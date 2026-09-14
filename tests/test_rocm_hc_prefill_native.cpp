// SPDX-License-Identifier: MIT
// Link the real ROCm backend. Synthetic weights/activations require no GGUF.
#include "../ds4_gpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

static constexpr size_t guard = 64;
static constexpr uint32_t output_dim = 24;
static unsigned cases;
static void check(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr,"ROCm HC prefill native FAIL: %s\n",message); std::exit(1); }
}
static uint32_t bits(float value) { uint32_t out; std::memcpy(&out,&value,4); return out; }
static float poison() { const uint32_t u=0x7fc53971; float v; std::memcpy(&v,&u,4); return v; }
static uint32_t word(uint32_t &state) { return state=state*1664525u+1013904223u; }
struct tensor {
    ds4_gpu_tensor *base=nullptr, *view=nullptr;
    size_t count;
    explicit tensor(size_t n):count(n) {
        base=ds4_gpu_tensor_alloc((n+2*guard)*sizeof(float));
        view=base?ds4_gpu_tensor_view(base,guard*sizeof(float),n*sizeof(float)):nullptr;
        check(base&&view,"guarded tensor allocation"); reset();
    }
    ~tensor() { ds4_gpu_tensor_free(view); ds4_gpu_tensor_free(base); }
    tensor(const tensor&)=delete; tensor &operator=(const tensor&)=delete;
    void reset() {
        std::vector<float> init(count+2*guard,poison());
        check(ds4_gpu_tensor_write(base,0,init.data(),init.size()*sizeof(float)),"poison output");
    }
    std::vector<float> read() const {
        std::vector<float> v(count+2*guard);
        check(ds4_gpu_tensor_read(base,0,v.data(),v.size()*sizeof(float)),"readback");
        for(size_t i=0;i<guard;++i)
            check(bits(v[i])==bits(poison())&&bits(v[v.size()-1-i])==bits(poison()),"tensor canaries");
        return v;
    }
};
struct model_fixture {
    void *model=nullptr;
    size_t offset, bytes;
    std::vector<uint8_t> original;
    model_fixture() {
        offset=size_t(getpagesize());
        const size_t weight_bytes=size_t(28672)*output_dim*2;
        bytes=weight_bytes+2*offset;
        check(posix_memalign(&model,offset,bytes)==0,"aligned synthetic model");
        std::memset(model,0xa5,bytes);
        auto *p=reinterpret_cast<uint16_t *>(static_cast<char *>(model)+offset);
        uint32_t random=7519;
        for(size_t i=0;i<weight_bytes/2;++i)
            p[i]=uint16_t((word(random)&0x8000u)|0x2000u|(word(random)&0x7ffu));
        original.assign(static_cast<uint8_t *>(model),static_cast<uint8_t *>(model)+bytes);
        const uint64_t woffset=offset,wbytes=weight_bytes;
        check(ds4_gpu_set_model_map_spans(model,bytes,&woffset,&wbytes,1,wbytes),"register synthetic weights");
    }
    void verify() const { check(!std::memcmp(model,original.data(),bytes),"model and model guards immutable"); }
    ~model_fixture() { std::free(model); }
};
static void compare(const tensor &a,const tensor &b,uint32_t n,uint32_t rows) {
    const auto av=a.read(),bv=b.read();
    size_t mismatches=0; double maximum=0;
    for(size_t i=0;i<a.count;++i) {
        check(std::isfinite(av[guard+i])&&std::isfinite(bv[guard+i]),"every output written and finite");
        mismatches+=bits(av[guard+i])!=bits(bv[guard+i]);
        maximum=std::max(maximum,std::abs(double(av[guard+i])-double(bv[guard+i])));
    }
    if(mismatches)std::fprintf(stderr,"K=%u rows=%u mismatch=%zu/%zu maxabs=%.9g\n",n,rows,mismatches,a.count,maximum);
    check(mismatches==0,"real baseline norm+GEMM vs folded norm+same GEMM bits");
}
static std::vector<float> make_input(uint32_t n,uint32_t rows,unsigned pattern) {
    std::vector<float> x(size_t(n)*rows);
    uint32_t random=1239+n+rows;
    for(size_t i=0;i<x.size();++i) {
        float v=std::ldexp(float(int(word(random)%1025u)-512),int(i%17u)-12);
        if(pattern==1)v=i%2?0.f:-0.f;
        if(pattern==2)v=i%2?1e-10f:-1e-10f;
        if(pattern==3)v=i%2?1e10f:-1e10f;
        x[i]=v;
    }
    return x;
}
static bool run(model_fixture &model,uint32_t n,uint32_t rows,unsigned pattern,bool bench) {
    tensor x(size_t(n)*rows),norm(size_t(n)*rows),baseline(size_t(output_dim)*rows),candidate(baseline.count);
    const auto input=make_input(n,rows,pattern);
    check(ds4_gpu_tensor_write(x.view,0,input.data(),input.size()*sizeof(float)),"upload input");
    const float eps=pattern%2?1e-5f:1e-6f;
    auto reference=[&]() {
        return ds4_gpu_rms_norm_plain_rows_tensor(norm.view,x.view,n,rows,eps)&&
            ds4_gpu_matmul_f16_tensor(baseline.view,model.model,model.bytes,model.offset,n,output_dim,norm.view,rows);
    };
    auto actual=[&]() {
        return ds4_gpu_matmul_f16_rms_fold_tensor(candidate.view,model.model,model.bytes,model.offset,n,output_dim,x.view,rows,eps);
    };
    check(ds4_gpu_begin_commands(),"begin actual baseline");
    check(reference(),"baseline RMS and GEMM dispatch");
    check(ds4_gpu_end_commands()&&ds4_gpu_synchronize(),"baseline complete");
    check(ds4_gpu_begin_commands(),"begin folded candidate");
    const int result=actual();
    check(ds4_gpu_end_commands()&&ds4_gpu_synchronize(),"candidate complete");
    if(result==0) {
        for(float value:candidate.read())check(bits(value)==bits(poison()),"decline preserves output");
        return false;
    }
    check(result==1,"post-submission failure is fatal, without fallback");
    compare(baseline,candidate,n,rows);
    auto input_after=x.read();
    check(!std::memcmp(input_after.data()+guard,input.data(),input.size()*sizeof(float)),"input and row offsets immutable");
    norm.read(); model.verify(); ++cases;
    if(bench) {
        constexpr unsigned repeats=8,warmup=2,blocks=7;
        std::vector<double> samples[2];
        for(unsigned b=0;b<warmup+blocks;++b)for(unsigned step=0;step<4;++step) {
            const unsigned arm=step==1||step==2;
            const auto start=std::chrono::steady_clock::now();
            check(ds4_gpu_begin_commands(),"begin timed full API batch");
            for(unsigned repeat=0;repeat<repeats;++repeat)
                check(arm?actual()==1:reference(),"timed actual API dispatch");
            check(ds4_gpu_end_commands()&&ds4_gpu_synchronize(),"timed GPU completion");
            const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
            if(b>=warmup)samples[arm].push_back(us);
        }
        double median[2],mean[2]={0,0};
        std::printf("BENCH full API K=%u M=%u N=%u repeats=%u warmup_ABBA=%u samples_per_arm=%u\n",n,output_dim,rows,repeats,warmup,2*blocks);
        for(unsigned arm=0;arm<2;++arm) {
            std::printf("  %s raw_us=",arm?"candidate":"baseline");
            for(size_t i=0;i<samples[arm].size();++i) {
                std::printf("%s%.6f",i?",":"",samples[arm][i]);
                mean[arm]+=samples[arm][i]/samples[arm].size();
            }
            std::sort(samples[arm].begin(),samples[arm].end());
            median[arm]=(samples[arm][blocks-1]+samples[arm][blocks])*0.5;
            std::printf(" mean=%.6f median=%.6f range=%.6f..%.6f\n",mean[arm],median[arm],samples[arm].front(),samples[arm].back());
        }
        std::printf("  median stage throughput gain=%+.3f%%; CPU encoding, synchronization, RMS, conversion and actual BLAS included; allocation, model I/O and full model execution excluded.\n",100*(median[0]/median[1]-1));
        compare(baseline,candidate,n,rows); model.verify();
        input_after=x.read();
        check(!std::memcmp(input_after.data()+guard,input.data(),input.size()*sizeof(float)),"timed input unchanged");
        std::fflush(stdout);
    }
    return true;
}
int main(int argc,char **argv) {
    if(argc>2||(argc==2&&std::strcmp(argv[1],"--bench"))) {
        std::fprintf(stderr,"Usage: %s [--bench]\n",argv[0]); return 2;
    }
    check(ds4_gpu_init(),"ROCm initialization requires a supported GPU");
    ds4_gpu_set_quality(false); ds4_gpu_set_ssd_streaming(false);
    const auto previous=ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_PREFILL);
    bool available=true;
    {
        model_fixture model;
        // The existing 28672-wide projection already consumes half RHS for
        // small batches; only the 16384-wide <=8-token path is excluded.
        for(uint32_t rows:{2u,8u}) {
            if(!run(model,28672,rows,0,false)) {available=false;break;}
        }
        for(uint32_t n:{16384u,28672u}) {
            if(!available)break;
            for(uint32_t rows:{9u,127u,128u,129u,512u,2048u}) {
                const unsigned patterns=rows<=129?4u:1u;
                for(unsigned pattern=0;pattern<patterns;++pattern) {
                    ds4_gpu_exchange_execution_phase(pattern%2?DS4_GPU_PHASE_AUTO:DS4_GPU_PHASE_PREFILL);
                    if(!run(model,n,rows,pattern,false)) {available=false;break;}
                }
                if(!available)break;
            }
            if(!available)break;
        }
        if(available&&argc==2) {
            ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_PREFILL);
            for(uint32_t n:{16384u,28672u})
                for(uint32_t rows:{128u,512u,2048u})
                    check(run(model,n,rows,0,true),"benchmark candidate must be available");
        }
        model.verify();
        // Backend mappings are released before freeing the registered host map.
        ds4_gpu_exchange_execution_phase(previous); ds4_gpu_cleanup();
    }
    if(!available) {
        std::puts("SKIP: eligible ROCm folded HC prefill candidate unavailable; no parity or performance claim."); return 77;
    }
    std::printf("PASS: %u native real-backend HC prefill bitwise cases, guards and immutable input/model; operands checked separately by test_rocm_hc_prefill.py --rocm.\n",cases);
}
