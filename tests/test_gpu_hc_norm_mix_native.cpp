// SPDX-License-Identifier: MIT
// Link against the real backend (CORE_OBJS); no copied GPU implementation.
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

enum { N=16384, M=24, GUARD=64 };
static void check(bool ok,const char *message) {
    if(!ok){std::fprintf(stderr,"HC norm/mix native FAIL: %s\n",message);std::exit(1);}
}
static float poison(){uint32_t u=0x7fc12789u;float v;std::memcpy(&v,&u,4);return v;}
static uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
static uint32_t word(uint32_t &state){return state=state*1664525u+1013904223u;}
struct tensor {
    ds4_gpu_tensor *base=nullptr,*view=nullptr;
    size_t count;
    explicit tensor(size_t n):count(n) {
        base=ds4_gpu_tensor_alloc((n+2*GUARD)*4);
        view=base?ds4_gpu_tensor_view(base,GUARD*4,n*4):nullptr;
        check(base&&view,"guarded allocation"); reset();
    }
    ~tensor(){ds4_gpu_tensor_free(view);ds4_gpu_tensor_free(base);}
    tensor(const tensor &)=delete; tensor &operator=(const tensor &)=delete;
    void reset(){std::vector<float> v(count+2*GUARD,poison());check(ds4_gpu_tensor_write(base,0,v.data(),v.size()*4),"poison write");}
    std::vector<float> read()const {
        std::vector<float> v(count+2*GUARD);
        check(ds4_gpu_tensor_read(base,0,v.data(),v.size()*4),"device readback");
        for(size_t i=0;i<GUARD;++i)check(bits(v[i])==bits(poison())&&bits(v[v.size()-1-i])==bits(poison()),"device guards");
        return v;
    }
};
static void compare(const std::vector<float> &a,const std::vector<float> &b,unsigned seed) {
    double max_abs=0,max_rel=0;size_t mismatches=0;
    for(unsigned i=0;i<M;++i) {
        const float av=a[GUARD+i],bv=b[GUARD+i];
        check(std::isfinite(av)&&std::isfinite(bv),"every output must be written and finite");
        const double delta=std::abs(double(av)-double(bv));
        max_abs=std::max(max_abs,delta);
        max_rel=std::max(max_rel,delta/std::max(1e-30,std::abs(double(av))));
        mismatches+=bits(av)!=bits(bv);
    }
    if(mismatches)std::fprintf(stderr,"seed=%u mismatched=%zu/%u max_abs=%.9g max_rel=%.9g\n",seed,mismatches,M,max_abs,max_rel);
    check(!mismatches,"actual baseline/candidate must be bitwise equal");
}
int main(int argc,char **argv) {
    check(argc==1||(argc==2&&!std::strcmp(argv[1],"--bench")),"usage: test_gpu_hc_norm_mix_native [--bench]");
    check(ds4_gpu_init(),"backend initialization (a supported GPU is required)");
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const auto previous=ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_DECODE);
    const size_t page=size_t(getpagesize()), weight_bytes=size_t(N)*M*2, model_bytes=weight_bytes+2*page;
    void *model=nullptr;
    check(!posix_memalign(&model,page,model_bytes),"aligned model allocation");
    std::memset(model,0xa5,model_bytes);
    auto *weights=reinterpret_cast<uint16_t *>(static_cast<char *>(model)+page);
    uint32_t state=9137;
    for(size_t i=0;i<weight_bytes/2;++i)weights[i]=uint16_t((word(state)&0x8000u)|0x2800u|(word(state)&0x7ffu));
    const std::vector<uint8_t> model_original(static_cast<uint8_t *>(model),static_cast<uint8_t *>(model)+model_bytes);
    const uint64_t offset=page, size=weight_bytes;
    check(ds4_gpu_set_model_map_spans(model,model_bytes,&offset,&size,1,weight_bytes),"model span registration");
    bool candidate_supported=true;
    unsigned cases=0;
    {
        tensor x(N),norm(N),reference(M),actual(M);
        auto baseline=[&](float eps) {
            return ds4_gpu_rms_norm_plain_tensor(norm.view,x.view,N,eps)&&
                ds4_gpu_matmul_f16_tensor(reference.view,model,model_bytes,page,N,M,norm.view,1);
        };
        auto candidate=[&](float eps) {
            return ds4_gpu_hc_rms_norm_mix_f16_tensor(actual.view,x.view,model,model_bytes,page,N,M,eps);
        };
        std::printf("HC norm/mix graph_available=%d; explicit candidate tested independently.\n",ds4_gpu_hc_rms_norm_mix_f16_available());
        for(unsigned seed=0;seed<64;++seed) {
            std::vector<float> input(N);
            for(unsigned i=0;i<N;++i) {
                float v=std::ldexp(float(int(word(state)%1025u)-512),int(i%17u)-12);
                if(seed%8==0)v=0;
                if(seed%8==1)v=i%2?1e-10f:-1e-10f;
                if(seed%8==2)v=i%2?1e10f:-1e10f;
                input[i]=v;
            }
            check(ds4_gpu_tensor_write(x.view,0,input.data(),N*4),"activation upload");
            reference.reset();actual.reset();norm.reset();
            const float eps=seed&1?1e-6f:1e-5f;
            check(ds4_gpu_begin_commands(),"begin baseline commands");
            check(baseline(eps),"baseline actual API dispatch");
            check(ds4_gpu_end_commands()&&ds4_gpu_synchronize(),"baseline completion");
            check(ds4_gpu_begin_commands(),"begin candidate commands");
            const int result=candidate(eps);
            check(ds4_gpu_end_commands()&&ds4_gpu_synchronize(),"candidate completion");
            if(result==0 && seed==0) {
                const auto untouched=actual.read();
                for(float v:untouched)check(bits(v)==bits(poison()),"declined candidate touched output");
                candidate_supported=false;break;
            }
            check(result==1,"candidate declined or failed after writer");
            compare(reference.read(),actual.read(),seed);
            const auto after=x.read();
            check(!std::memcmp(after.data()+GUARD,input.data(),N*4),"input unchanged");
            norm.read();++cases;
        }
        if(candidate_supported && ds4_gpu_decode_graphs_supported()) {
            ds4_gpu_decode_graphs_invalidate();
            ds4_decode_graph_key key{};
            key.cur_hc=ds4_gpu_tensor_contents(x.view);
            key.after_attn_hc=ds4_gpu_tensor_contents(actual.view);
            key.attn_norm=ds4_gpu_tensor_contents(norm.view);
            for(unsigned epoch=0;epoch<2;++epoch) {
                if(epoch) {
                    // The real batched F16 API doubles the common half-input
                    // allocation from 32 KiB to 64 KiB. It must invalidate the
                    // graph before freeing the previous captured scratch.
                    tensor x2(2*N),out2(2*M);
                    std::vector<float> input2(2*N,0.125f);
                    check(ds4_gpu_tensor_write(x2.view,0,input2.data(),input2.size()*4),"growth input upload");
                    check(ds4_gpu_begin_commands(),"begin actual scratch growth");
                    check(ds4_gpu_matmul_f16_tensor(out2.view,model,model_bytes,page,N,M,x2.view,2),"actual scratch growth API");
                    check(ds4_gpu_end_commands()&&ds4_gpu_synchronize(),"scratch growth completion");
                    out2.read();x2.read();
                }
                for(unsigned replay=0;replay<5;++replay) {
                    std::vector<float> input(N);
                    for(unsigned i=0;i<N;++i)input[i]=(int(word(state)%1025u)-512)/1024.f;
                    check(ds4_gpu_tensor_write(x.view,0,input.data(),N*4),"fresh graph input upload");
                    reference.reset();actual.reset();
                    check(ds4_gpu_begin_commands(),"begin graph reference");
                    check(baseline(1e-6f),"graph reference API");
                    check(ds4_gpu_end_commands()&&ds4_gpu_synchronize(),"graph reference completion");
                    const int mode=ds4_gpu_decode_graph_begin(&key);
                    check(mode==(replay==0?-1:replay==1?0:1),
                          "expected actual warm/capture/replay sequence, including scratch invalidation");
                    if(mode!=1)check(candidate(1e-6f)==1,"graph candidate API");
                    if(mode==0)check(ds4_gpu_decode_graph_end(&key)==0,"actual CUDA graph capture and first launch");
                    check(ds4_gpu_synchronize(),"actual CUDA graph replay completion");
                    compare(reference.read(),actual.read(),64+epoch*5+replay);
                    const auto after=x.read();
                    check(!std::memcmp(after.data()+GUARD,input.data(),N*4),"graph input unchanged");
                    ++cases;
                }
            }
            ds4_gpu_decode_graphs_invalidate();
            std::puts("PASS: two actual CUDA graph warm/capture/three-replay sequences with fresh inputs, output poison and intervening common-scratch growth/invalidation.");
        } else if(candidate_supported) {
            std::puts("SKIP: backend decode-graph capture unavailable; eager API parity remains verified.");
        }
        if(candidate_supported && argc==2) {
            constexpr unsigned repeats=64, blocks=8;
            std::vector<double> samples[2];
            for(unsigned b=0;b<blocks;++b)for(unsigned step=0;step<4;++step) {
                const unsigned arm=(step==1||step==2);
                const auto start=std::chrono::steady_clock::now();
                check(ds4_gpu_begin_commands(),"begin benchmark batch");
                for(unsigned repeat=0;repeat<repeats;++repeat)
                    check(arm?candidate(1e-6f)==1:baseline(1e-6f),"benchmark API dispatch");
                check(ds4_gpu_end_commands()&&ds4_gpu_synchronize(),"benchmark completion");
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
                if(b)samples[arm].push_back(us);
            }
            double mean[2]={0,0},median[2]={0,0};
            for(unsigned arm=0;arm<2;++arm) {
                std::printf("SAMPLES %s us=",arm?"candidate":"baseline");
                for(size_t i=0;i<samples[arm].size();++i){std::printf("%s%.6f",i?",":"",samples[arm][i]);mean[arm]+=samples[arm][i]/samples[arm].size();}
                std::putchar('\n');std::sort(samples[arm].begin(),samples[arm].end());
                median[arm]=(samples[arm][6]+samples[arm][7])*0.5;
            }
            std::printf("BENCH full API norm+projection, CPU dispatch and synchronized GPU work included; allocations/model upload excluded; warm fixed input, ABBA14samples/arm x64calls. baseline_mean_us=%.4f candidate_mean_us=%.4f mean_speedup_pct=%.3f baseline_median_us=%.4f candidate_median_us=%.4f\n",mean[0],mean[1],100*(mean[0]/mean[1]-1),median[0],median[1]);
            compare(reference.read(),actual.read(),64);
        }
    }
    check(!std::memcmp(model,model_original.data(),model_bytes),"model and model guards unchanged");
    ds4_gpu_exchange_execution_phase(previous);
    ds4_gpu_cleanup();std::free(model);
    if(!candidate_supported){std::puts("SKIP: explicit HC norm/mix candidate is unavailable on this backend/device/configuration; no performance result.");return 77;}
    std::printf("PASS: %u native real-backend HC norm/mix bitwise cases with guards and immutable inputs.\n",cases);
    return 0;
}
