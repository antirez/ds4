// Production gather and (on HIP) the unchanged original D4 quantizer are
// inserted by test_rocm_mmq_quant_reuse.py. No historical checkout required.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(bool ok, const char *what) {
    if(!ok) {std::fprintf(stderr,"MMQ quant reuse FAIL: %s\n",what);std::exit(1);}
}
#ifdef TEST_NATIVE
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define QK8_1 32
#define WARP_SIZE 32
#else
struct alignas(16) uint4 {uint32_t x,y,z,w;};
struct Dim {uint32_t x=0,y=0,z=0;};
static Dim blockIdx,threadIdx,blockDim;
#endif

// PRODUCTION_SOURCE

namespace reuse=ds4_mmq_quant_reuse;
static constexpr size_t guard=4; // uint4 vectors, preserving16-byte alignment
static const uint4 poison={0xdeadabedu,0x1337faefu,0x4caf7811u,0x8903dd27u};
static bool same(const std::vector<uint4>& a,const std::vector<uint4>& b) {
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(uint4))==0;
}
static std::vector<int32_t> assignment_map(uint32_t tokens,uint32_t assignments,uint32_t pattern) {
    std::vector<int32_t> ids(assignments,0);
    std::mt19937 random(713);
    for(uint32_t a=0;a<assignments;++a) {
        if(pattern==0) ids[a]=int32_t(a%tokens);
        if(pattern==1) ids[a]=int32_t(tokens-1-(a%tokens));
        if(pattern==2) ids[a]=int32_t(random()%tokens);
        // Same pre-zeroed unwritten tail semantics as mm_ids_helper.
        if(pattern==3) ids[a]=a<assignments/2 ? int32_t(a%tokens) : 0;
    }
    return ids;
}
static void reference_gather(std::vector<uint4>& expected,const std::vector<uint4>& compact,
                             const std::vector<int32_t>& ids,uint32_t tokens,uint32_t ksegs) {
    for(uint32_t k=0;k<ksegs;++k)
    for(size_t a=0;a<ids.size();++a) {
        const size_t src=guard+(size_t(k)*tokens+size_t(ids[a]))*9;
        const size_t dst=guard+(size_t(k)*ids.size()+a)*9;
        std::memcpy(expected.data()+dst,compact.data()+src,144);
    }
}

static void test_policy() {
    size_t cases=0;
    for(bool iq2:{false,true}) for(bool gfx:{false,true}) for(bool direct:{false,true})
    for(int k:{0,256,1024,4096,8192})
    for(int n:{0,1,127,128,129,511,512,1024,2047,2048,2049,4096})
    for(int experts:{1,255,256,288}) for(int top:{1,5,6,8}) {
        const bool expected=iq2&&gfx&&!direct&&k==4096&&n>=128&&n<=2048&&experts==256&&top==6;
        check(reuse::select(iq2,gfx,direct,k,n,experts,top)==expected,"HIP IQ2 gfx1151 production scope");++cases;
    }
    check(reuse::compact_bytes(4096,2048)==9u*1024u*1024u,"maximum extra scratch9MiB");
    check(reuse::compact_bytes(4096,128)==576u*1024u,"minimum scratch576KiB");
    std::printf("PASS %zu production scope policies\n",cases);
}

#ifndef TEST_NATIVE
int main() {
    test_policy();size_t cases=0;
    for(uint32_t tokens:{1u,7u,32u,127u,128u,129u,512u,2048u})
    for(uint32_t ksegs:{1u,2u,32u})
    for(uint32_t tail:{0u,1u,17u})
    for(uint32_t pattern=0;pattern<5;++pattern) {
        const uint32_t assignments=tokens*6-tail*(tokens>3);
        const auto ids=assignment_map(tokens,assignments,pattern);
        std::vector<uint4> compact(guard+size_t(tokens)*ksegs*9+guard,poison);
        for(size_t r=0;r<size_t(tokens)*ksegs;++r) {
            // Include signed zero, NaN payload and infinity among raw F32
            // scale bits; the gather must preserve all bits without arithmetic.
            compact[guard+r*9]={0u,0x80000000u,0x7fc00101u+uint32_t(r%255),0x7f800000u};
            for(uint32_t v=1;v<9;++v) {
                const uint32_t x=uint32_t(r*9+v)*2654435761u;
                compact[guard+r*9+v]={x,x^0x89abcdefu,x+1u,x*17u};
            }
        }
        const auto before=compact;
        // Include enough overread-tail storage for a MMQ column tile. It must
        // remain exactly the prior memset pattern, as must outer canaries.
        const size_t payload=size_t(assignments)*ksegs*9,tail_vectors=64*9;
        std::vector<uint4> actual(guard+payload+tail_vectors+guard,poison);
        std::memset(actual.data()+guard,pattern%2 ? 0xff : 0,(payload+tail_vectors)*sizeof(uint4));
        auto expected=actual;reference_gather(expected,compact,ids,tokens,ksegs);
        blockDim.x=reuse::gather_threads;
        // One extra block tests the out-of-range vector guard.
        for(blockIdx.y=0;blockIdx.y<ksegs;++blockIdx.y)
        for(blockIdx.x=0;blockIdx.x<=(assignments*9+255)/256;++blockIdx.x)
        for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x)
            ds4_mmq_gather_q8_1_records_kernel(compact.data()+guard,ids.data(),actual.data()+guard,tokens,assignments);
        check(same(actual,expected),"actual gather: record/scale bits, map0, vector tail, MMQ padding and canaries");
        check(same(compact,before),"immutable compact records/guards");++cases;
    }
    // Empty assignment grid must return before dereferencing either input.
    blockDim.x=256;blockIdx.x=blockIdx.y=0;
    for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
        ds4_mmq_gather_q8_1_records_kernel(nullptr,nullptr,nullptr,1,0);
    std::printf("PASS %zu actual gather cases (raw144-byte records, guards, zero-map tails)\n",cases);
    std::puts("Host validation does not emulate quantization or establish a GPU speedup.");
}
#else
static void hip_check(hipError_t rc,const char *what) {
    if(rc!=hipSuccess) {std::fprintf(stderr,"%s: %s\n",what,hipGetErrorString(rc));std::exit(1);}
}
template<class T> struct Device {
    T *p=nullptr;size_t count;
    explicit Device(const std::vector<T>& input):count(input.size()) {
        hip_check(hipMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)),"allocation");
        hip_check(hipMemcpy(p,input.data(),count*sizeof(T),hipMemcpyHostToDevice),"upload");
    }
    ~Device(){hipFree(p);}
    Device(const Device&)=delete;
    std::vector<T> read() const {
        std::vector<T> output(count);
        hip_check(hipMemcpy(output.data(),p,count*sizeof(T),hipMemcpyDeviceToHost),"readback");return output;
    }
};
static std::vector<int32_t> sorted_top6_map(uint32_t tokens,uint32_t dropped) {
    // Stable expert-sorted true top-k: six distinct experts per token, using
    // all256 experts. Dropped assignments reproduce the zeroed mmid tail.
    std::vector<int32_t> ids(tokens*6,0);uint32_t cursor=0;
    for(uint32_t e=0;e<256;++e)
    for(uint32_t t=0;t<tokens;++t)
    for(uint32_t slot=0;slot<6;++slot) {
        const uint32_t pair=t*6+slot;
        if(pair+ dropped<tokens*6 && (t*17+slot*37)%256==e) ids[cursor++]=int32_t(t);
    }
    check(cursor==tokens*6-dropped,"sorted top-k map count");return ids;
}

static void quantize_original(const float *x,const int32_t *ids,void *out,
                              uint32_t k,uint32_t padded_k,uint32_t rows,hipStream_t stream) {
    const dim3 grid(rows,(padded_k+511)/512,1);
    quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4><<<grid,128,0,stream>>>(
        x,ids,out,k,k,k,k, padded_k,rows,1);
}
static void gather(const void *compact,const int32_t *ids,void *out,
                   uint32_t n,uint32_t assignments,uint32_t padded_k,hipStream_t stream) {
    const dim3 grid((assignments*9+255)/256,padded_k/128,1);
    ds4_mmq_gather_q8_1_records_kernel<<<grid,256,0,stream>>>(compact,ids,out,n,assignments);
}

template<class Launch> static double measure(Launch launch,int repetitions,hipStream_t stream) {
    hipEvent_t begin,end;hip_check(hipEventCreate(&begin),"event");hip_check(hipEventCreate(&end),"event");
    hip_check(hipEventRecord(begin,stream),"event begin");
    for(int r=0;r<repetitions;++r) launch();
    hip_check(hipEventRecord(end,stream),"event end");hip_check(hipEventSynchronize(end),"event wait");
    float ms=0;hip_check(hipEventElapsedTime(&ms,begin,end),"elapsed");
    hipEventDestroy(begin);hipEventDestroy(end);return double(ms)*1000/repetitions;
}
template<class A,class B> static void benchmark(A before,B after,uint32_t n,hipStream_t stream) {
    for(int r=0;r<4;++r) {before();after();}
    hip_check(hipStreamSynchronize(stream),"warmup");
    std::vector<double> a,b;
    for(int sample=0;sample<12;++sample) {
        double x,y;
        if(sample%2) {y=measure(after,16,stream);x=measure(before,16,stream);}
        else {x=measure(before,16,stream);y=measure(after,16,stream);}
        a.push_back(x);b.push_back(y);
        std::printf("sample,N%u,K4096,topk6,%d,original_us,%.6f,reuse_us,%.6f\n",n,sample,x,y);
    }
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
    const double ma=(a[5]+a[6])*0.5,mb=(b[5]+b[6])*0.5;
    std::printf("median,N%u,original_us,%.6f,reuse_us,%.6f,speedup_percent,%.3f\n",n,ma,mb,100*(ma/mb-1));
}

static void native_case(uint32_t n,uint32_t k,uint32_t pattern,bool timing,hipStream_t stream) {
    const uint32_t padded_k=4096,assignments=n*6,ksegs=padded_k/128;
    const auto ids=pattern<2 ? sorted_top6_map(n,pattern==1 ? n/3+1 : 0) :
                   assignment_map(n,assignments,pattern==2 ? 2 : 4);
    // Four floats per quantizer lane; views retain16-byte alignment with
    // nonzero input offsets, so row and K indexing cannot assume basezero.
    std::vector<float> x(16+size_t(n)*k+16,-1203.0f);
    for(uint32_t t=0;t<n;++t)
    for(uint32_t kk=0;kk<k;++kk) {
        float value=float(int((uint64_t(t)*701+kk*29)%4001)-2000)*0.0009765625f;
        if(t%17==16 || kk/32%19==18) value=0.0f;
        x[16+size_t(t)*k+kk]=value;
    }
    const size_t payload=size_t(assignments)*ksegs*9;
    // Match the production gfx1151 default overread tail: the generator
    // verifies get_mmq_x_max_host still selects64 columns for this device.
    const size_t tail_vectors=64*9;
    std::vector<uint4> init(guard+payload+tail_vectors+guard,poison);
    Device<uint4> a(init),b(init);
    std::vector<uint4> compact_init(guard+size_t(n)*ksegs*9+guard,poison);
    Device<uint4> compact(compact_init);
    Device<float> dx(x);Device<int32_t> dm(ids);
    const int fill=pattern==1 ? 0xff : 0;
    auto before=[&] {
        hip_check(hipMemsetAsync(a.p+guard,fill,(payload+tail_vectors)*16,stream),"baseline memset");
        quantize_original(dx.p+16,dm.p,a.p+guard,k,padded_k,assignments,stream);
        hip_check(hipGetLastError(),"baseline quantize launch");
    };
    auto after=[&] {
        hip_check(hipMemsetAsync(b.p+guard,fill,(payload+tail_vectors)*16,stream),"reuse memset");
        quantize_original(dx.p+16,nullptr,compact.p+guard,k,padded_k,n,stream);
        hip_check(hipGetLastError(),"token quantize launch before gather");
        gather(compact.p+guard,dm.p,b.p+guard,n,assignments,padded_k,stream);
        hip_check(hipGetLastError(),"record gather launch");
    };
    before();after();hip_check(hipStreamSynchronize(stream),"preparation validation");
    const auto av=a.read(),bv=b.read(),cv=compact.read();
    check(same(av,bv),"original quantizer IDs vs quant-token+gather exact full buffer bits");
    auto expected=init;
    std::memset(expected.data()+guard,fill,(payload+tail_vectors)*16);
    reference_gather(expected,cv,ids,n,ksegs);
    check(same(bv,expected),"native gather matches CPU record copy and untouched MMQ tail");
    for(size_t i=0;i<guard;++i) {
        check(std::memcmp(&cv[i],&compact_init[i],16)==0,"compact prefix guard");
        const size_t j=cv.size()-guard+i;
        check(std::memcmp(&cv[j],&compact_init[j],16)==0,"compact suffix guard");
    }
    check(dm.read()==ids && dx.read()==x,"immutable IDs and activation guards");
    // Every nonzero input group must publish a non-poison, positive finite
    // scale. Examine integer bits so compiler finite-math flags cannot hide
    // a skipped store; zero blocks legitimately produce scalezero.
    for(uint32_t ks=0;ks<ksegs;++ks)
    for(uint32_t t=0;t<n;++t) {
        uint32_t scales[4];std::memcpy(scales,&cv[guard+(size_t(ks)*n+t)*9],16);
        for(uint32_t g=0;g<4;++g) {
            const uint32_t kk=ks*128+g*32;
            bool nonzero=false;
            for(uint32_t j=0;j<32 && kk+j<k;++j) nonzero |= x[16+size_t(t)*k+kk+j]!=0;
            if(nonzero) check(scales[g]>0 && scales[g]<0x7f800000u,"nonzero group wrote finite positive scale");
            else check(scales[g]==0,"zero group scale bits");
        }
    }
    std::printf("PASS native N%u K%u Kp%u map%u stream%s payload%zuB\n",n,k,padded_k,pattern,stream ? "nondefault" : "default",payload*16);
    if(timing) {
        std::puts("GPU preparation benchmark includes gathered memset and all quant/gather launches; fixed warm inputs, maps and allocations outside timing, MMQ compute excluded;default64-column tail.");
        benchmark(before,after,n,stream);
        hip_check(hipStreamSynchronize(stream),"benchmark completion");
        check(same(a.read(),b.read()),"timed preparation remains bit-identical");
    }
}

int main(int argc,char **argv) {
    test_policy();const int device=argc>1 ? std::atoi(argv[1]) : 0;
    const bool bench=argc>2 && std::strcmp(argv[2],"--bench")==0;
    hip_check(hipSetDevice(device),"device");hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop,device),"properties");
    check(std::strncmp(prop.gcnArchName,"gfx1151",7)==0 && prop.warpSize==32,"production quant-reuse validation requires gfx1151 wave32");
    std::printf("GPU %s %s wave%d\n",prop.name,prop.gcnArchName,prop.warpSize);
    hipStream_t nondefault;hip_check(hipStreamCreateWithFlags(&nondefault,hipStreamNonBlocking),"stream");
    size_t cases=0;
    for(hipStream_t stream:{hipStream_t(nullptr),nondefault})
    for(uint32_t n:{128u,129u,512u,2048u})
    for(uint32_t k:{3840u,4096u})
    for(uint32_t pattern=0;pattern<4;++pattern) {
        native_case(n,k,pattern,false,stream);++cases;
    }
    std::printf("PASS %zu native exact quantization reuse cases\n",cases);
    if(bench) for(uint32_t n:{128u,512u,1024u,2048u}) native_case(n,4096,0,true,nullptr);
    hip_check(hipStreamDestroy(nondefault),"destroy stream");
}

#endif
