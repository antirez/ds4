// Python inserts the real kernel bodies after adapting their barrier stages.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csetjmp>
#include <cfloat>
#include <vector>
#include "cuda/ds4_hc_norm_mix.cuh"

using __half = _Float16;
static float __half2float(__half x) { return float(x); }
static __half __float2half(float x) { return __half(x); }
static float __fmul_rn(float a, float b) { volatile float p = a*b; return p; }
static float rsqrtf(float x) { return 1.0f/std::sqrt(x); }
static struct { unsigned x,y; } threadIdx, blockIdx, blockDim = {256,0};
static bool collecting;
static float leaves[256], reduced;
static std::jmp_buf collected;
static float reduce_stage(float sum) {
    if (collecting) { leaves[threadIdx.x] = sum; std::longjmp(collected, 1); }
    return reduced;
}
static void finish_reduce() {
    for (unsigned stride = 128; stride; stride >>= 1)
        for (unsigned lane = 0; lane < stride; ++lane) {
            volatile float next = leaves[lane] + leaves[lane+stride];
            leaves[lane] = next;
        }
    reduced = leaves[0];
}
#define __syncthreads() ((void)0)
static void check(bool ok, const char *why);

// PRODUCTION_SOURCE

static void check(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr,"HC norm/mix FAIL: %s\n",why); std::exit(1); }
}
enum { N = 16384, M = 24, G = 16 };
static float poison() { uint32_t bits=0x7fc12567u; float v; std::memcpy(&v,&bits,4); return v; }
static uint32_t bits(float v) { uint32_t u; std::memcpy(&u,&v,4); return u; }
template<class F> static void norm_run(F kernel) {
    blockIdx.x=0; blockDim.x=256; collecting=true;
    for (threadIdx.x=0; threadIdx.x<256; ++threadIdx.x)
        if (!setjmp(collected)) kernel();
    finish_reduce(); collecting=false;
    for (threadIdx.x=0; threadIdx.x<256; ++threadIdx.x) kernel();
}
static void check_float_buffer(const std::vector<float> &v, size_t active) {
    for (size_t i=0;i<v.size();++i)
        check(i>=G && i<G+active ? (bits(v[i])&0x7f800000u)!=0x7f800000u :
                                  bits(v[i])==bits(poison()), "finite writers / float guards");
}

static void arithmetic_case(unsigned seed) {
    std::vector<float> x(N+2*G), norm(N+2*G,poison()), scale(1+2*G,poison());
    std::vector<float> reference(M+2*G,poison()), actual(M+2*G,poison());
    std::vector<__half> weights(size_t(M)*N);
    const uint16_t half_guard=0x7e35; __half hguard; std::memcpy(&hguard,&half_guard,2);
    std::vector<__half> cuda_reference(N+2*G,hguard), cuda_actual(N+2*G,hguard);
    uint32_t state=seed+17;
    auto word=[&](){state=state*1664525u+1013904223u;return state;};
    std::fill(x.begin(),x.end(),poison());
    for(unsigned i=0;i<N;++i) {
        float v=std::ldexp(float(int(word()%1025u)-512),int(i%17u)-12);
        if(seed%8==0)v=0.f;
        if(seed%8==1)v=i%2?1e-10f:-1e-10f;
        if(seed%8==2)v=i%2?1e10f:-1e10f;
        if(seed%8==3 && i%13==0)v=-0.f;
        x[G+i]=v;
    }
    for(__half &v:weights)v=__half((int(word()%1025u)-512)/1024.f);
    const auto original=x; const auto original_weights=weights;
    const float eps=seed&1?1e-6f:1e-5f;
    norm_run([&](){rms_norm_plain_kernel(norm.data()+G,x.data()+G,N,1,eps);});
    norm_run([&](){ds4_hc_rms_scale_kernel(scale.data()+G,x.data()+G,N,eps);});
    check_float_buffer(norm,N); check_float_buffer(scale,1);
    blockDim.x=32;
    for(blockIdx.x=0;blockIdx.x<M;++blockIdx.x) {
        // The real barrier publishes every lane before lane zero reduces.
        for(unsigned order=1;order<=32;++order) {
            threadIdx.x=order%32;
            matmul_f16_ordered_chunks_kernel(reference.data()+G,weights.data(),norm.data()+G,N,M,1);
            ds4_hc_f16_project_scaled_ordered_kernel(actual.data()+G,weights.data(),x.data()+G,scale.data()+G,N,M);
        }
    }
    check_float_buffer(reference,M); check_float_buffer(actual,M);
    check(!std::memcmp(reference.data(),actual.data(),reference.size()*4),"ROCm ordered projection bits");

    norm_run([&](){rms_norm_plain_batch8_kernel(norm.data()+G,x.data()+G,N,1,eps);});
    for(unsigned i=0;i<N;++i)cuda_reference[G+i]=__float2half(norm[G+i]);
    norm_run([&](){rms_norm_plain_f16_batch8_kernel(cuda_actual.data()+G,x.data()+G,N,1,eps);});
    check(!std::memcmp(cuda_reference.data(),cuda_actual.data(),cuda_actual.size()*2),
          "CUDA normalized half input bits / guards");
    for(unsigned i=0;i<N;++i) {
        uint16_t h;std::memcpy(&h,&cuda_actual[G+i],2);
        check((h&0x7c00u)!=0x7c00u,"CUDA half writer is finite");
    }
    check(!std::memcmp(x.data(),original.data(),x.size()*4),"input and input guards unchanged");
    check(!std::memcmp(weights.data(),original_weights.data(),weights.size()*2),"weights unchanged");
}

static void policy_cases() {
    namespace p=ds4_hc_norm_mix;
    const uintptr_t out=0x100000u,x=0x200000u,model=0x400000u;
    auto eligible=[&](uint32_t n=N,uint32_t m=M,ds4_gpu_execution_phase phase=DS4_GPU_PHASE_DECODE,
            bool quality=false,uintptr_t op=0x100000u,uint64_t ob=p::out_bytes,
            uintptr_t xp=0x200000u,uint64_t xb=p::x_bytes,uintptr_t mp=0x400000u,
            uint64_t mb=p::weight_bytes+16u,uint64_t offset=16u) {
        return p::eligible(n,m,phase,quality,op,ob,xp,xb,mp,mb,offset);
    };
    check(eligible(),"eligible HC decode shape");
    for(auto phase:{DS4_GPU_PHASE_AUTO,DS4_GPU_PHASE_PREFILL,DS4_GPU_PHASE_DECODE,
                    DS4_GPU_PHASE_VERIFY,DS4_GPU_PHASE_BATCH_DECODE,DS4_GPU_PHASE_MIXED})
        check(eligible(N,M,phase)==(phase==DS4_GPU_PHASE_AUTO||phase==DS4_GPU_PHASE_DECODE),"phase eligibility");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,true),"quality fallback");
    for(uint32_t n:{0u,16383u,16385u,32768u,UINT32_MAX})check(!eligible(n),"width fallback");
    for(uint32_t m:{0u,23u,25u,48u,UINT32_MAX})check(!eligible(N,m),"row-count fallback");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,0),"null output");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes-1),"short output");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,0),"null input");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,x,p::x_bytes-1),"short input");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out+2),"unaligned output");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,x+2),"unaligned input");
    const uintptr_t aliases[]={x,x+4,uintptr_t(x+p::x_bytes-4),model+16,uintptr_t(model+16+p::weight_bytes-4)};
    for(uintptr_t alias:aliases)
        check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,alias),"output overlap rejected");
    check(eligible(N,M,DS4_GPU_PHASE_DECODE,false,x+p::x_bytes),"adjacent buffers admitted");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,x,p::x_bytes,0),"null model");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,x,p::x_bytes,model,p::weight_bytes+15,16),"truncated weights");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,x,p::x_bytes,model,p::weight_bytes+16,17),"unaligned or short weights");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,x,p::x_bytes,model,p::weight_bytes,UINT64_MAX),"weight-offset overflow");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,UINTPTR_MAX-3),"output range wrap");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,UINTPTR_MAX-3),"input range wrap");
    check(!eligible(N,M,DS4_GPU_PHASE_DECODE,false,out,p::out_bytes,x,p::x_bytes,UINTPTR_MAX-1,4,0),"model range wrap");
    check(!p::range_valid(1,0)&&p::range_valid(UINTPTR_MAX,1)&&!p::range_valid(UINTPTR_MAX,2),"range endpoints");
    check(!p::overlaps(x,4,x+4,4)&&p::overlaps(x,4,x+3,4)&&!p::overlaps(x,0,x,4),"overlap endpoints");
    std::puts("PASS: actual HC policy shapes, all phases, quality, capacities, nulls, alignment, aliasing and pointer/offset overflow.");
}
int main() {
    policy_cases();
    wrapper_cases();
    for(unsigned seed=0;seed<64;++seed)arithmetic_case(seed);
    std::puts("PASS: 64 actual-source ROCm norm+ordered projection and CUDA norm-to-half bitwise cases, guards and immutable inputs. Host barrier staging does not test GPU scheduling, FTZ or native BLAS.");
}
