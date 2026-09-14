// The Python driver inserts actual production functions at the marker below.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>
#include "ds4_gpu_phase.h"

#ifdef TEST_NATIVE
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cudaGetLastError hipGetLastError
static int cuda_ok(hipError_t rc, const char *what) {
    if (rc != hipSuccess) std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
    return rc == hipSuccess;
}
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ && defined(__gfx1151__)
#define DS4_ROCM_Q4_GFX1151_WMMA_ROWTILE_DEVICE 1
using ds4_q4_half16_t = _Float16 __attribute__((ext_vector_type(16)));
using ds4_q4_float8_t = float __attribute__((ext_vector_type(8)));
using ds4_q4_uchar16_t = uint8_t __attribute__((ext_vector_type(16)));
#else
#define DS4_ROCM_Q4_GFX1151_WMMA_ROWTILE_DEVICE 0
#endif
__device__ static float dev_f16_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}
#else
using __half = _Float16;
static __half __float2half(float value) { return __half(value); }
struct alignas(4) half2 { __half x,y; };
struct alignas(16) float4 { float x,y,z,w; };
static half2 __floats2half2_rn(float x,float y){return {__half(x),__half(y)};}
struct dim3 { uint32_t x,y,z; dim3(uint32_t a=1,uint32_t b=1,uint32_t c=1):x(a),y(b),z(c){} };
static dim3 blockDim,blockIdx,threadIdx;
#endif

#include "rocm/ds4_rocm_q4_activation.cuh"
#include "rocm/ds4_rocm_q4_scales.cuh"
#include "rocm/ds4_rocm_q4_wmma_load.cuh"
namespace activation = ds4_rocm_q4_activation;
enum { CUDA_QK_K=256, ROCM_Q4_WMMA_TOKEN_TILE=64, ROCM_Q4_WMMA_K_TILE=32,
       ROCM_Q4_WMMA_K128_TILE=128, ROCM_Q4_WMMA_K128_LDS_PITCH=144,
       ROCM_Q4_WMMA_FRAGMENT=16 };
struct cuda_block_q4_K { uint16_t d,dmin; uint8_t scales[12],qs[128]; };
static_assert(sizeof(cuda_block_q4_K)==144,"Q4 GGUF ABI");
static void check(bool ok,const char *why) {
    if(!ok){std::fprintf(stderr,"Q4 activation FAIL: %s\n",why);std::exit(1);}
}
static uint16_t bits(__half h){uint16_t b;std::memcpy(&b,&h,2);return b;}

#ifndef TEST_NATIVE
static void *g_cuda_tmp,*g_rocm_q4_attn_q_b_transient_f16_scratch;
static uint64_t g_cuda_tmp_bytes,g_rocm_q4_attn_q_b_transient_f16_scratch_bytes;
static uint64_t g_rocm_q4_prefill_wmma_launches,g_rocm_q4_prefill_wmma_k128_launches,
    g_rocm_q4_prefill_wmma_k64_launches,g_rocm_q4_prefill_wmma_k64_load4_launches;
static bool g_ssd_streaming_mode,g_quality_mode,gfx1151=true;
static uint32_t allocated,freed,conversion_calls,half_calls,float_calls,k64_calls,k32_calls;
static uint64_t requested;
static int pending_error,fail_convert,fail_half,fail_allocate,k64_control=-1,k128_disable=-1;
static void *allocation_result;
static int cudaGetLastError(){const int rc=pending_error;pending_error=0;return rc;}
static int cuda_ok(int rc,const char *){return rc==0;}
static int rocm_q4_qb_gfx1151_wave32_device(){return gfx1151;}
static void rocm_q4_K_prefill_stats_register(){}
static uint32_t rocm_q4_K_prefill_wmma_row_tile(uint32_t m){return m>=8192?256:m>=1024?128:64;}
static int rocm_q4_attn_q_b_env_bool(const char *key){
    return std::strstr(key,"DISABLE") ? k128_disable : k64_control;
}
static void *cuda_tmp_alloc(uint64_t bytes,const char *){
    ++allocated;requested=bytes;
    if(g_cuda_tmp_bytes>=bytes)return g_cuda_tmp;
    if(g_cuda_tmp){++freed;g_cuda_tmp=nullptr;g_cuda_tmp_bytes=0;}
    if(fail_allocate)return nullptr;
    g_cuda_tmp=allocation_result;g_cuda_tmp_bytes=bytes;return g_cuda_tmp;
}
static void conversion_launch(dim3 geometry,__half *,const float *,uint64_t count){
    check(geometry.x==(count+255)/256&&geometry.y==256,"production conversion grid");
    ++conversion_calls;pending_error=fail_convert;
}
template<class... Args> static void wmma_half_launch(dim3 grid,Args...){
    check(grid.x==128&&grid.z==1,"production F16 WMMA grid");
    ++half_calls;pending_error=fail_half;
}
template<class... Args> static void wmma_float_launch(dim3,Args...){++float_calls;}
template<class... Args> static void rocm_q4_K_prefill_wmma_k64_enqueue(Args...){++k64_calls;}
template<class... Args> static void rocm_q4_K_prefill_wmma_enqueue(Args...){++k32_calls;}
#endif

// PRODUCTION_SOURCE

#ifndef TEST_NATIVE
static size_t test_conversion_and_staging(){
    size_t cases=0;
    for(uint32_t n:{256u,257u,511u,512u,1024u,2048u}){
        // Exactly ending allocations expose reads past the last active token.
        std::vector<float> source(size_t(n)*activation::inner);
        for(size_t i=0;i<source.size();++i){
            constexpr uint32_t low[]={0u,0x0fffu,0x1000u,0x1001u};
            const uint32_t raw=(uint32_t(i%65536u)<<16u)|low[(i/65536u)%4u];
            std::memcpy(source.data()+i,&raw,4);
        }
        std::vector<uint16_t> converted(16+source.size()+16,0x5ad5);
        auto out=reinterpret_cast<__half*>(converted.data()+16);
        blockDim=dim3(256);
        for(uint64_t i=0;i<(source.size()+255u)/256u*256u;++i){
            blockIdx.x=uint32_t(i/256);threadIdx.x=uint32_t(i%256);
            f32_to_f16_kernel(out,source.data(),source.size());
        }
        for(size_t i=0;i<source.size();++i)
            check(converted[16+i]==bits(__half(source[i])),"actual conversion FP16 bits");
        for(uint32_t i=0;i<16;++i)
            check(converted[i]==0x5ad5&&converted[16+source.size()+i]==0x5ad5,"conversion guards");
        const auto before=converted;
        for(uint32_t tok0=0;tok0<n;tok0+=64)
        for(uint32_t k0=0;k0<1024;k0+=128){
            std::vector<uint16_t> stage(16+64*144+16,0x5ad5),expected=stage,baseline=stage;
            for(uint32_t t=0;t<64;++t)for(uint32_t k=0;k<128;++k)
                expected[16+t*144+k]=tok0+t<n ? bits(__half(source[size_t(tok0+t)*1024+k0+k])) : 0;
            blockDim.x=512;
            for(uint32_t tid=0;tid<512;++tid){
                activation::stage_thread(reinterpret_cast<__half*>(stage.data()+16),out,n,tok0,k0,tid,512);
                stage_f32_baseline(reinterpret_cast<_Float16*>(baseline.data()+16),source.data(),n,tok0,k0,tid);
            }
            check(stage==expected&&stage==baseline,"actual half8 versus actual F32 staging, scalar oracle, tails and LDS padding");
            ++cases;
        }
        check(before==converted,"staging source immutable");
    }
    return cases;
}

static void reset(){
    (void)ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_AUTO);
    allocated=freed=conversion_calls=half_calls=float_calls=k64_calls=k32_calls=0;
    requested=0;pending_error=fail_convert=fail_half=fail_allocate=0;
    g_ssd_streaming_mode=g_quality_mode=false;gfx1151=true;k64_control=k128_disable=-1;
    g_cuda_tmp=g_rocm_q4_attn_q_b_transient_f16_scratch=nullptr;
    g_cuda_tmp_bytes=g_rocm_q4_attn_q_b_transient_f16_scratch_bytes=0;
    allocation_result=reinterpret_cast<void*>(uintptr_t(0x500000000ull));
    unsetenv("DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K64_LOAD4");
}
static size_t test_dispatch(){
    size_t cases=0;
    auto out=reinterpret_cast<float*>(uintptr_t(0x100000000ull));
    auto w=reinterpret_cast<const char*>(uintptr_t(0x200000000ull));
    auto x=reinterpret_cast<const float*>(uintptr_t(0x300000000ull));
    auto launch=[&](uint32_t n=512,uint32_t g=1,uint32_t k=1024,uint32_t m=32768,
                    uint64_t rb=576,uint64_t xs=1024,uint64_t gs=0,uint64_t os=32768){
        return rocm_q4_K_prefill_wmma_launch(out,w,x,n,g,k,m,rb,xs,gs,os,"test");
    };
    for(uint32_t n:{255u,256u,257u,511u,512u,1024u,2048u,2049u,4096u}){
        reset();check(launch(n)==1,"valid K128 launch");
        const bool reused=n>=256&&n<=2048;
        check(allocated==reused&&conversion_calls==reused&&half_calls==reused&&float_calls==!reused,"bounded production selection");
        if(reused)check(requested==uint64_t(n)*1024*2&&requested<=activation::max_bytes,"at most four MiB");
        ++cases;
    }
    for(auto phase:{DS4_GPU_PHASE_AUTO,DS4_GPU_PHASE_PREFILL,DS4_GPU_PHASE_DECODE,
                    DS4_GPU_PHASE_VERIFY,DS4_GPU_PHASE_BATCH_DECODE,DS4_GPU_PHASE_MIXED}){
        reset();
        check(ds4_gpu_exchange_execution_phase(phase)==DS4_GPU_PHASE_AUTO,"phase setup");
        check(launch()==1,"phase must retain a valid WMMA fallback");
        const bool reused=phase==DS4_GPU_PHASE_AUTO||phase==DS4_GPU_PHASE_PREFILL;
        check(allocated==reused&&conversion_calls==reused&&half_calls==reused&&float_calls==!reused,
              "only AUTO/PREFILL prepare half RHS; other phases retain F32 K128");
        check(ds4_gpu_get_execution_phase()==phase,"dispatch must not change caller phase");
        ++cases;
    }
    for(uint32_t which=0;which<12;++which){
        reset();int result;
        if(which==0){g_ssd_streaming_mode=true;result=launch();}
        else if(which==1){g_quality_mode=true;result=launch();}
        else if(which==2){gfx1151=false;result=launch();}
        else if(which==3){k128_disable=1;result=launch();}
        else if(which==4){k64_control=0;result=launch();}
        else if(which==5)result=launch(512,2);
        else if(which==6)result=launch(512,1,2048);
        else if(which==7)result=launch(512,1,1024,8192);
        else if(which==8)result=launch(512,1,1024,32768,580);
        else if(which==9)result=launch(512,1,1024,32768,576,1028);
        else if(which==10)result=launch(512,1,1024,32768,576,1024,1024);
        else result=launch(512,1,1024,32768,576,1024,0,32772);
        check(result==1&&allocated==0&&conversion_calls==0&&half_calls==0,"unchanged fallback admission");++cases;
    }
    for(uint32_t which=0;which<4;++which){
        reset();g_cuda_tmp=which==0?static_cast<void*>(out):which==1?const_cast<char*>(w):which==2?const_cast<float*>(x):allocation_result;
        g_cuda_tmp_bytes=16;
        if(which==3){g_rocm_q4_attn_q_b_transient_f16_scratch=g_cuda_tmp;g_rocm_q4_attn_q_b_transient_f16_scratch_bytes=64;}
        check(launch()==1&&float_calls==1&&allocated==0&&freed==0,"alias blocks allocator before free");++cases;
    }
    for(uint32_t which=0;which<4;++which){
        reset();
        if(which==0)fail_allocate=1;
        if(which==1)allocation_result=const_cast<float*>(x);
        if(which==2)allocation_result=reinterpret_cast<void*>(uintptr_t(allocation_result)+2);
        if(which==3){g_cuda_tmp=allocation_result;g_cuda_tmp_bytes=16;fail_allocate=1;}
        check(launch()==1&&allocated==1&&conversion_calls==0&&half_calls==0&&float_calls==1,"allocation/pre-submit failure safely falls back");++cases;
    }
    reset();g_cuda_tmp=allocation_result;g_cuda_tmp_bytes=activation::max_bytes;
    check(launch()==1&&freed==0&&half_calls==1,"reuse existing arena without growth");++cases;
    reset();fail_convert=1;
    check(launch()==0&&conversion_calls==1&&half_calls==0&&float_calls==0,"conversion failure never replays baseline");++cases;
    reset();fail_half=1;
    check(launch()==0&&conversion_calls==1&&half_calls==1&&float_calls==0,"consumer failure never replays baseline");++cases;
    reset();
    check(ds4_rocm_bench_q4_K_wmma_k128_enqueue(out,w,x,512,1,1024,32768,576,1024,0,32768)==1&&float_calls==1,"baseline hook stays F32");
    check(ds4_rocm_bench_q4_K_wmma_k128_half_enqueue(out,w,x,allocation_result,activation::max_bytes,512)==1&&conversion_calls==1&&half_calls==1,"candidate hook includes conversion");
    check(ds4_rocm_bench_q4_K_wmma_k128_half_enqueue(out,w,x,const_cast<float*>(x),activation::max_bytes,512)==0,"benchmark alias rejected");++cases;
    return cases;
}

int main(){
    const auto staging=test_conversion_and_staging();
    const auto dispatch=test_dispatch();
    std::printf("PASS Q4 activation: %zu conversion/staging tiles and %zu extracted dispatch/fault cases\n",staging,dispatch);
    std::puts("Host-only: HIP compilation, WMMA bit parity, occupancy and timing remain required.");
}
#else
static void hip_check(hipError_t rc,const char *why){check(cuda_ok(rc,why),why);}
template<typename T> struct Device {
    T *ptr=nullptr;
    size_t count;
    explicit Device(const std::vector<T>& values):count(values.size()){
        hip_check(hipMalloc(reinterpret_cast<void**>(&ptr),count*sizeof(T)),"allocate");
        hip_check(hipMemcpy(ptr,values.data(),count*sizeof(T),hipMemcpyHostToDevice),"upload");
    }
    ~Device(){hipFree(ptr);}
    Device(const Device&)=delete;
    std::vector<T> read(){
        std::vector<T> result(count);
        hip_check(hipMemcpy(result.data(),ptr,count*sizeof(T),hipMemcpyDeviceToHost),"read");
        return result;
    }
};
template<class Launch> static double elapsed(Launch launch,int repeat){
    hipEvent_t begin,end;
    hip_check(hipEventCreate(&begin),"create event");hip_check(hipEventCreate(&end),"create event");
    hip_check(hipEventRecord(begin),"start event");
    for(int i=0;i<repeat;++i)launch();
    hip_check(hipEventRecord(end),"stop event");hip_check(hipEventSynchronize(end),"event sync");
    float ms=0;hip_check(hipEventElapsedTime(&ms,begin,end),"event elapsed");
    hipEventDestroy(begin);hipEventDestroy(end);return double(ms)*1000/repeat;
}
template<class Before,class After> static void time_pair(uint32_t n,Before before,After after){
    for(int i=0;i<4;++i){before();after();}
    hip_check(hipDeviceSynchronize(),"warmup");
    std::vector<double>a,b;
    for(int sample=0;sample<12;++sample){
        double x,y;
        if(sample%2){y=elapsed(after,8);x=elapsed(before,8);}
        else{x=elapsed(before,8);y=elapsed(after,8);}
        a.push_back(x);b.push_back(y);
        std::printf("sample,Q-B,N%u,%d,F32_K128_us,%.6f,convert_plus_F16_K128_us,%.6f\n",n,sample,x,y);
    }
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
    const double ma=(a[5]+a[6])*0.5,mb=(b[5]+b[6])*0.5;
    std::printf("median,Q-B,N%u,F32_K128_us,%.6f,convert_plus_F16_K128_us,%.6f,speedup_percent,%.3f,warm_weights,allocation_excluded\n",n,ma,mb,100*(ma/mb-1));
}
static void native_case(uint32_t n,bool bench){
    constexpr size_t guard=16;
    constexpr uint32_t sentinel=0x7fc12345u;
    const size_t xe=size_t(n)*activation::inner,oe=size_t(n)*activation::outputs;
    std::vector<cuda_block_q4_K> weights(activation::weight_bytes/144+2);
    uint32_t random=0x824ebd1u;
    auto next=[&](){random^=random<<13;random^=random>>17;random^=random<<5;return random;};
    for(auto &w:weights){
        w.d=bits(__half(0.03125f));w.dmin=bits(__half(0.015625f));
        for(auto &s:w.scales)s=uint8_t(next());for(auto&q:w.qs)q=uint8_t(next());
    }
    std::vector<float> x(4+xe+4,-1234.0f);
    for(size_t i=0;i<xe;++i){
        // Finite inputs include exact halves and values on either side of
        // F16 rounding boundaries; the source pointer is only 16B aligned.
        uint32_t raw=0x3f000000u+(next()&0x007fffffu);
        if(i&1u)raw|=0x80000000u;
        if(i%257u==0u)raw=0u;
        if(i%263u==0u)raw=0x80000000u;
        std::memcpy(x.data()+4+i,&raw,4);
    }
    std::vector<uint32_t> output(guard+oe+guard,sentinel);
    std::vector<uint16_t> scratch(guard+xe+guard,0x5ad5);
    Device<cuda_block_q4_K> dw(weights);Device<float> dx(x);
    Device<uint32_t> a(output),b(output);Device<uint16_t> dh(scratch);
    const auto w=dw.ptr+1;
    auto baseline=[&](){check(ds4_rocm_bench_q4_K_wmma_k128_enqueue(a.ptr+guard,w,dx.ptr+4,
        n,1,activation::inner,activation::outputs,activation::row_bytes,activation::inner,0,activation::outputs)==1,"baseline enqueue");};
    auto candidate=[&](){check(ds4_rocm_bench_q4_K_wmma_k128_half_enqueue(b.ptr+guard,w,dx.ptr+4,
        dh.ptr+guard,xe*sizeof(uint16_t),n)==1,"conversion+candidate enqueue");};
    baseline();candidate();hip_check(hipGetLastError(),"launch");
    const auto ar=a.read(),br=b.read();
    const auto hr=dh.read();
    check(ar==br,"native F32/F16 input K128 exact FP32 outputs");
    for(size_t i=0;i<ar.size();++i){
        if(i<guard||i>=guard+oe)check(ar[i]==sentinel,"output guards");
        else check((ar[i]&0x7f800000u)!=0x7f800000u,"every output written and finite");
    }
    for(size_t i=0;i<hr.size();++i){
        if(i<guard||i>=guard+xe)check(hr[i]==0x5ad5,"conversion scratch guards");
        else check(hr[i]==bits(__half(x[4+i-guard])),"native conversion exact F16 bits");
    }
    const auto wr=dw.read();
    const auto xr=dx.read();
    check(std::memcmp(wr.data(),weights.data(),weights.size()*sizeof(weights[0]))==0,"weights immutable");
    check(std::memcmp(xr.data(),x.data(),x.size()*sizeof(float))==0,"inputs and guards immutable");
    std::printf("PASS native Q-B F32/convert+F16 N%u K1024 M32768 scratch_bytes%zu\n",n,xe*2);
    if(bench&&(n==256||n==512||n==1024||n==2048))time_pair(n,baseline,candidate);
}
int main(int argc,char**argv){
    const int device=argc>1?std::atoi(argv[1]):0;
    const bool bench=argc>2&&std::strcmp(argv[2],"--bench")==0;
    hip_check(hipSetDevice(device),"device");hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop,device),"device properties");
    check(std::strncmp(prop.gcnArchName,"gfx1151",7)==0&&prop.warpSize==32,"gfx1151 wave32 required");
    std::printf("GPU %s %s wave%d\n",prop.name,prop.gcnArchName,prop.warpSize);
    auto baseline=rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel<256u,16u,1u,float>;
    auto candidate=rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel<256u,16u,1u,__half>;
    hipFuncAttributes a{},b{};
    hip_check(hipFuncGetAttributes(&a,reinterpret_cast<const void*>(baseline)),"baseline resources");
    hip_check(hipFuncGetAttributes(&b,reinterpret_cast<const void*>(candidate)),"candidate resources");
    std::printf("resources,F32,registers%d,LDS%zu,maxthreads%d,F16,registers%d,LDS%zu,maxthreads%d\n",
        a.numRegs,a.sharedSizeBytes,a.maxThreadsPerBlock,b.numRegs,b.sharedSizeBytes,b.maxThreadsPerBlock);
    for(uint32_t n:{256u,257u,511u,512u,1024u,2048u})native_case(n,bench);
}
#endif
