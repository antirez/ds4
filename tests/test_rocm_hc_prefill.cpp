// SPDX-License-Identifier: MIT
// The runner inserts real production bodies, never a copied RMS algorithm.
#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#ifdef DS4_HC_PREFILL_NATIVE
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
using __half = _Float16;
static __half __float2half(float x) { return __half(x); }
static float __fmul_rn(float a,float b) { volatile float p=a*b; return p; }
static float rsqrtf(float x) { return 1.f/std::sqrt(x); }
static struct { unsigned x; } threadIdx,blockIdx,blockDim={256};
static bool collecting;
static float leaves[256],reduced;
static std::jmp_buf collected;
static float reduce_stage(float sum) {
    if(collecting) { leaves[threadIdx.x]=sum; std::longjmp(collected,1); }
    return reduced;
}
static void finish_reduce() {
    for(unsigned step=128;step;step>>=1)for(unsigned lane=0;lane<step;++lane) {
        volatile float sum=leaves[lane]+leaves[lane+step]; leaves[lane]=sum;
    }
    reduced=leaves[0];
}
#endif
#include "rocm/ds4_rocm_rms_f16.cuh"

static void check(bool ok,const char *message) {
    if(!ok) { std::fprintf(stderr,"ROCm HC prefill FAIL: %s\n",message); std::exit(1); }
}
static uint32_t bits(float value) { uint32_t u; std::memcpy(&u,&value,4); return u; }
static float float_poison() { uint32_t u=0x7fc71935; float v; std::memcpy(&v,&u,4); return v; }
static __half half_poison() { uint16_t u=0x7e79; __half v; std::memcpy(&v,&u,2); return v; }
static uint32_t word(uint32_t &state) { return state=state*1664525u+1013904223u; }
#ifdef DS4_HC_PREFILL_NATIVE
static constexpr size_t guard=128; // GPU timing retains 256-byte operand alignment.
#else
static constexpr size_t guard=17; // Host address stress includes non-vector-aligned offsets.
#endif
static unsigned arithmetic_cases;

#ifndef DS4_HC_PREFILL_NATIVE
struct ds4_gpu_tensor {void *ptr; uint64_t bytes;};
static bool g_quality_mode,g_glm_model,g_cublas_ready;
static int g_cublas=19,stream_result,capture_result,capture_value;
static void *stream_value;
using hipStream_t=void *;
using hipStreamCaptureStatus=int;
enum {HIPBLAS_STATUS_SUCCESS=0,hipSuccess=0,hipStreamCaptureStatusNone=0};
static int hipblasGetStream(int handle,hipStream_t *out) {check(handle==19,"stream handle");*out=stream_value;return stream_result;}
static int hipStreamIsCapturing(hipStream_t stream,hipStreamCaptureStatus *out) {check(!stream,"capture uses default stream");*out=capture_value;return capture_result;}
static thread_local ds4_gpu_execution_phase phase=DS4_GPU_PHASE_AUTO;
extern "C" ds4_gpu_execution_phase ds4_gpu_get_execution_phase() {return phase;}
extern "C" ds4_gpu_execution_phase ds4_gpu_exchange_execution_phase(ds4_gpu_execution_phase p) {auto old=phase;phase=p;return old;}
static void *g_cuda_tmp,*allocation_result;
static uint64_t g_cuda_tmp_bytes;
static const char *resolved_weight;
static bool override_weight;
static unsigned allocations,launches,consumers;
static int fail_alloc,fail_norm,fail_consumer,pending_error;
static uint64_t expected_n,expected_rows;
static const char *cuda_model_range_ptr(const void *model,uint64_t offset,uint64_t size,const char *) {
    check(size==expected_n*24*2,"resolved weight shape");
    return override_weight?resolved_weight:static_cast<const char *>(model)+offset;
}
static void *cuda_tmp_alloc(uint64_t bytes,const char *) {
    check(!launches&&!consumers,"allocation before writers"); ++allocations;
    check(bytes==expected_n*expected_rows*2,"scratch holds exactly normalized half operand");
    if(fail_alloc)return nullptr;
    g_cuda_tmp=allocation_result;g_cuda_tmp_bytes=bytes;return g_cuda_tmp;
}
static void mock_norm_launch(uint32_t grid,uint32_t threads,__half *out,const float *,uint32_t n,uint32_t rows,float) {
    check(grid==expected_rows&&threads==256&&n==expected_n&&rows==expected_rows,"same one-block-per-row RMS launch");
    check(out==allocation_result,"producer uses prepared half scratch");
    ++launches;pending_error=fail_norm;
}
static int cudaGetLastError() {const int result=pending_error;pending_error=0;return result;}
static bool cuda_ok(int status,const char *) {return status==0;}
static int rocm_matmul_f16_prepared_tensor(ds4_gpu_tensor *,const __half *,const __half *x,
        uint64_t n,uint64_t m,uint64_t rows,bool stop_on_error) {
    check(launches==1&&n==expected_n&&m==24&&rows==expected_rows&&x==allocation_result,
          "consumer gets exact producer operand/dimensions after one successful launch");
    check(stop_on_error,"consumer cannot retry after a submission failure");
    ++consumers;return !fail_consumer;
}
#endif

// PRODUCTION_SOURCE

static std::vector<float> input_data(uint32_t n,uint32_t rows,unsigned pattern) {
    std::vector<float> x(guard+size_t(n)*rows+guard,float_poison());
    uint32_t state=1923+n+rows;
    for(size_t i=0;i<size_t(n)*rows;++i) {
        float v=std::ldexp(float(int(word(state)%1025u)-512),int(i%17u)-12);
        if(pattern==1)v=i%2?0.f:-0.f;
        if(pattern==2)v=i%2?1e-10f:-1e-10f;
        if(pattern==3)v=i%2?1e10f:-1e10f;
        x[guard+i]=v;
    }
    return x;
}
static void operands_equal(const std::vector<__half> &a,const std::vector<__half> &b,size_t count) {
    check(!std::memcmp(a.data(),b.data(),a.size()*2),"normalized FP16 operand bits, including guards");
    const __half poison=half_poison();
    for(size_t i=0;i<a.size();++i) {
        uint16_t u;std::memcpy(&u,&a[i],2);
        if(i>=guard&&i<guard+count)check((u&0x7c00u)!=0x7c00u,"every half operand written and finite");
        else check(!std::memcmp(&a[i],&poison,2),"half operand canary");
    }
}
#ifndef DS4_HC_PREFILL_NATIVE
template<class F> static void norm_run(uint32_t rows,F kernel) {
    blockDim.x=256;
    for(blockIdx.x=0;blockIdx.x<rows;++blockIdx.x) {
        collecting=true;
        for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
            if(!setjmp(collected))kernel();
        finish_reduce();collecting=false;
        for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)kernel();
    }
    // Actual kernel row guard must return before any reduction or write.
    for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)kernel();
}
static void arithmetic_case(uint32_t n,uint32_t rows,unsigned pattern) {
    const size_t count=size_t(n)*rows;
    auto x=input_data(n,rows,pattern);const auto original=x;
    std::vector<float> norm(count+2*guard,float_poison());
    std::vector<__half> reference(count+2*guard,half_poison()),actual=reference;
    const float eps=pattern%2?1e-5f:1e-6f;
    norm_run(rows,[&](){rms_norm_plain_kernel(norm.data()+guard,x.data()+guard,n,rows,eps);});
    for(blockIdx.x=0;blockIdx.x<(count+255)/256;++blockIdx.x)
        for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
            f32_to_f16_kernel(reference.data()+guard,norm.data()+guard,count);
    norm_run(rows,[&](){rocm_hc_rms_norm_f16_kernel(actual.data()+guard,x.data()+guard,n,rows,eps);});
    operands_equal(reference,actual,count);
    for(size_t i=0;i<norm.size();++i)
        check(i>=guard&&i<guard+count ? std::isfinite(norm[i]) : bits(norm[i])==bits(float_poison()),
              "baseline normalized float writers and guards");
    check(!std::memcmp(x.data(),original.data(),x.size()*4),"input and guarded row offsets immutable");
    ++arithmetic_cases;
}
static void policy_cases() {
    using ds4_rocm_hc_prefill::sizes;
    struct config {
        uint64_t n=16384,m=24,rows=9;
        ds4_gpu_execution_phase phase=DS4_GPU_PHASE_PREFILL;
        bool quality=false,glm=false,ready=true;
        uintptr_t out=UINT64_C(0x1000000000000000),x=UINT64_C(0x2000000000000000),model=UINT64_C(0x3000000000000000);
        uint64_t outcap=9*24*4,xcap=9*16384*4,modelsize=16384*24*2+16,offset=16;
        float eps=1e-6f;
    } p;
    auto call=[&](sizes *result) {return ds4_rocm_hc_prefill::eligible(p.n,p.m,p.rows,p.phase,
        p.quality,p.glm,p.ready,p.out,p.outcap,p.x,p.xcap,p.model,p.modelsize,p.offset,p.eps,result);};
    unsigned count=0;
    auto expect=[&](bool accepted) {
        sizes need{11,22,33,44};const auto before=need;
        check(call(&need)==accepted,"pure policy admission");
        if(accepted)check(need.weights==p.n*p.m*2&&need.input==p.rows*p.n*4&&
            need.output==p.rows*p.m*4&&need.half==p.rows*p.n*2,"policy exact byte sizes");
        else check(!std::memcmp(&need,&before,sizeof(need)),"declined policy does not publish partial sizes");
        ++count;
    };
    expect(true);
    for(auto value:{DS4_GPU_PHASE_AUTO,DS4_GPU_PHASE_PREFILL,DS4_GPU_PHASE_DECODE,
                    DS4_GPU_PHASE_VERIFY,DS4_GPU_PHASE_BATCH_DECODE,DS4_GPU_PHASE_MIXED}) {
        p=config{};p.phase=value;expect(value==DS4_GPU_PHASE_AUTO||value==DS4_GPU_PHASE_PREFILL);
    }
    for(uint64_t n:{UINT64_C(0),UINT64_C(4096),UINT64_C(16383),UINT64_C(16385),UINT64_C(28671),UINT64_C(28673),UINT64_MAX}) {
        p=config{};p.n=n;expect(false);
    }
    for(uint64_t m:{UINT64_C(0),UINT64_C(23),UINT64_C(25),UINT64_MAX}) {p=config{};p.m=m;expect(false);}
    for(uint64_t n:{UINT64_C(16384),UINT64_C(28672)})for(uint64_t rows:{UINT64_C(0),UINT64_C(1),UINT64_C(2),UINT64_C(8),UINT64_C(9),UINT64_C(128),uint64_t(INT32_MAX),UINT64_MAX}) {
        p=config{};p.n=n;p.rows=rows;p.outcap=p.xcap=UINT64_MAX;p.modelsize=n*24*2+16;
        expect(rows>=2&&rows<=INT32_MAX&&(n!=16384||rows>8));
    }
    p=config{};p.quality=true;expect(false);
    p=config{};p.glm=true;expect(false);
    p=config{};p.ready=false;expect(false);
    for(float eps:{0.f,-1.f,INFINITY,-INFINITY,NAN}) {p=config{};p.eps=eps;expect(false);}
    p=config{};p.eps=std::numeric_limits<float>::max();expect(true);
    p=config{};p.out=0;expect(false);p=config{};p.x=0;expect(false);p=config{};p.model=0;expect(false);
    p=config{};p.outcap--;expect(false);p=config{};p.xcap--;expect(false);p=config{};p.modelsize--;expect(false);
    p=config{};p.out+=2;expect(false);p=config{};p.x+=2;expect(false);p=config{};p.offset--;expect(false);
    p=config{};p.offset=UINT64_MAX;expect(false);
    p=config{};p.out=p.x;expect(false);p=config{};p.out=p.x+p.xcap-4;expect(false);
    p=config{};p.out=p.x+p.xcap;expect(true);
    p=config{};p.out=p.model+p.offset;expect(false);
    p=config{};p.out=p.model+p.offset+16384*24*2-4;expect(false);
    p=config{};p.out=UINTPTR_MAX-3;expect(false);p=config{};p.x=UINTPTR_MAX-3;expect(false);
    p=config{};p.model=UINTPTR_MAX-3;expect(false);
    p=config{};check(!call(nullptr),"null size output rejected");++count;
    std::printf("PASS: %u actual-policy shapes/phases/epsilon/alias/alignment/overflow cases.\n",count);
}
static void wrapper_cases() {
    constexpr uint64_t max_n=28672,rows=9;
    std::vector<float> input(max_n*rows),output(24*rows);
    std::vector<uint16_t> model(max_n*24+8),scratch(max_n*rows+16);
    ds4_gpu_tensor x{},out{};
    unsigned count=0;
    auto reset=[&]() {
        expected_n=16384;expected_rows=rows;
        x={input.data(),16384*rows*4};out={output.data(),24*rows*4};
        phase=DS4_GPU_PHASE_PREFILL;g_quality_mode=g_glm_model=false;g_cublas_ready=true;
        g_cuda_tmp=nullptr;g_cuda_tmp_bytes=0;allocation_result=scratch.data();
        stream_value=nullptr;stream_result=capture_result=capture_value=0;
        override_weight=false;resolved_weight=nullptr;
        allocations=launches=consumers=0;fail_alloc=fail_norm=fail_consumer=pending_error=0;
    };
    auto call=[&](float eps=1e-6f) {return ds4_gpu_matmul_f16_rms_fold_tensor(&out,model.data(),model.size()*2,
        16,expected_n,24,&x,expected_rows,eps);};
    auto decline=[&]() {check(call()==0&&!launches&&!consumers,"actual wrapper decline before writer");++count;};
    reset();check(call()==1&&allocations==1&&launches==1&&consumers==1,"actual wrapper producer+same consumer success");++count;
    reset();expected_n=max_n;x.bytes=max_n*rows*4;check(call()==1&&launches==1&&consumers==1,"extended 28672 HC width");++count;
    reset();phase=DS4_GPU_PHASE_AUTO;check(call()==1,"AUTO compatibility");++count;
    reset();fail_norm=1;check(call()==-1&&launches==1&&consumers==0,"producer failure is -1 without consumer/fallback");++count;
    reset();fail_consumer=1;check(call()==-1&&launches==1&&consumers==1,"consumer failure is -1 without fallback");++count;
    reset();fail_alloc=1;decline();check(allocations==1,"allocation fault exercised");
    for(auto p:{DS4_GPU_PHASE_DECODE,DS4_GPU_PHASE_VERIFY,DS4_GPU_PHASE_BATCH_DECODE,DS4_GPU_PHASE_MIXED}) {
        reset();phase=p;decline();check(!allocations,"phase rejects before allocation");
    }
    reset();g_quality_mode=true;decline();reset();g_glm_model=true;decline();reset();g_cublas_ready=false;decline();
    reset();stream_result=1;decline();check(!allocations,"stream query failure preflight");
    reset();stream_value=reinterpret_cast<void *>(uintptr_t(4));decline();check(!allocations,"nondefault stream preflight");
    reset();capture_result=1;decline();check(!allocations,"capture query failure preflight");
    reset();capture_value=1;decline();check(!allocations,"capture preflight before scratch growth");
    reset();x.bytes--;decline();reset();out.bytes--;decline();
    reset();x.ptr=nullptr;decline();reset();out.ptr=nullptr;decline();
    reset();out.ptr=x.ptr;decline();check(!allocations,"input/output alias precedes allocation");
    for(float eps:{0.f,-1.f,INFINITY,-INFINITY,NAN}) {
        reset();check(call(eps)==0&&!launches&&!consumers&&!allocations,"epsilon preflight");++count;
    }
    reset();override_weight=true;resolved_weight=nullptr;decline();
    reset();override_weight=true;resolved_weight=reinterpret_cast<const char *>(out.ptr);decline();
    reset();override_weight=true;resolved_weight=reinterpret_cast<const char *>(model.data())+1;decline();
    for(void *alias:{static_cast<void *>(input.data()),static_cast<void *>(output.data()),static_cast<void *>(model.data()+8)}) {
        reset();g_cuda_tmp=alias;g_cuda_tmp_bytes=2;decline();check(!allocations,"old scratch alias rejected before freeing/growing");
        reset();allocation_result=alias;decline();check(allocations==1,"returned scratch alias rejected before launch");
    }
    reset();allocation_result=reinterpret_cast<char *>(scratch.data())+1;decline();
    reset();g_cuda_tmp=scratch.data();g_cuda_tmp_bytes=scratch.size()*2;
    check(call()==1&&launches==1&&consumers==1,"existing disjoint scratch reuse");++count;
    std::printf("PASS: %u extracted-wrapper cases: stream/capture, bounds/aliases, allocation and producer/consumer faults, no post-write fallback.\n",count);
}
#else
static void hip_check(hipError_t status,const char *message) {
    if(status!=hipSuccess) {std::fprintf(stderr,"HIP %s: %s\n",message,hipGetErrorString(status));std::exit(1);}
}
template<class T> struct device_buffer {
    T *ptr=nullptr;size_t count;
    explicit device_buffer(size_t n):count(n) {hip_check(hipMalloc(&ptr,n*sizeof(T)),"allocate");}
    ~device_buffer() {hip_check(hipFree(ptr),"free");}
    void put(const std::vector<T> &v) {check(v.size()==count,"upload size");hip_check(hipMemcpy(ptr,v.data(),count*sizeof(T),hipMemcpyHostToDevice),"upload");}
    std::vector<T> get() {std::vector<T> v(count);hip_check(hipMemcpy(v.data(),ptr,count*sizeof(T),hipMemcpyDeviceToHost),"readback");return v;}
};
static void arithmetic_case(uint32_t n,uint32_t rows,unsigned pattern,bool bench=false) {
    const size_t count=size_t(n)*rows;
    const auto x=input_data(n,rows,pattern);
    std::vector<float> initial_norm(count+2*guard,float_poison());
    std::vector<__half> initial_half(count+2*guard,half_poison());
    device_buffer<float> dx(x.size()),norm(initial_norm.size());
    device_buffer<__half> a(initial_half.size()),b(initial_half.size());
    dx.put(x);norm.put(initial_norm);a.put(initial_half);b.put(initial_half);
    const float eps=pattern%2?1e-5f:1e-6f;
    const uint32_t grid_rows=bench?rows:rows+1;
    auto baseline=[&]() {
        rms_norm_plain_kernel<<<grid_rows,256>>>(norm.ptr+guard,dx.ptr+guard,n,rows,eps);
        hip_check(hipGetLastError(),"baseline RMS launch");
        f32_to_f16_kernel<<<(count+255)/256,256>>>(a.ptr+guard,norm.ptr+guard,count);
        hip_check(hipGetLastError(),"baseline cast launch");
    };
    auto candidate=[&]() {
        rocm_hc_rms_norm_f16_kernel<<<grid_rows,256>>>(b.ptr+guard,dx.ptr+guard,n,rows,eps);
        hip_check(hipGetLastError(),"candidate RMS-half launch");
    };
    baseline();candidate();hip_check(hipDeviceSynchronize(),"operand completion");
    operands_equal(a.get(),b.get(),count);
    const auto after=dx.get();check(!std::memcmp(after.data(),x.data(),x.size()*4),"device input and guards unchanged");
    const auto normalized=norm.get();
    for(size_t i=0;i<guard;++i)
        check(bits(normalized[i])==bits(float_poison())&&bits(normalized[normalized.size()-1-i])==bits(float_poison()),"device F32 scratch guards");
    ++arithmetic_cases;
    if(bench) {
        constexpr unsigned repeats=16,blocks=7,warmup=2;
        hipEvent_t start,stop;hip_check(hipEventCreate(&start),"event");hip_check(hipEventCreate(&stop),"event");
        std::vector<double> samples[2];
        for(unsigned block=0;block<warmup+blocks;++block)for(unsigned step=0;step<4;++step) {
            const unsigned arm=step==1||step==2;
            hip_check(hipEventRecord(start,nullptr),"event start");
            for(unsigned i=0;i<repeats;++i) {if(arm)candidate();else baseline();}
            hip_check(hipEventRecord(stop,nullptr),"event stop");hip_check(hipEventSynchronize(stop),"event wait");
            float ms=0;hip_check(hipEventElapsedTime(&ms,start,stop),"event elapsed");
            if(block>=warmup)samples[arm].push_back(ms*1e3/repeats);
        }
        double medians[2];
        std::printf("BENCH operand preparation K=%u rows=%u repeats=%u warmup_ABBA=%u samples_per_arm=%u\n",n,rows,repeats,warmup,2*blocks);
        for(unsigned arm=0;arm<2;++arm) {
            double mean=0;std::printf("  %s raw_us=",arm?"fused":"RMS+cast");
            for(size_t i=0;i<samples[arm].size();++i) {mean+=samples[arm][i]/samples[arm].size();std::printf("%s%.6f",i?",":"",samples[arm][i]);}
            std::sort(samples[arm].begin(),samples[arm].end());medians[arm]=(samples[arm][blocks-1]+samples[arm][blocks])*0.5;
            std::printf(" mean=%.6f median=%.6f range=%.6f..%.6f\n",mean,medians[arm],samples[arm].front(),samples[arm].back());
        }
        std::printf("  median operand-stage throughput gain=%+.3f%%; excludes GEMM, model, allocation and host transfers.\n",100*(medians[0]/medians[1]-1));
        hip_check(hipEventDestroy(start),"destroy event");hip_check(hipEventDestroy(stop),"destroy event");
        operands_equal(a.get(),b.get(),count);std::fflush(stdout);
    }
}
#endif

int main(int argc,char **argv) {
#ifndef DS4_HC_PREFILL_NATIVE
    (void)argc;(void)argv;
    policy_cases();wrapper_cases();lt_test::cases();graph_test::cases();
#else
    check(argc==1||(argc==2&&!std::strcmp(argv[1],"--bench")),"usage: operand fixture [--bench]");
    int device=0;hip_check(hipGetDevice(&device),"current device");
    hipDeviceProp_t prop{};hip_check(hipGetDeviceProperties(&prop,device),"device properties");
    std::printf("ROCm operand fixture: %s (%s), wave=%d\n",prop.name,prop.gcnArchName,prop.warpSize);
#endif
    for(uint32_t n:{1u,7u,255u,256u,257u,4096u,16384u,28672u})
        for(uint32_t rows:{1u,3u,9u})
            for(unsigned pattern=0;pattern<4;++pattern)arithmetic_case(n,rows,pattern);
    arithmetic_case(16384,128,0);arithmetic_case(28672,129,0);
#ifdef DS4_HC_PREFILL_NATIVE
    if(argc==2)for(uint32_t n:{16384u,28672u})for(uint32_t rows:{128u,512u,2048u})arithmetic_case(n,rows,0,true);
    std::printf("PASS: %u native HIP bitwise operand cases, guards and immutable input.\n",arithmetic_cases);
#else
    std::printf("PASS: %u actual-source bitwise operand cases, row/tail guards and immutable input; host staging does not validate GPU scheduling, FTZ or HIP assembly.\n",arithmetic_cases);
#endif
}
