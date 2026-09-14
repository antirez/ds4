// SPDX-License-Identifier: MIT
// The Python runner inserts actual production source at the marked locations.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <chrono>
#include <utility>
#ifdef TEST_NATIVE
#ifdef TEST_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#define GPU(name) hip##name
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#define GPU(name) cuda##name
#endif
#include "cuda/ds4_indexer_prepare.cuh"
#include "ds4_indexer_prepared_launch.cuh"
#ifdef TEST_HIP
#include "rocm/ds4_rocm_indexer_prepared.cuh"
#include "rocm/ds4_rocm_indexer_registers.cuh"
#else
#include "cuda/ds4_cuda_indexer_prepared.cuh"
#endif
#else
using __half = _Float16;
struct __half2 { __half x,y; };
static __half __float2half(float x) { return __half(x); }
static __half2 __floats2half2_rn(float x,float y) { return {__float2half(x),__float2half(y)}; }
static __half2 __halves2half2(__half x,__half y) { return {x,y}; }
static __half __ushort_as_half(unsigned short bits) { __half v; std::memcpy(&v,&bits,2); return v; }
static struct { unsigned x; } threadIdx,blockIdx,blockDim{256},gridDim;
#endif
#include "ds4_indexer_prepared.h"

static void check(bool condition,const char *message) {
    if(!condition) { std::fprintf(stderr,"indexer prepared FAIL: %s\n",message); std::exit(1); }
}
static uint32_t bits(float v) { uint32_t u;std::memcpy(&u,&v,4);return u; }
static uint16_t half_bits(__half v) { uint16_t u;std::memcpy(&u,&v,2);return u; }
static float from_bits(uint32_t u) {float v;std::memcpy(&v,&u,4);return v;}
static __half half_poison() {uint16_t u=0x7e79;__half v;std::memcpy(&v,&u,2);return v;}
static float float_poison() {return from_bits(0x7fc71935);}
static uint32_t random_word(uint32_t &s) {return s=s*1664525u+1013904223u;}
// Independent integer IEEE-754 binary32 -> binary16 RNE oracle. Inputs here
// are finite, but overflow/NaN handling keeps the helper well-defined.
static uint16_t half_reference(float value) {
    const uint32_t u=bits(value),sign=(u>>16)&0x8000u,exponent=(u>>23)&255u,mantissa=u&0x7fffffu;
    if(exponent==255)return uint16_t(sign|0x7c00u|(mantissa?0x0200u:0));
    const int e=int(exponent)-127;
    if(e>15)return uint16_t(sign|0x7c00u);
    if(e < -25)return uint16_t(sign);
    const uint32_t mant=mantissa|0x800000u;
    const unsigned shift=e < -14?unsigned(-e-1):13u;
    uint32_t rounded=mant>>shift;
    const uint32_t remainder=mant&((uint32_t(1)<<shift)-1u),mid=uint32_t(1)<<(shift-1u);
    rounded+=(remainder>mid||(remainder==mid&&(rounded&1u)));
    if(e < -14)return uint16_t(sign|rounded);
    return uint16_t(sign|((uint32_t(e+14)<<10)+rounded));
}
static constexpr size_t guard=128; // 256-byte alignment for half, 512 for F32.
#ifdef TEST_NATIVE
static unsigned score_cases;
#else
static unsigned operand_cases,stage_cases;
#endif
static std::vector<float> input(size_t count,unsigned pattern,uint32_t seed) {
    std::vector<float> v(count+2*guard,float_poison());
    for(size_t i=0;i<count;++i) {
        const uint32_t u=random_word(seed);
        float x=from_bits(0x3f000000u|(u&0x7fffffu))-.75f;
        if(pattern==1)x=(i&1)?0.f:-0.f;
        if(pattern==2) {
            const uint32_t midpoint=0x3f801000u+uint32_t(i%17)*0x2000u;
            x=from_bits(midpoint+uint32_t(int(i%3)-1));
            if(i&4)x=-x;
        }
        if(pattern==3)x=std::ldexp(float(int(i%31)-15),-10)*(i&1?-1.f:1.f);
        if(pattern==4)x=std::ldexp(float(int(i%15)-7),-25);
        v[guard+i]=x;
    }
    return v;
}
static void verify_half(const std::vector<__half>& h,const std::vector<float>& f,size_t count) {
    check(h.size()==count+2*guard,"half size");
    for(size_t i=0;i<h.size();++i)
        check(half_bits(h[i])==(i<guard||i>=guard+count?half_bits(half_poison()):half_reference(f[i])),
              "actual preparation bits, complete write coverage and half guards");
}
template<class T> static bool identical(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(T));
}

// PRODUCTION_SOURCE
#ifndef TEST_NATIVE
static void policy_cases() {
    unsigned cases=0;
    ds4_indexer_plan_request r{};ds4_indexer_prepared_caps c{};
    auto reset=[&]() {
        r={DS4_INDEXER_BACKEND_CUDA,DS4_GPU_PHASE_AUTO,4096,128,12000,64,128,4,false};
        c={true,false,false,false,true,1024,65536,UINT32_MAX,65535,UINT64_MAX};
    };
    auto decline=[&]() {
        ds4_indexer_prepared_plan p;std::memset(&p,0xa7,sizeof(p));const auto before=p;
        check(!ds4_indexer_prepared_build(&r,&c,&p)&&!std::memcmp(&p,&before,sizeof(p)),"policy decline leaves output unchanged");++cases;
    };
    for(auto backend:{DS4_INDEXER_BACKEND_CUDA,DS4_INDEXER_BACKEND_HIP})
    for(auto phase:{DS4_GPU_PHASE_AUTO,DS4_GPU_PHASE_PREFILL,DS4_GPU_PHASE_DECODE,DS4_GPU_PHASE_VERIFY,DS4_GPU_PHASE_BATCH_DECODE,DS4_GPU_PHASE_MIXED})
    for(uint32_t nc:{1u,127u,128u,129u,4096u,65536u})
    for(uint32_t nt:{2u,15u,16u,17u,31u,32u,33u,128u,512u}) {
        reset();r.backend=backend;r.phase=phase;r.n_comp=nc;r.n_tokens=nt;
        if(phase!=DS4_GPU_PHASE_AUTO&&phase!=DS4_GPU_PHASE_PREFILL) {decline();continue;}
        ds4_indexer_prepared_plan p{};check(ds4_indexer_prepared_build(&r,&c,&p),"admitted shape/phase");
        check(p.q_elements==uint64_t(nt)*64*128&&p.k_elements==uint64_t(nc)*128&&
            p.q_half_bytes==p.q_elements*2&&p.k_half_bytes==p.k_elements*2&&
            p.k_offset==(p.q_half_bytes+255)/256*256&&p.scratch_bytes==p.k_offset+p.k_half_bytes&&
            p.prepare_blocks==std::min<uint64_t>(65535,(p.q_elements+p.k_elements+511)/512)&&
            p.score_grid_x==(nc+127)/128&&p.score_grid_y==(nt+(backend==DS4_INDEXER_BACKEND_CUDA?31:15))/(backend==DS4_INDEXER_BACKEND_CUDA?32:16)&&
            p.shared_bytes==(backend==DS4_INDEXER_BACKEND_CUDA?43520u:45056u),"independent byte/stride/grid calculations");++cases;
    }
    reset();r.quality=true;decline();reset();r.n_tokens=1;decline();reset();r.n_tokens=0;decline();
    reset();r.n_head=63;decline();reset();r.head_dim=127;decline();reset();r.ratio=0;decline();
    reset();r.n_comp=0;decline();reset();r.n_comp=UINT32_MAX;decline();
    reset();r.n_tokens=UINT32_MAX;decline();reset();r.pos0=UINT32_MAX;decline();
    // Six is outside the API's named phases but within this enum's C++
    // representable range (0..7); casting 99 would itself be undefined.
    reset();r.phase=static_cast<ds4_gpu_execution_phase>(6);decline();reset();r.backend=DS4_INDEXER_BACKEND_METAL;decline();
    reset();c.device_supported=false;decline();reset();c.capturing=true;decline();reset();c.native_mxf4=true;decline();
    reset();c.max_threads=255;decline();reset();c.max_shared_bytes=43519;decline();
    reset();c.max_grid_x=0;decline();reset();c.max_grid_x=31;decline();reset();c.max_grid_y=3;decline();
    reset();c.max_buffer_bytes=1;decline();reset();c.register_scores=true;decline();
    reset();c.register_scores=true;r.backend=DS4_INDEXER_BACKEND_HIP;c.register_scores_supported=false;decline();
    reset();c.register_scores=true;r.backend=DS4_INDEXER_BACKEND_HIP;c.max_shared_bytes=37887;decline();
    reset();c.register_scores=true;r.backend=DS4_INDEXER_BACKEND_HIP;c.max_shared_bytes=37888;
    ds4_indexer_prepared_plan p{};check(ds4_indexer_prepared_build(&r,&c,&p)&&p.shared_bytes==37888&&p.register_scores,"explicit supported register candidate resource bound");++cases;
    reset();r.n_tokens=INT32_MAX;r.pos0=0;r.n_comp=1;c.max_grid_y=UINT32_MAX;
    check(ds4_indexer_prepared_build(&r,&c,&p)&&p.prepare_blocks==65535,"huge representable producer grid caps without overflow");++cases;
    c.max_grid_x=7;check(ds4_indexer_prepared_build(&r,&c,&p)&&p.prepare_blocks==7,"device grid cap retains grid-stride coverage");++cases;
    reset();check(!ds4_indexer_prepared_build(nullptr,&c,&p)&&!ds4_indexer_prepared_build(&r,nullptr,&p)&&!ds4_indexer_prepared_build(&r,&c,nullptr),"null planner inputs");cases+=3;
    check(ds4_indexer_prepared_build(&r,&c,&p),"buffer policy base plan");
    ds4_indexer_prepared_buffers b{};
    auto reset_buffers=[&]() {b={0x10000000,0x20000000,0x30000000,0x40000000,0x50000000,
        p.base.q_bytes,p.base.index_bytes,p.base.weight_bytes,p.base.score_bytes,p.scratch_bytes};};
    auto reject_buffer=[&]() {check(!ds4_indexer_prepared_buffers_valid(&p,&b),"invalid buffer geometry/alias rejected before producer");++cases;};
    reset_buffers();check(ds4_indexer_prepared_buffers_valid(&p,&b),"valid disjoint exact-capacity buffers");++cases;
    for(unsigned i=0;i<5;++i) {
        reset_buffers();uint64_t *capacity[]={&b.q_bytes,&b.key_bytes,&b.weight_bytes,&b.score_bytes,&b.scratch_bytes};--*capacity[i];reject_buffer();
        reset_buffers();uintptr_t *address[]={&b.q,&b.keys,&b.weights,&b.scores,&b.scratch};*address[i]=0;reject_buffer();
        reset_buffers();uintptr_t *misaligned[]={&b.q,&b.keys,&b.weights,&b.scores,&b.scratch};++*misaligned[i];reject_buffer();
        reset_buffers();uintptr_t *overflow[]={&b.q,&b.keys,&b.weights,&b.scores,&b.scratch};*overflow[i]=UINTPTR_MAX&~uintptr_t(255);reject_buffer();
    }
    for(unsigned writer=0;writer<2;++writer)for(unsigned reader=0;reader<4;++reader) {
        reset_buffers();uintptr_t *read[]={&b.q,&b.keys,&b.weights,writer?&b.scores:&b.scratch};
        *(writer?&b.scratch:&b.scores)=*read[reader];reject_buffer();
    }
    reset_buffers();b.keys=b.q;b.weights=b.q;check(ds4_indexer_prepared_buffers_valid(&p,&b),"read-only operands may alias");++cases;
    reset_buffers();b.scratch=b.scores+p.base.score_bytes;check(ds4_indexer_prepared_buffers_valid(&p,&b),"adjacent writer ranges do not overlap");++cases;
    reset_buffers();check(!ds4_indexer_prepared_buffers_valid(nullptr,&b)&&!ds4_indexer_prepared_buffers_valid(&p,nullptr),"null buffer validation arguments");cases+=2;
    std::printf("PASS: %u actual planner/buffer policy cases; phases, architecture capabilities, capture, MXF4, checked sizes, aliases and non-mutating decline.\n",cases);
}
#endif

#ifndef TEST_NATIVE
static void prepare_host(__half *qh,__half *kh,const float *q,const float *k,uint64_t nq,uint64_t nk,unsigned blocks) {
    gridDim.x=blocks;blockDim.x=256;
    for(blockIdx.x=0;blockIdx.x<blocks;++blockIdx.x)
        for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x)
            ds4_indexer_prepare_f16_kernel(qh,kh,q,k,nq,nk);
}
static void operand_case(size_t nq,size_t nk,unsigned pattern,unsigned blocks) {
    auto q=input(nq,pattern,8191),k=input(nk,pattern,371);const auto qo=q,ko=k;
    std::vector<__half> qh(nq+2*guard,half_poison()),kh(nk+2*guard,half_poison());
    prepare_host(qh.data()+guard,kh.data()+guard,q.data()+guard,k.data()+guard,nq,nk,blocks);
    verify_half(qh,q,nq);verify_half(kh,k,nk);
    check(identical(q,qo)&&identical(k,ko),"prepare leaves F32 operands immutable");++operand_cases;
}
// HOST_STAGING_CASES
#else
static void gpu_check(GPU(Error_t) status,const char *label) {
    if(status!=GPU(Success)) {std::fprintf(stderr,"GPU %s: %s\n",label,GPU(GetErrorString)(status));std::exit(1);}
}
template<class T> struct device_buffer {
    T *ptr=nullptr;size_t count;
    explicit device_buffer(size_t n):count(n) {gpu_check(GPU(Malloc)(reinterpret_cast<void **>(&ptr),n*sizeof(T)),"allocate");}
    ~device_buffer() {if(ptr)gpu_check(GPU(Free)(ptr),"free");}
    void put(const std::vector<T>& v) {check(v.size()==count,"upload size");gpu_check(GPU(Memcpy)(ptr,v.data(),count*sizeof(T),GPU(MemcpyHostToDevice)),"upload");}
    std::vector<T> get() const {std::vector<T> v(count);gpu_check(GPU(Memcpy)(v.data(),ptr,count*sizeof(T),GPU(MemcpyDeviceToHost)),"readback");return v;}
};
static const char *arm_name(unsigned arm) {return arm==0?"original F32 staging":arm==1?"prepare + LDS consumer":arm==2?"prepare + register consumer":"register consumer F32 staging";}
static void score_check(const std::vector<float>& reference,const std::vector<float>& actual,
        uint32_t nc,uint32_t nt,uint32_t pos0,uint32_t ratio,bool causal) {
    if(!identical(reference,actual))for(size_t i=0;i<reference.size();++i)if(bits(reference[i])!=bits(actual[i])) {
        std::fprintf(stderr,"score mismatch C=%u T=%u pos=%u index=%zu baseline=%08x candidate=%08x\n",nc,nt,pos0,i,bits(reference[i]),bits(actual[i]));
        check(false,"native score bits differ");
    }
    for(size_t i=0;i<actual.size();++i) {
        if(i<guard||i>=guard+size_t(nc)*nt)check(bits(actual[i])==bits(float_poison()),"score guards");
        else {
            const size_t j=i-guard;const bool hidden=causal&&j%nc>=(uint64_t(pos0)+j/nc+1)/ratio;
            check(hidden?bits(actual[i])==0xff800000u:std::isfinite(actual[i]),"causal -inf and every visible output written finite");
        }
    }
}
template<class K> static void resources(const char *label,K kernel) {
    GPU(FuncAttributes) attr{};int resident=0;
    gpu_check(GPU(FuncGetAttributes)(&attr,reinterpret_cast<const void *>(kernel)),"kernel attributes");
    gpu_check(GPU(OccupancyMaxActiveBlocksPerMultiprocessor)(&resident,reinterpret_cast<const void *>(kernel),256,0),"kernel occupancy");
    std::printf("RESOURCE %s registers_per_thread=%d static_LDS_bytes=%zu local_bytes=%zu max_threads=%d active_blocks_per_multiprocessor=%d\n",
        label,attr.numRegs,size_t(attr.sharedSizeBytes),size_t(attr.localSizeBytes),attr.maxThreadsPerBlock,resident);
}
static void native_case(uint32_t nc,uint32_t nt,unsigned pattern,uint32_t pos0,bool causal,float scale,bool bench=false) {
    constexpr uint32_t nh=64,hd=128,ratio=4;
    const size_t nq=size_t(nt)*nh*hd,nk=size_t(nc)*hd,ns=size_t(nt)*nc;
    const auto q=input(nq,pattern,719),k=input(nk,pattern,731),w=input(size_t(nt)*nh,pattern==1?1:0,137);
    device_buffer<float> dq(q.size()),dk(k.size()),dw(w.size()),ref(ns+2*guard),out(ns+2*guard);
    // Same contiguous Q-then-K layout as the checked plan. H64/D128 makes
    // the Q byte span a multiple of 256, so no internal alignment pad is needed.
    device_buffer<__half> scratch(nq+nk+2*guard);
    __half *const qh=scratch.ptr+guard,*const kh=qh+nq;
    dq.put(q);dk.put(k);dw.put(w);
    const std::vector<float> initial(ns+2*guard,float_poison());ref.put(initial);out.put(initial);
    scratch.put(std::vector<__half>(nq+nk+2*guard,half_poison()));
    auto check_preparation=[&]() {
        const auto h=scratch.get();
        for(size_t i=0;i<h.size();++i) {
            const uint16_t expected=i<guard||i>=guard+nq+nk?half_bits(half_poison()):
                i<guard+nq?half_reference(q[i]):half_reference(k[guard+i-guard-nq]);
            check(half_bits(h[i])==expected,"native contiguous Q/K preparation RNE bits and outer guards");
        }
    };
#ifdef TEST_HIP
    constexpr unsigned arms=4,tile_m=16;
#else
    constexpr unsigned arms=2,tile_m=32;
#endif
    const dim3 grid((nc+127)/128,(nt+tile_m-1)/tile_m);
    const unsigned prepare_blocks=unsigned(std::min<uint64_t>(65535,((uint64_t(nq)+nk)/2+255)/256));
    const ds4_indexer_plan_request request={
#ifdef TEST_HIP
        DS4_INDEXER_BACKEND_HIP,
#else
        DS4_INDEXER_BACKEND_CUDA,
#endif
        DS4_GPU_PHASE_PREFILL,nc,nt,pos0,nh,hd,ratio,false};
    const ds4_indexer_prepared_buffers buffers={reinterpret_cast<uintptr_t>(dq.ptr+guard),
        reinterpret_cast<uintptr_t>(dk.ptr+guard),reinterpret_cast<uintptr_t>(dw.ptr+guard),
        reinterpret_cast<uintptr_t>(out.ptr+guard),reinterpret_cast<uintptr_t>(qh),
        nq*4,nk*4,size_t(nt)*nh*4,ns*4,(nq+nk)*2};
    int current_device;gpu_check(GPU(GetDevice)(&current_device),"device for launch limits");
#ifdef TEST_HIP
    hipDeviceProp_t limits{};
#else
    cudaDeviceProp limits{};
#endif
    gpu_check(GPU(GetDeviceProperties)(&limits,current_device),"device launch limits");
    auto launch=[&](unsigned arm) {
        if((arm==1||arm==2)&&nt>=2) {
            const ds4_indexer_prepared_caps caps={true,false,false,arm==2,true,
                uint32_t(limits.maxThreadsPerBlock),uint32_t(limits.sharedMemPerBlock),
                uint32_t(limits.maxGridSize[0]),uint32_t(limits.maxGridSize[1]),UINT64_MAX};
            check(ds4_indexer_prepared_try(&request,&caps,&buffers,scale,int(causal),nullptr)==1,
                "actual prepared launcher must enqueue producer and chosen consumer");
            return;
        }
        if(arm==1||arm==2) {
            // T=1 is deliberately outside the prefill API, but is still a
            // useful raw-kernel tail test. Benchmarks always use the API.
            ds4_indexer_prepare_f16_kernel<<<prepare_blocks,256>>>(qh,kh,dq.ptr+guard,dk.ptr+guard,nq,nk);
            gpu_check(GPU(GetLastError)(),"prepare launch");
        }
        if(!arm)indexer_scores_wmma128_kernel<<<grid,256>>>(ref.ptr+guard,dq.ptr+guard,dw.ptr+guard,dk.ptr+guard,nc,nt,pos0,nh,hd,ratio,scale,int(causal));
        else if(arm==1)indexer_scores_wmma128_prepared_kernel<<<grid,256>>>(out.ptr+guard,qh,dw.ptr+guard,kh,nc,nt,pos0,nh,hd,ratio,scale,int(causal));
#ifdef TEST_HIP
        else if(arm==2)ds4_rocm_indexer_scores_registers_kernel<true><<<grid,256>>>(out.ptr+guard,qh,dw.ptr+guard,kh,nc,nt,pos0,nh,hd,ratio,scale,int(causal));
        else ds4_rocm_indexer_scores_registers_kernel<false><<<grid,256>>>(out.ptr+guard,dq.ptr+guard,dw.ptr+guard,dk.ptr+guard,nc,nt,pos0,nh,hd,ratio,scale,int(causal));
#endif
        gpu_check(GPU(GetLastError)(),"score launch");
    };
    launch(0);gpu_check(GPU(DeviceSynchronize)(),"baseline completion");const auto reference=ref.get();
    for(unsigned arm=1;arm<arms;++arm) {
        out.put(initial);launch(arm);gpu_check(GPU(DeviceSynchronize)(),"candidate completion");
        score_check(reference,out.get(),nc,nt,pos0,ratio,causal);
        if(arm!=3)check_preparation();++score_cases;
        if(!bench)continue;
        constexpr unsigned samples_per_arm=20,warmup_blocks=4;
        const unsigned repeats=ns<=size_t(128)*4096?8:ns<=size_t(128)*16384?4:1;
        std::vector<double> gpu_samples[2],wall_samples[2];
        GPU(Event_t) start,stop;gpu_check(GPU(EventCreate)(&start),"event");gpu_check(GPU(EventCreate)(&stop),"event");
        for(unsigned block=0;block<warmup_blocks+samples_per_arm/2;++block)for(unsigned step=0;step<4;++step) {
            const unsigned which=(step==1||step==2)?1:0;
            const auto wall_start=std::chrono::steady_clock::now();gpu_check(GPU(EventRecord)(start,nullptr),"event start");
            for(unsigned i=0;i<repeats;++i)launch(which?arm:0);
            gpu_check(GPU(EventRecord)(stop,nullptr),"event stop");gpu_check(GPU(EventSynchronize)(stop),"event completion");
            const double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-wall_start).count()/repeats;
            float elapsed;gpu_check(GPU(EventElapsedTime)(&elapsed,start,stop),"elapsed");
            if(block>=warmup_blocks) {gpu_samples[which].push_back(elapsed*1000/repeats);wall_samples[which].push_back(wall);}
        }
        gpu_check(GPU(EventDestroy)(start),"event free");gpu_check(GPU(EventDestroy)(stop),"event free");
        std::printf("BENCH C=%u T=%u H64 D128 pos=%u causal=%d scale=%.9g repeats=%u samples_per_arm=%u warmup_ABBA_blocks=%u\n",nc,nt,pos0,int(causal),scale,repeats,samples_per_arm,warmup_blocks);
        double median[2]={};
        for(unsigned side=0;side<2;++side)for(unsigned metric=0;metric<2;++metric) {
            auto values=metric?wall_samples[side]:gpu_samples[side];double sum=0;
            std::printf("  %s %s_raw_us=",arm_name(side?arm:0),metric?"wall":"GPU");
            for(size_t i=0;i<values.size();++i) {sum+=values[i];std::printf("%s%.5f",i?",":"",values[i]);}
            std::sort(values.begin(),values.end());const double m=(values[9]+values[10])*.5;
            std::printf(" mean=%.5f median=%.5f range=%.5f..%.5f\n",sum/values.size(),m,values.front(),values.back());if(!metric)median[side]=m;
        }
        std::printf("  score route median GPU throughput change=%+.3f%%; warm fixed inputs; preparation %s; excludes top-K, model, allocations and host transfers.\n",100*(median[0]/median[1]-1),arm==3?"absent in this explicit F32 register control":"included EVERY candidate call");
        score_check(ref.get(),out.get(),nc,nt,pos0,ratio,causal);check_preparation();std::fflush(stdout);
    }
    check(identical(dq.get(),q)&&identical(dk.get(),k)&&identical(dw.get(),w),"device Q/K/weights and guards immutable");
}
#endif

int main(int argc,char **argv) {
#ifndef TEST_NATIVE
    (void)argc;(void)argv;policy_cases();
    for(auto pair:{std::pair<size_t,size_t>{0,0},{2,2},{128,128},{256,384},{8192,3968},{8192*17,128*129}})
        for(unsigned pattern=0;pattern<5;++pattern)for(unsigned blocks:{1u,3u,31u})operand_case(pair.first,pair.second,pattern,blocks);
    staging_cases();
    std::printf("PASS: %u actual-source producer cases and %u actual-source staging cases; independent RNE bits, guards, tails and immutable inputs. Host tests do not emulate WMMA or GPU scheduling.\n",operand_cases,stage_cases);
#else
    check(argc==1||(argc==2&&!std::strcmp(argv[1],"--bench")),"usage: native fixture [--bench]");
    int device=0;gpu_check(GPU(GetDevice)(&device),"device");
#ifdef TEST_HIP
    hipDeviceProp_t prop{};
#else
    cudaDeviceProp prop{};
#endif
    gpu_check(GPU(GetDeviceProperties)(&prop,device),"properties");
#ifdef TEST_HIP
    std::printf("HIP device=%s arch=%s wave=%d\n",prop.name,prop.gcnArchName,prop.warpSize);
    check(!std::strncmp(prop.gcnArchName,"gfx1151",7)&&prop.warpSize==32,"native register fixture requires gfx1151 wave32");
#else
    std::printf("CUDA device=%s SM=%d.%d warp=%d\n",prop.name,prop.major,prop.minor,prop.warpSize);
    check(prop.major>=7&&prop.warpSize==32,"native fixture requires SM70+ warp32");
#endif
    resources("prepare",ds4_indexer_prepare_f16_kernel);resources("baseline",indexer_scores_wmma128_kernel);resources("prepared-LDS",indexer_scores_wmma128_prepared_kernel);
#ifdef TEST_HIP
    resources("prepared-register",ds4_rocm_indexer_scores_registers_kernel<true>);
    resources("F32-register",ds4_rocm_indexer_scores_registers_kernel<false>);
#endif
    for(auto shape:{std::pair<uint32_t,uint32_t>{1,1},{31,7},{32,8},{33,9},{127,15},{128,16},{129,17},{511,31},{512,32},{513,33},{1025,127},{4096,128},{513,129}})
        for(unsigned pattern=0;pattern<4;++pattern) {
            native_case(shape.first,shape.second,pattern,pattern==0?0:shape.first*4-shape.second%4,true,.011048543457f);
        }
    for(float scale:{0.f,-0.f,-.011048543457f,.1f})for(unsigned pattern:{0u,2u,3u,4u})
        native_case(513,9,pattern,1024,true,scale);
    native_case(129,17,2,0,false,.011048543457f);
    if(argc==2)for(uint32_t nc:{4096u,16384u,65536u})for(uint32_t nt:{128u,512u})native_case(nc,nt,0,nc*4-nt,true,.011048543457f,true);
    std::printf("PASS: %u native bitwise candidate score comparisons, preparation RNE bits, causal masks, guards and immutable inputs; benchmark is prepare+score only.\n",score_cases);
#endif
}
