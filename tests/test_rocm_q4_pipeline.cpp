// SPDX-License-Identifier: MIT
// Host helper oracle; Python inserts the existing native K128 fixture prefix.
#ifndef TEST_NATIVE
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using __half = _Float16;
struct half2 { _Float16 x, y; };
static half2 __floats2half2_rn(float x, float y) { return {__half(x), __half(y)}; }
#include "rocm/ds4_rocm_q4_pipeline.cuh"
namespace pipeline = ds4_rocm_q4_pipeline;
static void check(bool ok, const char *why) {
    if (!ok) { std::fprintf(stderr,"Q4 pipeline FAIL: %s\n",why); std::exit(1); }
}
static uint16_t half_bits(__half h) { uint16_t u; std::memcpy(&u,&h,2); return u; }

template<typename A> static size_t host_case(uint32_t k, uint32_t n, bool padded) {
    const uint64_t stride = k + (padded ? 16u : 0u);
    const uint64_t group_stride = stride*n + (padded ? 16u : 0u);
    const size_t total = group_stride*2u;
    std::vector<A> x(total);
    check((uintptr_t(x.data()) & 15u) == 0u,"host source alignment");
    for (size_t i=0;i<total;++i) {
        if constexpr (std::is_same<A,__half>::value) {
            // Include all binary16 encodings; F16 staging is pure bit transport.
            uint16_t bits=uint16_t(i*97u); std::memcpy(&x[i],&bits,2);
        } else {
            uint32_t bits = 0x3d000000u + uint32_t((i*7919u)&0x02ffffffu);
            if(i&1u)bits|=0x80000000u;
            if(i%13u==0u)bits=0x33800000u; // half subnormal
            if(i%17u==0u)bits=0x33000000u; // half ties-to-even boundary
            if(i%19u==0u)bits=0x80000000u;
            std::memcpy(&x[i],&bits,4);
        }
    }
    const auto original=x;
    constexpr size_t guard=16;
    size_t count=0;
    for(uint32_t group=0;group<2u;++group)for(uint32_t tok0=0;tok0<n;tok0+=64u){
        std::vector<pipeline::packet<A>> prefetched(pipeline::threads);
        for(uint32_t tid=0;tid<pipeline::threads;++tid)
            prefetched[tid]=pipeline::load_vector(x.data(),n,tok0,group,0u,
                tid*pipeline::packet<A>::lanes,stride,group_stride);
        for(uint32_t k0=0;k0<k;k0+=128u){
            std::vector<uint16_t> actual(guard+64u*144u+guard,0x5ad5u),expected=actual;
            for(uint32_t t=0;t<64u;++t)for(uint32_t c=0;c<128u;++c){
                uint16_t value=0;
                if(tok0+t<n){
                    const size_t offset=group*group_stride+uint64_t(tok0+t)*stride+k0+c;
                    if constexpr(std::is_same<A,__half>::value)std::memcpy(&value,&x[offset],2);
                    else value=half_bits(__half(x[offset]));
                }
                expected[guard+t*144u+c]=value;
            }
            for(uint32_t tid=0;tid<pipeline::threads;++tid)
                pipeline::stage_thread(reinterpret_cast<_Float16*>(actual.data()+guard),
                    prefetched[tid],x.data(),n,tok0,group,k0,tid,stride,group_stride);
            check(actual==expected,"prefetched staging, token tails, stride padding and LDS guards");
            // The next packet is read while the current tile remains intact.
            if(k0+128u<k)for(uint32_t tid=0;tid<pipeline::threads;++tid)
                prefetched[tid]=pipeline::load_vector(x.data(),n,tok0,group,k0+128u,
                    tid*pipeline::packet<A>::lanes,stride,group_stride);
            check(actual==expected,"lookahead must not overwrite current LDS");
            ++count;
        }
    }
    check(std::memcmp(x.data(),original.data(),total*sizeof(A))==0,"input immutable");
    return count;
}
int main(){
    const auto out=reinterpret_cast<void*>(uintptr_t(0x100000000ull));
    const auto w=reinterpret_cast<void*>(uintptr_t(0x200000000ull));
    const auto x=reinterpret_cast<void*>(uintptr_t(0x300000000ull));
    auto admit=[&](uint32_t n=512u,uint32_t g=1u,uint32_t k=1024u,uint32_t m=32768u,
                   uint64_t rb=576u,uint64_t xs=1024u,uint64_t xgs=0u,uint64_t os=32768u,
                   const void *op=nullptr,const void *wp=nullptr,const void *xp=nullptr,
                   bool half=false,bool gfx=true){
        return pipeline::admit(n,g,k,m,rb,xs,xgs,os,op?op:out,wp?wp:w,xp?xp:x,half,gfx);
    };
    check(admit()&&admit(512u,1u,1024u,32768u,576u,1024u,0u,32768u,nullptr,nullptr,nullptr,true),
          "valid F32/F16 admission");
    check(!admit(512u,1u,1024u,32768u,576u,1024u,0u,32768u,nullptr,nullptr,nullptr,false,false),"device gate");
    check(!admit(0u)&&!admit(UINT32_MAX)&&!admit(4194241u),"token/grid bounds");
    check(!admit(512u,2u)&&!admit(512u,1u,1025u),"group/K scope");
    check(!admit(512u,1u,1024u,32768u,575u)&&!admit(512u,1u,1024u,32768u,577u),"weight layout");
    check(!admit(512u,1u,1024u,32768u,576u,0u)&&!admit(512u,1u,1024u,32768u,576u,1025u),"input stride");
    check(!admit(512u,1u,1024u,32768u,576u,1024u,1024u)&&
          !admit(512u,1u,1024u,32768u,576u,1024u,0u,0u),"group/output stride");
    check(!admit(512u,1u,1024u,32768u,UINT64_MAX-1u)&&
          !admit(512u,1u,1024u,32768u,576u,UINT64_MAX-15u)&&
          !admit(512u,1u,1024u,32768u,576u,1024u,0u,UINT64_MAX),"span overflow");
    check(!admit(512u,1u,1024u,32768u,576u,1024u,0u,32768u,x)&&
          !admit(512u,1u,1024u,32768u,576u,1024u,0u,32768u,w),"output overlaps input/weights");
    for(unsigned operand=0;operand<3u;++operand){
        const void *unaligned=reinterpret_cast<void*>(uintptr_t(operand==0u?out:operand==1u?w:x)+1u);
        check(!admit(512u,1u,1024u,32768u,576u,1024u,0u,32768u,
                     operand==0u?unaligned:nullptr,operand==1u?unaligned:nullptr,operand==2u?unaligned:nullptr),"operand alignment");
    }
    check(!admit(512u,1u,1024u,32768u,576u,1024u,0u,32768u,nullptr,nullptr,
                 reinterpret_cast<void*>(UINTPTR_MAX-15u)),"pointer end overflow");
    size_t count=0;
    for(uint32_t k:{256u,512u,1024u,2048u})
        for(uint32_t n:{1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,128u,129u,512u})
            for(bool padded:{false,true}){
                count+=host_case<float>(k,n,padded);
                count+=host_case<__half>(k,n,padded);
            }
    std::printf("PASS Q4 pipeline: %zu staged tiles; raw F32 lookahead/F16 bit transport, "
                "two groups, padded strides, tails and guards.\n",count);
    std::puts("Host-only: WMMA, scheduling/ISA, VGPR spills, occupancy and speed require gfx1151.");
}
#else
// NATIVE_TEST
#include "rocm/ds4_rocm_q4_pipeline.cuh"
// PRODUCTION_PIPELINE_HOOK

template<typename A> static void pipeline_resources(const char *label) {
    hipFuncAttributes base{},candidate{};
    const auto old_kernel=rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel<256u,16u,1u,A>;
    const auto new_kernel=rocm_matmul_q4_K_prefill_wmma_k128_pipeline_rowtile_strided_kernel<256u,16u,1u,A>;
    hip_check(hipFuncGetAttributes(&base,reinterpret_cast<const void*>(old_kernel)),"baseline resource query");
    hip_check(hipFuncGetAttributes(&candidate,reinterpret_cast<const void*>(new_kernel)),"pipeline resource query");
    int base_blocks=0,candidate_blocks=0;
    hip_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&base_blocks,old_kernel,512,0),"baseline occupancy");
    hip_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&candidate_blocks,new_kernel,512,0),"pipeline occupancy");
    std::printf("resources,%s,baseline_vgpr,%d,pipeline_vgpr,%d,baseline_local_bytes,%zu,pipeline_local_bytes,%zu,"
                "baseline_LDS,%zu,pipeline_LDS,%zu,baseline_blocks,%d,pipeline_blocks,%d\n",
                label,base.numRegs,candidate.numRegs,base.localSizeBytes,candidate.localSizeBytes,
                base.sharedSizeBytes,candidate.sharedSizeBytes,base_blocks,candidate_blocks);
    check(base.sharedSizeBytes==candidate.sharedSizeBytes,"pipeline must not add an LDS buffer");
}

static void native_pipeline_case(uint32_t k,uint32_t m,uint32_t n,bool bench){
    constexpr size_t guard=16;
    constexpr uint32_t sentinel=0x7fc12345u;
    const size_t xe=size_t(n)*k,oe=size_t(n)*m;
    std::vector<cuda_block_q4_K> weights(size_t(m)*(k/256u)+2u);
    uint32_t random=0x24e18fd1u;
    auto next=[&](){random^=random<<13;random^=random>>17;random^=random<<5;return random;};
    for(auto &w:weights){
        w.d=bits(__half(0.03125f));w.dmin=bits(__half(0.015625f));
        for(auto &s:w.scales)s=uint8_t(next());for(auto&q:w.qs)q=uint8_t(next());
    }
    std::vector<float> x(4u+xe+4u,-1234.f);
    for(size_t i=0;i<xe;++i){
        uint32_t v=0x3e000000u+(next()&0x01ffffffu);
        if(i&1u)v|=0x80000000u;
        if(i%23u==0u)v=0x33000000u;
        if(i%29u==0u)v=0x80000000u;
        std::memcpy(x.data()+4u+i,&v,4);
    }
    std::vector<uint16_t> half_storage(guard+xe+guard,0x5ad5u);
    std::vector<uint32_t> poisoned(guard+oe+guard,sentinel);
    Device<cuda_block_q4_K> dw(weights);Device<float> dx(x);
    Device<uint16_t> dh(half_storage);
    Device<uint32_t> af(poisoned),bf(poisoned),ah(poisoned),bh(poisoned);
    const uint64_t row_bytes=uint64_t(k/256u)*144u;
    const dim3 grid((m+255u)/256u,(n+63u)/64u,1u);
    auto conversion=[&](){
        f32_to_f16_kernel<<<(xe+255u)/256u,256u>>>(reinterpret_cast<__half*>(dh.ptr+guard),dx.ptr+4u,xe);
        hip_check(hipGetLastError(),"activation conversion");
    };
    auto baseline_f32=[&](){
        rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel<256u,16u,1u,float><<<grid,512u>>>(
            reinterpret_cast<float*>(af.ptr+guard),reinterpret_cast<const char*>(dw.ptr+1u),dx.ptr+4u,
            n,1u,k,m,row_bytes,k,0u,m);
        hip_check(hipGetLastError(),"baseline F32");
    };
    auto pipeline_f32=[&](){
        check(ds4_rocm_bench_q4_K_wmma_k128_pipeline_enqueue(bf.ptr+guard,dw.ptr+1u,dx.ptr+4u,
            n,1u,k,m,row_bytes,k,0u,m,false)==1,"pipeline F32 hook");
    };
    auto baseline_f16=[&](){
        conversion();
        rocm_matmul_q4_K_prefill_wmma_k128_p144_rowtile_strided_kernel<256u,16u,1u,__half><<<grid,512u>>>(
            reinterpret_cast<float*>(ah.ptr+guard),reinterpret_cast<const char*>(dw.ptr+1u),
            reinterpret_cast<const __half*>(dh.ptr+guard),n,1u,k,m,row_bytes,k,0u,m);
        hip_check(hipGetLastError(),"baseline F16");
    };
    auto pipeline_f16=[&](){
        conversion();
        check(ds4_rocm_bench_q4_K_wmma_k128_pipeline_enqueue(bh.ptr+guard,dw.ptr+1u,dh.ptr+guard,
            n,1u,k,m,row_bytes,k,0u,m,true)==1,"pipeline F16 hook");
    };
    for(int repeat=0;repeat<3;++repeat){baseline_f32();pipeline_f32();baseline_f16();pipeline_f16();}
    const auto ref=af.read(),got=bf.read(),href=ah.read(),hgot=bh.read();
    check(ref==got&&ref==href&&ref==hgot,"all four kernels must produce bitwise identical FP32 output");
    for(size_t i=0;i<ref.size();++i){
        if(i<guard||i>=guard+oe)check(ref[i]==sentinel,"output guards");
        else check((ref[i]&0x7f800000u)!=0x7f800000u,"finite written output");
    }
    const auto half_result=dh.read();
    for(size_t i=0;i<half_result.size();++i){
        if(i<guard||i>=guard+xe)check(half_result[i]==0x5ad5u,"half guards");
        else check(half_result[i]==bits(__half(x[4u+i-guard])),"conversion oracle");
    }
    const auto wr=dw.read();const auto xr=dx.read();
    check(std::memcmp(wr.data(),weights.data(),weights.size()*sizeof(weights[0]))==0,"weight guards/input immutable");
    check(std::memcmp(xr.data(),x.data(),x.size()*sizeof(x[0]))==0,"activation guards/input immutable");
    std::printf("PASS native pipeline K%u M%u N%u F32/F16 bitwise, guards, three repeated dispatches\n",k,m,n);
    if(bench){
        for(unsigned mode=0;mode<2u;++mode){
            std::vector<double> base,candidate;
            for(unsigned sample=0;sample<14u;++sample){
                double a,b;
                auto old_run=[&](){if(mode)baseline_f16();else baseline_f32();};
                auto new_run=[&](){if(mode)pipeline_f16();else pipeline_f32();};
                if(sample&1u){b=elapsed(new_run,4);a=elapsed(old_run,4);}
                else{a=elapsed(old_run,4);b=elapsed(new_run,4);}
                if(sample<2u)continue;
                base.push_back(a);candidate.push_back(b);
                std::printf("sample,N%u,%s,%u,baseline_us,%.6f,pipeline_us,%.6f\n",n,mode?"copy+F16":"F32",sample-2u,a,b);
            }
            std::sort(base.begin(),base.end());std::sort(candidate.begin(),candidate.end());
            const double a=(base[5]+base[6])*0.5,b=(candidate[5]+candidate[6])*0.5;
            std::printf("median,N%u,%s,baseline_us,%.6f,pipeline_us,%.6f,speedup_percent,%.3f,allocation_excluded\n",
                        n,mode?"copy+F16":"F32",a,b,100.0*(a/b-1.0));
        }
    }
}
int main(int argc,char**argv){
    const int device=argc>1?std::atoi(argv[1]):0;
    const bool bench=argc>2&&std::strcmp(argv[2],"--bench")==0;
    hip_check(hipSetDevice(device),"device");hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop,device),"device query");
    check(std::strncmp(prop.gcnArchName,"gfx1151",7)==0&&prop.warpSize==32,"gfx1151 wave32 required");
    std::printf("GPU %s %s; benchmark-only pipeline, no production dispatch\n",prop.name,prop.gcnArchName);
    pipeline_resources<float>("F32");pipeline_resources<__half>("F16");
    for(uint32_t n:{1u,15u,16u,17u,63u,64u,65u,129u})native_pipeline_case(256u,8193u,n,false);
    native_pipeline_case(512u,8192u,128u,false);
    for(uint32_t n:{128u,256u,257u,512u,1024u,2048u})native_pipeline_case(1024u,32768u,n,bench&&n!=257u);
}
#endif
