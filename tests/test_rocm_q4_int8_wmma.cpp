// SPDX-License-Identifier: MIT
// TEST ONLY: exact Q4_K x Q8_K on RDNA3 INT8 WMMA. No production dispatch.
// Run through test_rocm_q4_int8_wmma.py; host mode checks the lane layout and
// arithmetic model, never the hardware instruction or its occupancy.
// RDNA3 input replication and accumulator mapping:
// https://gpuopen.com/learn/wmma_on_rdna3/ and cuda/mmq/mma.cuh.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#ifdef TEST_NATIVE
#include <hip/hip_runtime.h>
#define HD __host__ __device__
#else
#define HD
#define __device__
#define __forceinline__ inline
struct alignas(16) int4 { int32_t x,y,z,w; };
#endif
struct alignas(4) cuda_block_q4_K { uint16_t d,dmin; uint8_t scales[12],qs[128]; };
struct cuda_block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(cuda_block_q4_K)==144 && sizeof(cuda_block_q8_K)==292, "GGUF ABI");
enum { ROCM_Q4_PREFILL_TOKEN_TILE=8 };
HD static float dev_f16_to_f32(uint16_t bits) { _Float16 h; __builtin_memcpy(&h,&bits,2); return float(h); }
__device__ static int32_t __dp4a(int32_t a,int32_t b,int32_t acc) {
#ifdef TEST_NATIVE
    union bits4 { int32_t i; char4 v; } av,bv; av.i=a; bv.i=b;
    return amd_mixed_dot(av.v,bv.v,acc,false);
#else
    for(unsigned i=0;i<4;++i) acc+=int32_t(int8_t(uint32_t(a)>>(8*i)))*int32_t(int8_t(uint32_t(b)>>(8*i)));
    return acc;
#endif
}
#include "rocm/ds4_rocm_q4_dot.cuh"

HD static unsigned result_row(unsigned lane) { return lane&15u; }
HD static unsigned result_col(unsigned lane,unsigned item) { return 2u*item+(lane>>4u); }
HD static unsigned q4_value(const cuda_block_q4_K& w,unsigned group,unsigned k) {
    return (w.qs[(group/2)*32+k]>>((group&1u)*4))&15u;
}
HD static uint32_t weight_word(const cuda_block_q4_K& w,unsigned g,unsigned k0) {
    uint32_t packed=0;
    for(unsigned byte=0;byte<4;++byte)packed|=q4_value(w,g,k0+byte)<<(8*byte);
    return packed;
}
HD static uint32_t activation_word(const cuda_block_q8_K& x,unsigned g,unsigned k0) {
    uint32_t packed=0;
    for(unsigned byte=0;byte<4;++byte)packed|=uint32_t(uint8_t(x.qs[g*32+k0+byte]))<<(8*byte);
    return packed;
}
HD static unsigned metadata(const cuda_block_q4_K& w,unsigned g,bool minimum) {
    if(g<4) return w.scales[g+(minimum?4:0)]&63u;
    return minimum ? (w.scales[g+4]>>4)|((w.scales[g]>>6)<<4) :
                     (w.scales[g+4]&15u)|((w.scales[g-4]>>6)<<4);
}
HD static float finish_block(float acc,float yd,const cuda_block_q4_K& w,int dot,int mins) {
    const float xd=dev_f16_to_f32(w.d), xmin=dev_f16_to_f32(w.dmin);
    // Same FP32 expression and block order as the production dot helper.
    acc += yd*xd*float(dot)-yd*xmin*float(mins);
    return acc;
}
HD static float reduce8(const float p[8]) {
    // Lane zero of width-eight shfl_down with offsets 4,2,1.
    return ((p[0]+p[4])+(p[2]+p[6]))+((p[1]+p[5])+(p[3]+p[7]));
}

#ifdef TEST_NATIVE
using i4 = int32_t __attribute__((ext_vector_type(4)));
using i8 = int32_t __attribute__((ext_vector_type(8)));
__global__ static void candidate(float *out,const cuda_block_q4_K *w,
        const cuda_block_q8_K *x,unsigned m,unsigned n,unsigned nb) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ && defined(__gfx1151__)
    const unsigned lane=threadIdx.x&31u, wave=threadIdx.x>>5u;
    const unsigned row=blockIdx.x*16+result_row(lane), token=blockIdx.y*16+(lane&15u);
    float acc[8]={};
    for(unsigned b=wave;b<nb;b+=8) {
        // Invalid matrix rows/columns duplicate row zero and are never stored.
        // Do not create address-taken local zero blocks: they would spill.
        const auto *wr=w+uint64_t(row<m ? row : 0)*nb+b;
        const auto *xr=x+uint64_t(token<n ? token : 0)*nb+b;
        int weighted[8]={}, minimum[8]={};
        for(unsigned g=0;g<8;++g) {
            i8 dot{};
            for(unsigned half=0;half<2;++half) {
                i4 av{},bv{};
#pragma unroll
                for(unsigned word=0;word<4;++word) {
                    av[word]=int32_t(weight_word(*wr,g,half*16+word*4));
                    bv[word]=int32_t(activation_word(*xr,g,half*16+word*4));
                }
                // RDNA3 mirrored input lanes and I-major result layout from
                // cuda/mmq/mma.cuh. Both 16-lane halves supply identical rows.
                // clamp=false is exact: a group sums at most32*15*128=61440
                // in magnitude, and even 8*63*61440 fits signed INT32.
                dot=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,av,true,bv,dot,false);
            }
            const int sc=metadata(*wr,g,false), mn=metadata(*wr,g,true);
            const int bs=int(xr->bsums[2*g])+int(xr->bsums[2*g+1]);
#pragma unroll
            for(unsigned j=0;j<8;++j) {
                weighted[j]+=sc*dot[j];
                minimum[j]+=mn*__shfl_sync(UINT64_MAX,bs,result_col(lane,j),32);
            }
        }
#pragma unroll
        for(unsigned j=0;j<8;++j)
            acc[j]=finish_block(acc[j],__shfl_sync(UINT64_MAX,xr->d,result_col(lane,j),32),*wr,weighted[j],minimum[j]);
    }
    __shared__ float partial[8*256];
#pragma unroll
    for(unsigned j=0;j<8;++j)
        partial[wave*256+result_col(lane,j)*16+result_row(lane)]=acc[j];
    __syncthreads();
    const unsigned local=threadIdx.x, r=blockIdx.x*16+local%16, t=blockIdx.y*16+local/16;
    if(r<m && t<n) {
        float p[8];
#pragma unroll
        for(unsigned j=0;j<8;++j) p[j]=partial[j*256+local];
        out[uint64_t(t)*m+r]=reduce8(p);
    }
#else
    (void)out;(void)w;(void)x;(void)m;(void)n;(void)nb;
#endif
}
// Numerical baseline uses the actual production block helper and K partition.
// This deliberately is not a performance surrogate for the full TILE8 kernel.
__global__ static void baseline(float *out,const cuda_block_q4_K *w,
        const cuda_block_q8_K *x,unsigned m,unsigned n,unsigned nb) {
    unsigned row=blockIdx.x*32+threadIdx.x/8,lane=threadIdx.x&7u,t=blockIdx.y;
    if(row>=m || t>=n) return;
    float acc[8]={};
    for(unsigned b=lane;b<nb;b+=8) {
        const auto *y=x+uint64_t(t)*nb+b;
        rocm_dot_q4_K_q8_K_block8_reuse_weights(w+uint64_t(row)*nb+b,y,y,y,y,y,y,y,y,1,acc);
    }
    float v=acc[0];
    const uint64_t mask=uint64_t(255u)<<((threadIdx.x&31u)&~7u);
    for(unsigned offset=4;offset;offset>>=1) v+=__shfl_down_sync(mask,v,offset,8);
    if(!lane) out[uint64_t(t)*m+row]=v;
}
static void hip_check(hipError_t e) { if(e!=hipSuccess) { std::fprintf(stderr,"HIP: %s\n",hipGetErrorString(e));std::exit(1); } }
#endif

static void check(bool ok,const char *why) { if(!ok) { std::fprintf(stderr,"FAIL %s\n",why);std::exit(1); } }
static uint16_t hb(float x) { _Float16 h=_Float16(x);uint16_t u;std::memcpy(&u,&h,2);return u; }
static constexpr size_t guard=32;
static constexpr float poison=-123456.75f;
static void run_case(unsigned m,unsigned n,unsigned nb,std::mt19937& rng,bool rich=false) {
    std::vector<cuda_block_q4_K> w(size_t(m)*nb);
    std::vector<cuda_block_q8_K> x(size_t(n)*nb);
    for(auto& b:w) {
        b.d=hb(float(int(rng()%65)-32)/128);b.dmin=hb(float(rng()%33)/128);
        if(rich) {
            b.d=uint16_t(((1u+rng()%29u)<<10u)|(rng()&1023u)|((rng()&1u)<<15u));
            b.dmin=uint16_t(((1u+rng()%29u)<<10u)|(rng()&1023u));
        }
        for(auto& s:b.scales)s=uint8_t(rng());for(auto& q:b.qs)q=uint8_t(rng());
    }
    for(auto& b:x) {
        b.d=float(int(rng()%65)-32)/256;
        if(rich) {
            const uint32_t bits=((96u+rng()%48u)<<23u)|(rng()&0x7fffffu)|((rng()&1u)<<31u);
            std::memcpy(&b.d,&bits,4);
        }
        for(auto& q:b.qs)q=int8_t(rng());
        for(unsigned g=0;g<16;++g) { int sum=0;for(unsigned k=0;k<16;++k)sum+=b.qs[g*16+k];b.bsums[g]=int16_t(sum); }
    }
    std::vector<float> reference(guard+size_t(m)*n+guard,poison), actual=reference;
    const uint32_t unwritten=0x7fc5a11au;
    for(size_t i=guard;i<guard+size_t(m)*n;++i)std::memcpy(&actual[i],&unwritten,4);
#ifndef TEST_NATIVE
    // Matrix model uses exactly the native fragment ownership. Scalar oracle
    // calls production code with eight independent K-block accumulators.
    for(unsigned t0=0;t0<n;t0+=16)for(unsigned r0=0;r0<m;r0+=16)
    for(unsigned lane=0;lane<32;++lane)for(unsigned j=0;j<8;++j) {
        const unsigned r=r0+result_row(lane),t=t0+result_col(lane,j);
        if(r>=m||t>=n)continue;
        float parts[8]={}, ref_parts[8]={};
        for(unsigned split=0;split<8;++split)for(unsigned b=split;b<nb;b+=8) {
            const auto& wb=w[size_t(r)*nb+b];const auto& xb=x[size_t(t)*nb+b];
            int weighted=0,mins=0;
            for(unsigned g=0;g<8;++g) {
                int dot=0;
                for(unsigned h=0;h<2;++h)for(unsigned word=0;word<4;++word)
                    dot=__dp4a(int32_t(weight_word(wb,g,h*16+word*4)),
                               int32_t(activation_word(xb,g,h*16+word*4)),dot);
                weighted+=int(metadata(wb,g,false))*dot;
                mins+=int(metadata(wb,g,true))*(int(xb.bsums[g*2])+int(xb.bsums[g*2+1]));
            }
            parts[split]=finish_block(parts[split],xb.d,wb,weighted,mins);
            float a[8]={ref_parts[split]};
            rocm_dot_q4_K_q8_K_block8_reuse_weights(&wb,&xb,&xb,&xb,&xb,&xb,&xb,&xb,&xb,1,a);
            ref_parts[split]=a[0];
        }
        reference[guard+size_t(t)*m+r]=reduce8(ref_parts);
        actual[guard+size_t(t)*m+r]=reduce8(parts);
    }
#else
    cuda_block_q4_K *dw=nullptr;cuda_block_q8_K *dx=nullptr;float *da=nullptr,*db=nullptr;
    hip_check(hipMalloc(&dw,w.size()*sizeof(w[0])));hip_check(hipMalloc(&dx,x.size()*sizeof(x[0])));
    hip_check(hipMalloc(&da,actual.size()*4));hip_check(hipMalloc(&db,reference.size()*4));
    hip_check(hipMemcpy(dw,w.data(),w.size()*sizeof(w[0]),hipMemcpyHostToDevice));
    hip_check(hipMemcpy(dx,x.data(),x.size()*sizeof(x[0]),hipMemcpyHostToDevice));
    hip_check(hipMemcpy(da,actual.data(),actual.size()*4,hipMemcpyHostToDevice));
    hip_check(hipMemcpy(db,reference.data(),reference.size()*4,hipMemcpyHostToDevice));
    baseline<<<dim3((m+31)/32,n),256>>>(db+guard,dw,dx,m,n,nb);
    hip_check(hipGetLastError());
    candidate<<<dim3((m+15)/16,(n+15)/16),256>>>(da+guard,dw,dx,m,n,nb);
    hip_check(hipGetLastError());hip_check(hipDeviceSynchronize());
    hip_check(hipMemcpy(actual.data(),da,actual.size()*4,hipMemcpyDeviceToHost));
    hip_check(hipMemcpy(reference.data(),db,reference.size()*4,hipMemcpyDeviceToHost));
    std::vector<cuda_block_q4_K> w_after(w.size());std::vector<cuda_block_q8_K> x_after(x.size());
    hip_check(hipMemcpy(w_after.data(),dw,w.size()*sizeof(w[0]),hipMemcpyDeviceToHost));
    hip_check(hipMemcpy(x_after.data(),dx,x.size()*sizeof(x[0]),hipMemcpyDeviceToHost));
    check(!std::memcmp(w.data(),w_after.data(),w.size()*sizeof(w[0])) &&
          !std::memcmp(x.data(),x_after.data(),x.size()*sizeof(x[0])),"immutable inputs");
    hip_check(hipFree(dw));hip_check(hipFree(dx));hip_check(hipFree(da));hip_check(hipFree(db));
#endif
    for(size_t i=0;i<actual.size();++i) {
        uint32_t bits;std::memcpy(&bits,&actual[i],4);
        check((bits&0x7f800000u)!=0x7f800000u,"finite and written output (integer bits)");
        if(std::memcmp(&actual[i],&reference[i],4)) {
            std::fprintf(stderr,"m=%u n=%u K=%u i=%zu actual=%a expected=%a\n",m,n,nb*256,i,actual[i],reference[i]);
            check(false,"bitwise production-dot parity");
        }
        if(i<guard||i>=guard+size_t(m)*n)check(actual[i]==poison,"output guards");
    }
    std::printf("PASS m=%u n=%u K=%u full_mantissa=%u\n",m,n,nb*256,unsigned(rich));
}
int main() {
#ifdef TEST_NATIVE
    int count=0;hip_check(hipGetDeviceCount(&count));if(!count)return 77;
    hipDeviceProp_t p{};hip_check(hipGetDeviceProperties(&p,0));
    if(std::strncmp(p.gcnArchName,"gfx1151",7)||p.warpSize!=32) { std::fprintf(stderr,"requires gfx1151 wave32\n");return 77; }
    hipFuncAttributes a{};hip_check(hipFuncGetAttributes(&a,reinterpret_cast<const void*>(candidate)));
    std::printf("device=%s arch=%s candidate_regs=%d static_lds=%zu local=%zu\n",p.name,p.gcnArchName,a.numRegs,a.sharedSizeBytes,a.localSizeBytes);
#else
    unsigned seen[256]={};
    for(unsigned lane=0;lane<32;++lane)for(unsigned j=0;j<8;++j)++seen[result_col(lane,j)*16+result_row(lane)];
    for(unsigned v:seen)check(v==1,"RDNA3 result lane map bijection");
    std::puts("HOST MODEL ONLY: no WMMA instruction, no GPU race or speed validation");
#endif
    std::mt19937 rng(0x128256u);
    for(unsigned nb:{1u,2u,8u,32u})run_case(16,16,nb,rng);
    run_case(17,19,32,rng);run_case(15,128,32,rng);run_case(1,1,1,rng);
    run_case(16,16,32,rng,true);run_case(17,19,33,rng,true);
    return 0;
}
