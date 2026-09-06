// SPDX-License-Identifier: MIT
// Compile the actual production dot with host FP16/dp4a shims. This tests
// staging and integer/FP32 operand order, not GPU instructions or scheduling.
#include "rocm/ds4_rocm_q4_lds.cuh"
#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define __device__
#define __forceinline__ inline
struct alignas(16) int4 { int32_t x,y,z,w; };
struct cuda_block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
struct alignas(4) cuda_block_q4_K { uint16_t d,dmin; uint8_t scales[12],qs[128]; };
enum { ROCM_Q4_PREFILL_TOKEN_TILE=8 };
static float dev_f16_to_f32(uint16_t bits) { _Float16 f; std::memcpy(&f,&bits,2); return (float)f; }
static int32_t __dp4a(int32_t a,int32_t b,int32_t acc) {
  for (uint32_t i=0;i<4;i++) acc += (int32_t)(int8_t)((uint32_t)a>>(8*i))*(int32_t)(int8_t)((uint32_t)b>>(8*i));
  return acc;
}
static uint32_t rng=0x486fabd9u;
static uint32_t next() { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
static void halfbits(uint16_t &u) { const _Float16 f=(_Float16)((int32_t)(next()%2049)-1024)/(_Float16)32;std::memcpy(&u,&f,2); }
__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

#include "rocm/ds4_rocm_q4_dot.cuh"

// Independently enumerate Q4 logical groups, nibble indices and Q8 elements.
// Integer products/sums fit int32 even for the deliberately malformed int16
// sum extrema below: 8 * 63 * 65536 < INT32_MAX. Preserve the reference FP32
// block order; an F16 GEMM approximation would not be an exact oracle.
static float scalar_block(const cuda_block_q4_K &w,
                          const cuda_block_q8_K &x, float acc) {
  int32_t dot=0, mins=0;
  for (unsigned g=0;g<8;g++) {
    const unsigned sc=g<4 ? w.scales[g]&63u :
      (w.scales[g+4]&15u)|((w.scales[g-4]>>6u)<<4u);
    const unsigned mn=g<4 ? w.scales[g+4]&63u :
      (w.scales[g+4]>>4u)|((w.scales[g]>>6u)<<4u);
    int32_t group=0;
    for(unsigned k=0;k<32;k++) {
      const unsigned q=(w.qs[(g/2)*32+k]>>((g%2)*4))&15u;
      group+=(int32_t)q*(int32_t)x.qs[g*32+k];
    }
    dot+=(int32_t)sc*group;
    mins+=(int32_t)mn*((int32_t)x.bsums[g*2]+(int32_t)x.bsums[g*2+1]);
  }
  const float xd=dev_f16_to_f32(w.d), xmin=dev_f16_to_f32(w.dmin), yd=x.d;
  acc += yd*xd*(float)dot - yd*xmin*(float)mins;
  return acc;
}

int main() {
  unsigned cases=0;
  for (uint32_t trial=0;trial<1600;trial++) {
    cuda_block_q4_K w[8];
    cuda_block_q8_K packed[8][8];
    alignas(16) ds4_rocm_q4_lds::aligned_q8_K aligned[8][8];
    for (auto &weight:w) {
      halfbits(weight.d);halfbits(weight.dmin);
      if (trial%17==0) weight.d=0x7bffu; // maximum finite half
      if (trial%13==0) weight.dmin=0x0001u; // half subnormal
      for (auto &s:weight.scales) s=(uint8_t)next();
      for (auto &q:weight.qs) q=(uint8_t)next();
    }
    for (auto &token:packed) for (auto &block:token) {
      block.d=(float)((int32_t)(next()%4097)-2048)/512.0f;
      for (auto &q:block.qs) q=trial%19==0?-128:trial%23==0?127:(int8_t)next();
      for (uint32_t s=0;s<16;s++) {
        int32_t sum=0; for(uint32_t i=0;i<16;i++) sum+=block.qs[s*16+i];
        block.bsums[s]=trial%29==0 ? INT16_MIN : trial%31==0 ? INT16_MAX : (int16_t)sum;
      }
    }
    for (uint32_t n=1;n<=8;n++) {
      std::memset(aligned,0xab,sizeof(aligned));
      for(uint32_t tid=0;tid<256;tid++) ds4_rocm_q4_lds::copy_thread_aligned<8>(
        reinterpret_cast<uint32_t*>(aligned), reinterpret_cast<const uint32_t*>(packed),tid,256,n,8,8*73);
      float a[8],b[8],ref[8];
      for(uint32_t p=0;p<8;p++) a[p]=b[p]=ref[p]=(float)((int32_t)(next()%4097)-2048)/256.0f;
      for (uint32_t lane=0;lane<8;lane++) {
        rocm_dot_q4_K_q8_K_block8_reuse_weights(w+lane,packed[0]+lane,packed[1]+lane,packed[2]+lane,packed[3]+lane,packed[4]+lane,packed[5]+lane,packed[6]+lane,packed[7]+lane,n,a);
        rocm_dot_q4_K_q8_K_block8_reuse_weights(w+lane,aligned[0]+lane,aligned[1]+lane,aligned[2]+lane,aligned[3]+lane,aligned[4]+lane,aligned[5]+lane,aligned[6]+lane,aligned[7]+lane,n,b);
        for(uint32_t p=0;p<n;p++) ref[p]=scalar_block(w[lane],packed[p][lane],ref[p]);
        if(std::memcmp(a,b,sizeof(a)) || std::memcmp(a,ref,sizeof(a))) {
          std::fprintf(stderr,"FAIL trial=%u n=%u lane=%u\n",trial,n,lane);return 1;
        }
        cases++;
      }
    }
  }
  std::printf("PASS production dot packed/aligned: %u cases (native shim only)\n",cases);
}
