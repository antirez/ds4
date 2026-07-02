// DS4 ROCm direct MFMA wrappers.

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

#include <stdint.h>

typedef _Float16 __attribute__((ext_vector_type(4))) ds4_rocm_f16x4_t;
typedef _Float16 __attribute__((ext_vector_type(8))) ds4_rocm_f16x8_t;
typedef float    __attribute__((ext_vector_type(4))) ds4_rocm_f32x4_t;

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define DS4_ROCM_ARCH_CDNA3 1
#else
#define DS4_ROCM_ARCH_CDNA3 0
#endif

#if defined(__gfx950__)
#define DS4_ROCM_ARCH_CDNA4 1
#else
#define DS4_ROCM_ARCH_CDNA4 0
#endif

#if DS4_ROCM_ARCH_CDNA4
#define DS4_ROCM_MFMA_F16_K 32u
#define DS4_ROCM_MFMA_F16_K_PER_LANE 8u
typedef ds4_rocm_f16x8_t ds4_rocm_mfma_f16_frag_t;
#elif DS4_ROCM_ARCH_CDNA3
#define DS4_ROCM_MFMA_F16_K 16u
#define DS4_ROCM_MFMA_F16_K_PER_LANE 4u
typedef ds4_rocm_f16x4_t ds4_rocm_mfma_f16_frag_t;
#else
#define DS4_ROCM_MFMA_F16_K 16u
#define DS4_ROCM_MFMA_F16_K_PER_LANE 4u
typedef ds4_rocm_f16x4_t ds4_rocm_mfma_f16_frag_t;
#endif

__device__ __forceinline__ static ds4_rocm_f32x4_t ds4_rocm_f32x4_zero(void) {
    ds4_rocm_f32x4_t v;
#pragma unroll
    for (uint32_t i = 0; i < 4u; i++) v[i] = 0.0f;
    return v;
}

__device__ __forceinline__ static ds4_rocm_f32x4_t ds4_rocm_mfma_f16_16x16(
        ds4_rocm_mfma_f16_frag_t a,
        ds4_rocm_mfma_f16_frag_t b,
        ds4_rocm_f32x4_t c) {
#if DS4_ROCM_ARCH_CDNA4
    return __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, 0, 0);
#elif DS4_ROCM_ARCH_CDNA3
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
#else
    (void)a;
    (void)b;
    return c;
#endif
}

#endif
