// SPDX-License-Identifier: MIT
// Shared addressing and reduction tree; usable by the native CPU oracle.
#ifndef DS4_ROCM_Q4_QB_EPILOGUE_LAYOUT_CUH
#define DS4_ROCM_Q4_QB_EPILOGUE_LAYOUT_CUH
#include <stdint.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_QB_EP_INLINE __host__ __device__ __forceinline__
#else
#define DS4_QB_EP_INLINE inline
#endif

namespace ds4_rocm_q4_qb_epilogue {
enum { lanes = 32, values = 16, head_dim = 512, rotation = 64,
       heads = 64, heads_per_block = 8, threads = 256 };

DS4_QB_EP_INLINE bool shape(uint32_t tokens, uint32_t nh,
                            uint32_t dim, uint32_t rot) {
    return tokens >= 256u && tokens <= 4096u && nh == heads &&
           dim == head_dim && rot == rotation;
}

DS4_QB_EP_INLINE bool select(uint32_t tokens, uint32_t nh,
                            uint32_t dim, uint32_t rot, bool gfx1151_wave32,
                            bool quality, bool ssd, bool disabled) {
    return shape(tokens, nh, dim, rot) && gfx1151_wave32 &&
           !quality && !ssd && !disabled;
}

DS4_QB_EP_INLINE uint32_t column(uint32_t lane, uint32_t slot) {
    return lane + lanes * slot;
}

// The legacy 256-thread kernel first forms x[i]^2 + x[i+256]^2.
// Its stride-128/64/32 stages combine eight such partials per lane.
// Do NOT replace this with a sequential sum of the sixteen retained values.
DS4_QB_EP_INLINE float fold_columns(const float *v) {
#if defined(__clang__)
    // ROCm builds use -ffast-math. Without this local restriction LLVM may
    // flatten the register tree into an unordered vector reduction. Retain
    // multiply/add contraction within each legacy sum += v*v expression.
#pragma clang fp reassociate(off) contract(on)
#endif
    float partial[8];
#if defined(__HIPCC__) || defined(__CUDACC__)
#pragma unroll
#endif
    for (uint32_t j = 0; j < 8u; ++j) {
        float sum = 0.0f;
        sum += v[j] * v[j];
        sum += v[j + 8u] * v[j + 8u];
        partial[j] = sum;
    }
#if defined(__HIPCC__) || defined(__CUDACC__)
#pragma unroll
#endif
    for (uint32_t stride = 4u; stride; stride >>= 1u) {
#if defined(__HIPCC__) || defined(__CUDACC__)
#pragma unroll
#endif
        for (uint32_t j = 0; j < stride; ++j)
            partial[j] += partial[j + stride];
    }
    return partial[0];
}
} // namespace ds4_rocm_q4_qb_epilogue
#undef DS4_QB_EP_INLINE
#endif
