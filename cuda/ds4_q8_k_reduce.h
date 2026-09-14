// SPDX-License-Identifier: MIT
// Exact signed-maximum tree for a 256-value Q8_K activation block.
#ifndef DS4_Q8_K_REDUCE_H
#define DS4_Q8_K_REDUCE_H

#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_Q8_MAX_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q8_MAX_INLINE inline
#endif

typedef struct {
    float magnitude;
    uint32_t index;
} ds4_q8_K_maximum;

DS4_Q8_MAX_INLINE static ds4_q8_K_maximum ds4_q8_K_max_choose(
        ds4_q8_K_maximum left, ds4_q8_K_maximum right) {
    return right.magnitude > left.magnitude ? right : left;
}

// Preserve the original stride-128/64/32 operands. Equal magnitudes keep
// the left subtree, whose winner need not have the lowest source index.
DS4_Q8_MAX_INLINE static ds4_q8_K_maximum ds4_q8_K_max_fold(
        const float *magnitudes, uint32_t lane) {
    ds4_q8_K_maximum leaves[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) {
        const uint32_t index = lane + 32u * i;
        leaves[i].magnitude = magnitudes[index];
        leaves[i].index = index;
    }
    const ds4_q8_K_maximum a = ds4_q8_K_max_choose(leaves[0], leaves[4]);
    const ds4_q8_K_maximum b = ds4_q8_K_max_choose(leaves[2], leaves[6]);
    const ds4_q8_K_maximum c = ds4_q8_K_max_choose(leaves[1], leaves[5]);
    const ds4_q8_K_maximum d = ds4_q8_K_max_choose(leaves[3], leaves[7]);
    return ds4_q8_K_max_choose(ds4_q8_K_max_choose(a, b),
                               ds4_q8_K_max_choose(c, d));
}

#if defined(__CUDACC__) || defined(__HIPCC__)
// Only physical lanes 0..31 of the first wave call this helper, including
// on wave64. All 256 shared leaves must have been published by a CTA fence.
__device__ __forceinline__ static ds4_q8_K_maximum ds4_q8_K_max_first_warp(
        const float *magnitudes, uint32_t lane) {
    ds4_q8_K_maximum best = ds4_q8_K_max_fold(magnitudes, lane);
#if defined(MASK_T)
    const MASK_T mask = (MASK_T)0xffffffffu;
#else
    const uint32_t mask = 0xffffffffu;
#endif
#pragma unroll
    for (uint32_t stride = 16u; stride != 0u; stride >>= 1u) {
        ds4_q8_K_maximum other;
        other.magnitude = __shfl_down_sync(mask, best.magnitude, stride, 32);
        other.index = __shfl_down_sync(mask, best.index, stride, 32);
        if (lane < stride) best = ds4_q8_K_max_choose(best, other);
    }
    return best;
}
#endif

#undef DS4_Q8_MAX_INLINE
#endif
