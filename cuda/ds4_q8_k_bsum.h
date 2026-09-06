// SPDX-License-Identifier: MIT
// Include after the CUDA/HIP device shuffle intrinsics. ROCm supplies MASK_T;
// CUDA uses its native 32-bit participant mask.
#ifndef DS4_Q8_K_BSUM_H
#define DS4_Q8_K_BSUM_H

__device__ __forceinline__ static int32_t ds4_q8_K_bsum16(
        int32_t q, uint32_t tid, uint32_t wave_size) {
    // Each 16-lane subgroup owns one contiguous Q8_K bsum. Cast before
    // shifting so subgroups in the upper half of wave64 name their own lanes.
    const uint32_t wave_lane = tid & (wave_size - 1u);
#if defined(MASK_T)
    const MASK_T mask = (MASK_T)0xffffu << (wave_lane & ~15u);
#else
    const uint32_t mask = 0xffffu << (wave_lane & ~15u);
#endif
    #pragma unroll
    for (uint32_t offset = 8u; offset != 0u; offset >>= 1u)
        q += __shfl_down_sync(mask, q, offset, 16);
    return q;
}

#endif
