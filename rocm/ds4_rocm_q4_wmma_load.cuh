// SPDX-License-Identifier: MIT
// Addressing/admission shared by the K64 float4 loader and native host tests.
#ifndef DS4_ROCM_Q4_WMMA_LOAD_CUH
#define DS4_ROCM_Q4_WMMA_LOAD_CUH
#include <stdint.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_Q4_LOAD_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_LOAD_INLINE inline
#endif

namespace ds4_rocm_q4_wmma_load {
enum { tokens = 64u, columns = 64u, pitch = 80u };

DS4_Q4_LOAD_INLINE bool aligned4(
        const void *x, uint64_t token_stride, uint64_t group_stride) {
    return x && ((uintptr_t)x & 15u) == 0u &&
           (token_stride & 3u) == 0u && (group_stride & 3u) == 0u;
}

// Called only after the existing WMMA architecture/precision/storage selector.
DS4_Q4_LOAD_INLINE bool select(
        bool disabled, bool k64, bool k128, uint32_t row_tile,
        const void *x, uint64_t token_stride, uint64_t group_stride) {
    return !disabled && k64 && !k128 &&
           (row_tile == 128u || row_tile == 256u) &&
           aligned4(x, token_stride, group_stride);
}

DS4_Q4_LOAD_INLINE uint32_t token(uint32_t j) { return j >> 6u; }
DS4_Q4_LOAD_INLINE uint32_t column(uint32_t j) { return j & 63u; }
DS4_Q4_LOAD_INLINE uint32_t destination(uint32_t j) {
    return token(j) * pitch + column(j);
}
DS4_Q4_LOAD_INLINE uint64_t source(
        uint32_t j, uint32_t tok0, uint32_t group, uint32_t k0,
        uint64_t token_stride, uint64_t group_stride) {
    return ((uint64_t)tok0 + token(j)) * token_stride +
           (uint64_t)group * group_stride + k0 + column(j);
}

// The final WMMA stage only reads LDS. Subsequent register/output operations
// cannot overwrite it, so only stages followed by another producer need a
// reuse barrier. The separate producer-to-consumer barrier is unconditional.
DS4_Q4_LOAD_INLINE bool needs_reuse_barrier(
        uint32_t block, uint32_t blocks, uint32_t stage, uint32_t stages) {
    return block + 1u < blocks || stage + 1u < stages;
}
} // namespace ds4_rocm_q4_wmma_load
#undef DS4_Q4_LOAD_INLINE
#endif
