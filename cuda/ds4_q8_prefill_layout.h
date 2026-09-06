// SPDX-License-Identifier: MIT
// Host-testable admission and addressing for the CUDA prefill Q8_0 producer.
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define DS4_Q8_PREFILL_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q8_PREFILL_INLINE inline
#endif

namespace ds4_cuda_q8_prefill {
enum { lanes = 32, warps = 8, threads = lanes * warps };

// logical_tokens must not be the packed token*group row count. Exact-decode
// entry points deliberately do not call this selector, even for many rows.
inline bool select(uint64_t logical_tokens, uint64_t in_dim,
                   bool single_gb10, bool quality, bool disable_warp,
                   bool disable_prefill) {
    return single_gb10 && !quality && !disable_warp && !disable_prefill &&
           logical_tokens >= 256u && logical_tokens <= 8192u &&
           in_dim >= 256u && in_dim <= 16384u;
}

DS4_Q8_PREFILL_INLINE uint64_t tiles(uint64_t blocks) {
    return blocks / warps + (blocks % warps != 0u);
}

DS4_Q8_PREFILL_INLINE uint64_t quant_block(uint64_t tile, uint32_t tid) {
    return tile * warps + tid / lanes;
}
} // namespace ds4_cuda_q8_prefill
#undef DS4_Q8_PREFILL_INLINE
