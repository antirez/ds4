// SPDX-License-Identifier: MIT
// Shared CUDA/HIP Q4_K transient-prefill addressing; also used by CPU tests.
#pragma once
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_Q4_DQ_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_DQ_INLINE inline
#endif

namespace ds4_q4_dequant {
enum { block_values = 256, block_bytes = 144, values_per_thread = 16,
       chunks_per_block = 16, threads = 256 };

// Call only from transient weight expansion, never persistent-cache prewarm.
// Vector loads/stores require aligned, disjoint buffers. Existing callers
// validate buffer lengths and ownership before reaching this selection.
inline bool select(uint64_t k, uint64_t m, uint64_t tokens,
                   uintptr_t src, uintptr_t dst, bool target_device,
                   bool quality, bool ssd, bool disabled) {
    if (k != 1024u || m != 32768u || tokens < 256u || tokens > 8192u ||
        !target_device || quality || ssd || disabled || !src || !dst ||
        ((src | dst) & 15u)) return false;
    const uint64_t source_bytes = (k / block_values) * m * block_bytes;
    const uint64_t output_bytes = k * m * 2u;
    if (source_bytes > UINTPTR_MAX - src || output_bytes > UINTPTR_MAX - dst)
        return false;
    return src <= dst ? dst - src >= source_bytes : src - dst >= output_bytes;
}

DS4_Q4_DQ_INLINE uint64_t block(uint64_t chunk) { return chunk >> 4u; }
DS4_Q4_DQ_INLINE uint32_t within(uint64_t chunk) { return (chunk & 15u) * 16u; }
DS4_Q4_DQ_INLINE uint32_t group(uint64_t chunk) { return (chunk >> 1u) & 7u; }
DS4_Q4_DQ_INLINE uint32_t packed_offset(uint64_t chunk) {
    return ((chunk >> 2u) & 3u) * 32u + (chunk & 1u) * 16u;
}
DS4_Q4_DQ_INLINE void scale_min(uint32_t g, const uint8_t *s,
                                uint8_t *scale, uint8_t *minimum) {
    if (g < 4u) {
        *scale = s[g] & 63u;
        *minimum = s[g + 4u] & 63u;
    } else {
        *scale = (s[g + 4u] & 15u) | ((s[g - 4u] >> 6u) << 4u);
        *minimum = (s[g + 4u] >> 4u) | ((s[g] >> 6u) << 4u);
    }
}
DS4_Q4_DQ_INLINE uint32_t nibble(uint32_t word, uint32_t byte, uint32_t g) {
    return (word >> (8u * byte + ((g & 1u) * 4u))) & 15u;
}
DS4_Q4_DQ_INLINE uint32_t half_pair(uint16_t lo, uint16_t hi) {
    return (uint32_t)lo | ((uint32_t)hi << 16u);
}
} // namespace ds4_q4_dequant
#undef DS4_Q4_DQ_INLINE
