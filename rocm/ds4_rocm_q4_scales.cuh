// SPDX-License-Identifier: MIT
// Exact paired Q4_K scale/minimum decoding for ROCm prefill.
#ifndef DS4_ROCM_Q4_SCALES_CUH
#define DS4_ROCM_Q4_SCALES_CUH

#include <stdint.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_Q4_SCALES_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_SCALES_INLINE inline
#endif

namespace ds4_rocm_q4_scales {
struct pair {
    uint16_t scales;
    uint16_t minima;
};

DS4_Q4_SCALES_INLINE uint16_t load_word(const uint8_t *src) {
    // Q4_K stores bytes, not uint16_t objects. A fixed-size memcpy avoids
    // aliasing/alignment assumptions and also accepts unaligned host views.
    uint16_t word;
    __builtin_memcpy(&word, src, sizeof(word));
    return word;
}

// qpair is 0..3. The low/high byte belongs to qgroup 2*qpair / 2*qpair+1.
// AMD GPUs and the GGUF block layout are little endian. Decode both groups
// together so their metadata loads and bit operations are shared. The two
// packed results live only across the corresponding low/high payload use.
DS4_Q4_SCALES_INLINE pair load_pair(const uint8_t *src, uint32_t qpair) {
    if (qpair < 2u) {
        return {static_cast<uint16_t>(load_word(src + qpair * 2u) & 0x3f3fu),
                static_cast<uint16_t>(load_word(src + 4u + qpair * 2u) & 0x3f3fu)};
    }
    const uint16_t extra = load_word(src + 4u + qpair * 2u);
    const uint16_t scale_high = load_word(src + (qpair - 2u) * 2u);
    const uint16_t min_high = load_word(src + qpair * 2u);
    return {
        static_cast<uint16_t>((extra & 0x0f0fu) | ((scale_high >> 2u) & 0x3030u)),
        static_cast<uint16_t>(((extra >> 4u) & 0x0f0fu) | ((min_high >> 2u) & 0x3030u))};
}
} // namespace ds4_rocm_q4_scales

#undef DS4_Q4_SCALES_INLINE
#endif
