// SPDX-License-Identifier: MIT
// Exact Q8_K staging used only by the Q4 tiled-prefill kernels.
#ifndef DS4_ROCM_Q4_LDS_CUH
#define DS4_ROCM_Q4_LDS_CUH

#include <stdint.h>
#include <stddef.h>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_Q4_LDS_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_LDS_INLINE inline
#endif

namespace ds4_rocm_q4_lds {
enum { block_words = 73u }; // sizeof(Q8_K) / sizeof(uint32_t)

// LDS-only layout. Keep the external Q8_K scratch packed; align the hot qs
// payload of every staged block so the dot can read four words at a time.
// Every Q4 group consumes two adjacent Q8 sums. Add them once during staging,
// then share the result across all 32 output rows. Keeping 32-bit group sums
// retains the 76-word pitch: a 72-word pitch would make eight K-block lanes
// collide on four LDS banks instead of eight for scalar metadata reads.
struct alignas(16) aligned_q8_K {
    float d;
    uint32_t padding[3];
    int8_t qs[256];
    int32_t group_sums[8];
};
static_assert(sizeof(aligned_q8_K) == 304u &&
              offsetof(aligned_q8_K, qs) == 16u &&
              offsetof(aligned_q8_K, group_sums) == 272u,
              "Q4 aligned LDS block layout");

DS4_Q4_LDS_INLINE int32_t pair_sum(uint32_t packed) {
    int16_t pair[2];
    __builtin_memcpy(pair, &packed, sizeof(packed));
    return (int32_t)pair[0] + (int32_t)pair[1];
}

DS4_Q4_LDS_INLINE bool aligned_scope(
        uint32_t blocks, uint32_t out_dim, uint32_t n_tok, uint32_t groups,
        bool ssd, bool quality, bool gfx1151, bool disabled) {
    return !disabled && !ssd && !quality && gfx1151 && groups == 1u &&
           blocks == 32u && out_dim == 4096u &&
           n_tok >= 256u && n_tok <= 4096u;
}

template<uint32_t BLOCKS>
DS4_Q4_LDS_INLINE void copy_thread_aligned(
        uint32_t *dst, const uint32_t *src, uint32_t tid, uint32_t threads,
        uint32_t nt, uint32_t nb, uint64_t src_stride_words) {
    static_assert(BLOCKS == 4u || BLOCKS == 8u, "Q4 LDS tile must be TILE4/8");
    // One wave per packed block gives contiguous reads without division by
    // 73 for each word. Threads is a positive multiple of 32. No thread reads
    // inactive K/token slots; the three padding words are never consumed.
    const uint32_t lane = tid & 31u;
    for (uint32_t slot = tid >> 5u; slot < nt * BLOCKS; slot += threads >> 5u) {
        const uint32_t p = slot / BLOCKS;
        const uint32_t b = slot % BLOCKS;
        if (b < nb) {
            const uint32_t *block = src + (uint64_t)p * src_stride_words + b * block_words;
            for (uint32_t w = lane; w < block_words; w += 32u) {
                // Copy object representations: the words contain float,
                // byte and short fields, not an array of uint32_t objects.
                uint32_t value;
                __builtin_memcpy(&value, block + w, sizeof(value));
                if (w >= 65u) {
                    const int32_t sum = pair_sum(value);
                    __builtin_memcpy(&value, &sum, sizeof(value));
                }
                __builtin_memcpy(dst + slot * 76u + (w == 0u ? 0u : w + 3u),
                                 &value, sizeof(value));
            }
        }
    }
}

// src already points at this tile's first token/group/K block. A token may
// have other groups or K blocks between the ranges being staged. Destination
// rows always keep their full BLOCKS pitch, including an uninitialized K tail.
template<uint32_t BLOCKS>
DS4_Q4_LDS_INLINE void copy_thread(
        uint32_t *dst, const uint32_t *src, uint32_t tid, uint32_t threads,
        uint32_t nt, uint32_t nb, uint64_t src_stride_words) {
    const uint32_t pitch = BLOCKS * block_words;
    const uint32_t words = nt * pitch;
    if (nb == BLOCKS && src_stride_words == pitch) {
        // In particular, K1024 TILE4 is one contiguous copy. No per-word
        // block division, remainder, token index or 64-bit row multiply.
        for (uint32_t i = tid; i < words; i += threads) dst[i] = src[i];
    } else {
        for (uint32_t i = tid; i < words; i += threads) {
            const uint32_t p = i / pitch;
            const uint32_t word = i - p * pitch;
            if (word < nb * block_words)
                dst[i] = src[(uint64_t)p * src_stride_words + word];
        }
    }
}

// Four-word copies are valid only when every token row has the same 16-byte
// alignment. Q8_K blocks themselves are only four-byte aligned (292 bytes).
DS4_Q4_LDS_INLINE bool vector_copy_aligned(
        const uint32_t *dst, const uint32_t *src, uint64_t src_stride_words) {
    return (((uintptr_t)dst | (uintptr_t)src) & 15u) == 0u &&
           (src_stride_words & 3u) == 0u;
}

DS4_Q4_LDS_INLINE void copy_four(uint32_t *dst, const uint32_t *src) {
    // Fixed-size memcpy preserves the packed block's aliasing semantics.
    // Alignment is established by the caller, not assumed for arbitrary Q8_K.
    __builtin_memcpy(__builtin_assume_aligned(dst, 16),
                     __builtin_assume_aligned(src, 16), 16);
}

template<uint32_t BLOCKS>
DS4_Q4_LDS_INLINE void copy_thread_vector(
        uint32_t *dst, const uint32_t *src, uint32_t tid, uint32_t threads,
        uint32_t nt, uint32_t nb, uint64_t src_stride_words) {
    static_assert(BLOCKS == 4u || BLOCKS == 8u, "Q4 LDS tile must be TILE4/8");
    if (!vector_copy_aligned(dst, src, src_stride_words)) {
        copy_thread<BLOCKS>(dst, src, tid, threads, nt, nb, src_stride_words);
        return;
    }
    constexpr uint32_t pitch = BLOCKS * block_words;
    constexpr uint32_t pitch4 = pitch / 4u;
    const uint32_t vectors = nt * pitch4;
    if (nb == BLOCKS && src_stride_words == pitch) {
        for (uint32_t i = tid; i < vectors; i += threads)
            copy_four(dst + 4u * i, src + 4u * i);
    } else {
        const uint32_t valid_words = nb * block_words;
        for (uint32_t i = tid; i < vectors; i += threads) {
            const uint32_t p = i / pitch4;
            const uint32_t word = (i - p * pitch4) * 4u;
            if (word + 4u <= valid_words) {
                copy_four(dst + 4u * i,
                          src + (uint64_t)p * src_stride_words + word);
            } else {
                // Never read beyond the last valid Q8_K block or initialize
                // padding which the existing scalar schedule leaves alone.
                for (uint32_t j = 0u; j < 4u && word + j < valid_words; ++j)
                    dst[4u * i + j] = src[(uint64_t)p * src_stride_words + word + j];
            }
        }
    }
}

template<uint32_t BLOCKS, bool VECTOR>
DS4_Q4_LDS_INLINE void copy_thread_selected(
        uint32_t *dst, const uint32_t *src, uint32_t tid, uint32_t threads,
        uint32_t nt, uint32_t nb, uint64_t src_stride_words) {
    if constexpr (VECTOR)
        copy_thread_vector<BLOCKS>(dst, src, tid, threads, nt, nb, src_stride_words);
    else
        copy_thread<BLOCKS>(dst, src, tid, threads, nt, nb, src_stride_words);
}

// All workgroup threads use the same K loop. The final tile is read only:
// register reduction/output stores cannot overwrite LDS, so no reuse fence
// is necessary after it. The producer-to-consumer barrier remains mandatory.
DS4_Q4_LDS_INLINE bool needs_reuse_barrier(uint32_t b0, uint32_t blocks) {
    return blocks - b0 > 8u; // caller guarantees b0 < blocks
}
} // namespace ds4_rocm_q4_lds

#undef DS4_Q4_LDS_INLINE
#endif
