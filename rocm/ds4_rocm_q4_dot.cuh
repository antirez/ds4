// SPDX-License-Identifier: MIT
// Shared production dot body for native GPU kernels and the host arithmetic
// oracle. Include after the Q4/Q8 layouts, half/dp4a helpers and token tile.
#ifndef DS4_ROCM_Q4_DOT_CUH
#define DS4_ROCM_Q4_DOT_CUH

#include "ds4_rocm_q4_lds.cuh"
#include <type_traits>

template<typename Q8Block>
__device__ __forceinline__ static void
rocm_dot_q4_K_q8_K_block8_reuse_weights(
        const cuda_block_q4_K *x,
        const Q8Block *y0,
        const Q8Block *y1,
        const Q8Block *y2,
        const Q8Block *y3,
        const Q8Block *y4,
        const Q8Block *y5,
        const Q8Block *y6,
        const Q8Block *y7,
        uint32_t n,
        float acc[ROCM_Q4_PREFILL_TOKEN_TILE]) {
    const Q8Block *ys[ROCM_Q4_PREFILL_TOKEN_TILE] = {
        y0, y1, y2, y3, y4, y5, y6, y7,
    };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int32_t isum[ROCM_Q4_PREFILL_TOKEN_TILE] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t summs[ROCM_Q4_PREFILL_TOKEN_TILE] = {0, 0, 0, 0, 0, 0, 0, 0};

    /* A 32-byte Q4 payload stores the low and high nibbles for two adjacent
     * 32-value groups.  Load those eight packed words once, then reuse them
     * for both groups and every token in the tile.  This makes weight reuse
     * explicit instead of relying on the compiler or vector cache to hoist
     * repeated loads out of the token loop. */
    #pragma unroll
    for (uint32_t jp = 0u; jp < 4u; jp++) {
        const uint32_t j0 = 2u * jp;
        const uint32_t j1 = j0 + 1u;
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j0, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j1, x->scales, &sc1, &m1);

        int32_t qw[8];
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; i++) {
            qw[i] = *reinterpret_cast<const int32_t *>(
                x->qs + jp * 32u + i * 4u);
        }

        #pragma unroll
        for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
            if (p < n) {
                const Q8Block *y = ys[p];
                int32_t dot0 = 0;
                int32_t dot1 = 0;
                if constexpr (std::is_same<Q8Block, ds4_rocm_q4_lds::aligned_q8_K>::value) {
                    // Only the LDS layout establishes this alignment. Load
                    // one low/high pair at a time to limit live registers;
                    // keep the scalar kernel's dp4a order within each dot.
                    #pragma unroll
                    for (uint32_t i = 0u; i < 8u; i += 4u) {
                        int4 lo, hi;
                        __builtin_memcpy(&lo, __builtin_assume_aligned(
                            y->qs + j0 * 32u + i * 4u, 16), sizeof(lo));
                        __builtin_memcpy(&hi, __builtin_assume_aligned(
                            y->qs + j1 * 32u + i * 4u, 16), sizeof(hi));
                        dot0 = __dp4a(qw[i] & 0x0f0f0f0f, lo.x, dot0);
                        dot1 = __dp4a((qw[i] >> 4) & 0x0f0f0f0f, hi.x, dot1);
                        dot0 = __dp4a(qw[i+1u] & 0x0f0f0f0f, lo.y, dot0);
                        dot1 = __dp4a((qw[i+1u] >> 4) & 0x0f0f0f0f, hi.y, dot1);
                        dot0 = __dp4a(qw[i+2u] & 0x0f0f0f0f, lo.z, dot0);
                        dot1 = __dp4a((qw[i+2u] >> 4) & 0x0f0f0f0f, hi.z, dot1);
                        dot0 = __dp4a(qw[i+3u] & 0x0f0f0f0f, lo.w, dot0);
                        dot1 = __dp4a((qw[i+3u] >> 4) & 0x0f0f0f0f, hi.w, dot1);
                    }
                } else {
                    #pragma unroll
                    for (uint32_t i = 0u; i < 8u; i++) {
                        const int32_t w0 = qw[i] & 0x0f0f0f0f;
                        const int32_t w1 = (qw[i] >> 4) & 0x0f0f0f0f;
                        dot0 = __dp4a(w0, *reinterpret_cast<const int32_t *>(
                                               y->qs + j0 * 32u + i * 4u), dot0);
                        dot1 = __dp4a(w1, *reinterpret_cast<const int32_t *>(
                                               y->qs + j1 * 32u + i * 4u), dot1);
                    }
                }
                isum[p] += (int32_t)sc0 * dot0;
                isum[p] += (int32_t)sc1 * dot1;
                if constexpr (std::is_same<Q8Block, ds4_rocm_q4_lds::aligned_q8_K>::value) {
                    // These exact integer pair sums were prepared once per
                    // activation block, rather than again for each weight row.
                    summs[p] += (int32_t)m0 * y->group_sums[j0];
                    summs[p] += (int32_t)m1 * y->group_sums[j1];
                } else {
                    summs[p] += (int32_t)m0 *
                        (int32_t)(y->bsums[2u * j0] + y->bsums[2u * j0 + 1u]);
                    summs[p] += (int32_t)m1 *
                        (int32_t)(y->bsums[2u * j1] + y->bsums[2u * j1 + 1u]);
                }
            }
        }
    }

    #pragma unroll
    for (uint32_t p = 0u; p < ROCM_Q4_PREFILL_TOKEN_TILE; p++) {
        if (p < n) {
            const float yd = ys[p]->d;
            acc[p] += yd * xd * (float)isum[p] -
                      yd * xmin * (float)summs[p];
        }
    }
}

#endif
