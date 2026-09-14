/* SPDX-License-Identifier: MIT */
#ifndef DS4_CUDA_INDEXER_PREPARED_CUH
#define DS4_CUDA_INDEXER_PREPARED_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <string.h>

/* Candidate consumer of the same RNE F16 operands used by the existing
 * indexer_scores_wmma128_kernel. Q and K are prepared once per scoring call;
 * weights and final scores remain F32. Only the two LDS staging loops differ
 * from that kernel: WMMA shape/order, register mapping, head reduction, final
 * scale and causal masks are unchanged.
 *
 * Launch contract: SM >= 70, 256 threads, ceil(n_comp/128) by ceil(n_tokens/32)
 * blocks, D=128 (current wrapper additionally restricts H=64). Q [T,H,128] and
 * K [C,128] are contiguous half arrays with 4-byte-aligned bases, disjoint from
 * the output. Their producer must use the baseline __float2half rounding.
 * Dimensions, position arithmetic, capacities and launch resource limits are
 * validated before enqueue by the wrapper; this raw kernel is not a fallback
 * or an admission API. Static LDS is 43520 bytes, exactly as in the baseline.
 *
 * Fixed-size memcpy gives the compiler a 32-bit half2 copy without violating
 * C++ aliasing rules. Padded LDS rows (136 half values) preserve 4-byte pair
 * alignment; the shared bases are explicitly aligned for WMMA loads.
 */
__global__ static void indexer_scores_wmma128_prepared_kernel(
        float *scores,
        const __half *q,
        const float *weights,
        const __half *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 32u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 32u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 32u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    /* The padded stride avoids ldmatrix bank conflicts. Two token tiles
     * share each staged index fragment, while WMMA accumulators remain in
     * registers through the head reduction. */
    __shared__ __align__(32) __half a_sh[32 * 136];
    __shared__ __align__(32) __half b_sh[128 * 136];

    const uint32_t lane = tid & 31u;
    const uint32_t quad = lane >> 2u;
    const uint32_t tcol = (lane & 3u) * 2u;

    float acc0[8], acc1[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) {
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;
    }

    // DS4_INDEXER_PREPARED_K_BEGIN
    // Two already-rounded operands per lane: one 32-bit global read and LDS
    // store, without repeating F32 -> F16 conversion in every token tile.
    for (uint32_t i = tid * 2u; i < 128u * 128u; i += 512u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        __half2 v = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
        if (comp < n_comp) {
            memcpy(&v, index_comp + (uint64_t)comp * head_dim + d, sizeof(v));
        }
        memcpy(b_sh + d + c * 136u, &v, sizeof(v));
    }
    // DS4_INDEXER_PREPARED_K_END
    __syncthreads();

    const uint32_t t0_lo = tile_t + quad;
    const uint32_t t0_hi = tile_t + quad + 8u;
    const uint32_t t1_lo = tile_t + 16u + quad;
    const uint32_t t1_hi = tile_t + 24u + quad;

    for (uint32_t h = 0; h < n_head; h++) {
        // DS4_INDEXER_PREPARED_Q_BEGIN
        for (uint32_t i = tid * 2u; i < 32u * 128u; i += 512u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            __half2 v = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
            if (token < n_tokens) {
                memcpy(&v, q + ((uint64_t)token * n_head + h) * head_dim + d, sizeof(v));
            }
            memcpy(a_sh + r * 136u + d, &v, sizeof(v));
        }
        // DS4_INDEXER_PREPARED_Q_END
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
        wmma::fill_fragment(c0, 0.0f);
        wmma::fill_fragment(c1, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a0, a_sh + k0, 136);
            wmma::load_matrix_sync(a1, a_sh + 16u * 136u + k0, 136);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 136u + k0, 136);
            wmma::mma_sync(c0, a0, b_frag, c0);
            wmma::mma_sync(c1, a1, b_frag, c1);
        }

        const float w0_lo = t0_lo < n_tokens ? weights[(uint64_t)t0_lo * n_head + h] : 0.0f;
        const float w0_hi = t0_hi < n_tokens ? weights[(uint64_t)t0_hi * n_head + h] : 0.0f;
        const float w1_lo = t1_lo < n_tokens ? weights[(uint64_t)t1_lo * n_head + h] : 0.0f;
        const float w1_hi = t1_hi < n_tokens ? weights[(uint64_t)t1_hi * n_head + h] : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            acc0[i] += fmaxf(c0.x[i], 0.0f) * ((i & 2) ? w0_hi : w0_lo);
            acc1[i] += fmaxf(c1.x[i], 0.0f) * ((i & 2) ? w1_hi : w1_lo);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t row = quad + ((i & 2) ? 8u : 0u);
        const uint32_t col = tcol + (i & 1) + ((i & 4) ? 8u : 0u);
        const uint32_t comp = tile_c + warp * 16u + col;
        for (uint32_t rt = 0; rt < 2u; rt++) {
            const uint32_t token = tile_t + rt * 16u + row;
            if (token < n_tokens && comp < n_comp) {
                float out = (rt ? acc1[i] : acc0[i]) * scale;
                if (causal) {
                    const uint32_t visible = (pos0 + token + 1u) / ratio;
                    if (comp >= visible) out = -INFINITY;
                }
                scores[(uint64_t)token * n_comp + comp] = out;
            }
        }
    }
#endif
}

#endif
