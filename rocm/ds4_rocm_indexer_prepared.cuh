// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <stdint.h>
#include <string.h>

// Prepared-input reference candidate: retain the existing rocWMMA MMA,
// shared score epilogue, head order and masking. Only Q/K staging changes
// to already-rounded half2 packets. Launch 256 threads in wave32 mode,
// ceil(C/128) x ceil(T/16), H=64, D=128; static LDS is 45056 bytes.
// Q[T,H,128] and K[C,128] have 4-byte aligned contiguous half rows. The
// caller must validate geometry, spans, stream state and device capability
// before launch. Native validation is required before automatic dispatch.
// Fixed-size memcpy avoids aliasing half storage through a half2 pointer.
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
#if __CUDA_ARCH__ >= 700 || defined(__HIP_DEVICE_COMPILE__)
#ifdef __HIP_PLATFORM_AMD__
    namespace wmma = rocwmma;
#else
    namespace wmma = nvcuda::wmma;
#endif
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
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

    __shared__ __align__(32) __half a_sh[16 * 128];
    __shared__ __align__(32) __half b_sh[128 * 128];
    __shared__ __align__(32) float c_sh[8 * 16 * 16];

    float acc[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc[i] = 0.0f;

    // DS4_INDEXER_PREPARED_K_BEGIN
    for (uint32_t i = tid * 2u; i < 128u * 128u; i += 512u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        __half2 v = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
        if (comp < n_comp) {
            memcpy(&v, index_comp + (uint64_t)comp * head_dim + d, sizeof(v));
        }
        memcpy(b_sh + d + c * 128u, &v, sizeof(v));
    }
    // DS4_INDEXER_PREPARED_K_END
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        // DS4_INDEXER_PREPARED_Q_BEGIN
        for (uint32_t i = tid * 2u; i < 16u * 128u; i += 512u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            __half2 v = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
            if (token < n_tokens) {
                memcpy(&v, q + ((uint64_t)token * n_head + h) * head_dim + d, sizeof(v));
            }
            memcpy(a_sh + r * 128u + d, &v, sizeof(v));
        }
        // DS4_INDEXER_PREPARED_Q_END
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token0 = tile_t + (local0 >> 4u);
        const float w0 = token0 < n_tokens ? weights[(uint64_t)token0 * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                acc[slot] += fmaxf(c_sh[i], 0.0f) * w0;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc[slot] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}
