// SPDX-License-Identifier: MIT
// Benchmark-only candidate derived from adamlawi/ds4 b4922c9, ds4_cuda.cu.
// No production dispatch includes this file. Include after the production
// Q4_K/Q8_K types, dot8, quarter-warp reducer and Q8_K quantizer.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

namespace ds4_q4_tok8_candidate {
constexpr uint32_t quant_rows_per_launch = 65535u;
struct Plan {
    uint32_t rank, groups, tokens, k, blocks, low_dim, rows;
    uint32_t grid_x, grid_y, quant_launches;
    size_t weight_bytes, input_bytes, q8_bytes, output_bytes;
};
static inline bool bytes(uint64_t count, uint64_t width, size_t *out) {
    if (count > SIZE_MAX / width) return false;
    *out = (size_t)(count * width);
    return true;
}
static inline bool plan(uint64_t rank, uint64_t groups, uint64_t tokens,
                        uint64_t k, Plan *out) {
    if (!out || !rank || !groups || !tokens || !k || k % 256u ||
        rank > UINT32_MAX || groups > UINT32_MAX || k > UINT32_MAX ||
        tokens > 65535u * 8u || rank * groups > UINT32_MAX ||
        tokens * groups > UINT32_MAX) return false;
    Plan p{};
    p.rank = (uint32_t)rank; p.groups = (uint32_t)groups;
    p.tokens = (uint32_t)tokens; p.k = (uint32_t)k;
    p.blocks = p.k / 256u; p.low_dim = (uint32_t)(rank * groups);
    p.rows = (uint32_t)(tokens * groups);
    p.grid_x = (uint32_t)(((uint64_t)p.low_dim + 31u) / 32u);
    p.grid_y = (uint32_t)((tokens + 7u) / 8u);
    p.quant_launches = (uint32_t)(((uint64_t)p.rows + 65534u) / 65535u);
    if (!bytes((uint64_t)p.low_dim * p.blocks, 144u, &p.weight_bytes) ||
        !bytes((uint64_t)p.rows * k, sizeof(float), &p.input_bytes) ||
        !bytes((uint64_t)p.rows * p.blocks, 292u, &p.q8_bytes) ||
        !bytes(tokens * p.low_dim, sizeof(float), &p.output_bytes)) return false;
    *out = p;
    return true;
}
static inline uint32_t quant_batch_rows(const Plan &p, uint64_t row0) {
    if (row0 >= p.rows) return 0;
    const uint64_t left = p.rows - row0;
    return (uint32_t)(left < quant_rows_per_launch ? left : quant_rows_per_launch);
}
} // namespace ds4_q4_tok8_candidate

#ifdef __CUDACC__
__global__ static void attention_output_q4_K_grouped_tok8_kernel(
        float *low,
        const char *w_base,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes,
        uint32_t xq_blocks,
        uint32_t rank,
        uint32_t n_groups,
        uint32_t n_tokens) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t low_dim = n_groups * rank;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    if (row >= low_dim) return;

    const uint32_t group = row / rank;
    const uint32_t tok0 = blockIdx.y * 8u;
    const uint32_t remaining = n_tokens - tok0;
    const uint32_t np = remaining < 8u ? remaining : 8u;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(
        w_base + (uint64_t)row * row_bytes);
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL,
                                      NULL, NULL, NULL, NULL};
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < np) {
            xqb[p] = xq + ((uint64_t)(tok0 + p) * n_groups + group) *
                            xq_blocks;
        }
    }
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block8(
            wr + b,
            xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL,
            np, acc);
    }
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < np) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                low[(uint64_t)(tok0 + p) * low_dim + row] = acc[p];
            }
        }
    }
}

// The remote version put all token/group rows in grid.y. Split only the
// quantizer's row domain; each Q8_K row keeps its exact 256-thread reduction.
static cudaError_t ds4_q4_tok8_quantize(
        const ds4_q4_tok8_candidate::Plan &p, cuda_block_q8_K *q8,
        const float *x, cudaStream_t stream) {
    for (uint64_t row0 = 0; row0 < p.rows;) {
        const uint32_t count = ds4_q4_tok8_candidate::quant_batch_rows(p, row0);
        q8_K_quantize_kernel<<<dim3(p.blocks, count), 256, 0, stream>>>(
            q8 + row0 * p.blocks, x + row0 * p.k, p.k, count);
        const cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        row0 += count;
    }
    return cudaSuccess;
}
static cudaError_t ds4_q4_tok8_matmul(
        const ds4_q4_tok8_candidate::Plan &p, float *out,
        const cuda_block_q4_K *w, const cuda_block_q8_K *q8,
        cudaStream_t stream) {
    attention_output_q4_K_grouped_tok8_kernel<<<
        dim3(p.grid_x, p.grid_y), 256, 0, stream>>>(
            out, reinterpret_cast<const char *>(w), q8,
            (uint64_t)p.blocks * sizeof(cuda_block_q4_K), p.blocks,
            p.rank, p.groups, p.tokens);
    return cudaGetLastError();
}
#endif
