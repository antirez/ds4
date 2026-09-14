// SPDX-License-Identifier: MIT
#pragma once

#include <stdint.h>
#include <limits.h>

// Generic NVIDIA MMVQ owns one row (four for small K) at N=1 and two rows
// at N=2..8. Keep complete four-row cohorts for both paths: every output is
// overwritten and neither loader needs padded weights. Bound signed weight
// block indices and the uint32 column/output offsets for the multi-token
// store. Device/backend and opt-out checks are host policy; the persistent
// K1024 experiment keeps its original epilogue.
static inline bool ds4_q4_mmvq_epilogue_shape_ok(int M, int N, int K) {
    return M > 0 && M % 4 == 0 && N >= 1 && N <= 8 && K > 0 && K % 256 == 0 &&
           (uint64_t)M * (uint64_t)N <= UINT32_MAX &&
           (uint64_t)M * (uint64_t)(K / 256) <= INT_MAX;
}

// Grouped decode flattens (token, group) into grid.y. In addition to full
// row cohorts, bound the signed weight-block index and unsigned channel
// offsets used by canonical MMVQ. Strides are in Q8_1 blocks, not bytes.
static inline bool ds4_q4_mmvq_grouped_shape_ok(
        int M, int K, int n_tokens, int n_groups, int64_t stride_col_y) {
    if (!ds4_q4_mmvq_epilogue_shape_ok(M, 1, K) ||
        n_tokens <= 0 || n_tokens > 8 || n_groups <= 0 || n_groups > 16 ||
        stride_col_y < K / 32 || stride_col_y > INT_MAX) return false;
    const uint64_t channels = (uint64_t)n_tokens * (uint64_t)n_groups;
    return (uint64_t)n_groups * (uint64_t)M * (uint64_t)(K / 256) <= INT_MAX &&
           channels * (uint64_t)M <= UINT32_MAX &&
           (uint64_t)stride_col_y <= UINT32_MAX / channels;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
static inline uint32_t ds4_q4_mmvq_sanitize_bits(uint32_t bits) {
    // Preserve every finite bit pattern, including -0 and subnormals; map
    // all NaNs and both infinities to +0, like ds4_mmq_sanitize_f32_kernel.
    return (bits & 0x7f800000u) == 0x7f800000u ? 0u : bits;
}
