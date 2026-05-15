// SPDX-License-Identifier: MIT
// ds4_mmq.h - public C ABI for ds4's quantized matmul kernels.
//
// All functions are extern "C" so ds4.c / ds4_cuda.cu can call them
// without C++ compilation. Functions return 0 on success and non-zero on
// failure (with stderr error message). Device pointers are caller-owned.
//
// Phase 0: skeleton only. Q8_0 dense entry compiles and instantiates
// mul_mat_q_case<Q8_0> but is not yet wired into ds4_cuda.cu.
// Phase 1: Q8_1 activation quantizer wrapper added.
// Phase 2: Q8_0 dense entry verified against cublas+dequant baseline.
// Phase 3: Q2_K + IQ2_XXS dense entries.
// Phase 4: MoE _id variants of all three.

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One-time init. Sets the current CUDA device and triggers lazy population
// of the device-info singleton. Safe to call repeatedly.
//
//   device: CUDA device ordinal (0 for the primary GPU).
// Returns 0 on success.
int ds4_mmq_init(int device);

// Query whether ds4_mmq is willing to handle a given matmul. Returns
//   1 if mmq is faster than dequant+cublas for this shape on this device,
//   0 otherwise (caller should fall back to its existing dequant+cublas path).
//
// Wraps ggml_cuda_should_use_mmq. type_x uses ds4 quant codes which match
// ggml's enum:
//   8  = Q8_0
//   10 = Q2_K
//   16 = IQ2_XXS
//
//   ne11:      batch dimension (number of activation columns / tokens).
//   n_experts: 0 for dense matmul, >0 for MoE (e.g. 256 for V4 Flash).
int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts);

// Dense matmul entry points. Per-type wrappers that all share the same
// underlying mul_mat_q template, parameterised by the weight quant type.
//
// All three variants compute:
//
//   out[col, row] = sum_k W[row, k] * X[k, col]      0 <= row < M, 0 <= col < N
//
// Layouts (matching ggml + llama.cpp mmq conventions, all on device):
//   W:       [M rows, K cols], row-major, packed in the type-specific block
//            format. K must be a multiple of 256.
//   X_f32:   [N rows, K cols] F32 row-major (logical [K, N] with K
//            innermost - i.e. for each "column" col of the logical [K, N]
//            matrix, K contiguous floats live at X[col*K .. col*K + K]).
//   out_f32: caller-allocated, M*N floats. mmq writes in column-major:
//            out[col*M + row]. Callers expecting row-major must transpose.
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q8_0_dense(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

int ds4_mmq_q2_K_dense(
    const void  * W_q2_K,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

int ds4_mmq_iq2_xxs_dense(
    const void  * W_iq2_xxs,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

// MoE matmul entry points. For each (token, slot-within-token's-top-k) pair
// the kernel computes:
//
//   out[col, row] = sum_k W[ids[token, slot], row, k] * X[token, k]
//
// where col = token * n_expert_used + slot, row in [0, M).  The caller is
// responsible for any downstream sum-weighted-by-router-weights reduction
// across the n_expert_used dimension (Phase 5 wires this into ds4's
// existing moe_sum_kernel).
//
// Layouts:
//   W:       device pointer, [n_experts, M rows, K cols] in the
//            type-specific block format.  Per-expert slab is M*K/blck
//            blocks stored contiguously; experts are stacked.
//   X_f32:   device pointer, [n_tokens, K] F32 row-major (K innermost).
//   ids:     device pointer, [n_tokens, n_expert_used] int32_t row-major.
//            ids[t*n_expert_used + s] is the expert id for token t's
//            s-th routing slot.  Values must be in [0, n_experts).
//   out_f32: caller-allocated, M * n_tokens * n_expert_used floats.
//            Column-major: out[col*M + row].
//
// K must be a multiple of 256.  n_expert_used must be one of the values
// the vendored mm_ids_helper template specialises on: 2, 4, 6, 8, 16, 32
// (or any other value, which falls back to the generic path).  For V4
// Flash, n_expert_used = 6.
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q8_0_moe(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q2_K_moe(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_iq2_xxs_moe(
    const void    * W,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_f32,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

#ifdef __cplusplus
} // extern "C"
#endif
