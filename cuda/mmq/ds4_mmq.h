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

int ds4_mmq_q4_K_dense(
    const void  * W_q4_K,
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

int ds4_mmq_q4_K_moe(
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

// Paired MoE entries. Compute gate AND up over the same activation in a
// single call so the Q8_1 quantize of X (and the mm_ids_helper bookkeeping)
// happens once instead of twice. Both weights must be the same quant type
// and the same shape (M, K, n_experts); out_a / out_b have the same layout
// as a single ds4_mmq_<type>_moe call. Saves one launch of
// quantize_mmq_q8_1_cuda and one ggml_cuda_launch_mm_ids_helper per MoE
// block. See ds4_mmq.cu / routed_moe_launch for the wiring.
//
// Returns 0 on success; on error neither output is guaranteed valid.

int ds4_mmq_iq2_xxs_moe_pair(
    const void    * W_a,
    const void    * W_b,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_a,
    float         * out_b,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe_pair(
    const void    * W_a,
    const void    * W_b,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_a,
    float         * out_b,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// MoE vector matmul entries (Step 6). Same signature and semantics as the
// ds4_mmq_<type>_moe entries above, but route through llama.cpp's mmvq
// kernels instead of mmq. mmvq is structurally optimised for small batch
// counts (single-token decode, short prefill), where mmq's tile-based
// approach wastes work on empty columns.
//
// Constraints:
//   - n_tokens * something must fit under mmvq's per-arch batch cap
//     (MMVQ_MAX_BATCH_SIZE = 8 on Blackwell). Specifically, ncols_dst as
//     computed by the wrapper must be <= 8. The wrapper rejects with -1
//     if the request is too large.
//   - K must be a multiple of 256 (same as the mmq path).
//
// Unlike the mmq path, mmvq consumes a CANONICAL block_q8_1 buffer (not
// the interleaved block_q8_1_mmq the mmq path uses). The wrapper builds
// the canonical buffer internally; callers cannot reuse a Q8_1 buffer
// previously built for the mmq path.
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q8_0_moe_vec(
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

int ds4_mmq_q2_K_moe_vec(
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

int ds4_mmq_iq2_xxs_moe_vec(
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

int ds4_mmq_q4_K_moe_vec(
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

// Pair-fused MoE vector matmul entries (Step 6). Computes
//
//   out[col, row] = (W_a[ids, row, :] @ X[token, :])
//                 * silu(W_b[ids, row, :] @ X[token, :])
//
// in a SINGLE mmvq launch via mmvq's built-in fusion (fusion.gate = W_b,
// fusion.glu_op = GGML_GLU_OP_SWIGLU). The kernel applies silu to the
// fusion.gate matmul and multiplies into the main matmul: pass the
// SwiGLU "up" weights as W_a and the SwiGLU "gate" weights as W_b to
// match ds4's expected silu(gate)*up semantics. The DeepSeek V4 clamp
// and router-weight multiplication are NOT applied by the kernel - the
// caller is expected to apply them as a small post-process (or to skip
// clamp if clamp==0).
//
// Constraints:
//   - n_tokens = 1 ONLY. mmvq supports fusion only at ncols_dst = 1.
//   - K must be a multiple of 256.
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_iq2_xxs_moe_pair_vec(
    const void    * W_a,
    const void    * W_b,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_silu,
    int             M,
    int             K,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

int ds4_mmq_q4_K_moe_pair_vec(
    const void    * W_a,
    const void    * W_b,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_silu,
    int             M,
    int             K,
    int             n_experts,
    int             n_expert_used,
    cudaStream_t    stream);

// Dense vector matmul entry (Step 6). Same shape semantics as
// ds4_mmq_q8_0_dense but routed through mmvq for batch counts that
// favour the vec path (n_tokens <= 8 on Blackwell).
//
// Returns 0 on success, non-zero on validation or launch failure.

int ds4_mmq_q8_0_dense_vec(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

// Set the thread-local stream that the internal cuda pool uses for
// cudaMallocAsync / cudaFreeAsync.  Defaults to cudaStreamPerThread.
// Step 8 (CUDA Graphs) calls this with the capture stream so pool
// allocations land on the captured stream and don't invalidate capture.
// Pass NULL to reset to cudaStreamPerThread.
void ds4_pool_set_stream(cudaStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif
