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

// Dense Q8_0 matmul (Phase 2 will validate correctness).
//   out[i, j] = sum_k W[i, k] * X[k, j]
//
// Shapes:
//   W_q8_0:  device pointer, [M rows, K cols] in Q8_0 row-major block layout
//            (M * K / 32 * sizeof(block_q8_0) bytes total). K must be
//            divisible by 256.
//   X_f32:   device pointer, [K rows, N cols] F32 row-major.
//   out_f32: device pointer, [M rows, N cols] F32 row-major (caller-allocated).
//   stream:  CUDA stream to launch on.
//
// Returns 0 on success, non-zero on launch failure.
int ds4_mmq_q8_0_dense(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    cudaStream_t  stream);

#ifdef __cplusplus
} // extern "C"
#endif
