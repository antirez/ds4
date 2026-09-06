// SPDX-License-Identifier: MIT
// Internal CUDA Q4_K dense-prefill experiment.  This header intentionally
// exposes only the pre-quantized enqueue boundary; allocation and Q8_1
// quantization stay owned by ds4_mmq.cu.

#pragma once

#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_runtime.h>
#endif

#include <stddef.h>

#if !defined(GGML_USE_HIP)

#ifdef __cplusplus
extern "C" {
#endif

// Returns non-zero when the requested CUDA compute capability can execute
// the m128n128, 16-warp integer-MMA kernel.
int ds4_mmq_q4_K_dense_16warp_available(int cc);

// Conservative standalone production admission gate. Availability and shape
// are both checked; it admits M>=1024 and only complete 128x128 output tiles.
// The K envelope covers the production 8192-wide attention output projection.
// The dispatcher separately enforces the m128n128 reference selector and its
// candidate-grid efficiency gate. A false result must fall back. The pair
// dispatcher has a separate per-leg M>=512 gate.
int ds4_mmq_q4_K_dense_16warp_supported(int cc, int M, int N, int K);

// Opt in the 56 KiB dynamic-shared-memory launch on the current device.
// Call once during device initialization (and once again after switching to a
// different device) before enqueue.  The operation is idempotent.
int ds4_mmq_q4_K_dense_16warp_prepare(void);

// Enqueue-only dense Q4_K GEMM over an already resident canonical MMQ Q8_1
// activation buffer.
//
//   W      raw row-major block_q4_K, [M][K/256]
//   q8_ds4 block_q8_1_mmq DS4 (half scale + half sum), [K/128][N]
//   out    column-major float, [N][M]
//
// ds4_mmq_q4_K_dense_16warp_prepare must have succeeded on the current device.
// The kernel owns the complete K reduction for every output tile: it never
// uses stream-K and writes every valid output exactly once.  No allocation,
// memset, quantization, synchronization, or host/device copy is performed.
// Returns 0 after a successful enqueue and a negative value otherwise.
int ds4_mmq_q4_K_dense_16warp_enqueue(
    const void   * W,
    const void   * q8_ds4,
    float        * out,
    int            M,
    int            N,
    int            K,
    cudaStream_t   stream);

// Return the caller-owned fixup storage required by the canonical Stream-K
// partition for this dense MxN output shape and SM count.  Zero means either
// that no fixup is necessary (the selected grid owns complete tiles) or that
// the arguments/size cannot be represented; enqueue repeats all validation.
// The storage, when non-zero, is a byte buffer and need only remain valid until
// the work already enqueued on `stream` has completed.
size_t ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
    int M,
    int N,
    int nsm);

// Enqueue the same 16-warp arithmetic using canonical CUDA MMQ Stream-K
// scheduling and its exact Q4_K fixup reduction tree.  W, q8_ds4 and out use
// the layouts documented above.  `scratch` must provide at least the size
// returned by ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes; it may be null
// when that function returns zero.  The routine performs only asynchronous
// memset/kernel operations and does not allocate or synchronize.
// ds4_mmq_q4_K_dense_16warp_prepare must have succeeded on the current device.
int ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
    const void   * W,
    const void   * q8_ds4,
    float        * out,
    void         * scratch,
    size_t         scratch_bytes,
    int            M,
    int            N,
    int            K,
    int            nsm,
    cudaStream_t   stream);

#ifdef __cplusplus
}
#endif

#endif // !defined(GGML_USE_HIP)
