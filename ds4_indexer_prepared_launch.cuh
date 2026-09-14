// SPDX-License-Identifier: MIT
#pragma once
#include "ds4_indexer_prepared.h"
#include "cuda/ds4_indexer_prepare.cuh"
#if defined(__HIPCC__)
#include "rocm/ds4_rocm_indexer_prepared.cuh"
#include "rocm/ds4_rocm_indexer_registers.cuh"
using ds4_indexer_prepared_stream_t = hipStream_t;
#define DS4_INDEXER_PREPARED_GPU(name) hip##name
#define DS4_INDEXER_PREPARED_BACKEND DS4_INDEXER_BACKEND_HIP
#elif defined(__CUDACC__)
#include "cuda/ds4_cuda_indexer_prepared.cuh"
using ds4_indexer_prepared_stream_t = cudaStream_t;
#define DS4_INDEXER_PREPARED_GPU(name) cuda##name
#define DS4_INDEXER_PREPARED_BACKEND DS4_INDEXER_BACKEND_CUDA
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
// Experimental eager pipeline, used by the native comparison fixture. No
// production dispatch selects it until native parity and timings establish
// a device/shape policy. The caller supplies capabilities for the CURRENT
// device and scratch that remains alive until this stream finishes. CUDA
// requires SM >= 70; HIP requires supported wave32 rocWMMA, and the register
// variant specifically requires a gfx1151 image and validated fragment layout.
// Set native_mxf4 when that higher-priority path applies. Inputs are never
// cached across calls; every successful call converts the current Q/K anew.
//
// Return 0 only for a decline before any GPU work, 1 for successful enqueue,
// and -1 for a runtime/launch error. After an error, the caller must propagate
// failure instead of falling back on potentially partially-written scratch.
// Asynchronous execution errors remain the caller's stream-sync responsibility.
static int ds4_indexer_prepared_try(
        const ds4_indexer_plan_request *request,
        const ds4_indexer_prepared_caps *caps,
        const ds4_indexer_prepared_buffers *buffers,
        float scale, int causal, ds4_indexer_prepared_stream_t stream) {
    ds4_indexer_prepared_plan plan = {};
    if (!request || request->backend != DS4_INDEXER_PREPARED_BACKEND ||
        !ds4_indexer_prepared_build(request, caps, &plan) ||
        !ds4_indexer_prepared_buffers_valid(&plan, buffers)) return 0;
    DS4_INDEXER_PREPARED_GPU(StreamCaptureStatus) capture;
    if (DS4_INDEXER_PREPARED_GPU(StreamIsCapturing)(stream, &capture) !=
            DS4_INDEXER_PREPARED_GPU(Success)) return -1;
    if (capture == DS4_INDEXER_PREPARED_GPU(StreamCaptureStatusInvalidated)) return -1;
    if (capture != DS4_INDEXER_PREPARED_GPU(StreamCaptureStatusNone)) return 0;
    if (DS4_INDEXER_PREPARED_GPU(PeekAtLastError)() !=
            DS4_INDEXER_PREPARED_GPU(Success)) return -1;
    __half *q_half = reinterpret_cast<__half *>(buffers->scratch);
    __half *k_half = reinterpret_cast<__half *>(buffers->scratch + plan.k_offset);
    const float *q = reinterpret_cast<const float *>(buffers->q);
    const float *keys = reinterpret_cast<const float *>(buffers->keys);
    const float *weights = reinterpret_cast<const float *>(buffers->weights);
    float *scores = reinterpret_cast<float *>(buffers->scores);
    ds4_indexer_prepare_f16_kernel<<<plan.prepare_blocks, 256, 0, stream>>>(
        q_half, k_half, q, keys, plan.q_elements, plan.k_elements);
    if (DS4_INDEXER_PREPARED_GPU(GetLastError)() !=
            DS4_INDEXER_PREPARED_GPU(Success)) return -1;
    const dim3 score_grid(plan.score_grid_x, plan.score_grid_y);
#if defined(__HIPCC__)
    if (plan.register_scores) {
        ds4_rocm_indexer_scores_registers_kernel<true><<<score_grid, 256, 0, stream>>>(
            scores, q_half, weights, k_half, request->n_comp, request->n_tokens,
            request->pos0, request->n_head, request->head_dim, request->ratio, scale, causal);
    } else
#endif
    {
        indexer_scores_wmma128_prepared_kernel<<<score_grid, 256, 0, stream>>>(
            scores, q_half, weights, k_half, request->n_comp, request->n_tokens,
            request->pos0, request->n_head, request->head_dim, request->ratio, scale, causal);
    }
    return DS4_INDEXER_PREPARED_GPU(GetLastError)() ==
        DS4_INDEXER_PREPARED_GPU(Success) ? 1 : -1;
}
#endif
#undef DS4_INDEXER_PREPARED_GPU
#undef DS4_INDEXER_PREPARED_BACKEND
