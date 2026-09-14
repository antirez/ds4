// SPDX-License-Identifier: MIT
// HC prefill preparation: preserve the standalone F32 RMS reduction and its
// normalized F32 rounding boundary, then write the already-required FP16 RHS.
#pragma once

#include <float.h>
#include <limits.h>
#include <stdint.h>
#include "../cuda/ds4_hc_norm_mix.cuh"

namespace ds4_rocm_hc_prefill {
struct sizes {
    uint64_t weights, input, output, half;
};

// Restrict this entry to HC rows that already consume FP16 RHS through the
// prepared GEMM dispatcher. The <=8-token 16384x24 path has its own tiny-wave
// kernel and remains unchanged, as do decode/verifier/mixed batches.
static inline bool eligible(uint64_t n, uint64_t m, uint64_t rows,
        ds4_gpu_execution_phase phase, bool quality, bool glm, bool blas_ready,
        uintptr_t out, uint64_t out_capacity, uintptr_t x, uint64_t x_capacity,
        uintptr_t model, uint64_t model_size, uint64_t weight_offset,
        float eps, sizes *bytes) {
    using namespace ds4_hc_norm_mix;
    if (!bytes || (n != 16384u && n != 28672u) || m != 24u ||
        rows <= 1u || rows > INT32_MAX || (n == 16384u && rows <= 8u) ||
        !ds4_gpu_execution_phase_allows_prefill(phase) || quality || glm || !blas_ready ||
        !(eps > 0.0f && eps <= FLT_MAX)) return false;
    // The bounded dimensions above make these products representable in u64.
    const sizes need = {m * n * sizeof(uint16_t), rows * n * sizeof(float),
                        rows * m * sizeof(float), rows * n * sizeof(uint16_t)};
    if (!range_valid(out, need.output) || !range_valid(x, need.input) ||
        !range_valid(model, model_size) || out_capacity < need.output ||
        x_capacity < need.input || (out & 3u) || (x & 3u) ||
        weight_offset > model_size || need.weights > model_size - weight_offset ||
        ((model + weight_offset) & 1u) ||
        overlaps(out, need.output, x, need.input) ||
        overlaps(out, need.output, model + weight_offset, need.weights)) return false;
    *bytes = need;
    return true;
}
} // namespace ds4_rocm_hc_prefill

#if defined(__HIPCC__) || defined(__CUDACC__)
__global__ static void rocm_hc_rms_norm_f16_kernel(
        __half *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    __half *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        // The standalone RMS stores this product to F32 before conversion.
        // Retain that fence even under HIP fast math; do not move scale after
        // the projection or reassociate the normalized operand.
        orow[i] = __float2half(ds4_hc_normalized_f32(xr[i], scale));
    }
}
#endif
