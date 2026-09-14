// SPDX-License-Identifier: MIT
// HC decode RMS preparation. CUDA keeps its cuBLAS projection; ROCm keeps
// its ordered F32 activation matvec. No Sinkhorn or HC coefficients change.
#pragma once

#include <stdint.h>
#include "../ds4_gpu_phase.h"

namespace ds4_hc_norm_mix {
enum : uint32_t { width = 16384u, outputs = 24u };
enum : uint64_t {
    x_bytes = (uint64_t)width * sizeof(float),
    out_bytes = (uint64_t)outputs * sizeof(float),
    weight_bytes = (uint64_t)width * outputs * sizeof(uint16_t),
    half_bytes = (uint64_t)width * sizeof(uint16_t)
};

static inline bool range_valid(uintptr_t ptr, uint64_t bytes) {
    return ptr != 0u && bytes != 0u && bytes - 1u <= UINTPTR_MAX - ptr;
}

static inline bool overlaps(uintptr_t a, uint64_t a_bytes,
                            uintptr_t b, uint64_t b_bytes) {
    if (a_bytes == 0u || b_bytes == 0u) return false;
    return a <= b ? b - a < a_bytes : a - b < b_bytes;
}

// Pure preflight shared with host tests. The dispatcher is intentionally
// limited to one HC decode row; direct callers retain AUTO compatibility.
static inline bool eligible(uint32_t n, uint32_t out_dim,
                            ds4_gpu_execution_phase phase, bool quality,
                            uintptr_t out, uint64_t out_capacity,
                            uintptr_t x, uint64_t x_capacity,
                            uintptr_t model, uint64_t model_size,
                            uint64_t weight_offset) {
    return n == width && out_dim == outputs && !quality &&
           (phase == DS4_GPU_PHASE_AUTO || phase == DS4_GPU_PHASE_DECODE) &&
           out_capacity >= out_bytes && x_capacity >= x_bytes &&
           range_valid(out, out_bytes) && range_valid(x, x_bytes) &&
           range_valid(model, model_size) &&
           (out & 3u) == 0u && (x & 3u) == 0u &&
           weight_offset <= model_size &&
           weight_bytes <= model_size - weight_offset &&
           ((model + weight_offset) & 1u) == 0u &&
           !overlaps(out, out_bytes, x, x_bytes) &&
           !overlaps(out, out_bytes, model + weight_offset, weight_bytes);
}
} // namespace ds4_hc_norm_mix

#if defined(__CUDACC__) || defined(__HIPCC__)
// Mirrors rms_norm_plain_kernel: ascending stride-256 sums, then the same
// shared-memory 128/64/.../1 tree and rsqrtf expression. Only the output is
// reduced from a full F32 row to one F32 scale. Keep n as a runtime argument
// so the production compiler sees the same loop bounds as the reference.
__global__ static void ds4_hc_rms_scale_kernel(
        float *scale_out, const float *x, uint32_t n, float eps) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u) scale_out[0] = rsqrtf(partial[0] / (float)n + eps);
}

// The standalone RMS kernel stores x*scale to F32 before the matvec. Keep
// that rounding boundary even under ROCm -ffast-math: the empty register asm
// prevents reassociation into the weight product, without a local-memory
// volatile spill. CUDA's explicit rounded multiply supplies the same fence.
__device__ __forceinline__ static float ds4_hc_normalized_f32(float x, float scale) {
#if defined(__HIP_DEVICE_COMPILE__)
    float normalized = x * scale;
    asm volatile("" : "+v"(normalized));
    return normalized;
#elif defined(__CUDA_ARCH__)
    return __fmul_rn(x, scale);
#else
    return x * scale;
#endif
}

// Same 32 contiguous chunks and ascending scalar reduction as
// matmul_f16_ordered_chunks_kernel. This does not quantize activations to
// FP16: ROCm's one-token reference consumes the normalized F32 values.
__global__ static void ds4_hc_f16_project_scaled_ordered_kernel(
        float *out, const __half *w, const float *x, const float *scale_in,
        uint64_t in_dim, uint64_t out_dim) {
    const uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out_dim) return;
    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float scale = scale_in[0];
    for (uint64_t i = k0; i < k1; i++) {
        const float normalized = ds4_hc_normalized_f32(x[i], scale);
        sum += __half2float(wr[i]) * normalized;
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0u) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[row] = total;
    }
}
#endif
