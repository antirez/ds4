#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kHeadTile = 4u;

__device__ __forceinline__ float cache_load(
        const char *cache,
        uint64_t index,
        bool cache_f16) {
    return cache_f16 ?
        __half2float(reinterpret_cast<const __half *>(cache)[index]) :
        reinterpret_cast<const float *>(cache)[index];
}

__global__ void glm_selected_multihead_kernel(
        float *lora_out,
        const float *qk_low,
        const char *kv_lora_cache,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope) {
    const uint32_t token = blockIdx.y;
    const uint32_t head0 = blockIdx.x * kHeadTile;
    if (token >= n_tokens || head0 >= n_head || n_selected == 0u) return;
    const uint32_t remaining = n_head - head0;
    const uint32_t head_count = remaining < kHeadTile ? remaining : kHeadTile;
    const float scale = rsqrtf(static_cast<float>(qk_nope));
    const int32_t *selected_row =
        selected + static_cast<uint64_t>(token) * n_selected;
    extern __shared__ float shared[];
    float *scores = shared;
    float *reduce = scores + static_cast<uint64_t>(kHeadTile) * n_selected;
    float local_max[kHeadTile];
#pragma unroll
    for (uint32_t h = 0; h < kHeadTile; h++) local_max[h] = -INFINITY;

    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        const int32_t row_i = selected_row[s];
        const bool valid = row_i >= 0 && static_cast<uint32_t>(row_i) < cache_cap;
        float dot[kHeadTile];
#pragma unroll
        for (uint32_t h = 0; h < kHeadTile; h++) dot[h] = 0.0f;
        if (valid) {
            const uint64_t cache_base =
                static_cast<uint64_t>(static_cast<uint32_t>(row_i)) *
                kv_lora_dim;
            const float *low = qk_low +
                (static_cast<uint64_t>(token) * n_head + head0) *
                kv_lora_dim;
            for (uint32_t j = 0; j < kv_lora_dim; j++) {
                const float kv = cache_load(
                    kv_lora_cache, cache_base + j, cache_f16);
#pragma unroll
                for (uint32_t h = 0; h < kHeadTile; h++) {
                    if (h < head_count) {
                        dot[h] += low[static_cast<uint64_t>(h) *
                                      kv_lora_dim + j] * kv;
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t h = 0; h < kHeadTile; h++) {
            if (h < head_count) {
                const float score = valid ? dot[h] * scale : -INFINITY;
                scores[static_cast<uint64_t>(h) * n_selected + s] = score;
                local_max[h] = fmaxf(local_max[h], score);
            }
        }
    }

#pragma unroll
    for (uint32_t h = 0; h < kHeadTile; h++) {
        reduce[static_cast<uint64_t>(h) * blockDim.x + threadIdx.x] =
            local_max[h];
    }
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (uint32_t h = 0; h < kHeadTile; h++) {
                float *head_reduce =
                    reduce + static_cast<uint64_t>(h) * blockDim.x;
                head_reduce[threadIdx.x] =
                    fmaxf(head_reduce[threadIdx.x],
                          head_reduce[threadIdx.x + stride]);
            }
        }
        __syncthreads();
    }
    if (!isfinite(reduce[0])) {
        for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
#pragma unroll
            for (uint32_t h = 0; h < kHeadTile; h++) {
                if (h < head_count) {
                    lora_out[((static_cast<uint64_t>(token) * n_head +
                               head0 + h) * kv_lora_dim) + j] = 0.0f;
                }
            }
        }
        return;
    }

    float local_sum[kHeadTile];
#pragma unroll
    for (uint32_t h = 0; h < kHeadTile; h++) local_sum[h] = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
#pragma unroll
        for (uint32_t h = 0; h < kHeadTile; h++) {
            if (h < head_count) {
                float *head_scores =
                    scores + static_cast<uint64_t>(h) * n_selected;
                const float weight = expf(
                    head_scores[s] -
                    reduce[static_cast<uint64_t>(h) * blockDim.x]);
                head_scores[s] = weight;
                local_sum[h] += weight;
            }
        }
    }
#pragma unroll
    for (uint32_t h = 0; h < kHeadTile; h++) {
        reduce[static_cast<uint64_t>(h) * blockDim.x + threadIdx.x] =
            local_sum[h];
    }
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (uint32_t h = 0; h < kHeadTile; h++) {
                float *head_reduce =
                    reduce + static_cast<uint64_t>(h) * blockDim.x;
                head_reduce[threadIdx.x] += head_reduce[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        float acc[kHeadTile];
#pragma unroll
        for (uint32_t h = 0; h < kHeadTile; h++) acc[h] = 0.0f;
        for (uint32_t s = 0; s < n_selected; s++) {
            const int32_t row_i = selected_row[s];
            const bool valid =
                row_i >= 0 && static_cast<uint32_t>(row_i) < cache_cap;
            if (valid) {
                const float kv = cache_load(
                    kv_lora_cache,
                    static_cast<uint64_t>(static_cast<uint32_t>(row_i)) *
                        kv_lora_dim + j,
                    cache_f16);
#pragma unroll
                for (uint32_t h = 0; h < kHeadTile; h++) {
                    if (h < head_count) {
                        acc[h] += scores[static_cast<uint64_t>(h) *
                                         n_selected + s] * kv;
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t h = 0; h < kHeadTile; h++) {
            if (h < head_count) {
                const float denom = fmaxf(
                    reduce[static_cast<uint64_t>(h) * blockDim.x],
                    1.0e-20f);
                lora_out[((static_cast<uint64_t>(token) * n_head +
                           head0 + h) * kv_lora_dim) + j] = acc[h] / denom;
            }
        }
    }
}

}  // namespace

extern "C" int ds4_rocm_glm_selected_multihead_launch(
        float *lora_out,
        const float *qk_low,
        const char *kv_lora_cache,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope) {
    if (!lora_out || !qk_low || !kv_lora_cache || !selected ||
        n_tokens == 0u || n_selected == 0u || cache_cap == 0u ||
        n_head == 0u || kv_lora_dim == 0u || qk_nope == 0u) {
        return 0;
    }
    const dim3 grid((n_head + kHeadTile - 1u) / kHeadTile,
                    n_tokens,
                    1u);
    const size_t shared_bytes =
        static_cast<size_t>(kHeadTile) * (n_selected + 256u) * sizeof(float);
    hipLaunchKernelGGL(glm_selected_multihead_kernel,
                       grid,
                       dim3(256u),
                       shared_bytes,
                       0,
                       lora_out,
                       qk_low,
                       kv_lora_cache,
                       selected,
                       n_tokens,
                       n_selected,
                       cache_cap,
                       cache_f16,
                       n_head,
                       kv_lora_dim,
                       qk_nope);
    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        std::fprintf(stderr,
                     "ds4: ROCm GLM selected multihead launch failed: %s\n",
                     hipGetErrorString(error));
        return 0;
    }
    return 1;
}
