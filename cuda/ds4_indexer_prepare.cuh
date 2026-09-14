// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#include <string.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
// Q and K both have 128-wide rows. Convert each input exactly once per call,
// with the same round-to-nearest-even conversion used by the old tile loads.
// F32 sources require only scalar (four-byte) alignment. Destination packets
// are raw half2 bits, copied with memcpy to avoid a half/half2 aliasing cast.
__global__ static void ds4_indexer_prepare_f16_kernel(
        __half *q_half, __half *k_half, const float *q, const float *keys,
        uint64_t q_elements, uint64_t k_elements) {
    const uint64_t q_pairs = q_elements / 2u;
    const uint64_t pairs = q_pairs + k_elements / 2u;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t pair = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         pair < pairs; pair += stride) {
        const bool is_q = pair < q_pairs;
        const uint64_t i = 2u * (is_q ? pair : pair - q_pairs);
        const float *src = is_q ? q : keys;
        __half *dst = is_q ? q_half : k_half;
        const __half2 value = __floats2half2_rn(src[i], src[i + 1u]);
        memcpy(dst + i, &value, sizeof(value));
    }
}
#endif
