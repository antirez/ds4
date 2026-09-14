// SPDX-License-Identifier: MIT
// Include after CUDA/HIP runtime and half headers. Both backends instantiate
// this kernel with their native 144-byte GGUF Q4_K block type.
#pragma once
#include <stddef.h>
#include "ds4_q4_dequant_layout.h"

template<typename BlockQ4>
__global__ static void ds4_q4_dequant_f16_pair32_kernel(
        __half *dst, const BlockQ4 *src, uint64_t total_blocks) {
    namespace dq = ds4_q4_dequant;
    static_assert(sizeof(BlockQ4) == dq::block_bytes &&
                  offsetof(BlockQ4, d) == 0 && offsetof(BlockQ4, dmin) == 2 &&
                  offsetof(BlockQ4, scales) == 4 && offsetof(BlockQ4, qs) == 16,
                  "Q4_K GGUF ABI");
    const uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if ((task >> 3u) >= total_blocks) return;
    const uint64_t lo_chunk = dq::low_chunk(task);
    // Complete Q4_K rows are contiguous: no runtime 64-bit row division.
    const BlockQ4 *xb = src + (task >> 3u);
    const float d = __half2float(__ushort_as_half(xb->d));
    const float dmin = __half2float(__ushort_as_half(xb->dmin));

    // The low/high groups share one aligned 16-byte packed vector and the
    // block scale. Each thread writes its low group before decoding the high
    // group, so no 32-value output array needs to stay live. The final load
    // stays inside qs, never in the next block's metadata.
    const uint4 packed = *reinterpret_cast<const uint4 *>(xb->qs + dq::packed_offset(lo_chunk));
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
    for (uint32_t hi = 0; hi < 2u; ++hi) {
        const uint64_t chunk = lo_chunk + 2u * hi;
        const uint32_t g = dq::group(chunk);
        uint8_t scale, minimum;
        dq::scale_min(g, xb->scales, &scale, &minimum);
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
        for (uint32_t h = 0; h < 2u; ++h) {
            uint32_t pairs[4];
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t k = h * 8u + j * 2u;
                const uint32_t q0 = dq::nibble(words[k / 4u], k & 3u, g);
                const uint32_t q1 = dq::nibble(words[k / 4u], (k & 3u) + 1u, g);
                // Same F32 algebra and single RN-to-F16 boundary as the scalar
                // kernels. Do not replace with half arithmetic or change FMA flags.
                const __half a = __float2half_rn(
                    (d * (float)scale) * (float)q0 - dmin * (float)minimum);
                const __half b = __float2half_rn(
                    (d * (float)scale) * (float)q1 - dmin * (float)minimum);
                pairs[j] = dq::half_pair(__half_as_ushort(a), __half_as_ushort(b));
            }
            *reinterpret_cast<uint4 *>(dst + chunk * 16u + h * 8u) =
                make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
        }
    }
}
