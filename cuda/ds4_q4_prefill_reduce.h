// SPDX-License-Identifier: MIT
// Exact 256-thread reduction for the resident Q4 prefill RMS/RoPE epilogue.
// Include after cuda_runtime.h in CUDA translation units.
#pragma once

#if defined(__CUDACC__)
#define DS4_Q4_REDUCE_INLINE __host__ __device__ __forceinline__
#else
#define DS4_Q4_REDUCE_INLINE inline
#endif

// Materialize each rounded addition: the former shared-memory stages were
// separated by barriers, so reassociating this tree would change RMS scales.
DS4_Q4_REDUCE_INLINE static float ds4_q4_prefill_add(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    volatile float rounded = a + b;
    return rounded;
#endif
}

// One lane computes precisely its original stride-128, stride-64 and
// stride-32 subtree. The input is the unmodified array of 256 partial sums.
DS4_Q4_REDUCE_INLINE static float ds4_q4_prefill_reduce_lane(
        const float *partial, unsigned lane) {
    const float a = ds4_q4_prefill_add(partial[lane], partial[lane + 128u]);
    const float b = ds4_q4_prefill_add(partial[lane + 64u], partial[lane + 192u]);
    const float c = ds4_q4_prefill_add(partial[lane + 32u], partial[lane + 160u]);
    const float d = ds4_q4_prefill_add(partial[lane + 96u], partial[lane + 224u]);
    return ds4_q4_prefill_add(ds4_q4_prefill_add(a, b),
                              ds4_q4_prefill_add(c, d));
}

#if defined(__CUDACC__)
// All 256 threads must call this helper. Only the first warp reads the
// shared leaves and runs the final 16/8/4/2/1 tree; the second CTA barrier
// publishes its result. This replaces nine CTA barriers with two, without
// changing the per-thread sum of squares or any reduction operand order.
__device__ __forceinline__ static float ds4_q4_prefill_reduce_256(
        float sum, float *partial) {
    const unsigned tid = threadIdx.x;
    partial[tid] = sum;
    __syncthreads();
    if (tid < 32u) {
        float folded = ds4_q4_prefill_reduce_lane(partial, tid);
#pragma unroll
        for (unsigned stride = 16u; stride > 0u; stride >>= 1u) {
            const float other = __shfl_down_sync(0xffffffffu, folded, stride);
            if (tid < stride) folded = ds4_q4_prefill_add(folded, other);
        }
        if (tid == 0u) partial[0] = folded;
    }
    __syncthreads();
    return partial[0];
}
#endif

#undef DS4_Q4_REDUCE_INLINE
