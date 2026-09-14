// SPDX-License-Identifier: MIT
// CUDA activation quantizers. The false specialization is the original
// shared-memory oracle; true only changes the independent 32-lane Q8_0 max.
// Include from CUDA translation units after cuda_runtime.h and math.h.
#pragma once

template<bool WARP_REDUCE>
__device__ __forceinline__ static float ds4_cuda_q8_0_warp_amax(
        float a, float *shared) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    if (WARP_REDUCE) {
        // All 32 lanes participate, including zero-padded input tails.
        // Keep precisely the legacy 16,8,4,2,1 comparison tree, then broadcast
        // lane 0: using each lane's suffix maximum would change quantization.
        #pragma unroll
        for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
            const float other = __shfl_down_sync(0xffffffffu, a, stride, 32);
            if (lane < stride) a = fmaxf(a, other);
        }
        return __shfl_sync(0xffffffffu, a, 0, 32);
    }
    shared[tid] = a;
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    return shared[tid - lane];
}

template<bool WARP_REDUCE>
__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    const float amax = ds4_cuda_q8_0_warp_amax<WARP_REDUCE>(a, vals);
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

template<bool WARP_REDUCE>
__global__ static void quantize_q8_0_group_slice_rows_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t group_dim,
        uint64_t blocks,
        uint32_t n_groups_total,
        uint32_t group0,
        uint32_t group_cnt) {
    const uint64_t b = blockIdx.x;
    const uint64_t packed_row = blockIdx.y;
    if (b >= blocks) return;
    const uint64_t token = packed_row / group_cnt;
    const uint64_t group = group0 + packed_row - token * group_cnt;
    const uint64_t i0 = b * 32u;
    const uint64_t bn = group_dim - i0 < 32u ? group_dim - i0 : 32u;
    const float *xr = x +
        (token * n_groups_total + group) * group_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    const float amax = ds4_cuda_q8_0_warp_amax<WARP_REDUCE>(a, vals);
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0u) xscale[packed_row * blocks + b] = d;
    int8_t *dst = xq + (packed_row * blocks + b) * 32u;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

// Q8_K uses its original 256-thread signed-max reduction, including ties.
template<bool WARP_REDUCE, typename BlockQ8K>
__global__ static void q8_K_q8_0_quantize_kernel(
        BlockQ8K *out,
        int8_t *q8_0,
        float *q8_0_scale,
        const float *x,
        uint32_t in_dim,
        uint32_t n_rows) {
    const uint32_t b = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / 256u) return;
    const float *xr = x + (uint64_t)row * in_dim +
                      (uint64_t)b * 256u;
    BlockQ8K *yb = out +
        (uint64_t)row * (in_dim / 256u) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const float v = tid < 256u ? xr[tid] : 0.0f;

    const float q8_amax = ds4_cuda_q8_0_warp_amax<WARP_REDUCE>(
        tid < 256u ? fabsf(v) : 0.0f, abs_part);
    const uint32_t q8_blocks = in_dim / 32u;
    const uint32_t q8_block = b * 8u + warp;
    const float d = q8_amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0u) {
        q8_0_scale[(uint64_t)row * q8_blocks + q8_block] = d;
    }
    int qv = (int)lrintf(v * id);
    qv = qv > 127 ? 127 : (qv < -128 ? -128 : qv);
    q8_0[((uint64_t)row * q8_blocks + q8_block) * 32u + lane] =
        (int8_t)qv;
    // Only the legacy Q8_0 reduction reads shared storage before it is reused.
    if (!WARP_REDUCE) __syncthreads();

    abs_part[tid] = tid < 256u ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    const float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < 256u) yb->qs[tid] = 0;
        if (tid < 256u / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < 256u) {
        int kv = (int)lrintf(iscale_s * xr[tid]);
        if (kv > 127) kv = 127;
        if (kv < -128) kv = -128;
        yb->qs[tid] = (int8_t)kv;
    }
    __syncthreads();
    if (tid < 256u / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

static inline void ds4_cuda_launch_q8_0_quantize(
        bool warp_reduce, dim3 grid, cudaStream_t stream,
        int8_t *xq, float *xscale, const float *x,
        uint64_t in_dim, uint64_t blocks) {
    if (warp_reduce) {
        quantize_q8_0_f32_kernel<true><<<grid, 32, 0, stream>>>(
            xq, xscale, x, in_dim, blocks);
    } else {
        quantize_q8_0_f32_kernel<false><<<grid, 32, 0, stream>>>(
            xq, xscale, x, in_dim, blocks);
    }
}

static inline void ds4_cuda_launch_q8_0_group_slice_quantize(
        bool warp_reduce, dim3 grid, cudaStream_t stream,
        int8_t *xq, float *xscale, const float *x,
        uint64_t group_dim, uint64_t blocks, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt) {
    if (warp_reduce) {
        quantize_q8_0_group_slice_rows_kernel<true><<<grid, 32, 0, stream>>>(
            xq, xscale, x, group_dim, blocks, n_groups_total, group0, group_cnt);
    } else {
        quantize_q8_0_group_slice_rows_kernel<false><<<grid, 32, 0, stream>>>(
            xq, xscale, x, group_dim, blocks, n_groups_total, group0, group_cnt);
    }
}

template<typename BlockQ8K>
static inline void ds4_cuda_launch_q8_dual_quantize(
        bool warp_reduce, dim3 grid, cudaStream_t stream,
        BlockQ8K *out, int8_t *q8_0, float *q8_0_scale, const float *x,
        uint32_t in_dim, uint32_t n_rows) {
    if (warp_reduce) {
        q8_K_q8_0_quantize_kernel<true><<<grid, 256, 0, stream>>>(
            out, q8_0, q8_0_scale, x, in_dim, n_rows);
    } else {
        q8_K_q8_0_quantize_kernel<false><<<grid, 256, 0, stream>>>(
            out, q8_0, q8_0_scale, x, in_dim, n_rows);
    }
}
