// DS4 ROCm Q8_0 matmul / grouped-output / HC-expand kernels.
//
// Included from ds4_cuda.cu in the same translation unit so kernel helpers stay
// private/static while we gradually split the custom ROCm backend into modules.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <rocwmma/rocwmma.hpp>
#endif

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static DS4_ROCM_UNUSED void matmul_q8_0_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint64_t blocks = (in_dim + 31) / 32;
    const unsigned char *wr = w + row * blocks * 34;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(xr[i0 + i]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = 0;
        for (uint64_t i = 0; i < bn; i++) {
            int q = (int)lrintf(xr[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dot += (int)qs[i] * q;
        }
        acc += __half2float(*scale_h) * d * (float)dot;
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

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
    a = warp_max_f32(a);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const float d = __shfl(a, 0, 32) / 127.0f;
#else
    const float d = __shfl_sync(FULL_WARP_MASK, a, 0, 32) / 127.0f;
#endif
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

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_q8_0_preq_rows_w32_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t rows_per_block,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34u;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb = xq + b * 32u;
        const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void shared_gate_up_swiglu_q8_0_pair_preq_warp8_kernel(
        float *gate,
        float *up,
        float *mid,
        const unsigned char *wg,
        const unsigned char *wu,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a,
        int store_gate_up,
        float clamp) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *gr = wg + row * blocks * 34u;
    const unsigned char *ur = wu + row * blocks * 34u;
    float g = 0.0f;
    float u = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb = xq + b * 32u;
        const float xs = xscale[b];
        const __half *gscale_h = (const __half *)(gr + b * 34u);
        const int8_t *gqs = (const int8_t *)(gr + b * 34u + 2u);
        const __half *uscale_h = (const __half *)(ur + b * 34u);
        const int8_t *uqs = (const int8_t *)(ur + b * 34u + 2u);
        const int gdot = dot_i8_block(gqs, xqb, bn, use_dp4a);
        const int udot = dot_i8_block(uqs, xqb, bn, use_dp4a);
        g += __half2float(*gscale_h) * xs * (float)gdot;
        u += __half2float(*uscale_h) * xs * (float)udot;
    }
    g = warp_sum_f32(g);
    u = warp_sum_f32(u);
    if (lane == 0u) {
        if (store_gate_up) {
            gate[row] = g;
            up[row] = u;
        }
        float sg = g;
        float su = u;
        if (clamp > 1.0e-6f) {
            sg = fminf(sg, clamp);
            su = fminf(fmaxf(su, -clamp), clamp);
        }
        mid[row] = (sg / (1.0f + expf(-sg))) * su;
    }
}

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__device__ static float q8_0_scale_scalar(const unsigned char *blk) {
    const uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    return __half2float(__ushort_as_half((unsigned short)bits));
}

__device__ static float q8_0_scale_broadcast_w32(const unsigned char *blk) {
    float d = 0.0f;
    if ((threadIdx.x & 31u) == 0u) d = q8_0_scale_scalar(blk);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __shfl(d, 0, 32);
#else
    return __shfl_sync(FULL_WARP_MASK, d, 0, 32);
#endif
}

__device__ static float q8_block_sum_w32(float v) {
    __shared__ float sh[32];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wid = tid >> 5u;
    const uint32_t nwarp = (blockDim.x + 31u) >> 5u;
    v = warp_sum_f32(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = (tid < nwarp) ? sh[lane] : 0.0f;
    if (wid == 0u) v = warp_sum_f32(v);
    if (tid == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__global__ static void matmul_q8_0_f32_small_block_w32_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint64_t out_dim,
        uint64_t row_bytes) {
    const uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out_dim) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t waves_per_block = blockDim.x >> 5u;
    const unsigned char *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = wave; b < n_blocks; b += waves_per_block) {
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * x[((uint64_t)b << 5u) + lane];
    }
    acc = q8_block_sum_w32(acc);
    if (tid == 0u) out[row] = acc;
}

__global__ static void matmul_q8_0_f32_warp8_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34u;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i = b * 32u + lane;
        if (i < in_dim) {
            const unsigned char *blk = wr + b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc += d * (float)q * x[i];
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_q8_0_f32_sharedx_warp_rows_w32_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint64_t out_dim,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t in_dim = n_blocks << 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(b << 5u) + lane];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

__global__ static void matmul_q8_0_f32_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34u;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i = b * 32u + lane;
        if (i < in_dim) {
            const unsigned char *blk = wr + b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc += d * (float)q * xr[i];
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

template <uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel(
        float *gate,
        float *up,
        float *mid,
        const unsigned char *wg,
        const unsigned char *wu,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        int store_gate_up) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (t0 >= n_tok) return;
    const bool row_valid = row < out_dim;
    const unsigned char *wgr = wg + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const unsigned char *wur = wu + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const uint32_t in_dim = n_blocks << 5u;
    float accg[TOK_TILE];
    float accu[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) {
        accg[u] = 0.0f;
        accu[u] = 0.0f;
    }

    for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const uint32_t b_count = ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
        for (uint32_t j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
            const uint32_t u = j / (BLOCKS_TILE * 32u);
            const uint32_t r = j - u * (BLOCKS_TILE * 32u);
            const uint32_t bb = r >> 5u;
            const uint32_t k = r & 31u;
            const uint32_t t = t0 + u;
            shx[j] = (t < n_tok && bb < b_count)
                ? x[(uint64_t)t * in_dim + ((uint64_t)(b0 + bb) << 5u) + k]
                : 0.0f;
        }
        __syncthreads();
        if (row_valid) {
            for (uint32_t bb = 0; bb < b_count; bb++) {
                const unsigned char *bg = wgr + (uint64_t)(b0 + bb) * 34u;
                const unsigned char *bu = wur + (uint64_t)(b0 + bb) * 34u;
                const float dg = q8_0_scale_broadcast_w32(bg);
                const float du = q8_0_scale_broadcast_w32(bu);
                const float wvg = dg * (float)((const int8_t *)(bg + 2u))[lane];
                const float wvu = du * (float)((const int8_t *)(bu + 2u))[lane];
#pragma unroll
                for (uint32_t u = 0; u < TOK_TILE; u++) {
                    const float xv = shx[(u * BLOCKS_TILE + bb) * 32u + lane];
                    accg[u] += wvg * xv;
                    accu[u] += wvu * xv;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) {
        accg[u] = warp_sum_f32(accg[u]);
        accu[u] = warp_sum_f32(accu[u]);
    }
    if (lane == 0u && row_valid) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) {
                const uint64_t off = (uint64_t)t * out_dim + row;
                const float g = accg[u];
                const float uv = accu[u];
                if (store_gate_up) {
                    gate[off] = g;
                    up[off] = uv;
                }
                mid[off] = (g / (1.0f + expf(-g))) * uv;
            }
        }
    }
}

template <uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (t0 >= n_tok) return;
    const bool row_valid = row < out_dim;
    const unsigned char *wr = w + (uint64_t)(row_valid ? row : 0u) * row_bytes;
    const uint32_t in_dim = n_blocks << 5u;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = 0.0f;

    for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const uint32_t b_count = ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
        for (uint32_t j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
            const uint32_t u = j / (BLOCKS_TILE * 32u);
            const uint32_t r = j - u * (BLOCKS_TILE * 32u);
            const uint32_t bb = r >> 5u;
            const uint32_t k = r & 31u;
            const uint32_t t = t0 + u;
            shx[j] = (t < n_tok && bb < b_count)
                ? x[(uint64_t)t * in_dim + ((uint64_t)(b0 + bb) << 5u) + k]
                : 0.0f;
        }
        __syncthreads();
        if (row_valid) {
            for (uint32_t bb = 0; bb < b_count; bb++) {
                const unsigned char *blk = wr + (uint64_t)(b0 + bb) * 34u;
                const float d = q8_0_scale_broadcast_w32(blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[lane];
                const float wv = d * (float)q;
#pragma unroll
                for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] += wv * shx[(u * BLOCKS_TILE + bb) * 32u + lane];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = warp_sum_f32(acc[u]);
    if (lane == 0u && row_valid) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tok) out[(uint64_t)t * out_dim + row] = acc[u];
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
typedef _Float16 __attribute__((ext_vector_type(16))) ds4_q8_half16_t;
typedef float    __attribute__((ext_vector_type(8)))  ds4_q8_float8_t;

/* Four-wave, 64x64 output-tile Q8_0 batched GEMM for large prefill chunks.
 * This is the hipfire/llama.cpp-style MMQ shape adapted to DS4's existing
 * F32 activation buffers: each block stages a 64-token x 32-K activation tile
 * into LDS as f16, while each wave owns 16 output rows and computes four
 * 16-token WMMA columns.  It is opt-in from host code because it only wins once
 * the token batch is large enough to amortize the bigger tile. */
__launch_bounds__(128, 2)
__global__ static void matmul_q8_0_f32_batch_wmma_4w_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes) {
    constexpr uint32_t M_TILE = 64u;
    constexpr uint32_t N_TILE = 64u;
    constexpr uint32_t K_TILE = 32u;
    constexpr uint32_t WARPS = 4u;
    constexpr uint32_t M_PER_WARP = M_TILE / WARPS;
    constexpr uint32_t N_TILES_PER_WARP = N_TILE / 16u;

    const uint32_t block_m = (uint32_t)blockIdx.x * M_TILE;
    const uint32_t block_n = (uint32_t)blockIdx.y * N_TILE;
    if (block_m >= out_dim || block_n >= n_tokens) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t warp_m = block_m + warp_id * M_PER_WARP;
    const uint32_t my_row = warp_m + lane16;
    const uint32_t safe_row = my_row < out_dim ? my_row : (out_dim - 1u);
    const unsigned char *row_base = w + (uint64_t)safe_row * row_bytes;
    const uint32_t n_blocks = in_dim >> 5u;

    ds4_q8_float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ds4_q8_float8_t acc1 = acc0;
    ds4_q8_float8_t acc2 = acc0;
    ds4_q8_float8_t acc3 = acc0;

    __shared__ _Float16 lds_x[N_TILE * K_TILE];

    for (uint32_t bi = 0; bi < n_blocks; bi++) {
        for (uint32_t j = tid; j < N_TILE * K_TILE; j += blockDim.x) {
            const uint32_t nt = j >> 5u;
            const uint32_t kk = j & 31u;
            const uint32_t tok = block_n + nt;
            float xv = 0.0f;
            if (tok < n_tokens) xv = x[(uint64_t)tok * in_dim + bi * 32u + kk];
            lds_x[j] = (_Float16)xv;
        }
        __syncthreads();

        const unsigned char *bp = row_base + (uint64_t)bi * 34u;
        _Float16 sc;
        {
            uint16_t s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const int8_t *w0 = (const int8_t *)(bp + 2u);
        const int8_t *w1 = (const int8_t *)(bp + 18u);
        ds4_q8_half16_t a0;
        ds4_q8_half16_t a1;
#pragma unroll
        for (uint32_t i = 0; i < 16u; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }

#pragma unroll
        for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
            const uint32_t nt = ntile * 16u + lane16;
            const _Float16 *xb = lds_x + nt * K_TILE;
            const ds4_q8_half16_t b0 = *(const ds4_q8_half16_t *)(xb);
            const ds4_q8_half16_t b1 = *(const ds4_q8_half16_t *)(xb + 16u);
            if (ntile == 0u) {
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc0);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc0);
            } else if (ntile == 1u) {
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc1);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc1);
            } else if (ntile == 2u) {
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc2);
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc2);
            } else {
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc3);
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc3);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
        const uint32_t tok = block_n + ntile * 16u + lane16;
        if (tok >= n_tokens) continue;
        ds4_q8_float8_t acc = ntile == 0u ? acc0 : (ntile == 1u ? acc1 : (ntile == 2u ? acc2 : acc3));
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t row = warp_m + 2u * j + (lane >> 4u);
            if (row < out_dim) out[(uint64_t)tok * out_dim + row] = acc[j];
        }
    }
}

template <int TILES_N=8, int BM=16, int BN=16, int BK=16>
__global__ static void matmul_q8_0_f32_batch_wmma_onthefly_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shB = shA + BM * BK;
    float *shC = reinterpret_cast<float *>(shB + TILES_N * BK * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t t0 = (uint32_t)blockIdx.y * BM;
    const uint32_t row0 = (uint32_t)blockIdx.x * TILES_N * BN;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < TILES_N) rocwmma::fill_fragment(acc, 0.0f);

    for (uint32_t k0 = 0; k0 < in_dim; k0 += BK) {
        for (uint32_t j = tid; j < BM * BK; j += blockDim.x) {
            const uint32_t m = j / BK;
            const uint32_t kk = j - m * BK;
            const uint32_t t = t0 + m;
            shA[j] = (t < n_tokens && k0 + kk < in_dim)
                ? __float2half(x[(uint64_t)t * in_dim + k0 + kk])
                : __float2half(0.0f);
        }
        for (uint32_t j = tid; j < TILES_N * BK * BN; j += blockDim.x) {
            const uint32_t tn = j / (BK * BN);
            const uint32_t rem = j - tn * BK * BN;
            const uint32_t kk = rem / BN;
            const uint32_t nn = rem - kk * BN;
            const uint32_t row = row0 + tn * BN + nn;
            const uint32_t k = k0 + kk;
            if (row < out_dim && k < in_dim) {
                const unsigned char *blk = w + (uint64_t)row * row_bytes + (uint64_t)(k >> 5u) * 34u;
                const float d = __half2float(*(const half *)blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[k & 31u];
                shB[j] = __float2half(d * (float)q);
            } else {
                shB[j] = __float2half(0.0f);
            }
        }
        __syncthreads();
        if (wave < TILES_N) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::load_matrix_sync(b, shB + wave * BK * BN, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    if (wave < TILES_N) rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = tid; j < TILES_N * BM * BN; j += blockDim.x) {
        const uint32_t tn = j / (BM * BN);
        const uint32_t rem = j - tn * BM * BN;
        const uint32_t m = rem / BN;
        const uint32_t nn = rem - m * BN;
        const uint32_t t = t0 + m;
        const uint32_t row = row0 + tn * BN + nn;
        if (t < n_tokens && row < out_dim) out[(uint64_t)t * out_dim + row] = shC[j];
    }
}
#endif

__global__ static void matmul_q8_0_pair_f32_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34u : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34u : NULL;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i = b * 32u + lane;
        if (i < in_dim) {
            if (wr0) {
                const unsigned char *blk = wr0 + b * 34u;
                const float d = q8_0_scale_broadcast_w32(blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[lane];
                acc0 += d * (float)q * x[i];
            }
            if (wr1) {
                const unsigned char *blk = wr1 + b * 34u;
                const float d = q8_0_scale_broadcast_w32(blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[lane];
                acc1 += d * (float)q * x[i];
            }
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void matmul_q8_0_pair_f32_sharedx_warp_rows_w32_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const float *x,
        uint32_t n_blocks,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t in_dim = n_blocks << 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out0_dim && row >= out1_dim) return;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * row_bytes : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * row_bytes : NULL;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const float xv = shx[(b << 5u) + lane];
        if (wr0) {
            const unsigned char *blk = wr0 + (uint64_t)b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc0 += d * (float)q * xv;
        }
        if (wr1) {
            const unsigned char *blk = wr1 + (uint64_t)b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc1 += d * (float)q * xv;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0u) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void shared_gate_up_swiglu_q8_0_rows_w32_kernel(
        float *gate,
        float *up,
        float *mid,
        const unsigned char *wg,
        const unsigned char *wu,
        const float *x,
        uint32_t n_blocks,
        uint64_t out_dim,
        uint64_t row_bytes,
        int store_gate_up,
        float clamp) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *row_g = wg + row * row_bytes;
    const unsigned char *row_u = wu + row * row_bytes;
    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *bg = row_g + (uint64_t)b * 34u;
        const unsigned char *bu = row_u + (uint64_t)b * 34u;
        const float dg = q8_0_scale_broadcast_w32(bg);
        const float du = q8_0_scale_broadcast_w32(bu);
        const int8_t qg = ((const int8_t *)(bg + 2u))[lane];
        const int8_t qu = ((const int8_t *)(bu + 2u))[lane];
        const float xv = x[((uint64_t)b << 5) + lane];
        acc_g += dg * (float)qg * xv;
        acc_u += du * (float)qu * xv;
    }
    const float g = warp_sum_f32(acc_g);
    const float u = warp_sum_f32(acc_u);
    if (lane == 0u) {
        if (store_gate_up) {
            gate[row] = g;
            up[row] = u;
        }
        float sg = g;
        float su = u;
        if (clamp > 1.0e-6f) {
            sg = fminf(sg, clamp);
            su = fminf(fmaxf(su, -clamp), clamp);
        }
        mid[row] = (sg / (1.0f + expf(-sg))) * su;
    }
}

__global__ static void matmul_q8_0_hc_expand_preq_rows_w32_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        uint32_t rows_per_block,
        int has_add,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34u;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb = xq + b * 32u;
        const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                hc_acc += residual_hc[(uint64_t)src_hc * n_embd + d] * comb[(uint64_t)src_hc * n_hc + dst_hc];
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_hc_expand_f32_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34u;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i = b * 32u + lane;
        if (i < in_dim) {
            const unsigned char *blk = wr + b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc += d * (float)q * x[i];
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_hc_expand_f32_sharedx_warp_rows_w32_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint64_t out_dim,
        uint64_t row_bytes,
        uint32_t n_embd,
        uint32_t n_hc,
        int has_add) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t in_dim = n_blocks << 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(b << 5u) + lane];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__device__ static float warp_sum_f32_oldhip_w32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_down(v, offset, 32);
#else
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
    }
    return v;
}

__device__ static float q8_0_scale_broadcast_oldhip_w32(const unsigned char *blk) {
    float d = 0.0f;
    if ((threadIdx.x & 31u) == 0u) d = q8_0_scale_scalar(blk);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __shfl(d, 0, 32);
#else
    return __shfl_sync(FULL_WARP_MASK, d, 0, 32);
#endif
}

__global__ static void matmul_q8_0_hc_partial16_w32_kernel(
        float *partial,
        const unsigned char *w,
        const float *x,
        uint32_t out_dim,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t b0 = split << 4;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *wr = w + (uint64_t)row * row_bytes;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const uint32_t b = b0 + bb;
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = q8_0_scale_broadcast_oldhip_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(bb << 5) + lane];
    }
    acc = warp_sum_f32_oldhip_w32(acc);
    if (lane == 0u) partial[(uint64_t)split * out_dim + row] = acc;
}

__global__ static void matmul_q8_0_hc_partial_w32_kernel(
        float *partial,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint64_t row_bytes,
        uint32_t n_splits) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t chunk = (n_blocks + n_splits - 1u) / n_splits;
    const uint32_t b0 = split * chunk;
    const uint32_t b1 = min(n_blocks, b0 + chunk);
    const uint32_t chunk_blocks = b1 > b0 ? b1 - b0 : 0u;
    for (uint32_t i = tid; i < (chunk_blocks << 5); i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t row = blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const unsigned char *wr = w + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t bb = 0; bb < chunk_blocks; bb++) {
        const uint32_t b = b0 + bb;
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = q8_0_scale_broadcast_oldhip_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(bb << 5) + lane];
    }
    acc = warp_sum_f32_oldhip_w32(acc);
    if (lane == 0u) partial[(uint64_t)split * out_dim + row] = acc;
}

__global__ static void hc_expand_partial_kernel(
        float *out_hc,
        float *block_out,
        const float *partial,
        const float *residual_hc,
        const float *split,
        uint32_t out_dim,
        uint32_t n_hc,
        uint32_t n_splits,
        int store_block_out) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_splits; s++) acc += partial[(uint64_t)s * out_dim + row];
    if (store_block_out) block_out[row] = acc;
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float v = acc * post[dst];
        for (uint32_t src = 0; src < n_hc; src++) {
            v += comb[dst + (uint64_t)src * n_hc] * residual_hc[(uint64_t)src * out_dim + row];
        }
        out_hc[(uint64_t)dst * out_dim + row] = v;
    }
}

__global__ static void hc_expand_add_partial_kernel(
        float *out_hc,
        float *block_out,
        const float *partial,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t out_dim,
        uint32_t n_hc,
        uint32_t n_splits,
        int store_block_out) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_splits; s++) acc += partial[(uint64_t)s * out_dim + row];
    if (store_block_out) block_out[row] = acc;
    const float block = acc + block_add[row];
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float v = block * post[dst];
        for (uint32_t src = 0; src < n_hc; src++) {
            v += comb[dst + (uint64_t)src * n_hc] * residual_hc[(uint64_t)src * out_dim + row];
        }
        out_hc[(uint64_t)dst * out_dim + row] = v;
    }
}

__global__ static void hc_expand_add_partial4_kernel(
        float *out_hc,
        float *block_out,
        const float *partial,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t out_dim,
        uint32_t n_hc,
        int store_block_out) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < 4u; s++) acc += partial[(uint64_t)s * out_dim + row];
    if (store_block_out) block_out[row] = acc;
    const float block = acc + block_add[row];
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float v = block * post[dst];
        for (uint32_t src = 0; src < n_hc; src++) {
            v += comb[dst + (uint64_t)src * n_hc] * residual_hc[(uint64_t)src * out_dim + row];
        }
        out_hc[(uint64_t)dst * out_dim + row] = v;
    }
}

__global__ static void hc_expand_partial16_kernel(
        float *out_hc,
        float *block_out,
        const float *partial,
        const float *residual_hc,
        const float *split,
        uint32_t out_dim,
        uint32_t n_hc,
        int store_block_out) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < 16u; s++) acc += partial[(uint64_t)s * out_dim + row];
    if (store_block_out) block_out[row] = acc;
    const float *post = split + n_hc;
    const float *comb = split + 2u * n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float v = acc * post[dst];
        for (uint32_t src = 0; src < n_hc; src++) {
            v += comb[dst + (uint64_t)src * n_hc] * residual_hc[(uint64_t)src * out_dim + row];
        }
        out_hc[(uint64_t)dst * out_dim + row] = v;
    }
}

__global__ static void grouped_q8_0_a_f32_warp8_kernel(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim) return;
    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34u;
    const float *x = heads + group * group_dim;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i = b * 32u + lane;
        if (i < group_dim) {
            const unsigned char *blk = wr + b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc += d * (float)q * x[i];
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[row] = acc;
}

__global__ static void grouped_q8_0_a_f32_sharedx_rows_w32_2row_kernel(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint64_t rank,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = (blockDim.x >> 5u) << 1u;
    const uint32_t group_dim = n_blocks << 5u;
    const uint64_t total = (uint64_t)n_groups * rank;
    const uint64_t base_idx = (uint64_t)blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint64_t base_gtmp = base_idx / rank;
    const uint32_t g = (uint32_t)(base_gtmp % n_groups);
    const float *x = heads + (uint64_t)g * group_dim;
    for (uint32_t i = tid; i < group_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t idx0 = base_idx + ((uint64_t)wave << 1u);
    if (idx0 >= total) return;
    const uint64_t row0 = idx0 % rank;
    const uint64_t idx1 = idx0 + 1u;
    const uint64_t tensor_row0 = (uint64_t)g * rank + row0;
    const unsigned char *wr0 = w + tensor_row0 * row_bytes;
    const unsigned char *wr1 = wr0 + row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const float xv = shx[(b << 5u) + lane];
        const unsigned char *blk0 = wr0 + (uint64_t)b * 34u;
        const float d0 = q8_0_scale_broadcast_w32(blk0);
        const int8_t q0 = ((const int8_t *)(blk0 + 2u))[lane];
        acc0 += d0 * (float)q0 * xv;
        if (row0 + 1u < rank && idx1 < total) {
            const unsigned char *blk1 = wr1 + (uint64_t)b * 34u;
            const float d1 = q8_0_scale_broadcast_w32(blk1);
            const int8_t q1 = ((const int8_t *)(blk1 + 2u))[lane];
            acc1 += d1 * (float)q1 * xv;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0u) {
        low[idx0] = acc0;
        if (row0 + 1u < rank && idx1 < total) low[idx1] = acc1;
    }
}

__global__ static void grouped_q8_0_a_partial16_w32_kernel(
        float *partial,
        const unsigned char *w,
        const float *heads,
        uint32_t n_groups,
        uint32_t rank,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t rows_per_block = blockDim.x >> 5;
    const uint32_t split = blockIdx.y;
    const uint32_t total = n_groups * rank;
    const uint32_t base_idx = blockIdx.x * rows_per_block;
    if (base_idx >= total) return;
    const uint32_t g = (base_idx / rank) % n_groups;
    const uint32_t b0 = split << 4;
    const float *x = heads + (uint64_t)g * 4096u;
    for (uint32_t i = tid; i < 512u; i += blockDim.x) shx[i] = x[((uint64_t)b0 << 5) + i];
    __syncthreads();

    const uint32_t idx = base_idx + wave;
    if (idx >= total) return;
    const uint32_t row = idx % rank;
    const unsigned char *wr = w + (uint64_t)((uint64_t)g * rank + row) * row_bytes;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t bb = 0; bb < 16u; bb++) {
        const uint32_t b = b0 + bb;
        const unsigned char *blk = wr + (uint64_t)b * 34u;
        const float d = q8_0_scale_broadcast_oldhip_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * shx[(bb << 5) + lane];
    }
    acc = warp_sum_f32_oldhip_w32(acc);
    if (lane == 0u) partial[(uint64_t)split * total + idx] = acc;
}

__global__ static void q8_partial_sum8_kernel(float *out, const float *partial, uint32_t out_dim) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < 8u; s++) acc += partial[(uint64_t)s * out_dim + row];
    out[row] = acc;
}

__global__ static void grouped_q8_0_a_f32_batch_warp8_kernel(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;
    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34u;
    const float *x = heads + (tok * (uint64_t)n_groups + group) * group_dim;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i = b * 32u + lane;
        if (i < group_dim) {
            const unsigned char *blk = wr + b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc += d * (float)q * x[i];
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

template <uint32_t TOK_TILE, uint32_t BLOCKS_TILE>
__global__ static void grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
    const uint32_t g = blockIdx.x / row_blocks;
    const uint32_t row0 = (blockIdx.x - g * row_blocks) * rows_per_block + wave;
    const uint32_t t0 = blockIdx.y * TOK_TILE;
    if (g >= n_groups || t0 >= n_tokens) return;
    const uint32_t group_dim = n_blocks << 5u;
    const bool row_valid = row0 < rank;
    const unsigned char *wr = w + ((uint64_t)g * rank + (row_valid ? row0 : 0u)) * row_bytes;
    float acc[TOK_TILE];
#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = 0.0f;

    for (uint32_t b0 = 0; b0 < n_blocks; b0 += BLOCKS_TILE) {
        const uint32_t b_count = ((b0 + BLOCKS_TILE) <= n_blocks) ? BLOCKS_TILE : (n_blocks - b0);
        for (uint32_t j = tid; j < TOK_TILE * BLOCKS_TILE * 32u; j += blockDim.x) {
            const uint32_t u = j / (BLOCKS_TILE * 32u);
            const uint32_t r = j - u * (BLOCKS_TILE * 32u);
            const uint32_t bb = r >> 5u;
            const uint32_t k = r & 31u;
            const uint32_t t = t0 + u;
            const uint64_t xoff = ((uint64_t)t * n_groups + g) * group_dim + ((uint64_t)(b0 + bb) << 5u) + k;
            shx[j] = (t < n_tokens && bb < b_count) ? heads[xoff] : 0.0f;
        }
        __syncthreads();
        if (row_valid) {
            for (uint32_t bb = 0; bb < b_count; bb++) {
                const unsigned char *blk = wr + (uint64_t)(b0 + bb) * 34u;
                const float d = q8_0_scale_broadcast_w32(blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[lane];
                const float wv = d * (float)q;
#pragma unroll
                for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] += wv * shx[(u * BLOCKS_TILE + bb) * 32u + lane];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t u = 0; u < TOK_TILE; u++) acc[u] = warp_sum_f32(acc[u]);
    if (lane == 0u && row_valid) {
#pragma unroll
        for (uint32_t u = 0; u < TOK_TILE; u++) {
            const uint32_t t = t0 + u;
            if (t < n_tokens) low[((uint64_t)t * n_groups + g) * rank + row0] = acc[u];
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
template <int TILES_N=8, int BM=16, int BN=16, int BK=16>
__global__ static void grouped_q8_0_a_f32_batch_wmma_onthefly_kernel(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim,
        uint32_t rank,
        uint64_t row_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shB = shA + BM * BK;
    float *shC = reinterpret_cast<float *>(shB + TILES_N * BK * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t row_tiles_per_group = (rank + TILES_N * BN - 1u) / (TILES_N * BN);
    const uint32_t g = (uint32_t)blockIdx.x / row_tiles_per_group;
    const uint32_t row_tile = (uint32_t)blockIdx.x - g * row_tiles_per_group;
    const uint32_t row0 = row_tile * TILES_N * BN;
    const uint32_t t0 = (uint32_t)blockIdx.y * BM;
    if (g >= n_groups) return;

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < TILES_N) rocwmma::fill_fragment(acc, 0.0f);

    for (uint32_t k0 = 0; k0 < group_dim; k0 += BK) {
        for (uint32_t j = tid; j < BM * BK; j += blockDim.x) {
            const uint32_t m = j / BK;
            const uint32_t kk = j - m * BK;
            const uint32_t t = t0 + m;
            const uint32_t k = k0 + kk;
            shA[j] = (t < n_tokens && k < group_dim)
                ? __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + k])
                : __float2half(0.0f);
        }
        for (uint32_t j = tid; j < TILES_N * BK * BN; j += blockDim.x) {
            const uint32_t tn = j / (BK * BN);
            const uint32_t rem = j - tn * BK * BN;
            const uint32_t kk = rem / BN;
            const uint32_t nn = rem - kk * BN;
            const uint32_t row = row0 + tn * BN + nn;
            const uint32_t k = k0 + kk;
            if (row < rank && k < group_dim) {
                const unsigned char *blk = w + ((uint64_t)g * rank + row) * row_bytes + (uint64_t)(k >> 5u) * 34u;
                const float d = __half2float(*(const half *)blk);
                const int8_t q = ((const int8_t *)(blk + 2u))[k & 31u];
                shB[j] = __float2half(d * (float)q);
            } else {
                shB[j] = __float2half(0.0f);
            }
        }
        __syncthreads();
        if (wave < TILES_N) {
            rocwmma::load_matrix_sync(a, shA, BK);
            rocwmma::load_matrix_sync(b, shB + wave * BK * BN, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    if (wave < TILES_N) rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = tid; j < TILES_N * BM * BN; j += blockDim.x) {
        const uint32_t tn = j / (BM * BN);
        const uint32_t rem = j - tn * BM * BN;
        const uint32_t m = rem / BN;
        const uint32_t nn = rem - m * BN;
        const uint32_t t = t0 + m;
        const uint32_t row = row0 + tn * BN + nn;
        if (t < n_tokens && row < rank) low[((uint64_t)t * n_groups + g) * rank + row] = shC[j];
    }
}
#endif

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const float scale = q8_0_scale_scalar(blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

__global__ static void dequant_q8_0_to_f16_transpose_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    const uint64_t row = gid / in_dim;
    const uint64_t i = gid - row * in_dim;
    const uint64_t b = i / 32u;
    const uint64_t j = i - b * 32u;
    const unsigned char *blk = w + (row * blocks + b) * 34u;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2u + j);
    out[i * out_dim + row] = __hmul(scale, __float2half((float)q));
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}
