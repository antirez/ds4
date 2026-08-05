/* proto_m2_hc.cu — M2-Inc1 prototype: fuse the per-layer HC stage chain
 *
 *   baseline (production, 4 launches):
 *     rms_norm_plain(flat, cur, 16384)            <<<1,256>>>
 *     matmul_f16_splitk(partial, W24x16384, flat) <<<(24,32),256>>>
 *     matmul_f16_splitk_combine(mix, partial)     <<<1,256>>>
 *     hc_split_weighted_sum_norm_fused(...)       <<<1,256>>>
 *
 *   V1 fused-2 (2 launches): dots on RAW x + sumsq partials in one launch
 *     (linearity: dot(w, x*s) == s*dot(w, x)); finish kernel does
 *     scale+combine+sinkhorn+weighted-sum-norm in one block.
 *
 *   V2 coop-1 (1 cooperative launch, 96 blocks, 4 grid.sync()).
 *
 * Parity vs double-precision host reference for all variants (incl. the
 * production baseline — the noise yardstick). Timing: 43 rotating weight
 * sets (L2-realistic), CUDA events. Graph capture/replay correctness test
 * for V1 and V2 (V2 probes whether cooperative launch is capturable).
 *
 * Build: nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 proto_m2_hc.cu -o proto_m2_hc
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define HC_DIM   16384u   /* n_hc * n_embd */
#define MIX_HC   24u
#define N_EMBD   4096u
#define N_HC     4u
#define KSPLIT   32u
#define SINK_IT  20u
#define RMS_EPS  1.0e-6f
#define HC_EPS   1.0e-6f
#define N_LAYERS 43
#define REPS     200

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

/* ---------- production helpers (copied verbatim semantics) ---------- */
__device__ __forceinline__ static float warp_sum_f32(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}
__device__ static float block_reduce_sum_256(float v) {
    v = warp_sum_f32(v);
    __shared__ float s[8];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (lane == 0u) s[warp] = v;
    __syncthreads();
    float r = 0.0f;
    if (warp == 0u) {
        r = (lane < 8u) ? s[lane] : 0.0f;
        r += __shfl_down_sync(0xffffffffu, r, 4);
        r += __shfl_down_sync(0xffffffffu, r, 2);
        r += __shfl_down_sync(0xffffffffu, r, 1);
    }
    return r;
}

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

/* ---------- V0: production baseline kernels ---------- */
__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
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
        orow[i] = xr[i] * scale;
    }
}

__global__ static void matmul_f16_splitk_kernel(
        float *partial, const __half *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint32_t ksplit, int vec_ok) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint32_t kseg = blockIdx.y;
    if (row >= out_dim) return;
    uint64_t seg = (in_dim + (uint64_t)ksplit - 1u) / (uint64_t)ksplit;
    seg = (seg + 7u) & ~(uint64_t)7u;
    const uint64_t k0 = (uint64_t)kseg * seg;
    if (k0 >= in_dim) {
        if (threadIdx.x == 0u) partial[row * (uint64_t)ksplit + (uint64_t)kseg] = 0.0f;
        return;
    }
    uint64_t k1 = k0 + seg;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const uint32_t tid = threadIdx.x;
    const uint32_t nt = blockDim.x;
    float sum = 0.0f;
    if (vec_ok) {
        const uint64_t vend = k0 + ((k1 - k0) & ~(uint64_t)7u);
        for (uint64_t i = k0 + (uint64_t)tid * 8u; i < vend; i += (uint64_t)nt * 8u) {
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(x + i);
            const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                 + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        for (uint64_t i = vend + tid; i < k1; i += nt) sum += __half2float(wr[i]) * x[i];
    } else {
        for (uint64_t i = k0 + tid; i < k1; i += nt) sum += __half2float(wr[i]) * x[i];
    }
    sum = block_reduce_sum_256(sum);
    if (threadIdx.x == 0u) partial[row * (uint64_t)ksplit + (uint64_t)kseg] = sum;
}

__global__ static void matmul_f16_splitk_combine_kernel(
        float *out, const float *partial, uint64_t out_dim, uint32_t ksplit) {
    const uint64_t row = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (row >= out_dim) return;
    const float *p = partial + row * (uint64_t)ksplit;
    float s = 0.0f;
    for (uint32_t k = 0u; k < ksplit; k++) s += p[k];
    out[row] = s;
}

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out, float *norm_out, float *split,
        const float *mix, const float *residual_hc,
        const float *scale, const float *base, const float *norm_w,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_rows,
        uint32_t sinkhorn_iters, float epsv, float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }
    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

/* ---------- V1: fused-2 ---------- */
/* K1: grid (MIX_HC+1, KSPLIT). rows 0..23 = dot partials on RAW x (identical
 * loop structure to production splitk, just unnormed input). row 24 = sumsq
 * partials of x over the same segment tiling. */
__global__ static void hc_stage_dots_kernel(
        float *partial,          /* [MIX_HC * KSPLIT] */
        float *sumsq_partial,    /* [KSPLIT] */
        const __half *w, const float *x,
        uint64_t in_dim, uint32_t ksplit) {
    const uint32_t row = blockIdx.x;
    const uint32_t kseg = blockIdx.y;
    uint64_t seg = (in_dim + (uint64_t)ksplit - 1u) / (uint64_t)ksplit;
    seg = (seg + 7u) & ~(uint64_t)7u;
    const uint64_t k0 = (uint64_t)kseg * seg;
    uint64_t k1 = k0 + seg;
    if (k1 > in_dim) k1 = in_dim;
    const uint32_t tid = threadIdx.x;
    const uint32_t nt = blockDim.x;
    if (row == MIX_HC) { /* sumsq lane */
        float ss = 0.0f;
        if (k0 < in_dim) {
            for (uint64_t i = k0 + tid; i < k1; i += nt) { float v = x[i]; ss += v * v; }
        }
        ss = block_reduce_sum_256(ss);
        if (tid == 0u) sumsq_partial[kseg] = ss;
        return;
    }
    if (k0 >= in_dim) {
        if (tid == 0u) partial[(uint64_t)row * ksplit + kseg] = 0.0f;
        return;
    }
    const __half *wr = w + (uint64_t)row * in_dim;
    float sum = 0.0f;
    const uint64_t vend = k0 + ((k1 - k0) & ~(uint64_t)7u);
    for (uint64_t i = k0 + (uint64_t)tid * 8u; i < vend; i += (uint64_t)nt * 8u) {
        const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
        const float4 xa = *reinterpret_cast<const float4 *>(x + i);
        const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
        const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
        const float2 w0 = __half22float2(h[0]);
        const float2 w1 = __half22float2(h[1]);
        const float2 w2 = __half22float2(h[2]);
        const float2 w3 = __half22float2(h[3]);
        sum += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
             + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
    }
    for (uint64_t i = vend + tid; i < k1; i += nt) sum += __half2float(wr[i]) * x[i];
    sum = block_reduce_sum_256(sum);
    if (tid == 0u) partial[(uint64_t)row * ksplit + kseg] = sum;
}

/* K2: one block. scale from sumsq partials; mix = scale * sum(partials)
 * (fixed kseg-ascending order); sinkhorn; weighted-sum-norm. */
__global__ static void hc_stage_finish_kernel(
        float *out, float *norm_out, float *split, float *mix_out,
        const float *partial, const float *sumsq_partial,
        const float *residual_hc,
        const float *scale3, const float *base, const float *norm_w,
        uint32_t n_embd, uint32_t in_dim, uint32_t ksplit,
        uint32_t sinkhorn_iters, float epsv, float rms_eps, float norm_eps) {
    const uint32_t d = threadIdx.x;
    __shared__ float rms_scale_sh;
    __shared__ float sp_sh[MIX_HC];
    if (d == 0u) {
        float ss = 0.0f;
        for (uint32_t k = 0u; k < ksplit; k++) ss += sumsq_partial[k];
        rms_scale_sh = rsqrtf(ss / (float)in_dim + rms_eps);
    }
    __syncthreads();
    if (d < MIX_HC) {
        const float *p = partial + (uint64_t)d * ksplit;
        float s = 0.0f;
        for (uint32_t k = 0u; k < ksplit; k++) s += p[k];
        mix_out[d] = s * rms_scale_sh;
    }
    __syncthreads();
    if (d == 0u) {
        hc4_split_one(sp_sh, mix_out, scale3, base, sinkhorn_iters, epsv);
        for (uint32_t i = 0; i < MIX_HC; i++) split[i] = sp_sh[i];
    }
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)h * n_embd + col] * sp_sh[h];
        }
        out[col] = acc;
        sum += acc * acc;
    }
    __shared__ float partial_sh[256];
    partial_sh[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial_sh[d] += partial_sh[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial_sh[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        norm_out[col] = out[col] * norm_scale * norm_w[col];
    }
}

/* ---------- V2: coop-1 ---------- */
/* 96 blocks x 256 threads, 4 grid syncs.
 * P1: blocks 0..95 -> dot partials (row = b>>2, subseg = b&3, 4 subsegs of
 *     4096 elems); blocks 0..31 also sumsq over 512-elem segs.
 * P2: block 0 -> rms scale, mix, sinkhorn -> globals.
 * P3: all blocks -> out cols + sumsq2 block partials.
 * P4: block 0 -> norm scale -> global.
 * P5: all blocks -> norm_out. */
#ifndef COOP_BLOCKS
#define COOP_BLOCKS 96u
#endif
#ifndef SUBSEG
#define SUBSEG (COOP_BLOCKS / 24u)   /* dot subsegments per row */
#endif
__global__ static void hc_stage_coop_kernel(
        float *out, float *norm_out, float *split, float *mix_out,
        float *scratch,          /* [MIX_HC*4 dots | 32 sumsq | COOP_BLOCKS sumsq2 | 2 scales] */
        const __half *w, const float *x, const float *residual_hc,
        const float *scale3, const float *base, const float *norm_w,
        uint32_t n_embd, uint32_t in_dim,
        uint32_t sinkhorn_iters, float epsv, float rms_eps, float norm_eps) {
    cg::grid_group grid = cg::this_grid();
    float *dots   = scratch;                    /* 24*SUBSEG */
    float *ssq    = scratch + MIX_HC * SUBSEG;  /* 32 */
    float *ssq2   = ssq + 32u;                  /* COOP_BLOCKS */
    float *scales = ssq2 + COOP_BLOCKS;         /* 2 */
    const uint32_t b = blockIdx.x;
    const uint32_t d = threadIdx.x;
    const uint32_t nt = blockDim.x;

    /* P1 dots: 24 rows x SUBSEG subsegments */
    {
        const uint32_t row = b / SUBSEG;
        const uint32_t sub = b % SUBSEG;
        const uint64_t seg_len = in_dim / SUBSEG;
        if (row < MIX_HC) {
            const uint64_t k0 = (uint64_t)sub * seg_len;
            const uint64_t k1 = k0 + seg_len;
            const __half *wr = w + (uint64_t)row * in_dim;
            float sum = 0.0f;
            for (uint64_t i = k0 + (uint64_t)d * 8u; i < k1; i += (uint64_t)nt * 8u) {
                const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
                const float4 xa = *reinterpret_cast<const float4 *>(x + i);
                const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
                const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
                const float2 w0 = __half22float2(h[0]);
                const float2 w1 = __half22float2(h[1]);
                const float2 w2 = __half22float2(h[2]);
                const float2 w3 = __half22float2(h[3]);
                sum += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                     + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
            }
            sum = block_reduce_sum_256(sum);
            if (d == 0u) dots[(uint64_t)row * SUBSEG + sub] = sum;
        }
    }
    /* P1 sumsq: blocks 0..31 over 512-elem segments */
    if (b < 32u) {
        const uint64_t k0 = (uint64_t)b * 512u;
        float ss = 0.0f;
        for (uint64_t i = k0 + d; i < k0 + 512u; i += nt) { float v = x[i]; ss += v * v; }
        ss = block_reduce_sum_256(ss);
        if (d == 0u) ssq[b] = ss;
    }
    grid.sync();

    /* P2: block 0 computes scale, mix, sinkhorn */
    if (b == 0u) {
        __shared__ float rms_scale_sh;
        __shared__ float sp_sh[MIX_HC];
        if (d == 0u) {
            float ss = 0.0f;
            for (uint32_t k = 0u; k < 32u; k++) ss += ssq[k];
            rms_scale_sh = rsqrtf(ss / (float)in_dim + rms_eps);
            scales[0] = rms_scale_sh;
        }
        __syncthreads();
        if (d < MIX_HC) {
            float s = 0.0f;
            for (uint32_t k = 0u; k < SUBSEG; k++) s += dots[(uint64_t)d * SUBSEG + k];
            mix_out[d] = s * rms_scale_sh;
        }
        __syncthreads();
        if (d == 0u) {
            hc4_split_one(sp_sh, mix_out, scale3, base, sinkhorn_iters, epsv);
            for (uint32_t i = 0; i < MIX_HC; i++) split[i] = sp_sh[i];
        }
    }
    grid.sync();

    /* P3: all blocks, out cols + sumsq2 partials */
    {
        float s0 = split[0], s1 = split[1], s2 = split[2], s3 = split[3];
        float sum = 0.0f;
        for (uint32_t col = b * nt + d; col < n_embd; col += COOP_BLOCKS * nt) {
            float acc = residual_hc[col] * s0
                      + residual_hc[n_embd + col] * s1
                      + residual_hc[2u * n_embd + col] * s2
                      + residual_hc[3u * n_embd + col] * s3;
            out[col] = acc;
            sum += acc * acc;
        }
        sum = block_reduce_sum_256(sum);
        if (d == 0u) ssq2[b] = sum;
    }
    grid.sync();

    /* P4: block 0 total norm scale */
    if (b == 0u && d == 0u) {
        float ss = 0.0f;
        for (uint32_t k = 0u; k < COOP_BLOCKS; k++) ss += ssq2[k];
        scales[1] = rsqrtf(ss / (float)n_embd + norm_eps);
    }
    grid.sync();

    /* P5: norm_out */
    {
        const float ns = scales[1];
        for (uint32_t col = b * nt + d; col < n_embd; col += COOP_BLOCKS * nt) {
            norm_out[col] = out[col] * ns * norm_w[col];
        }
    }
}

/* ---------- V3: coop, 3 syncs + warp-parallel sinkhorn ---------- */
/* Differences vs V2:
 *  - P2 sinkhorn runs on 16 lanes of warp 0 (lane = r*4+c): row sums via
 *    shfl_xor over bits 0..1, col sums over bits 2..3; pre/post sigmoids on
 *    lanes 0..7 in parallel.
 *  - P4 is folded into P5: every block redundantly reduces the ssq2 partials
 *    (COOP_BLOCKS floats from L2) so the P4 grid.sync disappears. */
__device__ __forceinline__ static float quad_sum_rows(float v) { /* sum over c (bits 0..1) */
    v += __shfl_xor_sync(0xffffu, v, 1, 16);
    v += __shfl_xor_sync(0xffffu, v, 2, 16);
    return v;
}
__device__ __forceinline__ static float quad_sum_cols(float v) { /* sum over r (bits 2..3) */
    v += __shfl_xor_sync(0xffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffu, v, 8, 16);
    return v;
}
__device__ static void hc4_split_warp16(float *out, const float *mix, const float *scale,
                                        const float *base, uint32_t sinkhorn_iters, float epsv,
                                        uint32_t lane) {
    /* lanes 0..15 active; lane = r*4+c */
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (lane < 8u) {
        float z = mix[lane] * (lane < 4u ? pre_scale : post_scale) + base[lane];
        out[lane] = lane < 4u ? 1.0f / (1.0f + expf(-z)) + epsv
                              : 2.0f / (1.0f + expf(-z));
    }
    float v = mix[8u + lane] * comb_scale + base[8u + lane];
    /* row softmax */
    float m = v;
    m = fmaxf(m, __shfl_xor_sync(0xffffu, m, 1, 16));
    m = fmaxf(m, __shfl_xor_sync(0xffffu, m, 2, 16));
    float c = expf(v - m);
    float s = quad_sum_rows(c);
    c = c / s + epsv;
    /* first col normalize */
    s = quad_sum_cols(c) + epsv;
    c /= s;
    for (uint32_t it = 1; it < sinkhorn_iters; it++) {
        s = quad_sum_rows(c) + epsv;
        c /= s;
        s = quad_sum_cols(c) + epsv;
        c /= s;
    }
    out[8u + lane] = c;
}

__global__ static void hc_stage_coop3_kernel(
        float *out, float *norm_out, float *split, float *mix_out,
        float *scratch,
        const __half *w, const float *x, const float *residual_hc,
        const float *scale3, const float *base, const float *norm_w,
        uint32_t n_embd, uint32_t in_dim,
        uint32_t sinkhorn_iters, float epsv, float rms_eps, float norm_eps) {
    cg::grid_group grid = cg::this_grid();
    float *dots   = scratch;
    float *ssq    = scratch + MIX_HC * SUBSEG;
    float *ssq2   = ssq + 32u;
    const uint32_t b = blockIdx.x;
    const uint32_t d = threadIdx.x;
    const uint32_t nt = blockDim.x;

    /* P1 dots + sumsq (same as V2) */
    {
        const uint32_t row = b / SUBSEG;
        const uint32_t sub = b % SUBSEG;
        const uint64_t seg_len = in_dim / SUBSEG;
        if (row < MIX_HC) {
            const uint64_t k0 = (uint64_t)sub * seg_len;
            const uint64_t k1 = k0 + seg_len;
            const __half *wr = w + (uint64_t)row * in_dim;
            float sum = 0.0f;
            for (uint64_t i = k0 + (uint64_t)d * 8u; i < k1; i += (uint64_t)nt * 8u) {
                const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
                const float4 xa = *reinterpret_cast<const float4 *>(x + i);
                const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
                const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
                const float2 w0 = __half22float2(h[0]);
                const float2 w1 = __half22float2(h[1]);
                const float2 w2 = __half22float2(h[2]);
                const float2 w3 = __half22float2(h[3]);
                sum += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                     + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
            }
            sum = block_reduce_sum_256(sum);
            if (d == 0u) dots[(uint64_t)row * SUBSEG + sub] = sum;
        }
    }
    if (b < 32u) {
        const uint64_t seg32 = in_dim / 32u;
        const uint64_t k0 = (uint64_t)b * seg32;
        float ss = 0.0f;
        for (uint64_t i = k0 + d; i < k0 + seg32; i += nt) { float v = x[i]; ss += v * v; }
        ss = block_reduce_sum_256(ss);
        if (d == 0u) ssq[b] = ss;
    }
    grid.sync();

    /* P2: block 0, warp 0: scale + mix (lanes 0..23), sinkhorn (lanes 0..15) */
    if (b == 0u && d < 32u) {
        float rms_scale;
        {
            float ss = 0.0f;
            for (uint32_t k = 0u; k < 32u; k++) ss += ssq[k];
            rms_scale = rsqrtf(ss / (float)in_dim + rms_eps);
        }
        if (d < MIX_HC) {
            float s = 0.0f;
            for (uint32_t k = 0u; k < SUBSEG; k++) s += dots[(uint64_t)d * SUBSEG + k];
            mix_out[d] = s * rms_scale;
        }
        __syncwarp();
        if (d < 16u) hc4_split_warp16(split, mix_out, scale3, base, sinkhorn_iters, epsv, d);
    }
    grid.sync();

    /* P3: out cols + ssq2 partials */
    {
        float s0 = split[0], s1 = split[1], s2 = split[2], s3 = split[3];
        float sum = 0.0f;
        for (uint32_t col = b * nt + d; col < n_embd; col += COOP_BLOCKS * nt) {
            float acc = residual_hc[col] * s0
                      + residual_hc[n_embd + col] * s1
                      + residual_hc[2u * n_embd + col] * s2
                      + residual_hc[3u * n_embd + col] * s3;
            out[col] = acc;
            sum += acc * acc;
        }
        sum = block_reduce_sum_256(sum);
        if (d == 0u) ssq2[b] = sum;
    }
    grid.sync();

    /* P5: every block redundantly reduces ssq2, then writes norm_out */
    {
        __shared__ float ns_sh;
        if (d == 0u) {
            float ss = 0.0f;
            for (uint32_t k = 0u; k < COOP_BLOCKS; k++) ss += ssq2[k];
            ns_sh = rsqrtf(ss / (float)n_embd + norm_eps);
        }
        __syncthreads();
        const float ns = ns_sh;
        for (uint32_t col = b * nt + d; col < n_embd; col += COOP_BLOCKS * nt) {
            norm_out[col] = out[col] * ns * norm_w[col];
        }
    }
}

/* ---------- V4: V3 + q8_0/q8_1 activation emission in the final phase ----
 * (M2-Inc1b) The norm_out values are quantized in-register by the warp that
 * writes them: each warp of the P5 col loop holds 32 consecutive columns =
 * exactly one q8 block.  q8_0 mirrors quantize_q8_0_f32_kernel (d=max/127,
 * lrintf, clamp); q8_1 mirrors the vendored quantize_q8_1 (roundf, ds=(d,sum),
 * shfl_xor butterflies) so both are BIT-EXACT vs the standalone kernels. */
typedef struct { __half2 ds; int8_t qs[32]; } block_q8_1_t;

__global__ static void hc_stage_coop3_q8_kernel(
        float *out, float *norm_out, float *split, float *mix_out,
        float *scratch,
        const __half *w, const float *x, const float *residual_hc,
        const float *scale3, const float *base, const float *norm_w,
        int8_t *q80_xq, float *q80_scale, block_q8_1_t *q81,
        uint32_t n_embd, uint32_t in_dim,
        uint32_t sinkhorn_iters, float epsv, float rms_eps, float norm_eps) {
    cg::grid_group grid = cg::this_grid();
    float *dots   = scratch;
    float *ssq    = scratch + MIX_HC * SUBSEG;
    float *ssq2   = ssq + 32u;
    const uint32_t b = blockIdx.x;
    const uint32_t d = threadIdx.x;
    const uint32_t nt = blockDim.x;

    /* P1 dots + sumsq (same as V3) */
    {
        const uint32_t row = b / SUBSEG;
        const uint32_t sub = b % SUBSEG;
        const uint64_t seg_len = in_dim / SUBSEG;
        if (row < MIX_HC) {
            const uint64_t k0 = (uint64_t)sub * seg_len;
            const uint64_t k1 = k0 + seg_len;
            const __half *wr = w + (uint64_t)row * in_dim;
            float sum = 0.0f;
            for (uint64_t i = k0 + (uint64_t)d * 8u; i < k1; i += (uint64_t)nt * 8u) {
                const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
                const float4 xa = *reinterpret_cast<const float4 *>(x + i);
                const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
                const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
                const float2 w0 = __half22float2(h[0]);
                const float2 w1 = __half22float2(h[1]);
                const float2 w2 = __half22float2(h[2]);
                const float2 w3 = __half22float2(h[3]);
                sum += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                     + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
            }
            sum = block_reduce_sum_256(sum);
            if (d == 0u) dots[(uint64_t)row * SUBSEG + sub] = sum;
        }
    }
    if (b < 32u) {
        const uint64_t seg32 = in_dim / 32u;
        const uint64_t k0 = (uint64_t)b * seg32;
        float ss = 0.0f;
        for (uint64_t i = k0 + d; i < k0 + seg32; i += nt) { float v = x[i]; ss += v * v; }
        ss = block_reduce_sum_256(ss);
        if (d == 0u) ssq[b] = ss;
    }
    grid.sync();

    /* P2 (same as V3) */
    if (b == 0u && d < 32u) {
        float rms_scale;
        {
            float ss = 0.0f;
            for (uint32_t k = 0u; k < 32u; k++) ss += ssq[k];
            rms_scale = rsqrtf(ss / (float)in_dim + rms_eps);
        }
        if (d < MIX_HC) {
            float s = 0.0f;
            for (uint32_t k = 0u; k < SUBSEG; k++) s += dots[(uint64_t)d * SUBSEG + k];
            mix_out[d] = s * rms_scale;
        }
        __syncwarp();
        if (d < 16u) hc4_split_warp16(split, mix_out, scale3, base, sinkhorn_iters, epsv, d);
    }
    grid.sync();

    /* P3 (same as V3) */
    {
        float s0 = split[0], s1 = split[1], s2 = split[2], s3 = split[3];
        float sum = 0.0f;
        for (uint32_t col = b * nt + d; col < n_embd; col += COOP_BLOCKS * nt) {
            float acc = residual_hc[col] * s0
                      + residual_hc[n_embd + col] * s1
                      + residual_hc[2u * n_embd + col] * s2
                      + residual_hc[3u * n_embd + col] * s3;
            out[col] = acc;
            sum += acc * acc;
        }
        sum = block_reduce_sum_256(sum);
        if (d == 0u) ssq2[b] = sum;
    }
    grid.sync();

    /* P5 + q8 emission: warp lanes hold 32 consecutive cols (all strides are
     * multiples of 32), so each warp emits exactly one q8 block per pass. */
    {
        __shared__ float ns_sh;
        if (d == 0u) {
            float ss = 0.0f;
            for (uint32_t k = 0u; k < COOP_BLOCKS; k++) ss += ssq2[k];
            ns_sh = rsqrtf(ss / (float)n_embd + norm_eps);
        }
        __syncthreads();
        const float ns = ns_sh;
        const uint32_t lane = d & 31u;
        for (uint32_t col = b * nt + d; col < n_embd; col += COOP_BLOCKS * nt) {
            const float v = out[col] * ns * norm_w[col];
            norm_out[col] = v;
            if (q80_xq || q81) {
                const uint32_t qb = col >> 5u;
                float amax = fabsf(v);
                for (uint32_t off = 16u; off > 0u; off >>= 1u)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
                if (q80_xq) {
                    const float d8 = amax / 127.0f;
                    const float id8 = d8 != 0.0f ? 1.0f / d8 : 0.0f;
                    int q = (int)lrintf(v * id8);
                    q = q > 127 ? 127 : (q < -128 ? -128 : q);
                    q80_xq[(uint64_t)qb * 32u + lane] = (int8_t)q;
                    if (lane == 0u) q80_scale[qb] = d8;
                }
                if (q81) {
                    float s = v;
                    for (uint32_t off = 16u; off > 0u; off >>= 1u)
                        s += __shfl_xor_sync(0xffffffffu, s, off, 32);
                    const float d1 = amax / 127.0f;
                    const int8_t q = amax == 0.0f ? (int8_t)0 : (int8_t)roundf(v / d1);
                    q81[qb].qs[lane] = q;
                    if (lane == 0u) q81[qb].ds = __floats2half2_rn(d1, s);
                }
            }
        }
    }
}

/* reference quantizers (verbatim semantics of the production kernels) */
__global__ static void ref_quantize_q8_0_kernel(
        int8_t *xq, float *xscale, const float *x, uint32_t in_dim) {
    const uint32_t b = blockIdx.x;
    const uint32_t i0 = b * 32u;
    const float *xr = x + i0;
    float a = threadIdx.x < 32u ? fabsf(xr[threadIdx.x]) : 0.0f;
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[b] = d;
    int v = (int)lrintf(xr[threadIdx.x] * id);
    v = v > 127 ? 127 : (v < -128 ? -128 : v);
    xq[b * 32u + threadIdx.x] = (int8_t)v;
    (void)in_dim;
}
__global__ static void ref_quantize_q8_1_kernel(block_q8_1_t *y, const float *x, uint32_t n) {
    const uint32_t i = blockIdx.x * 32u + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi);
    float sum = xi;
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
        sum += __shfl_xor_sync(0xffffffffu, sum, off, 32);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? (int8_t)0 : (int8_t)roundf(xi / d);
    y[blockIdx.x].qs[threadIdx.x] = q;
    if (threadIdx.x == 0) y[blockIdx.x].ds = __floats2half2_rn(d, sum);
}

/* ---------- host reference (double) ---------- */
static void ref_chain(double *out, double *norm_out, double *split24, double *mix24,
                      const float *x, const __half *w_h, const float *scale3,
                      const float *base, const float *norm_w) {
    /* rms norm */
    double ss = 0.0;
    for (uint32_t i = 0; i < HC_DIM; i++) ss += (double)x[i] * x[i];
    double rs = 1.0 / sqrt(ss / HC_DIM + (double)RMS_EPS);
    /* matmul */
    for (uint32_t r = 0; r < MIX_HC; r++) {
        double s = 0.0;
        for (uint32_t i = 0; i < HC_DIM; i++)
            s += (double)__half2float(w_h[(uint64_t)r * HC_DIM + i]) * ((double)x[i] * rs);
        mix24[r] = s;
    }
    /* sinkhorn (double port of hc4_split_one) */
    double sp[24];
    {
        const double pre = scale3[0], post = scale3[1], comb = scale3[2];
        for (int i = 0; i < 4; i++) sp[i] = 1.0 / (1.0 + exp(-(mix24[i] * pre + base[i]))) + (double)HC_EPS;
        for (int i = 0; i < 4; i++) sp[4 + i] = 2.0 / (1.0 + exp(-(mix24[4 + i] * post + base[4 + i])));
        double c[16];
        for (int r = 0; r < 4; r++) {
            double m = -1e300;
            for (int col = 0; col < 4; col++) {
                double v = mix24[8 + r * 4 + col] * comb + base[8 + r * 4 + col];
                c[r * 4 + col] = v; if (v > m) m = v;
            }
            double s = 0.0;
            for (int col = 0; col < 4; col++) { double v = exp(c[r * 4 + col] - m); c[r * 4 + col] = v; s += v; }
            for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + (double)HC_EPS;
        }
        for (int col = 0; col < 4; col++) {
            double s = (double)HC_EPS;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
        for (uint32_t it = 1; it < SINK_IT; it++) {
            for (int r = 0; r < 4; r++) {
                double s = (double)HC_EPS;
                for (int col = 0; col < 4; col++) s += c[r * 4 + col];
                for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
            }
            for (int col = 0; col < 4; col++) {
                double s = (double)HC_EPS;
                for (int r = 0; r < 4; r++) s += c[r * 4 + col];
                for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
            }
        }
        for (int i = 0; i < 16; i++) sp[8 + i] = c[i];
    }
    for (int i = 0; i < 24; i++) split24[i] = sp[i];
    /* weighted sum + norm (residual = raw x) */
    double ss2 = 0.0;
    for (uint32_t col = 0; col < N_EMBD; col++) {
        double acc = 0.0;
        for (uint32_t h = 0; h < 4; h++) acc += (double)x[(uint64_t)h * N_EMBD + col] * sp[h];
        out[col] = acc;
        ss2 += acc * acc;
    }
    double ns = 1.0 / sqrt(ss2 / N_EMBD + (double)RMS_EPS);
    for (uint32_t col = 0; col < N_EMBD; col++) norm_out[col] = out[col] * ns * (double)norm_w[col];
}

static double max_rel(const float *a, const double *r, uint32_t n) {
    double m = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - r[i]);
        double den = fabs(r[i]) > 1e-12 ? fabs(r[i]) : 1e-12;
        double e = d / den;
        if (e > m) m = e;
    }
    return m;
}

/* ---------- launch wrappers ---------- */
struct Bufs {
    __half *w[N_LAYERS];
    float *norm_w[N_LAYERS];
    float *scale3, *base;
    float *x;                     /* cur_hc, 16384 */
    float *flat, *mix, *split, *out, *norm_out;
    float *partial;               /* 24*32 */
    float *ssq;                   /* 32 */
    float *coop_scratch;          /* 24*4+32+96+2 */
    int8_t *q80; float *q80s;     /* V4 emission: q8_0 codes + scales */
    block_q8_1_t *q81;            /* V4 emission: q8_1 blocks */
    int8_t *q80r; float *q80sr;   /* reference quantizer outputs */
    block_q8_1_t *q81r;
};

static void launch_v0(const Bufs &B, int li, cudaStream_t st) {
    rms_norm_plain_kernel<<<1, 256, 0, st>>>(B.flat, B.x, HC_DIM, 1, RMS_EPS);
    dim3 sg(MIX_HC, KSPLIT, 1);
    matmul_f16_splitk_kernel<<<sg, 256, 0, st>>>(B.partial, B.w[li], B.flat, HC_DIM, MIX_HC, KSPLIT, 1);
    matmul_f16_splitk_combine_kernel<<<1, 256, 0, st>>>(B.mix, B.partial, MIX_HC, KSPLIT);
    hc_split_weighted_sum_norm_fused_kernel<<<1, 256, 0, st>>>(
            B.out, B.norm_out, B.split, B.mix, B.x, B.scale3, B.base, B.norm_w[li],
            N_EMBD, N_HC, 1, SINK_IT, HC_EPS, RMS_EPS);
}

static void launch_v1(const Bufs &B, int li, cudaStream_t st) {
    dim3 g1(MIX_HC + 1u, KSPLIT, 1);
    hc_stage_dots_kernel<<<g1, 256, 0, st>>>(B.partial, B.ssq, B.w[li], B.x, HC_DIM, KSPLIT);
    hc_stage_finish_kernel<<<1, 256, 0, st>>>(
            B.out, B.norm_out, B.split, B.mix, B.partial, B.ssq, B.x,
            B.scale3, B.base, B.norm_w[li],
            N_EMBD, HC_DIM, KSPLIT, SINK_IT, HC_EPS, RMS_EPS, RMS_EPS);
}

static cudaError_t launch_coop(const Bufs &B, int li, cudaStream_t st, void *kfn) {
    uint32_t n_embd = N_EMBD, in_dim = HC_DIM, sink = SINK_IT;
    float epsv = HC_EPS, rms_eps = RMS_EPS, norm_eps = RMS_EPS;
    void *args[] = {
        (void *)&B.out, (void *)&B.norm_out, (void *)&B.split, (void *)&B.mix,
        (void *)&B.coop_scratch,
        (void *)&B.w[li], (void *)&B.x, (void *)&B.x,
        (void *)&B.scale3, (void *)&B.base, (void *)&B.norm_w[li],
        (void *)&n_embd, (void *)&in_dim, (void *)&sink,
        (void *)&epsv, (void *)&rms_eps, (void *)&norm_eps };
    return cudaLaunchCooperativeKernel(kfn,
            dim3(COOP_BLOCKS, 1, 1), dim3(256, 1, 1), args, 0, st);
}
static cudaError_t launch_v2(const Bufs &B, int li, cudaStream_t st) {
    return launch_coop(B, li, st, (void *)hc_stage_coop_kernel);
}
static cudaError_t launch_v3(const Bufs &B, int li, cudaStream_t st) {
    return launch_coop(B, li, st, (void *)hc_stage_coop3_kernel);
}
static cudaError_t launch_v4(const Bufs &B, int li, cudaStream_t st) {
    uint32_t n_embd = N_EMBD, in_dim = HC_DIM, sink = SINK_IT;
    float epsv = HC_EPS, rms_eps = RMS_EPS, norm_eps = RMS_EPS;
    void *args[] = {
        (void *)&B.out, (void *)&B.norm_out, (void *)&B.split, (void *)&B.mix,
        (void *)&B.coop_scratch,
        (void *)&B.w[li], (void *)&B.x, (void *)&B.x,
        (void *)&B.scale3, (void *)&B.base, (void *)&B.norm_w[li],
        (void *)&B.q80, (void *)&B.q80s, (void *)&B.q81,
        (void *)&n_embd, (void *)&in_dim, (void *)&sink,
        (void *)&epsv, (void *)&rms_eps, (void *)&norm_eps };
    return cudaLaunchCooperativeKernel((void *)hc_stage_coop3_q8_kernel,
            dim3(COOP_BLOCKS, 1, 1), dim3(256, 1, 1), args, 0, st);
}

int main() {
    Bufs B;
    srand(12345);
    const uint64_t wcnt = (uint64_t)MIX_HC * HC_DIM;
    __half *wh_host = (__half *)malloc(wcnt * sizeof(__half));
    float *nw_host = (float *)malloc(N_EMBD * sizeof(float));
    float *x_host = (float *)malloc(HC_DIM * sizeof(float));
    for (int l = 0; l < N_LAYERS; l++) {
        CK(cudaMalloc(&B.w[l], wcnt * sizeof(__half)));
        CK(cudaMalloc(&B.norm_w[l], N_EMBD * sizeof(float)));
        for (uint64_t i = 0; i < wcnt; i++) wh_host[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.02f);
        for (uint32_t i = 0; i < N_EMBD; i++) nw_host[i] = 0.8f + 0.4f * (float)rand() / RAND_MAX;
        CK(cudaMemcpy(B.w[l], wh_host, wcnt * sizeof(__half), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(B.norm_w[l], nw_host, N_EMBD * sizeof(float), cudaMemcpyHostToDevice));
    }
    float sc_host[3] = { 0.9f, 1.1f, 1.05f };
    float base_host[MIX_HC];
    for (uint32_t i = 0; i < MIX_HC; i++) base_host[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.4f;
    for (uint32_t i = 0; i < HC_DIM; i++) x_host[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    CK(cudaMalloc(&B.scale3, 3 * sizeof(float)));
    CK(cudaMalloc(&B.base, MIX_HC * sizeof(float)));
    CK(cudaMalloc(&B.x, HC_DIM * sizeof(float)));
    CK(cudaMalloc(&B.flat, HC_DIM * sizeof(float)));
    CK(cudaMalloc(&B.mix, MIX_HC * sizeof(float)));
    CK(cudaMalloc(&B.split, MIX_HC * sizeof(float)));
    CK(cudaMalloc(&B.out, N_EMBD * sizeof(float)));
    CK(cudaMalloc(&B.norm_out, N_EMBD * sizeof(float)));
    CK(cudaMalloc(&B.partial, MIX_HC * KSPLIT * sizeof(float)));
    CK(cudaMalloc(&B.ssq, KSPLIT * sizeof(float)));
    CK(cudaMalloc(&B.coop_scratch, 512 * sizeof(float)));
    const uint32_t qblocks = N_EMBD / 32u;
    CK(cudaMalloc(&B.q80, qblocks * 32u));
    CK(cudaMalloc(&B.q80s, qblocks * sizeof(float)));
    CK(cudaMalloc(&B.q81, qblocks * sizeof(block_q8_1_t)));
    CK(cudaMalloc(&B.q80r, qblocks * 32u));
    CK(cudaMalloc(&B.q80sr, qblocks * sizeof(float)));
    CK(cudaMalloc(&B.q81r, qblocks * sizeof(block_q8_1_t)));
    CK(cudaMemcpy(B.scale3, sc_host, 3 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B.base, base_host, MIX_HC * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B.x, x_host, HC_DIM * sizeof(float), cudaMemcpyHostToDevice));

    /* coop support check */
    int dev = 0, coop_ok = 0;
    CK(cudaDeviceGetAttribute(&coop_ok, cudaDevAttrCooperativeLaunch, dev));
    int max_blocks = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, hc_stage_coop_kernel, 256, 0));
    int n_sm = 0;
    CK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
    printf("coop launch supported=%d, coop kernel max co-resident blocks=%d (%d SMs, need %u)\n",
           coop_ok, max_blocks * n_sm, n_sm, COOP_BLOCKS);

    /* ---- parity (layer 0 weights) ---- */
    double *r_out = (double *)malloc(N_EMBD * sizeof(double));
    double *r_norm = (double *)malloc(N_EMBD * sizeof(double));
    double r_split[24], r_mix[24];
    /* re-generate layer-0 host weights deterministically: redo copy back from device */
    CK(cudaMemcpy(wh_host, B.w[0], wcnt * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(nw_host, B.norm_w[0], N_EMBD * sizeof(float), cudaMemcpyDeviceToHost));
    ref_chain(r_out, r_norm, r_split, r_mix, x_host, wh_host, sc_host, base_host, nw_host);

    float *h_out = (float *)malloc(N_EMBD * sizeof(float));
    float *h_norm = (float *)malloc(N_EMBD * sizeof(float));
    float h_split[24], h_mix[24];
    cudaStream_t st;
    CK(cudaStreamCreate(&st));

    const char *names[5] = { "V0-baseline", "V1-fused2", "V2-coop1", "V3-coop3sync", "V4-coop3+q8emit" };
    for (int v = 0; v < 5; v++) {
        CK(cudaMemsetAsync(B.out, 0, N_EMBD * sizeof(float), st));
        CK(cudaMemsetAsync(B.norm_out, 0, N_EMBD * sizeof(float), st));
        if (v == 0) launch_v0(B, 0, st);
        else if (v == 1) launch_v1(B, 0, st);
        else {
            cudaError_t e = v == 2 ? launch_v2(B, 0, st) : (v == 3 ? launch_v3(B, 0, st) : launch_v4(B, 0, st));
            if (e != cudaSuccess) { printf("%s: coop launch FAILED: %s\n", names[v], cudaGetErrorString(e)); continue; }
        }
        CK(cudaStreamSynchronize(st));
        if (v == 4) {
            /* q8 emission bit-exactness vs the reference quantizers run on
             * this very norm_out */
            ref_quantize_q8_0_kernel<<<qblocks, 32, 0, st>>>(B.q80r, B.q80sr, B.norm_out, N_EMBD);
            ref_quantize_q8_1_kernel<<<qblocks, 32, 0, st>>>(B.q81r, B.norm_out, N_EMBD);
            CK(cudaStreamSynchronize(st));
            int8_t hq[2][128 * 32]; float hs[2][128];
            block_q8_1_t h1[2][128];
            CK(cudaMemcpy(hq[0], B.q80, qblocks * 32u, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hq[1], B.q80r, qblocks * 32u, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hs[0], B.q80s, qblocks * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hs[1], B.q80sr, qblocks * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h1[0], B.q81, qblocks * sizeof(block_q8_1_t), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h1[1], B.q81r, qblocks * sizeof(block_q8_1_t), cudaMemcpyDeviceToHost));
            int bad80 = memcmp(hq[0], hq[1], qblocks * 32u) != 0 ||
                        memcmp(hs[0], hs[1], qblocks * sizeof(float)) != 0;
            int bad81 = memcmp(h1[0], h1[1], qblocks * sizeof(block_q8_1_t)) != 0;
            printf("V4 q8 emission: q8_0 %s, q8_1 %s (vs reference quantizers, bit-exact)\n",
                   bad80 ? "MISMATCH" : "OK", bad81 ? "MISMATCH" : "OK");
        }
        CK(cudaMemcpy(h_out, B.out, N_EMBD * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_norm, B.norm_out, N_EMBD * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_split, B.split, 24 * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_mix, B.mix, 24 * sizeof(float), cudaMemcpyDeviceToHost));
        double e_mix = 0, e_split = 0;
        for (int i = 0; i < 24; i++) {
            double dm = fabs((double)h_mix[i] - r_mix[i]) / fmax(fabs(r_mix[i]), 1e-12);
            double dsp = fabs((double)h_split[i] - r_split[i]) / fmax(fabs(r_split[i]), 1e-12);
            if (dm > e_mix) e_mix = dm;
            if (dsp > e_split) e_split = dsp;
        }
        printf("%s parity: mix %.3e split %.3e out %.3e norm_out %.3e\n",
               names[v], e_mix, e_split, max_rel(h_out, r_out, N_EMBD), max_rel(h_norm, r_norm, N_EMBD));
    }

    /* ---- timing: rotate 43 layers, REPS sweeps ---- */
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int v = 0; v < 5; v++) {
        /* warmup */
        bool bad = false;
        for (int l = 0; l < N_LAYERS && !bad; l++) {
            if (v == 0) launch_v0(B, l, st);
            else if (v == 1) launch_v1(B, l, st);
            else if ((v == 2 ? launch_v2(B, l, st)
                             : (v == 3 ? launch_v3(B, l, st) : launch_v4(B, l, st))) != cudaSuccess) bad = true;
        }
        if (bad) { printf("%s: skipped (launch failed)\n", names[v]); continue; }
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(e0, st));
        for (int r = 0; r < REPS; r++)
            for (int l = 0; l < N_LAYERS; l++) {
                if (v == 0) launch_v0(B, l, st);
                else if (v == 1) launch_v1(B, l, st);
                else if (v == 2) (void)launch_v2(B, l, st);
                else if (v == 3) (void)launch_v3(B, l, st);
                else (void)launch_v4(B, l, st);
            }
        CK(cudaEventRecord(e1, st));
        CK(cudaStreamSynchronize(st));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("%s: %.2f us/chain (x86 per step = %.2f ms/step)\n",
               names[v], 1000.0f * ms / (REPS * N_LAYERS), 2.0f * ms / REPS);
    }

    /* ---- graph capture/replay test ---- */
    for (int v = 1; v < 5; v++) {
        cudaGraph_t graph;
        cudaError_t e = cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
        if (e != cudaSuccess) { printf("%s capture begin failed: %s\n", names[v], cudaGetErrorString(e)); continue; }
        if (v == 1) launch_v1(B, 0, st);
        else {
            e = v == 2 ? launch_v2(B, 0, st) : (v == 3 ? launch_v3(B, 0, st) : launch_v4(B, 0, st));
            if (e != cudaSuccess) {
                printf("%s: coop launch UNDER CAPTURE failed: %s\n", names[v], cudaGetErrorString(e));
                cudaStreamEndCapture(st, &graph);
                cudaGetLastError();
                continue;
            }
        }
        e = cudaStreamEndCapture(st, &graph);
        if (e != cudaSuccess) { printf("%s capture end failed: %s\n", names[v], cudaGetErrorString(e)); cudaGetLastError(); continue; }
        cudaGraphExec_t gexec;
        e = cudaGraphInstantiate(&gexec, graph, NULL, NULL, 0);
        if (e != cudaSuccess) { printf("%s graph instantiate failed: %s\n", names[v], cudaGetErrorString(e)); cudaGraphDestroy(graph); cudaGetLastError(); continue; }
        /* eager result */
        if (v == 1) launch_v1(B, 0, st); else if (v == 2) (void)launch_v2(B, 0, st);
        else if (v == 3) (void)launch_v3(B, 0, st); else (void)launch_v4(B, 0, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(h_out, B.norm_out, N_EMBD * sizeof(float), cudaMemcpyDeviceToHost));
        /* replayed result */
        CK(cudaMemsetAsync(B.norm_out, 0, N_EMBD * sizeof(float), st));
        CK(cudaGraphLaunch(gexec, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(h_norm, B.norm_out, N_EMBD * sizeof(float), cudaMemcpyDeviceToHost));
        int identical = memcmp(h_out, h_norm, N_EMBD * sizeof(float)) == 0;
        printf("%s graph capture/replay: %s (eager==replay bit-%s)\n",
               names[v], "OK", identical ? "IDENTICAL" : "DIFFERENT");
        cudaGraphExecDestroy(gexec);
        cudaGraphDestroy(graph);
    }
    printf("PROTO_M2_HC_DONE\n");
    return 0;
}
