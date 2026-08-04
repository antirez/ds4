// proto_q8_warp8.cu — Inc4 prototype (megakernel program).
//
// The nsys truth table (nsys33) put the three CUSTOM warp8 Q8_0 decode
// kernels at 21.2 ms/step combined (hc_expand 9.09 / grouped_a 6.98 /
// pair 5.13) running at 150-195 GB/s on the raw 34-byte block layout.
// proto_q8_aligned proved the aligned SoA layout ([half dq[nblk]][pad64]
// [int8 qs[nblk*32]], derived kind 5) reaches 235-245 GB/s at these K.
//
// This proto A/Bs each warp8 kernel VERBATIM against an aligned variant
// with the SAME warp8 structure, prequant inputs (xq int8 + xscale float,
// quantize_q8_0_f32 semantics), accumulation order, and epilogue — only
// the weight load path changes (unaligned 34B-stride assembles -> two
// aligned int4 per block).  Production Flash shapes:
//   hc_expand : out=4096 in=8192 (attn_output_b)  + out=4096 in=2048 (shared_down)
//   pair      : in=4096 out0=1024 out1=512 (q_a+kv) + out0=out1=2048 (shared gate+up)
//   grouped   : group_dim=4096 rank=1024 n_groups=8 (attn_output_a)
//
// L2-defeating rotation (>=256 MB of weight copies) + double host reference
// parity, per the proto_q8_aligned method rules.
//
// Build (box):
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 -I.. \
//        proto_q8_warp8.cu -lcudart -lcuda -o proto_q8_warp8

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

namespace {

// ---- device helpers copied verbatim from ds4_cuda.cu ----------------------
__device__ __forceinline__ int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}
__device__ __forceinline__ int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}
__device__ __forceinline__ int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}
__device__ __forceinline__ int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}
__device__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Aligned per-block dot: two 16B loads + 8 dp4a, same integer result.
__device__ __forceinline__ int32_t dot_i8x32_aligned(const int8_t *qs32, const int8_t *xqb) {
    const int4 w0 = *(const int4 *)qs32;
    const int4 w1 = *(const int4 *)(qs32 + 16);
    const int *u = (const int *)xqb;
    int sumi = 0;
    sumi = __dp4a(w0.x, u[0], sumi);
    sumi = __dp4a(w0.y, u[1], sumi);
    sumi = __dp4a(w0.z, u[2], sumi);
    sumi = __dp4a(w0.w, u[3], sumi);
    sumi = __dp4a(w1.x, u[4], sumi);
    sumi = __dp4a(w1.y, u[5], sumi);
    sumi = __dp4a(w1.z, u[6], sumi);
    sumi = __dp4a(w1.w, u[7], sumi);
    return sumi;
}

// ==== 1. hc_expand ==========================================================
// Baseline: verbatim matmul_q8_0_hc_expand_preq_warp8_kernel (ds4_cuda.cu).
__global__ void hc_expand_base_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split, const unsigned char *w,
        const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_embd, uint32_t n_hc,
        uint64_t blocks, int has_add, int use_dp4a) {
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

// Aligned variant: identical loop/epilogue; weight loads via [dq|qs] artifact.
__global__ void hc_expand_aligned_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const int8_t *qs_art, const __half *dq_art,
        const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_embd, uint32_t n_hc,
        uint64_t blocks, int has_add) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const uint64_t rbase = row * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const int sumi = dot_i8x32_aligned(qs_art + (rbase + b) * 32u, xq + b * 32u);
        acc += __half2float(dq_art[rbase + b]) * xscale[b] * (float)sumi;
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

// ==== 2. pair ===============================================================
__global__ void pair_base_kernel(
        float *out0, float *out1, const unsigned char *w0, const unsigned char *w1,
        const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim, uint64_t blocks,
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

__global__ void pair_aligned_kernel(
        float *out0, float *out1,
        const int8_t *qs0, const __half *dq0,
        const int8_t *qs1, const __half *dq1,
        const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim, uint64_t blocks) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    const uint64_t rbase = row * blocks;
    const int in0 = row < out0_dim;
    const int in1 = row < out1_dim;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        if (in0) {
            const int sumi = dot_i8x32_aligned(qs0 + (rbase + b) * 32u, xqb);
            acc0 += __half2float(dq0[rbase + b]) * xs * (float)sumi;
        }
        if (in1) {
            const int sumi = dot_i8x32_aligned(qs1 + (rbase + b) * 32u, xqb);
            acc1 += __half2float(dq1[rbase + b]) * xs * (float)sumi;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

// ==== 3. grouped ============================================================
__global__ void grouped_base_kernel(
        float *low, const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t blocks, int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
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

__global__ void grouped_aligned_kernel(
        float *low, const int8_t *qs_art, const __half *dq_art,
        const int8_t *xq, const float *xscale,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t rbase = row * blocks;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const int sumi = dot_i8x32_aligned(qs_art + (rbase + b) * 32u, xqr + b * 32u);
        acc += __half2float(dq_art[rbase + b]) * xsr[b] * (float)sumi;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

// ==== host helpers ==========================================================
// Mirror quantize_q8_0_f32_kernel semantics (lrintf, clamp 127/-128).
static void quantize_q8_0_host(const float *x, int8_t *xq, float *xscale, int K) {
    for (int b = 0; b < K / 32; b++) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        xscale[b] = d;
        for (int i = 0; i < 32; i++) {
            int v = (int)lrintf(x[b * 32 + i] * id);
            v = v > 127 ? 127 : (v < -128 ? -128 : v);
            xq[b * 32 + i] = (int8_t)v;
        }
    }
}

struct RawArt {
    std::vector<uint8_t> W;    // raw 34B-block rows
    std::vector<uint8_t> ART;  // [dq halves][pad64][qs 32B]
    uint64_t dq_bytes;
};

static RawArt make_weights(std::mt19937 &rng, long long rows, int blocks) {
    RawArt r;
    const long long nblk = rows * blocks;
    r.W.resize((size_t)nblk * 34);
    for (auto &b : r.W) b = (uint8_t)(rng() & 0xff);
    for (long long blk = 0; blk < nblk; blk++) {
        uint16_t h = (uint16_t)(0x2c00 | (rng() & 0xff));
        memcpy(&r.W[blk * 34], &h, 2);
    }
    r.dq_bytes = ((uint64_t)nblk * 2u + 63u) & ~63ull;
    r.ART.resize(r.dq_bytes + (uint64_t)nblk * 32u);
    for (long long blk = 0; blk < nblk; blk++) {
        memcpy(&r.ART[blk * 2], &r.W[blk * 34], 2);
        memcpy(&r.ART[r.dq_bytes + (uint64_t)blk * 32u], &r.W[blk * 34 + 2], 32);
    }
    return r;
}

// Double reference GEMV row: sum_b dq[b] * xscale[b] * (int dot)
static double ref_row(const uint8_t *wrow, const int8_t *xq, const float *xs, int blocks) {
    double ref = 0.0;
    for (int b = 0; b < blocks; b++) {
        const uint8_t *blk = wrow + (size_t)b * 34;
        uint16_t h; memcpy(&h, blk, 2);
        const float dw = __half2float(__ushort_as_half(h));
        const int8_t *q = (const int8_t *)(blk + 2);
        const int8_t *u = xq + (size_t)b * 32;
        int s = 0;
        for (int j = 0; j < 32; j++) s += (int)q[j] * (int)u[j];
        ref += (double)dw * (double)xs[b] * (double)s;
    }
    return ref;
}

struct DevCopies {
    uint8_t *dW, *dArt;
    int n_copies;
    size_t w_sz, art_sz;
    const uint8_t *w(int i) const { return dW + (size_t)(i % n_copies) * w_sz; }
    const uint8_t *art(int i) const { return dArt + (size_t)(i % n_copies) * art_sz; }
    const int8_t *art_qs(int i, uint64_t dqb) const { return (const int8_t *)(art(i) + dqb); }
    const __half *art_dq(int i) const { return (const __half *)art(i); }
};

static DevCopies upload_rotating(const RawArt &r, size_t min_footprint = 256ull << 20) {
    DevCopies d;
    d.w_sz = r.W.size(); d.art_sz = r.ART.size();
    d.n_copies = (int)(min_footprint / (d.w_sz + 1)) + 1;
    CK(cudaMalloc(&d.dW, d.w_sz * d.n_copies));
    CK(cudaMalloc(&d.dArt, d.art_sz * d.n_copies));
    for (int c = 0; c < d.n_copies; c++) {
        CK(cudaMemcpy(d.dW + (size_t)c * d.w_sz, r.W.data(), d.w_sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.dArt + (size_t)c * d.art_sz, r.ART.data(), d.art_sz, cudaMemcpyHostToDevice));
    }
    return d;
}

static float time_loop(cudaStream_t stream, cudaEvent_t e0, cudaEvent_t e1, int iters,
                       void (*launch)(int, void *), void *ud) {
    launch(0, ud);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++) launch(i, ud);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms = 0.0f; CK(cudaEventElapsedTime(&ms, e0, e1));
    return ms / iters;
}

static int check_pair_tol(const std::vector<float> &base, const std::vector<float> &al,
                          const std::vector<double> &ref, const char *tag) {
    double eb_max = 0.0, ea_max = 0.0; int bad = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double eb = fabs((double)base[i] - ref[i]);
        const double ea = fabs((double)al[i] - ref[i]);
        if (eb > eb_max) eb_max = eb;
        if (ea > ea_max) ea_max = ea;
        if (ea > eb * 4.0 + 1e-2) bad++;
    }
    printf("  parity(%s): max_abs_err base=%.3e aligned=%.3e bad=%d -> %s\n",
           tag, eb_max, ea_max, bad, bad == 0 ? "PASS" : "FAIL");
    return bad;
}

// ==== section runners =======================================================
struct HcCtx {
    cudaStream_t stream; DevCopies dev; uint64_t dq_bytes;
    float *out_hc, *block_out; const float *block_add, *residual, *split;
    const int8_t *xq; const float *xs;
    uint64_t in_dim, out_dim, blocks; uint32_t n_embd, n_hc; int base;
};
static void hc_launch(int i, void *p) {
    HcCtx *c = (HcCtx *)p;
    const unsigned grid = ((unsigned)c->out_dim + 7u) / 8u;
    if (c->base) {
        hc_expand_base_kernel<<<grid, 256, 0, c->stream>>>(
            c->out_hc, c->block_out, c->block_add, c->residual, c->split,
            c->dev.w(i), c->xq, c->xs, c->in_dim, c->out_dim, c->n_embd, c->n_hc,
            c->blocks, 1, 1);
    } else {
        hc_expand_aligned_kernel<<<grid, 256, 0, c->stream>>>(
            c->out_hc, c->block_out, c->block_add, c->residual, c->split,
            c->dev.art_qs(i, c->dq_bytes), c->dev.art_dq(i), c->xq, c->xs,
            c->in_dim, c->out_dim, c->n_embd, c->n_hc, c->blocks, 1);
    }
}

static int run_hc_expand(const char *label, int out_dim, int in_dim, int iters, uint32_t seed) {
    const int blocks = in_dim / 32, n_hc = 4;
    std::mt19937 rng(seed);
    RawArt r = make_weights(rng, out_dim, blocks);
    std::vector<float> X(in_dim);
    for (auto &v : X) v = ((float)(rng() % 2000) - 1000.0f) / 500.0f;
    std::vector<int8_t> XQ(in_dim); std::vector<float> XS(blocks);
    quantize_q8_0_host(X.data(), XQ.data(), XS.data(), in_dim);
    std::vector<float> ADD(out_dim), RES((size_t)n_hc * out_dim), SPL(2 * n_hc + n_hc * n_hc);
    for (auto &v : ADD) v = ((float)(rng() % 2000) - 1000.0f) / 1000.0f;
    for (auto &v : RES) v = ((float)(rng() % 2000) - 1000.0f) / 1000.0f;
    for (auto &v : SPL) v = ((float)(rng() % 2000) - 1000.0f) / 2000.0f;

    DevCopies dev = upload_rotating(r);
    float *dOutHc, *dBlk, *dAdd, *dRes, *dSpl, *dXs; int8_t *dXq;
    CK(cudaMalloc(&dOutHc, sizeof(float) * n_hc * out_dim));
    CK(cudaMalloc(&dBlk, sizeof(float) * out_dim));
    CK(cudaMalloc(&dAdd, sizeof(float) * out_dim));
    CK(cudaMalloc(&dRes, sizeof(float) * RES.size()));
    CK(cudaMalloc(&dSpl, sizeof(float) * SPL.size()));
    CK(cudaMalloc(&dXq, XQ.size()));
    CK(cudaMalloc(&dXs, sizeof(float) * XS.size()));
    CK(cudaMemcpy(dAdd, ADD.data(), sizeof(float) * ADD.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dRes, RES.data(), sizeof(float) * RES.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSpl, SPL.data(), sizeof(float) * SPL.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXq, XQ.data(), XQ.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXs, XS.data(), sizeof(float) * XS.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    HcCtx c = {stream, dev, r.dq_bytes, dOutHc, dBlk, dAdd, dRes, dSpl,
               dXq, dXs, (uint64_t)in_dim, (uint64_t)out_dim, (uint64_t)blocks,
               (uint32_t)out_dim, (uint32_t)n_hc, 1};
    const float ms_base = time_loop(stream, e0, e1, iters, hc_launch, &c);
    std::vector<float> bBlk(out_dim), bHc((size_t)n_hc * out_dim);
    CK(cudaMemcpy(bBlk.data(), dBlk, sizeof(float) * out_dim, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bHc.data(), dOutHc, sizeof(float) * bHc.size(), cudaMemcpyDeviceToHost));
    c.base = 0;
    const float ms_al = time_loop(stream, e0, e1, iters, hc_launch, &c);
    std::vector<float> aBlk(out_dim), aHc((size_t)n_hc * out_dim);
    CK(cudaMemcpy(aBlk.data(), dBlk, sizeof(float) * out_dim, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(aHc.data(), dOutHc, sizeof(float) * aHc.size(), cudaMemcpyDeviceToHost));

    // double refs
    std::vector<double> rBlk(out_dim), rHc((size_t)n_hc * out_dim);
    for (int d = 0; d < out_dim; d++) {
        rBlk[d] = ref_row(&r.W[(size_t)d * blocks * 34], XQ.data(), XS.data(), blocks);
        const double bv = rBlk[d] + (double)ADD[d];
        for (int dst = 0; dst < n_hc; dst++) {
            double hc = bv * (double)SPL[n_hc + dst];
            for (int src = 0; src < n_hc; src++)
                hc += (double)SPL[2 * n_hc + dst + src * n_hc] * (double)RES[(size_t)src * out_dim + d];
            rHc[(size_t)dst * out_dim + d] = hc;
        }
    }
    const double gb = (double)out_dim * blocks * 34 / 1e9;
    printf("%-22s out=%d in=%d (%.1f MB/call, iters=%d)\n", label, out_dim, in_dim, gb * 1000, iters);
    printf("  baseline warp8     : %.4f ms -> %6.1f GB/s\n", ms_base, gb / (ms_base / 1e3));
    printf("  aligned  warp8     : %.4f ms -> %6.1f GB/s  (%+.1f%%)\n",
           ms_al, gb / (ms_al / 1e3), 100.0 * (ms_base / ms_al - 1.0));
    int bad = check_pair_tol(bBlk, aBlk, rBlk, "block_out");
    bad += check_pair_tol(bHc, aHc, rHc, "out_hc");

    CK(cudaFree(dev.dW)); CK(cudaFree(dev.dArt));
    CK(cudaFree(dOutHc)); CK(cudaFree(dBlk)); CK(cudaFree(dAdd)); CK(cudaFree(dRes));
    CK(cudaFree(dSpl)); CK(cudaFree(dXq)); CK(cudaFree(dXs));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaStreamDestroy(stream));
    return bad;
}

struct PairCtx {
    cudaStream_t stream; DevCopies d0, d1; uint64_t dqb0, dqb1;
    float *out0, *out1; const int8_t *xq; const float *xs;
    uint64_t in_dim, out0_dim, out1_dim, blocks; int base;
};
static void pair_launch(int i, void *p) {
    PairCtx *c = (PairCtx *)p;
    const uint64_t max_out = c->out0_dim > c->out1_dim ? c->out0_dim : c->out1_dim;
    const unsigned grid = ((unsigned)max_out + 7u) / 8u;
    if (c->base) {
        pair_base_kernel<<<grid, 256, 0, c->stream>>>(
            c->out0, c->out1, c->d0.w(i), c->d1.w(i), c->xq, c->xs,
            c->in_dim, c->out0_dim, c->out1_dim, c->blocks, 1);
    } else {
        pair_aligned_kernel<<<grid, 256, 0, c->stream>>>(
            c->out0, c->out1,
            c->d0.art_qs(i, c->dqb0), c->d0.art_dq(i),
            c->d1.art_qs(i, c->dqb1), c->d1.art_dq(i),
            c->xq, c->xs, c->in_dim, c->out0_dim, c->out1_dim, c->blocks);
    }
}

static int run_pair(const char *label, int out0, int out1, int in_dim, int iters, uint32_t seed) {
    const int blocks = in_dim / 32;
    std::mt19937 rng(seed);
    RawArt r0 = make_weights(rng, out0, blocks);
    RawArt r1 = make_weights(rng, out1, blocks);
    std::vector<float> X(in_dim);
    for (auto &v : X) v = ((float)(rng() % 2000) - 1000.0f) / 500.0f;
    std::vector<int8_t> XQ(in_dim); std::vector<float> XS(blocks);
    quantize_q8_0_host(X.data(), XQ.data(), XS.data(), in_dim);

    DevCopies d0 = upload_rotating(r0, 128ull << 20);
    DevCopies d1 = upload_rotating(r1, 128ull << 20);
    float *dOut0, *dOut1, *dXs; int8_t *dXq;
    CK(cudaMalloc(&dOut0, sizeof(float) * out0));
    CK(cudaMalloc(&dOut1, sizeof(float) * out1));
    CK(cudaMalloc(&dXq, XQ.size()));
    CK(cudaMalloc(&dXs, sizeof(float) * XS.size()));
    CK(cudaMemcpy(dXq, XQ.data(), XQ.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXs, XS.data(), sizeof(float) * XS.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    PairCtx c = {stream, d0, d1, r0.dq_bytes, r1.dq_bytes, dOut0, dOut1,
                 dXq, dXs, (uint64_t)in_dim, (uint64_t)out0, (uint64_t)out1, (uint64_t)blocks, 1};
    const float ms_base = time_loop(stream, e0, e1, iters, pair_launch, &c);
    std::vector<float> b0(out0), b1(out1);
    CK(cudaMemcpy(b0.data(), dOut0, sizeof(float) * out0, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b1.data(), dOut1, sizeof(float) * out1, cudaMemcpyDeviceToHost));
    c.base = 0;
    const float ms_al = time_loop(stream, e0, e1, iters, pair_launch, &c);
    std::vector<float> a0(out0), a1(out1);
    CK(cudaMemcpy(a0.data(), dOut0, sizeof(float) * out0, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(a1.data(), dOut1, sizeof(float) * out1, cudaMemcpyDeviceToHost));

    std::vector<double> ref0(out0), ref1(out1);
    for (int d = 0; d < out0; d++) ref0[d] = ref_row(&r0.W[(size_t)d * blocks * 34], XQ.data(), XS.data(), blocks);
    for (int d = 0; d < out1; d++) ref1[d] = ref_row(&r1.W[(size_t)d * blocks * 34], XQ.data(), XS.data(), blocks);

    const double gb = ((double)out0 + out1) * blocks * 34 / 1e9;
    printf("%-22s out0=%d out1=%d in=%d (%.1f MB/call, iters=%d)\n", label, out0, out1, in_dim, gb * 1000, iters);
    printf("  baseline warp8     : %.4f ms -> %6.1f GB/s\n", ms_base, gb / (ms_base / 1e3));
    printf("  aligned  warp8     : %.4f ms -> %6.1f GB/s  (%+.1f%%)\n",
           ms_al, gb / (ms_al / 1e3), 100.0 * (ms_base / ms_al - 1.0));
    int bad = check_pair_tol(b0, a0, ref0, "out0");
    bad += check_pair_tol(b1, a1, ref1, "out1");

    CK(cudaFree(d0.dW)); CK(cudaFree(d0.dArt)); CK(cudaFree(d1.dW)); CK(cudaFree(d1.dArt));
    CK(cudaFree(dOut0)); CK(cudaFree(dOut1)); CK(cudaFree(dXq)); CK(cudaFree(dXs));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaStreamDestroy(stream));
    return bad;
}

struct GroupCtx {
    cudaStream_t stream; DevCopies dev; uint64_t dqb;
    float *low; const int8_t *xq; const float *xs;
    uint64_t group_dim, rank; uint32_t n_groups; uint64_t blocks; int base;
};
static void grouped_launch(int i, void *p) {
    GroupCtx *c = (GroupCtx *)p;
    const uint64_t low_dim = (uint64_t)c->n_groups * c->rank;
    const unsigned grid = ((unsigned)low_dim + 7u) / 8u;
    if (c->base) {
        grouped_base_kernel<<<grid, 256, 0, c->stream>>>(
            c->low, c->dev.w(i), c->xq, c->xs, c->group_dim, c->rank,
            c->n_groups, 1u, c->blocks, 1);
    } else {
        grouped_aligned_kernel<<<grid, 256, 0, c->stream>>>(
            c->low, c->dev.art_qs(i, c->dqb), c->dev.art_dq(i), c->xq, c->xs,
            c->group_dim, c->rank, c->n_groups, 1u, c->blocks);
    }
}

static int run_grouped(const char *label, int group_dim, int rank, int n_groups, int iters, uint32_t seed) {
    const int blocks = group_dim / 32;
    const long long low_dim = (long long)n_groups * rank;
    std::mt19937 rng(seed);
    RawArt r = make_weights(rng, low_dim, blocks);
    // per-group activation rows
    std::vector<float> X((size_t)n_groups * group_dim);
    for (auto &v : X) v = ((float)(rng() % 2000) - 1000.0f) / 500.0f;
    std::vector<int8_t> XQ(X.size()); std::vector<float> XS((size_t)n_groups * blocks);
    for (int g = 0; g < n_groups; g++)
        quantize_q8_0_host(&X[(size_t)g * group_dim], &XQ[(size_t)g * group_dim],
                           &XS[(size_t)g * blocks], group_dim);

    DevCopies dev = upload_rotating(r);
    float *dLow, *dXs; int8_t *dXq;
    CK(cudaMalloc(&dLow, sizeof(float) * low_dim));
    CK(cudaMalloc(&dXq, XQ.size()));
    CK(cudaMalloc(&dXs, sizeof(float) * XS.size()));
    CK(cudaMemcpy(dXq, XQ.data(), XQ.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dXs, XS.data(), sizeof(float) * XS.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    GroupCtx c = {stream, dev, r.dq_bytes, dLow, dXq, dXs,
                  (uint64_t)group_dim, (uint64_t)rank, (uint32_t)n_groups, (uint64_t)blocks, 1};
    const float ms_base = time_loop(stream, e0, e1, iters, grouped_launch, &c);
    std::vector<float> bl(low_dim);
    CK(cudaMemcpy(bl.data(), dLow, sizeof(float) * low_dim, cudaMemcpyDeviceToHost));
    c.base = 0;
    const float ms_al = time_loop(stream, e0, e1, iters, grouped_launch, &c);
    std::vector<float> al(low_dim);
    CK(cudaMemcpy(al.data(), dLow, sizeof(float) * low_dim, cudaMemcpyDeviceToHost));

    std::vector<double> ref(low_dim);
    for (long long d = 0; d < low_dim; d++) {
        const int g = (int)(d / rank);
        ref[d] = ref_row(&r.W[(size_t)d * blocks * 34], &XQ[(size_t)g * group_dim],
                         &XS[(size_t)g * blocks], blocks);
    }

    const double gb = (double)low_dim * blocks * 34 / 1e9;
    printf("%-22s gdim=%d rank=%d groups=%d (%.1f MB/call, iters=%d)\n",
           label, group_dim, rank, n_groups, gb * 1000, iters);
    printf("  baseline warp8     : %.4f ms -> %6.1f GB/s\n", ms_base, gb / (ms_base / 1e3));
    printf("  aligned  warp8     : %.4f ms -> %6.1f GB/s  (%+.1f%%)\n",
           ms_al, gb / (ms_al / 1e3), 100.0 * (ms_base / ms_al - 1.0));
    int bad = check_pair_tol(bl, al, ref, "low");

    CK(cudaFree(dev.dW)); CK(cudaFree(dev.dArt));
    CK(cudaFree(dLow)); CK(cudaFree(dXq)); CK(cudaFree(dXs));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaStreamDestroy(stream));
    return bad;
}

} // namespace

int main(int argc, char **argv) {
    const uint32_t seed = argc > 1 ? (uint32_t)atoi(argv[1]) : 1234u;
    printf("PROTO_Q8_WARP8 (Inc4: aligned-SoA variants of the custom warp8 q8_0 kernels)\n");
    int bad = 0;
    bad += run_hc_expand("hc_expand(output_b)", 4096, 8192, 1500, seed);
    bad += run_hc_expand("hc_expand(shr_down)", 4096, 2048, 2000, seed + 1);
    bad += run_pair("pair(q_a+kv)",       1024, 512, 4096, 2000, seed + 2);
    bad += run_pair("pair(shr_gate+up)",  2048, 2048, 4096, 2000, seed + 3);
    bad += run_grouped("grouped(output_a)", 4096, 1024, 8, 1500, seed + 4);
    printf(bad == 0 ? "ALL PASS\n" : "FAILURES: %d\n", bad);
    return bad == 0 ? 0 : 2;
}
