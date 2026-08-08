/* proto_m2_router.cu — M2-Inc3 prototype: fused router
 * (f16 split-K logits matmul + combine + top-6 select) as ONE cooperative
 * kernel, replacing the production 3-kernel chain
 *   matmul_f16_splitk_kernel (grid 256x8, 256 thr)
 * + matmul_f16_splitk_combine_kernel (1 blk, 256 thr)
 * + router_select_warp_topk_kernel (1 blk, 32x4)
 * at the router shape in_dim=4096, out_dim=256 (=n_expert), n_tok=1, ksplit=8.
 *
 * Gate: all four outputs (logits, probs, selected, weights) BIT-IDENTICAL to
 * the reference chain, in topk-bias / no-bias / hash modes, incl. exact-tie
 * logits; capture==eager bit-identity; then per-layer timing vs the chain.
 *
 * Bit-exactness argument for the matmul phase: the ref (row,kseg) block gives
 * each thread tid<64 exactly one 8-half chunk (seg 512 = 64 chunks); its
 * block_reduce_sum_256 reduces warp 0 (chunks 0..31) and warp 1 (chunks
 * 32..63) with warp_sum_f32, then the cross-warp tree only adds those two
 * warp sums and exact zeros.  The fused warp-per-tile form computes the SAME
 * chunk values at the SAME lanes, applies the SAME warp_sum_f32 trees, and
 * adds s0+s1 -- identical up to x+0.0f==x (breaks only for a partial that is
 * exactly -0.0f, i.e. an all-zero activation segment; unreachable for
 * rms-normed activations, and select-invariant even then).
 *
 * Build (GB10):
 *   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 proto_m2_router.cu -o proto_m2_router
 */
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define N_IN     4096u
#define N_EXPERT 256u
#define KSPLIT   8u
#define SEG      512u
#define N_TOP    6u

/* ------------------------------------------------------------------ */
/* Vendored production device functions + kernels (ds4_cuda.cu, verbatim) */

struct ds4_decode_scalars {
    uint32_t pos0;
    uint32_t raw_row;
    uint32_t raw_start;
    uint32_t n_raw;
    uint32_t n_comp;
    uint32_t emit_phase;
    uint32_t comp_row;
    uint32_t index_row;
    uint32_t flags;
    uint32_t token;
};

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
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

__global__ static void matmul_f16_splitk_kernel(
        float *partial,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t ksplit,
        int vec_ok) {
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
        float *out,
        const float *partial,
        uint64_t out_dim,
        uint32_t ksplit) {
    const uint64_t row = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (row >= out_dim) return;
    const float *p = partial + row * (uint64_t)ksplit;
    float s = 0.0f;
    for (uint32_t k = 0u; k < ksplit; k++) s += p[k];
    out[row] = s;
}

__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode,
        const struct ds4_decode_scalars *scalars) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs + (uint64_t)t * 256u;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[4][256];
    float local_prob[8];
    float local_score[8];

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t]
                                 : (scalars ? (int32_t)scalars->token : token_scalar);
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
    }
}

/* ------------------------------------------------------------------ */
/* FUSED: one cooperative kernel for the whole router stage. */

__global__ static void router_fused_coop_kernel(
        float *logits,            /* [256] */
        float *partials,          /* [256*8] dedicated scratch */
        int32_t *selected,        /* [6] */
        float *weights,           /* [6] */
        float *probs,             /* [256] */
        const __half *w,          /* [256][4096] f16 */
        const float *x,           /* [4096] */
        const float *bias,        /* [256] or NULL */
        const int32_t *hash,      /* [hash_rows][6] or NULL */
        const int32_t *tokens,    /* decode: NULL */
        int32_t token_scalar,
        uint32_t hash_rows,
        int has_bias,
        int hash_mode,
        const struct ds4_decode_scalars *scalars) {
    __shared__ float sbias[256];
    __shared__ float sprob[256];
    __shared__ int32_t shash[6];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;

    /* Phase 0 (block 0): prefetch the tail's small model-map reads so their
     * latency hides behind the matmul phase.  Verbatim copies -- the select
     * below consumes identical values. */
    if (blockIdx.x == 0u) {
        if (has_bias) sbias[threadIdx.x] = bias[threadIdx.x];
        if (hash_mode && threadIdx.x == 0u) {
            int32_t tok = tokens ? tokens[0]
                                 : (scalars ? (int32_t)scalars->token : token_scalar);
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) shash[j] = row[j];
        }
    }

    /* Phase 1: split-K logits matmul, one warp per (row, kseg) tile.
     * Chunk values, lane placement, and both warp_sum_f32 trees replicate the
     * ref block's warps 0/1 exactly; s0+s1 matches its cross-warp tree, which
     * otherwise only adds exact zeros. */
    const uint32_t nwarp = gridDim.x * (blockDim.x >> 5u);
    for (uint32_t t = blockIdx.x * (blockDim.x >> 5u) + warp;
         t < N_EXPERT * KSPLIT; t += nwarp) {
        const uint32_t row = t >> 3u;
        const uint32_t kseg = t & 7u;
        const uint64_t k0 = (uint64_t)kseg * SEG;
        const __half *wr = w + (uint64_t)row * N_IN;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        {
            const uint64_t i = k0 + (uint64_t)lane * 8u;
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(x + i);
            const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum0 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                  + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        {
            const uint64_t i = k0 + ((uint64_t)lane + 32u) * 8u;
            const uint4 wv = *reinterpret_cast<const uint4 *>(wr + i);
            const float4 xa = *reinterpret_cast<const float4 *>(x + i);
            const float4 xb = *reinterpret_cast<const float4 *>(x + i + 4);
            const __half2 *h = reinterpret_cast<const __half2 *>(&wv);
            const float2 w0 = __half22float2(h[0]);
            const float2 w1 = __half22float2(h[1]);
            const float2 w2 = __half22float2(h[2]);
            const float2 w3 = __half22float2(h[3]);
            sum1 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                  + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
        }
        const float s0 = warp_sum_f32(sum0);
        const float s1 = warp_sum_f32(sum1);
        if (lane == 0u) partials[(uint64_t)row * KSPLIT + kseg] = s0 + s1;
    }

    cooperative_groups::this_grid().sync();
    if (blockIdx.x != 0u) return;

    /* Phase 2: combine partials in fixed kseg order (verbatim combine kernel;
     * out_dim 256 == one 256-thread block). */
    {
        const uint64_t row = (uint64_t)threadIdx.x;
        const float *p = partials + row * KSPLIT;
        float s = 0.0f;
        for (uint32_t k = 0u; k < KSPLIT; k++) s += p[k];
        logits[row] = s;
    }
    __syncthreads();

    /* Phase 3: verbatim router_select_warp_topk_kernel warp body (t = 0,
     * row_in_block = 0); bias/hash reads swapped for the prefetched copies. */
    if (warp != 0u) return;
    const float *log = logits;
    float local_prob[8];
    float local_score[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? sbias[e] : 0.0f);
        sprob[e] = p;
        probs[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0u) {
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = shash[j];
                selected[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[(uint32_t)e] : 0.0f;
                weights[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) weights[j] = weights[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            selected[j] = (int32_t)out_idx[j];
            weights[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) weights[j] = weights[j] / sum * 1.5f;
    }
}

/* V2: identical numerics, but each warp owns TWO tiles per pass with all four
 * DRAM-latency w-vector loads hoisted before any math, doubling per-warp
 * memory-level parallelism (x reads stay L1-resident after warmup). */
__global__ static void router_fused_coop_kernel_v2(
        float *logits,
        float *partials,
        int32_t *selected,
        float *weights,
        float *probs,
        const __half *w,
        const float *x,
        const float *bias,
        const int32_t *hash,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        int has_bias,
        int hash_mode,
        const struct ds4_decode_scalars *scalars) {
    __shared__ float sbias[256];
    __shared__ float sprob[256];
    __shared__ int32_t shash[6];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;

    if (blockIdx.x == 0u) {
        if (has_bias) sbias[threadIdx.x] = bias[threadIdx.x];
        if (hash_mode && threadIdx.x == 0u) {
            int32_t tok = tokens ? tokens[0]
                                 : (scalars ? (int32_t)scalars->token : token_scalar);
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) shash[j] = row[j];
        }
    }

    const uint32_t nwarp = gridDim.x * (blockDim.x >> 5u);
    const uint32_t gwarp = blockIdx.x * (blockDim.x >> 5u) + warp;
    for (uint32_t t0 = gwarp; t0 < N_EXPERT * KSPLIT; t0 += 2u * nwarp) {
        const uint32_t t1 = t0 + nwarp;
        const int have1 = t1 < N_EXPERT * KSPLIT;
        const uint64_t k0a = (uint64_t)(t0 & 7u) * SEG;
        const __half *wra = w + (uint64_t)(t0 >> 3u) * N_IN;
        const uint64_t ia0 = k0a + (uint64_t)lane * 8u;
        const uint64_t ia1 = k0a + ((uint64_t)lane + 32u) * 8u;
        const uint64_t k0b = have1 ? (uint64_t)(t1 & 7u) * SEG : k0a;
        const __half *wrb = have1 ? w + (uint64_t)(t1 >> 3u) * N_IN : wra;
        const uint64_t ib0 = k0b + (uint64_t)lane * 8u;
        const uint64_t ib1 = k0b + ((uint64_t)lane + 32u) * 8u;
        /* all four w vectors in flight before any dependent math */
        const uint4 wva0 = *reinterpret_cast<const uint4 *>(wra + ia0);
        const uint4 wva1 = *reinterpret_cast<const uint4 *>(wra + ia1);
        const uint4 wvb0 = *reinterpret_cast<const uint4 *>(wrb + ib0);
        const uint4 wvb1 = *reinterpret_cast<const uint4 *>(wrb + ib1);
        float s0, s1;
        {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            {
                const float4 xa = *reinterpret_cast<const float4 *>(x + ia0);
                const float4 xb = *reinterpret_cast<const float4 *>(x + ia0 + 4);
                const __half2 *h = reinterpret_cast<const __half2 *>(&wva0);
                const float2 w0 = __half22float2(h[0]);
                const float2 w1 = __half22float2(h[1]);
                const float2 w2 = __half22float2(h[2]);
                const float2 w3 = __half22float2(h[3]);
                sum0 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                      + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
            }
            {
                const float4 xa = *reinterpret_cast<const float4 *>(x + ia1);
                const float4 xb = *reinterpret_cast<const float4 *>(x + ia1 + 4);
                const __half2 *h = reinterpret_cast<const __half2 *>(&wva1);
                const float2 w0 = __half22float2(h[0]);
                const float2 w1 = __half22float2(h[1]);
                const float2 w2 = __half22float2(h[2]);
                const float2 w3 = __half22float2(h[3]);
                sum1 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                      + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
            }
            s0 = warp_sum_f32(sum0);
            s1 = warp_sum_f32(sum1);
        }
        float s2, s3;
        if (have1) {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            {
                const float4 xa = *reinterpret_cast<const float4 *>(x + ib0);
                const float4 xb = *reinterpret_cast<const float4 *>(x + ib0 + 4);
                const __half2 *h = reinterpret_cast<const __half2 *>(&wvb0);
                const float2 w0 = __half22float2(h[0]);
                const float2 w1 = __half22float2(h[1]);
                const float2 w2 = __half22float2(h[2]);
                const float2 w3 = __half22float2(h[3]);
                sum0 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                      + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
            }
            {
                const float4 xa = *reinterpret_cast<const float4 *>(x + ib1);
                const float4 xb = *reinterpret_cast<const float4 *>(x + ib1 + 4);
                const __half2 *h = reinterpret_cast<const __half2 *>(&wvb1);
                const float2 w0 = __half22float2(h[0]);
                const float2 w1 = __half22float2(h[1]);
                const float2 w2 = __half22float2(h[2]);
                const float2 w3 = __half22float2(h[3]);
                sum1 += w0.x * xa.x + w0.y * xa.y + w1.x * xa.z + w1.y * xa.w
                      + w2.x * xb.x + w2.y * xb.y + w3.x * xb.z + w3.y * xb.w;
            }
            s2 = warp_sum_f32(sum0);
            s3 = warp_sum_f32(sum1);
        }
        if (lane == 0u) {
            partials[(uint64_t)(t0 >> 3u) * KSPLIT + (t0 & 7u)] = s0 + s1;
            if (have1) partials[(uint64_t)(t1 >> 3u) * KSPLIT + (t1 & 7u)] = s2 + s3;
        }
    }

    cooperative_groups::this_grid().sync();
    if (blockIdx.x != 0u) return;

    {
        const uint64_t row = (uint64_t)threadIdx.x;
        const float *p = partials + row * KSPLIT;
        float s = 0.0f;
        for (uint32_t k = 0u; k < KSPLIT; k++) s += p[k];
        logits[row] = s;
    }
    __syncthreads();

    if (warp != 0u) return;
    const float *log = logits;
    float local_prob[8];
    float local_score[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? sbias[e] : 0.0f);
        sprob[e] = p;
        probs[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0u) {
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = shash[j];
                selected[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[(uint32_t)e] : 0.0f;
                weights[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) weights[j] = weights[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            selected[j] = (int32_t)out_idx[j];
            weights[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) weights[j] = weights[j] / sum * 1.5f;
    }
}

/* ------------------------------------------------------------------ */
/* Host harness */

struct outs {
    float logits[N_EXPERT];
    float probs[N_EXPERT];
    int32_t sel[N_TOP];
    float wgt[N_TOP];
};

static void fill_rand(float *p, size_t n, unsigned seed, float scale) {
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 2.0f * scale;
    }
}

struct dev_bufs {
    __half *w;
    float *x, *bias, *logits, *probs, *partials_ref, *partials_fu, *wgt;
    int32_t *hash, *sel;
    ds4_decode_scalars *scalars;
    uint32_t hash_rows;
};

static void run_ref(const dev_bufs &b, int has_bias, int hash_mode, int32_t token_via_scalars) {
    dim3 sgrid(N_EXPERT, KSPLIT, 1);
    matmul_f16_splitk_kernel<<<sgrid, 256>>>(b.partials_ref, b.w, b.x, N_IN, N_EXPERT, KSPLIT, 1);
    matmul_f16_splitk_combine_kernel<<<1, 256>>>(b.logits, b.partials_ref, N_EXPERT, KSPLIT);
    dim3 block(32, 4, 1);
    router_select_warp_topk_kernel<<<1, block>>>(b.sel, b.wgt, b.probs,
            has_bias && !hash_mode ? b.bias : NULL,
            hash_mode ? b.hash : NULL,
            b.logits, NULL, /*token_scalar=*/-1, b.hash_rows, 1,
            has_bias && !hash_mode, hash_mode,
            token_via_scalars ? b.scalars : NULL);
    CHECK(cudaGetLastError());
}

static int launch_fused_k(void *kernel, const dev_bufs &b, int has_bias, int hash_mode,
                          int32_t token_via_scalars, unsigned grid_blocks, cudaStream_t stream) {
    float *logits = b.logits, *partials = b.partials_fu, *probs = b.probs, *wgt = b.wgt, *x = b.x;
    const __half *w = b.w;
    const float *bias = has_bias && !hash_mode ? b.bias : NULL;
    const int32_t *hash = hash_mode ? b.hash : NULL;
    int32_t *sel = b.sel;
    const int32_t *tokens = NULL;
    int32_t token_scalar = -1;
    uint32_t hash_rows = b.hash_rows;
    int hb = has_bias && !hash_mode;
    int hm = hash_mode;
    const ds4_decode_scalars *scalars = token_via_scalars ? b.scalars : NULL;
    void *args[] = {
        (void *)&logits, (void *)&partials, (void *)&sel, (void *)&wgt, (void *)&probs,
        (void *)&w, (void *)&x, (void *)&bias, (void *)&hash, (void *)&tokens,
        (void *)&token_scalar, (void *)&hash_rows, (void *)&hb, (void *)&hm, (void *)&scalars };
    cudaError_t err = cudaLaunchCooperativeKernel(
            kernel,
            dim3(grid_blocks, 1, 1), dim3(256, 1, 1), args, 0, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "coop launch failed (%u blk): %s\n", grid_blocks, cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

static int launch_fused(const dev_bufs &b, int has_bias, int hash_mode, int32_t token_via_scalars,
                        unsigned grid_blocks, cudaStream_t stream) {
    return launch_fused_k((void *)router_fused_coop_kernel, b, has_bias, hash_mode,
                          token_via_scalars, grid_blocks, stream);
}

static void read_outs(const dev_bufs &b, outs *o) {
    CHECK(cudaMemcpy(o->logits, b.logits, sizeof(o->logits), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(o->probs, b.probs, sizeof(o->probs), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(o->sel, b.sel, sizeof(o->sel), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(o->wgt, b.wgt, sizeof(o->wgt), cudaMemcpyDeviceToHost));
}

static int cmp_outs(const outs &a, const outs &bb, const char *tag) {
    int bad = 0;
    if (memcmp(a.logits, bb.logits, sizeof(a.logits))) { printf("  [%s] logits MISMATCH\n", tag); bad = 1; }
    if (memcmp(a.probs, bb.probs, sizeof(a.probs)))    { printf("  [%s] probs MISMATCH\n", tag); bad = 1; }
    if (memcmp(a.sel, bb.sel, sizeof(a.sel)))          { printf("  [%s] selected MISMATCH\n", tag); bad = 1; }
    if (memcmp(a.wgt, bb.wgt, sizeof(a.wgt)))          { printf("  [%s] weights MISMATCH\n", tag); bad = 1; }
    if (bad) {
        for (int i = 0; i < 8; i++) printf("    logit[%d] ref=%.9g fused=%.9g\n", i, a.logits[i], bb.logits[i]);
        for (int i = 0; i < 6; i++) printf("    sel[%d] ref=%d fused=%d  w ref=%.9g fused=%.9g\n",
                                           i, a.sel[i], bb.sel[i], a.wgt[i], bb.wgt[i]);
    }
    return bad;
}

int main() {
    int dev = 0, coop = 0, n_sm = 0, bps = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
    CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, router_fused_coop_kernel, 256, 0));
    const unsigned max_blocks = (unsigned)(bps * n_sm);
    printf("coop=%d n_sm=%d blocks/sm=%d max co-resident=%u\n", coop, n_sm, bps, max_blocks);
    if (!coop) { printf("NO cooperative launch support\n"); return 1; }

    dev_bufs b = {};
    b.hash_rows = 1000;
    CHECK(cudaMalloc(&b.w, (size_t)N_EXPERT * N_IN * sizeof(__half)));
    CHECK(cudaMalloc(&b.x, N_IN * sizeof(float)));
    CHECK(cudaMalloc(&b.bias, N_EXPERT * sizeof(float)));
    CHECK(cudaMalloc(&b.logits, N_EXPERT * sizeof(float)));
    CHECK(cudaMalloc(&b.probs, N_EXPERT * sizeof(float)));
    CHECK(cudaMalloc(&b.partials_ref, N_EXPERT * KSPLIT * sizeof(float)));
    CHECK(cudaMalloc(&b.partials_fu, N_EXPERT * KSPLIT * sizeof(float)));
    CHECK(cudaMalloc(&b.sel, N_TOP * sizeof(int32_t)));
    CHECK(cudaMalloc(&b.wgt, N_TOP * sizeof(float)));
    CHECK(cudaMalloc(&b.hash, (size_t)b.hash_rows * N_TOP * sizeof(int32_t)));
    CHECK(cudaMalloc(&b.scalars, sizeof(ds4_decode_scalars)));

    float *hw = (float *)malloc((size_t)N_EXPERT * N_IN * sizeof(float));
    __half *hwh = (__half *)malloc((size_t)N_EXPERT * N_IN * sizeof(__half));
    float *hx = (float *)malloc(N_IN * sizeof(float));
    float *hb = (float *)malloc(N_EXPERT * sizeof(float));
    int32_t *hh = (int32_t *)malloc((size_t)b.hash_rows * N_TOP * sizeof(int32_t));

    /* -------- parity sweep -------- */
    int total_bad = 0, n_cases = 0;
    const unsigned grids[] = { 48u, 96u, 128u, 192u, 256u };
    for (unsigned seed = 1; seed <= 10; seed++) {
        fill_rand(hw, (size_t)N_EXPERT * N_IN, seed * 7919u, 0.05f);
        if (seed == 3) {
            /* exact-tie case: duplicate rows -> equal logits (and equal bias) ->
             * tie-break by lower index must survive fusion */
            for (int r = 0; r < 6; r++)
                memcpy(hw + (size_t)(100 + r) * N_IN, hw + (size_t)(10 + r) * N_IN, N_IN * sizeof(float));
        }
        for (size_t i = 0; i < (size_t)N_EXPERT * N_IN; i++) hwh[i] = __float2half(hw[i]);
        fill_rand(hx, N_IN, seed * 104729u, 1.0f);
        if (seed == 5) for (size_t i = 0; i < N_IN; i++) hx[i] = 0.0f;  /* all-equal logits */
        fill_rand(hb, N_EXPERT, seed * 31u, 0.3f);
        if (seed == 3) for (int r = 0; r < 6; r++) hb[100 + r] = hb[10 + r];
        unsigned s = seed * 2654435761u;
        for (size_t i = 0; i < (size_t)b.hash_rows * N_TOP; i++) {
            s = s * 1664525u + 1013904223u;
            hh[i] = (int32_t)(s % 300u) - 20;  /* includes out-of-range ids */
        }
        CHECK(cudaMemcpy(b.w, hwh, (size_t)N_EXPERT * N_IN * sizeof(__half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(b.x, hx, N_IN * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(b.bias, hb, N_EXPERT * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(b.hash, hh, (size_t)b.hash_rows * N_TOP * sizeof(int32_t), cudaMemcpyHostToDevice));
        ds4_decode_scalars sc = {};
        sc.token = (seed * 37u) % 1200u;   /* sometimes >= hash_rows -> clamp path */
        CHECK(cudaMemcpy(b.scalars, &sc, sizeof(sc), cudaMemcpyHostToDevice));

        struct { int has_bias, hash_mode, via_scalars; const char *name; } cases[] = {
            { 1, 0, 0, "bias" }, { 0, 0, 0, "nobias" }, { 0, 1, 1, "hash" },
        };
        for (auto &c : cases) {
            outs oref, ofu;
            run_ref(b, c.has_bias, c.hash_mode, c.via_scalars);
            CHECK(cudaDeviceSynchronize());
            read_outs(b, &oref);
            for (unsigned g : grids) {
                if (g > max_blocks) continue;
                /* scrub outputs so a silent non-write is caught */
                CHECK(cudaMemset(b.logits, 0xCB, N_EXPERT * sizeof(float)));
                CHECK(cudaMemset(b.probs, 0xCB, N_EXPERT * sizeof(float)));
                CHECK(cudaMemset(b.sel, 0xCB, N_TOP * sizeof(int32_t)));
                CHECK(cudaMemset(b.wgt, 0xCB, N_TOP * sizeof(float)));
                if (!launch_fused(b, c.has_bias, c.hash_mode, c.via_scalars, g, 0)) return 1;
                CHECK(cudaDeviceSynchronize());
                read_outs(b, &ofu);
                char tag[64];
                snprintf(tag, sizeof(tag), "seed%u %s g%u", seed, c.name, g);
                total_bad += cmp_outs(oref, ofu, tag);
                n_cases++;
                CHECK(cudaMemset(b.logits, 0xCB, N_EXPERT * sizeof(float)));
                CHECK(cudaMemset(b.probs, 0xCB, N_EXPERT * sizeof(float)));
                CHECK(cudaMemset(b.sel, 0xCB, N_TOP * sizeof(int32_t)));
                CHECK(cudaMemset(b.wgt, 0xCB, N_TOP * sizeof(float)));
                if (!launch_fused_k((void *)router_fused_coop_kernel_v2,
                                    b, c.has_bias, c.hash_mode, c.via_scalars, g, 0)) return 1;
                CHECK(cudaDeviceSynchronize());
                read_outs(b, &ofu);
                snprintf(tag, sizeof(tag), "V2 seed%u %s g%u", seed, c.name, g);
                total_bad += cmp_outs(oref, ofu, tag);
                n_cases++;
            }
        }
    }
    printf("parity: %d/%d cases bit-identical\n", n_cases - total_bad, n_cases);
    if (total_bad) return 1;

    /* -------- capture/replay bit-identity -------- */
    {
        const unsigned g = max_blocks < 256u ? max_blocks : 256u;
        outs oeager, oreplay;
        if (!launch_fused(b, 1, 0, 0, g, 0)) return 1;
        CHECK(cudaDeviceSynchronize());
        read_outs(b, &oeager);
        cudaStream_t cs;
        CHECK(cudaStreamCreate(&cs));
        cudaGraph_t graph;
        cudaGraphExec_t gexec;
        CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        if (!launch_fused(b, 1, 0, 0, g, cs)) return 1;
        CHECK(cudaStreamEndCapture(cs, &graph));
        CHECK(cudaGraphInstantiate(&gexec, graph, NULL, NULL, 0));
        for (int r = 0; r < 3; r++) {
            CHECK(cudaMemset(b.logits, 0xCB, N_EXPERT * sizeof(float)));
            CHECK(cudaGraphLaunch(gexec, cs));
            CHECK(cudaStreamSynchronize(cs));
            read_outs(b, &oreplay);
            if (cmp_outs(oeager, oreplay, "capture-replay")) return 1;
        }
        printf("capture/replay: bit-identical (3 replays, %u blk)\n", g);
        CHECK(cudaGraphExecDestroy(gexec));
        CHECK(cudaGraphDestroy(graph));
        CHECK(cudaStreamDestroy(cs));
    }

    /* -------- timing: rotating 68-layer weight set (L2-defeated) -------- */
    {
        const int NL = 68;
        __half *wl[NL];
        for (int l = 0; l < NL; l++) {
            CHECK(cudaMalloc(&wl[l], (size_t)N_EXPERT * N_IN * sizeof(__half)));
            CHECK(cudaMemcpy(wl[l], b.w, (size_t)N_EXPERT * N_IN * sizeof(__half), cudaMemcpyDeviceToDevice));
        }
        const int ITERS = 3000, WARM = 300;
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        float ms;

        /* ref chain */
        dev_bufs bl = b;
        for (int it = 0; it < WARM; it++) { bl.w = wl[it % NL]; run_ref(bl, 1, 0, 0); }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(e0));
        for (int it = 0; it < ITERS; it++) { bl.w = wl[it % NL]; run_ref(bl, 1, 0, 0); }
        CHECK(cudaEventRecord(e1));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        printf("ref chain   : %8.3f us/layer\n", ms * 1000.0f / ITERS);

        int bps2 = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps2, router_fused_coop_kernel_v2, 256, 0));
        const unsigned max_blocks2 = (unsigned)(bps2 * n_sm);
        printf("V2 blocks/sm=%d max co-resident=%u\n", bps2, max_blocks2);
        struct { void *k; const char *name; unsigned cap; } variants[] = {
            { (void *)router_fused_coop_kernel, "V1", max_blocks },
            { (void *)router_fused_coop_kernel_v2, "V2", max_blocks2 },
        };
        for (auto &v : variants) {
            for (unsigned g : grids) {
                if (g > v.cap) continue;
                for (int it = 0; it < WARM; it++) { bl.w = wl[it % NL]; launch_fused_k(v.k, bl, 1, 0, 0, g, 0); }
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(e0));
                for (int it = 0; it < ITERS; it++) { bl.w = wl[it % NL]; launch_fused_k(v.k, bl, 1, 0, 0, g, 0); }
                CHECK(cudaEventRecord(e1));
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventElapsedTime(&ms, e0, e1));
                printf("fused %s g=%3u : %8.3f us/layer\n", v.name, g, ms * 1000.0f / ITERS);
            }
        }
        for (int l = 0; l < NL; l++) CHECK(cudaFree(wl[l]));
    }

    printf("PROTO_M2_ROUTER OK\n");
    return 0;
}
