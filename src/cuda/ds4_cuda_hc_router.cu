#include "ds4_cuda_internal.h"



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



/* Parallel hc4 split for the fused decode kernels: lanes 0..31 of warp 0
 * cooperate (guards inside; all lanes must enter for the __syncwarp()s).
 * BIT-IDENTICAL to hc4_split_one: rows (and columns) are normalized
 * INDEPENDENTLY in the serial version, so distributing one row/column per lane
 * with serial per-lane reductions preserves every element's exact float
 * operation sequence -- only genuinely independent work runs concurrently.
 * This removes the ~1k-cycle single-thread critical section (24 expf + 20
 * Sinkhorn iterations) that ran while 255 threads idled, x2 per layer per
 * decode token. c is a per-block shared[16] scratch. */
__device__ static void hc4_split_par(float *out, const float *mix, const float *scale,
                                     const float *base, uint32_t sinkhorn_iters, float epsv,
                                     uint32_t lane, float *c) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    if (lane < 4u) {
        float z = mix[lane] * pre_scale + base[lane];
        out[lane] = 1.0f / (1.0f + expf(-z)) + epsv;
    } else if (lane < 8u) {
        float z = mix[lane] * post_scale + base[lane];
        out[lane] = 2.0f / (1.0f + expf(-z));
    }
    if (lane < 4u) {
        const uint32_t r = lane;
        float v[4];
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            v[col] = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            m = fmaxf(m, v[col]);
        }
        float ss = 0.0f;
        for (int col = 0; col < 4; col++) {
            v[col] = expf(v[col] - m);
            ss += v[col];
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = v[col] / ss + epsv;
    }
    __syncwarp();
    if (lane < 4u) {
        const uint32_t col = lane;
        float ss = epsv;
        for (int r = 0; r < 4; r++) ss += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= ss;
    }
    __syncwarp();
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        if (lane < 4u) {
            const uint32_t r = lane;
            float ss = epsv;
            for (int col = 0; col < 4; col++) ss += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= ss;
        }
        __syncwarp();
        if (lane < 4u) {
            const uint32_t col = lane;
            float ss = epsv;
            for (int r = 0; r < 4; r++) ss += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= ss;
        }
        __syncwarp();
    }
    if (lane < 16u) out[8 + lane] = c[lane];
}


__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}



__global__ static void hc_weighted_sum_kernel(float *out, const ds4_hc_t *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += ds4_hc_load(x, (uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d) *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}



__global__ static void hc_expand_kernel(
        ds4_hc_t *out_hc,
        const float *block_out,
        const float *block_add,
        const ds4_hc_t *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = ds4_hc_load(residual_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d);
        acc += comb_v * res_v;
    }
    ds4_hc_store(out_hc, (uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d, acc);
}



__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const ds4_hc_t *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    __shared__ float hc4_c[16];
    if (d < 32u) hc4_split_par(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv, d, hc4_c);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += ds4_hc_load(residual_hc, (uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col) * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}



/* BIT-EXACT register-staging rewrite (2026-07-21).  At decode n_rows==1, so this
 * runs as ONE block on a 48-SM GPU and every stall is exposed.  The old shape
 * wrote `acc` to global `out`, then -- after a full block barrier -- read the
 * SAME value straight back to scale it, i.e. a read-after-write round trip
 * through memory for a value that was already in a register.  With BLK a
 * compile-time constant the per-thread column set {d, d+BLK, ...} is a
 * compile-time-bounded VEC, so the accs live in registers across the barrier
 * and the second pass loads nothing.
 *
 * BIT-EXACT: `v` reloaded from `out` was the f32 that `acc` already held (one
 * store + one load of the same 32-bit pattern -- no rounding in between), the
 * column order and the 256-partial pairwise tree are untouched, and `out` is
 * still written with the same values.  Only the reload disappears.
 *
 * `residual_hc` is an HC CARRIER (BF16 storage under task #62) and is read
 * through ds4_hc_load exactly as the generic kernel below does. */
template <uint32_t BLK, uint32_t VEC>
__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const ds4_hc_t *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    __shared__ float hc4_c[16];
    if (d < 32u) hc4_split_par(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv, d, hc4_c);
    __syncthreads();

    const uint64_t rbase = (uint64_t)t * 4u * n_embd;
    const uint64_t obase = (uint64_t)t * n_embd;
    float accs[VEC];
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t u = 0; u < VEC; u++) {
        const uint32_t col = d + u * BLK;
        if (col < n_embd) {
            float acc = 0.0f;
            #pragma unroll
            for (uint32_t h = 0; h < 4; h++) {
                acc += ds4_hc_load(residual_hc, rbase + (uint64_t)h * n_embd + col) * sp[h];
            }
            out[obase + col] = acc;
            accs[u] = acc;
            sum += acc * acc;
        } else {
            accs[u] = 0.0f;
        }
    }

    __shared__ float partial[BLK];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    #pragma unroll
    for (uint32_t u = 0; u < VEC; u++) {
        const uint32_t col = d + u * BLK;
        if (col < n_embd) norm_out[obase + col] = accs[u] * norm_scale * norm_w[col];
    }
}


/* Generic fallback for n_embd > BLK*VEC (byte-identical to the pre-2026-07-21
 * kernel: same column order, same tree, same global round trip). */
__global__ static void hc_split_weighted_sum_norm_fused_generic_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const ds4_hc_t *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    __shared__ float hc4_c[16];
    if (d < 32u) hc4_split_par(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv, d, hc4_c);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += ds4_hc_load(residual_hc, (uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col) * sp[h];
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



__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}



__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}



__global__ static void router_select_kernel(
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
        int hash_mode) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;

    for (int i = 0; i < 256; i++) prob[i] = sqrtf(softplus_dev(log[i]));

    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int i = 0; i < 6; i++) sel[i] = row[i];
    } else {
        for (int i = 0; i < 6; i++) sel[i] = -1;
        for (int i = 0; i < 256; i++) {
            float score = prob[i] + (has_bias ? bias[i] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > prob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        int e = sel[i];
        float v = (e >= 0 && e < 256) ? prob[e] : 0.0f;
        w[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int i = 0; i < 6; i++) w[i] = w[i] / sum * 1.5f;
}



__global__ static void router_select_parallel_kernel(
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
        int hash_mode) {
    uint32_t t = blockIdx.x;
    uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;
    __shared__ float sprob[256];

    const float p = sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0) return;
    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int j = 0; j < 6; j++) sel[j] = row[j];
    } else {
        for (int j = 0; j < 6; j++) sel[j] = -1;
        for (int e = 0; e < 256; e++) {
            float score = sprob[e] + (has_bias ? bias[e] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > sprob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < 6; j++) {
        int e = sel[j];
        float v = (e >= 0 && e < 256) ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < 6; j++) w[j] = w[j] / sum * 1.5f;
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
        int hash_mode) {
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
            int32_t tok = tokens ? tokens[t] : token_scalar;
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



__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}



__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}



__global__ static void directional_steering_project_kernel(
        float       *x,
        const float *directions,
        uint32_t     layer,
        uint32_t     width,
        uint32_t     rows,
        float        scale) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || width == 0) return;

    float *xr = x + (uint64_t)row * width;
    const float *dir = directions + (uint64_t)layer * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += xr[i] * dir[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    const float coeff = scale * partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        xr[i] -= coeff * dir[i];
    }
}


int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}


int ds4_gpu_shared_gate_up_swiglu_mxfp8_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_matmul_mxfp8_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_matmul_mxfp8_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}


int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}


int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}


int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (ok && has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) ok = 0;
        else bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) ok = 0;
        else hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) ok = 0;
    }
    /* Launch-path dispatch flags: read the environment once per process. */
    static const int no_warp_select = getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") != NULL;
    static const int no_parallel_select = getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") != NULL;
    if (ok) {
        if (!no_warp_select && !no_parallel_select) {
            dim3 block(32, 4, 1);
            router_select_warp_topk_kernel<<<1, block>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                         bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                         has_bias && !hash_mode, hash_mode);
        } else if (!no_parallel_select) {
            router_select_parallel_kernel<<<1, 256>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                      bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                      has_bias && !hash_mode, hash_mode);
        } else {
            router_select_kernel<<<1, 1>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                          bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                          has_bias && !hash_mode, hash_mode);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}


int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) return 0;
        hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    /* Launch-path dispatch flags: read the environment once per process. */
    static const int no_warp_select = getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") != NULL;
    static const int no_parallel_select = getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") != NULL;
    if (!no_warp_select && !no_parallel_select) {
        dim3 block(32, 4, 1);
        router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block>>>((int32_t *)selected->ptr,
                                                                        (float *)weights->ptr,
                                                                        (float *)probs->ptr,
                                                                        bias,
                                                                        hash,
                                                                        (const float *)logits->ptr,
                                                                        (const int32_t *)tokens->ptr,
                                                                        0,
                                                                        hash_rows,
                                                                        n_tokens,
                                                                        has_bias && !hash_mode,
                                                                        hash_mode);
    } else if (!no_parallel_select) {
        router_select_parallel_kernel<<<n_tokens, 256>>>((int32_t *)selected->ptr,
                                                         (float *)weights->ptr,
                                                         (float *)probs->ptr,
                                                         bias,
                                                         hash,
                                                         (const float *)logits->ptr,
                                                         (const int32_t *)tokens->ptr,
                                                         0,
                                                         hash_rows,
                                                         n_tokens,
                                                         has_bias && !hash_mode,
                                                         hash_mode);
    } else {
        router_select_kernel<<<n_tokens, 1>>>((int32_t *)selected->ptr,
                                              (float *)weights->ptr,
                                              (float *)probs->ptr,
                                              bias,
                                              hash,
                                              (const float *)logits->ptr,
                                              (const int32_t *)tokens->ptr,
                                              0,
                                              hash_rows,
                                              n_tokens,
                                              has_bias && !hash_mode,
                                              hash_mode);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}


int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (scale_offset > model_size || model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        mix->bytes < mix_bytes || out->bytes < mix_bytes) return 0;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    uint32_t n_rows = (uint32_t)(mix->bytes / mix_bytes);
    if (out->bytes / mix_bytes < n_rows) n_rows = (uint32_t)(out->bytes / mix_bytes);
    hc_split_sinkhorn_kernel<<<(n_rows + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)mix->ptr,
        scale,
        base,
        n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc_split_sinkhorn launch");
}


int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    /* n_tokens comes from the OUTPUT; bound the carrier input too, mirroring
     * ds4_gpu_hc_split_weighted_sum_tensor. The BF16 narrowing removed the
     * accidental 2x margin that made an over-read harmless (task #62). */
    if (residual_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * DS4_HC_ELT_SIZE ||
        weights->bytes < (uint64_t)n_tokens * n_hc * sizeof(float)) return 0;
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const ds4_hc_t *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}


int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    uint32_t stride = (uint32_t)(2u * n_hc + n_hc * n_hc);
    /* Bound the carrier input as well as the output (see the sibling above). */
    if (residual_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * DS4_HC_ELT_SIZE ||
        split->bytes < (uint64_t)n_tokens * stride * sizeof(float)) return 0;
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const ds4_hc_t *)residual_hc->ptr, (const float *)split->ptr,
        n_embd, n_hc, n_tokens, stride);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum_split launch");
}


int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * DS4_HC_ELT_SIZE;
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const ds4_hc_t *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}


int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * DS4_HC_ELT_SIZE;
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        norm_out->bytes < out->bytes ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset ||
        norm_weight_offset > model_size ||
        (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
        return 0;
    }
    const uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset,
            3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset,
            mix_bytes, "hc_base");
    const float *norm_w = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset,
            (uint64_t)n_embd * sizeof(float), "hc_norm_weight");
    if (!scale || !base || !norm_w) return 0;
#define DS4_HCFUSED_BLK 256u
#define DS4_HCFUSED_VEC 16u
    if (n_embd <= DS4_HCFUSED_BLK * DS4_HCFUSED_VEC) {
        hc_split_weighted_sum_norm_fused_kernel<DS4_HCFUSED_BLK, DS4_HCFUSED_VEC>
                <<<(uint32_t)n_rows, DS4_HCFUSED_BLK>>>(
                (float *)out->ptr,
                (float *)norm_out->ptr,
                (float *)split->ptr,
                (const float *)mix->ptr,
                (const ds4_hc_t *)residual_hc->ptr,
                scale,
                base,
                norm_w,
                n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
        return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
    }
    hc_split_weighted_sum_norm_fused_generic_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)norm_out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const ds4_hc_t *)residual_hc->ptr,
            scale,
            base,
            norm_w,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
}


int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}


int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * DS4_HC_ELT_SIZE));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((ds4_hc_t *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const ds4_hc_t *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}


int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * DS4_HC_ELT_SIZE));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((ds4_hc_t *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const ds4_hc_t *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}



int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * DS4_HC_ELT_SIZE));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((ds4_hc_t *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const ds4_hc_t *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}



int ds4_gpu_shared_down_hc_expand_mxfp8_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return ds4_gpu_matmul_mxfp8_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1) &&
           ds4_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
                                                residual_hc, split, n_embd, n_hc);
}



int ds4_gpu_matmul_fp8_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    static const int disable_fused = getenv("DS4_CUDA_DISABLE_FP8_HC_EXPAND_FUSED") != NULL;
    if (!disable_fused) {
        return cuda_matmul_fp8_hc_expand_tensor_labeled(out_hc, block_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        x,
                                                        NULL,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "fp8_hc_expand");
    }
    return ds4_gpu_matmul_mxfp8_tensor(block_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                            split, n_embd, n_hc);
}



/* ---------------------------------------------------------------------------
 * Fused plain-RMSNorm + f16 HC-mix GEMV (2026-07-21, BIT-EXACT).
 *
 * The decode graph ran, three times per token per layer-pair:
 *     rms_norm_plain_tensor(flat_hc, cur_hc, hc_dim)      // grid 1  x 256
 *     matmul_f16_tensor(hc_mix, hc_attn_fn, flat_hc, 1)   // grid 24 x 256
 * On a 48-SM GB10 that is one SM followed by 24 SMs, with a 64 KB f32 store and
 * a 24-way re-read of it in between, for 24 (or 4, at the output head) dot
 * products.  The nsys profile put the pair at ~5.4% of decode; the GEMV alone
 * moved 786 KB in 28.3 us = 27.8 GB/s, ~12% of achievable -- i.e. it was never
 * bandwidth, it was 8 warps on 24 SMs with no memory-level parallelism.
 *
 * This kernel keeps the SAME two reductions but runs the norm redundantly in
 * each of the out_dim blocks, so the serial 1-block stage disappears, the
 * 64 KB scratch round trip disappears (flat_hc has no other consumer -- every
 * call site feeds it straight into this GEMV), one launch disappears, and both
 * loops get compile-time strides so UNROLL loads are in flight per warp.
 *
 * `x` is an HC residual CARRIER (cur_hc / after_attn_hc), so it is read through
 * ds4_hc_load -- BF16 storage promoted to f32 -- matching rms_norm_plain_kernel
 * exactly (task #62).  Reading it as float* would be silent corruption.
 *
 * WHY IT IS BIT-EXACT, not merely close:
 *   - The sum-of-squares keeps thread t on exactly {t, t+BLK, ...} in
 *     increasing order and reuses the identical 256-entry pairwise tree, so
 *     `scale` is the same float rsqrtf produced before.
 *   - flat[i] was stored as f32 and reloaded; here x[i]*scale is consumed from
 *     the register that held that same f32.  Store/load of an f32 is the
 *     identity, so the multiplicand is bit-identical.
 *   - The dot product keeps the same 256 partials over the same index subsets
 *     in the same order and the same pairwise tree as matmul_f16_kernel.
 * There is no split-K, no atomic, and no cross-thread re-association anywhere.
 * ------------------------------------------------------------------------- */
template <uint32_t BLK, uint32_t UNROLL>
__global__ static void hc_norm_mix_f16_kernel(
        float *out,
        const __half *w,
        const ds4_hc_t *x,
        uint32_t n,
        uint32_t out_dim,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;
    const uint32_t tid = threadIdx.x;
    __shared__ float partial[BLK];

    /* stage 1: plain RMSNorm scale over x -- same order as rms_norm_plain_kernel */
    float sum = 0.0f;
    uint32_t i = tid;
    for (; i + (UNROLL - 1u) * BLK < n; i += BLK * UNROLL) {
        float v[UNROLL];
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) v[u] = ds4_hc_load(x, i + u * BLK);
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) sum += v[u] * v[u];
    }
    for (; i < n; i += BLK) {
        float v = ds4_hc_load(x, i);
        sum += v * v;
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    __syncthreads();   /* partial[] is reused below */

    /* stage 2: dot(w[row], normed x) -- same order as matmul_f16_kernel */
    const __half *wr = w + (uint64_t)row * n;
    float dot = 0.0f;
    i = tid;
    for (; i + (UNROLL - 1u) * BLK < n; i += BLK * UNROLL) {
        float xv[UNROLL];
        float wv[UNROLL];
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) xv[u] = ds4_hc_load(x, i + u * BLK);
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) wv[u] = __half2float(wr[i + u * BLK]);
        /* pinned roundings: __fmul_rn reproduces the f32 that rms_norm_plain
         * stored into flat_hc, __fmaf_rn reproduces matmul_f16_kernel's
         * contracted `sum += w * x`.  Written explicitly so no future
         * contraction/reassociation decision by nvcc can silently move this
         * off byte-identical. */
        #pragma unroll
        for (uint32_t u = 0; u < UNROLL; u++) dot = __fmaf_rn(wv[u], __fmul_rn(xv[u], scale), dot);
    }
    for (; i < n; i += BLK) {
        dot = __fmaf_rn(__half2float(wr[i]), __fmul_rn(ds4_hc_load(x, i), scale), dot);
    }
    partial[tid] = dot;
    __syncthreads();
    for (uint32_t stride = BLK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[row] = partial[0];
}


int ds4_gpu_hc_norm_mix_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   eps) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    /* x is an HC residual carrier: DS4_HC_ELT_SIZE bytes per sample. */
    if (x->bytes < in_dim * DS4_HC_ELT_SIZE || out->bytes < out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    hc_norm_mix_f16_kernel<256, 8><<<(uint32_t)out_dim, 256>>>(
            (float *)out->ptr, (const __half *)wptr, (const ds4_hc_t *)x->ptr,
            (uint32_t)in_dim, (uint32_t)out_dim, eps);
    return cuda_ok(cudaGetLastError(), "hc norm mix f16 launch");
}
