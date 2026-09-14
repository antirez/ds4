/* Qwen3.8-Flash-Next vision tower glue around kernel_qwen4_dense_mm:
 * layernorm, bias/activation, 2D rope and the attention over one image.
 * Rows are patches in 2x2 merge-window order, so the merger is a plain
 * reshape and a row's grid position follows from its window index. */

struct ds4_metal_args_qwen4_vis {
    uint32_t rows;
    uint32_t width;
    uint32_t grid_w;     /* patch grid width (rope) */
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t mode;       /* bias_act: 0 tanh gelu, 1 erf gelu, 2 bias only */
    float    eps;
    float    scale;
};

/* x = a0 + a1 + bias + pos: both temporal conv taps see the same frame */
kernel void kernel_qwen4_vis_patch_finish(
        constant ds4_metal_args_qwen4_vis & args,
        device const float *a0,
        device const float *a1,
        device const float *bias,
        device const float *pos,
        device float       *x,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.rows * args.width) return;
    x[gid] = a0[gid] + a1[gid] + bias[gid % args.width] + pos[gid];
}

/* LayerNorm of one row per 256-thread threadgroup */
kernel void kernel_qwen4_vis_layernorm(
        constant ds4_metal_args_qwen4_vis & args,
        device const float *x,
        device const float *w,
        device const float *b,
        device float       *out,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (row >= args.rows) return;
    device const float *src = x + (uint64_t)row * args.width;
    threadgroup float red[8];
    float s = 0.0f;
    for (uint i = tid; i < args.width; i += 256) s += src[i];
    s = simd_sum(s);
    if (tiisg == 0) red[sgitg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = 0.0f;
    for (uint i = 0; i < 8; i++) s += red[i];
    const float mean = s / (float)args.width;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint i = tid; i < args.width; i += 256) { const float d = src[i] - mean; ss += d * d; }
    ss = simd_sum(ss);
    if (tiisg == 0) red[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (uint i = 0; i < 8; i++) ss += red[i];
    const float r = rsqrt(ss / (float)args.width + args.eps);
    device float *dst = out + (uint64_t)row * args.width;
    for (uint i = tid; i < args.width; i += 256) dst[i] = (src[i] - mean) * r * w[i] + b[i];
}

/* qkv rows (+bias) -> q, k, v [rows][n_head*head_dim].  q and k get the 2D
 * rope: pair (i, i + D/2) rotates by hpos*inv(i) for i < D/4 and by
 * wpos*inv(i - D/4) above, inv(j) = 10000^(-2j/(D/2)). */
kernel void kernel_qwen4_vis_qkv_rope(
        constant ds4_metal_args_qwen4_vis & args,
        device const float *qkv,
        device const float *bias,
        device float       *q,
        device float       *k,
        device float       *v,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= args.rows) return;
    const uint D = args.head_dim, H = args.n_head, W = H * D;
    const uint hd = D / 2, qd = D / 4;
    const uint blk = row / 4, within = row % 4, w2 = args.grid_w / 2;
    const float hpos = (float)((blk / w2) * 2 + within / 2);
    const float wpos = (float)((blk % w2) * 2 + within % 2);
    device const float *src = qkv + (uint64_t)row * 3 * W;
    const uint64_t o = (uint64_t)row * W;
    for (uint idx = tid; idx < 2 * H * hd; idx += 256) {
        const uint part = idx / (H * hd), rem = idx % (H * hd);
        const uint h = rem / hd, i = rem % hd;
        const uint base = part * W + h * D;
        const float x0 = src[base + i] + bias[base + i];
        const float x1 = src[base + i + hd] + bias[base + i + hd];
        const uint j = i < qd ? i : i - qd;
        const float theta = (i < qd ? hpos : wpos) * pow(10000.0f, -2.0f * (float)j / (float)hd);
        const float c = cos(theta), s = sin(theta);
        device float *dst = part == 0 ? q : k;
        dst[o + h * D + i] = x0 * c - x1 * s;
        dst[o + h * D + i + hd] = x0 * s + x1 * c;
    }
    for (uint i = tid; i < W; i += 256) v[o + i] = src[2 * W + i] + bias[2 * W + i];
}

/* Bidirectional attention over the image rows, one simdgroup per
 * (row, head); lanes hold dims lane, lane+32 and (head_dim > 64) lane+64. */
kernel void kernel_qwen4_vis_attention(
        constant ds4_metal_args_qwen4_vis & args,
        device const float *q,
        device const float *k,
        device const float *v,
        device float       *out,
        uint2 tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, head = tg.y;
    if (row >= args.rows) return;
    const uint D = args.head_dim, W = args.n_head * D;
    const bool has2 = D > 64 && lane < D - 64;
    const uint64_t qb = (uint64_t)row * W + head * D;
    const float q0 = q[qb + lane], q1 = q[qb + lane + 32], q2 = has2 ? q[qb + lane + 64] : 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, m = -INFINITY, denom = 0.0f;
    for (uint key = 0; key < args.rows; key++) {
        const uint64_t kb = (uint64_t)key * W + head * D;
        const float k2 = has2 ? k[kb + lane + 64] : 0.0f;
        const float score = simd_sum(q0 * k[kb + lane] + q1 * k[kb + lane + 32] + q2 * k2) * args.scale;
        const float nm = max(m, score);
        const float old = m == -INFINITY ? 0.0f : exp(m - nm);
        const float p = exp(score - nm);
        denom = denom * old + p;
        acc0 = acc0 * old + p * v[kb + lane];
        acc1 = acc1 * old + p * v[kb + lane + 32];
        acc2 = acc2 * old + p * (has2 ? v[kb + lane + 64] : 0.0f);
        m = nm;
    }
    out[qb + lane] = acc0 / denom;
    out[qb + lane + 32] = acc1 / denom;
    if (has2) out[qb + lane + 64] = acc2 / denom;
}

/* x += add + bias */
kernel void kernel_qwen4_vis_bias_residual(
        constant ds4_metal_args_qwen4_vis & args,
        device float       *x,
        device const float *add,
        device const float *bias,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.rows * args.width) return;
    x[gid] += add[gid] + bias[gid % args.width];
}

/* Abramowitz-Stegun 7.1.26 erf (|err| < 1.5e-7); Metal has no erf */
static inline float qwen4_vis_erf(float x) {
    const float sgn = x < 0.0f ? -1.0f : 1.0f;
    x = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t +
                            0.254829592f) * t * exp(-x * x);
    return sgn * y;
}

/* x = act(x + bias) */
kernel void kernel_qwen4_vis_bias_act(
        constant ds4_metal_args_qwen4_vis & args,
        device float       *x,
        device const float *bias,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.rows * args.width) return;
    float t = x[gid] + bias[gid % args.width];
    if (args.mode == 0) {
        /* fast-math tanh overflows past ~44; the result is saturated there anyway */
        const float u = clamp(0.7978845608f * (t + 0.044715f * t * t * t), -30.0f, 30.0f);
        t = 0.5f * t * (1.0f + tanh(u));
    } else if (args.mode == 1) {
        t = 0.5f * t * (1.0f + qwen4_vis_erf(t * 0.70710678f));
    }
    x[gid] = t;
}
