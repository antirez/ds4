#include <metal_stdlib>
using namespace metal;

struct ds4_dflash2_conv_args {
    uint32_t n_tok;
    uint32_t hidden;
    uint32_t ksize;
    uint32_t group;
    uint32_t tap;
};
kernel void kernel_dflash2_grouped_conv(
        constant ds4_dflash2_conv_args & args [[buffer(0)]],
        device const float *hidden [[buffer(1)]],
        device const float *dynamic [[buffer(2)]],
        device const float *base [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint32_t n = args.n_tok * args.hidden;
    if (gid >= n) return;
    const uint32_t t = gid / args.hidden;
    const uint32_t ch = gid % args.hidden;
    const uint32_t groups = args.hidden / args.group;
    const uint32_t g = ch / args.group;
    float acc = 0.0f;
    const uint32_t dyn_tok = args.ksize * groups;
    for (uint32_t off = 0; off < args.ksize; off++) {
        const int src = (int)t - (int)off;
        const float v = src >= 0 ? hidden[(uint32_t)src * args.hidden + ch] : 0.0f;
        const float kb = base[args.hidden * (off + args.ksize * args.tap) + ch];
        const float dyn = dynamic[(ulong)t * 2u * dyn_tok + (ulong)args.tap * dyn_tok +
                                  (ulong)off * groups + g];
        acc += (kb + dyn) * v;
    }
    out[gid] = acc;
}

struct ds4_dflash2_sdpa_args {
    uint32_t n_tok;
    uint32_t n_ctx;
    uint32_t n_head;
    uint32_t n_kv;
    uint32_t head_dim;
    uint32_t causal;
    uint32_t window;
};

kernel void kernel_dflash2_sdpa(
        constant ds4_dflash2_sdpa_args & args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k_ctx [[buffer(2)]],
        device const float *v_ctx [[buffer(3)]],
        device const float *k_prop [[buffer(4)]],
        device const float *v_prop [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint gid [[thread_position_in_grid]]) {
    const uint32_t n = args.n_tok * args.n_head;
    if (gid >= n) return;
    const uint32_t t = gid / args.n_head;
    const uint32_t h = gid % args.n_head;
    const uint32_t heads_per_kv = args.n_head / args.n_kv;
    const uint32_t kv_h = h / heads_per_kv;
    const float scale = 1.0f / sqrt((float)args.head_dim);
    device const float *qh = q + ((ulong)t * args.n_head + h) * args.head_dim;
    device float *oh = out + ((ulong)t * args.n_head + h) * args.head_dim;
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc[128];
    for (uint32_t d = 0; d < args.head_dim && d < 128u; d++) acc[d] = 0.0f;
    const uint32_t all_keys = args.n_ctx + args.n_tok;
    const uint32_t max_key = args.causal ? args.n_ctx + t + 1u : all_keys;
    const uint32_t min_key =
        args.window != 0u && max_key > args.window ? max_key - args.window : 0u;
    for (uint32_t s = min_key; s < max_key; s++) {
        device const float *kh;
        device const float *vh;
        if (s < args.n_ctx) {
            kh = k_ctx + ((ulong)s * args.n_kv + kv_h) * args.head_dim;
            vh = v_ctx + ((ulong)s * args.n_kv + kv_h) * args.head_dim;
        } else {
            kh = k_prop + ((ulong)(s - args.n_ctx) * args.n_kv + kv_h) * args.head_dim;
            vh = v_prop + ((ulong)(s - args.n_ctx) * args.n_kv + kv_h) * args.head_dim;
        }
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) dot += qh[d] * kh[d];
        dot *= scale;
        const float m_curr = m_prev > dot ? m_prev : dot;
        const float alpha = exp(m_prev - m_curr);
        const float beta = exp(dot - m_curr);
        l_prev = l_prev * alpha + beta;
        for (uint32_t d = 0; d < args.head_dim && d < 128u; d++) {
            acc[d] = acc[d] * alpha + beta * vh[d];
        }
        m_prev = m_curr;
    }
    const float inv_l = 1.0f / (l_prev > 1e-8f ? l_prev : 1.0f);
    for (uint32_t d = 0; d < args.head_dim && d < 128u; d++) oh[d] = acc[d] * inv_l;
}
