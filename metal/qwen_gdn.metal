#include <metal_stdlib>
using namespace metal;

constant uint QWEN_GDN_V_HEADS = 48u;
constant uint QWEN_GDN_K_HEADS = 16u;
constant uint QWEN_GDN_HEAD_DIM = 128u;
constant uint QWEN_GDN_CONV_K = 4u;

struct ds4_qwen_gdn_conv_args {
    uint32_t n_channels;
    uint32_t layer;
    uint32_t n_tok;
    uint32_t n_layers;
    uint32_t snapshot;
};

struct ds4_qwen_gdn_core_args {
    uint32_t layer;
    uint32_t n_tok;
    uint32_t v_heads;
    uint32_t qkv_dim;
    uint32_t z_dim;
    uint32_t n_layers;
    uint32_t snapshot;
    uint32_t pre_normalized;
    uint32_t split_output;
};


kernel void kernel_qwen_gdn_conv(
        constant ds4_qwen_gdn_conv_args & args,
        device const float * qkv,
        device const float * conv_w,
        device float * conv,
        device float * mixed,
        device float * conv_steps,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_channels) return;
    device float *h = conv + ((uint)args.layer * args.n_channels + gid) * QWEN_GDN_CONV_K;
    const device float *w = conv_w + gid * QWEN_GDN_CONV_K;
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint layer_off = ((uint)args.layer * args.n_channels + gid) * QWEN_GDN_CONV_K;
    const uint step_stride = 64u * args.n_channels * QWEN_GDN_CONV_K;
    for (uint t = 0; t < ntok; t++) {
        const float x = qkv[t * args.n_channels + gid];
        h[0] = h[1];
        h[1] = h[2];
        h[2] = h[3];
        h[3] = x;
        float acc = w[0]*h[0] + w[1]*h[1] + w[2]*h[2] + w[3]*h[3];
        mixed[t * args.n_channels + gid] = acc / (1.0f + exp(-acc));
        if (ntok > 1u) {
            device float *hs = conv_steps + t * step_stride + layer_off;
            hs[0] = h[0]; hs[1] = h[1]; hs[2] = h[2]; hs[3] = h[3];
        }

    }
}



kernel void kernel_qwen_gdn_core(
        constant ds4_qwen_gdn_core_args & args,
        device const float * mixed,
        device const float * z,
        device const float * alpha,
        device const float * beta,
        device const float * A_log,
        device const float * dt_bias,
        device const float * snorm,
        device float * state,
        device float * core,
        device float * state_steps,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  lid   [[thread_index_in_threadgroup]]) {

    const uint vh = tgpig.x;
    const uint j = lid;
    if (vh >= QWEN_GDN_V_HEADS || j >= QWEN_GDN_HEAD_DIM) return;

    const uint kh = vh % QWEN_GDN_K_HEADS;
    threadgroup float tq[QWEN_GDN_HEAD_DIM];
    threadgroup float tk[QWEN_GDN_HEAD_DIM];
    threadgroup float toh[QWEN_GDN_HEAD_DIM];
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint sg = lid / 32u;

    for (uint t = 0; t < ntok; t++) {
        const device float *mixed_t = mixed + t * 10240u;
        const device float *z_t = z + t * 6144u;
        const device float *alpha_t = alpha + t * 48u;
        const device float *beta_t = beta + t * 48u;
        device float *core_t = core + t * 6144u;

        const device float *qsrc = mixed_t + kh * QWEN_GDN_HEAD_DIM;
        const device float *ksrc = mixed_t + 2048u + kh * QWEN_GDN_HEAD_DIM;
        tq[j] = qsrc[j];
        tk[j] = ksrc[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qn = tq[j] * tq[j];
        float kn = tk[j] * tk[j];
        qn += simd_shuffle_xor(qn, 1u);
        qn += simd_shuffle_xor(qn, 2u);
        qn += simd_shuffle_xor(qn, 4u);
        qn += simd_shuffle_xor(qn, 8u);
        qn += simd_shuffle_xor(qn, 16u);
        kn += simd_shuffle_xor(kn, 1u);
        kn += simd_shuffle_xor(kn, 2u);
        kn += simd_shuffle_xor(kn, 4u);
        kn += simd_shuffle_xor(kn, 8u);
        kn += simd_shuffle_xor(kn, 16u);
        threadgroup float qn_sg[4];
        threadgroup float kn_sg[4];
        if ((lid & 31u) == 0u) {
            qn_sg[sg] = qn;
            kn_sg[sg] = kn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        qn = qn_sg[0] + qn_sg[1] + qn_sg[2] + qn_sg[3];
        kn = kn_sg[0] + kn_sg[1] + kn_sg[2] + kn_sg[3];
        qn = 1.0f / max(sqrt(qn), 1e-6f);
        kn = 1.0f / max(sqrt(kn), 1e-6f);
        tq[j] *= qn;
        tk[j] *= kn;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float vj = mixed_t[4096u + vh * QWEN_GDN_HEAD_DIM + j];
        float g = alpha_t[vh] + dt_bias[vh];
        if (g > 20.0f) {
        } else if (g < -20.0f) {
            g = exp(g);
        } else {
            g = log(1.0f + exp(g));
        }
        const float decay = exp(g * A_log[vh]);
        const float b = 1.0f / (1.0f + exp(-beta_t[vh]));

        device float *Sj = state + ((uint)args.layer * QWEN_GDN_V_HEADS + vh) *
                           QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM + j * QWEN_GDN_HEAD_DIM;
        float kv = 0.0f;
        for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u) {
            float4 s = *(device float4 *)(Sj + i);
            s *= decay;
            kv += s.x * tk[i] + s.y * tk[i + 1] + s.z * tk[i + 2] + s.w * tk[i + 3];
            *(device float4 *)(Sj + i) = s;
        }
        const float delta = (vj - kv) * b;
        float oh = 0.0f;
        const float qscale = 1.0f / sqrt((float)QWEN_GDN_HEAD_DIM);
        for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u) {
            float4 s = *(device float4 *)(Sj + i);
            s.x += tk[i] * delta;
            s.y += tk[i + 1] * delta;
            s.z += tk[i + 2] * delta;
            s.w += tk[i + 3] * delta;
            *(device float4 *)(Sj + i) = s;
            oh += s.x * tq[i] + s.y * tq[i + 1] + s.z * tq[i + 2] + s.w * tq[i + 3];
        }
        oh *= qscale;
        toh[j] = oh;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float ss = toh[j] * toh[j];
        ss += simd_shuffle_xor(ss, 1u);
        ss += simd_shuffle_xor(ss, 2u);
        ss += simd_shuffle_xor(ss, 4u);
        ss += simd_shuffle_xor(ss, 8u);
        ss += simd_shuffle_xor(ss, 16u);
        threadgroup float ss_sg[4];
        if ((lid & 31u) == 0u) ss_sg[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss = (ss_sg[0] + ss_sg[1] + ss_sg[2] + ss_sg[3]) / (float)QWEN_GDN_HEAD_DIM;
        const float nscale = 1.0f / sqrt(ss + 1e-6f);
        const float zj = z_t[vh * QWEN_GDN_HEAD_DIM + j];
        core_t[vh * QWEN_GDN_HEAD_DIM + j] = oh * nscale * snorm[j] * (zj / (1.0f + exp(-zj)));
        if (ntok > 1u) {
            const uint state_off = ((uint)args.layer * QWEN_GDN_V_HEADS + vh) *
                                   QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM + j * QWEN_GDN_HEAD_DIM;
            const uint step_stride = 64u * QWEN_GDN_V_HEADS * QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM;
            device float *dst = state_steps + t * step_stride + state_off;
            for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u)
                *(device float4 *)(dst + i) = *(device float4 *)(Sj + i);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

    }
}


struct ds4_qwen_split_q_args {
    uint32_t n_head;
    uint32_t head_dim;
};

kernel void kernel_qwen_split_gated_q(
        constant ds4_qwen_split_q_args & args,
        device const float * q_raw,
        device float * q,
        device float * gate,
        uint gid [[thread_position_in_grid]]) {
    const uint n = args.n_head * args.head_dim;
    if (gid >= n) return;
    const uint h = gid / args.head_dim;
    const uint d = gid % args.head_dim;
    q[gid] = q_raw[h * args.head_dim * 2u + d];
    gate[gid] = q_raw[h * args.head_dim * 2u + args.head_dim + d];
}

struct ds4_qwen_rope_args {
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos;
    float freq_base;
};

kernel void kernel_qwen_rope_rotate_half(
        constant ds4_qwen_rope_args & args,
        device float * x,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint n_half = args.n_rot / 2u;
    if (h >= args.n_head || lid >= n_half) return;
    const float theta_scale = pow(args.freq_base, -2.0f / (float)args.n_rot);
    float theta = (float)args.pos;
    for (uint i = 0; i < lid; i++) theta *= theta_scale;
    device float *xh = x + h * args.head_dim;
    const float c = cos(theta);
    const float s = sin(theta);
    const float x0 = xh[lid];
    const float x1 = xh[lid + n_half];
    xh[lid] = x0 * c - x1 * s;
    xh[lid + n_half] = x0 * s + x1 * c;
}

struct ds4_qwen_fullattn_args {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t pos;
    uint32_t layer;
    uint32_t cap;
    uint32_t gated;
};

kernel void kernel_qwen_attn_decode(
        constant ds4_qwen_fullattn_args & args,
        device const float * q,
        device const float * k_cache,
        device const float * v_cache,
        device const float * gate,
        device float * heads,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint d = lid;
    if (h >= args.n_head || d >= args.head_dim) return;
    const uint kv_dim = args.n_head_kv * args.head_dim;
    const uint kv_h = h / (args.n_head / args.n_head_kv);
    const device float *base_k = k_cache + ((uint)args.layer * args.cap) * kv_dim;
    const device float *base_v = v_cache + ((uint)args.layer * args.cap) * kv_dim;
    threadgroup float qh[256];
    qh[d] = q[h * args.head_dim + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = 1.0f / sqrt((float)args.head_dim);
    float m_prev = -1.0e30f;
    float l_prev = 0.0f;
    float acc = 0.0f;
    const uint sg = lid / 32u;
    for (uint t = 0; t <= args.pos; t++) {
        const device float *kt = base_k + t * kv_dim + kv_h * args.head_dim;
        float partial = qh[d] * kt[d];
        partial += simd_shuffle_xor(partial, 1u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 8u);
        partial += simd_shuffle_xor(partial, 16u);
        threadgroup float dot_sg[8];
        if ((lid & 31u) == 0u) dot_sg[sg] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (dot_sg[0] + dot_sg[1] + dot_sg[2] + dot_sg[3]
                     + dot_sg[4] + dot_sg[5] + dot_sg[6] + dot_sg[7]) * scale;
        const float m_curr = max(m_prev, score);
        const float al = exp(m_prev - m_curr);
        const float be = exp(score - m_curr);
        l_prev = l_prev * al + be;
        const device float *vt = base_v + t * kv_dim + kv_h * args.head_dim;
        acc = acc * al + be * vt[d];
        m_prev = m_curr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float out = acc / (l_prev > 1e-8f ? l_prev : 1.0f);
    if (args.gated) {
        const float g = gate[h * args.head_dim + d];
        out *= 1.0f / (1.0f + exp(-g));
    }

    heads[h * args.head_dim + d] = out;
}

/* Row-batched twins of the decode path: a speculative round verifies several
   rows at once, and one dispatch per layer beats one dispatch per row. */
struct ds4_qwen_split_q_rows_args {
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_tok;
};

kernel void kernel_qwen_split_gated_q_rows(
        constant ds4_qwen_split_q_rows_args & args,
        device const float * q_raw,
        device float * q,
        device float * gate,
        uint gid [[thread_position_in_grid]]) {
    const uint per_row = args.n_head * args.head_dim;
    if (gid >= per_row * args.n_tok) return;
    const uint t = gid / per_row;
    const uint idx = gid % per_row;
    const uint h = idx / args.head_dim;
    const uint d = idx % args.head_dim;
    const device float *src = q_raw + (uint64_t)t * per_row * 2u;
    q[gid] = src[h * args.head_dim * 2u + d];
    gate[gid] = src[h * args.head_dim * 2u + args.head_dim + d];
}

struct ds4_qwen_rope_rows_args {
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    float freq_base;
    uint32_t n_tok;
};

kernel void kernel_qwen_rope_rotate_half_rows(
        constant ds4_qwen_rope_rows_args & args,
        device float * x,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    const uint n_half = args.n_rot / 2u;
    if (h >= args.n_head || t >= args.n_tok || lid >= n_half) return;
    const float theta_scale = pow(args.freq_base, -2.0f / (float)args.n_rot);
    float theta = (float)(args.pos0 + t);
    for (uint i = 0; i < lid; i++) theta *= theta_scale;
    device float *xh = x + (uint64_t)t * args.n_head * args.head_dim + h * args.head_dim;
    const float c = cos(theta);
    const float s = sin(theta);
    const float x0 = xh[lid];
    const float x1 = xh[lid + n_half];
    xh[lid] = x0 * c - x1 * s;
    xh[lid + n_half] = x0 * s + x1 * c;
}

struct ds4_qwen_fullattn_rows_args {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t layer;
    uint32_t cap;
    uint32_t gated;
    uint32_t n_tok;
};

kernel void kernel_qwen_attn_decode_rows(
        constant ds4_qwen_fullattn_rows_args & args,
        device const float * q,
        device const float * k_cache,
        device const float * v_cache,
        device const float * gate,
        device float * heads,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    const uint d = lid;
    if (h >= args.n_head || t >= args.n_tok || d >= args.head_dim) return;
    const uint kv_dim = args.n_head_kv * args.head_dim;
    const uint kv_h = h / (args.n_head / args.n_head_kv);
    const uint row_off = t * args.n_head * args.head_dim;
    const device float *base_k = k_cache + ((uint)args.layer * args.cap) * kv_dim;
    const device float *base_v = v_cache + ((uint)args.layer * args.cap) * kv_dim;
    threadgroup float qh[256];
    qh[d] = q[row_off + h * args.head_dim + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = 1.0f / sqrt((float)args.head_dim);
    const uint last = args.pos0 + t;
    float m_prev = -1.0e30f;
    float l_prev = 0.0f;
    float acc = 0.0f;
    const uint sg = lid / 32u;
    for (uint p = 0; p <= last; p++) {
        const device float *kt = base_k + p * kv_dim + kv_h * args.head_dim;
        float partial = qh[d] * kt[d];
        partial += simd_shuffle_xor(partial, 1u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 8u);
        partial += simd_shuffle_xor(partial, 16u);
        threadgroup float dot_sg[8];
        if ((lid & 31u) == 0u) dot_sg[sg] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = (dot_sg[0] + dot_sg[1] + dot_sg[2] + dot_sg[3]
                     + dot_sg[4] + dot_sg[5] + dot_sg[6] + dot_sg[7]) * scale;
        const float m_curr = max(m_prev, score);
        const float al = exp(m_prev - m_curr);
        const float be = exp(score - m_curr);
        l_prev = l_prev * al + be;
        const device float *vt = base_v + p * kv_dim + kv_h * args.head_dim;
        acc = acc * al + be * vt[d];
        m_prev = m_curr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float out = acc / (l_prev > 1e-8f ? l_prev : 1.0f);
    if (args.gated) {
        const float g = gate[row_off + h * args.head_dim + d];
        out *= 1.0f / (1.0f + exp(-g));
    }
    heads[row_off + h * args.head_dim + d] = out;
}

struct ds4_qwen_q6k_args {
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_bytes;
};

kernel void kernel_qwen_q6k_matvec(
        constant ds4_qwen_q6k_args & args,
        device const char * w,
        device const float * x,
        device float * out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.out_dim) return;
    device const char *row = w + (uint64_t)gid * args.row_bytes;
    const int nb = (int)args.in_dim / 256;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        device const uchar *ql = (device const uchar *)(row + (uint)i * 210u);
        device const uchar *qh = ql + 128;
        device const char *scales = (device const char *)(qh + 64);
        const float d = float(*((device const half *)(scales + 16)));
        device const float *yb = x + (uint)i * 256;
        for (int n128 = 0; n128 < 256; n128 += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = ((int)(ql[l + 0]  & 0x0F) | (((int)qh[l] << 4) & 0x30)) - 32;
                const int q2 = ((int)(ql[l + 32] & 0x0F) | (((int)qh[l] << 2) & 0x30)) - 32;
                const int q3 = ((int)(ql[l + 0]  >> 4)    | (((int)qh[l] << 0) & 0x30)) - 32;
                const int q4 = ((int)(ql[l + 32] >> 4)    | (((int)qh[l] >> 2) & 0x30)) - 32;
                sumf += d * (float)scales[is + 0] * (float)q1 * yb[n128 + l + 0];
                sumf += d * (float)scales[is + 2] * (float)q2 * yb[n128 + l + 32];
                sumf += d * (float)scales[is + 4] * (float)q3 * yb[n128 + l + 64];
                sumf += d * (float)scales[is + 6] * (float)q4 * yb[n128 + l + 96];
            }
            ql += 64;
            qh += 32;
            scales += 8;
        }
    }
    out[gid] = sumf;
}

kernel void kernel_qwen_copy_f32(
        device const float * src,
        device float * dst,
        constant uint & n,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = src[gid];
}

struct ds4_affine2_matvec_args {
    uint32_t n_embd;
    uint32_t n_out;
    uint32_t q_row_bytes;
    uint32_t s_row_bytes;
    uint32_t b_row_bytes;
};

kernel void kernel_affine2_g64_matvec(
    constant ds4_affine2_matvec_args & args,
    device const char  * w_q,
    device const char  * w_scales,
    device const char  * w_biases,
    device const float * x,
    device float       * out,
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    const uint row = tgpig.x * 4 + sgitg;
    if (row >= args.n_out) return;

    device const uint32_t * q_row = (device const uint32_t *)(w_q + (uint64_t)row * args.q_row_bytes);
    device const half     * s_row = (device const half *)(w_scales + (uint64_t)row * args.s_row_bytes);
    device const half     * b_row = (device const half *)(w_biases + (uint64_t)row * args.b_row_bytes);

    const uint total_u32 = args.n_embd / 16;
    float thread_sum = 0.0f;

    for (uint u_idx = tiisg; u_idx < total_u32; u_idx += 32) {
        const uint g = u_idx / 4;
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);

        const uint32_t u = q_row[u_idx];
        device const float * x_ptr = x + u_idx * 16;

        const float4 x0 = *(device const float4 *)(x_ptr + 0);
        const float4 x1 = *(device const float4 *)(x_ptr + 4);
        const float4 x2 = *(device const float4 *)(x_ptr + 8);
        const float4 x3 = *(device const float4 *)(x_ptr + 12);

        const float4 q0 = float4(float(u & 3), float((u >> 2) & 3), float((u >> 4) & 3), float((u >> 6) & 3));
        const float4 q1 = float4(float((u >> 8) & 3), float((u >> 10) & 3), float((u >> 12) & 3), float((u >> 14) & 3));
        const float4 q2 = float4(float((u >> 16) & 3), float((u >> 18) & 3), float((u >> 20) & 3), float((u >> 22) & 3));
        const float4 q3 = float4(float((u >> 24) & 3), float((u >> 26) & 3), float((u >> 28) & 3), float((u >> 30) & 3));

        const float q_dot = dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3);
        const float x_dot = (x0.x + x0.y + x0.z + x0.w) +
                            (x1.x + x1.y + x1.z + x1.w) +
                            (x2.x + x2.y + x2.z + x2.w) +
                            (x3.x + x3.y + x3.z + x3.w);

        thread_sum += scale * q_dot + bias * x_dot;
    }

    const float total = simd_sum(thread_sum);
    if (tiisg == 0) {
        out[row] = total;
    }
}

