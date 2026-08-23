// kernel_qwen_gdn_core_rows4 is derived from llama.cpp's ggml-metal
// kernel_gated_delta_net_impl.
// Copyright (c) 2023-2026 The ggml authors. MIT licensed; see ../LICENSE.

#include <metal_stdlib>
using namespace metal;

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
    const uint n_layers = args.n_layers == 0u ? 64u : args.n_layers;
    const uint step_stride = n_layers * args.n_channels * QWEN_GDN_CONV_K;
    for (uint t = 0; t < ntok; t++) {
        const float x = qkv[t * args.n_channels + gid];
        h[0] = h[1];
        h[1] = h[2];
        h[2] = h[3];
        h[3] = x;
        float acc = w[0]*h[0] + w[1]*h[1] + w[2]*h[2] + w[3]*h[3];
        mixed[t * args.n_channels + gid] = acc / (1.0f + exp(-acc));
        if (args.snapshot && ntok > 1u) {
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

    const uint v_heads = args.v_heads == 0u ? 48u : args.v_heads;
    const uint qkv_dim = args.qkv_dim == 0u ? 10240u : args.qkv_dim;
    const uint z_dim = args.z_dim == 0u ? (v_heads * QWEN_GDN_HEAD_DIM) : args.z_dim;
    const uint n_layers = args.n_layers == 0u ? 64u : args.n_layers;

    const uint vh = tgpig.x;
    const uint j = lid;
    if (vh >= v_heads || j >= QWEN_GDN_HEAD_DIM) return;

    const uint kh = vh % QWEN_GDN_K_HEADS;
    threadgroup float tq[QWEN_GDN_HEAD_DIM];
    threadgroup float tk[QWEN_GDN_HEAD_DIM];
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint sg = lid / 32u;
    device float *Sj = state + ((uint)args.layer * v_heads + vh) *
                       QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM + j * QWEN_GDN_HEAD_DIM;
    float Srow[QWEN_GDN_HEAD_DIM];
    for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u) {
        float4 s = *(device float4 *)(Sj + i);
        Srow[i] = s.x; Srow[i + 1u] = s.y; Srow[i + 2u] = s.z; Srow[i + 3u] = s.w;
    }

    for (uint t = 0; t < ntok; t++) {
        const device float *mixed_t = mixed + t * qkv_dim;
        const device float *z_t = z + t * z_dim;
        const device float *alpha_t = alpha + t * v_heads;
        const device float *beta_t = beta + t * v_heads;
        device float *core_t = core + t * z_dim;

        const device float *qsrc = mixed_t + kh * QWEN_GDN_HEAD_DIM;
        const device float *ksrc = mixed_t + (QWEN_GDN_K_HEADS * QWEN_GDN_HEAD_DIM) + kh * QWEN_GDN_HEAD_DIM;
        tq[j] = qsrc[j];
        tk[j] = ksrc[j];

        if (args.pre_normalized) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
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
            tq[j] *= 1.0f / max(sqrt(qn), 1e-6f);
            tk[j] *= 1.0f / max(sqrt(kn), 1e-6f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float vj = mixed_t[(2u * QWEN_GDN_K_HEADS * QWEN_GDN_HEAD_DIM) + vh * QWEN_GDN_HEAD_DIM + j];
        float g = alpha_t[vh] + dt_bias[vh];
        if (g > 20.0f) {
        } else if (g < -20.0f) {
            g = exp(g);
        } else {
            g = log(1.0f + exp(g));
        }
        const float decay = exp(g * A_log[vh]);
        const float b = 1.0f / (1.0f + exp(-beta_t[vh]));

        float kv = 0.0f;
        for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u) {
            Srow[i] *= decay;
            Srow[i + 1u] *= decay;
            Srow[i + 2u] *= decay;
            Srow[i + 3u] *= decay;
            kv += Srow[i] * tk[i] + Srow[i + 1u] * tk[i + 1u] +
                  Srow[i + 2u] * tk[i + 2u] + Srow[i + 3u] * tk[i + 3u];
        }
        const float delta = (vj - kv) * b;
        float oh = 0.0f;
        const float qscale = 1.0f / sqrt((float)QWEN_GDN_HEAD_DIM);
        for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u) {
            Srow[i] += tk[i] * delta;
            Srow[i + 1u] += tk[i + 1u] * delta;
            Srow[i + 2u] += tk[i + 2u] * delta;
            Srow[i + 3u] += tk[i + 3u] * delta;
            oh += Srow[i] * tq[i] + Srow[i + 1u] * tq[i + 1u] +
                  Srow[i + 2u] * tq[i + 2u] + Srow[i + 3u] * tq[i + 3u];
        }
        oh *= qscale;
        if (args.split_output) {
            core_t[vh * QWEN_GDN_HEAD_DIM + j] = oh;
        } else {
            float ss = oh * oh;
            ss += simd_shuffle_xor(ss, 1u);
            ss += simd_shuffle_xor(ss, 2u);
            ss += simd_shuffle_xor(ss, 4u);
            ss += simd_shuffle_xor(ss, 8u);
            ss += simd_shuffle_xor(ss, 16u);
            threadgroup float ss_sg[4];
            if ((lid & 31u) == 0u) ss_sg[sg] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ss = (ss_sg[0] + ss_sg[1] + ss_sg[2] + ss_sg[3]) /
                 (float)QWEN_GDN_HEAD_DIM;
            const float nscale = 1.0f / sqrt(ss + 1e-6f);
            const float zj = z_t[vh * QWEN_GDN_HEAD_DIM + j];
            core_t[vh * QWEN_GDN_HEAD_DIM + j] =
                oh * nscale * snorm[j] * (zj / (1.0f + exp(-zj)));
        }
        if (args.snapshot && ntok > 1u) {
            const uint state_off = ((uint)args.layer * v_heads + vh) *
                                   QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM + j * QWEN_GDN_HEAD_DIM;
            const uint step_stride = n_layers * v_heads * QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM;
            device float *dst = state_steps + t * step_stride + state_off;
            for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u)
                *(device float4 *)(dst + i) = float4(Srow[i], Srow[i + 1u], Srow[i + 2u], Srow[i + 3u]);
        }



        if (t + 1u < ntok) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = 0; i < QWEN_GDN_HEAD_DIM; i += 4u)
        *(device float4 *)(Sj + i) = float4(Srow[i], Srow[i + 1u], Srow[i + 2u], Srow[i + 3u]);
}
// llama.cpp-style recurrent geometry: one 4-simdgroup threadgroup owns four
// state rows, and each lane keeps only four contiguous state columns.
// Q/K are pre-normalized by kernel_qwen_gdn_qk_l2; output normalization is
// handled by kernel_qwen_gdn_output_norm.
kernel void kernel_qwen_gdn_core_rows4(
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
        uint3 group [[threadgroup_position_in_grid]],
        uint3 tid [[thread_position_in_threadgroup]]) {
    (void)z;
    (void)snorm;
    const uint v_heads = args.v_heads == 0u ? 48u : args.v_heads;
    const uint qkv_dim = args.qkv_dim == 0u ? 10240u : args.qkv_dim;
    const uint z_dim = args.z_dim == 0u
        ? v_heads * QWEN_GDN_HEAD_DIM : args.z_dim;
    const uint n_layers = args.n_layers == 0u ? 64u : args.n_layers;
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint vh = group.y;
    const uint j = group.x * 4u + tid.y;
    const uint lane = tid.x;
    if (vh >= v_heads || j >= QWEN_GDN_HEAD_DIM || lane >= 32u ||
        tid.y >= 4u) return;

    const uint kh = vh % QWEN_GDN_K_HEADS;
    device float *Sj = state + ((uint)args.layer * v_heads + vh) *
        QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM + j * QWEN_GDN_HEAD_DIM;
    const uint i = lane * 4u;
    float4 s = *(device float4 *)(Sj + i);
    const float A = A_log[vh];
    const float dt = dt_bias[vh];
    const float qscale = 1.0f / sqrt((float)QWEN_GDN_HEAD_DIM);

    for (uint t = 0; t < ntok; t++) {
        const device float *mixed_t = mixed + t * qkv_dim;
        const device float *q = mixed_t + kh * QWEN_GDN_HEAD_DIM;
        const device float *k = mixed_t +
            QWEN_GDN_K_HEADS * QWEN_GDN_HEAD_DIM + kh * QWEN_GDN_HEAD_DIM;
        const float4 q4 = *(device const float4 *)(q + i);
        const float4 k4 = *(device const float4 *)(k + i);
        const float vj = mixed_t[
            2u * QWEN_GDN_K_HEADS * QWEN_GDN_HEAD_DIM +
            vh * QWEN_GDN_HEAD_DIM + j];

        float g = alpha[t * v_heads + vh] + dt;
        if (g > 20.0f) {
        } else if (g < -20.0f) {
            g = exp(g);
        } else {
            g = log(1.0f + exp(g));
        }
        s *= exp(g * A);
        const float sk = simd_sum(dot(s, k4));
        const float b = 1.0f /
            (1.0f + exp(-beta[t * v_heads + vh]));
        s += k4 * ((vj - sk) * b);
        const float oh = simd_sum(dot(s, q4)) * qscale;
        if (lane == 0u) {
            core[t * z_dim + vh * QWEN_GDN_HEAD_DIM + j] = oh;
        }

        if (args.snapshot && ntok > 1u) {
            const uint state_off = ((uint)args.layer * v_heads + vh) *
                QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM +
                j * QWEN_GDN_HEAD_DIM;
            const uint step_stride = n_layers * v_heads *
                QWEN_GDN_HEAD_DIM * QWEN_GDN_HEAD_DIM;
            device float *dst = state_steps + t * step_stride + state_off;
            *(device float4 *)(dst + i) = s;
        }
    }
    *(device float4 *)(Sj + i) = s;
}

kernel void kernel_qwen_gdn_qk_l2(
        constant ds4_qwen_gdn_core_args & args,
        device float * mixed,
        uint group [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint qkv_dim = args.qkv_dim == 0u ? 10240u : args.qkv_dim;
    const uint t = group / QWEN_GDN_K_HEADS;
    const uint kh = group % QWEN_GDN_K_HEADS;
    if (t >= ntok || lid >= QWEN_GDN_HEAD_DIM) return;
    device float *q = mixed + t * qkv_dim + kh * QWEN_GDN_HEAD_DIM;
    device float *k = q + QWEN_GDN_K_HEADS * QWEN_GDN_HEAD_DIM;
    const float qv = q[lid];
    const float kv = k[lid];
    float qn = qv * qv;
    float kn = kv * kv;
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
    threadgroup float qsum[4];
    threadgroup float ksum[4];
    const uint sg = lid / 32u;
    if ((lid & 31u) == 0u) {
        qsum[sg] = qn;
        ksum[sg] = kn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qn = qsum[0] + qsum[1] + qsum[2] + qsum[3];
    kn = ksum[0] + ksum[1] + ksum[2] + ksum[3];
    q[lid] = qv * (1.0f / max(sqrt(qn), 1e-6f));
    k[lid] = kv * (1.0f / max(sqrt(kn), 1e-6f));
}
kernel void kernel_qwen_gdn_output_norm(
        constant ds4_qwen_gdn_core_args & args,
        device float * core,
        device const float * z,
        device const float * snorm,
        uint group [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
    const uint v_heads = args.v_heads == 0u ? 48u : args.v_heads;
    const uint z_dim = args.z_dim == 0u
        ? v_heads * QWEN_GDN_HEAD_DIM : args.z_dim;
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint t = group / v_heads;
    const uint vh = group % v_heads;
    if (t >= ntok || lid >= QWEN_GDN_HEAD_DIM) return;
    device float *out = core + t * z_dim + vh * QWEN_GDN_HEAD_DIM;
    const float oh = out[lid];
    float ss = oh * oh;
    ss += simd_shuffle_xor(ss, 1u);
    ss += simd_shuffle_xor(ss, 2u);
    ss += simd_shuffle_xor(ss, 4u);
    ss += simd_shuffle_xor(ss, 8u);
    ss += simd_shuffle_xor(ss, 16u);
    threadgroup float sums[4];
    const uint sg = lid / 32u;
    if ((lid & 31u) == 0u) sums[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = (sums[0] + sums[1] + sums[2] + sums[3]) /
         (float)QWEN_GDN_HEAD_DIM;
    const float nscale = 1.0f / sqrt(ss + 1e-6f);
    const float zj = z[t * z_dim + vh * QWEN_GDN_HEAD_DIM + lid];
    out[lid] = oh * nscale * snorm[lid] *
               (zj / (1.0f + exp(-zj)));
}
/* ds4 keeps recurrent V heads grouped by their shared K head:
 * [v-slot][k-head][dim]. MLX's projection consumes [k-head][v-slot][dim].
 * Restore that original order before the unmodified MLX out_proj QMV so its
 * 64-value group accumulation order remains byte-identical. */
kernel void kernel_qwen_gdn_to_mlx_order(
        device const float *src,
        device float *dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= 48u * QWEN_GDN_HEAD_DIM) return;
    const uint new_head = gid / QWEN_GDN_HEAD_DIM;
    const uint d = gid % QWEN_GDN_HEAD_DIM;
    const uint v_slot = new_head / QWEN_GDN_K_HEADS;
    const uint k_head = new_head % QWEN_GDN_K_HEADS;
    const uint mlx_head = k_head * 3u + v_slot;
    dst[mlx_head * QWEN_GDN_HEAD_DIM + d] = src[gid];
}
kernel void kernel_qwen_gdn_mlx_output_norm(
        device float *core,
        device const float *z,
        device const float *snorm,
        uint vh [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    if (vh >= 48u || lane >= 32u) return;
    device float *x = core + vh * QWEN_GDN_HEAD_DIM;
    threadgroup float inv[1];
    float acc = 0.0f;
    for (uint i = 0; i < 4u; i++) {
        const float v = x[lane * 4u + i];
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane == 0u) inv[0] = precise::rsqrt(acc / 128.0f + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4u; i++) {
        const uint d = lane * 4u + i;
        const float rms = ds4_q4_64a_runtime_value(x[d] * inv[0]);
        const float normed = ds4_q4_64a_runtime_value(snorm[d] * rms);
        const float gate = z[vh * QWEN_GDN_HEAD_DIM + d];
        const float sigmoid_base = 1.0f /
            (1.0f + precise::exp(fabs(gate)));
        const float sigmoid = gate < 0.0f ? sigmoid_base : 1.0f - sigmoid_base;
        x[d] = ds4_q4_64a_runtime_value(normed * (gate * sigmoid));
    }
}


/* Chunked parallel prefill for the recurrent core (FLA/mamba2-style UT
 * transform).  The serial recurrence
 *     S_t = d_t*S_{t-1} + khat_t (x) delta_t,
 *     delta_t = b_t*(v_t - d_t*S_{t-1} khat_t)
 * is affine in the incoming state, so within a chunk of B tokens the
 * strictly-lower-triangular system
 *     delta_r + sum_{t<r} T[r][t]*delta_t = b_r*v_r - b_r*p_r*(S0 k_r),
 *     T[r][t] = b_r * (p_r/p_t) * <k_t, k_r>   (t < r),
 *     p_r = prod_{m<=r} d_m   (within-chunk cumulative decay, p_{-1} = 1),
 * solved by forward substitution yields both the chunk outputs and the
 * chunk-end state:
 *     oh_r  = (p_r*(S0 q_r) + sum_{t<=r} (p_r/p_t)*delta_t*<k_t,q_r>) / sqrt(D),
 *     S_end = p_R*S0 - sum_r (p_R/p_r)*z_r k_r^T + H_c,
 *     z     = (I+T)^{-1} ((b*p) (x) (S0 k)),
 *     H_c   = sum_r (p_R/p_r)*delta^0_r k_r^T   (zero-initial-state chunk end),
 * where delta^0 solves the same system with the S0-independent rhs b (x) v.
 * Three passes replace the serial token loop for long prefills:
 *   A  kernel_qwen_gdn_chunk_prep: per-(chunk,head) pairwise key dots, T
 *      build, delta^0 solve, H_c and per-token meta (p, b, p_R/p).
 *   B  kernel_qwen_gdn_state_scan: one threadgroup per head walks chunks
 *      sequentially, applying the inter-chunk transition and recording the
 *      state entering every chunk (S0) plus the final state.
 *   C  kernel_qwen_gdn_chunk_out: fully parallel per-token outputs from S0;
 *      writes raw oh (split_output layout) consumed by
 *      kernel_qwen_gdn_output_norm exactly like the serial kernel.
 * Numerics: fp32 accumulation throughout; the cumulative log-decay is
 * clamped at -80 so within-chunk decay ratios stay finite.  Relative to the
 * serial kernel this reassociates the dot/sum tree (expected relative drift
 * < 1e-3 on realistic gate distributions; grows only in the strong-decay
 * regime where magnitudes are negligible).  The softplus clamp branches
 * intentionally mirror kernel_qwen_gdn_core_rows4. */
struct ds4_qwen_gdn_chunk_args {
    uint32_t layer;
    uint32_t n_tok;
    uint32_t v_heads;
    uint32_t qkv_dim;
    uint32_t n_chunks;
    uint32_t _pad;
};

constant uint QWEN_GDN_CHUNK_TOK = 64u;

static inline void qwen_gdn_tri_decode(uint idx, thread uint & r, thread uint & t) {
    r = 0u;
    while (((r + 1u) * (r + 2u)) / 2u <= idx) r++;
    t = idx - (r * (r + 1u)) / 2u;
}

kernel void kernel_qwen_gdn_chunk_prep(
        constant ds4_qwen_gdn_chunk_args & args,
        device const float * mixed,
        device const float * alpha,
        device const float * beta,
        device const float * A_log,
        device const float * dt_bias,
        device float * chunk_H,
        device float * chunk_T,
        device float * chunk_meta,
        uint3 group [[threadgroup_position_in_grid]],
        uint  lid   [[thread_index_in_threadgroup]]) {
    const uint B = QWEN_GDN_CHUNK_TOK;
    const uint D = QWEN_GDN_HEAD_DIM;
    const uint v_heads = args.v_heads == 0u ? 48u : args.v_heads;
    const uint qkv_dim = args.qkv_dim == 0u ? 10240u : args.qkv_dim;
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint c = group.x;
    const uint vh = group.y;
    if (c >= args.n_chunks || vh >= v_heads || lid >= D) return;

    const uint kh = vh % QWEN_GDN_K_HEADS;
    const uint a = c * B;
    const uint len = min(B, ntok - a);
    const uint koff = QWEN_GDN_K_HEADS * D + kh * D;
    const float A = A_log[vh];
    const float dtb = dt_bias[vh];
    const ulong ch = (ulong)c * v_heads + vh;

    threadgroup float lgd[QWEN_GDN_CHUNK_TOK];
    threadgroup float p[QWEN_GDN_CHUNK_TOK];
    threadgroup float bb[QWEN_GDN_CHUNK_TOK];
    /* packed lower triangle including the diagonal: <k_t, k_r>, t <= r */
    threadgroup float gg[QWEN_GDN_CHUNK_TOK * (QWEN_GDN_CHUNK_TOK + 1u) / 2u];

    for (uint r = lid; r < B; r += D) {
        if (r < len) {
            const uint t = a + r;
            float g = alpha[t * v_heads + vh] + dtb;
            if (g > 20.0f) {
            } else if (g < -20.0f) {
                g = exp(g);
            } else {
                g = log(1.0f + exp(g));
            }
            lgd[r] = g * A;
            bb[r] = 1.0f / (1.0f + exp(-beta[t * v_heads + vh]));
        } else {
            lgd[r] = 0.0f;
            bb[r] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float acc = 0.0f;
        for (uint r = 0; r < B; r++) {
            acc += lgd[r];
            if (acc < -80.0f) acc = -80.0f;
            p[r] = exp(acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float pR = p[len - 1u];

    const uint npairs = B * (B + 1u) / 2u;
    for (uint idx = lid; idx < npairs; idx += D) {
        uint r, t;
        qwen_gdn_tri_decode(idx, r, t);
        float acc = 0.0f;
        if (r < len) {
            const device float *kr = mixed + (a + r) * qkv_dim + koff;
            const device float *kt = mixed + (a + t) * qkv_dim + koff;
            for (uint i = 0; i < D; i++) acc += kr[i] * kt[i];
        }
        gg[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define QWEN_GDN_G_AT(r_, t_) \
    gg[((t_) <= (r_) ? ((t_) * ((t_) + 1u)) / 2u + (r_) \
                     : ((r_) * ((r_) + 1u)) / 2u + (t_))]

    device float *mb = chunk_meta + ch * (3ul * B);
    for (uint r = lid; r < B; r += D) {
        mb[r] = p[r];
        mb[B + r] = bb[r];
        mb[2u * B + r] = r < len ? pR / p[r] : 0.0f;
    }

    device float *tb = chunk_T + ch * (ulong)B * B;
    const uint nstrict = B * (B - 1u) / 2u;
    for (uint idx = lid; idx < nstrict; idx += D) {
        uint r, t;
        qwen_gdn_tri_decode(idx + 1u, r, t);
        tb[r * B + t] = bb[r] * (p[r] / p[t]) * QWEN_GDN_G_AT(t, r);
    }

    /* forward substitution for delta^0 (S0-independent rhs) and the
       zero-state chunk-end state row H_c[lid][:] */
    device float *hb = chunk_H + (ch * D + lid) * D;
    float dz[QWEN_GDN_CHUNK_TOK];
    for (uint r = 0; r < len; r++) {
        const float vr = mixed[(a + r) * qkv_dim +
            2u * QWEN_GDN_K_HEADS * D + vh * D + lid];
        float d = bb[r] * vr;
        for (uint t = 0; t < r; t++)
            d -= bb[r] * (p[r] / p[t]) * QWEN_GDN_G_AT(t, r) * dz[t];
        dz[r] = d;
    }
    for (uint i = 0; i < D; i++) {
        float acc = 0.0f;
        for (uint r = 0; r < len; r++)
            acc += (pR / p[r]) * dz[r] *
                   mixed[(a + r) * qkv_dim + koff + i];
        hb[i] = acc;
    }
#undef QWEN_GDN_G_AT
}

kernel void kernel_qwen_gdn_state_scan(
        constant ds4_qwen_gdn_chunk_args & args,
        device const float * mixed,
        device float * state,
        device const float * chunk_H,
        device const float * chunk_T,
        device const float * chunk_meta,
        device float * chunk_s0,
        uint group [[threadgroup_position_in_grid]],
        uint  lid  [[thread_index_in_threadgroup]]) {
    const uint B = QWEN_GDN_CHUNK_TOK;
    const uint D = QWEN_GDN_HEAD_DIM;
    const uint v_heads = args.v_heads == 0u ? 48u : args.v_heads;
    const uint qkv_dim = args.qkv_dim == 0u ? 10240u : args.qkv_dim;
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint vh = group;
    if (vh >= v_heads || lid >= D) return;
    const uint kh = vh % QWEN_GDN_K_HEADS;
    const uint koff = QWEN_GDN_K_HEADS * D + kh * D;

    device float *srow = state +
        (((ulong)args.layer * v_heads + vh) * D + lid) * D;
    float s[QWEN_GDN_HEAD_DIM];
    for (uint i = 0; i < D; i += 4u) {
        const float4 v = *(device float4 *)(srow + i);
        s[i] = v.x; s[i + 1u] = v.y; s[i + 2u] = v.z; s[i + 3u] = v.w;
    }

    float z[QWEN_GDN_CHUNK_TOK];
    for (uint c = 0; c < args.n_chunks; c++) {
        const ulong ch = (ulong)c * v_heads + vh;
        const device float *mb = chunk_meta + ch * (3ul * B);
        const device float *tb = chunk_T + ch * (ulong)B * B;
        const device float *hb = chunk_H + (ch * D + lid) * D;
        const uint a = c * B;
        const uint len = min(B, ntok - a);
        const float pR = mb[len - 1u];

        /* record the state entering this chunk */
        device float *s0dst = chunk_s0 + (ch * D + lid) * D;
        for (uint i = 0; i < D; i += 4u)
            *(device float4 *)(s0dst + i) =
                float4(s[i], s[i + 1u], s[i + 2u], s[i + 3u]);

        /* z = (I+T)^{-1} ((b*p) (x) (S0 k)), one row per lane */
        for (uint r = 0; r < len; r++) {
            const device float *kr = mixed + (a + r) * qkv_dim + koff;
            float y = 0.0f;
            for (uint i = 0; i < D; i++) y += s[i] * kr[i];
            float zr = mb[r] * mb[B + r] * y;
            for (uint t = 0; t < r; t++) zr -= tb[r * B + t] * z[t];
            z[r] = zr;
        }

        /* S_end = p_R*S0 - sum_r (p_R/p_r) z_r k_r^T + H_c */
        for (uint i = 0; i < D; i++) {
            float acc = pR * s[i] + hb[i];
            for (uint r = 0; r < len; r++)
                acc -= mb[2u * B + r] * z[r] *
                       mixed[(a + r) * qkv_dim + koff + i];
            s[i] = acc;
        }
    }
    for (uint i = 0; i < D; i += 4u)
        *(device float4 *)(srow + i) =
            float4(s[i], s[i + 1u], s[i + 2u], s[i + 3u]);
}

kernel void kernel_qwen_gdn_chunk_out(
        constant ds4_qwen_gdn_chunk_args & args,
        device const float * mixed,
        device const float * chunk_s0,
        device const float * chunk_T,
        device const float * chunk_meta,
        device float * core,
        uint3 group [[threadgroup_position_in_grid]],
        uint  lid   [[thread_index_in_threadgroup]]) {
    const uint B = QWEN_GDN_CHUNK_TOK;
    const uint D = QWEN_GDN_HEAD_DIM;
    const uint v_heads = args.v_heads == 0u ? 48u : args.v_heads;
    const uint qkv_dim = args.qkv_dim == 0u ? 10240u : args.qkv_dim;
    const uint z_dim = v_heads * QWEN_GDN_HEAD_DIM;
    const uint ntok = args.n_tok == 0u ? 1u : args.n_tok;
    const uint c = group.x;
    const uint vh = group.y;
    if (c >= args.n_chunks || vh >= v_heads || lid >= D) return;

    const uint kh = vh % QWEN_GDN_K_HEADS;
    const uint a = c * B;
    const uint len = min(B, ntok - a);
    const uint qoff = kh * D;
    const uint koff = QWEN_GDN_K_HEADS * D + kh * D;
    const uint voff = 2u * QWEN_GDN_K_HEADS * D + vh * D;
    const ulong ch = (ulong)c * v_heads + vh;
    const device float *mb = chunk_meta + ch * (3ul * B);
    const device float *tb = chunk_T + ch * (ulong)B * B;
    const device float *s0 = chunk_s0 + (ch * D + lid) * D;
    const float qscale = 1.0f / sqrt((float)D);

    /* <k_t, q_r> for t <= r, packed lower triangle, shared per (chunk,head) */
    threadgroup float qg[QWEN_GDN_CHUNK_TOK * (QWEN_GDN_CHUNK_TOK + 1u) / 2u];
    const uint npairs = B * (B + 1u) / 2u;
    for (uint idx = lid; idx < npairs; idx += D) {
        uint r, t;
        qwen_gdn_tri_decode(idx, r, t);
        float acc = 0.0f;
        if (r < len) {
            const device float *kt = mixed + (a + t) * qkv_dim + koff;
            const device float *qr = mixed + (a + r) * qkv_dim + qoff;
            for (uint i = 0; i < D; i++) acc += kt[i] * qr[i];
        }
        qg[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* per-lane (value column j = lid): delta solve + outputs */
    float dl[QWEN_GDN_CHUNK_TOK];
    for (uint r = 0; r < len; r++) {
        const device float *kr = mixed + (a + r) * qkv_dim + koff;
        const device float *qr = mixed + (a + r) * qkv_dim + qoff;
        float y = 0.0f;
        float av = 0.0f;
        for (uint i = 0; i < D; i++) {
            y += s0[i] * kr[i];
            av += s0[i] * qr[i];
        }
        float d = mb[B + r] * mixed[(a + r) * qkv_dim + voff + lid] -
                  mb[r] * mb[B + r] * y;
        for (uint t = 0; t < r; t++) d -= tb[r * B + t] * dl[t];
        dl[r] = d;
        float e = 0.0f;
        for (uint t = 0; t <= r; t++)
            e += dl[t] * qg[(t * (t + 1u)) / 2u + r] / mb[t];
        core[(a + r) * z_dim + vh * D + lid] =
            mb[r] * (av + e) * qscale;
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
    if (h >= args.n_head || args.n_rot == 0u || (args.n_rot & 1u) != 0u ||
        args.n_rot > args.head_dim) return;
    const uint n_half = args.n_rot / 2u;
    const float theta_scale = pow(args.freq_base, -2.0f / (float)args.n_rot);
    device float *xh = x + h * args.head_dim;
    for (uint d = lid; d < n_half; d += 32u) {
        float theta = (float)args.pos;
        for (uint i = 0; i < d; i++) theta *= theta_scale;
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = xh[d];
        const float x1 = xh[d + n_half];
        xh[d] = x0 * c - x1 * s;
        xh[d + n_half] = x0 * s + x1 * c;
    }
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

static inline float qwen_attn_gate(float out, float gate) {
#if defined(DS4_Q4_64A_MLX_BF16)
    const float o = ds4_q4_64a_runtime_value(out);
    const float g = ds4_q4_64a_runtime_value(gate);
    const float exp_abs = ds4_q4_64a_runtime_value(precise::exp(fabs(g)));
    const float denom = ds4_q4_64a_runtime_value(1.0f + exp_abs);
    float sigmoid = ds4_q4_64a_runtime_value(1.0f / denom);
    if (g >= 0.0f) sigmoid = ds4_q4_64a_runtime_value(1.0f - sigmoid);
    return ds4_q4_64a_runtime_value(o * sigmoid);
#else
    return out * (1.0f / (1.0f + exp(-gate)));
#endif
}

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
    const uint lane = lid;
    const uint head_dim = args.head_dim;
    const uint nvec = (head_dim + 31u) / 32u;
    if (h >= args.n_head || lane >= 32u || head_dim == 0u || head_dim > 256u ||
        args.n_head_kv == 0u || (args.n_head % args.n_head_kv) != 0u ||
        args.pos >= args.cap) return;
    const uint kv_dim = args.n_head_kv * head_dim;
    const uint kv_h = h / (args.n_head / args.n_head_kv);
    const device float *base_k = k_cache + ((uint)args.layer * args.cap) * kv_dim;
    const device float *base_v = v_cache + ((uint)args.layer * args.cap) * kv_dim;
    const device float *qh = q + h * head_dim;
    float qreg[8];
    float acc[8];
    for (uint i = 0; i < 8u; i++) {
        const uint d = lane + i * 32u;
        qreg[i] = (i < nvec && d < head_dim) ? qh[d] : 0.0f;
        acc[i] = 0.0f;
    }
    const float scale = rsqrt((float)head_dim);
    float m_prev = -1.0e30f;
    float l_prev = 0.0f;
    for (uint t = 0; t <= args.pos; t++) {
        const device float *kt = base_k + t * kv_dim + kv_h * head_dim;
        float partial = 0.0f;
        for (uint i = 0; i < nvec; i++) {
            const uint d = lane + i * 32u;
            if (d < head_dim) partial += qreg[i] * kt[d];
        }
        const float score = simd_sum(partial) * scale;
        const float m_curr = max(m_prev, score);
        const float al = exp(m_prev - m_curr);
        const float be = exp(score - m_curr);
        l_prev = l_prev * al + be;
        const device float *vt = base_v + t * kv_dim + kv_h * head_dim;
        for (uint i = 0; i < nvec; i++) {
            const uint d = lane + i * 32u;
            if (d < head_dim) acc[i] = acc[i] * al + be * vt[d];
        }
        m_prev = m_curr;
    }
    const float inv = 1.0f / (l_prev > 1e-8f ? l_prev : 1.0f);
    device float *outp = heads + h * head_dim;
    for (uint i = 0; i < nvec; i++) {
        const uint d = lane + i * 32u;
        if (d >= head_dim) continue;
        float out = acc[i] * inv;
        if (args.gated) out = qwen_attn_gate(out, gate[h * head_dim + d]);
        outp[d] = out;
    }
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
    if (h >= args.n_head || t >= args.n_tok || args.n_rot == 0u ||
        (args.n_rot & 1u) != 0u || args.n_rot > args.head_dim) return;
    const uint n_half = args.n_rot / 2u;
    const float theta_scale = pow(args.freq_base, -2.0f / (float)args.n_rot);
    const float log2_theta_scale = log2(theta_scale);
    device float *xh = x + (uint64_t)t * args.n_head * args.head_dim + h * args.head_dim;
    for (uint d = lid; d < n_half; d += 32u) {
        const float theta = (float)(args.pos0 + t) * exp2((float)d * log2_theta_scale);
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = xh[d];
        const float x1 = xh[d + n_half];
        xh[d] = x0 * c - x1 * s;
        xh[d + n_half] = x0 * s + x1 * c;
    }
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
    const uint lane = lid;
    const uint head_dim = args.head_dim;
    const uint nvec = (head_dim + 31u) / 32u;
    if (h >= args.n_head || t >= args.n_tok || lane >= 32u ||
        head_dim == 0u || head_dim > 256u || args.n_head_kv == 0u ||
        (args.n_head % args.n_head_kv) != 0u || args.pos0 >= args.cap ||
        args.n_tok > args.cap - args.pos0) return;
    const uint kv_dim = args.n_head_kv * head_dim;
    const uint kv_h = h / (args.n_head / args.n_head_kv);
    const uint row_off = t * args.n_head * head_dim;
    const device float *base_k = k_cache + ((uint)args.layer * args.cap) * kv_dim;
    const device float *base_v = v_cache + ((uint)args.layer * args.cap) * kv_dim;
    const device float *qh = q + row_off + h * head_dim;
    float qreg[8];
    float acc[8];
    for (uint i = 0; i < 8u; i++) {
        const uint d = lane + i * 32u;
        qreg[i] = (i < nvec && d < head_dim) ? qh[d] : 0.0f;
        acc[i] = 0.0f;
    }
    const float scale = rsqrt((float)head_dim);
    const uint last = args.pos0 + t;
    float m_prev = -1.0e30f;
    float l_prev = 0.0f;
    for (uint p = 0; p <= last; p++) {
        const device float *kt = base_k + p * kv_dim + kv_h * head_dim;
        float partial = 0.0f;
        for (uint i = 0; i < nvec; i++) {
            const uint d = lane + i * 32u;
            if (d < head_dim) partial += qreg[i] * kt[d];
        }
        const float score = simd_sum(partial) * scale;
        const float m_curr = max(m_prev, score);
        const float al = exp(m_prev - m_curr);
        const float be = exp(score - m_curr);
        l_prev = l_prev * al + be;
        const device float *vt = base_v + p * kv_dim + kv_h * head_dim;
        for (uint i = 0; i < nvec; i++) {
            const uint d = lane + i * 32u;
            if (d < head_dim) acc[i] = acc[i] * al + be * vt[d];
        }
        m_prev = m_curr;
    }
    const float inv = 1.0f / (l_prev > 1e-8f ? l_prev : 1.0f);
    device float *outp = heads + row_off + h * head_dim;
    for (uint i = 0; i < nvec; i++) {
        const uint d = lane + i * 32u;
        if (d >= head_dim) continue;
        float out = acc[i] * inv;
        if (args.gated) out = qwen_attn_gate(out, gate[row_off + h * head_dim + d]);
        outp[d] = out;
    }
}

/* Shadow-backed tiled decode attention for qwen full-attn layers. Attends
 * the persistent F16 KV shadow maintained by ds4_gpu_qwen_attn_flash_rows
 * (ds4_metal.m) instead of rescanning the F32 cache serially per token.
 * Layout of one slot's shadow buffer: K region at offset 0, V region right
 * after; within a region head kv_h lives at kv_h * kv_head_stride half
 * elements (kv_head_stride = align_up(cap,64) * head_dim), token t at
 * t * head_dim, dims contiguous. The current token is NOT in the shadow yet:
 * it arrives via k_new/v_new and is written into the F32 cache here exactly
 * like the legacy kernel so later steps can convert it incrementally.
 *
 * Keys are processed in blocks of 128 tokens: each lane scores whole K rows
 * for its own tokens (q staged once in threadgroup memory), the block max /
 * sum come from two simd reductions, the accumulator is rescaled once per
 * block, and V accumulation reads each lane's 8-dim slice with the weight
 * broadcast from the owning lane. */
struct ds4_qwen_gqa_decode_shadow_args {
    uint32_t pos;             /* current token position; attends [0, pos] */
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint64_t kv_head_stride;  /* half elements per KV head plane */
};

#define QWEN_DECODE_SHADOW_BT 128u

kernel void kernel_qwen_gqa_attn_decode_shadow(
        constant ds4_qwen_gqa_decode_shadow_args & args,
        device const float * q,
        device const float * k_new,
        device const float * v_new,
        device float * k_cache,
        device float * v_cache,
        device const half * shadow_k,
        device const half * shadow_v,
        device float * heads_out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint ngrp = args.n_head / args.n_head_kv;
    const uint hd = args.head_dim;
    if (h >= args.n_head || lane >= 32u || ngrp == 0u || hd != 256u) return;
    const uint kv_h = h / ngrp;
    const uint kv_dim = args.n_head_kv * hd;

    /* Persist the new row into the F32 cache exactly like the legacy path.
     * k_new/v_new hold the whole KV row (n_head_kv * head_dim), so the
     * source needs the same kv_base offset as the destination: reading
     * k_new[i] would copy KV head 0 into every head's cache slot and
     * poison the F16 shadow from the next step onwards. */
    if (h % ngrp == 0u) {
        const uint kv_base = kv_h * hd;
        device float *kd = k_cache + (uint64_t)args.pos * kv_dim + kv_base;
        device float *vd = v_cache + (uint64_t)args.pos * kv_dim + kv_base;
        for (uint i = lane; i < hd; i += 32u) {
            kd[i] = k_new[kv_base + i];
            vd[i] = v_new[kv_base + i];
        }
    }

    threadgroup float tq[256];
    device const float *qh = q + (uint64_t)h * hd;
    for (uint i = lane; i < hd; i += 32u) tq[i] = qh[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half *kh =
        shadow_k + (uint64_t)kv_h * args.kv_head_stride;
    device const half *vh =
        shadow_v + (uint64_t)kv_h * args.kv_head_stride;

    const float scale = rsqrt((float)hd);
    float m_prev = -1.0e30f;
    float l_prev = 0.0f;
    float acc[8];
    for (uint j = 0; j < 8u; j++) acc[j] = 0.0f;

    for (uint t0 = 0; t0 < args.pos; t0 += QWEN_DECODE_SHADOW_BT) {
        const uint len = min(QWEN_DECODE_SHADOW_BT, args.pos - t0);
        float sloc[QWEN_DECODE_SHADOW_BT / 32u];
        for (uint i = 0; i < QWEN_DECODE_SHADOW_BT / 32u; i++)
            sloc[i] = -1.0e30f;
        /* Scores: every lane streams whole K rows for its own tokens. */
        for (uint i = 0; i < len; i += 32u) {
            const uint t = t0 + i + lane;   /* tail lanes: t >= pos keeps -1e30f */
            float dot = 0.0f;
            if (t < args.pos) {
                device const half *kt = kh + (uint64_t)t * hd;
                for (uint d = 0; d < hd; d += 4u) {
                    const half4 kv4 = *(device const half4 *)(kt + d);
                    dot += ((float)kv4.x * tq[d] +
                            (float)kv4.y * tq[d + 1] +
                            (float)kv4.z * tq[d + 2] +
                            (float)kv4.w * tq[d + 3]);
                }
                sloc[i / 32u] = dot * scale;
            }
            /* t >= pos (partial block tail): sloc keeps its -1e30f init,
             * contributing zero weight after the softmax shift. */
        }
        /* Block-wide online softmax update: two simd reductions per block
         * instead of one per token. */
        float m_b = sloc[0];
        for (uint i = 1; i < QWEN_DECODE_SHADOW_BT / 32u; i++)
            m_b = max(m_b, sloc[i]);
        m_b = simd_max(m_b);
        const float m_new = max(m_prev, m_b);
        const float alpha = exp(m_prev - m_new);
        float lsum = 0.0f;
        for (uint i = 0; i < QWEN_DECODE_SHADOW_BT / 32u; i++) {
            sloc[i] = exp(sloc[i] - m_new);
            lsum += sloc[i];
        }
        l_prev = l_prev * alpha + simd_sum(lsum);
        for (uint j = 0; j < 8u; j++) acc[j] *= alpha;
        /* V accumulation: weight broadcast from the owning lane. */
        for (uint i = 0; i < len; i++) {
            const float w = simd_shuffle(sloc[i >> 5], i & 31u);
            device const half *vt = vh + (uint64_t)(t0 + i) * hd;
            const uint base = lane * 8u;
            const half4 v0 = *(device const half4 *)(vt + base);
            const half4 v1 = *(device const half4 *)(vt + base + 4);
            acc[0] += w * (float)v0.x;
            acc[1] += w * (float)v0.y;
            acc[2] += w * (float)v0.z;
            acc[3] += w * (float)v0.w;
            acc[4] += w * (float)v1.x;
            acc[5] += w * (float)v1.y;
            acc[6] += w * (float)v1.z;
            acc[7] += w * (float)v1.w;
        }
        m_prev = m_new;
    }

    /* Current token: read directly from the new-K/V buffers. */
    {
        device const float *kt = k_new + (uint64_t)kv_h * hd;
        device const float *vt = v_new + (uint64_t)kv_h * hd;
        float qreg[8];
        for (uint j = 0; j < 8u; j++) qreg[j] = tq[lane * 8u + j];
        float partial = 0.0f;
        for (uint j = 0; j < 8u; j++)
            partial += qreg[j] * kt[lane * 8u + j];
        const float score = simd_sum(partial) * scale;
        const float m_curr = max(m_prev, score);
        const float al = exp(m_prev - m_curr);
        const float be = exp(score - m_curr);
        l_prev = l_prev * al + be;
        for (uint j = 0; j < 8u; j++)
            acc[j] = acc[j] * al + be * vt[lane * 8u + j];
        m_prev = m_curr;
    }

    const float inv = 1.0f / (l_prev > 1e-8f ? l_prev : 1.0f);
    device float *oh = heads_out + (uint64_t)h * hd;
    for (uint j = 0; j < 8u; j++)
        oh[lane * 8u + j] = acc[j] * inv;
}

kernel void kernel_qwen_causal_mask_f16(
        device half * mask,
        constant uint4 & args,
        uint2 gid [[thread_position_in_grid]]) {
    const uint cache_len = args.x;
    const uint n_tok = args.y;
    const uint pos0 = args.z;
    const uint q = gid.y;
    const uint k = gid.x;
    if (q >= n_tok || k >= cache_len) return;
    mask[(ulong)q * cache_len + k] = (k <= pos0 + q) ? half(0.0f) : half(-65504.0f);
}

kernel void kernel_qwen_attn_apply_gate_rows(
        device float * heads,
        device const float * gate,
        constant uint32_t & n,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    heads[gid] = qwen_attn_gate(heads[gid], gate[gid]);
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

struct ds4_qwen_q6k_gemm_args {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t _pad;
    uint64_t row_bytes;
};

/* Shared-weight Q6_K GEMM: dequant one output row once, dot all tokens. */
kernel void kernel_qwen_q6k_gemm(
        constant ds4_qwen_q6k_gemm_args & args,
        device const char * w,
        device const float * x,
        device float * out,
        uint tg [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    if (tg >= args.out_dim) return;
    const uint n_tok = args.n_tok;
    if (n_tok == 0u || n_tok > 64u || lane >= 32u) return;
    threadgroup float dq[256];
    threadgroup float acc[64];
    for (uint tok = lane; tok < n_tok; tok += 32u) acc[tok] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const char *row = w + (uint64_t)tg * args.row_bytes;
    const int nb = (int)args.in_dim / 256;
    const uint l = lane;
    for (int i = 0; i < nb; i++) {
        device const uchar *ql = (device const uchar *)(row + (uint)i * 210u);
        device const uchar *qh = ql + 128;
        device const char *scales = (device const char *)(qh + 64);
        const float d = float(*((device const half *)(scales + 16)));
        for (int n128 = 0; n128 < 256; n128 += 128) {
            const int is = (int)(l / 16u);
            const int q1 = ((int)(ql[l + 0]  & 0x0F) | (((int)qh[l] << 4) & 0x30)) - 32;
            const int q2 = ((int)(ql[l + 32] & 0x0F) | (((int)qh[l] << 2) & 0x30)) - 32;
            const int q3 = ((int)(ql[l + 0]  >> 4)    | (((int)qh[l] << 0) & 0x30)) - 32;
            const int q4 = ((int)(ql[l + 32] >> 4)    | (((int)qh[l] >> 2) & 0x30)) - 32;
            dq[n128 + l + 0]  = d * (float)scales[is + 0] * (float)q1;
            dq[n128 + l + 32] = d * (float)scales[is + 2] * (float)q2;
            dq[n128 + l + 64] = d * (float)scales[is + 4] * (float)q3;
            dq[n128 + l + 96] = d * (float)scales[is + 6] * (float)q4;
            ql += 64;
            qh += 32;
            scales += 8;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint tok = lane; tok < n_tok; tok += 32u) {
            device const float *yb = x + (uint64_t)tok * args.in_dim + (uint)i * 256u;
            float s = 0.0f;
            for (int k = 0; k < 256; k++) s += dq[k] * yb[k];
            acc[tok] += s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint tok = lane; tok < n_tok; tok += 32u)
        out[(uint64_t)tok * args.out_dim + tg] = acc[tok];
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
