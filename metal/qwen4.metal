/* Qwen3.8-Flash-Next kernels: hyper-connection mix/combine, gated delta net
 * (conv, scan, gated norm), PLE n-gram gate + dilated conv, softmax top-k
 * router, QSA indexer, gated GQA attention, routed/dense expert matmuls and
 * the MTP input.  Transients are f32; the GDN state is f32 [head][dv][dk]
 * so each simdgroup lane owns a contiguous dk slice.  Math mirrors the
 * qwen4_ref_* scalar reference in ds4.c. */

static inline float qwen4_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

static inline float qwen4_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

static inline float qwen4_silu(float x) {
    return x * qwen4_sigmoid(x);
}

static inline float qwen4_row_dot(device const char *row, device const float *x,
                                  uint weight_type, uint in_dim, ushort tiisg);

/* --- hyper-connections -------------------------------------------------- */

struct ds4_metal_args_qwen4_hc_norm {
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t n_inject;     /* 0 or n_hc: inject rows dotted with this stream's xn */
    float    eps;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

#define QWEN4_HC_CHUNKS 8   /* threadgroups per stream; each recomputes the stream RMS */

/* element readers for the hc mixer weights: f16, f32 and q8_0 rows */
struct qwen4_w_f16 {
    device const half *p;
    qwen4_w_f16(device const char *base) : p((device const half *)base) {}
    float at(uint64_t i) const { return (float)p[i]; }
};
struct qwen4_w_f32 {
    device const float *p;
    qwen4_w_f32(device const char *base) : p((device const float *)base) {}
    float at(uint64_t i) const { return p[i]; }
};
struct qwen4_w_q8 {
    device const char *p;
    qwen4_w_q8(device const char *base) : p(base) {}
    float at(uint64_t i) const {
        device const char *b = p + (i >> 5) * 34;
        return (float)(*(device const half *)b) * (float)b[2 + (i & 31u)];
    }
};

/* Grouped RMSNorm of one (stream chunk, token): xn = R * rsqrt(mean(R^2)
 * + eps) * gamma over the chunk, plus the chunk's partial dot with each
 * inject row (consumers sum the hc*chunks partials and apply
 * 2*sigmoid(./hc)).  The low-rank projection itself is a GEMV on xn. */
template <typename W>
kernel void kernel_qwen4_hc_norm(
        constant ds4_metal_args_qwen4_hc_norm & args,
        device const float *R,          /* [T][hc*E] */
        device const float *gamma,      /* [hc*E] */
        device const char  *w_inject,   /* [n_inject][hc*E] */
        device float       *xn,         /* [T][hc*E] */
        device float       *inj_part,   /* [T][hc*chunks][n_inject] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint s = tgpig.x / QWEN4_HC_CHUNKS;
    const uint chunk = tgpig.x % QWEN4_HC_CHUNKS;
    const uint tok = tgpig.y;
    if (s >= args.n_hc || tok >= args.n_tokens) return;
    const uint E = args.n_embd, dim = E * args.n_hc;
    const uint nth = ntg.x, nsg = nth / 32;
    threadgroup float red[5][32];
    device const float *r = R + ((uint64_t)tok * args.n_hc + s) * E;
    device const float *g = gamma + s * E;
    device float *o = xn + ((uint64_t)tok * args.n_hc + s) * E;
    const W w(w_inject);
    float ss = 0.0f;
    for (uint i = tid; i < E; i += nth) ss += r[i] * r[i];
    ss = simd_sum(ss);
    if (tiisg == 0) red[0][sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint q = 0; q < nsg; q++) tot += red[0][q];
    const float inv = rsqrt(tot / (float)E + args.eps);
    const uint per = (E + QWEN4_HC_CHUNKS - 1) / QWEN4_HC_CHUNKS;
    const uint i0 = chunk * per, i1 = min(E, i0 + per);
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint i = i0 + tid; i < i1; i += nth) {
        const float v = r[i] * inv * g[i];
        o[i] = v;
        for (uint j = 0; j < 4; j++) {
            if (j < args.n_inject) acc[j] += w.at((uint64_t)j * dim + s * E + i) * v;
        }
    }
    for (uint j = 0; j < 4; j++) {
        if (j >= args.n_inject) break;
        const float a = simd_sum(acc[j]);
        if (tiisg == 0) red[1 + j][sgitg] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < args.n_inject) {
        float a = 0.0f;
        for (uint q = 0; q < nsg; q++) a += red[1 + tid][q];
        inj_part[((uint64_t)tok * args.n_hc * QWEN4_HC_CHUNKS + s * QWEN4_HC_CHUNKS + chunk) * args.n_inject + tid] = a;
    }
}

#define QWEN4_HC_NORM_INSTANCE(SUFFIX, W) \
template [[host_name("kernel_qwen4_hc_norm_" #SUFFIX)]] \
kernel void kernel_qwen4_hc_norm<W>(constant ds4_metal_args_qwen4_hc_norm &, device const float *, \
        device const float *, device const char *, device float *, device float *, uint3, ushort, ushort3, ushort, ushort);
QWEN4_HC_NORM_INSTANCE(f16, qwen4_w_f16)
QWEN4_HC_NORM_INSTANCE(f32, qwen4_w_f32)
QWEN4_HC_NORM_INSTANCE(q8, qwen4_w_q8)

/* 2*sigmoid(inj/hc) with inj[s] = sum of the hc*chunks norm partials for s */
static inline float qwen4_hc_inject_weight(device const float *inj_part, uint hc, uint s) {
    float a = 0.0f;
    for (uint src = 0; src < hc * QWEN4_HC_CHUNKS; src++) a += inj_part[src * hc + s];
    return 2.0f * qwen4_sigmoid(a / (float)hc);
}

struct ds4_metal_args_qwen4_hc_gate_mix {
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t n_rank;
};

/* mixed[d] = mean over streams of sigmoid(w_up[s*E+d] . silu(lo/hc)) *
 * xn[s*E+d], lo being the raw low-rank projection.  One simdgroup per d:
 * the four 8-lane groups stream the four stream rows (contiguous n_rank
 * weights each) and shuffle-reduce, first within the group, then across
 * the streams. */
template <typename W>
kernel void kernel_qwen4_hc_gate_mix(
        constant ds4_metal_args_qwen4_hc_gate_mix & args,
        device const float *xn,       /* [T][hc*E] */
        device const float *lo,       /* [T][n_rank] raw */
        device const char  *w_up,     /* [hc*E][n_rank] */
        device float       *mixed,    /* [T][E] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hc = args.n_hc;
    const uint E = args.n_embd;
    const uint nsg = ntg.x / 32;
    const uint d = tgpig.x * nsg + sgitg;
    const uint tok = tgpig.y;
    if (d >= E || tok >= args.n_tokens) return;
    const uint s = tiisg / 8, lane = tiisg % 8;
    device const float *l = lo + (uint64_t)tok * args.n_rank;
    const W w(w_up);
    const uint64_t row = (uint64_t)(s * E + d) * args.n_rank;
    float acc = 0.0f;
    for (uint r = lane; r < args.n_rank; r += 8) acc += w.at(row + r) * qwen4_silu(l[r] / (float)hc);
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    float g = qwen4_sigmoid(acc) * xn[(uint64_t)tok * E * hc + s * E + d];
    g += simd_shuffle_xor(g, 8);
    g += simd_shuffle_xor(g, 16);
    if (tiisg == 0) mixed[(uint64_t)tok * E + d] = g / (float)hc;
}

#define QWEN4_HC_MIX_INSTANCE(SUFFIX, W) \
template [[host_name("kernel_qwen4_hc_gate_mix_" #SUFFIX)]] \
kernel void kernel_qwen4_hc_gate_mix<W>(constant ds4_metal_args_qwen4_hc_gate_mix &, device const float *, \
        device const float *, device const char *, device float *, uint3, ushort3, ushort, ushort);
QWEN4_HC_MIX_INSTANCE(f16, qwen4_w_f16)
QWEN4_HC_MIX_INSTANCE(f32, qwen4_w_f32)
QWEN4_HC_MIX_INSTANCE(q8, qwen4_w_q8)

struct ds4_metal_args_qwen4_hc_combine {
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t pad0;
};

/* R[s][d] += 2*sigmoid(inj[s]/hc) * out[d]; the inject weights are reduced
 * once per threadgroup. */
kernel void kernel_qwen4_hc_combine(
        constant ds4_metal_args_qwen4_hc_combine & args,
        device float       *R,        /* [T][hc*E] */
        device const float *out,      /* [T][E] */
        device const float *inj,      /* [T][hc*chunks][hc] norm partials */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint d = tgpig.x * ntg.x + tid;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    threadgroup float wgt[8];
    if (tid < args.n_hc) {
        wgt[tid] = qwen4_hc_inject_weight(inj + (uint64_t)tok * args.n_hc * QWEN4_HC_CHUNKS * args.n_hc, args.n_hc, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= args.n_embd) return;
    const float o = out[(uint64_t)tok * args.n_embd + d];
    device float *r = R + (uint64_t)tok * args.n_embd * args.n_hc;
    for (uint s = 0; s < args.n_hc; s++) r[s * args.n_embd + d] += wgt[s] * o;
}

/* --- gated delta net ---------------------------------------------------- */

struct ds4_metal_args_qwen4_conv_stream {
    uint32_t n_tokens;
    uint32_t n_channels;
    uint32_t conv_kernel;   /* <= 4 */
    uint32_t apply_silu;
};

/* Depthwise causal conv over the token axis.  x holds the raw inputs for
 * all tokens and is overwritten with the (optionally silu'd) output; state
 * holds the K-1 previous raw inputs oldest first and is advanced past the
 * processed tokens.  One thread per channel, tokens sequential. */
kernel void kernel_qwen4_conv_stream(
        constant ds4_metal_args_qwen4_conv_stream & args,
        device float       *x,        /* [T][C] */
        device float       *state,    /* [K-1][C] */
        device const float *weight,   /* [C][K] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint c = tgpig.x * ntg.x + tid;
    if (c >= args.n_channels) return;
    const uint C = args.n_channels;
    const uint K = args.conv_kernel;

    float win[3];
    for (uint t = 0; t + 1 < K; t++) win[t] = state[t * C + c];
    float taps[4];
    for (uint t = 0; t < K; t++) taps[t] = weight[c * K + t];

    for (uint tok = 0; tok < args.n_tokens; tok++) {
        const float raw = x[tok * C + c];
        float acc = taps[K - 1] * raw;
        for (uint t = 0; t + 1 < K; t++) acc += taps[t] * win[t];
        for (uint t = 0; t + 2 < K; t++) win[t] = win[t + 1];
        win[K - 2] = raw;
        x[tok * C + c] = args.apply_silu ? qwen4_silu(acc) : acc;
    }
    for (uint t = 0; t + 1 < K; t++) state[t * C + c] = win[t];
}

struct ds4_metal_args_qwen4_gdn_prep {
    uint32_t n_tokens;
    uint32_t n_k_head;
    uint32_t n_v_head;
    uint32_t head_dim;
};

/* After the conv: L2-normalize each q and k head (q also scaled by
 * 1/sqrt(D)), turn the alpha/beta projections into the decay g =
 * exp(ssm_a * softplus(a + dt_bias)) and beta = sigmoid(b).  One simdgroup
 * per (token, k-head); the v-head scalars are handled by the first head. */
kernel void kernel_qwen4_gdn_prep(
        constant ds4_metal_args_qwen4_gdn_prep & args,
        device float       *qkv,      /* [T][2*Hk*D + Hv*D], conv output, q/k normalized in place */
        device float       *a,        /* [T][Hv]: alpha in, g out */
        device float       *b,        /* [T][Hv]: b in, beta out */
        device const float *ssm_a,    /* [Hv] = -exp(A_log) */
        device const float *dt_bias,  /* [Hv] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint tok = tgpig.y;
    if (h >= args.n_k_head || tok >= args.n_tokens) return;
    const uint D = args.head_dim;
    const uint npt = D / 32;
    const uint conv_dim = 2 * args.n_k_head * D + args.n_v_head * D;
    device float *q = qkv + (uint64_t)tok * conv_dim + h * D + tiisg * npt;
    device float *k = q + args.n_k_head * D;

    float sq = 0.0f, sk = 0.0f;
    for (uint i = 0; i < npt; i++) {
        sq += q[i] * q[i];
        sk += k[i] * k[i];
    }
    sq = simd_sum(sq);
    sk = simd_sum(sk);
    const float qs = rsqrt(sq + 1e-6f) * rsqrt((float)D);
    const float ks = rsqrt(sk + 1e-6f);
    for (uint i = 0; i < npt; i++) {
        q[i] *= qs;
        k[i] *= ks;
    }
    if (h == 0) {
        device float *ga = a + (uint64_t)tok * args.n_v_head;
        device float *gb = b + (uint64_t)tok * args.n_v_head;
        for (uint j = tiisg; j < args.n_v_head; j += 32) {
            ga[j] = exp(ssm_a[j] * qwen4_softplus(ga[j] + dt_bias[j]));
            gb[j] = qwen4_sigmoid(gb[j]);
        }
    }
}

struct ds4_metal_args_qwen4_gdn_scan {
    uint32_t n_tokens;
    uint32_t n_k_head;
    uint32_t n_v_head;
    uint32_t head_dim;
    uint32_t snap_tok;     /* copy the state after this token into snap_state (UINT32_MAX: never) */
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

/* Sequential gated delta scan.  One simdgroup per (v-head, dv) state row;
 * lane j owns S[dk = j*npt .. +npt-1] in registers for the whole token
 * loop.  Value head h reads key head h % Hk (tiled GGUF order).  Per token:
 * S *= g, u = S k, delta = beta (v - u), S += k delta, o = S q. */
kernel void kernel_qwen4_gdn_scan(
        constant ds4_metal_args_qwen4_gdn_scan & args,
        device const float *qkv,      /* [T][2*Hk*D + Hv*D], q/k normalized */
        device const float *ga,       /* [T][Hv] decay g */
        device const float *gb,       /* [T][Hv] beta */
        device float       *state,    /* [Hv][D][D] as [dv][dk] */
        device float       *out,      /* [T][Hv*D] */
        device float       *snap_state,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dv = tgpig.x;
    const uint h = tgpig.y;
    if (dv >= args.head_dim || h >= args.n_v_head) return;
    const uint D = args.head_dim;
    const uint npt = D / 32;
    const uint dk0 = tiisg * npt;
    const uint kh = h % args.n_k_head;
    const uint conv_dim = 2 * args.n_k_head * D + args.n_v_head * D;

    float s[4];
    device float *srow = state + ((uint64_t)h * D + dv) * D + dk0;
    device float *snaprow = snap_state + ((uint64_t)h * D + dv) * D + dk0;
    for (uint i = 0; i < npt; i++) s[i] = srow[i];

    for (uint tok = 0; tok < args.n_tokens; tok++) {
        device const float *q = qkv + (uint64_t)tok * conv_dim + kh * D + dk0;
        device const float *k = q + args.n_k_head * D;
        device const float *v = qkv + (uint64_t)tok * conv_dim + 2 * args.n_k_head * D + h * D;
        const float g = ga[(uint64_t)tok * args.n_v_head + h];
        const float beta = gb[(uint64_t)tok * args.n_v_head + h];

        float u = 0.0f;
        for (uint i = 0; i < npt; i++) {
            s[i] *= g;
            u += s[i] * k[i];
        }
        u = simd_sum(u);
        const float delta = (v[dv] - u) * beta;
        float o = 0.0f;
        for (uint i = 0; i < npt; i++) {
            s[i] += k[i] * delta;
            o += s[i] * q[i];
        }
        o = simd_sum(o);
        if (tiisg == 0) out[((uint64_t)tok * args.n_v_head + h) * D + dv] = o;
        if (tok == args.snap_tok) {
            for (uint i = 0; i < npt; i++) snaprow[i] = s[i];
        }
    }
    for (uint i = 0; i < npt; i++) srow[i] = s[i];
}

/* Prefill scan for head_dim 128: one simdgroup per (v-head, 4 consecutive dv
 * rows), so k/q/g/beta are loaded once per four state rows and the eight
 * reductions per token overlap.  Same math and state layout as above. */
kernel void kernel_qwen4_gdn_scan_r4(
        constant ds4_metal_args_qwen4_gdn_scan & args,
        device const float *qkv,
        device const float *ga,
        device const float *gb,
        device float       *state,
        device float       *out,
        device float       *snap_state,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dv0 = tgpig.x * 4;
    const uint h = tgpig.y;
    if (dv0 >= args.head_dim || h >= args.n_v_head) return;
    const uint D = 128, dk0 = tiisg * 4;
    const uint kh = h % args.n_k_head;
    const uint conv_dim = 2 * args.n_k_head * D + args.n_v_head * D;

    float4 s[4];
    device float *srow = state + ((uint64_t)h * D + dv0) * D + dk0;
    device float *snaprow = snap_state + ((uint64_t)h * D + dv0) * D + dk0;
    for (uint r = 0; r < 4; r++) s[r] = *(device const float4 *)(srow + r * D);

    for (uint tok = 0; tok < args.n_tokens; tok++) {
        device const float *base = qkv + (uint64_t)tok * conv_dim;
        const float4 q = *(device const float4 *)(base + kh * D + dk0);
        const float4 k = *(device const float4 *)(base + (args.n_k_head + kh) * D + dk0);
        const float4 v = *(device const float4 *)(base + 2 * args.n_k_head * D + h * D + dv0);
        const float g = ga[(uint64_t)tok * args.n_v_head + h];
        const float beta = gb[(uint64_t)tok * args.n_v_head + h];
        float u[4], o[4];
        for (uint r = 0; r < 4; r++) { s[r] *= g; u[r] = dot(s[r], k); }
        for (uint r = 0; r < 4; r++) u[r] = simd_sum(u[r]);
        for (uint r = 0; r < 4; r++) { s[r] += k * ((v[r] - u[r]) * beta); o[r] = dot(s[r], q); }
        for (uint r = 0; r < 4; r++) o[r] = simd_sum(o[r]);
        if (tiisg == 0) {
            *(device float4 *)(out + ((uint64_t)tok * args.n_v_head + h) * D + dv0) = float4(o[0], o[1], o[2], o[3]);
        }
        if (tok == args.snap_tok) {
            for (uint r = 0; r < 4; r++) *(device float4 *)(snaprow + r * D) = s[r];
        }
    }
    for (uint r = 0; r < 4; r++) *(device float4 *)(srow + r * D) = s[r];
}

struct ds4_metal_args_qwen4_gdn_out {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    float    eps;
};

/* Per-head RMSNorm of the scan output, scaled by ssm_norm and gated by
 * sigmoid(z). */
kernel void kernel_qwen4_gdn_out(
        constant ds4_metal_args_qwen4_gdn_out & args,
        device float       *o,        /* [T][H*D], in place */
        device const float *z,        /* [T][H*D] */
        device const float *weight,   /* [D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint tok = tgpig.y;
    if (h >= args.n_head || tok >= args.n_tokens) return;
    const uint D = args.head_dim;
    const uint npt = D / 32;
    const uint64_t base = ((uint64_t)tok * args.n_head + h) * D + tiisg * npt;
    float ss = 0.0f;
    for (uint i = 0; i < npt; i++) ss += o[base + i] * o[base + i];
    ss = simd_sum(ss);
    const float r = rsqrt(ss / (float)D + args.eps);
    for (uint i = 0; i < npt; i++) {
        o[base + i] = o[base + i] * r * weight[tiisg * npt + i] * qwen4_sigmoid(z[base + i]);
    }
}

/* --- PLE ---------------------------------------------------------------- */

struct ds4_metal_args_qwen4_ple_gate {
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_hc;
    float    eps;
};

/* Per token: grouped norms of the projected n-gram key and of the residual,
 * per-stream signed-sqrt sigmoid gate, gated value (kept raw for the
 * residual) and its grouped norm (conv input). */
kernel void kernel_qwen4_ple_gate(
        constant ds4_metal_args_qwen4_ple_gate & args,
        device const float *R,          /* [T][hc*E] */
        device const float *key,        /* [T][hc*E] raw key projection */
        device const float *value,      /* [T][E] */
        device const float *g_key,      /* [hc*E] */
        device const float *g_query,    /* [hc*E] */
        device const float *g_conv,     /* [hc*E] */
        device float       *gated,      /* [T][hc*E] */
        device float       *normed,     /* [T][hc*E] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens) return;
    const uint E = args.n_embd;
    const uint nth = ntg.x;
    const uint nsg = nth / 32;
    threadgroup float red[3][32];

    for (uint s = 0; s < args.n_hc; s++) {
        device const float *kr = key + ((uint64_t)tok * args.n_hc + s) * E;
        device const float *rr = R + ((uint64_t)tok * args.n_hc + s) * E;
        device const float *gk = g_key + s * E;
        device const float *gq = g_query + s * E;
        float sk = 0.0f, sr = 0.0f;
        for (uint i = tid; i < E; i += nth) {
            sk += kr[i] * kr[i];
            sr += rr[i] * rr[i];
        }
        sk = simd_sum(sk);
        sr = simd_sum(sr);
        if (tiisg == 0) { red[0][sgitg] = sk; red[1][sgitg] = sr; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tk = 0.0f, tr = 0.0f;
        for (uint g = 0; g < nsg; g++) { tk += red[0][g]; tr += red[1][g]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ik = rsqrt(tk / (float)E + args.eps);
        const float ir = rsqrt(tr / (float)E + args.eps);
        float dot = 0.0f;
        for (uint i = tid; i < E; i += nth) dot += (kr[i] * ik * gk[i]) * (rr[i] * ir * gq[i]);
        dot = simd_sum(dot);
        if (tiisg == 0) red[2][sgitg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float td = 0.0f;
        for (uint g = 0; g < nsg; g++) td += red[2][g];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gsc = td * rsqrt((float)E);
        const float mag = sqrt(max(fabs(gsc), 1e-6f));
        gsc = qwen4_sigmoid(gsc > 0.0f ? mag : (gsc < 0.0f ? -mag : 0.0f));

        device float *gd = gated + ((uint64_t)tok * args.n_hc + s) * E;
        device const float *val = value + (uint64_t)tok * E;
        float sg = 0.0f;
        for (uint i = tid; i < E; i += nth) {
            const float v = gsc * val[i];
            gd[i] = v;
            sg += v * v;
        }
        sg = simd_sum(sg);
        if (tiisg == 0) red[0][sgitg] = sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tg = 0.0f;
        for (uint g = 0; g < nsg; g++) tg += red[0][g];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float ig = rsqrt(tg / (float)E + args.eps);
        device float *nd = normed + ((uint64_t)tok * args.n_hc + s) * E;
        device const float *gc = g_conv + s * E;
        for (uint i = tid; i < E; i += nth) nd[i] = gd[i] * ig * gc[i];
    }
}

struct ds4_metal_args_qwen4_ple_conv {
    uint32_t n_tokens;
    uint32_t n_channels;
    uint32_t conv_kernel;
    uint32_t dilation;
    uint32_t weight_f16;   /* taps stored as half */
    uint32_t snap_tok;     /* copy the history after this token into snap_history */
    uint32_t pad1;
    uint32_t pad2;
};

/* R += gated + silu(dilated depthwise conv of normed).  history holds the
 * (K-1)*dilation previous normed rows oldest first and is advanced.  One
 * thread per channel, tokens sequential. */
kernel void kernel_qwen4_ple_conv(
        constant ds4_metal_args_qwen4_ple_conv & args,
        device float       *R,          /* [T][C] */
        device const float *gated,      /* [T][C] */
        device const float *normed,     /* [T][C] */
        device float       *history,    /* [(K-1)*dil][C] */
        device const char  *weight,     /* [C][K] taps, f32 or f16 */
        device float       *snap_history,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint c = tgpig.x * ntg.x + tid;
    if (c >= args.n_channels) return;
    const uint C = args.n_channels;
    const uint K = args.conv_kernel;
    const uint dil = args.dilation;
    const uint H = (K - 1) * dil;   /* <= 9 */

    float hist[9];
    for (uint t = 0; t < H; t++) hist[t] = history[t * C + c];
    float taps[4];
    for (uint t = 0; t < K; t++) taps[t] = (args.weight_f16 ? (float)((device const half *)weight)[c * K + t] : ((device const float *)weight)[c * K + t]);

    for (uint tok = 0; tok < args.n_tokens; tok++) {
        const float cur = normed[tok * C + c];
        float acc = taps[K - 1] * cur;
        for (uint k = 0; k + 1 < K; k++) acc += taps[k] * hist[H - (K - 1 - k) * dil];
        for (uint t = 0; t + 1 < H; t++) hist[t] = hist[t + 1];
        hist[H - 1] = cur;
        R[tok * C + c] += gated[tok * C + c] + qwen4_silu(acc);
        if (tok == args.snap_tok) {
            for (uint t = 0; t < H; t++) snap_history[t * C + c] = hist[t];
        }
    }
    for (uint t = 0; t < H; t++) history[t * C + c] = hist[t];
}

/* --- router ------------------------------------------------------------- */

struct ds4_metal_args_qwen4_router {
    uint32_t n_tokens;
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t gate_type;    /* shared-expert gate row type; in_dim 0 disables */
    uint32_t in_dim;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

#define QWEN4_ROUTER_MAX_EXPERT 512
#define QWEN4_ROUTER_MAX_USED 16
#define QWEN4_ROUTER_LANE_MAX 2      /* experts per lane: 512 / (8 simdgroups * 32) */

/* softmax over the router logits, top-k by probability (lower index wins
 * ties), renormalized weights, plus the shared expert's gate logit (one row
 * dotted with x).  One 256-thread threadgroup per token; every lane keeps
 * its LANE_MAX logits in registers, each simdgroup ranks its own experts by
 * repeated simd argmax and simdgroup 0 merges the candidates. */
kernel void kernel_qwen4_router_topk(
        constant ds4_metal_args_qwen4_router & args,
        device const float *logits,     /* [T][n_expert] */
        device int32_t     *selected,   /* [T][n_used] */
        device float       *weights,    /* [T][n_used] */
        device const float *x,          /* [T][in_dim] */
        device const char  *w_gate,     /* [in_dim] shared gate row */
        device float       *shared_gate,/* [T] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens) return;
    const uint NE = args.n_expert;
    const uint nth = ntg.x;
    const uint nsg = nth / 32;
    threadgroup float redf[32];
    threadgroup float redg[32];
    threadgroup float cand_v[8 * QWEN4_ROUTER_MAX_USED];
    threadgroup int cand_i[8 * QWEN4_ROUTER_MAX_USED];
    device const float *lg = logits + (uint64_t)tok * NE;
    /* shared gate logit: f32 rows as float4 over the whole threadgroup, other
     * types through the generic row dot on the last simdgroup */
    float gpart = 0.0f;
    if (args.in_dim && args.gate_type == 0 && (args.in_dim % 4) == 0) {
        device const float4 *w4 = (device const float4 *)w_gate;
        device const float4 *x4 = (device const float4 *)(x + (uint64_t)tok * args.in_dim);
        const uint n4 = args.in_dim / 4;
        float4 acc4 = 0.0f;
        for (uint i = tid; i < n4; i += nth) acc4 += w4[i] * x4[i];
        gpart = acc4.x + acc4.y + acc4.z + acc4.w;
        gpart = simd_sum(gpart);
    } else if (args.in_dim && sgitg == nsg - 1) {
        gpart = qwen4_row_dot(w_gate, x + (uint64_t)tok * args.in_dim, args.gate_type, args.in_dim, tiisg);
    }
    if (args.in_dim && tiisg == 0) redg[sgitg] = gpart;

    /* lane-owned logits: e = sgitg*32 + tiisg + 32*nsg*k */
    float mine[QWEN4_ROUTER_LANE_MAX];
    float mx = -3.0e38f;
    for (uint k = 0; k < QWEN4_ROUTER_LANE_MAX; k++) {
        const uint e = (uint)sgitg * 32 + tiisg + 32 * nsg * k;
        mine[k] = e < NE ? lg[e] : -3.0e38f;
        mx = max(mx, mine[k]);
    }
    mx = simd_max(mx);
    if (tiisg == 0) redf[sgitg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = redf[0];
    for (uint g = 1; g < nsg; g++) mx = max(mx, redf[g]);
    if (args.in_dim && tid == 0) {
        float gl = 0.0f;
        for (uint g = 0; g < nsg; g++) gl += redg[g];
        shared_gate[tok] = gl;
    }
    float sum = 0.0f;
    for (uint k = 0; k < QWEN4_ROUTER_LANE_MAX; k++) {
        const uint e = (uint)sgitg * 32 + tiisg + 32 * nsg * k;
        mine[k] = e < NE ? exp(mine[k] - mx) : -1.0f;
        if (e < NE) sum += mine[k];
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) redf[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint g = 0; g < nsg; g++) sum += redf[g];
    const float inv = 1.0f / sum;
    for (uint k = 0; k < QWEN4_ROUTER_LANE_MAX; k++) if (mine[k] >= 0.0f) mine[k] *= inv;

    /* local ranking per simdgroup */
    for (uint r = 0; r < args.n_used; r++) {
        float bv = -1.0f;
        int bi = 0x7fffffff;
        for (uint k = 0; k < QWEN4_ROUTER_LANE_MAX; k++) {
            if (mine[k] > bv) { bv = mine[k]; bi = (int)((uint)sgitg * 32 + tiisg + 32 * nsg * k); }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_xor(bv, off);
            const int oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (tiisg == 0) { cand_v[sgitg * args.n_used + r] = bv; cand_i[sgitg * args.n_used + r] = bi; }
        for (uint k = 0; k < QWEN4_ROUTER_LANE_MAX; k++) {
            if ((int)((uint)sgitg * 32 + tiisg + 32 * nsg * k) == bi) mine[k] = -1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const uint n_cand = nsg * args.n_used;
        float cv[4];
        int ci[4];
        for (uint j = 0; j < 4; j++) {
            const uint c = tiisg + 32 * j;
            cv[j] = c < n_cand ? cand_v[c] : -1.0f;
            ci[j] = c < n_cand ? cand_i[c] : 0x7fffffff;
        }
        float wsum = 0.0f;
        for (uint r = 0; r < args.n_used; r++) {
            float bv = -1.0f;
            int bi = 0x7fffffff;
            for (uint j = 0; j < 4; j++) {
                if (cv[j] > bv || (cv[j] == bv && ci[j] < bi)) { bv = cv[j]; bi = ci[j]; }
            }
            for (uint off = 16; off > 0; off >>= 1) {
                const float ov = simd_shuffle_xor(bv, off);
                const int oi = simd_shuffle_xor(bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            for (uint j = 0; j < 4; j++) if (ci[j] == bi) cv[j] = -1.0f;
            if (tiisg == 0) {
                selected[(uint64_t)tok * args.n_used + r] = bi;
                weights[(uint64_t)tok * args.n_used + r] = bv;
            }
            wsum += bv;
        }
        if (tiisg < args.n_used) weights[(uint64_t)tok * args.n_used + tiisg] /= wsum;
    }
}

/* --- gated GQA attention + QSA indexer ---------------------------------- */

struct ds4_metal_args_qwen4_attn_prep {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t n_idx_head;
    uint32_t idx_dim;
    uint32_t pos0;
    uint32_t cache_cap;
    float    rope_base;
    float    eps;
    uint32_t pad0;
    float    rope_mscale;      /* YaRN magnitude scale on cos/sin (1 without) */
    float    rope_freq[32];    /* per-pair inverse frequencies (YaRN-adjusted) */
};

/* Interleaved multimodal NeoX rope: pair i takes the (t, h, w) position
 * component i % 3.  Text tokens carry one position in all three lanes. */
static inline void qwen4_rope_neox(thread float *x, uint n_rot, uint4 p3, constant float *freq, float mscale) {
    const uint nh = n_rot / 2;
    for (uint i = 0; i < nh; i++) {
        const uint m = i % 3;
        const uint pos = m == 0 ? p3.x : (m == 1 ? p3.y : p3.z);
        const float theta = (float)pos * freq[i];
        const float c = cos(theta) * mscale, s = sin(theta) * mscale;
        const float x0 = x[i], x1 = x[i + nh];
        x[i] = x0 * c - x1 * s;
        x[i + nh] = x0 * s + x1 * c;
    }
}

/* Per (token, head) simdgroup: RMSNorm + NeoX rope of one query head (from
 * the [q|gate] interleaved projection, gate copied out raw), of one kv head
 * (written to the f16 cache at pos0+tok) or of one indexer query head; the
 * raw indexer key is copied to its cache.  Heads are enumerated
 * q(0..H-1), k(H..H+Hkv-1), iq(H+Hkv..+Hi-1), ik(last). */
kernel void kernel_qwen4_attn_prep(
        constant ds4_metal_args_qwen4_attn_prep & args,
        device const float *qg,        /* [T][H*2*D] */
        device const float *kproj,     /* [T][Hkv*D] */
        device const float *vproj,     /* [T][Hkv*D] */
        device const float *iq,        /* [T][Hi*Di] */
        device const float *ik,        /* [T][Di] */
        device const float *g_q,       /* [D] */
        device const float *g_k,       /* [D] */
        device const float *g_iq,      /* [Di] */
        device float       *q_out,     /* [T][H*D] */
        device float       *gate_out,  /* [T][H*D] */
        device half        *k_cache,   /* [cap][Hkv*D] */
        device half        *v_cache,   /* [cap][Hkv*D] */
        device float       *iq_out,    /* [T][Hi*Di] */
        device float       *ik_cache,  /* [cap][Di] raw */
        device const uint4 *pos3,      /* [cap] rope positions (t, h, w) */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.x;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint H = args.n_head, Hkv = args.n_head_kv, D = args.head_dim;
    const uint Hi = args.n_idx_head, Di = args.idx_dim;
    const uint pos = args.pos0 + tok;
    const uint4 p3 = pos3[pos];
    float v[8];   /* D/32 <= 8 */
    threadgroup float row[576];
    float tmp[64];

    if (slot < H) {
        const uint npt = D / 32;
        device const float *src = qg + ((uint64_t)tok * H + slot) * 2 * D;
        float ss = 0.0f;
        for (uint i = 0; i < npt; i++) { v[i] = src[tiisg * npt + i]; ss += v[i] * v[i]; }
        ss = simd_sum(ss);
        const float r = rsqrt(ss / (float)D + args.eps);
        for (uint i = 0; i < npt; i++) v[i] = v[i] * r * g_q[tiisg * npt + i];
        /* rope needs the pairs (i, i+n_rot/2): stage through threadgroup memory */
        for (uint i = 0; i < npt; i++) row[tiisg * npt + i] = v[i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            for (uint i = 0; i < args.n_rot; i++) tmp[i] = row[i];
            qwen4_rope_neox(tmp, args.n_rot, p3, args.rope_freq, args.rope_mscale);
            for (uint i = 0; i < args.n_rot; i++) row[i] = tmp[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device float *dq = q_out + ((uint64_t)tok * H + slot) * D;
        device float *dg = gate_out + ((uint64_t)tok * H + slot) * D;
        for (uint i = 0; i < npt; i++) {
            dq[tiisg * npt + i] = row[tiisg * npt + i];
            dg[tiisg * npt + i] = src[D + tiisg * npt + i];
        }
        return;
    }
    if (slot < H + Hkv) {
        const uint h = slot - H;
        const uint npt = D / 32;
        device const float *src = kproj + ((uint64_t)tok * Hkv + h) * D;
        device const float *vs = vproj + ((uint64_t)tok * Hkv + h) * D;
        float ss = 0.0f;
        for (uint i = 0; i < npt; i++) { v[i] = src[tiisg * npt + i]; ss += v[i] * v[i]; }
        ss = simd_sum(ss);
        const float r = rsqrt(ss / (float)D + args.eps);
        for (uint i = 0; i < npt; i++) v[i] = v[i] * r * g_k[tiisg * npt + i];
        for (uint i = 0; i < npt; i++) row[tiisg * npt + i] = v[i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            for (uint i = 0; i < args.n_rot; i++) tmp[i] = row[i];
            qwen4_rope_neox(tmp, args.n_rot, p3, args.rope_freq, args.rope_mscale);
            for (uint i = 0; i < args.n_rot; i++) row[i] = tmp[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device half *dk = k_cache + ((uint64_t)pos * Hkv + h) * D;
        device half *dv = v_cache + ((uint64_t)pos * Hkv + h) * D;
        for (uint i = 0; i < npt; i++) {
            dk[tiisg * npt + i] = (half)row[tiisg * npt + i];
            dv[tiisg * npt + i] = (half)vs[tiisg * npt + i];
        }
        return;
    }
    if (slot < H + Hkv + Hi) {
        const uint h = slot - H - Hkv;
        const uint npt = Di / 32;
        device const float *src = iq + ((uint64_t)tok * Hi + h) * Di;
        float ss = 0.0f;
        for (uint i = 0; i < npt; i++) { v[i] = src[tiisg * npt + i]; ss += v[i] * v[i]; }
        ss = simd_sum(ss);
        const float r = rsqrt(ss / (float)Di + args.eps);
        for (uint i = 0; i < npt; i++) v[i] = v[i] * r * g_iq[tiisg * npt + i];
        for (uint i = 0; i < npt; i++) row[tiisg * npt + i] = v[i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            for (uint i = 0; i < args.n_rot; i++) tmp[i] = row[i];
            qwen4_rope_neox(tmp, args.n_rot, p3, args.rope_freq, args.rope_mscale);
            for (uint i = 0; i < args.n_rot; i++) row[i] = tmp[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device float *dq = iq_out + ((uint64_t)tok * Hi + h) * Di;
        for (uint i = 0; i < npt; i++) dq[tiisg * npt + i] = row[tiisg * npt + i];
        return;
    }
    {
        const uint npt = Di / 32;
        device const float *src = ik + (uint64_t)tok * Di;
        device float *dst = ik_cache + (uint64_t)pos * Di;
        for (uint i = 0; i < npt; i++) dst[tiisg * npt + i] = src[tiisg * npt + i];
    }
}

struct ds4_metal_args_qwen4_idx_block {
    uint32_t block0;      /* first block to (re)build */
    uint32_t n_blocks;
    uint32_t ratio;
    uint32_t idx_dim;
    uint32_t n_rot;
    float    rope_base;
    float    eps;
    uint32_t pad0;
    float    rope_mscale;
    float    rope_freq[32];
};

/* Block key b: mean of its ratio raw indexer keys, RMSNorm (gamma), NeoX
 * rope at the block's first token position.  One simdgroup per block. */
kernel void kernel_qwen4_idx_block_key(
        constant ds4_metal_args_qwen4_idx_block & args,
        device const float *ik_cache,   /* [cap][Di] */
        device const float *g_ik,       /* [Di] */
        device const uint4 *pos3,       /* [cap] */
        device half        *block_key,  /* [n_blocks_cap][Di] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint b = args.block0 + tgpig.x;
    if (tgpig.x >= args.n_blocks) return;
    const uint Di = args.idx_dim;
    const uint npt = Di / 32;
    float v[4];
    float tmp[64];
    threadgroup float row[128];
    for (uint i = 0; i < npt; i++) {
        float acc = 0.0f;
        for (uint t = 0; t < args.ratio; t++) acc += ik_cache[((uint64_t)b * args.ratio + t) * Di + tiisg * npt + i];
        v[i] = acc / (float)args.ratio;
    }
    float ss = 0.0f;
    for (uint i = 0; i < npt; i++) ss += v[i] * v[i];
    ss = simd_sum(ss);
    const float r = rsqrt(ss / (float)Di + args.eps);
    for (uint i = 0; i < npt; i++) row[tiisg * npt + i] = v[i] * r * g_ik[tiisg * npt + i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        for (uint i = 0; i < args.n_rot; i++) tmp[i] = row[i];
        qwen4_rope_neox(tmp, args.n_rot, pos3[b * args.ratio], args.rope_freq, args.rope_mscale);
        for (uint i = 0; i < args.n_rot; i++) row[i] = tmp[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device half *dst = block_key + (uint64_t)b * Di;
    for (uint i = 0; i < npt; i++) dst[tiisg * npt + i] = (half)row[tiisg * npt + i];
}

struct ds4_metal_args_qwen4_idx_score {
    uint32_t n_tokens;
    uint32_t n_blocks;    /* scored per token: blocks < n_visible(tok) */
    uint32_t n_idx_head;
    uint32_t idx_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint32_t pad0;
    uint32_t pad1;
};

/* score[tok][b] = sum over indexer heads of relu(q_h . key_b); blocks not
 * yet complete for the token's position score -inf.  One lane per block. */
kernel void kernel_qwen4_idx_score(
        constant ds4_metal_args_qwen4_idx_score & args,
        device const float *iq,          /* [T][Hi*Di] */
        device const half  *block_key,   /* [n_blocks][Di] */
        device float       *score,       /* [T][n_blocks] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint b = tgpig.x * ntg.x + tid;
    const uint tok = tgpig.y;
    if (b >= args.n_blocks || tok >= args.n_tokens) return;
    const uint Di = args.idx_dim;
    const uint visible = (args.pos0 + tok + 1) / args.ratio;
    if (b >= visible) {
        score[(uint64_t)tok * args.n_blocks + b] = -3.0e38f;
        return;
    }
    device const half *key = block_key + (uint64_t)b * Di;
    device const float *q = iq + (uint64_t)tok * args.n_idx_head * Di;
    float sum = 0.0f;
    for (uint h = 0; h < args.n_idx_head; h++) {
        float dot = 0.0f;
        for (uint d = 0; d < Di; d++) dot += q[h * Di + d] * (float)key[d];
        sum += max(dot, 0.0f);
    }
    score[(uint64_t)tok * args.n_blocks + b] = sum;
}

/* Matrix-unit block scorer for 4 heads x 128 dims: a threadgroup scores 16
 * tokens x 64 blocks.  Per K half the 64 keys and the 64 query rows (16
 * tokens x 4 heads, rounded to half, stored transposed) are staged so both
 * operands load straight; each simdgroup accumulates a 32 x 32 quarter of
 * the blocks x rows product, then the relu'd head rows are summed per
 * token.  Same output as kernel_qwen4_idx_score. */
kernel void kernel_qwen4_idx_score_mm(
        constant ds4_metal_args_qwen4_idx_score & args,
        device const float *iq,          /* [T][4*128] */
        device const half  *block_key,   /* [n_blocks][128] */
        device float       *score,       /* [T][n_blocks] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint b0 = tgpig.x * 64, t0 = tgpig.y * 16;
    if (b0 >= args.n_blocks || t0 >= args.n_tokens) return;
    threadgroup uint4 buf[1040];
    threadgroup half *Bk = (threadgroup half *)buf;          /* [block][k half] */
    threadgroup half *Qt = Bk + 64 * 64;                     /* [k half][row], stride 66 */
    threadgroup float *C = (threadgroup float *)buf;         /* [row][block] after the loop */
    const uint sb = (sgitg >> 1) * 32, sr = (sgitg & 1) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < 128; k0 += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64 * 8; i += 128) {
            const uint blk = i >> 3, seg = i & 7, gb = b0 + blk;
            const uint4 v = gb < args.n_blocks ? ((device const uint4 *)(block_key + (uint64_t)gb * 128 + k0))[seg] : uint4(0u);
            ((threadgroup uint4 *)(Bk + blk * 64))[seg] = v;
        }
        for (uint i = tid; i < 64 * 64; i += 128) {
            const uint kx = i >> 6, r = i & 63, tok = t0 + (r >> 2), h = r & 3;
            Qt[kx * 66 + r] = tok < args.n_tokens ? (half)iq[(uint64_t)tok * 512 + h * 128 + k0 + kx] : (half)0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 64; kk += 8) {
            simdgroup_half8x8 a[4], b[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], Bk + (sb + i * 8) * 64 + kk, 64, 0, false);
            for (uint j = 0; j < 4; j++) simdgroup_load(b[j], Qt + kk * 66 + sr + j * 8, 66, 0, false);
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], C + (sr + j * 8) * 64 + sb + i * 8, 64, 0, true);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 16 * 64; i += 128) {
        const uint lt = i >> 6, lb = i & 63, tok = t0 + lt, gb = b0 + lb;
        if (tok >= args.n_tokens || gb >= args.n_blocks) continue;
        const uint visible = (args.pos0 + tok + 1) / args.ratio;
        float sum = -3.0e38f;
        if (gb < visible) {
            sum = 0.0f;
            for (uint h = 0; h < 4; h++) sum += max(C[(lt * 4 + h) * 64 + lb], 0.0f);
        }
        score[(uint64_t)tok * args.n_blocks + gb] = sum;
    }
}

struct ds4_metal_args_qwen4_idx_select {
    uint32_t n_tokens;
    uint32_t n_blocks;    /* row stride of score */
    uint32_t top_k;
    uint32_t pad0;
};

/* Exact top-k per token row without sorting: scores are >= 0 or -inf, so
 * their float bits order them and a 4-pass radix select over 8-bit digits
 * finds the k-th largest key and how many of its equals to take.  The gather
 * then keeps every block above that key and, among equals, the lowest block
 * indices (invisible blocks score -inf and sit above every visible one, so
 * they never win a tie).  One threadgroup per token; blocks above the key
 * come out in ascending block order, the equals after them. */
kernel void kernel_qwen4_idx_select(
        constant ds4_metal_args_qwen4_idx_select & args,
        device const float *score,     /* [T][n_blocks] */
        device int32_t     *sel,       /* [T][top_k] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens) return;
    const uint nth = ntg.x;
    device const float *row = score + (uint64_t)tok * args.n_blocks;
    device int32_t *out = sel + (uint64_t)tok * args.top_k;
    const uint n = args.n_blocks;
    threadgroup atomic_uint hist[256];
    threadgroup uint scan[32];
    threadgroup uint found[2];
    uint prefix = 0, need = args.top_k;
    for (uint pass = 0; pass < 4; pass++) {
        const uint shift = 24 - 8 * pass;
        const uint mask_hi = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint i = tid; i < 256; i += nth) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* 16 loads per thread in flight before the counting */
        for (uint b = tid; b < n; b += 16u * nth) {
            float v[16];
            for (uint u = 0; u < 16; u++) { const uint i = b + u * nth; v[u] = i < n ? row[i] : -1.0f; }
            for (uint u = 0; u < 16; u++) {
                if (b + u * nth >= n) break;
                const uint key = as_type<uint>(max(v[u], 0.0f));
                if ((key & mask_hi) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        /* suffix sums from the top digit down: the first 256 threads hold one
         * digit each, higher threads contribute zeros */
        const uint c = tid < 256 ? atomic_load_explicit(&hist[255 - tid], memory_order_relaxed) : 0u;
        const uint p = simd_prefix_inclusive_sum(c);
        if (tiisg == 31 && sgitg < 8) scan[sgitg] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256) {
            uint above = p - c;                     /* elements in digits above this one */
            for (uint g = 0; g < sgitg; g++) above += scan[g];
            if (above < need && above + c >= need) {
                found[0] = prefix | ((255u - tid) << shift);
                found[1] = need - above;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = found[0]; need = found[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /* gather: contiguous chunk per thread, ranks by exclusive scan */
    const uint chunk = (n + nth - 1) / nth;
    const uint b0 = min((uint)tid * chunk, n), b1 = min(b0 + chunk, n);
    uint n_gt = 0, n_eq = 0;
    for (uint b = b0; b < b1; b += 8) {
        float v[8];
        for (uint u = 0; u < 8; u++) v[u] = b + u < b1 ? row[b + u] : -1.0f;
        for (uint u = 0; u < 8 && b + u < b1; u++) {
            const uint key = as_type<uint>(max(v[u], 0.0f));
            n_gt += key > prefix; n_eq += key == prefix;
        }
    }
    uint r_gt, r_eq;
    for (uint which = 0; which < 2; which++) {
        const uint v = which == 0 ? n_gt : n_eq;
        const uint p = simd_prefix_exclusive_sum(v);
        if (tiisg == 31) scan[sgitg] = p + v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            const uint nsg = (nth + 31) / 32;
            const uint sv = tiisg < nsg ? scan[tiisg] : 0u;
            const uint sp = simd_prefix_exclusive_sum(sv);
            if (tiisg < nsg) scan[tiisg] = sp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (which == 0) r_gt = p + scan[sgitg]; else r_eq = p + scan[sgitg];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint eq_base = args.top_k - need;
    for (uint b = b0; b < b1; b += 8) {
        float v[8];
        for (uint u = 0; u < 8; u++) v[u] = b + u < b1 ? row[b + u] : -1.0f;
        for (uint u = 0; u < 8 && b + u < b1; u++) {
            const uint key = as_type<uint>(max(v[u], 0.0f));
            if (key > prefix) out[r_gt++] = (int32_t)(b + u);
            else if (key == prefix) { if (r_eq < need) out[eq_base + r_eq] = (int32_t)(b + u); r_eq++; }
        }
    }
}

struct ds4_metal_args_qwen4_idx_expand {
    uint32_t n_tokens;
    uint32_t n_sel_blocks;   /* k_eff */
    uint32_t ratio;
    uint32_t pos0;
    uint32_t sel_stride;     /* row stride of sel_tokens */
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

/* Selected blocks -> token list (block tokens, then the incomplete tail up
 * to and including the query position); n_sel[tok] gets the count.  The
 * host only routes tokens with more complete blocks than the budget here. */
kernel void kernel_qwen4_idx_expand(
        constant ds4_metal_args_qwen4_idx_expand & args,
        device const int32_t *sel_blocks,   /* [T][n_sel_blocks] */
        device int32_t       *sel_tokens,   /* [T][sel_stride] */
        device uint32_t      *n_sel,        /* [T] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens) return;
    const uint pos = args.pos0 + tok;
    const uint n_blk = args.n_sel_blocks;
    const uint tail_start = ((pos + 1) / args.ratio) * args.ratio;
    device int32_t *dst = sel_tokens + (uint64_t)tok * args.sel_stride;
    device const int32_t *blk = sel_blocks + (uint64_t)tok * n_blk;
    for (uint i = tid; i < n_blk * args.ratio; i += ntg.x) {
        dst[i] = blk[i / args.ratio] * (int32_t)args.ratio + (int32_t)(i % args.ratio);
    }
    for (uint t = tail_start + tid; t <= pos; t += ntg.x) {
        dst[n_blk * args.ratio + (t - tail_start)] = (int32_t)t;
    }
    if (tid == 0) n_sel[tok] = n_blk * args.ratio + (pos + 1 - tail_start);
}

struct ds4_metal_args_qwen4_attn_decode {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t pos0;        /* dense mode: token attends positions 0..pos0+tok */
    uint32_t use_sel;     /* 1: read sel_tokens/n_sel */
    uint32_t sel_stride;
    float    scale;
    uint32_t n_splits;    /* key ranges per (token, kv head); > 1 writes partials */
    uint32_t keys_per_split;
    uint32_t pad0;
    uint32_t pad1;
};

#define QWEN4_ATTN_NSG 4          /* simdgroups per threadgroup, each owning a slice of the q-head group */
#define QWEN4_ATTN_HPS 3          /* q heads per simdgroup: group <= NSG * HPS */

/* Decode attention for one (key split, kv head, token).  The simdgroups of
 * a threadgroup share the K/V rows but own disjoint query heads, so every
 * lane keeps at most HPS heads in registers; lane j owns dims j*NPT..+NPT-1
 * (NPT = D/32 is a template constant so the per-lane arrays stay in
 * registers).  With one split the gated output is written directly;
 * otherwise each head leaves (m, l, acc) partials for the merge kernel. */
template <uint NPT>
kernel void kernel_qwen4_attn_decode(
        constant ds4_metal_args_qwen4_attn_decode & args,
        device const float   *q,          /* [T][H*D] */
        device const float   *gate,       /* [T][H*D] */
        device const half    *k_cache,    /* [cap][Hkv*D] */
        device const half    *v_cache,    /* [cap][Hkv*D] */
        device const int32_t *sel_tokens, /* [T][sel_stride] */
        device const uint32_t *n_sel,     /* [T] */
        device float         *out,        /* [T][H*D] */
        device float         *part,       /* [T][Hkv][n_splits][group][2+D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint split = tgpig.x;
    const uint kvh = tgpig.y;
    const uint tok = tgpig.z;
    if (split >= args.n_splits || kvh >= args.n_head_kv || tok >= args.n_tokens) return;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    constexpr uint D = NPT * 32;
    const uint group = H / Hkv;
    const uint hps = (group + QWEN4_ATTN_NSG - 1) / QWEN4_ATTN_NSG;
    const uint g0 = (uint)sgitg * hps;
    if (g0 >= group) return;
    const uint ng = min(hps, group - g0);
    const uint n = args.use_sel ? n_sel[tok] : args.pos0 + tok + 1;
    const uint k0 = split * args.keys_per_split;
    const uint k1 = min(n, k0 + args.keys_per_split);
    device const int32_t *sel = sel_tokens + (uint64_t)tok * args.sel_stride;

    float qv[QWEN4_ATTN_HPS][NPT];
    float m[QWEN4_ATTN_HPS], l[QWEN4_ATTN_HPS], acc[QWEN4_ATTN_HPS][NPT];
#pragma unroll
    for (uint g = 0; g < QWEN4_ATTN_HPS; g++) {
        const uint h = kvh * group + g0 + min(g, ng - 1u);
        device const float *qh = q + ((uint64_t)tok * H + h) * D + tiisg * NPT;
#pragma unroll
        for (uint i = 0; i < NPT; i++) qv[g][i] = qh[i] * args.scale;
        m[g] = -3.0e38f;
        l[g] = 0.0f;
#pragma unroll
        for (uint i = 0; i < NPT; i++) acc[g][i] = 0.0f;
    }
    for (uint idx = k0; idx < k1; idx++) {
        const uint p = args.use_sel ? (uint)sel[idx] : idx;
        device const half *kr = k_cache + ((uint64_t)p * Hkv + kvh) * D + tiisg * NPT;
        device const half *vr = v_cache + ((uint64_t)p * Hkv + kvh) * D + tiisg * NPT;
        float kv[NPT], vv[NPT];
#pragma unroll
        for (uint i = 0; i < NPT; i++) { kv[i] = (float)kr[i]; vv[i] = (float)vr[i]; }
#pragma unroll
        for (uint g = 0; g < QWEN4_ATTN_HPS; g++) {
            if (g < ng) {
                float s = 0.0f;
#pragma unroll
                for (uint i = 0; i < NPT; i++) s += qv[g][i] * kv[i];
                s = simd_sum(s);
                const float m_new = max(m[g], s);
                const float corr = exp(m[g] - m_new);
                const float w = exp(s - m_new);
                l[g] = l[g] * corr + w;
#pragma unroll
                for (uint i = 0; i < NPT; i++) acc[g][i] = acc[g][i] * corr + w * vv[i];
                m[g] = m_new;
            }
        }
    }
#pragma unroll
    for (uint g = 0; g < QWEN4_ATTN_HPS; g++) {
        if (g >= ng) break;
        const uint h = kvh * group + g0 + g;
        if (args.n_splits == 1) {
            device float *dst = out + ((uint64_t)tok * H + h) * D + tiisg * NPT;
            device const float *gt = gate + ((uint64_t)tok * H + h) * D + tiisg * NPT;
            const float inv = l[g] > 0.0f ? 1.0f / l[g] : 0.0f;
#pragma unroll
            for (uint i = 0; i < NPT; i++) dst[i] = acc[g][i] * inv * qwen4_sigmoid(gt[i]);
        } else {
            device float *dst = part + ((((uint64_t)tok * Hkv + kvh) * args.n_splits + split) * group + g0 + g) * (2u + D);
            if (tiisg == 0) { dst[0] = m[g]; dst[1] = l[g]; }
#pragma unroll
            for (uint i = 0; i < NPT; i++) dst[2u + tiisg * NPT + i] = acc[g][i];
        }
    }
}

/* Merge the split partials of one (token, head) and apply the gate. */
template <uint NPT>
kernel void kernel_qwen4_attn_merge(
        constant ds4_metal_args_qwen4_attn_decode & args,
        device const float *part,
        device const float *gate,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint tok = tgpig.y;
    if (h >= args.n_head || tok >= args.n_tokens) return;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    constexpr uint D = NPT * 32;
    const uint group = H / Hkv;
    const uint kvh = h / group, g = h % group;
    const uint64_t stride = (uint64_t)group * (2u + D);
    device const float *base = part + (((uint64_t)tok * Hkv + kvh) * args.n_splits * group + g) * (2u + D);
    float mm = -3.0e38f;
    for (uint s = 0; s < args.n_splits; s++) mm = max(mm, base[s * stride]);
    float ll = 0.0f;
    float o[NPT];
#pragma unroll
    for (uint i = 0; i < NPT; i++) o[i] = 0.0f;
    for (uint s = 0; s < args.n_splits; s++) {
        device const float *p = base + s * stride;
        const float c = p[1] > 0.0f ? exp(p[0] - mm) : 0.0f;
        ll += p[1] * c;
#pragma unroll
        for (uint i = 0; i < NPT; i++) o[i] += p[2u + tiisg * NPT + i] * c;
    }
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    device float *dst = out + ((uint64_t)tok * H + h) * D + tiisg * NPT;
    device const float *gt = gate + ((uint64_t)tok * H + h) * D + tiisg * NPT;
#pragma unroll
    for (uint i = 0; i < NPT; i++) dst[i] = o[i] * inv * qwen4_sigmoid(gt[i]);
}

#define QWEN4_ATTN_INSTANCE(NPT_) \
template [[host_name("kernel_qwen4_attn_decode_npt" #NPT_)]] \
kernel void kernel_qwen4_attn_decode<NPT_>(constant ds4_metal_args_qwen4_attn_decode &, device const float *, \
        device const float *, device const half *, device const half *, device const int32_t *, device const uint32_t *, \
        device float *, device float *, uint3, ushort, ushort); \
template [[host_name("kernel_qwen4_attn_merge_npt" #NPT_)]] \
kernel void kernel_qwen4_attn_merge<NPT_>(constant ds4_metal_args_qwen4_attn_decode &, device const float *, \
        device const float *, device float *, uint3, ushort);
QWEN4_ATTN_INSTANCE(8)
QWEN4_ATTN_INSTANCE(4)
QWEN4_ATTN_INSTANCE(1)

#define QWEN4_AMM_KT 16

/* Prefill attention on simdgroup matrices: one (kv head, token) per
 * threadgroup, the query heads of the group as two 8-row tiles, keys in
 * tiles of 16 staged from the selected positions (dense mode: every causal
 * position).  Simdgroup s owns row tile s & 1 and dim half s >> 1 (16
 * accumulators); the two simdgroups of a row tile each score one key half,
 * exchange the scores and run the same online softmax.  Same output as
 * kernel_qwen4_attn_decode without key splits, with the queries rounded to
 * half. */
kernel void kernel_qwen4_attn_mm(
        constant ds4_metal_args_qwen4_attn_decode & args,
        device const float   *q,          /* [T][H*D] */
        device const float   *gate,       /* [T][H*D] */
        device const half    *k_cache,    /* [cap][Hkv*D] */
        device const half    *v_cache,    /* [cap][Hkv*D] */
        device const int32_t *sel_tokens, /* [T][sel_stride] */
        device const uint32_t *n_sel,     /* [T] */
        device float         *out,        /* [T][H*D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint kvh = tgpig.x, tok = tgpig.y;
    if (kvh >= args.n_head_kv || tok >= args.n_tokens) return;
    constexpr uint D = 256;
    const uint H = args.n_head, Hkv = args.n_head_kv, group = H / Hkv;
    const uint qpos = args.pos0 + tok;
    const uint n = args.use_sel ? n_sel[tok] : qpos + 1;
    device const int32_t *sel = sel_tokens + (uint64_t)tok * args.sel_stride;
    const uint rt = sgitg & 1u, dh = sgitg >> 1;
    const uint lr = tiisg >> 2, lc = (tiisg & 3u) * 4;   /* this lane's row and first of four key columns */

    threadgroup half KV[2 * QWEN4_AMM_KT * D];           /* keys, then values; the epilogue reuses it as floats */
    threadgroup half *Ks = KV, *Vs = KV + QWEN4_AMM_KT * D;
    threadgroup half Qs[16 * D];                          /* scaled queries as half */
    threadgroup float Sx[2][2][64];                       /* [row tile][key half] scores */
    threadgroup half  Ps[4][128];                         /* per simdgroup 8 x 16 probabilities */
    threadgroup float Dg[4][64];                          /* per simdgroup diagonal factors */
    threadgroup float Id[64];
    threadgroup int   kpos[QWEN4_AMM_KT];

    for (uint i = tid; i < 16 * D; i += 128) {
        const uint r = i / D, d = i % D;
        Qs[i] = r < group ? (half)(q[((uint64_t)tok * H + kvh * group + r) * D + d] * args.scale) : (half)0.0h;
    }
    if (tid < 64) Id[tid] = (tid >> 3) == (tid & 7u) ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 I;
    simdgroup_load(I, Id, 8, 0, false);
    simdgroup_float8x8 O[16];
#pragma unroll
    for (uint j = 0; j < 16; j++) O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m_row = -3.0e38f, l_row = 0.0f;

    for (uint t0 = 0; t0 < n; t0 += QWEN4_AMM_KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < QWEN4_AMM_KT) {
            const uint idx = t0 + tid;
            const int p = idx < n ? (args.use_sel ? sel[idx] : (int)idx) : -1;
            kpos[tid] = p > (int)qpos ? -1 : p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   /* 16 rows of K and V, 64 bytes per thread each */
            const uint key = tid >> 3, seg = tid & 7u;
            const int p = kpos[key];
            const uint64_t row = ((uint64_t)max(p, 0) * Hkv + kvh) * D;
            device const uint4 *kr = (device const uint4 *)(k_cache + row) + seg * 4;
            device const uint4 *vr = (device const uint4 *)(v_cache + row) + seg * 4;
            threadgroup uint4 *kd = (threadgroup uint4 *)(Ks + key * D) + seg * 4;
            threadgroup uint4 *vd = (threadgroup uint4 *)(Vs + key * D) + seg * 4;
#pragma unroll
            for (uint u = 0; u < 4; u++) { kd[u] = p >= 0 ? kr[u] : uint4(0u); vd[u] = p >= 0 ? vr[u] : uint4(0u); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* scores of this row tile against key half dh, four independent chains */
        simdgroup_float8x8 S[4];
#pragma unroll
        for (uint i = 0; i < 4; i++) S[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma unroll
        for (uint kk = 0; kk < 8; kk++) {
#pragma unroll
            for (uint i = 0; i < 4; i++) {
                simdgroup_half8x8 Qt, Kt;
                simdgroup_load(Qt, Qs + (rt * 8) * D + (kk * 4 + i) * 8, D, 0, false);
                simdgroup_load(Kt, Ks + (dh * 8) * D + (kk * 4 + i) * 8, D, 0, true);
                simdgroup_multiply_accumulate(S[i], Qt, Kt, S[i]);
            }
        }
#pragma unroll
        for (uint i = 1; i < 4; i++) simdgroup_multiply_accumulate(S[0], I, S[i], S[0]);
        simdgroup_store(S[0], Sx[rt][dh], 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sv[4], pv[4];
        bool valid[4];
        float mx = -3.0e38f;
#pragma unroll
        for (uint c = 0; c < 4; c++) {
            const uint key = lc + c;
            valid[c] = kpos[key] >= 0;
            sv[c] = valid[c] ? Sx[rt][key >> 3][lr * 8 + (key & 7u)] : -3.0e38f;
            mx = max(mx, sv[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m_row, mx);
        const float corr = exp(m_row - m_new);
        float rs = 0.0f;
#pragma unroll
        for (uint c = 0; c < 4; c++) {
            pv[c] = valid[c] ? exp(sv[c] - m_new) : 0.0f;
            rs += pv[c];
            Ps[sgitg][lr * 16 + lc + c] = (half)pv[c];
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 2);
        l_row = l_row * corr + rs;
        m_row = m_new;
        const bool rescale = simd_any(corr != 1.0f);
        if (rescale) {
            for (uint i = tiisg; i < 64; i += 32) Dg[sgitg][i] = (i >> 3) == (i & 7u) ? simd_shuffle(corr, (ushort)((i >> 3) * 4)) : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale) {
            simdgroup_float8x8 Dm;
            simdgroup_load(Dm, Dg[sgitg], 8, 0, false);
#pragma unroll
            for (uint j = 0; j < 16; j++) { simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O[j]); O[j] = t; }
        }
        simdgroup_half8x8 P0, P1;
        simdgroup_load(P0, Ps[sgitg], 16, 0, false);
        simdgroup_load(P1, Ps[sgitg] + 8, 16, 0, false);
#pragma unroll
        for (uint j = 0; j < 16; j++) {
            simdgroup_half8x8 V0, V1;
            simdgroup_load(V0, Vs + dh * 128 + j * 8, D, 0, false);
            simdgroup_load(V1, Vs + 8 * D + dh * 128 + j * 8, D, 0, false);
            simdgroup_multiply_accumulate(O[j], P0, V0, O[j]);
            simdgroup_multiply_accumulate(O[j], P1, V1, O[j]);
        }
    }

    /* normalize, then hand the tile to the epilogue through the K/V area */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
        for (uint i = tiisg; i < 64; i += 32) Dg[sgitg][i] = (i >> 3) == (i & 7u) ? simd_shuffle(inv, (ushort)((i >> 3) * 4)) : 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 Dm;
        simdgroup_load(Dm, Dg[sgitg], 8, 0, false);
        threadgroup float *Osc = (threadgroup float *)KV + (rt * 8) * D + dh * 128;
#pragma unroll
        for (uint j = 0; j < 16; j++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, Dm, O[j]);
            simdgroup_store(t, Osc + j * 8, D, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 16 * D; i += 128) {
        const uint r = i / D, d = i % D;
        if (r >= group) continue;
        const uint64_t o = ((uint64_t)tok * H + kvh * group + r) * D + d;
        out[o] = ((threadgroup float *)KV)[r * D + d] * qwen4_sigmoid(gate[o]);
    }
}

/* --- routed experts ----------------------------------------------------- */

#define QWEN4_MOE_NSG 4
#define QWEN4_MOE_NR0 2

struct ds4_metal_args_qwen4_moe {
    uint32_t n_tokens;
    uint32_t n_slots;      /* routed slots; slot n_slots is the shared expert when has_shared */
    uint32_t in_dim;       /* row length of the expert matrix */
    uint32_t out_rows;     /* rows per expert */
    uint32_t weight_type;  /* 0 f32, 1 f16, 8 q8_0, 10 q2_K, 12 q4_K, 16 iq2_xxs, 39 mxfp4 */
    uint32_t row_bytes;
    uint64_t expert_bytes;
    uint32_t has_shared;
    uint32_t shared_type;
    uint32_t shared_row_bytes;
    uint32_t pad0;
};

/* dot of one quantized expert row with x, lanes split as in the K3 kernels:
 * ix = block stride, it = element pair inside the block */
static inline float qwen4_row_dot(device const char *row, device const float *x,
                                  uint weight_type, uint in_dim, ushort tiisg) {
    float acc = 0.0f;
    if (weight_type == 8) {
        const short ix = tiisg / 8, it = tiisg % 8;
        const uint nb = in_dim / 32;
        for (uint ib = (uint)ix; ib < nb; ib += 4) {
            device const char *b = row + (uint64_t)ib * 34;
            device const float *y = x + ib * 32 + (uint)it * 2;
            const float d = (float)(*(device const half *)b);
            device const char *q = b + 2 + it * 2;
            acc += d * (y[0] * (float)q[0] + y[1] * (float)q[1] +
                        y[16] * (float)q[16] + y[17] * (float)q[17]);
        }
    } else if (weight_type == 39) {
        const short ix = tiisg / 8, it = tiisg % 8;
        const uint nb = in_dim / 32;
        for (uint ib = (uint)ix; ib < nb; ib += 4) {
            device const uchar *b = (device const uchar *)(row + (uint64_t)ib * 17);
            device const float *y = x + ib * 32 + (uint)it * 2;
            const float d = ds4_metal_e8m0_to_f32(b[0]);
            const uint q0 = b[1 + it * 2], q1 = b[2 + it * 2];
            acc += d * (y[0] * ds4_metal_mxfp4_values[q0 & 0xfu] +
                        y[1] * ds4_metal_mxfp4_values[q1 & 0xfu] +
                        y[16] * ds4_metal_mxfp4_values[q0 >> 4] +
                        y[17] * ds4_metal_mxfp4_values[q1 >> 4]);
        }
    } else if (weight_type == 2) {
        const short ix = tiisg / 8, it = tiisg % 8;
        const uint nb = in_dim / 32;
        for (uint ib = (uint)ix; ib < nb; ib += 4) {
            device const uchar *b = (device const uchar *)(row + (uint64_t)ib * 18);
            device const float *y = x + ib * 32 + (uint)it * 2;
            const float d = (float)(*(device const half *)b);
            const uint q0 = b[2 + it * 2], q1 = b[3 + it * 2];
            acc += d * (y[0] * ((float)(q0 & 0xfu) - 8.0f) + y[1] * ((float)(q1 & 0xfu) - 8.0f) +
                        y[16] * ((float)(q0 >> 4) - 8.0f) + y[17] * ((float)(q1 >> 4) - 8.0f));
        }
    } else if (weight_type == 12) {
        /* q4_K: 256-element super-blocks (d, dmin, 12 packed 6-bit scale/min pairs, 128 nibble bytes);
         * lane owns 8 consecutive elements of every block: group = lane/4, l = (lane%4)*8 */
        const uint nb = in_dim / 256;
        const uint group = tiisg / 4, l = (tiisg % 4) * 8;
        for (uint ib = 0; ib < nb; ib++) {
            device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 144);
            const float d = (float)(*(device const half *)blk);
            const float dmin = (float)(*(device const half *)(blk + 2));
            device const uchar *sc = blk + 4;
            uint s, mn;
            if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
            else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
            const float ds = d * (float)s, dm = dmin * (float)mn;
            device const uchar *qs = blk + 16 + (group >> 1) * 32 + l;
            const uint shift = (group & 1u) * 4u;
            device const float *y = x + ib * 256 + group * 32 + l;
            for (uint i = 0; i < 8; i++) acc += (ds * (float)((qs[i] >> shift) & 0xFu) - dm) * y[i];
        }
    } else if (weight_type == 10) {
        /* q2_K: 84-byte super-blocks of 256 (16 scale/min nibble pairs, 64 packed 2-bit bytes, d, dmin);
         * lane owns 8 consecutive elements: group = lane/2 (16 per block), l = (lane%2)*8 */
        const uint nb = in_dim / 256;
        const uint group = tiisg / 2, l = (tiisg % 2) * 8;
        const uint q_base = 32u * (group / 8u) + 16u * (group & 1u), shift = ((group / 2u) & 3u) * 2u;
        for (uint ib = 0; ib < nb; ib++) {
            device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 84);
            const float d = (float)(*(device const half *)(blk + 80));
            const float dmin = (float)(*(device const half *)(blk + 82));
            const uint sc = blk[group];
            const float ds = d * (float)(sc & 0xFu), dm = dmin * (float)(sc >> 4);
            device const uchar *qs = blk + 16 + q_base + l;
            device const float *y = x + ib * 256 + group * 16 + l;
            for (uint i = 0; i < 8; i++) acc += (ds * (float)((qs[i] >> shift) & 3u) - dm) * y[i];
        }
    } else if (weight_type == 16) {
        /* iq2_xxs: 66-byte super-blocks of 256 (d, 8 x 4 u16: 4 grid bytes + 28 sign bits + 4-bit
         * scale per 32); lane owns one 8-value grid entry: sub-block = lane/4, entry = lane%4 */
        const uint nb = in_dim / 256;
        const uint ib32 = tiisg / 4, j = tiisg % 4;
        for (uint ib = 0; ib < nb; ib++) {
            device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 66);
            const float d = (float)(*(device const half *)blk);
            device const ushort *q2 = (device const ushort *)(blk + 2) + 4 * ib32;
            const uint aux_g = (uint)q2[0] | ((uint)q2[1] << 16);
            const uint aux_s = (uint)q2[2] | ((uint)q2[3] << 16);
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2xxs_grid + ((aux_g >> (8 * j)) & 0xFFu));
            const uint signs = ds4_metal_ksigns_iq2xs[(aux_s >> (7 * j)) & 127u];
            device const float *y = x + ib * 256 + ib32 * 32 + j * 8;
            float part = 0.0f;
            for (uint i = 0; i < 8; i++) part += (float)grid[i] * ((signs >> i) & 1u ? -y[i] : y[i]);
            acc += dl * part;
        }
    } else if (weight_type == 30) {
        device const ushort *w = (device const ushort *)row;
        for (uint i = tiisg * 4; i < in_dim; i += 128) {
            acc += as_type<float>((uint)w[i] << 16) * x[i] + as_type<float>((uint)w[i + 1] << 16) * x[i + 1] +
                   as_type<float>((uint)w[i + 2] << 16) * x[i + 2] + as_type<float>((uint)w[i + 3] << 16) * x[i + 3];
        }
    } else if (weight_type == 1) {
        device const half *w = (device const half *)row;
        for (uint i = tiisg * 4; i < in_dim; i += 128) {
            acc += (float)w[i] * x[i] + (float)w[i + 1] * x[i + 1] + (float)w[i + 2] * x[i + 2] + (float)w[i + 3] * x[i + 3];
        }
    } else {
        device const float *w = (device const float *)row;
        for (uint i = tiisg * 4; i < in_dim; i += 128) {
            acc += w[i] * x[i] + w[i + 1] * x[i + 1] + w[i + 2] * x[i + 2] + w[i + 3] * x[i + 3];
        }
    }
    return simd_sum(acc);
}

/* --- small multi-output GEMV (rows via the generic row dot) ------------- */

struct ds4_metal_args_qwen4_gemv {
    uint32_t n_tokens;
    uint32_t in_dim;
    uint32_t n_out;
    uint32_t pad0;
    uint32_t out_rows[4];
    uint32_t types[4];
    uint32_t row_bytes[4];
};

/* Up to four projections of the same input in one dispatch (e.g. the
 * attention k/v/indexer-q/indexer-k rows); also the fallback for weight types
 * the tuned dense GEMV lacks (bf16).  One simdgroup per two rows. */
kernel void kernel_qwen4_multi_gemv(
        constant ds4_metal_args_qwen4_gemv & args,
        device const float *x,          /* [T][in_dim] */
        device const char  *w0,
        device const char  *w1,
        device const char  *w2,
        device const char  *w3,
        device float       *o0,         /* [T][out_rows[i]] */
        device float       *o1,
        device float       *o2,
        device float       *o3,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint total = args.out_rows[0] + args.out_rows[1] + args.out_rows[2] + args.out_rows[3];
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    const uint r0 = (tgpig.x * 4 + (uint)sgitg) * 2;
    for (uint r = r0; r < r0 + 2 && r < total; r++) {
        uint i = 0, local = r;
        while (i + 1 < args.n_out && local >= args.out_rows[i]) { local -= args.out_rows[i]; i++; }
        device const char *w = i == 0 ? w0 : i == 1 ? w1 : i == 2 ? w2 : w3;
        device float *o = i == 0 ? o0 : i == 1 ? o1 : i == 2 ? o2 : o3;
        const float v = qwen4_row_dot(w + (uint64_t)local * args.row_bytes[i], xt, args.types[i], args.in_dim, tiisg);
        if (tiisg == 0) o[(uint64_t)tok * args.out_rows[i] + local] = v;
    }
}

/* mid[t][s][r] = silu(gate_row . x) * (up_row . x) for the selected expert;
 * slot n_slots (when has_shared) is the shared expert from its own bases. */
kernel void kernel_qwen4_moe_mid(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *selected,   /* [T][n_slots] */
        device const float   *x,          /* [T][in_dim] */
        device float         *mid,        /* [T][n_slots+has_shared][out_rows] */
        device const char    *sh_gate,    /* shared expert gate rows */
        device const char    *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * QWEN4_MOE_NSG + (uint)sgitg) * QWEN4_MOE_NR0;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *gb = shared ? sh_gate : gate_base;
    device const char *ub = shared ? sh_up : up_base;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    for (uint r = row0; r < row0 + QWEN4_MOE_NR0 && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * row_bytes;
        const float g = qwen4_row_dot(gb + off, xt, type, args.in_dim, tiisg);
        const float u = qwen4_row_dot(ub + off, xt, type, args.in_dim, tiisg);
        if (tiisg == 0) {
            mid[((uint64_t)tok * n_out + slot) * args.out_rows + r] = qwen4_silu(g) * u;
        }
    }
}

/* part[t][s][r] = down_row . mid[t][s]; slot n_slots is the shared expert */
kernel void kernel_qwen4_moe_down(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *down_base,
        device const int32_t *selected,   /* [T][n_slots] */
        device const float   *mid,        /* [T][n_slots+has_shared][in_dim] */
        device float         *part,       /* [T][n_slots+has_shared][out_rows] */
        device const char    *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * QWEN4_MOE_NSG + (uint)sgitg) * QWEN4_MOE_NR0;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *db = shared ? sh_down : down_base;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *m = mid + pair * args.in_dim;
    for (uint r = row0; r < row0 + QWEN4_MOE_NR0 && r < args.out_rows; r++) {
        const float v = qwen4_row_dot(db + ebase + (uint64_t)r * row_bytes, m, type, args.in_dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
    }
}

struct ds4_metal_args_qwen4_moe_reduce {
    uint32_t n_tokens;
    uint32_t n_slots;
    uint32_t dim;
    uint32_t shared_src;    /* 0 none, 1 part slot n_slots, 2 shared buffer */
    uint32_t n_hc;          /* > 0: also R[s][d] += 2*sigmoid(inj[s]/hc) * out[d] */
    uint32_t part_stride;   /* slots per token in part */
    uint32_t pad1;
    uint32_t pad2;
};

/* out = sum_s weights[s] * part[s] (+ sigmoid(shared_gate) * shared), with
 * the hyper-connection combine folded in when n_hc is set. */
kernel void kernel_qwen4_moe_reduce(
        constant ds4_metal_args_qwen4_moe_reduce & args,
        device const float *part,        /* [T][part_stride][dim] */
        device const float *weights,     /* [T][n_slots] */
        device const float *shared_gate, /* [T] raw logit */
        device float       *out,         /* [T][dim] */
        device float       *R,           /* [T][n_hc*dim] */
        device const float *inj,         /* [T][n_hc*chunks][n_hc] norm partials */
        device const float *shared,      /* [T][dim] when shared_src == 2 */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint d = tgpig.x * ntg.x + tid;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    threadgroup float wgt[8];
    if (args.n_hc && tid < args.n_hc) {
        wgt[tid] = qwen4_hc_inject_weight(inj + (uint64_t)tok * args.n_hc * QWEN4_HC_CHUNKS * args.n_hc, args.n_hc, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= args.dim) return;
    float acc = 0.0f;
    for (uint s = 0; s < args.n_slots; s++) {
        acc += weights[(uint64_t)tok * args.n_slots + s] *
               part[((uint64_t)tok * args.part_stride + s) * args.dim + d];
    }
    if (args.shared_src == 1) {
        acc += qwen4_sigmoid(shared_gate[tok]) * part[((uint64_t)tok * args.part_stride + args.n_slots) * args.dim + d];
    } else if (args.shared_src == 2) {
        acc += qwen4_sigmoid(shared_gate[tok]) * shared[(uint64_t)tok * args.dim + d];
    }
    out[(uint64_t)tok * args.dim + d] = acc;
    if (args.n_hc) {
        device float *r = R + (uint64_t)tok * args.dim * args.n_hc;
        for (uint s = 0; s < args.n_hc; s++) r[s * args.dim + d] += wgt[s] * acc;
    }
}

/* --- prefill: expert-grouped GEMMs -------------------------------------- */

struct ds4_metal_args_qwen4_moe_mm {
    uint32_t n_tokens;
    uint32_t n_slots;
    uint32_t n_out;        /* slots per token in the mid/part layout */
    uint32_t in_dim;
    uint32_t out_rows;
    uint32_t weight_type;  /* 8 q8_0, 10 q2_K, 12 q4_K, 16 iq2_xxs, 39 mxfp4 */
    uint32_t row_bytes;
    uint32_t list_cap;
    uint64_t expert_bytes;
    uint32_t n_expert;
    uint32_t tiles_per_launch;
    uint32_t pad0;
    uint32_t pad1;
};

#define QWEN4_MM_ROWS 32
#define QWEN4_MM_TOKS 8

/* Per-expert (token, slot) lists from the router selection.  One threadgroup
 * per dispatch; counts live in threadgroup memory until the end. */
kernel void kernel_qwen4_moe_build_lists(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const int32_t *selected,   /* [T][n_slots] */
        device int32_t       *lists,      /* [n_expert][list_cap]: t*n_slots + slot */
        device int32_t       *counts,     /* [n_expert] */
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    threadgroup atomic_int cnt[QWEN4_ROUTER_MAX_EXPERT];
    const uint nth = ntg.x;
    for (uint e = tid; e < args.n_expert; e += nth) atomic_store_explicit(&cnt[e], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_pairs = args.n_tokens * args.n_slots;
    for (uint p = tid; p < n_pairs; p += nth) {
        const uint e = (uint)selected[p];
        if (e >= args.n_expert) continue;
        const int slot = atomic_fetch_add_explicit(&cnt[e], 1, memory_order_relaxed);
        if ((uint)slot < args.list_cap) lists[(uint64_t)e * args.list_cap + (uint)slot] = (int32_t)p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < args.n_expert; e += nth) {
        int c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
        counts[e] = c < (int)args.list_cap ? c : (int)args.list_cap;
    }
}

/* dequantize 8 consecutive values (quarter q of 32-wide block b of a row) */
template <typename D>
static inline void qwen4_mm_stage8(device const char *row, uint b, uint q, uint type, threadgroup D *dst) {
    if (type == 2) {
        /* q4_0: 18-byte blocks of 32 (f16 scale, 16 nibble bytes; low nibbles first) */
        device const uchar *blk = (device const uchar *)(row + (uint64_t)b * 18);
        const float d = (float)(*(device const half *)blk);
        device const uchar *qs = blk + 2 + (q & 1u) * 8;
        const bool hi = q >= 2;
        for (uint i = 0; i < 8; i++) {
            const uint nib = hi ? (qs[i] >> 4) : (qs[i] & 0xFu);
            dst[i] = (D)(d * ((float)nib - 8.0f));
        }
        return;
    }
    if (type == 12) {
        const uint sb = b / 8, group = b % 8, l = q * 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 144);
        const float d = (float)(*(device const half *)blk);
        const float dmin = (float)(*(device const half *)(blk + 2));
        device const uchar *sc = blk + 4;
        uint s, mn;
        if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
        else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
        const float ds = d * (float)s, dm = dmin * (float)mn;
        device const uchar *qs = blk + 16 + (group >> 1) * 32 + l;
        const uint shift = (group & 1u) * 4u;
        for (uint i = 0; i < 8; i++) dst[i] = (D)(ds * (float)((qs[i] >> shift) & 0xFu) - dm);
        return;
    }
    if (type == 10) {
        const uint sb = b / 8, group = (b % 8) * 2 + q / 2, l = (q % 2) * 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 84);
        const float d = (float)(*(device const half *)(blk + 80));
        const float dmin = (float)(*(device const half *)(blk + 82));
        const uint sc = blk[group];
        const float ds = d * (float)(sc & 0xFu), dm = dmin * (float)(sc >> 4);
        device const uchar *qs = blk + 16 + 32u * (group / 8u) + 16u * (group & 1u) + l;
        const uint shift = ((group / 2u) & 3u) * 2u;
        for (uint i = 0; i < 8; i++) dst[i] = (D)(ds * (float)((qs[i] >> shift) & 3u) - dm);
        return;
    }
    if (type == 16) {
        const uint sb = b / 8, ib32 = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 66);
        const float d = (float)(*(device const half *)blk);
        device const ushort *q2 = (device const ushort *)(blk + 2) + 4 * ib32;
        const uint aux_g = (uint)q2[0] | ((uint)q2[1] << 16);
        const uint aux_s = (uint)q2[2] | ((uint)q2[3] << 16);
        const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
        constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2xxs_grid + ((aux_g >> (8 * q)) & 0xFFu));
        const uint signs = ds4_metal_ksigns_iq2xs[(aux_s >> (7 * q)) & 127u];
        for (uint i = 0; i < 8; i++) dst[i] = (D)(dl * (float)grid[i] * ((signs >> i) & 1u ? -1.0f : 1.0f));
        return;
    }
    if (type == 8) {
        device const char *blk = row + (uint64_t)b * 34;
        const float d = (float)(*(device const half *)blk);
        const packed_char4 q0 = *(device const packed_char4 *)(blk + 2 + q * 8);   /* 2-byte aligned */
        const packed_char4 q1 = *(device const packed_char4 *)(blk + 6 + q * 8);
        dst[0] = (D)(d * (float)q0.x); dst[1] = (D)(d * (float)q0.y); dst[2] = (D)(d * (float)q0.z); dst[3] = (D)(d * (float)q0.w);
        dst[4] = (D)(d * (float)q1.x); dst[5] = (D)(d * (float)q1.y); dst[6] = (D)(d * (float)q1.z); dst[7] = (D)(d * (float)q1.w);
    } else {
        device const uchar *blk = (device const uchar *)(row + (uint64_t)b * 17);
        const float d = ds4_metal_e8m0_to_f32(blk[0]);
        const uint base = (q & 1u) * 8;
        const bool hi = q >= 2;
        for (uint i = 0; i < 8; i++) {
            const uint byte = blk[1 + base + i];
            dst[i] = (D)(d * ds4_metal_mxfp4_values[hi ? (byte >> 4) : (byte & 0xfu)]);
        }
    }
}

/* dequantize 16 consecutive values (quarters q0 and q0 + 1 of block b, q0
 * even): the K-quant scales are unpacked once and the nibbles read as one
 * 16-byte word; other types take two 8-value steps */
template <typename D>
static inline void qwen4_mm_stage16(device const char *row, uint b, uint q0, uint type, threadgroup D *dst) {
    if (type == 12) {
        const uint sb = b / 8, group = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 144);
        const float d = (float)(*(device const half *)blk);
        const float dmin = (float)(*(device const half *)(blk + 2));
        device const uchar *sc = blk + 4;
        uint s, mn;
        if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
        else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
        const float ds = d * (float)s, dm = dmin * (float)mn;
        const uint4 v = *(device const uint4 *)(blk + 16 + (group >> 1) * 32 + q0 * 8);
        const uint shift = (group & 1u) * 4u;
        for (uint i = 0; i < 16; i++) dst[i] = (D)(ds * (float)((v[i >> 2] >> (8u * (i & 3u) + shift)) & 0xFu) - dm);
        return;
    }
    if (type == 10) {
        const uint sb = b / 8, group = (b % 8) * 2 + q0 / 2;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 84);
        const float d = (float)(*(device const half *)(blk + 80));
        const float dmin = (float)(*(device const half *)(blk + 82));
        const uint sc = blk[group];
        const float ds = d * (float)(sc & 0xFu), dm = dmin * (float)(sc >> 4);
        device const uint *qs = (device const uint *)(blk + 16 + 32u * (group / 8u) + 16u * (group & 1u));
        const uint shift = ((group / 2u) & 3u) * 2u;
        for (uint i = 0; i < 16; i++) dst[i] = (D)(ds * (float)((qs[i >> 2] >> (8u * (i & 3u) + shift)) & 3u) - dm);
        return;
    }
    qwen4_mm_stage8(row, b, q0, type, dst);
    qwen4_mm_stage8(row, b, q0 + 1, type, dst + 8);
}

#define QWEN4_MM_KS 64   /* K per staging step (two 32-blocks) */
#define QWEN4_MM_NT 4    /* 8-token tiles per staged weight tile */
#define QWEN4_MM_TT (QWEN4_MM_TOKS * QWEN4_MM_NT)

/* mid[t][slot][r] = silu(gate . x) * (up . x) for every (token, slot) routed
 * to expert e, as 32-row x 32-token tiles: A (weights, half) and B
 * (activations, half) are staged in threadgroup memory per 64-wide K step
 * and multiplied with simdgroup matrices into float accumulators, so each
 * expert row is read once per token tile. */
kernel void kernel_qwen4_moe_mm_mid(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const float   *x,          /* [T][in_dim] */
        device float         *mid,        /* [T][n_out][out_rows] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint rb = tgpig.x, e = tgpig.y;
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    threadgroup half Ag[QWEN4_MM_ROWS * QWEN4_MM_KS];
    threadgroup half Au[QWEN4_MM_ROWS * QWEN4_MM_KS];
    threadgroup half Bs[QWEN4_MM_KS * QWEN4_MM_TT];
    threadgroup float Cs[4][2][64];
    device const char *gbase = gate_base + (uint64_t)e * args.expert_bytes;
    device const char *ubase = up_base + (uint64_t)e * args.expert_bytes;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * QWEN4_MM_ROWS;
    const uint nk = args.in_dim / QWEN4_MM_KS;
    for (uint tile = tgpig.z; tile * QWEN4_MM_TT < count; tile += args.tiles_per_launch) {
        const uint t0 = tile * QWEN4_MM_TT;
        const uint n_tile = min((uint)QWEN4_MM_TT, count - t0);
        simdgroup_float8x8 Cg[QWEN4_MM_NT], Cu[QWEN4_MM_NT];
        for (uint nt = 0; nt < QWEN4_MM_NT; nt++) {
            Cg[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            Cu[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        const uint my_tok = tid % QWEN4_MM_TT;
        const int my_pair = my_tok < n_tile ? list[t0 + my_tok] : -1;
        const uint my_t = my_pair >= 0 ? (uint)my_pair / args.n_slots : 0;
        for (uint kb = 0; kb < nk; kb++) {
            /* A: 32 rows x 64 k; thread = (row, 16-wide slice) */
            {
                const uint r = tid / 4, q = tid % 4;   /* q: 16-value half of one of the two 32-blocks */
                threadgroup half *dg = Ag + r * QWEN4_MM_KS + q * 16;
                threadgroup half *du = Au + r * QWEN4_MM_KS + q * 16;
                if (row0 + r < args.out_rows) {
                    device const char *grow = gbase + (uint64_t)(row0 + r) * args.row_bytes;
                    device const char *urow = ubase + (uint64_t)(row0 + r) * args.row_bytes;
                    const uint b = kb * 2 + (q >> 1), quarter0 = (q & 1) * 2;
                    qwen4_mm_stage16(grow, b, quarter0, args.weight_type, dg);
                    qwen4_mm_stage16(urow, b, quarter0, args.weight_type, du);
                } else {
                    for (uint i = 0; i < 16; i++) { dg[i] = 0.0h; du[i] = 0.0h; }
                }
            }
            /* B: 64 k x 32 tokens; thread = (token, 16 k values) */
            {
                const uint tok = tid % QWEN4_MM_TT, kq = tid / QWEN4_MM_TT;
                device const float *xr = x + (uint64_t)my_t * args.in_dim + kb * QWEN4_MM_KS + kq * 16;
                for (uint j = 0; j < 16; j += 4) {
                    const float4 v = my_pair >= 0 ? *(device const float4 *)(xr + j) : float4(0.0f);
                    Bs[(kq * 16 + j + 0) * QWEN4_MM_TT + tok] = (half)v.x;
                    Bs[(kq * 16 + j + 1) * QWEN4_MM_TT + tok] = (half)v.y;
                    Bs[(kq * 16 + j + 2) * QWEN4_MM_TT + tok] = (half)v.z;
                    Bs[(kq * 16 + j + 3) * QWEN4_MM_TT + tok] = (half)v.w;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint sub = 0; sub < QWEN4_MM_KS / 8; sub++) {
                simdgroup_half8x8 ag, au, b;
                simdgroup_load(ag, Ag + (sgitg * 8) * QWEN4_MM_KS + sub * 8, QWEN4_MM_KS, 0, false);
                simdgroup_load(au, Au + (sgitg * 8) * QWEN4_MM_KS + sub * 8, QWEN4_MM_KS, 0, false);
                for (uint nt = 0; nt < QWEN4_MM_NT; nt++) {
                    simdgroup_load(b, Bs + sub * 8 * QWEN4_MM_TT + nt * 8, QWEN4_MM_TT, 0, false);
                    simdgroup_multiply_accumulate(Cg[nt], ag, b, Cg[nt]);
                    simdgroup_multiply_accumulate(Cu[nt], au, b, Cu[nt]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint nt = 0; nt < QWEN4_MM_NT; nt++) {
            simdgroup_store(Cg[nt], Cs[sgitg][0], 8, 0, false);
            simdgroup_store(Cu[nt], Cs[sgitg][1], 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < 4 * 64; idx += 128) {
                const uint sg = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                const uint row = row0 + sg * 8 + r;
                if (tok >= n_tile || row >= args.out_rows) continue;
                const int pair = list[t0 + tok];
                const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
                const float g = Cs[sg][0][el], u = Cs[sg][1][el];
                mid[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = qwen4_silu(g) * u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/* part[t][slot][r] = down . mid[t][slot], same tiling with mid as B */
kernel void kernel_qwen4_moe_mm_down(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *down_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const float   *midv,       /* [T][n_out][in_dim] */
        device float         *part,       /* [T][n_out][out_rows] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint rb = tgpig.x, e = tgpig.y;
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    threadgroup half As[QWEN4_MM_ROWS * QWEN4_MM_KS];
    threadgroup half Bs[QWEN4_MM_KS * QWEN4_MM_TT];
    threadgroup float Cs[4][64];
    device const char *dbase = down_base + (uint64_t)e * args.expert_bytes;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * QWEN4_MM_ROWS;
    const uint nk = args.in_dim / QWEN4_MM_KS;
    for (uint tile = tgpig.z; tile * QWEN4_MM_TT < count; tile += args.tiles_per_launch) {
        const uint t0 = tile * QWEN4_MM_TT;
        const uint n_tile = min((uint)QWEN4_MM_TT, count - t0);
        simdgroup_float8x8 C[QWEN4_MM_NT];
        for (uint nt = 0; nt < QWEN4_MM_NT; nt++) C[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        const uint my_tok = tid % QWEN4_MM_TT;
        const int my_pair = my_tok < n_tile ? list[t0 + my_tok] : -1;
        const uint64_t my_row = my_pair >= 0 ?
            ((uint64_t)((uint)my_pair / args.n_slots) * args.n_out + (uint)my_pair % args.n_slots) : 0;
        for (uint kb = 0; kb < nk; kb++) {
            {
                const uint r = tid / 4, q = tid % 4;
                threadgroup half *dd = As + r * QWEN4_MM_KS + q * 16;
                if (row0 + r < args.out_rows) {
                    device const char *drow = dbase + (uint64_t)(row0 + r) * args.row_bytes;
                    const uint b = kb * 2 + (q >> 1), quarter0 = (q & 1) * 2;
                    qwen4_mm_stage16(drow, b, quarter0, args.weight_type, dd);
                } else {
                    for (uint i = 0; i < 16; i++) dd[i] = 0.0h;
                }
            }
            {
                const uint tok = tid % QWEN4_MM_TT, kq = tid / QWEN4_MM_TT;
                device const float *mr = midv + my_row * args.in_dim + kb * QWEN4_MM_KS + kq * 16;
                for (uint j = 0; j < 16; j += 4) {
                    const float4 v = my_pair >= 0 ? *(device const float4 *)(mr + j) : float4(0.0f);
                    Bs[(kq * 16 + j + 0) * QWEN4_MM_TT + tok] = (half)v.x;
                    Bs[(kq * 16 + j + 1) * QWEN4_MM_TT + tok] = (half)v.y;
                    Bs[(kq * 16 + j + 2) * QWEN4_MM_TT + tok] = (half)v.z;
                    Bs[(kq * 16 + j + 3) * QWEN4_MM_TT + tok] = (half)v.w;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint sub = 0; sub < QWEN4_MM_KS / 8; sub++) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, As + (sgitg * 8) * QWEN4_MM_KS + sub * 8, QWEN4_MM_KS, 0, false);
                for (uint nt = 0; nt < QWEN4_MM_NT; nt++) {
                    simdgroup_load(b, Bs + sub * 8 * QWEN4_MM_TT + nt * 8, QWEN4_MM_TT, 0, false);
                    simdgroup_multiply_accumulate(C[nt], a, b, C[nt]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint nt = 0; nt < QWEN4_MM_NT; nt++) {
            simdgroup_store(C[nt], Cs[sgitg], 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < 4 * 64; idx += 128) {
                const uint sg = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                const uint row = row0 + sg * 8 + r;
                if (tok >= n_tile || row >= args.out_rows) continue;
                const int pair = list[t0 + tok];
                const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
                part[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = Cs[sg][el];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/* --- prefill: dense tiled GEMM for f32/f16/q8_0 weights ----------------- */

struct ds4_metal_args_qwen4_dense_mm {
    uint32_t n_tokens;
    uint32_t in_dim;
    uint32_t out_rows;
    uint32_t weight_type;   /* 0 f32, 1 f16, 8 q8_0 */
    uint32_t row_bytes;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

#define QWEN4_DM_TOKS 32
#define QWEN4_DM_K 32

/* 8 consecutive weights of row `row` starting at element k0 (k0 % 8 == 0);
 * f32/f16 rows may end mid-tile (in_dim % 32 != 0), q8_0 rows cannot */
static inline void qwen4_dm_stage8(device const char *row, uint k0, uint k_end, uint type, threadgroup float *dst) {
    if (type == 8) {
        qwen4_mm_stage8<float>(row, k0 / 32, (k0 % 32) / 8, 8u, dst);
    } else if (type == 1) {
        device const half *w = (device const half *)row + k0;
        for (uint i = 0; i < 8; i++) dst[i] = k0 + i < k_end ? (float)w[i] : 0.0f;
    } else {
        device const float *w = (device const float *)row + k0;
        for (uint i = 0; i < 8; i++) dst[i] = k0 + i < k_end ? w[i] : 0.0f;
    }
}

/* out[t][r] = w[r] . x[t] as 32-row x 32-token tiles; weights are read once
 * per 32 tokens.  Grid (rows/32, tokens/32), 128 threads. */
kernel void kernel_qwen4_dense_mm(
        constant ds4_metal_args_qwen4_dense_mm & args,
        device const char  *w,          /* [out_rows] rows */
        device const float *x,          /* [T][in_dim] */
        device float       *out,        /* [T][out_rows] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = tgpig.x * QWEN4_MM_ROWS;
    const uint t0 = tgpig.y * QWEN4_DM_TOKS;
    if (row0 >= args.out_rows || t0 >= args.n_tokens) return;
    threadgroup float As[QWEN4_MM_ROWS * QWEN4_DM_K];
    threadgroup float Bs[QWEN4_DM_K * QWEN4_DM_TOKS];
    threadgroup float Cs[4][4][64];
    simdgroup_float8x8 C[4];
    for (uint j = 0; j < 4; j++) C[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint nk = (args.in_dim + QWEN4_DM_K - 1) / QWEN4_DM_K;
    for (uint kb = 0; kb < nk; kb++) {
        {
            const uint r = tid / 4, q = tid % 4;
            if (row0 + r < args.out_rows) {
                qwen4_dm_stage8(w + (uint64_t)(row0 + r) * args.row_bytes, kb * QWEN4_DM_K + q * 8, args.in_dim,
                                args.weight_type, As + r * QWEN4_DM_K + q * 8);
            } else {
                for (uint i = 0; i < 8; i++) As[r * QWEN4_DM_K + q * 8 + i] = 0.0f;
            }
        }
        {
            /* B: 32 k x 32 tokens, thread = (token, 8 k values) */
            const uint tok = tid % QWEN4_DM_TOKS, kq = tid / QWEN4_DM_TOKS;
            const uint t = t0 + tok;
            for (uint i = 0; i < 8; i++) {
                const uint k = kq * 8 + i;
                Bs[k * QWEN4_DM_TOKS + tok] = t < args.n_tokens && kb * QWEN4_DM_K + k < args.in_dim ?
                                              x[(uint64_t)t * args.in_dim + kb * QWEN4_DM_K + k] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sub = 0; sub < QWEN4_DM_K / 8; sub++) {
            simdgroup_float8x8 a;
            simdgroup_load(a, As + (sgitg * 8) * QWEN4_DM_K + sub * 8, QWEN4_DM_K, 0, false);
            for (uint j = 0; j < 4; j++) {
                simdgroup_float8x8 b;
                simdgroup_load(b, Bs + sub * 8 * QWEN4_DM_TOKS + j * 8, QWEN4_DM_TOKS, 0, false);
                simdgroup_multiply_accumulate(C[j], a, b, C[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 4; j++) simdgroup_store(C[j], Cs[sgitg][j], 8, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 4 * 4 * 64; idx += 128) {
        const uint sg = idx / 256, rem = idx % 256, j = rem / 64, el = rem % 64, r = el / 8, tok = el % 8;
        const uint row = row0 + sg * 8 + r;
        const uint t = t0 + j * 8 + tok;
        if (row < args.out_rows && t < args.n_tokens) out[(uint64_t)t * args.out_rows + row] = Cs[sg][j][el];
    }
}

struct ds4_metal_args_qwen4_hc_mix_rows {
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t n_rank;
};

/* prefill hc: lo_act = silu(lo/hc) (per token, n_rank) */
kernel void kernel_qwen4_hc_lo_act(
        constant ds4_metal_args_qwen4_hc_mix_rows & args,
        device const float *lo,
        device float       *lo_act,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_tokens * args.n_rank) return;
    lo_act[gid] = qwen4_silu(lo[gid] / (float)args.n_hc);
}

/* prefill hc: mixed[t][d] = mean_s sigmoid(u[t][s*E+d]) * xn[t][s*E+d] */
kernel void kernel_qwen4_hc_mix_rows(
        constant ds4_metal_args_qwen4_hc_mix_rows & args,
        device const float *u,          /* [T][hc*E] */
        device const float *xn,         /* [T][hc*E] */
        device float       *mixed,      /* [T][E] */
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, t = gid.y;
    if (d >= args.n_embd || t >= args.n_tokens) return;
    const uint64_t base = (uint64_t)t * args.n_embd * args.n_hc;
    float acc = 0.0f;
    for (uint s = 0; s < args.n_hc; s++) acc += qwen4_sigmoid(u[base + s * args.n_embd + d]) * xn[base + s * args.n_embd + d];
    mixed[(uint64_t)t * args.n_embd + d] = acc / (float)args.n_hc;
}

/* --- multi-token prediction input --------------------------------------- */

struct ds4_metal_args_qwen4_mtp_stage {
    uint32_t n_embd;
    uint32_t n_hc;
    uint32_t pad0;
    float    eps;
};

/* Rows of the concat input for the fused [W_e | W_h] projection: row 0 is
 * [e/rms(e) * g_e | 0], row 1+s is [0 | R_s/rms(R) * g_h[s]] with one RMS
 * over all hc streams.  One threadgroup per row. */
kernel void kernel_qwen4_mtp_stage(
        constant ds4_metal_args_qwen4_mtp_stage & args,
        device const float *e,          /* [E] next-token embedding */
        device const float *R,          /* [hc*E] pre-mixer streams */
        device const float *g_e,        /* [E] */
        device const float *g_h,        /* [hc*E] */
        device float       *cat,        /* [1+hc][2E] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    if (row > args.n_hc) return;
    const uint E = args.n_embd;
    const uint nth = ntg.x;
    const uint nsg = nth / 32;
    threadgroup float red[32];
    const bool emb = row == 0;
    device const float *src = emb ? e : R + (uint64_t)(row - 1u) * E;
    device const float *g = emb ? g_e : g_h + (uint64_t)(row - 1u) * E;
    device const float *rs = emb ? src : R;
    const uint n_red = emb ? E : E * args.n_hc;
    float ss = 0.0f;
    for (uint i = tid; i < n_red; i += nth) ss += rs[i] * rs[i];
    ss = simd_sum(ss);
    if (tiisg == 0) red[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint q = 0; q < nsg; q++) tot += red[q];
    const float inv = rsqrt(tot / (float)n_red + args.eps);
    device float *o = cat + (uint64_t)row * 2u * E;
    const uint lo = emb ? 0u : E;
    const uint hi = emb ? E : 0u;
    for (uint i = tid; i < E; i += nth) {
        o[lo + i] = src[i] * inv * g[i];
        o[hi + i] = 0.0f;
    }
}

struct ds4_metal_args_qwen4_mtp_combine {
    uint32_t n_embd;
    uint32_t n_hc;
};

/* R_out[s][d] = proj[0][d] + proj[1+s][d] */
kernel void kernel_qwen4_mtp_combine(
        constant ds4_metal_args_qwen4_mtp_combine & args,
        device const float *proj,       /* [1+hc][E] */
        device float       *R_out,      /* [hc*E] */
        uint gid [[thread_position_in_grid]]) {
    const uint E = args.n_embd;
    if (gid >= E * args.n_hc) return;
    const uint s = gid / E;
    const uint d = gid - s * E;
    R_out[gid] = proj[d] + proj[(uint64_t)(s + 1u) * E + d];
}

/* --- decode-path fusions ------------------------------------------------ */

struct ds4_metal_args_qwen4_gdn_front {
    uint32_t n_tokens;
    uint32_t n_k_head;
    uint32_t n_v_head;
    uint32_t head_dim;
    uint32_t conv_kernel;
    uint32_t weight_type;
    uint32_t in_dim;
    uint32_t row_bytes;
    uint32_t snap_tok;     /* copy the conv history after this token into snap_state */
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

/* conv_stream + alpha/beta projections + gdn_prep for a few tokens.  One
 * threadgroup per k-head owns q_h, k_h and the value heads tiled onto it
 * (j % Hk == h): its threads run the conv over those channels, then
 * simdgroups 0/1 normalize q/k and the others take the alpha/beta rows.
 * Tokens are walked in order; the conv history is advanced in place. */
kernel void kernel_qwen4_gdn_front(
        constant ds4_metal_args_qwen4_gdn_front & args,
        device float       *qkv,      /* [T][C] raw in; conv'd, q/k normalized out */
        device float       *state,    /* [K-1][C] */
        device const float *conv_w,   /* [C][K] */
        device const float *mixed,    /* [T][in_dim] */
        device const char  *w_alpha,  /* [Hv] rows */
        device const char  *w_beta,   /* [Hv] rows */
        device const float *ssm_a,    /* [Hv] */
        device const float *dt_bias,  /* [Hv] */
        device float       *ga,       /* [T][Hv] decay out */
        device float       *gb,       /* [T][Hv] beta out */
        device float       *snap_state,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint Hk = args.n_k_head, Hv = args.n_v_head, D = args.head_dim, K = args.conv_kernel;
    if (h >= Hk) return;
    const uint C = 2 * Hk * D + Hv * D;
    const uint per_k = Hv / Hk;
    const uint n_ch = (2 + per_k) * D;
    const uint nth = ntg.x;
    const uint nsg = nth / 32;
    const uint npt = D / 32;
    for (uint tok = 0; tok < args.n_tokens; tok++) {
        device float *row = qkv + (uint64_t)tok * C;
        for (uint cl = tid; cl < n_ch; cl += nth) {
            const uint grp = cl / D, i = cl - grp * D;
            const uint c = (grp == 0 ? h * D : grp == 1 ? Hk * D + h * D : 2 * Hk * D + (h + (grp - 2) * Hk) * D) + i;
            const float raw = row[c];
            float acc = conv_w[c * K + K - 1] * raw;
            for (uint t = 0; t + 1 < K; t++) acc += conv_w[c * K + t] * state[t * C + c];
            for (uint t = 0; t + 2 < K; t++) state[t * C + c] = state[(t + 1) * C + c];
            state[(K - 2) * C + c] = raw;
            row[c] = qwen4_silu(acc);
            if (tok == args.snap_tok) {
                for (uint t = 0; t + 1 < K; t++) snap_state[t * C + c] = state[t * C + c];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (sgitg < 2) {
            device float *v = row + (sgitg == 0 ? h * D : Hk * D + h * D);
            float ss = 0.0f;
            for (uint r = 0; r < npt; r++) ss += v[tiisg + 32 * r] * v[tiisg + 32 * r];
            ss = simd_sum(ss);
            const float sc = rsqrt(ss + 1e-6f) * (sgitg == 0 ? rsqrt((float)D) : 1.0f);
            for (uint r = 0; r < npt; r++) v[tiisg + 32 * r] *= sc;
        } else {
            device const float *x = mixed + (uint64_t)tok * args.in_dim;
            for (uint rr = (uint)sgitg - 2u; rr < 2 * per_k; rr += nsg - 2u) {
                const uint j = h + (rr / 2) * Hk;
                device const char *wrow = (rr & 1u) ? w_beta : w_alpha;
                const float v = qwen4_row_dot(wrow + (uint64_t)j * args.row_bytes, x, args.weight_type, args.in_dim, tiisg);
                if (tiisg == 0) {
                    if (rr & 1u) gb[(uint64_t)tok * Hv + j] = qwen4_sigmoid(v);
                    else ga[(uint64_t)tok * Hv + j] = exp(ssm_a[j] * qwen4_softplus(v + dt_bias[j]));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
