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

/* Large batches have enough (token, stream) groups to compute the stream RMS
 * once and reuse it for all eight chunks.  Keep the 128-thread RMS reduction,
 * each chunk's injection reduction, and the partial layout identical to the
 * original kernel: combining the chunk dots would change rounding. */
template <typename W>
kernel void kernel_qwen4_hc_norm_reuse(
        constant ds4_metal_args_qwen4_hc_norm & args,
        device const float *R,
        device const float *gamma,
        device const char  *w_inject,
        device float       *xn,
        device float       *inj_part,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint s = tgpig.x;
    const uint tok = tgpig.y;
    if (s >= args.n_hc || tok >= args.n_tokens) return;
    const uint E = args.n_embd, dim = E * args.n_hc;
    const uint nth = ntg.x, nsg = nth / 32;
    threadgroup float red[32];
    /* This kernel uses 128 threads. Keep each chunk's four SIMD partials
     * separate so its readers need no barrier before the next chunk writes. */
    threadgroup float inject_red[QWEN4_HC_CHUNKS][4][4];
    device const float *r = R + ((uint64_t)tok * args.n_hc + s) * E;
    device const float *g = gamma + s * E;
    device float *o = xn + ((uint64_t)tok * args.n_hc + s) * E;
    const W w(w_inject);
    float ss = 0.0f;
    for (uint i = tid; i < E; i += nth) ss += r[i] * r[i];
    ss = simd_sum(ss);
    if (tiisg == 0) red[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint q = 0; q < nsg; q++) tot += red[q];
    const float inv = rsqrt(tot / (float)E + args.eps);
    const uint per = (E + QWEN4_HC_CHUNKS - 1) / QWEN4_HC_CHUNKS;
    for (uint chunk = 0; chunk < QWEN4_HC_CHUNKS; chunk++) {
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
            if (tiisg == 0) inject_red[chunk][j][sgitg] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < args.n_inject) {
            float a = 0.0f;
            for (uint q = 0; q < nsg; q++) a += inject_red[chunk][tid][q];
            inj_part[((uint64_t)tok * args.n_hc * QWEN4_HC_CHUNKS + s * QWEN4_HC_CHUNKS + chunk) * args.n_inject + tid] = a;
        }
    }
}

#define QWEN4_HC_NORM_REUSE_INSTANCE(SUFFIX, W) \
template [[host_name("kernel_qwen4_hc_norm_reuse_" #SUFFIX)]] \
kernel void kernel_qwen4_hc_norm_reuse<W>(constant ds4_metal_args_qwen4_hc_norm &, device const float *, \
        device const float *, device const char *, device float *, device float *, uint3, ushort, ushort3, ushort, ushort);
QWEN4_HC_NORM_REUSE_INSTANCE(f16, qwen4_w_f16)
QWEN4_HC_NORM_REUSE_INSTANCE(f32, qwen4_w_f32)
QWEN4_HC_NORM_REUSE_INSTANCE(q8, qwen4_w_q8)

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
    for (uint r = lane; r < args.n_rank; r += 8) {
        if constexpr (is_same<W, qwen4_w_f16>::value) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            /* Match the F16 prefetch/reuse kernels without depending on
             * the device compiler to reassociate weight * silu(x). */
            const float x = l[r] * (1.0f / (float)hc);
            const float t = x * w.at(row + r);
            const float u = t * qwen4_sigmoid(x);
            acc = acc + u;
        } else {
            acc += w.at(row + r) * qwen4_silu(l[r] / (float)hc);
        }
    }
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

/* F16 gate/mix with eight terms loaded ahead per lane round. The generic,
 * prefetched and reuse kernels all pin acc += (x*w)*sigmoid(x) without FMA;
 * loading ahead only overlaps memory reads with the sigmoid chain. */
kernel void kernel_qwen4_hc_gate_mix_f16_pf(
        constant ds4_metal_args_qwen4_hc_gate_mix & args,
        device const float *xn,
        device const float *lo,
        device const char  *w_up,
        device float       *mixed,
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
    device const half *wr = (device const half *)w_up + (uint64_t)(s * E + d) * args.n_rank;
    const float inv_hc = 1.0f / (float)hc;
    float acc = 0.0f;
    uint r = lane;
    for (; r + 56u < args.n_rank; r += 64u) {
        float wv[8], lv[8];
        for (uint i = 0; i < 8; i++) { wv[i] = (float)wr[r + 8u * i]; lv[i] = l[r + 8u * i]; }
        for (uint i = 0; i < 8; i++) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            const float x = lv[i] * inv_hc;
            const float sig = qwen4_sigmoid(x);
            const float t = x * wv[i];
            const float u = t * sig;
            acc = acc + u;
        }
    }
    for (; r < args.n_rank; r += 8u) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
        const float x = l[r] * inv_hc;
        const float sig = qwen4_sigmoid(x);
        const float t = x * (float)wr[r];
        const float u = t * sig;
        acc = acc + u;
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    float g = qwen4_sigmoid(acc) * xn[(uint64_t)tok * E * hc + s * E + d];
    g += simd_shuffle_xor(g, 8);
    g += simd_shuffle_xor(g, 16);
    if (tiisg == 0) mixed[(uint64_t)tok * E + d] = g / (float)hc;
}

/* Reuse the low-rank sigmoid across streams and output rows. Keep x and
 * sigmoid(x) separate: storing silu(x) would change the single-row
 * accumulation's (x * weight) * sigmoid(x) rounding. */
kernel void kernel_qwen4_hc_gate_mix_f16_reuse(
        constant ds4_metal_args_qwen4_hc_gate_mix & args,
        device const float *xn,
        device const float *lo,
        device const char  *w_up,
        device float       *mixed,
        threadgroup float2 *low_sig [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hc = args.n_hc, E = args.n_embd, rank = args.n_rank;
    const uint d = tgpig.x * (ntg.x / 32) + sgitg;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    device const float *l = lo + (uint64_t)tok * rank;
    const float inv_hc = 1.0f / (float)hc;
    for (uint r = tid; r < rank; r += ntg.x) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
        const float x = l[r] * inv_hc;
        low_sig[r] = float2(x, qwen4_sigmoid(x));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= E) return;
    const uint s = tiisg / 8, lane = tiisg % 8;
    device const half *wr = (device const half *)w_up + (uint64_t)(s * E + d) * rank;
    float acc = 0.0f;
    uint r = lane;
    for (; r + 56u < rank; r += 64u) {
        float wv[8];
        float2 av[8];
        for (uint i = 0; i < 8; i++) { wv[i] = (float)wr[r + 8u * i]; av[i] = low_sig[r + 8u * i]; }
        for (uint i = 0; i < 8; i++) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            const float t = av[i].x * wv[i];
            const float u = t * av[i].y;
            acc = acc + u;
        }
    }
    for (; r < rank; r += 8u) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
        const float2 a = low_sig[r];
        const float t = a.x * (float)wr[r];
        const float u = t * a.y;
        acc = acc + u;
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    float g = qwen4_sigmoid(acc) * xn[(uint64_t)tok * E * hc + s * E + d];
    g += simd_shuffle_xor(g, 8);
    g += simd_shuffle_xor(g, 16);
    if (tiisg == 0) mixed[(uint64_t)tok * E + d] = g / (float)hc;
}

/* Two-token MTP verification: reuse each up-projection weight for both
 * rows, and activate the low-rank inputs once per threadgroup. */
template <typename W>
kernel void kernel_qwen4_hc_gate_mix_pair(
        constant ds4_metal_args_qwen4_hc_gate_mix & args,
        device const float *xn,
        device const float *lo,
        device const char *w_up,
        device float *mixed,
        threadgroup float2 *activated [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint E = args.n_embd, hc = args.n_hc, rank = args.n_rank;
    const uint d = tgpig.x * (ntg.x / 32) + sgitg;
    for (uint r = tid; r < rank; r += ntg.x) {
        activated[r] = float2(qwen4_silu(lo[r] / (float)hc),
                              qwen4_silu(lo[rank + r] / (float)hc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= E) return;
    const uint s = tiisg / 8, lane = tiisg % 8;
    const W w(w_up);
    const uint64_t row = (uint64_t)(s * E + d) * rank;
    float2 acc = 0.0f;
    for (uint r = lane; r < rank; r += 8) {
        if constexpr (is_same<W, qwen4_w_f16>::value) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            /* Paired F16 rows retain their original fused accumulation. */
            acc = fma(float2(w.at(row + r)), activated[r], acc);
        } else {
            acc += w.at(row + r) * activated[r];
        }
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    float2 v = float2(qwen4_sigmoid(acc.x) * xn[s * E + d],
                      qwen4_sigmoid(acc.y) * xn[E * hc + s * E + d]);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 16);
    if (tiisg == 0) {
        mixed[d] = v.x / (float)hc;
        mixed[E + d] = v.y / (float)hc;
    }
}

/* Paired F16 gate/mix with eight weights and activated pairs loaded ahead.
 * Both paired kernels pin the same FMA chain, unlike the single-row mixer. */
kernel void kernel_qwen4_hc_gate_mix_pair_f16_pf(
        constant ds4_metal_args_qwen4_hc_gate_mix & args,
        device const float *xn,
        device const float *lo,
        device const char *w_up,
        device float *mixed,
        threadgroup float2 *activated [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint E = args.n_embd, hc = args.n_hc, rank = args.n_rank;
    const uint d = tgpig.x * (ntg.x / 32) + sgitg;
    for (uint r = tid; r < rank; r += ntg.x) {
        activated[r] = float2(qwen4_silu(lo[r] / (float)hc),
                              qwen4_silu(lo[rank + r] / (float)hc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= E) return;
    const uint s = tiisg / 8, lane = tiisg % 8;
    device const half *wr = (device const half *)w_up + (uint64_t)(s * E + d) * rank;
    float2 acc = 0.0f;
    uint r = lane;
    for (; r + 56u < rank; r += 64u) {
        float wv[8];
        float2 av[8];
        for (uint i = 0; i < 8; i++) { wv[i] = (float)wr[r + 8u * i]; av[i] = activated[r + 8u * i]; }
        for (uint i = 0; i < 8; i++) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            acc = fma(float2(wv[i]), av[i], acc);
        }
    }
    for (; r < rank; r += 8u) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
        acc = fma(float2((float)wr[r]), activated[r], acc);
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    float2 v = float2(qwen4_sigmoid(acc.x) * xn[s * E + d],
                      qwen4_sigmoid(acc.y) * xn[E * hc + s * E + d]);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 16);
    if (tiisg == 0) {
        mixed[d] = v.x / (float)hc;
        mixed[E + d] = v.y / (float)hc;
    }
}

#define QWEN4_HC_MIX_PAIR_INSTANCE(SUFFIX, W) \
template [[host_name("kernel_qwen4_hc_gate_mix_pair_" #SUFFIX)]] \
kernel void kernel_qwen4_hc_gate_mix_pair<W>(constant ds4_metal_args_qwen4_hc_gate_mix &, device const float *, \
        device const float *, device const char *, device float *, threadgroup float2 *, uint3, ushort, ushort3, ushort, ushort);
QWEN4_HC_MIX_PAIR_INSTANCE(f16, qwen4_w_f16)
QWEN4_HC_MIX_PAIR_INSTANCE(f32, qwen4_w_f32)
QWEN4_HC_MIX_PAIR_INSTANCE(q8, qwen4_w_q8)

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

struct ds4_metal_args_qwen4_conv_stream_rows {
    uint32_t n_rows;
    uint32_t n_channels;
    uint32_t conv_kernel;
    uint32_t apply_silu;
    uint32_t state_stride;   /* floats between two slots of the history pool */
    uint32_t x_stride;       /* floats between two rows of the batch arena */
    uint32_t pad0;
    uint32_t pad1;
};

/* One decode token for every row of a batch, each against its own slot of the
 * convolution history.  Identical arithmetic to kernel_qwen4_conv_stream with
 * n_tokens = 1, so the two agree bit for bit; the rows merely share a
 * dispatch instead of taking one each. */
kernel void kernel_qwen4_conv_stream_rows(
        constant ds4_metal_args_qwen4_conv_stream_rows & args,
        device float       *x,
        device float       *state,
        device const float *weight,
        device const uint  *slots,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint c = tgpig.x * ntg.x + tid;
    const uint row = tgpig.y;
    if (c >= args.n_channels || row >= args.n_rows) return;
    const uint C = args.n_channels;
    const uint K = args.conv_kernel;

    device float *st = state + (uint64_t)slots[row] * args.state_stride;
    device float *xr = x + (uint64_t)row * args.x_stride;

    float win[3];
    for (uint t = 0; t + 1 < K; t++) win[t] = st[t * C + c];
    float taps[4];
    for (uint t = 0; t < K; t++) taps[t] = weight[c * K + t];

    const float raw = xr[c];
    float acc = taps[K - 1] * raw;
    for (uint t = 0; t + 1 < K; t++) acc += taps[t] * win[t];
    for (uint t = 0; t + 2 < K; t++) win[t] = win[t + 1];
    win[K - 2] = raw;
    xr[c] = args.apply_silu ? qwen4_silu(acc) : acc;
    for (uint t = 0; t + 1 < K; t++) st[t * C + c] = win[t];
}

/* One entry per session of a batch for the *_rows2 kernels: its recurrent
 * state and convolution history by GPU address (the caches stay private,
 * so a session's snapshot can keep swapping places with its state), its
 * first batch row and how many consecutive rows it owns (its token, then
 * its draft).  A non-zero snapshot address receives the state after the
 * first token, as the verify kernels write it. */
struct ds4_metal_qwen4_gdn_row {
    uint64_t state;
    uint64_t hist;
    uint64_t snap_state;
    uint64_t snap_hist;
    uint32_t row0;
    uint32_t n_tok;
    uint32_t pad0;
    uint32_t pad1;
};

/* kernel_qwen4_conv_stream_rows over the table: one or two tokens in order
 * per entry against its own history, the window after the first token
 * written to the entry's snapshot when it has one. */
kernel void kernel_qwen4_conv_stream_rows2(
        constant ds4_metal_args_qwen4_conv_stream_rows & args,
        device float       *x,
        device const float *weight,
        device const ds4_metal_qwen4_gdn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint c = tgpig.x * ntg.x + tid;
    const uint r = tgpig.y;
    if (c >= args.n_channels || r >= args.n_rows) return;
    const ds4_metal_qwen4_gdn_row e = rows[r];
    const uint C = args.n_channels;
    const uint K = args.conv_kernel;
    device float *st = reinterpret_cast<device float *>(e.hist);
    float win[3];
    for (uint t = 0; t + 1 < K; t++) win[t] = st[t * C + c];
    float taps[4];
    for (uint t = 0; t < K; t++) taps[t] = weight[c * K + t];
    for (uint tok = 0; tok < e.n_tok; tok++) {
        device float *xr = x + (uint64_t)(e.row0 + tok) * args.x_stride;
        const float raw = xr[c];
        float acc = taps[K - 1] * raw;
        for (uint t = 0; t + 1 < K; t++) acc += taps[t] * win[t];
        for (uint t = 0; t + 2 < K; t++) win[t] = win[t + 1];
        win[K - 2] = raw;
        xr[c] = args.apply_silu ? qwen4_silu(acc) : acc;
        if (tok == 0 && e.snap_hist) {
            device float *sh = reinterpret_cast<device float *>(e.snap_hist);
            for (uint t = 0; t + 1 < K; t++) sh[t * C + c] = win[t];
        }
    }
    for (uint t = 0; t + 1 < K; t++) st[t * C + c] = win[t];
}

#define QWEN4_CONV_BLOCK 64u
/* Only incoming block windows need a snapshot; raw rows inside each block
 * stay private to its channel thread until they have entered the window. */
kernel void kernel_qwen4_conv_halo(
        constant ds4_metal_args_qwen4_conv_stream &args,
        device const float *x, device const float *state, device float *halo,
        uint gid [[thread_position_in_grid]]) {
    const uint C = args.n_channels, H = args.conv_kernel - 1u;
    const uint blocks = 1u + (args.n_tokens - 1u) / QWEN4_CONV_BLOCK;
    if (gid >= (uint64_t)blocks * H * C) return;
    const uint block = gid / (H * C), t = gid / C % H, c = gid % C;
    halo[gid] = block == 0 ? state[t * C + c] : x[(uint64_t)(block * QWEN4_CONV_BLOCK - H + t) * C + c];
}

kernel void kernel_qwen4_conv_blocked(
        constant ds4_metal_args_qwen4_conv_stream &args,
        device float *x, device float *state, device const float *weight,
        device const float *halo,
        uint gid [[thread_position_in_grid]]) {
    const uint C = args.n_channels, K = args.conv_kernel;
    const uint c = gid % C, block = gid / C, start = block * QWEN4_CONV_BLOCK;
    if (start >= args.n_tokens) return;
    float win[3], taps[4];
    for (uint t = 0; t + 1u < K; t++) win[t] = halo[((uint64_t)block * (K - 1u) + t) * C + c];
    for (uint t = 0; t < K; t++) taps[t] = weight[c * K + t];
    const uint end = start + min(QWEN4_CONV_BLOCK, args.n_tokens - start);
    for (uint tok = start; tok < end; tok++) {
        const float raw = x[(uint64_t)tok * C + c];
        float acc = taps[K - 1u] * raw;
        for (uint t = 0; t + 1u < K; t++) acc += taps[t] * win[t];
        for (uint t = 0; t + 2u < K; t++) win[t] = win[t + 1u];
        win[K - 2u] = raw;
        x[(uint64_t)tok * C + c] = args.apply_silu ? qwen4_silu(acc) : acc;
    }
    if (end == args.n_tokens) for (uint t = 0; t + 1u < K; t++) state[t * C + c] = win[t];
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
    uint32_t snap2_tok;    /* second snapshot point for 3-row MTP verifies (UINT32_MAX: never) */
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
        device float       *snap2_state,
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
    device float *snap2row = snap2_state + ((uint64_t)h * D + dv) * D + dk0;
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
        if (tok == args.snap2_tok) {
            for (uint i = 0; i < npt; i++) snap2row[i] = s[i];
        }
    }
    for (uint i = 0; i < npt; i++) srow[i] = s[i];
}

/* Scan for head_dim 128: one simdgroup per (v-head, 4 consecutive dv
 * rows), so k/q/g/beta are loaded once per four state rows and the eight
 * reductions per token overlap.  Same math and state layout as above. */
struct ds4_metal_args_qwen4_gdn_scan_rows {
    uint32_t n_rows;
    uint32_t n_k_head;
    uint32_t n_v_head;
    uint32_t head_dim;
    uint32_t state_stride;   /* floats between two slots of the pool */
    uint32_t qkv_stride;     /* floats between two rows of the batch arena */
    uint32_t out_stride;
    uint32_t pad0;
};

/* One decode step for every row of a batch, each against its own slot of the
 * recurrent state pool.  Same arithmetic as kernel_qwen4_gdn_scan_r4 with
 * n_tokens = 1, so the result is bit-identical to running that kernel once per
 * row; what changes is that the rows share one dispatch, which is long enough
 * to reach steady state where sixteen short ones were not.  head_dim is 128,
 * as in kernel_qwen4_gdn_scan_r4. */
kernel void kernel_qwen4_gdn_scan_rows(
        constant ds4_metal_args_qwen4_gdn_scan_rows & args,
        device const float *qkv,
        device const float *ga,
        device const float *gb,
        device float       *state,
        device float       *out,
        device const uint  *slots,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dv0 = (tgpig.x * (ntg.x / 32) + sgitg) * 4;
    const uint h = tgpig.y;
    const uint row = tgpig.z;
    if (dv0 >= args.head_dim || h >= args.n_v_head || row >= args.n_rows) return;
    const uint D = 128, dk0 = tiisg * 4;
    const uint kh = h % args.n_k_head;

    device float *srow = state + (uint64_t)slots[row] * args.state_stride +
                         ((uint64_t)h * D + dv0) * D + dk0;
    device const float *base = qkv + (uint64_t)row * args.qkv_stride;
    const float4 q = *(device const float4 *)(base + kh * D + dk0);
    const float4 k = *(device const float4 *)(base + (args.n_k_head + kh) * D + dk0);
    const float4 v = *(device const float4 *)(base + 2 * args.n_k_head * D + h * D + dv0);
    const float g = ga[(uint64_t)row * args.n_v_head + h];
    const float beta = gb[(uint64_t)row * args.n_v_head + h];

    float4 s[4];
    for (uint r = 0; r < 4; r++) s[r] = *(device const float4 *)(srow + r * D);
    float u[4], o[4];
    for (uint r = 0; r < 4; r++) { s[r] *= g; u[r] = dot(s[r], k); }
    for (uint r = 0; r < 4; r++) u[r] = simd_sum(u[r]);
    for (uint r = 0; r < 4; r++) { s[r] += k * ((v[r] - u[r]) * beta); o[r] = dot(s[r], q); }
    for (uint r = 0; r < 4; r++) o[r] = simd_sum(o[r]);
    if (tiisg == 0) {
        *(device float4 *)(out + (uint64_t)row * args.out_stride + (uint64_t)h * D + dv0) =
            float4(o[0], o[1], o[2], o[3]);
    }
    for (uint r = 0; r < 4; r++) *(device float4 *)(srow + r * D) = s[r];
}

/* kernel_qwen4_gdn_scan_rows over the table: the per-token arithmetic of
 * kernel_qwen4_gdn_scan_r4, one or two tokens in order per entry, the
 * state after the first written to the entry's snapshot when it has one. */
kernel void kernel_qwen4_gdn_scan_rows2(
        constant ds4_metal_args_qwen4_gdn_scan_rows & args,
        device const float *qkv,
        device const float *ga,
        device const float *gb,
        device float       *out,
        device const ds4_metal_qwen4_gdn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dv0 = (tgpig.x * (ntg.x / 32) + sgitg) * 4;
    const uint h = tgpig.y;
    const uint r = tgpig.z;
    if (dv0 >= args.head_dim || h >= args.n_v_head || r >= args.n_rows) return;
    const ds4_metal_qwen4_gdn_row e = rows[r];
    const uint D = 128, dk0 = tiisg * 4;
    const uint kh = h % args.n_k_head;
    const uint64_t at = ((uint64_t)h * D + dv0) * D + dk0;
    device float *srow = reinterpret_cast<device float *>(e.state) + at;
    float4 s[4];
    for (uint i = 0; i < 4; i++) s[i] = *(device const float4 *)(srow + i * D);
    for (uint tok = 0; tok < e.n_tok; tok++) {
        const uint row = e.row0 + tok;
        device const float *base = qkv + (uint64_t)row * args.qkv_stride;
        const float4 q = *(device const float4 *)(base + kh * D + dk0);
        const float4 k = *(device const float4 *)(base + (args.n_k_head + kh) * D + dk0);
        const float4 v = *(device const float4 *)(base + 2 * args.n_k_head * D + h * D + dv0);
        const float g = ga[(uint64_t)row * args.n_v_head + h];
        const float beta = gb[(uint64_t)row * args.n_v_head + h];
        float u[4], o[4];
        for (uint i = 0; i < 4; i++) { s[i] *= g; u[i] = dot(s[i], k); }
        for (uint i = 0; i < 4; i++) u[i] = simd_sum(u[i]);
        for (uint i = 0; i < 4; i++) { s[i] += k * ((v[i] - u[i]) * beta); o[i] = dot(s[i], q); }
        for (uint i = 0; i < 4; i++) o[i] = simd_sum(o[i]);
        if (tiisg == 0) {
            *(device float4 *)(out + (uint64_t)row * args.out_stride + (uint64_t)h * D + dv0) =
                float4(o[0], o[1], o[2], o[3]);
        }
        if (tok == 0 && e.snap_state) {
            device float *snaprow = reinterpret_cast<device float *>(e.snap_state) + at;
            for (uint i = 0; i < 4; i++) *(device float4 *)(snaprow + i * D) = s[i];
        }
    }
    for (uint i = 0; i < 4; i++) *(device float4 *)(srow + i * D) = s[i];
}

kernel void kernel_qwen4_gdn_scan_r4(
        constant ds4_metal_args_qwen4_gdn_scan & args,
        device const float *qkv,
        device const float *ga,
        device const float *gb,
        device float       *state,
        device float       *out,
        device float       *snap_state,
        device float       *snap2_state,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dv0 = (tgpig.x * (ntg.x / 32) + sgitg) * 4;
    const uint h = tgpig.y;
    if (dv0 >= args.head_dim || h >= args.n_v_head) return;
    const uint D = 128, dk0 = tiisg * 4;
    const uint kh = h % args.n_k_head;
    const uint conv_dim = 2 * args.n_k_head * D + args.n_v_head * D;

    float4 s[4];
    device float *srow = state + ((uint64_t)h * D + dv0) * D + dk0;
    device float *snaprow = snap_state + ((uint64_t)h * D + dv0) * D + dk0;
    device float *snap2row = snap2_state + ((uint64_t)h * D + dv0) * D + dk0;
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
        if (tok == args.snap2_tok) {
            for (uint r = 0; r < 4; r++) *(device float4 *)(snap2row + r * D) = s[r];
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
    uint32_t snap2_tok;    /* second snapshot point for 3-row MTP verifies */
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
        device float       *snap2_history,
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
        if (tok == args.snap2_tok) {
            for (uint t = 0; t < H; t++) snap2_history[t * C + c] = hist[t];
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

/* One row of a decode batch for the *_rows attention kernels.  The caches
 * stay per session, so a row reaches its own through GPU addresses instead
 * of bound buffers; its transients are row r of the batch arena. */
struct ds4_metal_qwen4_attn_row {
    uint64_t k_cache;      /* [cap][Hkv*D] half */
    uint64_t v_cache;      /* [cap][Hkv*D] half */
    uint64_t ik_cache;     /* [cap][Di] float */
    uint64_t block_key;    /* [n_block_cap][Di] half */
    uint64_t pos3;         /* [cap] uint4 rope positions */
    uint32_t pos;          /* the row's token position */
    uint32_t n_blocks;     /* complete indexer blocks at pos: (pos + 1) / ratio */
    uint32_t use_sel;      /* 1: block-sparse attention, 0: dense over 0..pos */
    uint32_t pad0;
};

/* Per (token, head) simdgroup: RMSNorm + NeoX rope of one query head (from
 * the [q|gate] interleaved projection, gate copied out raw), of one kv head
 * (written to the f16 cache at pos0+tok) or of one indexer query head; the
 * raw indexer key is copied to its cache.  Heads are enumerated
 * q(0..H-1), k(H..H+Hkv-1), iq(H+Hkv..+Hi-1), ik(last). */
static inline void qwen4_attn_prep_slot(
        constant ds4_metal_args_qwen4_attn_prep &args, uint slot, uint tok, uint pos, uint4 p3,
        device const float *qg, device const float *kproj, device const float *vproj,
        device const float *iq, device const float *ik,
        device const float *g_q, device const float *g_k, device const float *g_iq,
        device float *q_out, device float *gate_out, device half *k_cache, device half *v_cache,
        device float *iq_out, device float *ik_cache, threadgroup float *row, ushort tiisg) {
    const uint H = args.n_head, Hkv = args.n_head_kv, D = args.head_dim;
    const uint Hi = args.n_idx_head, Di = args.idx_dim;
    float v[8];   /* D/32 <= 8 */
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

/* Predictor prefix rows need only K/V and raw indexer K. Keep the full
 * preparation kernel's lane walk, normalization and serial RoPE unchanged. */
kernel void kernel_qwen4_attn_cache_prep(
        constant ds4_metal_args_qwen4_attn_prep & args,
        device const float *kproj,
        device const float *vproj,
        device const float *ik,
        device const float *g_k,
        device half *k_cache,
        device half *v_cache,
        device float *ik_cache,
        device const uint4 *pos3,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.x, tok = tgpig.y;
    if (tok >= args.n_tokens || slot > args.n_head_kv) return;
    const uint Hkv = args.n_head_kv, D = args.head_dim, Di = args.idx_dim;
    const uint pos = args.pos0 + tok;
    float v[8];
    threadgroup float row[576];
    float tmp[64];
    if (slot < Hkv) {
        const uint h = slot;
        const uint npt = D / 32;
        const uint4 p3 = pos3[pos];
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
    const uint npt = Di / 32;
    device const float *src = ik + (uint64_t)tok * Di;
    device float *dst = ik_cache + (uint64_t)pos * Di;
    for (uint i = 0; i < npt; i++) dst[tiisg * npt + i] = src[tiisg * npt + i];
}

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
    const uint pos = args.pos0 + tok;
    threadgroup float row[576];
    qwen4_attn_prep_slot(args, slot, tok, pos, pos3[pos], qg, kproj, vproj, iq, ik, g_q, g_k, g_iq,
                         q_out, gate_out, k_cache, v_cache, iq_out, ik_cache, row, tiisg);
}

/* The same per (row, head) work for a decode batch: row r's projections and
 * outputs are row r of the batch transients, its caches and position come
 * from its table entry.  n_tokens counts the rows. */
kernel void kernel_qwen4_attn_prep_rows(
        constant ds4_metal_args_qwen4_attn_prep & args,
        device const float *qg,        /* [R][H*2*D] */
        device const float *kproj,     /* [R][Hkv*D] */
        device const float *vproj,     /* [R][Hkv*D] */
        device const float *iq,        /* [R][Hi*Di] */
        device const float *ik,        /* [R][Di] */
        device const float *g_q,
        device const float *g_k,
        device const float *g_iq,
        device float       *q_out,     /* [R][H*D] */
        device float       *gate_out,  /* [R][H*D] */
        device float       *iq_out,    /* [R][Hi*Di] */
        device const ds4_metal_qwen4_attn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint r = tgpig.y;
    if (r >= args.n_tokens) return;
    const ds4_metal_qwen4_attn_row e = rows[r];
    threadgroup float row[576];
    qwen4_attn_prep_slot(args, tgpig.x, r, e.pos, reinterpret_cast<device const uint4 *>(e.pos3)[e.pos],
                         qg, kproj, vproj, iq, ik, g_q, g_k, g_iq, q_out, gate_out,
                         reinterpret_cast<device half *>(e.k_cache), reinterpret_cast<device half *>(e.v_cache),
                         iq_out, reinterpret_cast<device float *>(e.ik_cache), row, tiisg);
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
static inline void qwen4_idx_block_key_one(
        constant ds4_metal_args_qwen4_idx_block &args, uint b,
        device const float *ik_cache, device const float *g_ik, device const uint4 *pos3,
        device half *block_key, threadgroup float *row, ushort tiisg) {
    const uint Di = args.idx_dim;
    const uint npt = Di / 32;
    float v[4];
    float tmp[64];
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

kernel void kernel_qwen4_idx_block_key(
        constant ds4_metal_args_qwen4_idx_block & args,
        device const float *ik_cache,   /* [cap][Di] */
        device const float *g_ik,       /* [Di] */
        device const uint4 *pos3,       /* [cap] */
        device half        *block_key,  /* [n_blocks_cap][Di] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (tgpig.x >= args.n_blocks) return;
    threadgroup float row[128];
    qwen4_idx_block_key_one(args, args.block0 + tgpig.x, ik_cache, g_ik, pos3, block_key, row, tiisg);
}

/* Decode batch: a row whose position ends a block completes that block's
 * key in its own cache; the others have nothing to do.  n_blocks counts the
 * rows. */
kernel void kernel_qwen4_idx_block_key_rows(
        constant ds4_metal_args_qwen4_idx_block & args,
        device const float *g_ik,       /* [Di] */
        device const ds4_metal_qwen4_attn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (tgpig.x >= args.n_blocks) return;
    const ds4_metal_qwen4_attn_row e = rows[tgpig.x];
    if ((e.pos + 1u) % args.ratio != 0u) return;
    threadgroup float row[128];
    qwen4_idx_block_key_one(args, e.pos / args.ratio, reinterpret_cast<device const float *>(e.ik_cache), g_ik,
                            reinterpret_cast<device const uint4 *>(e.pos3),
                            reinterpret_cast<device half *>(e.block_key), row, tiisg);
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

/* Same scores with the token's query rows staged once per threadgroup and
 * the block key read as half4 vectors; every dot still adds its 128 terms
 * in index order, so the sums are bit-identical to kernel_qwen4_idx_score. */
static inline void qwen4_idx_score_vec_row(
        constant ds4_metal_args_qwen4_idx_score &args, uint b, uint n_blocks, uint visible,
        device const float *q, device const half *block_key, device float *score, device uint *tile_max,
        threadgroup float *qs, ushort tid, ushort3 ntg, ushort tiisg) {
    const uint Di = args.idx_dim;
    const uint nq = args.n_idx_head * Di;
    for (uint i = tid; i < nq; i += ntg.x) qs[i] = q[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool live = b < n_blocks && b < visible;
    float sum = -3.0e38f;
    if (live) {
        device const half4 *key = (device const half4 *)(block_key + (uint64_t)b * Di);
        sum = 0.0f;
        for (uint h = 0; h < args.n_idx_head; h++) {
            const threadgroup float *qh = qs + h * Di;
            float dot = 0.0f;
            for (uint d = 0; d < Di; d += 4) {
                const half4 k = key[d >> 2];
                dot += qh[d] * (float)k.x;
                dot += qh[d + 1] * (float)k.y;
                dot += qh[d + 2] * (float)k.z;
                dot += qh[d + 3] * (float)k.w;
            }
            sum += max(dot, 0.0f);
        }
    }
    if (b < n_blocks) score[b] = sum;
    /* the selector's key of this block (-inf and missing blocks map to 0),
     * reduced over the aligned eight-lane group; every lane joins the
     * shuffles */
    uint key = live ? as_type<uint>(max(sum, 0.0f)) : 0u;
    key = max(key, simd_shuffle_xor(key, 1));
    key = max(key, simd_shuffle_xor(key, 2));
    key = max(key, simd_shuffle_xor(key, 4));
    const uint n_tiles = (n_blocks + 7u) / 8u;
    if ((tiisg & 7u) == 0u && (b >> 3) < n_tiles) tile_max[b >> 3] = key;
}

kernel void kernel_qwen4_idx_score_vec(
        constant ds4_metal_args_qwen4_idx_score & args,
        device const float *iq,          /* [T][Hi*Di] */
        device const half  *block_key,   /* [n_blocks][Di] */
        device float       *score,       /* [T][n_blocks] */
        device uint        *tile_max,    /* [T][(n_blocks+7)/8] score keys */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float qs[4 * 128];
    const uint tok = tgpig.y;
    const uint nq = args.n_idx_head * args.idx_dim;
    if (tok >= args.n_tokens || nq > 4u * 128u || (args.idx_dim & 3u) != 0u) return;
    const uint n_tiles = (args.n_blocks + 7u) / 8u;
    qwen4_idx_score_vec_row(args, tgpig.x * ntg.x + tid, args.n_blocks, (args.pos0 + tok + 1) / args.ratio,
                            iq + (uint64_t)tok * nq, block_key, score + (uint64_t)tok * args.n_blocks,
                            tile_max + (uint64_t)tok * n_tiles, qs, tid, ntg, tiisg);
}

/* Decode batch: row r scores its own block keys against its query row.
 * n_blocks is the score row stride (the widest row); each row's own count
 * comes from its table entry.  n_tokens counts the rows. */
kernel void kernel_qwen4_idx_score_rows(
        constant ds4_metal_args_qwen4_idx_score & args,
        device const float *iq,          /* [R][Hi*Di] */
        device float       *score,       /* [R][n_blocks] */
        device uint        *tile_max,    /* [R][(n_blocks+7)/8] */
        device const ds4_metal_qwen4_attn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float qs[4 * 128];
    const uint r = tgpig.y;
    const uint nq = args.n_idx_head * args.idx_dim;
    if (r >= args.n_tokens || nq > 4u * 128u || (args.idx_dim & 3u) != 0u) return;
    const ds4_metal_qwen4_attn_row e = rows[r];
    if (!e.use_sel) return;
    const uint n_tiles = (args.n_blocks + 7u) / 8u;
    qwen4_idx_score_vec_row(args, tgpig.x * ntg.x + tid, args.n_blocks, e.n_blocks, iq + (uint64_t)r * nq,
                            reinterpret_cast<device const half *>(e.block_key), score + (uint64_t)r * args.n_blocks,
                            tile_max + (uint64_t)r * n_tiles, qs, tid, ntg, tiisg);
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

#define QWEN4_IDX_PRE_TILES 512   /* surviving tiles held in threadgroup memory */
#define QWEN4_IDX_PRE_MAX_TILES 8192

/* Key of entry e of the compact set: tile tiles[e>>3], block e&7. */
static inline uint qwen4_idx_pre_block(threadgroup const ushort *tiles, uint e) {
    return (uint)tiles[e >> 3] * 8u + (e & 7u);
}

/* Radix pass shared by the tile bound and the compact select: the k-th
 * largest key among n keys, same digit order and suffix-sum ranking as
 * kernel_qwen4_idx_select.  MODE 0 reads tile maxima, MODE 1 the compact
 * keys (entries past the last block are skipped), MODE 2 a full score row. */
template <int MODE>
static inline void qwen4_idx_radix(device const uint *tm, threadgroup const uint *keys,
                                   threadgroup const ushort *tiles, device const float *row,
                                   uint n_blocks, uint n, uint k, uint nth, ushort tid, ushort sgitg, ushort tiisg,
                                   threadgroup atomic_uint *hist, threadgroup uint *scan, threadgroup uint *found,
                                   thread uint &prefix_out, thread uint &need_out) {
    uint prefix = 0, need = k;
    for (uint pass = 0; pass < 4; pass++) {
        const uint shift = 24 - 8 * pass;
        const uint mask_hi = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint i = tid; i < 256; i += nth) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < n; e += nth) {
            uint key;
            if (MODE == 0) key = tm[e];
            else if (MODE == 1) { if (qwen4_idx_pre_block(tiles, e) >= n_blocks) continue; key = keys[e]; }
            else key = as_type<uint>(max(row[e], 0.0f));
            if ((key & mask_hi) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFFu], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint c = tid < 256 ? atomic_load_explicit(&hist[255 - tid], memory_order_relaxed) : 0u;
        const uint p = simd_prefix_inclusive_sum(c);
        if (tiisg == 31 && sgitg < 8) scan[sgitg] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256) {
            uint above = p - c;
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
    prefix_out = prefix;
    need_out = need;
}

/* Prefiltered top-k for one or two decode tokens (see the host comment):
 * tau = k-th largest tile maximum, survivors = tiles with maximum >= tau,
 * then the full-row radix select and gather over the survivors' blocks in
 * threadgroup memory.  Rows with more than QWEN4_IDX_PRE_TILES survivors or
 * more than QWEN4_IDX_PRE_MAX_TILES tiles run the full-row algorithm. */
static inline void qwen4_idx_select_pre_row(
        uint n, uint top_k, device const float *row, device const uint *tm, device int32_t *out,
        uint nth, ushort tid, ushort sgitg, ushort tiisg,
        threadgroup atomic_uint *hist, threadgroup uint *scan, threadgroup uint *found,
        threadgroup uint *keys, threadgroup ushort *tiles, threadgroup uint &total_surv) {
    const uint n_tiles = (n + 7u) / 8u;
    /* 1. tau: the k-th largest tile maximum (every tile survives when there
     *    are at most k tiles) */
    uint tau = 0, tau_need = 0;
    bool compact = n_tiles <= QWEN4_IDX_PRE_MAX_TILES;
    if (compact && n_tiles > top_k) {
        qwen4_idx_radix<0>(tm, keys, tiles, row, n, n_tiles, top_k, nth, tid, sgitg, tiisg,
                           hist, scan, found, tau, tau_need);
    }
    /* 2. survivors in ascending tile order: contiguous tile chunk per thread,
     *    ranks by exclusive scan */
    uint n_surv = 0;
    const uint tchunk = (n_tiles + nth - 1) / nth;
    const uint t0 = min((uint)tid * tchunk, n_tiles), t1 = min(t0 + tchunk, n_tiles);
    if (compact) {
        for (uint t = t0; t < t1; t++) n_surv += tm[t] >= tau;
    }
    uint rank = simd_prefix_exclusive_sum(n_surv);
    if (tiisg == 31) scan[sgitg] = rank + n_surv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const uint nsg = (nth + 31) / 32;
        const uint sv = tiisg < nsg ? scan[tiisg] : 0u;
        const uint sp = simd_prefix_exclusive_sum(sv);
        if (tiisg < nsg) scan[tiisg] = sp;
        if (tiisg == nsg - 1) total_surv = sp + sv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    rank += scan[sgitg];
    compact = compact && total_surv <= QWEN4_IDX_PRE_TILES;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (compact) {
        for (uint t = t0; t < t1; t++) {
            if (tm[t] < tau) continue;
            tiles[rank] = (ushort)t;
            for (uint j = 0; j < 8; j++) {
                const uint b = t * 8u + j;
                keys[rank * 8u + j] = b < n ? as_type<uint>(max(row[b], 0.0f)) : 0u;
            }
            rank++;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_c = compact ? total_surv * 8u : n;

    /* 3. k-th key and equal count over the compact set (or the full row) */
    uint prefix, need;
    if (compact) {
        qwen4_idx_radix<1>(tm, keys, tiles, row, n, n_c, top_k, nth, tid, sgitg, tiisg,
                           hist, scan, found, prefix, need);
    } else {
        qwen4_idx_radix<2>(tm, keys, tiles, row, n, n, top_k, nth, tid, sgitg, tiisg,
                           hist, scan, found, prefix, need);
    }
    /* Missing blocks of the last tile carry key 0 and must not count as
     * equals when the k-th key is 0; the block test excludes them below. */

    /* 4. gather: contiguous entry chunk per thread, ranks by exclusive scan */
    const uint chunk = (n_c + nth - 1) / nth;
    const uint e0 = min((uint)tid * chunk, n_c), e1 = min(e0 + chunk, n_c);
    uint n_gt = 0, n_eq = 0;
    for (uint e = e0; e < e1; e++) {
        const uint b = compact ? qwen4_idx_pre_block(tiles, e) : e;
        if (b >= n) continue;
        const uint key = compact ? keys[e] : as_type<uint>(max(row[e], 0.0f));
        n_gt += key > prefix; n_eq += key == prefix;
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
    const uint eq_base = top_k - need;
    for (uint e = e0; e < e1; e++) {
        const uint b = compact ? qwen4_idx_pre_block(tiles, e) : e;
        if (b >= n) continue;
        const uint key = compact ? keys[e] : as_type<uint>(max(row[e], 0.0f));
        if (key > prefix) out[r_gt++] = (int32_t)b;
        else if (key == prefix) { if (r_eq < need) out[eq_base + r_eq] = (int32_t)b; r_eq++; }
    }
}

kernel void kernel_qwen4_idx_select_pre(
        constant ds4_metal_args_qwen4_idx_select & args,
        device const float *score,     /* [T][n_blocks] */
        device const uint  *tile_max,  /* [T][n_tiles] */
        device int32_t     *sel,       /* [T][top_k] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens) return;
    const uint n = args.n_blocks;
    const uint n_tiles = (n + 7u) / 8u;
    threadgroup atomic_uint hist[256];
    threadgroup uint scan[32];
    threadgroup uint found[2];
    threadgroup uint keys[QWEN4_IDX_PRE_TILES * 8];
    threadgroup ushort tiles[QWEN4_IDX_PRE_TILES];
    threadgroup uint total_surv;
    qwen4_idx_select_pre_row(n, args.top_k, score + (uint64_t)tok * n, tile_max + (uint64_t)tok * n_tiles,
                             sel + (uint64_t)tok * args.top_k, ntg.x, tid, sgitg, tiisg,
                             hist, scan, found, keys, tiles, total_surv);
}

/* Decode batch: the prefiltered select of each sparse row over its own
 * block count; n_blocks is the score row stride.  n_tokens counts the rows. */
kernel void kernel_qwen4_idx_select_rows(
        constant ds4_metal_args_qwen4_idx_select & args,
        device const float *score,     /* [R][n_blocks] */
        device const uint  *tile_max,  /* [R][(n_blocks+7)/8] */
        device int32_t     *sel,       /* [R][top_k] */
        device const ds4_metal_qwen4_attn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint r = tgpig.x;
    if (r >= args.n_tokens) return;
    const ds4_metal_qwen4_attn_row e = rows[r];
    if (!e.use_sel) return;
    const uint stride = args.n_blocks, tiles_stride = (stride + 7u) / 8u;
    threadgroup atomic_uint hist[256];
    threadgroup uint scan[32];
    threadgroup uint found[2];
    threadgroup uint keys[QWEN4_IDX_PRE_TILES * 8];
    threadgroup ushort tiles[QWEN4_IDX_PRE_TILES];
    threadgroup uint total_surv;
    qwen4_idx_select_pre_row(e.n_blocks, args.top_k, score + (uint64_t)r * stride,
                             tile_max + (uint64_t)r * tiles_stride, sel + (uint64_t)r * args.top_k,
                             ntg.x, tid, sgitg, tiisg, hist, scan, found, keys, tiles, total_surv);
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
static inline void qwen4_idx_expand_row(
        uint pos, uint n_blk, uint ratio, device const int32_t *blk, device int32_t *dst,
        device uint32_t *n_sel, ushort tid, ushort3 ntg) {
    const uint tail_start = ((pos + 1) / ratio) * ratio;
    for (uint i = tid; i < n_blk * ratio; i += ntg.x) {
        dst[i] = blk[i / ratio] * (int32_t)ratio + (int32_t)(i % ratio);
    }
    for (uint t = tail_start + tid; t <= pos; t += ntg.x) {
        dst[n_blk * ratio + (t - tail_start)] = (int32_t)t;
    }
    if (tid == 0) *n_sel = n_blk * ratio + (pos + 1 - tail_start);
}

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
    qwen4_idx_expand_row(args.pos0 + tok, args.n_sel_blocks, args.ratio,
                         sel_blocks + (uint64_t)tok * args.n_sel_blocks,
                         sel_tokens + (uint64_t)tok * args.sel_stride, n_sel + tok, tid, ntg);
}

/* Decode batch: each sparse row's token list at its own position; n_tokens
 * counts the rows. */
kernel void kernel_qwen4_idx_expand_rows(
        constant ds4_metal_args_qwen4_idx_expand & args,
        device const int32_t *sel_blocks,   /* [R][n_sel_blocks] */
        device int32_t       *sel_tokens,   /* [R][sel_stride] */
        device uint32_t      *n_sel,        /* [R] */
        device const ds4_metal_qwen4_attn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint r = tgpig.x;
    if (r >= args.n_tokens) return;
    const ds4_metal_qwen4_attn_row e = rows[r];
    if (!e.use_sel) return;
    qwen4_idx_expand_row(e.pos, args.n_sel_blocks, args.ratio,
                         sel_blocks + (uint64_t)r * args.n_sel_blocks,
                         sel_tokens + (uint64_t)r * args.sel_stride, n_sel + r, tid, ntg);
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
static inline void qwen4_attn_decode_tile(
        constant ds4_metal_args_qwen4_attn_decode &args, uint split, uint kvh, uint tok,
        uint n, uint n_splits, uint keys_per_split, uint part_splits, bool use_sel,
        device const float *q, device const float *gate,
        device const half *k_cache, device const half *v_cache, device const int32_t *sel,
        device float *out, device float *part, ushort sgitg, ushort tiisg) {
    const uint H = args.n_head, Hkv = args.n_head_kv;
    constexpr uint D = NPT * 32;
    const uint group = H / Hkv;
    const uint hps = (group + QWEN4_ATTN_NSG - 1) / QWEN4_ATTN_NSG;
    const uint g0 = (uint)sgitg * hps;
    if (g0 >= group) return;
    const uint ng = min(hps, group - g0);
    const uint k0 = split * keys_per_split;
    const uint k1 = min(n, k0 + keys_per_split);

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
        const uint p = use_sel ? (uint)sel[idx] : idx;
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
        if (n_splits == 1) {
            device float *dst = out + ((uint64_t)tok * H + h) * D + tiisg * NPT;
            device const float *gt = gate + ((uint64_t)tok * H + h) * D + tiisg * NPT;
            const float inv = l[g] > 0.0f ? 1.0f / l[g] : 0.0f;
#pragma unroll
            for (uint i = 0; i < NPT; i++) dst[i] = acc[g][i] * inv * qwen4_sigmoid(gt[i]);
        } else {
            device float *dst = part + ((((uint64_t)tok * Hkv + kvh) * part_splits + split) * group + g0 + g) * (2u + D);
            if (tiisg == 0) { dst[0] = m[g]; dst[1] = l[g]; }
#pragma unroll
            for (uint i = 0; i < NPT; i++) dst[2u + tiisg * NPT + i] = acc[g][i];
        }
    }
}

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
    const uint n = args.use_sel ? n_sel[tok] : args.pos0 + tok + 1;
    qwen4_attn_decode_tile<NPT>(args, split, kvh, tok, n, args.n_splits, args.keys_per_split, args.n_splits,
                                args.use_sel != 0, q, gate, k_cache, v_cache,
                                sel_tokens + (uint64_t)tok * args.sel_stride, out, part, sgitg, tiisg);
}

/* The split count and width the host picks for one row from its key count
 * (keys_per_split is the keys per split before the cap on splits). */
static inline uint2 qwen4_attn_row_splits(uint n_keys, uint split_keys, uint max_splits) {
    uint n_splits = (n_keys + split_keys - 1u) / split_keys;
    n_splits = min(max(n_splits, 1u), max_splits);
    return uint2(n_splits, (n_keys + n_splits - 1u) / n_splits);
}

/* Decode batch: one (key split, kv head, row).  A row attends its own caches
 * with the splits the single-row dispatch would give its key count, so its
 * partials and merge are the same arithmetic; args.n_splits is the partials'
 * split stride and the cap, keys_per_split the keys per split before it.
 * n_tokens counts the rows. */
template <uint NPT>
kernel void kernel_qwen4_attn_decode_rows(
        constant ds4_metal_args_qwen4_attn_decode & args,
        device const float   *q,          /* [R][H*D] */
        device const float   *gate,       /* [R][H*D] */
        device const int32_t *sel_tokens, /* [R][sel_stride] */
        device const uint32_t *n_sel,     /* [R] */
        device float         *out,        /* [R][H*D] */
        device float         *part,       /* [R][Hkv][n_splits][group][2+D] */
        device const ds4_metal_qwen4_attn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint split = tgpig.x;
    const uint kvh = tgpig.y;
    const uint r = tgpig.z;
    if (kvh >= args.n_head_kv || r >= args.n_tokens) return;
    const ds4_metal_qwen4_attn_row e = rows[r];
    const uint n_keys = e.use_sel ? args.sel_stride : e.pos + 1u;
    const uint2 sp = qwen4_attn_row_splits(n_keys, args.keys_per_split, args.n_splits);
    if (split >= sp.x) return;
    const uint n = e.use_sel ? n_sel[r] : e.pos + 1u;
    qwen4_attn_decode_tile<NPT>(args, split, kvh, r, n, sp.x, sp.y, args.n_splits, e.use_sel != 0, q, gate,
                                reinterpret_cast<device const half *>(e.k_cache),
                                reinterpret_cast<device const half *>(e.v_cache),
                                sel_tokens + (uint64_t)r * args.sel_stride, out, part, sgitg, tiisg);
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

/* Same merge with one thread per dim: lane d runs the split chain the
 * 32-lane kernel ran for dim d (m and l are recomputed identically per
 * lane), so the outputs match bit for bit with eight times the threads. */
template <uint NPT>
static inline void qwen4_attn_merge_wide_head(
        constant ds4_metal_args_qwen4_attn_decode &args, uint h, uint tok, uint n_splits, uint part_splits,
        device const float *part, device const float *gate, device float *out, ushort tid) {
    constexpr uint D = NPT * 32;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    const uint group = H / Hkv;
    const uint kvh = h / group, g = h % group;
    const uint64_t stride = (uint64_t)group * (2u + D);
    device const float *base = part + (((uint64_t)tok * Hkv + kvh) * part_splits * group + g) * (2u + D);
    float mm = -3.0e38f;
    for (uint s = 0; s < n_splits; s++) mm = max(mm, base[s * stride]);
    float ll = 0.0f;
    float o = 0.0f;
    /* sixteen splits' values in flight per round; the chain below still
     * visits the splits in order */
    for (uint s0 = 0; s0 < n_splits; s0 += 16u) {
        float pm[16], pl[16], po[16];
        const uint n_round = min(16u, n_splits - s0);
        for (uint i = 0; i < 16u; i++) {
            device const float *p = base + (s0 + min(i, n_round - 1u)) * stride;
            pm[i] = p[0];
            pl[i] = p[1];
            po[i] = p[2u + tid];
        }
        for (uint i = 0; i < n_round; i++) {
            const float c = pl[i] > 0.0f ? exp(pm[i] - mm) : 0.0f;
            ll += pl[i] * c;
            o += po[i] * c;
        }
    }
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    const uint64_t at = ((uint64_t)tok * H + h) * D + tid;
    out[at] = o * inv * qwen4_sigmoid(gate[at]);
}

template <uint NPT>
kernel void kernel_qwen4_attn_merge_wide(
        constant ds4_metal_args_qwen4_attn_decode & args,
        device const float *part,
        device const float *gate,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint tok = tgpig.y;
    if (h >= args.n_head || tok >= args.n_tokens || tid >= NPT * 32) return;
    qwen4_attn_merge_wide_head<NPT>(args, h, tok, args.n_splits, args.n_splits, part, gate, out, tid);
}

/* Decode batch: the merge of each row's own split count (rows with one
 * split were written directly).  n_tokens counts the rows. */
template <uint NPT>
kernel void kernel_qwen4_attn_merge_rows(
        constant ds4_metal_args_qwen4_attn_decode & args,
        device const float *part,
        device const float *gate,
        device float       *out,
        device const ds4_metal_qwen4_attn_row *rows,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x;
    const uint r = tgpig.y;
    if (h >= args.n_head || r >= args.n_tokens || tid >= NPT * 32) return;
    const ds4_metal_qwen4_attn_row e = rows[r];
    const uint n_keys = e.use_sel ? args.sel_stride : e.pos + 1u;
    const uint2 sp = qwen4_attn_row_splits(n_keys, args.keys_per_split, args.n_splits);
    if (sp.x == 1u) return;
    qwen4_attn_merge_wide_head<NPT>(args, h, r, sp.x, args.n_splits, part, gate, out, tid);
}

#define QWEN4_ATTN_MERGE_WIDE_INSTANCE(NPT_) \
template [[host_name("kernel_qwen4_attn_merge_wide_npt" #NPT_)]] \
kernel void kernel_qwen4_attn_merge_wide<NPT_>(constant ds4_metal_args_qwen4_attn_decode &, device const float *, \
        device const float *, device float *, uint3, ushort);
QWEN4_ATTN_MERGE_WIDE_INSTANCE(8)
QWEN4_ATTN_MERGE_WIDE_INSTANCE(4)

#define QWEN4_ATTN_ROWS_INSTANCE(NPT_) \
template [[host_name("kernel_qwen4_attn_decode_rows_npt" #NPT_)]] \
kernel void kernel_qwen4_attn_decode_rows<NPT_>(constant ds4_metal_args_qwen4_attn_decode &, device const float *, \
        device const float *, device const int32_t *, device const uint32_t *, device float *, device float *, \
        device const ds4_metal_qwen4_attn_row *, uint3, ushort, ushort); \
template [[host_name("kernel_qwen4_attn_merge_rows_npt" #NPT_)]] \
kernel void kernel_qwen4_attn_merge_rows<NPT_>(constant ds4_metal_args_qwen4_attn_decode &, device const float *, \
        device const float *, device float *, device const ds4_metal_qwen4_attn_row *, uint3, ushort);
QWEN4_ATTN_ROWS_INSTANCE(8)
QWEN4_ATTN_ROWS_INSTANCE(4)

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

/* Only SSD cache launches bind address tables. The resident specialization
 * retains its original contiguous expert addressing and arithmetic. */
constant bool qwen4_expert_addresses [[function_constant(906)]];
static inline device const char *qwen4_expert_base(device const char *base,
                                                  uint expert, uint64_t bytes) {
    if (is_function_constant_defined(qwen4_expert_addresses) && qwen4_expert_addresses) {
        return (device const char *)((device const uint64_t *)base)[expert];
    }
    return base + (uint64_t)expert * bytes;
}

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
    uint32_t n_total_expert;
    uint32_t slot_mask[3]; /* per-token routed/shared slots for SSD split passes */
    uint32_t masked_tokens; /* zero disables masking; an empty enabled row skips all slots */
    uint32_t list_cap;     /* grouped kernels: row stride of the per-expert pair lists */
    uint32_t pad0;
};
static_assert(sizeof(ds4_metal_args_qwen4_moe) == 72u, "Qwen MoE host/Metal argument layout");

/* SSD split launches compact the grid, but all addressing keeps the original
 * routed/shared slot. Resident specializations eliminate this remapping. */
static inline uint qwen4_moe_slot(constant ds4_metal_args_qwen4_moe &args,
                                  uint grid_slot, uint token) {
    if (is_function_constant_defined(qwen4_expert_addresses) && qwen4_expert_addresses) {
        if (args.masked_tokens) {
            if (token >= args.masked_tokens || token >= 3u) return 0xffffffffu;
            uint mask = args.slot_mask[token];
            for (uint i = 0; i < grid_slot && mask; i++) mask &= mask - 1u;
            return mask ? ctz(mask) : 0xffffffffu;
        }
    }
    return grid_slot;
}

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
        const uint nb = (in_dim + 255u) / 256u;
        const uint group = tiisg / 2, l = (tiisg % 2) * 8;
        const uint q_base = 32u * (group / 8u) + 16u * (group & 1u), shift = ((group / 2u) & 3u) * 2u;
        for (uint ib = 0; ib < nb; ib++) {
            /* Padded Q2_K down weights have no corresponding activation tail. */
            if (ib * 256u + group * 16u + l >= in_dim) continue;
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

/* Two Q8 projections share only activation loads. Keep each qwen4_row_dot
 * lane's block order, four-term expression and final SIMD reduction. */
static inline float2 qwen4_q8_pair_dot(device const char *gr, device const char *ur,
                                      device const float *x, uint in_dim, ushort tiisg) {
    const short ix = tiisg / 8, it = tiisg % 8;
    const uint nb = in_dim / 32u;
    float sumg = 0.0f, sumu = 0.0f;
    for (uint ib = (uint)ix; ib < nb; ib += 4) {
        device const char *bg = gr + (uint64_t)ib * 34;
        device const char *bu = ur + (uint64_t)ib * 34;
        device const float *y = x + ib * 32 + (uint)it * 2;
        const float dg = (float)(*(device const half *)bg);
        const float du = (float)(*(device const half *)bu);
        device const char *qg = bg + 2 + it * 2;
        device const char *qu = bu + 2 + it * 2;
        const float y0 = y[0], y1 = y[1], y16 = y[16], y17 = y[17];
        sumg += dg * (y0 * (float)qg[0] + y1 * (float)qg[1] +
                      y16 * (float)qg[16] + y17 * (float)qg[17]);
        sumu += du * (y0 * (float)qu[0] + y1 * (float)qu[1] +
                      y16 * (float)qu[16] + y17 * (float)qu[17]);
    }
    return float2(simd_sum(sumg), simd_sum(sumu));
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

/* Decode specialization keeps the original per-lane accumulation order. */
constant uint qwen4_mv_type [[function_constant(901)]];
constant uint qwen4_mv_shared_type [[function_constant(902)]];
constant uint qwen4_mv_dim [[function_constant(903)]];
constant uint qwen4_mv_rows [[function_constant(904)]];

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
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = qwen4_moe_slot(args, tgpig.y, tgpig.z);
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = is_function_constant_defined(qwen4_mv_rows) ? qwen4_mv_rows : 2u;
    const uint dim = is_function_constant_defined(qwen4_mv_dim) ? qwen4_mv_dim : args.in_dim;
    const uint wt = is_function_constant_defined(qwen4_mv_type) ? qwen4_mv_type : args.weight_type;
    const uint st = is_function_constant_defined(qwen4_mv_shared_type) ? qwen4_mv_shared_type : args.shared_type;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? st : wt;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    const uint expert = shared ? 0u : (uint)selected[(uint64_t)tok * args.n_slots + slot];
    device const char *gb = shared ? sh_gate : qwen4_expert_base(gate_base, expert, args.expert_bytes);
    device const char *ub = shared ? sh_up : qwen4_expert_base(up_base, expert, args.expert_bytes);
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
        const uint64_t off = (uint64_t)r * row_bytes;
        const float g = qwen4_row_dot(gb + off, xt, type, dim, tiisg);
        const float u = qwen4_row_dot(ub + off, xt, type, dim, tiisg);
        if (tiisg == 0) {
            mid[((uint64_t)tok * n_out + slot) * args.out_rows + r] = qwen4_silu(g) * u;
        }
    }
}

/* IQ2XXS gate/up input reuse, one or two rows per SIMD group. Each lane
 * keeps the generic dot's eight-value part and ordered dl * part additions;
 * only independent rows and projections share the activation loads. */
template <uint NR>
kernel void kernel_qwen4_moe_mid_iq2(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *gate_base,
        device const char *up_base,
        device const int32_t *selected,
        device const float *x,
        device float *mid,
        device const char *sh_gate,
        device const char *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = qwen4_moe_slot(args, tgpig.y, tgpig.z), tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    const uint64_t mid_base = ((uint64_t)tok * n_out + slot) * args.out_rows;
    if (slot == args.n_slots) {
        if (args.shared_type == 8u) {
            for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
                const uint64_t off = (uint64_t)r * args.shared_row_bytes;
                const float2 gu = qwen4_q8_pair_dot(sh_gate + off, sh_up + off, xt, args.in_dim, tiisg);
                if (tiisg == 0) mid[mid_base + r] = qwen4_silu(gu.x) * gu.y;
            }
        } else {
            for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
                const uint64_t off = (uint64_t)r * args.shared_row_bytes;
                const float g = qwen4_row_dot(sh_gate + off, xt, args.shared_type, args.in_dim, tiisg);
                const float u = qwen4_row_dot(sh_up + off, xt, args.shared_type, args.in_dim, tiisg);
                if (tiisg == 0) mid[mid_base + r] = qwen4_silu(g) * u;
            }
        }
        return;
    }

    const int32_t expert = selected[(uint64_t)tok * args.n_slots + slot];
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0) {
            for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) mid[mid_base + r] = 0.0f;
        }
        return;
    }
    device const char *gb = qwen4_expert_base(gate_base, (uint)expert, args.expert_bytes);
    device const char *ub = qwen4_expert_base(up_base, (uint)expert, args.expert_bytes);
    const uint nb = args.in_dim / 256;
    const uint ib32 = tiisg / 4, j = tiisg % 4;
    float sumg[NR] = {0.0f}, sumu[NR] = {0.0f};
    for (uint ib = 0; ib < nb; ib++) {
        device const float *yp = xt + ib * 256 + ib32 * 32 + j * 8;
        float y[8];
        for (uint i = 0; i < 8; i++) y[i] = yp[i];
        for (uint r = 0; r < NR && row0 + r < args.out_rows; r++) {
            const uint64_t off = (uint64_t)(row0 + r) * args.row_bytes + (uint64_t)ib * 66;
            device const uchar *bg = (device const uchar *)(gb + off);
            device const uchar *bu = (device const uchar *)(ub + off);
            const float dg = (float)(*(device const half *)bg);
            const float du = (float)(*(device const half *)bu);
            /* IQ2XXS blocks are only two-byte aligned. */
            device const ushort *qg = (device const ushort *)(bg + 2) + 4 * ib32;
            device const ushort *qu = (device const ushort *)(bu + 2) + 4 * ib32;
            const uint ag = (uint)qg[0] | ((uint)qg[1] << 16);
            const uint sg = (uint)qg[2] | ((uint)qg[3] << 16);
            const uint au = (uint)qu[0] | ((uint)qu[1] << 16);
            const uint su = (uint)qu[2] | ((uint)qu[3] << 16);
            const float dlg = dg * (0.5f + (float)(sg >> 28)) * 0.25f;
            const float dlu = du * (0.5f + (float)(su >> 28)) * 0.25f;
            constant const uchar *gridg = (constant const uchar *)(ds4_metal_iq2xxs_grid + ((ag >> (8 * j)) & 0xFFu));
            constant const uchar *gridu = (constant const uchar *)(ds4_metal_iq2xxs_grid + ((au >> (8 * j)) & 0xFFu));
            const uint signsg = ds4_metal_ksigns_iq2xs[(sg >> (7 * j)) & 127u];
            const uint signsu = ds4_metal_ksigns_iq2xs[(su >> (7 * j)) & 127u];
            float partg = 0.0f, partu = 0.0f;
            for (uint i = 0; i < 8; i++) {
                partg += (float)gridg[i] * ((signsg >> i) & 1u ? -y[i] : y[i]);
                partu += (float)gridu[i] * ((signsu >> i) & 1u ? -y[i] : y[i]);
            }
            sumg[r] += dlg * partg;
            sumu[r] += dlu * partu;
        }
    }
    for (uint r = 0; r < NR && row0 + r < args.out_rows; r++) {
        const float g = simd_sum(sumg[r]);
        const float u = simd_sum(sumu[r]);
        if (tiisg == 0) mid[mid_base + row0 + r] = qwen4_silu(g) * u;
    }
}

#define QWEN4_MOE_MID_IQ2_INSTANCE(NR_, NAME_) \
template [[host_name(NAME_)]] \
kernel void kernel_qwen4_moe_mid_iq2<NR_>(constant ds4_metal_args_qwen4_moe &, \
        device const char *, device const char *, device const int32_t *, device const float *, \
        device float *, device const char *, device const char *, uint3, ushort, ushort, ushort3);
QWEN4_MOE_MID_IQ2_INSTANCE(2, "kernel_qwen4_moe_mid_iq2")
QWEN4_MOE_MID_IQ2_INSTANCE(1, "kernel_qwen4_moe_mid_iq2_nr1")
#undef QWEN4_MOE_MID_IQ2_INSTANCE


/* Q4_K gate/up input reuse with the original qwen4_row_dot lane mapping
 * and accumulation order.  Each lane still visits every block in order and
 * adds its eight elements individually; only independent rows/projections
 * are interleaved.  NR output rows per SIMD group. */
template <uint NR>
kernel void kernel_qwen4_moe_mid_q4k(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *gate_base,
        device const char *up_base,
        device const int32_t *selected,
        device const float *x,
        device float *mid,
        device const char *sh_gate,
        device const char *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = qwen4_moe_slot(args, tgpig.y, tgpig.z), tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    const uint64_t mid_base = ((uint64_t)tok * n_out + slot) * args.out_rows;
    if (slot == args.n_slots) {
        for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
            const uint64_t off = (uint64_t)r * args.shared_row_bytes;
            const float2 gu = args.shared_type == 8u ?
                qwen4_q8_pair_dot(sh_gate + off, sh_up + off, xt, args.in_dim, tiisg) :
                float2(qwen4_row_dot(sh_gate + off, xt, args.shared_type, args.in_dim, tiisg),
                       qwen4_row_dot(sh_up + off, xt, args.shared_type, args.in_dim, tiisg));
            if (tiisg == 0) mid[mid_base + r] = qwen4_silu(gu.x) * gu.y;
        }
        return;
    }

    const int32_t expert = selected[(uint64_t)tok * args.n_slots + slot];
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0) {
            for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) mid[mid_base + r] = 0.0f;
        }
        return;
    }
    device const char *gb = qwen4_expert_base(gate_base, (uint)expert, args.expert_bytes);
    device const char *ub = qwen4_expert_base(up_base, (uint)expert, args.expert_bytes);
    const uint nb = args.in_dim / 256;
    const uint group = tiisg / 4, l = (tiisg % 4) * 8;
    const uint shift = (group & 1u) * 4u;
    float sumg[NR] = {0.0f}, sumu[NR] = {0.0f};
    for (uint ib = 0; ib < nb; ib++) {
        device const float *yp = xt + ib * 256 + group * 32 + l;
        /* Widen memory operations while preserving the scalar FP32 chain.
         * Packed types keep the original float/half alignment contract. */
        const packed_float4 y0 = *(device const packed_float4 *)yp;
        const packed_float4 y1 = *(device const packed_float4 *)(yp + 4u);
        const float y[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
        for (uint r = 0; r < NR && row0 + r < args.out_rows; r++) {
            const uint64_t off = (uint64_t)(row0 + r) * args.row_bytes + (uint64_t)ib * 144;
            device const uchar *bg = (device const uchar *)(gb + off);
            device const uchar *bu = (device const uchar *)(ub + off);
            const packed_half2 scales_g = *(device const packed_half2 *)bg;
            const packed_half2 scales_u = *(device const packed_half2 *)bu;
            const float dg = (float)scales_g.x, dmg = (float)scales_g.y;
            const float du = (float)scales_u.x, dmu = (float)scales_u.y;
            device const uchar *scg = bg + 4;
            device const uchar *scu = bu + 4;
            uint sg, mg, su, mu;
            if (group < 4) {
                sg = scg[group] & 63u; mg = scg[group + 4] & 63u;
                su = scu[group] & 63u; mu = scu[group + 4] & 63u;
            } else {
                sg = (scg[group + 4] & 0xFu) | ((scg[group - 4] & 0xC0u) >> 2);
                mg = (scg[group + 4] >> 4) | ((scg[group] & 0xC0u) >> 2);
                su = (scu[group + 4] & 0xFu) | ((scu[group - 4] & 0xC0u) >> 2);
                mu = (scu[group + 4] >> 4) | ((scu[group] & 0xC0u) >> 2);
            }
            const float dsg = dg * (float)sg, dmin_g = dmg * (float)mg;
            const float dsu = du * (float)su, dmin_u = dmu * (float)mu;
            /* Packed code loads change fast-math contraction on M1 Max.
             * Keep scalar bytes to match the original dot bit for bit. */
            device const uchar *qg = bg + 16 + (group >> 1) * 32 + l;
            device const uchar *qu = bu + 16 + (group >> 1) * 32 + l;
            for (uint i = 0; i < 8; i++) {
                sumg[r] += (dsg * (float)((qg[i] >> shift) & 0xFu) - dmin_g) * y[i];
                sumu[r] += (dsu * (float)((qu[i] >> shift) & 0xFu) - dmin_u) * y[i];
            }
        }
    }
    for (uint r = 0; r < NR && row0 + r < args.out_rows; r++) {
        const float g = simd_sum(sumg[r]);
        const float u = simd_sum(sumu[r]);
        if (tiisg == 0) mid[mid_base + row0 + r] = qwen4_silu(g) * u;
    }
}

#define QWEN4_MOE_MID_Q4K_INSTANCE(NR_, NAME_) \
template [[host_name(NAME_)]] \
kernel void kernel_qwen4_moe_mid_q4k<NR_>(constant ds4_metal_args_qwen4_moe &, \
        device const char *, device const char *, device const int32_t *, device const float *, \
        device float *, device const char *, device const char *, uint3, ushort3, ushort, ushort);
QWEN4_MOE_MID_Q4K_INSTANCE(2, "kernel_qwen4_moe_mid_q4k")
QWEN4_MOE_MID_Q4K_INSTANCE(1, "kernel_qwen4_moe_mid_q4k_nr1")

/* A decode batch over its distinct experts: the threadgroup of the first
 * (token, slot) pair in an expert's list computes its rows for every pair in
 * the list, four pairs per pass, so an expert's rows are read once per four
 * tokens that chose it instead of once per token.  Every token keeps its own
 * accumulators and block walk, so its outputs are kernel_qwen4_moe_mid_q4k's
 * bit for bit; the other pairs of the list return at once.  No shared slot:
 * the batch runs the shared expert as dense projections. */
/* One pass of the grouped mid kernel over NJ pairs of one expert: the
 * block's scales and the lane's eight nibbles come in as words (the byte
 * values are the same), the sub-scale decode and the dequantized weights
 * are computed once per block and row, and every pair accumulates in its
 * own named registers in the per-token kernel's order. */
template <uint NR, uint NJ>
static inline void qwen4_moe_mid_q4k_pass(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *gate_base, device const char *up_base,
        device const float *x, device float *mid, device const int32_t *list,
        uint64_t ebase, uint row0, uint nb, uint group, uint l, uint shift, ushort tiisg) {
    device const float *xs[NJ];
    uint pairs[NJ];
#pragma unroll
    for (uint j = 0; j < NJ; j++) {
        pairs[j] = (uint)list[j];
        xs[j] = x + (uint64_t)(pairs[j] / args.n_slots) * args.in_dim;
    }
    float sg_[NJ][NR], su_[NJ][NR];
#pragma unroll
    for (uint j = 0; j < NJ; j++) {
#pragma unroll
        for (uint r = 0; r < NR; r++) { sg_[j][r] = 0.0f; su_[j][r] = 0.0f; }
    }
    for (uint ib = 0; ib < nb; ib++) {
        const uint yo = ib * 256 + group * 32 + l;
        float y[NJ][8];
#pragma unroll
        for (uint j = 0; j < NJ; j++) {
            const float4 ya = *(device const float4 *)(xs[j] + yo);
            const float4 yb = *(device const float4 *)(xs[j] + yo + 4);
            y[j][0] = ya.x; y[j][1] = ya.y; y[j][2] = ya.z; y[j][3] = ya.w;
            y[j][4] = yb.x; y[j][5] = yb.y; y[j][6] = yb.z; y[j][7] = yb.w;
        }
#pragma unroll
        for (uint r = 0; r < NR; r++) {
            if (row0 + r >= args.out_rows) break;
            const uint64_t off = ebase + (uint64_t)(row0 + r) * args.row_bytes + (uint64_t)ib * 144;
            device const uchar *bg = (device const uchar *)(gate_base + off);
            device const uchar *bu = (device const uchar *)(up_base + off);
            const uint dg2 = *(device const uint *)bg, du2 = *(device const uint *)bu;
            const uint3 scg = *(device const uint3 *)(bg + 4), scu = *(device const uint3 *)(bu + 4);
            const uint2 qg2 = *(device const uint2 *)(bg + 16 + (group >> 1) * 32 + l);
            const uint2 qu2 = *(device const uint2 *)(bu + 16 + (group >> 1) * 32 + l);
            const float dg = (float)as_type<half>((ushort)(dg2 & 0xFFFFu));
            const float dmg = (float)as_type<half>((ushort)(dg2 >> 16));
            const float du = (float)as_type<half>((ushort)(du2 & 0xFFFFu));
            const float dmu = (float)as_type<half>((ushort)(du2 >> 16));
#define QWEN4_Q4K_SCB(W_, I_) ((((I_) < 4u ? (W_).x : (I_) < 8u ? (W_).y : (W_).z) >> (8u * ((I_) & 3u))) & 0xFFu)
            uint sg, mg, su, mu;
            if (group < 4) {
                sg = QWEN4_Q4K_SCB(scg, group) & 63u; mg = QWEN4_Q4K_SCB(scg, group + 4) & 63u;
                su = QWEN4_Q4K_SCB(scu, group) & 63u; mu = QWEN4_Q4K_SCB(scu, group + 4) & 63u;
            } else {
                sg = (QWEN4_Q4K_SCB(scg, group + 4) & 0xFu) | ((QWEN4_Q4K_SCB(scg, group - 4) & 0xC0u) >> 2);
                mg = (QWEN4_Q4K_SCB(scg, group + 4) >> 4) | ((QWEN4_Q4K_SCB(scg, group) & 0xC0u) >> 2);
                su = (QWEN4_Q4K_SCB(scu, group + 4) & 0xFu) | ((QWEN4_Q4K_SCB(scu, group - 4) & 0xC0u) >> 2);
                mu = (QWEN4_Q4K_SCB(scu, group + 4) >> 4) | ((QWEN4_Q4K_SCB(scu, group) & 0xC0u) >> 2);
            }
#undef QWEN4_Q4K_SCB
            const float dsg = dg * (float)sg, dmin_g = dmg * (float)mg;
            const float dsu = du * (float)su, dmin_u = dmu * (float)mu;
#pragma unroll
            for (uint i = 0; i < 8; i++) {
                const uint qgi = ((i < 4u ? qg2.x : qg2.y) >> (8u * (i & 3u))) & 0xFFu;
                const uint qui = ((i < 4u ? qu2.x : qu2.y) >> (8u * (i & 3u))) & 0xFFu;
                const float wg = dsg * (float)((qgi >> shift) & 0xFu) - dmin_g;
                const float wu = dsu * (float)((qui >> shift) & 0xFu) - dmin_u;
#pragma unroll
                for (uint j = 0; j < NJ; j++) {
                    sg_[j][r] += wg * y[j][i];
                    su_[j][r] += wu * y[j][i];
                }
            }
        }
    }
#pragma unroll
    for (uint r = 0; r < NR; r++) {
        if (row0 + r >= args.out_rows) break;
#pragma unroll
        for (uint j = 0; j < NJ; j++) {
            const float g = simd_sum(sg_[j][r]);
            const float u = simd_sum(su_[j][r]);
            if (tiisg == 0) mid[(uint64_t)pairs[j] * args.out_rows + row0 + r] = qwen4_silu(g) * u;
        }
    }
}

template <uint NR>
kernel void kernel_qwen4_moe_mid_q4k_grouped(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *gate_base,
        device const char *up_base,
        device const int32_t *selected,
        device const float *x,
        device float *mid,
        device const int32_t *lists,     /* [n_total_expert][list_cap]: t*n_slots + slot */
        device const int32_t *counts,    /* [n_total_expert] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint row0 = (tgpig.x * (ntg.x / 32) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= args.n_slots || tok >= args.n_tokens) return;
    const uint pair = tok * args.n_slots + slot;
    const int32_t expert = selected[pair];
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0) {
            for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) mid[(uint64_t)pair * args.out_rows + r] = 0.0f;
        }
        return;
    }
    device const int32_t *list = lists + (uint64_t)(uint)expert * args.list_cap;
    const uint count = min((uint)counts[expert], args.list_cap);
    if (count == 0 || (uint)list[0] != pair) return;
    const uint64_t ebase = (uint64_t)(uint)expert * args.expert_bytes;
    const uint nb = args.in_dim / 256;
    const uint group = tiisg / 4, l = (tiisg % 4) * 8;
    const uint shift = (group & 1u) * 4u;
    /* Up to four pairs per pass, the count a template constant so the inner
     * loops carry no per-pair branch; each pair keeps named accumulators (an
     * array indexed by the pair would leave the registers). */
    for (uint j0 = 0; j0 < count; j0 += 4u) {
        const uint nj = min(4u, count - j0);
        switch (nj) {
        case 1u: qwen4_moe_mid_q4k_pass<NR, 1>(args, gate_base, up_base, x, mid, list + j0, ebase, row0, nb, group, l, shift, tiisg); break;
        case 2u: qwen4_moe_mid_q4k_pass<NR, 2>(args, gate_base, up_base, x, mid, list + j0, ebase, row0, nb, group, l, shift, tiisg); break;
        case 3u: qwen4_moe_mid_q4k_pass<NR, 3>(args, gate_base, up_base, x, mid, list + j0, ebase, row0, nb, group, l, shift, tiisg); break;
        default: qwen4_moe_mid_q4k_pass<NR, 4>(args, gate_base, up_base, x, mid, list + j0, ebase, row0, nb, group, l, shift, tiisg); break;
        }
    }
}

template [[host_name("kernel_qwen4_moe_mid_q4k_grouped")]]
kernel void kernel_qwen4_moe_mid_q4k_grouped<2>(constant ds4_metal_args_qwen4_moe &,
        device const char *, device const char *, device const int32_t *, device const float *,
        device float *, device const int32_t *, device const int32_t *, uint3, ushort3, ushort, ushort);

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
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = qwen4_moe_slot(args, tgpig.y, tgpig.z);
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = is_function_constant_defined(qwen4_mv_rows) ? qwen4_mv_rows : 2u;
    const uint dim = is_function_constant_defined(qwen4_mv_dim) ? qwen4_mv_dim : args.in_dim;
    const uint wt = is_function_constant_defined(qwen4_mv_type) ? qwen4_mv_type : args.weight_type;
    const uint st = is_function_constant_defined(qwen4_mv_shared_type) ? qwen4_mv_shared_type : args.shared_type;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? st : wt;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    const uint expert = shared ? 0u : (uint)selected[(uint64_t)tok * args.n_slots + slot];
    device const char *db = shared ? sh_down : qwen4_expert_base(down_base, expert, args.expert_bytes);
    device const float *m = mid + pair * args.in_dim;
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
        const float v = qwen4_row_dot(db + (uint64_t)r * row_bytes, m, type, dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
    }
}

/* Q2_K NR2 down rows reuse each activation load across two independent
 * accumulators. Keep the generic lane mapping, block/element order and SIMD
 * reduction, including the unread 128-element weight padding at width 640. */
kernel void kernel_qwen4_moe_down_q2k(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *down_base,
        device const int32_t *selected,
        device const float *mid,
        device float *part,
        device const char *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = qwen4_moe_slot(args, tgpig.y, tgpig.z), tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * 2u;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    device const float *m = mid + pair * args.in_dim;
    if (slot == args.n_slots) {
        if (args.shared_type == 8u) {
            // Keep each Q8 row's lane mapping, block order, four-term
            // expression and SIMD reduction; reuse m loads across rows.
            const short ix = tiisg / 8, it = tiisg % 8;
            const uint nb = args.in_dim / 32u;
            float acc[2] = {0.0f, 0.0f};
            for (uint ib = (uint)ix; ib < nb; ib += 4) {
                device const float *y = m + ib * 32 + (uint)it * 2;
                const float y0 = y[0], y1 = y[1], y16 = y[16], y17 = y[17];
                for (uint r = 0; r < 2u && row0 + r < args.out_rows; r++) {
                    device const char *b = sh_down +
                        (uint64_t)(row0 + r) * args.shared_row_bytes + (uint64_t)ib * 34;
                    const float d = (float)(*(device const half *)b);
                    device const char *q = b + 2 + it * 2;
                    acc[r] += d * (y0 * (float)q[0] + y1 * (float)q[1] +
                                    y16 * (float)q[16] + y17 * (float)q[17]);
                }
            }
            for (uint r = 0; r < 2u && row0 + r < args.out_rows; r++) {
                const float v = simd_sum(acc[r]);
                if (tiisg == 0) part[pair * args.out_rows + row0 + r] = v;
            }
        } else {
            for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++) {
                const float v = qwen4_row_dot(sh_down + (uint64_t)r * args.shared_row_bytes,
                                              m, args.shared_type, args.in_dim, tiisg);
                if (tiisg == 0) part[pair * args.out_rows + r] = v;
            }
        }
        return;
    }
    device const char *db = qwen4_expert_base(down_base,
        (uint)selected[(uint64_t)tok * args.n_slots + slot], args.expert_bytes);
    const uint group = tiisg / 2u, l = (tiisg % 2u) * 8u;
    const uint q_base = 32u * (group / 8u) + 16u * (group & 1u);
    const uint shift = ((group / 2u) & 3u) * 2u;
    float acc[2] = {0.0f, 0.0f};
    /* Complete blocks keep all lanes active; only the last block has
     * 128 live values. No padded activation addresses are formed. */
    for (uint ib = 0; ib < 2u; ib++) {
        const uint column = ib * 256u + group * 16u + l;
        /* Widen loads only; preserve the eight-element FP32 chain below.
         * Packed types retain scalar activation and half-scale alignment. */
        const packed_float4 y0 = *(device const packed_float4 *)(m + column);
        const packed_float4 y1 = *(device const packed_float4 *)(m + column + 4u);
        const float y[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
        for (uint r = 0; r < 2u && row0 + r < args.out_rows; r++) {
            device const uchar *blk = (device const uchar *)(db +
                (uint64_t)(row0 + r) * args.row_bytes + (uint64_t)ib * 84u);
            const packed_half2 scales = *(device const packed_half2 *)(blk + 80);
            const float d = (float)scales.x;
            const float dmin = (float)scales.y;
            const uint sc = blk[group];
            const float ds = d * (float)(sc & 0xFu), dm = dmin * (float)(sc >> 4);
            device const uchar *qs = blk + 16u + q_base + l;
            const packed_uchar4 q0 = *(device const packed_uchar4 *)qs;
            const packed_uchar4 q1 = *(device const packed_uchar4 *)(qs + 4u);
            const uchar q[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            for (uint i = 0; i < 8u; i++) acc[r] += (ds * (float)((q[i] >> shift) & 3u) - dm) * y[i];
        }
    }
    if (tiisg < 16u) {
        const uint ib = 2u;
        const uint column = 512u + group * 16u + l;
        /* Widen loads only; preserve the eight-element FP32 chain below.
         * Packed types retain scalar activation and half-scale alignment. */
        const packed_float4 y0 = *(device const packed_float4 *)(m + column);
        const packed_float4 y1 = *(device const packed_float4 *)(m + column + 4u);
        const float y[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
        for (uint r = 0; r < 2u && row0 + r < args.out_rows; r++) {
            device const uchar *blk = (device const uchar *)(db +
                (uint64_t)(row0 + r) * args.row_bytes + (uint64_t)ib * 84u);
            const packed_half2 scales = *(device const packed_half2 *)(blk + 80);
            const float d = (float)scales.x;
            const float dmin = (float)scales.y;
            const uint sc = blk[group];
            const float ds = d * (float)(sc & 0xFu), dm = dmin * (float)(sc >> 4);
            device const uchar *qs = blk + 16u + q_base + l;
            const packed_uchar4 q0 = *(device const packed_uchar4 *)qs;
            const packed_uchar4 q1 = *(device const packed_uchar4 *)(qs + 4u);
            const uchar q[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            for (uint i = 0; i < 8u; i++) acc[r] += (ds * (float)((q[i] >> shift) & 3u) - dm) * y[i];
        }
    }
    for (uint r = 0; r < 2u && row0 + r < args.out_rows; r++) {
        const float v = simd_sum(acc[r]);
        if (tiisg == 0) part[pair * args.out_rows + row0 + r] = v;
    }
}

/* MXFP4 routed down rows with four blocks per lane requested before the
 * accumulation chain.  The shipped qwen4_row_dot loop for type 39 compiles
 * to s = t0*y0 + t1*y1 + t2*y16 + t3*y17 (left to right), acc += s*d; that
 * order is spelled out with reassociation pinned off (tests/test_qwen4_kernels.c
 * pins the rows against kernel_qwen4_moe_down). Shared slots retain their
 * original quantization and per-row accumulation order. */
#define QWEN4_MXFP4_PF_ACC_TO(ACC_, E_, Q0_, Q1_, Y0_, Y1_, Y16_, Y17_) do { \
    const float d_ = ds4_metal_e8m0_to_f32(E_); \
    const float t0_ = ds4_metal_mxfp4_values[(Q0_) & 0xfu], t1_ = ds4_metal_mxfp4_values[(Q1_) & 0xfu]; \
    const float t2_ = ds4_metal_mxfp4_values[(Q0_) >> 4], t3_ = ds4_metal_mxfp4_values[(Q1_) >> 4]; \
    float s_ = t0_ * (Y0_); \
    /*ACC1*/ s_ = fma(t1_, (Y1_), s_);\
    /*ACC2*/ s_ = fma(t2_, (Y16_), s_);\
    /*ACC3*/ s_ = fma(t3_, (Y17_), s_);\
    /*ACC4*/ (ACC_) = (ACC_) + s_ * d_;\
} while (0)
#define QWEN4_MXFP4_PF_ACC(E_, Q0_, Q1_, Y0_, Y1_, Y16_, Y17_) \
    QWEN4_MXFP4_PF_ACC_TO(acc, E_, Q0_, Q1_, Y0_, Y1_, Y16_, Y17_)

constant bool qwen4_mxfp4_pair_rows [[function_constant(907)]];

kernel void kernel_qwen4_moe_down_mxfp4_pf(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *down_base,
        device const int32_t *selected,
        device const float   *mid,
        device float         *part,
        device const char    *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = qwen4_moe_slot(args, tgpig.y, tgpig.z);
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = is_function_constant_defined(qwen4_mv_rows) ? qwen4_mv_rows : 2u;
    const uint dim = is_function_constant_defined(qwen4_mv_dim) ? qwen4_mv_dim : args.in_dim;
    const uint st = is_function_constant_defined(qwen4_mv_shared_type) ? qwen4_mv_shared_type : args.shared_type;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    device const float *m = mid + pair * args.in_dim;
    const bool pair_rows = is_function_constant_defined(qwen4_mxfp4_pair_rows) && qwen4_mxfp4_pair_rows;
    if (slot == args.n_slots) {
        if (pair_rows && nr == 2u && st == 8u && row0 + 1u < args.out_rows) {
            const uint64_t off = (uint64_t)row0 * args.shared_row_bytes;
            const float2 v = qwen4_q8_pair_dot(sh_down + off,
                sh_down + off + args.shared_row_bytes, m, dim, tiisg);
            if (tiisg == 0) {
                part[pair * args.out_rows + row0] = v.x;
                part[pair * args.out_rows + row0 + 1u] = v.y;
            }
        } else {
            for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
                const float v = qwen4_row_dot(sh_down + (uint64_t)r * args.shared_row_bytes, m, st, dim, tiisg);
                if (tiisg == 0) part[pair * args.out_rows + r] = v;
            }
        }
        return;
    }
    device const char *db = qwen4_expert_base(down_base,
        (uint)selected[(uint64_t)tok * args.n_slots + slot], args.expert_bytes);
    const uint ix = tiisg / 8, it = tiisg % 8;
    const uint nb = dim / 32;
    if (pair_rows && nr == 2u) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
        /* Batched rows benefit from interleaving independent chains. The
         * function constant removes this register footprint from single-token
         * decode. Activation loads are shared; block/FMA order is unchanged. */
        const bool two = row0 + 1u < args.out_rows;
        device const uchar *rowa = (device const uchar *)(db + (uint64_t)row0 * args.row_bytes);
        device const uchar *rowb = rowa + (two ? args.row_bytes : 0u);
        float acca = 0.0f, accb = 0.0f;
        uint ib = ix;
        for (; ib + 12u < nb; ib += 16u) {
            device const uchar *a0 = rowa + (uint64_t)ib * 17u;
            device const uchar *a1 = a0 + 4u * 17u, *a2 = a0 + 8u * 17u, *a3 = a0 + 12u * 17u;
            device const uchar *c0 = rowb + (uint64_t)ib * 17u;
            device const uchar *c1 = c0 + 4u * 17u, *c2 = c0 + 8u * 17u, *c3 = c0 + 12u * 17u;
            const uchar ea0 = a0[0], ea1 = a1[0], ea2 = a2[0], ea3 = a3[0];
            const uint pa0 = a0[1 + it * 2u], qa0 = a0[2 + it * 2u];
            const uint pa1 = a1[1 + it * 2u], qa1 = a1[2 + it * 2u];
            const uint pa2 = a2[1 + it * 2u], qa2 = a2[2 + it * 2u];
            const uint pa3 = a3[1 + it * 2u], qa3 = a3[2 + it * 2u];
            const uchar eb0 = c0[0], eb1 = c1[0], eb2 = c2[0], eb3 = c3[0];
            const uint pb0 = c0[1 + it * 2u], qb0 = c0[2 + it * 2u];
            const uint pb1 = c1[1 + it * 2u], qb1 = c1[2 + it * 2u];
            const uint pb2 = c2[1 + it * 2u], qb2 = c2[2 + it * 2u];
            const uint pb3 = c3[1 + it * 2u], qb3 = c3[2 + it * 2u];
            device const float *y0 = m + ib * 32u + it * 2u;
            device const float *y1 = y0 + 128u, *y2 = y0 + 256u, *y3 = y0 + 384u;
            const float y0a = y0[0], y0b = y0[1], y0c = y0[16], y0d = y0[17];
            const float y1a = y1[0], y1b = y1[1], y1c = y1[16], y1d = y1[17];
            const float y2a = y2[0], y2b = y2[1], y2c = y2[16], y2d = y2[17];
            const float y3a = y3[0], y3b = y3[1], y3c = y3[16], y3d = y3[17];
            QWEN4_MXFP4_PF_ACC_TO(acca, ea0, pa0, qa0, y0a, y0b, y0c, y0d);
            QWEN4_MXFP4_PF_ACC_TO(acca, ea1, pa1, qa1, y1a, y1b, y1c, y1d);
            QWEN4_MXFP4_PF_ACC_TO(acca, ea2, pa2, qa2, y2a, y2b, y2c, y2d);
            QWEN4_MXFP4_PF_ACC_TO(acca, ea3, pa3, qa3, y3a, y3b, y3c, y3d);
            QWEN4_MXFP4_PF_ACC_TO(accb, eb0, pb0, qb0, y0a, y0b, y0c, y0d);
            QWEN4_MXFP4_PF_ACC_TO(accb, eb1, pb1, qb1, y1a, y1b, y1c, y1d);
            QWEN4_MXFP4_PF_ACC_TO(accb, eb2, pb2, qb2, y2a, y2b, y2c, y2d);
            QWEN4_MXFP4_PF_ACC_TO(accb, eb3, pb3, qb3, y3a, y3b, y3c, y3d);
        }
        for (; ib < nb; ib += 4u) {
            device const uchar *a0 = rowa + (uint64_t)ib * 17u;
            device const uchar *c0 = rowb + (uint64_t)ib * 17u;
            const uchar ea0 = a0[0], eb0 = c0[0];
            const uint pa0 = a0[1 + it * 2u], qa0 = a0[2 + it * 2u];
            const uint pb0 = c0[1 + it * 2u], qb0 = c0[2 + it * 2u];
            device const float *y0 = m + ib * 32u + it * 2u;
            const float y0a = y0[0], y0b = y0[1], y0c = y0[16], y0d = y0[17];
            QWEN4_MXFP4_PF_ACC_TO(acca, ea0, pa0, qa0, y0a, y0b, y0c, y0d);
            QWEN4_MXFP4_PF_ACC_TO(accb, eb0, pb0, qb0, y0a, y0b, y0c, y0d);
        }
        const float va = simd_sum(acca), vb = simd_sum(accb);
        if (tiisg == 0) {
            part[pair * args.out_rows + row0] = va;
            if (two) part[pair * args.out_rows + row0 + 1u] = vb;
        }
        return;
    }
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
        device const uchar *row = (device const uchar *)(db + (uint64_t)r * args.row_bytes);
        float acc = 0.0f;
        uint ib = ix;
        for (; ib + 12u < nb; ib += 16u) {
            device const uchar *b0 = row + (uint64_t)ib * 17u;
            device const uchar *b1 = b0 + 4u * 17u, *b2 = b0 + 8u * 17u, *b3 = b0 + 12u * 17u;
            device const float *y0 = m + ib * 32u + it * 2u;
            device const float *y1 = y0 + 128u, *y2 = y0 + 256u, *y3 = y0 + 384u;
            const uchar e0 = b0[0], e1 = b1[0], e2 = b2[0], e3 = b3[0];
            const uint p0 = b0[1 + it * 2u], q0 = b0[2 + it * 2u];
            const uint p1 = b1[1 + it * 2u], q1 = b1[2 + it * 2u];
            const uint p2 = b2[1 + it * 2u], q2 = b2[2 + it * 2u];
            const uint p3 = b3[1 + it * 2u], q3 = b3[2 + it * 2u];
            const float y0a = y0[0], y0b = y0[1], y0c = y0[16], y0d = y0[17];
            const float y1a = y1[0], y1b = y1[1], y1c = y1[16], y1d = y1[17];
            const float y2a = y2[0], y2b = y2[1], y2c = y2[16], y2d = y2[17];
            const float y3a = y3[0], y3b = y3[1], y3c = y3[16], y3d = y3[17];
            QWEN4_MXFP4_PF_ACC(e0, p0, q0, y0a, y0b, y0c, y0d);
            QWEN4_MXFP4_PF_ACC(e1, p1, q1, y1a, y1b, y1c, y1d);
            QWEN4_MXFP4_PF_ACC(e2, p2, q2, y2a, y2b, y2c, y2d);
            QWEN4_MXFP4_PF_ACC(e3, p3, q3, y3a, y3b, y3c, y3d);
        }
        for (; ib < nb; ib += 4u) {
            device const uchar *b0 = row + (uint64_t)ib * 17u;
            device const float *y0 = m + ib * 32u + it * 2u;
            const uchar e0 = b0[0];
            const uint p0 = b0[1 + it * 2u], q0 = b0[2 + it * 2u];
            const float y0a = y0[0], y0b = y0[1], y0c = y0[16], y0d = y0[17];
            QWEN4_MXFP4_PF_ACC(e0, p0, q0, y0a, y0b, y0c, y0d);
        }
        const float v = simd_sum(acc);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
    }
}

#define QWEN4_MXFP4_GROUPED_ROWS(ACC_, M_, IB_) do { \
    device const float *y0 = (M_) + (IB_) * 32u + it * 2u; \
    device const float *y1 = y0 + 128u, *y2 = y0 + 256u, *y3 = y0 + 384u; \
    const float y0a = y0[0], y0b = y0[1], y0c = y0[16], y0d = y0[17]; \
    const float y1a = y1[0], y1b = y1[1], y1c = y1[16], y1d = y1[17]; \
    const float y2a = y2[0], y2b = y2[1], y2c = y2[16], y2d = y2[17]; \
    const float y3a = y3[0], y3b = y3[1], y3c = y3[16], y3d = y3[17]; \
    QWEN4_MXFP4_PF_ACC_TO(ACC_, e0, p0, q0, y0a, y0b, y0c, y0d); \
    QWEN4_MXFP4_PF_ACC_TO(ACC_, e1, p1, q1, y1a, y1b, y1c, y1d); \
    QWEN4_MXFP4_PF_ACC_TO(ACC_, e2, p2, q2, y2a, y2b, y2c, y2d); \
    QWEN4_MXFP4_PF_ACC_TO(ACC_, e3, p3, q3, y3a, y3b, y3c, y3d); \
} while (0)
#define QWEN4_MXFP4_GROUPED_TAIL(ACC_, M_, IB_) do { \
    device const float *y0 = (M_) + (IB_) * 32u + it * 2u; \
    const float y0a = y0[0], y0b = y0[1], y0c = y0[16], y0d = y0[17]; \
    QWEN4_MXFP4_PF_ACC_TO(ACC_, e0, p0, q0, y0a, y0b, y0c, y0d); \
} while (0)

/* One pass of the grouped down kernel over NJ pairs of one expert: the
 * per-token kernel's prefetched chain per row, one named accumulator per
 * pair, no per-pair branch inside the block loop. */
template <uint NJ>
static inline void qwen4_moe_down_mxfp4_pass(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *down_base, device const float *mid, device float *part,
        device const int32_t *list, uint64_t ebase, uint row0, uint nb, uint ix, uint it, ushort tiisg) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    uint pairs[NJ];
    device const float *ms[NJ];
#pragma unroll
    for (uint j = 0; j < NJ; j++) {
        pairs[j] = (uint)list[j];
        ms[j] = mid + (uint64_t)pairs[j] * args.in_dim;
    }
    /* both rows of the simdgroup walk their blocks together: two
     * independent chains per pair in flight instead of one after the other;
     * each chain's own order is the per-token kernel's */
    const bool two = row0 + 1u < args.out_rows;
    device const uchar *rowa = (device const uchar *)(down_base + ebase + (uint64_t)row0 * args.row_bytes);
    device const uchar *rowb = rowa + (two ? args.row_bytes : 0u);
    float acca[NJ], accb[NJ];
#pragma unroll
    for (uint j = 0; j < NJ; j++) { acca[j] = 0.0f; accb[j] = 0.0f; }
    uint ib = ix;
    for (; ib + 12u < nb; ib += 16u) {
        device const uchar *a0 = rowa + (uint64_t)ib * 17u;
        device const uchar *a1 = a0 + 4u * 17u, *a2 = a0 + 8u * 17u, *a3 = a0 + 12u * 17u;
        device const uchar *c0 = rowb + (uint64_t)ib * 17u;
        device const uchar *c1 = c0 + 4u * 17u, *c2 = c0 + 8u * 17u, *c3 = c0 + 12u * 17u;
        const uchar ea0 = a0[0], ea1 = a1[0], ea2 = a2[0], ea3 = a3[0];
        const uint pa0 = a0[1 + it * 2u], qa0 = a0[2 + it * 2u];
        const uint pa1 = a1[1 + it * 2u], qa1 = a1[2 + it * 2u];
        const uint pa2 = a2[1 + it * 2u], qa2 = a2[2 + it * 2u];
        const uint pa3 = a3[1 + it * 2u], qa3 = a3[2 + it * 2u];
        const uchar eb0 = c0[0], eb1 = c1[0], eb2 = c2[0], eb3 = c3[0];
        const uint pb0 = c0[1 + it * 2u], qb0 = c0[2 + it * 2u];
        const uint pb1 = c1[1 + it * 2u], qb1 = c1[2 + it * 2u];
        const uint pb2 = c2[1 + it * 2u], qb2 = c2[2 + it * 2u];
        const uint pb3 = c3[1 + it * 2u], qb3 = c3[2 + it * 2u];
#pragma unroll
        for (uint j = 0; j < NJ; j++) {
            device const float *y0 = ms[j] + ib * 32u + it * 2u;
            device const float *y1 = y0 + 128u, *y2 = y0 + 256u, *y3 = y0 + 384u;
            const float y0a = y0[0], y0b = y0[1], y0c = y0[16], y0d = y0[17];
            const float y1a = y1[0], y1b = y1[1], y1c = y1[16], y1d = y1[17];
            const float y2a = y2[0], y2b = y2[1], y2c = y2[16], y2d = y2[17];
            const float y3a = y3[0], y3b = y3[1], y3c = y3[16], y3d = y3[17];
            QWEN4_MXFP4_PF_ACC_TO(acca[j], ea0, pa0, qa0, y0a, y0b, y0c, y0d);
            QWEN4_MXFP4_PF_ACC_TO(acca[j], ea1, pa1, qa1, y1a, y1b, y1c, y1d);
            QWEN4_MXFP4_PF_ACC_TO(acca[j], ea2, pa2, qa2, y2a, y2b, y2c, y2d);
            QWEN4_MXFP4_PF_ACC_TO(acca[j], ea3, pa3, qa3, y3a, y3b, y3c, y3d);
            QWEN4_MXFP4_PF_ACC_TO(accb[j], eb0, pb0, qb0, y0a, y0b, y0c, y0d);
            QWEN4_MXFP4_PF_ACC_TO(accb[j], eb1, pb1, qb1, y1a, y1b, y1c, y1d);
            QWEN4_MXFP4_PF_ACC_TO(accb[j], eb2, pb2, qb2, y2a, y2b, y2c, y2d);
            QWEN4_MXFP4_PF_ACC_TO(accb[j], eb3, pb3, qb3, y3a, y3b, y3c, y3d);
        }
    }
    for (; ib < nb; ib += 4u) {
        device const uchar *a0 = rowa + (uint64_t)ib * 17u;
        device const uchar *c0 = rowb + (uint64_t)ib * 17u;
        const uchar ea0 = a0[0], eb0 = c0[0];
        const uint pa0 = a0[1 + it * 2u], qa0 = a0[2 + it * 2u];
        const uint pb0 = c0[1 + it * 2u], qb0 = c0[2 + it * 2u];
#pragma unroll
        for (uint j = 0; j < NJ; j++) {
            device const float *y0 = ms[j] + ib * 32u + it * 2u;
            const float y0a = y0[0], y0b = y0[1], y0c = y0[16], y0d = y0[17];
            QWEN4_MXFP4_PF_ACC_TO(acca[j], ea0, pa0, qa0, y0a, y0b, y0c, y0d);
            QWEN4_MXFP4_PF_ACC_TO(accb[j], eb0, pb0, qb0, y0a, y0b, y0c, y0d);
        }
    }
#pragma unroll
    for (uint j = 0; j < NJ; j++) {
        const float va = simd_sum(acca[j]);
        const float vb = simd_sum(accb[j]);
        if (tiisg == 0) {
            part[(uint64_t)pairs[j] * args.out_rows + row0] = va;
            if (two) part[(uint64_t)pairs[j] * args.out_rows + row0 + 1u] = vb;
        }
    }
}
#undef QWEN4_MXFP4_GROUPED_ROWS
#undef QWEN4_MXFP4_GROUPED_TAIL

/* kernel_qwen4_moe_down_mxfp4_pf over the batch's distinct experts, on the
 * same list ownership as kernel_qwen4_moe_mid_q4k_grouped: the owning
 * threadgroup walks each of its rows once per four pairs of the list, every
 * pair with its own accumulator and the pinned chain order, so each pair's
 * rows are the per-token kernel's bit for bit. */
kernel void kernel_qwen4_moe_down_mxfp4_grouped(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *down_base,
        device const int32_t *selected,
        device const float   *mid,
        device float         *part,
        device const int32_t *lists,
        device const int32_t *counts,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * 2u;
    if (row0 >= args.out_rows || slot >= args.n_slots || tok >= args.n_tokens) return;
    const uint pair = tok * args.n_slots + slot;
    const int32_t expert = selected[pair];
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0) {
            for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++) part[(uint64_t)pair * args.out_rows + r] = 0.0f;
        }
        return;
    }
    device const int32_t *list = lists + (uint64_t)(uint)expert * args.list_cap;
    const uint count = min((uint)counts[expert], args.list_cap);
    if (count == 0 || (uint)list[0] != pair) return;
    const uint64_t ebase = (uint64_t)(uint)expert * args.expert_bytes;
    const uint ix = tiisg / 8, it = tiisg % 8;
    const uint nb = args.in_dim / 32;
    /* Up to four pairs per pass, the count a template constant (see the mid
     * kernel). */
    for (uint j0 = 0; j0 < count; j0 += 4u) {
        const uint nj = min(4u, count - j0);
        switch (nj) {
        case 1u: qwen4_moe_down_mxfp4_pass<1>(args, down_base, mid, part, list + j0, ebase, row0, nb, ix, it, tiisg); break;
        case 2u: qwen4_moe_down_mxfp4_pass<2>(args, down_base, mid, part, list + j0, ebase, row0, nb, ix, it, tiisg); break;
        case 3u: qwen4_moe_down_mxfp4_pass<3>(args, down_base, mid, part, list + j0, ebase, row0, nb, ix, it, tiisg); break;
        default: qwen4_moe_down_mxfp4_pass<4>(args, down_base, mid, part, list + j0, ebase, row0, nb, ix, it, tiisg); break;
        }
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
    uint32_t tail_base; /* host binds the same value to function constant 905 */
    uint32_t expert_major; /* grid x = tile * row_blocks + row_block, z = 1 */
    uint32_t n_active_expert; /* zero retains original expert IDs in grid.y */
    uint32_t active_expert[512]; /* copied inline by the encoder; no CPU pointer */
};
static_assert(sizeof(ds4_metal_args_qwen4_moe_mm) == 2112u, "Qwen MM host/Metal argument layout");

static inline uint qwen4_moe_mm_expert(constant ds4_metal_args_qwen4_moe_mm &args, uint grid_expert) {
    if (!args.n_active_expert) return grid_expert;
    return grid_expert < args.n_active_expert ? args.active_expert[grid_expert] : args.n_expert;
}

/* Threadgroup -> (row block, first tile).  Expert-major order keeps one
 * expert's tiles adjacent so its rows are reused from cache; the K loop and
 * accumulation order of every tile are unchanged. */
static inline uint2 qwen4_moe_mm_block(constant ds4_metal_args_qwen4_moe_mm & args, uint3 tgpig) {
    if (!args.expert_major) return uint2(tgpig.x, tgpig.z);
    const uint n_rb = (args.out_rows + 31u) / 32u;   /* QWEN4_MM_ROWS-sized blocks */
    return uint2(tgpig.x % n_rb, tgpig.x / n_rb);
}

/* Zero retains runtime dispatch; a bound quantization removes the other
 * dequantizers without changing the tile arithmetic. */
constant uint qwen4_moe_weight_type [[function_constant(900)]];
constant uint qwen4_moe_tail_base [[function_constant(905)]];

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
 * 16-byte word; IQ2XXS shares the header and reads four-byte grid slices;
 * other types take two 8-value steps */
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
    if (type == 16) {
        const uint sb = b / 8, ib32 = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 66);
        const float d = (float)(*(device const half *)blk);
        /* A GGUF IQ2 block is only two-byte aligned, including alternating
         * 66-byte superblocks. Read the shared 32-value header with its true
         * alignment rather than repeating two independent stage8 decodes. */
        const packed_ushort4 header = *(device const packed_ushort4 *)(blk + 2 + 8 * ib32);
        const uint aux_g = (uint)header.x | ((uint)header.y << 16);
        const uint aux_s = (uint)header.z | ((uint)header.w << 16);
        const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
        const uint grid_pair = aux_g >> (8 * q0);
        const uint sign_pair = aux_s >> (7 * q0);
#pragma unroll
        for (uint h = 0; h < 2; h++) {
            const uint code = (grid_pair >> (8 * h)) & 0xFFu;
            const uint signs = ds4_metal_ksigns_iq2xs[(sign_pair >> (7 * h)) & 127u];
            /* The codebook is ulong-aligned. Each four-byte slice is aligned
             * for uchar4, so these loads never overread a grid entry. Keep
             * dl * grid * sign and the final D conversion exactly as stage8. */
            constant const uchar4 *grid = (constant const uchar4 *)(ds4_metal_iq2xxs_grid + code);
            const uchar4 lo = grid[0], hi = grid[1];
#pragma unroll
            for (uint i = 0; i < 4; i++) dst[h * 8 + i] = (D)(dl * (float)lo[i] * ((signs >> i) & 1u ? -1.0f : 1.0f));
#pragma unroll
            for (uint i = 0; i < 4; i++) dst[h * 8 + 4 + i] = (D)(dl * (float)hi[i] * ((signs >> (i + 4)) & 1u ? -1.0f : 1.0f));
        }
        return;
    }
    if (type == 39) {
        device const uchar *blk = (device const uchar *)(row + (uint64_t)b * 17);
        const float d = ds4_metal_e8m0_to_f32(blk[0]);
        const bool hi = q0 >= 2;
        /* Seventeen-byte blocks have no wider alignment guarantee. Read the
         * sixteen packed code bytes once, retaining stage8's multiplication
         * and final storage conversion for each low/high nibble. */
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            const packed_uchar4 codes = *(device const packed_uchar4 *)(blk + 1 + j * 4);
#pragma unroll
            for (uint i = 0; i < 4; i++) {
                const uint byte = codes[i];
                dst[j * 4 + i] = (D)(d * ds4_metal_mxfp4_values[hi ? (byte >> 4) : (byte & 0xfu)]);
            }
        }
        return;
    }
    qwen4_mm_stage8(row, b, q0, type, dst);
    qwen4_mm_stage8(row, b, q0 + 1, type, dst + 8);
}

/* Raw words of one 32-block (K step) of an expert row, for the tensor tiles'
 * register prefetch: Q4_K keeps the 16-byte header (d, dmin, scales) and the
 * 16 nibble bytes of the quarter pair; Q2_K keeps d|dmin, the group's scale
 * nibble and the 16 bit-plane bytes of its group pair; IQ2XXS keeps d, the
 * 32-bit grid indices and the sign/scale word of the block; MXFP4 keeps the
 * 17 block bytes. */
struct qwen4_raw16 { uint4 h; uint4 q; uchar m[17]; };
static inline qwen4_raw16 qwen4_load_raw16(device const char *row, uint b, uint q0, uint type) {
    qwen4_raw16 r;
    if (type == 12) {
        const uint sb = b / 8, group = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 144);
        r.h = *(device const uint4 *)blk;
        r.q = *(device const uint4 *)(blk + 16 + (group >> 1) * 32 + q0 * 8);
    } else if (type == 10) {
        const uint sb = b / 8, g = (b % 8) * 2 + q0 / 2;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 84);
        r.h.x = *(device const uint *)(blk + 80);   /* d | dmin << 16 */
        r.h.y = blk[g];                            /* scale | min nibbles */
        device const uint *qs = (device const uint *)(blk + 16 + 32u * (g / 8u) + 16u * (g & 1u));
        r.q.x = qs[0]; r.q.y = qs[1]; r.q.z = qs[2]; r.q.w = qs[3];
    } else if (type == 16) {
        const uint sb = b / 8, ib32 = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 66);
        device const ushort *q2 = (device const ushort *)(blk + 2) + 4 * ib32;
        const uint ag = (uint)q2[0] | ((uint)q2[1] << 16);
        const uint auxs = (uint)q2[2] | ((uint)q2[3] << 16);
        r.h.x = ((uint)*(device const ushort *)blk) | (ag << 16);
        r.h.y = (ag >> 16) | (auxs << 16);
        r.h.z = auxs >> 16;
    } else {
        device const uchar *blk = (device const uchar *)(row + (uint64_t)b * 17);
#pragma unroll
        for (uint i = 0; i < 17; i++) r.m[i] = blk[i];
    }
    return r;
}
static inline void qwen4_dequant_raw16(qwen4_raw16 r, uint b, uint q0, uint type, threadgroup half *dst) {
    if (type == 12) {
        const uint group = b % 8;
        const float d = (float)as_type<half>((ushort)(r.h.x & 0xFFFFu));
        const float dmin = (float)as_type<half>((ushort)(r.h.x >> 16));
        const uint scw[3] = { r.h.y, r.h.z, r.h.w };
#define QWEN4_SCB(i) ((scw[(i) >> 2] >> (8u * ((i) & 3u))) & 0xFFu)
        uint sN, mn;
        if (group < 4) { sN = QWEN4_SCB(group) & 63u; mn = QWEN4_SCB(group + 4) & 63u; }
        else { sN = (QWEN4_SCB(group + 4) & 0xFu) | ((QWEN4_SCB(group - 4) & 0xC0u) >> 2); mn = (QWEN4_SCB(group + 4) >> 4) | ((QWEN4_SCB(group) & 0xC0u) >> 2); }
#undef QWEN4_SCB
        const float ds = d * (float)sN, dm = dmin * (float)mn;
        const uint shift = (group & 1u) * 4u;
        for (uint i = 0; i < 16; i++) dst[i] = (half)(ds * (float)((r.q[i >> 2] >> (8u * (i & 3u) + shift)) & 0xFu) - dm);
        return;
    }
    if (type == 10) {
        const uint g = (b % 8) * 2 + q0 / 2;
        const float d = (float)as_type<half>((ushort)(r.h.x & 0xFFFFu));
        const float dmin = (float)as_type<half>((ushort)(r.h.x >> 16));
        const float ds = d * (float)(r.h.y & 0xFu), dm = dmin * (float)(r.h.y >> 4);
        const uint shift = ((g / 2u) & 3u) * 2u;
        for (uint i = 0; i < 16; i++) {
            const uint byte = (r.q[i >> 2] >> (8u * (i & 3u))) & 0xFFu;
            dst[i] = (half)(ds * (float)((byte >> shift) & 3u) - dm);
        }
        return;
    }
    if (type == 16) {
        const uint ag = (r.h.x >> 16) | (r.h.y << 16);
        const uint auxs = (r.h.y >> 16) | (r.h.z << 16);
        const float d = (float)as_type<half>((ushort)(r.h.x & 0xFFFFu));
        const float dl = d * (0.5f + (float)(auxs >> 28)) * 0.25f;
        for (uint h = 0; h < 2; h++) {
            const uint qq = q0 + h;
            constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2xxs_grid + ((ag >> (8u * qq)) & 0xFFu));
            const uint signs = ds4_metal_ksigns_iq2xs[(auxs >> (7u * qq)) & 127u];
            for (uint i = 0; i < 8; i++) dst[h * 8 + i] = (half)(dl * (float)grid[i] * ((signs >> i) & 1u ? -1.0f : 1.0f));
        }
        return;
    }
    const float d = ds4_metal_e8m0_to_f32(r.m[0]);
    const bool hi = q0 >= 2;
    for (uint i = 0; i < 16; i++) {
        const uint byte = r.m[1 + i];
        dst[i] = (half)(d * ds4_metal_mxfp4_values[hi ? (byte >> 4) : (byte & 0xfu)]);
    }
}

/* The K32 path reuses a Q4_K header over eight staging steps. Match
 * stage8's FP32 dequantization and round only the completed value to half. */
static inline void qwen4_mm_q4_header8(uint4 header, uint2 codes, uint group,
                                      threadgroup half *dst) {
    const float d = (float)as_type<half>((ushort)(header.x & 0xFFFFu));
    const float dmin = (float)as_type<half>((ushort)(header.x >> 16));
    const uint scw[3] = { header.y, header.z, header.w };
#define QWEN4_HEADER_BYTE(i) ((scw[(i) >> 2] >> (8u * ((i) & 3u))) & 0xFFu)
    uint scale, mn;
    if (group < 4u) {
        scale = QWEN4_HEADER_BYTE(group) & 63u;
        mn = QWEN4_HEADER_BYTE(group + 4u) & 63u;
    } else {
        scale = (QWEN4_HEADER_BYTE(group + 4u) & 15u) | ((QWEN4_HEADER_BYTE(group - 4u) & 0xC0u) >> 2);
        mn = (QWEN4_HEADER_BYTE(group + 4u) >> 4) | ((QWEN4_HEADER_BYTE(group) & 0xC0u) >> 2);
    }
#undef QWEN4_HEADER_BYTE
    const float ds = d * (float)scale, dm = dmin * (float)mn;
    const uint shift = (group & 1u) * 4u;
    for (uint i = 0; i < 8u; i++)
        dst[i] = (half)(ds * (float)((codes[i >> 2] >> (8u * (i & 3u) + shift)) & 15u) - dm);
}

/* mid[t][slot][r] = silu(gate . x) * (up . x) for every (token, slot) routed
 * to expert e, as 32-row x (8*NT)-token tiles: A (weights, half) and B
 * (activations, half) are staged in threadgroup memory per KS-wide K step
 * and multiplied with simdgroup matrices into float accumulators, so each
 * expert row is read once per token tile. The half-storage variants only
 * move the existing B-operand conversion to a shared copy of x and to the
 * mid epilogue; all matrix accumulators and activation math remain float. */
template <uint NT, typename X, bool HALF_SHADOW, uint KS, bool PACKED = false>
static inline void qwen4_moe_mm_mid_impl(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const X       *x,          /* [T][in_dim] */
        device float         *mid,        /* [T][n_out][out_rows] */
        device half          *mid_shadow,
        device const uint    *offsets,
        threadgroup half *Ag, threadgroup half *Au, threadgroup half *Bs,
        threadgroup float *Cs,
        uint3 tgpig, ushort tid, ushort sgitg) {
    constexpr uint TT = QWEN4_MM_TOKS * NT;
    static_assert(!PACKED || (NT == 4u && KS == 32u && HALF_SHADOW), "packed MoE geometry");
    static_assert(KS == 32u || KS == 64u, "supported MoE K staging widths");
    static_assert((KS * TT / 128u) % 4u == 0u, "B staging uses four-value loads");
    const uint2 block = qwen4_moe_mm_block(args, tgpig);
    const uint rb = block.x, e = qwen4_moe_mm_expert(args, tgpig.y);
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    if (!count) return;
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 8u ? 8u : remainder <= 16u ? 16u : remainder <= 32u ? 32u : 64u;
        if (TT < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != TT) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < TT) {
            work_count = count - remainder;
        }
    }
    device const char *gbase = qwen4_expert_base(gate_base, e, args.expert_bytes);
    device const char *ubase = qwen4_expert_base(up_base, e, args.expert_bytes);
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * QWEN4_MM_ROWS;
    const uint nk = args.in_dim / KS;
    for (uint tile = block.y; tile * TT < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * TT;
        const uint n_tile = min((uint)TT, work_count - tile * TT);
        simdgroup_float8x8 Cg[NT], Cu[NT];
        for (uint nt = 0; nt < NT; nt++) {
            Cg[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            Cu[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        const uint my_tok = tid % TT;
        const int my_pair = my_tok < n_tile ? list[t0 + my_tok] : -1;
        const uint my_t = my_pair >= 0 ? (uint)my_pair / args.n_slots : 0;
        uint4 gate_header, up_header;
        for (uint kb = 0; kb < nk; kb++) {
            /* A: 32 rows x KS columns; each thread stages KS/4 values */
            {
                const uint r = tid / 4, q = tid % 4;   /* quarter of this row's K tile */
                threadgroup half *dg = Ag + r * KS + q * (KS / 4u);
                threadgroup half *du = Au + r * KS + q * (KS / 4u);
                if (row0 + r < args.out_rows) {
                    device const char *grow = gbase + (uint64_t)(row0 + r) * args.row_bytes;
                    device const char *urow = ubase + (uint64_t)(row0 + r) * args.row_bytes;
                    const uint b = kb * (KS / 32u) + (KS == 64u ? q >> 1u : 0u);
                    const uint quarter0 = KS == 64u ? (q & 1u) * 2u : q;
                    const uint type = qwen4_moe_weight_type ? qwen4_moe_weight_type : args.weight_type;
                    if constexpr (KS == 32u) {
                        if (type == 12u) {
                            const uint sb = b / 8u, group = b % 8u;
                            device const uchar *gb = (device const uchar *)(grow + (uint64_t)sb * 144u);
                            device const uchar *ub = (device const uchar *)(urow + (uint64_t)sb * 144u);
                            if ((kb & 7u) == 0u) {
                                gate_header = *(device const uint4 *)gb;
                                up_header = *(device const uint4 *)ub;
                            }
                            const uint offset = 16u + (group >> 1u) * 32u + quarter0 * 8u;
                            qwen4_mm_q4_header8(gate_header, *(device const uint2 *)(gb + offset), group, dg);
                            qwen4_mm_q4_header8(up_header, *(device const uint2 *)(ub + offset), group, du);
                        } else {
                            qwen4_mm_stage8(grow, b, quarter0, type, dg);
                            qwen4_mm_stage8(urow, b, quarter0, type, du);
                        }
                    } else {
                        qwen4_mm_stage16(grow, b, quarter0, type, dg);
                        qwen4_mm_stage16(urow, b, quarter0, type, du);
                    }
                } else {
                    for (uint i = 0; i < KS / 4u; i++) { dg[i] = 0.0h; du[i] = 0.0h; }
                }
            }
            /* B: KS columns x TT tokens; distribute all K values over 128 threads. */
            if constexpr (!PACKED) {
                const uint tok = tid % TT, kq = tid / TT;
                constexpr uint K_PER_THREAD = KS * TT / 128;
                device const X *xr = x + (uint64_t)my_t * args.in_dim + kb * KS + kq * K_PER_THREAD;
                for (uint j = 0; j < K_PER_THREAD; j += 4) {
                    if constexpr (is_same<X, half>::value) {
                        const half4 v = my_pair >= 0 ? *(device const half4 *)(xr + j) : half4(0.0h);
                        Bs[(kq * K_PER_THREAD + j + 0) * TT + tok] = v.x;
                        Bs[(kq * K_PER_THREAD + j + 1) * TT + tok] = v.y;
                        Bs[(kq * K_PER_THREAD + j + 2) * TT + tok] = v.z;
                        Bs[(kq * K_PER_THREAD + j + 3) * TT + tok] = v.w;
                    } else {
                        const float4 v = my_pair >= 0 ? *(device const float4 *)(xr + j) : float4(0.0f);
                        Bs[(kq * K_PER_THREAD + j + 0) * TT + tok] = (half)v.x;
                        Bs[(kq * K_PER_THREAD + j + 1) * TT + tok] = (half)v.y;
                        Bs[(kq * K_PER_THREAD + j + 2) * TT + tok] = (half)v.z;
                        Bs[(kq * K_PER_THREAD + j + 3) * TT + tok] = (half)v.w;
                    }
                }
            }
            if constexpr (PACKED) simdgroup_barrier(mem_flags::mem_threadgroup);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint sub = 0; sub < KS / 8; sub++) {
                simdgroup_half8x8 ag, au, b;
                simdgroup_load(ag, Ag + (sgitg * 8) * KS + sub * 8, KS, 0, false);
                simdgroup_load(au, Au + (sgitg * 8) * KS + sub * 8, KS, 0, false);
                for (uint nt = 0; nt < NT; nt++) {
                    if constexpr (PACKED)
                        simdgroup_load(b, x + (uint64_t)(offsets[e] + t0) * args.in_dim +
                            (kb * KS + sub * 8u) * 32u + nt * 64u, 8u, 0, false);
                    else simdgroup_load(b, Bs + sub * 8 * TT + nt * 8, TT, 0, false);
                    simdgroup_multiply_accumulate(Cg[nt], ag, b, Cg[nt]);
                    simdgroup_multiply_accumulate(Cu[nt], au, b, Cu[nt]);
                }
            }
            if constexpr (PACKED) simdgroup_barrier(mem_flags::mem_threadgroup);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint nt = 0; nt < NT; nt++) {
            simdgroup_store(Cg[nt], Cs + ((uint)sgitg * 2u) * 64u, 8, 0, false);
            simdgroup_store(Cu[nt], Cs + ((uint)sgitg * 2u + 1u) * 64u, 8, 0, false);
            if constexpr (PACKED) simdgroup_barrier(mem_flags::mem_threadgroup);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint first = PACKED ? (uint)sgitg * 64u + (tid & 31u) : tid;
            const uint end = PACKED ? ((uint)sgitg + 1u) * 64u : 4u * 64u;
            for (uint idx = first; idx < end; idx += PACKED ? 32u : 128u) {
                const uint sg = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                const uint row = row0 + sg * 8 + r;
                if constexpr (PACKED) {
                    if (row >= args.out_rows) continue;
                    if (tok >= n_tile) {
                        mid_shadow[(uint64_t)(offsets[e] + t0) * args.out_rows +
                            (row / 8u) * 256u + (tok / 8u) * 64u + (row % 8u) * 8u + tok % 8u] = 0.0h;
                        continue;
                    }
                } else if (tok >= n_tile || row >= args.out_rows) continue;
                const int pair = list[t0 + tok];
                const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
                const float g = Cs[(sg * 2u) * 64u + el], u = Cs[(sg * 2u + 1u) * 64u + el];
                if constexpr (HALF_SHADOW) {
                    /* Round the completed FP32 activation, exactly where
                     * the down kernel would round its float mid operand. */
                    const float v = qwen4_silu(g) * u;
                    mid[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = v;
                    if constexpr (PACKED)
                        mid_shadow[(uint64_t)(offsets[e] + t0) * args.out_rows +
                            (row / 8u) * 256u + (tok / 8u) * 64u + (row % 8u) * 8u + tok % 8u] = (half)v;
                    else mid_shadow[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = (half)v;
                } else {
                    mid[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = qwen4_silu(g) * u;
                }
            }
            if constexpr (PACKED) simdgroup_barrier(mem_flags::mem_threadgroup);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/* Keep the existing float API and its intermediate observable to diagnostics.
 * The half-storage wrapper adds a shadow consumed only by the paired down
 * dispatch; it never changes the public mid tensor's element type. */
template <uint NT, uint KS>
kernel void kernel_qwen4_moe_mm_mid(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char *gate_base, device const char *up_base,
        device const int32_t *lists, device const int32_t *counts,
        device const float *x, device float *mid,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Ag[QWEN4_MM_ROWS * KS];
    threadgroup half Au[QWEN4_MM_ROWS * KS];
    threadgroup half Bs[KS * QWEN4_MM_TOKS * NT];
    threadgroup float Cs[4 * 2 * 64];
    qwen4_moe_mm_mid_impl<NT, float, false, KS>(args, gate_base, up_base,
        lists, counts, x, mid, nullptr, nullptr, Ag, Au, Bs, Cs, tgpig, tid, sgitg);
}

template <uint NT, uint KS>
kernel void kernel_qwen4_moe_mm_mid_f16(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char *gate_base, device const char *up_base,
        device const int32_t *lists, device const int32_t *counts,
        device const half *x, device float *mid, device half *mid_shadow,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Ag[QWEN4_MM_ROWS * KS];
    threadgroup half Au[QWEN4_MM_ROWS * KS];
    threadgroup half Bs[KS * QWEN4_MM_TOKS * NT];
    threadgroup float Cs[4 * 2 * 64];
    qwen4_moe_mm_mid_impl<NT, half, true, KS>(args, gate_base, up_base,
        lists, counts, x, mid, mid_shadow, nullptr, Ag, Au, Bs, Cs, tgpig, tid, sgitg);
}

#define QWEN4_MOE_MM_MID_F32_INSTANCE(NT_, NAME_) \
template [[host_name(NAME_)]] \
kernel void kernel_qwen4_moe_mm_mid<NT_, 64>(constant ds4_metal_args_qwen4_moe_mm &, \
    device const char *, device const char *, device const int32_t *, device const int32_t *, \
    device const float *, device float *, uint3, ushort, ushort);
QWEN4_MOE_MM_MID_F32_INSTANCE(1, "kernel_qwen4_moe_mm_mid_nt1")
QWEN4_MOE_MM_MID_F32_INSTANCE(2, "kernel_qwen4_moe_mm_mid_nt2")
QWEN4_MOE_MM_MID_F32_INSTANCE(4, "kernel_qwen4_moe_mm_mid")
/* 64-token tiles retain the existing K order and 8/16/32-token remainders. */
QWEN4_MOE_MM_MID_F32_INSTANCE(8, "kernel_qwen4_moe_mm_mid_nt8")
#undef QWEN4_MOE_MM_MID_F32_INSTANCE

#define QWEN4_MOE_MM_MID_F16_INSTANCE(NT_) \
template [[host_name("kernel_qwen4_moe_mm_mid_f16_nt" #NT_)]] \
kernel void kernel_qwen4_moe_mm_mid_f16<NT_, 64>(constant ds4_metal_args_qwen4_moe_mm &, \
    device const char *, device const char *, device const int32_t *, device const int32_t *, \
    device const half *, device float *, device half *, uint3, ushort, ushort);
QWEN4_MOE_MM_MID_F16_INSTANCE(1)
QWEN4_MOE_MM_MID_F16_INSTANCE(2)
QWEN4_MOE_MM_MID_F16_INSTANCE(4)
QWEN4_MOE_MM_MID_F16_INSTANCE(8)
#undef QWEN4_MOE_MM_MID_F16_INSTANCE

/* K32 lowers NT4 shared storage from 14 to 8 KiB without changing the
 * sequence of 8-wide matrix products. Existing entry points retain K64. */
template [[host_name("kernel_qwen4_moe_mm_mid_k32_nt4")]]
kernel void kernel_qwen4_moe_mm_mid<4, 32>(constant ds4_metal_args_qwen4_moe_mm &,
    device const char *, device const char *, device const int32_t *, device const int32_t *,
    device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen4_moe_mm_mid_f16_k32_nt4")]]
kernel void kernel_qwen4_moe_mm_mid_f16<4, 32>(constant ds4_metal_args_qwen4_moe_mm &,
    device const char *, device const char *, device const int32_t *, device const int32_t *,
    device const half *, device float *, device half *, uint3, ushort, ushort);

/* part[t][slot][r] = down . mid[t][slot], same tiling with mid as B */
template <uint NT, typename MID, uint KS, bool PACKED = false>
static inline void qwen4_moe_mm_down_impl(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *down_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const MID     *midv,       /* [T][n_out][in_dim] */
        device float         *part,       /* [T][n_out][out_rows] */
        device const uint *offsets,
        threadgroup half *As, threadgroup half *Bs, threadgroup float *Cs,
        uint3 tgpig, ushort tid, ushort sgitg) {
    constexpr uint TT = QWEN4_MM_TOKS * NT;
    static_assert(!PACKED || (NT == 4u && KS == 32u), "packed MoE geometry");
    static_assert(KS == 32u || KS == 64u, "supported MoE K staging widths");
    static_assert((KS * TT / 128u) % 4u == 0u, "B staging uses four-value loads");
    const uint2 block = qwen4_moe_mm_block(args, tgpig);
    const uint rb = block.x, e = qwen4_moe_mm_expert(args, tgpig.y);
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    if (!count) return;
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 8u ? 8u : remainder <= 16u ? 16u : remainder <= 32u ? 32u : 64u;
        if (TT < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != TT) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < TT) {
            work_count = count - remainder;
        }
    }
    device const char *dbase = qwen4_expert_base(down_base, e, args.expert_bytes);
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * QWEN4_MM_ROWS;
    const uint nk = args.in_dim / KS;
    for (uint tile = block.y; tile * TT < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * TT;
        const uint n_tile = min((uint)TT, work_count - tile * TT);
        simdgroup_float8x8 C[NT];
        for (uint nt = 0; nt < NT; nt++) C[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        const uint my_tok = tid % TT;
        const int my_pair = my_tok < n_tile ? list[t0 + my_tok] : -1;
        const uint64_t my_row = my_pair >= 0 ?
            ((uint64_t)((uint)my_pair / args.n_slots) * args.n_out + (uint)my_pair % args.n_slots) : 0;
        for (uint kb = 0; kb < nk; kb++) {
            {
                const uint r = tid / 4, q = tid % 4;
                threadgroup half *dd = As + r * KS + q * (KS / 4u);
                if (row0 + r < args.out_rows) {
                    device const char *drow = dbase + (uint64_t)(row0 + r) * args.row_bytes;
                    const uint b = kb * (KS / 32u) + (KS == 64u ? q >> 1u : 0u);
                    const uint quarter0 = KS == 64u ? (q & 1u) * 2u : q;
                    const uint type = qwen4_moe_weight_type ? qwen4_moe_weight_type : args.weight_type;
                    if constexpr (KS == 32u) qwen4_mm_stage8(drow, b, quarter0, type, dd);
                    else qwen4_mm_stage16(drow, b, quarter0, type, dd);
                } else {
                    for (uint i = 0; i < KS / 4u; i++) dd[i] = 0.0h;
                }
            }
            if constexpr (!PACKED) {
                const uint tok = tid % TT, kq = tid / TT;
                constexpr uint K_PER_THREAD = KS * TT / 128;
                device const MID *mr = midv + my_row * args.in_dim + kb * KS + kq * K_PER_THREAD;
                for (uint j = 0; j < K_PER_THREAD; j += 4) {
                    if constexpr (is_same<MID, half>::value) {
                        const half4 v = my_pair >= 0 ? *(device const half4 *)(mr + j) : half4(0.0h);
                        Bs[(kq * K_PER_THREAD + j + 0) * TT + tok] = v.x;
                        Bs[(kq * K_PER_THREAD + j + 1) * TT + tok] = v.y;
                        Bs[(kq * K_PER_THREAD + j + 2) * TT + tok] = v.z;
                        Bs[(kq * K_PER_THREAD + j + 3) * TT + tok] = v.w;
                    } else {
                        const float4 v = my_pair >= 0 ? *(device const float4 *)(mr + j) : float4(0.0f);
                        Bs[(kq * K_PER_THREAD + j + 0) * TT + tok] = (half)v.x;
                        Bs[(kq * K_PER_THREAD + j + 1) * TT + tok] = (half)v.y;
                        Bs[(kq * K_PER_THREAD + j + 2) * TT + tok] = (half)v.z;
                        Bs[(kq * K_PER_THREAD + j + 3) * TT + tok] = (half)v.w;
                    }
                }
            }
            if constexpr (PACKED) simdgroup_barrier(mem_flags::mem_threadgroup);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint sub = 0; sub < KS / 8; sub++) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, As + (sgitg * 8) * KS + sub * 8, KS, 0, false);
                for (uint nt = 0; nt < NT; nt++) {
                    if constexpr (PACKED)
                        simdgroup_load(b, midv + (uint64_t)(offsets[e] + t0) * args.in_dim +
                            (kb * KS + sub * 8u) * 32u + nt * 64u, 8u, 0, false);
                    else simdgroup_load(b, Bs + sub * 8 * TT + nt * 8, TT, 0, false);
                    simdgroup_multiply_accumulate(C[nt], a, b, C[nt]);
                }
            }
            if constexpr (PACKED) simdgroup_barrier(mem_flags::mem_threadgroup);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint nt = 0; nt < NT; nt++) {
            simdgroup_store(C[nt], Cs + (uint)sgitg * 64u, 8, 0, false);
            if constexpr (PACKED) simdgroup_barrier(mem_flags::mem_threadgroup);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint first = PACKED ? (uint)sgitg * 64u + (tid & 31u) : tid;
            const uint end = PACKED ? ((uint)sgitg + 1u) * 64u : 4u * 64u;
            for (uint idx = first; idx < end; idx += PACKED ? 32u : 128u) {
                const uint sg = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                const uint row = row0 + sg * 8 + r;
                if (tok >= n_tile || row >= args.out_rows) continue;
                const int pair = list[t0 + tok];
                const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
                part[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = Cs[sg * 64u + el];
            }
            if constexpr (PACKED) simdgroup_barrier(mem_flags::mem_threadgroup);
            else threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template <uint NT, typename MID, uint KS>
kernel void kernel_qwen4_moe_mm_down(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const char *down_base, device const int32_t *lists,
        device const int32_t *counts, device const MID *midv, device float *part,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half As[QWEN4_MM_ROWS * KS];
    threadgroup half Bs[KS * QWEN4_MM_TOKS * NT];
    threadgroup float Cs[4 * 64];
    qwen4_moe_mm_down_impl<NT, MID, KS>(args, down_base, lists, counts, midv,
        part, nullptr, As, Bs, Cs, tgpig, tid, sgitg);
}

template [[host_name("kernel_qwen4_moe_mm_down_nt1")]]
kernel void kernel_qwen4_moe_mm_down<1, float, 64>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen4_moe_mm_down_nt2")]]
kernel void kernel_qwen4_moe_mm_down<2, float, 64>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen4_moe_mm_down")]]
kernel void kernel_qwen4_moe_mm_down<4, float, 64>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen4_moe_mm_down_nt8")]]
kernel void kernel_qwen4_moe_mm_down<8, float, 64>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

#define QWEN4_MOE_MM_DOWN_F16_INSTANCE(NT_) \
template [[host_name("kernel_qwen4_moe_mm_down_f16_nt" #NT_)]] \
kernel void kernel_qwen4_moe_mm_down<NT_, half, 64>(constant ds4_metal_args_qwen4_moe_mm &, \
    device const char *, device const int32_t *, device const int32_t *, \
    device const half *, device float *, uint3, ushort, ushort);
QWEN4_MOE_MM_DOWN_F16_INSTANCE(1)
QWEN4_MOE_MM_DOWN_F16_INSTANCE(2)
QWEN4_MOE_MM_DOWN_F16_INSTANCE(4)
QWEN4_MOE_MM_DOWN_F16_INSTANCE(8)
#undef QWEN4_MOE_MM_DOWN_F16_INSTANCE

template [[host_name("kernel_qwen4_moe_mm_down_k32_nt4")]]
kernel void kernel_qwen4_moe_mm_down<4, float, 32>(constant ds4_metal_args_qwen4_moe_mm &,
    device const char *, device const int32_t *, device const int32_t *,
    device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen4_moe_mm_down_f16_k32_nt4")]]
kernel void kernel_qwen4_moe_mm_down<4, half, 32>(constant ds4_metal_args_qwen4_moe_mm &,
    device const char *, device const int32_t *, device const int32_t *,
    device const half *, device float *, uint3, ushort, ushort);

/* rows of floats -> halves (round to nearest even), four values per thread;
 * the tensor-op tiles read their activation operand from this copy. */
struct ds4_metal_args_qwen4_rows_f16 { uint32_t n4; };
kernel void kernel_qwen4_rows_f32_to_f16(
        constant ds4_metal_args_qwen4_rows_f16 & args,
        device const float4 *src,
        device half4         *dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i0 = gid * 4;
    if (i0 + 4 <= args.n4) {
        const float4 a = src[i0], b = src[i0 + 1], c = src[i0 + 2], d = src[i0 + 3];
        dst[i0] = half4(a); dst[i0 + 1] = half4(b); dst[i0 + 2] = half4(c); dst[i0 + 3] = half4(d);
    } else {
        for (uint i = i0; i < args.n4; i++) dst[i] = half4(src[i]);
    }
}

/* Packed operands retain expert-list order. Each expert gets ceil(count/32)*32
 * rows so every matrix load stays inside the allocation, including tile tails.
 * Offsets follow original expert IDs, including empty entries. Pass counts
 * can exclude whole experts while preserving each selected list's order. */
kernel void kernel_qwen4_moe_packed_map(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const int32_t *lists, device const int32_t *counts,
        device const uint *offsets, device int32_t *tokens,
        uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint expert = group.x;
    if (expert >= args.n_expert) return;
    const uint count = counts[expert], padded = offsets[expert + 1u] - offsets[expert];
    for (uint i = tid; i < padded; i += 128u)
        tokens[offsets[expert] + i] = i < count ? lists[(uint64_t)expert * args.list_cap + i] / (int)args.n_slots : -1;
}
struct qwen4_packed_gather_args { uint rows, dim; };
/* Pack each8x8 B operand contiguously. Loads from x remain float4;
 * a2KiB transpose makes stores and subsequent matrix loads contiguous. */
kernel void kernel_qwen4_moe_packed_gather(
        constant qwen4_packed_gather_args &args,
        device const int32_t *tokens, device const float *x, device half *packed,
        uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[32u * 32u];
    const uint ktiles = args.dim / 32u;
    const uint row0 = (group.x / ktiles) * 32u, k0 = (group.x % ktiles) * 32u;
    const uint t = tid / 4u, k = (tid % 4u) * 8u;
    const int token = tokens[row0 + t];
    const float4 va = token >= 0 ? *(device const float4 *)(x + (uint64_t)token * args.dim + k0 + k) : float4(0.0f);
    const float4 vb = token >= 0 ? *(device const float4 *)(x + (uint64_t)token * args.dim + k0 + k + 4u) : float4(0.0f);
    const uint dst = (k / 8u) * 256u + (t / 8u) * 64u + t % 8u;
    for (uint j=0;j<4u;j++) { tile[dst + j*8u]=(half)va[j]; tile[dst + (j+4u)*8u]=(half)vb[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half *out = packed + (uint64_t)row0 * args.dim + k0 * 32u + tid * 8u;
    *(device half4 *)out = *(threadgroup half4 *)(tile + tid * 8u);
    *(device half4 *)(out + 4u) = *(threadgroup half4 *)(tile + tid * 8u + 4u);
}

/* Each SIMD group owns eight weight rows and all 32 tokens. Packed B loads
 * are contiguous 8x8 matrices, so A and the epilogue need only SIMD barriers.
 * Public FP32 mid and the sequential 8-wide products retain their old order. */
kernel void kernel_qwen4_moe_packed_mid(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const char *gate_base, device const char *up_base,
        device const int32_t *lists, device const int32_t *counts,
        device const half *x, device float *mid, device half *mid_shadow,
        device const uint *offsets,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Ag[32 * 32], Au[32 * 32];
    threadgroup float Cs[4 * 2 * 64];
    qwen4_moe_mm_mid_impl<4, half, true, 32, true>(args, gate_base, up_base,
        lists, counts, x, mid, mid_shadow, offsets, Ag, Au, nullptr, Cs, tgpig, tid, sgitg);
}

kernel void kernel_qwen4_moe_packed_down(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const char *down_base, device const int32_t *lists,
        device const int32_t *counts, device const half *midv, device float *part,
        device const uint *offsets,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half As[32 * 32];
    threadgroup float Cs[4 * 64];
    qwen4_moe_mm_down_impl<4, half, 32, true>(args, down_base, lists, counts,
        midv, part, offsets, As, nullptr, Cs, tgpig, tid, sgitg);
}

#ifdef DS4_METAL_HAS_TENSOR
/* Routed expert tiles on the Metal 4 tensor ops (M5 neural accelerators):
 * 64 expert rows x NR1 tokens per threadgroup, K in 32-wide steps.  The
 * staged operands are the same halves the simdgroup kernels stage (the
 * Qwen dequantizers, activations rounded to half); only the cooperative
 * matmul's accumulation order differs, so outputs are close to, not
 * identical with, kernel_qwen4_moe_mm_mid/down (test_moe_mm_tiles_exact
 * bounds the difference).  With tail_base 64 (function constant 905) the
 * 64-token kernel keeps the full tiles and the 32-token kernel takes a
 * remainder of at most 32 tokens.  The activation operand comes pre-rounded to half
 * (kernel_qwen4_rows_f32_to_f16, one pass per call), so each K step gathers
 * 16 bytes per item; the mid epilogue also writes the half copy of `mid`
 * that the down tiles read.  The mid epilogue applies SiLU(gate)*up on the
 * cooperative tensors themselves (gate and up share one element layout),
 * so one float C tile goes through threadgroup memory.  Threadgroup
 * memory, mid: gate A 4 KB + up A 4 KB + B NR1/16 KB while staging, then
 * NR1/4 KB of C; down: A 4 KB + B NR1/16 KB, then NR1/4 KB of C. */
/* Staging copy of one token's 8-wide k slice (8 halves or 8 floats). */
template <typename XT>
inline void qwen4_nax_stage8(threadgroup XT *dst, device const XT *src) {
    if (src) {
        if constexpr (is_same<XT, half>::value) {
            *(threadgroup uint4 *)dst = *(device const uint4 *)src;
        } else {
            *(threadgroup float4 *)dst = *(device const float4 *)src;
            *(threadgroup float4 *)(dst + 4) = *(device const float4 *)(src + 4);
        }
    } else {
        if constexpr (is_same<XT, half>::value) {
            *(threadgroup uint4 *)dst = uint4(0u);
        } else {
            *(threadgroup float4 *)dst = float4(0.0f);
            *(threadgroup float4 *)(dst + 4) = float4(0.0f);
        }
    }
}

/* B operand element type: compensated tiles always stage halves (the float
 * input is split into a half part and a half residual at staging time). */
template <bool COMP, typename XT> struct qwen4_nax_btype { using type = XT; };
template <typename XT> struct qwen4_nax_btype<true, XT> { using type = half; };

template <int NR1, typename XT, bool COMP>
kernel void kernel_qwen4_moe_mm_mid_nax_t(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const XT      *x,          /* [T][in_dim] */
        device float         *mid,
        device half          *midh,       /* [T][n_out][out_rows], the down tiles' operand */
        device half          *midr,       /* [T][n_out][out_rows] residual of midh (COMP) */
        threadgroup char     *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NK = 32;
    constexpr int NB = NR1 * 4 / 128;   /* B staging items per thread (token, 8-wide k slice) */
    using BT = typename qwen4_nax_btype<COMP, XT>::type;
    uint rb, tile0;
    if (args.expert_major) { const uint n_rb = (args.out_rows + NR0 - 1u) / NR0; rb = tgpig.x % n_rb; tile0 = tgpig.x / n_rb; }
    else { rb = tgpig.x; tile0 = tgpig.z; }
    const uint e = qwen4_moe_mm_expert(args, tgpig.y);
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    if (!count) return;
    /* tails: with tail_base 64 the 64-token tiles keep the full tiles and the
     * 32-token kernel takes a remainder of at most 32 tokens */
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 32u ? 32u : 64u;
        if ((uint)NR1 < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != (uint)NR1) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < (uint)NR1) {
            work_count = count - remainder;
        }
    }
    threadgroup half *Ag = (threadgroup half *)shmem;                 /* [64][32] */
    threadgroup half *Au = (threadgroup half *)(shmem + 4096);        /* [64][32] */
    threadgroup BT *Bs = (threadgroup BT *)(shmem + 8192);            /* [NR1][32] */
    threadgroup half *Br = (threadgroup half *)(shmem + 8192 + NR1 * 64); /* [NR1][32] residual (COMP) */
    threadgroup float *Cs = (threadgroup float *)shmem;               /* [NR1 tok][64 row] after the K loop */
    device const char *gbase = qwen4_expert_base(gate_base, e, args.expert_bytes);
    device const char *ubase = qwen4_expert_base(up_base, e, args.expert_bytes);
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * NR0;
    const uint nk = args.in_dim / NK;
    const uint type = qwen4_moe_weight_type ? qwen4_moe_weight_type : args.weight_type;
    auto tA_g = tensor(Ag, dextents<int32_t, 2>(NK, NR0));
    auto tA_u = tensor(Au, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));   /* left operand: k contiguous, one token per column */
    auto tBr = tensor(Br, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    const uint ar = tid / 2, aq = tid % 2;       /* A staging: (row, 16-wide half of the 32-block) */
    for (uint tile = tile0; tile * NR1 < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * NR1;
        const uint n_tile = min((uint)NR1, work_count - tile * NR1);
        auto cG = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA_g), float>();
        auto cU = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA_u), float>();
#pragma unroll
        for (uint16_t i = 0; i < cG.get_capacity(); ++i) { if (cG.is_valid_element(i)) cG[i] = 0.0f; }
#pragma unroll
        for (uint16_t i = 0; i < cU.get_capacity(); ++i) { if (cU.is_valid_element(i)) cU[i] = 0.0f; }
        device const XT *xr[NB];
        threadgroup BT *bdst[NB];
        threadgroup half *rdst[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
            const uint item = (uint)tid + (uint)b * 128u, tok = item / 4u, kq = item % 4u;
            const int pair = tok < n_tile ? list[t0 + tok] : -1;
            xr[b] = pair >= 0 ? x + (uint64_t)((uint)pair / args.n_slots) * args.in_dim + kq * 8u : (device const XT *)0;
            bdst[b] = Bs + tok * NK + kq * 8u;
            if constexpr (COMP) rdst[b] = Br + tok * NK + kq * 8u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   /* previous tile's C tile consumed */
        const bool a_row = row0 + ar < args.out_rows;
        device const char *grow = gbase + (uint64_t)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        device const char *urow = ubase + (uint64_t)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        qwen4_raw16 rg = qwen4_load_raw16(grow, 0, aq * 2, type), ru = qwen4_load_raw16(urow, 0, aq * 2, type);
        for (uint kb = 0; kb < nk; kb++) {
            {
                threadgroup half *dg = Ag + ar * NK + aq * 16;
                threadgroup half *du = Au + ar * NK + aq * 16;
                if (a_row) {
                    qwen4_dequant_raw16(rg, kb, aq * 2, type, dg);
                    qwen4_dequant_raw16(ru, kb, aq * 2, type, du);
                } else {
                    for (uint i = 0; i < 16; i++) { dg[i] = 0.0h; du[i] = 0.0h; }
                }
                if (kb + 1 < nk) { rg = qwen4_load_raw16(grow, kb + 1, aq * 2, type); ru = qwen4_load_raw16(urow, kb + 1, aq * 2, type); }
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if constexpr (COMP) {
                    /* stage xh and the residual xr: x = xh + xr to ~2^-22 relative */
                    if (xr[b]) {
                        const float4 v0 = *(device const float4 *)(xr[b] + kb * NK);
                        const float4 v1 = *(device const float4 *)(xr[b] + kb * NK + 4);
                        const half4 h0 = half4(v0), h1 = half4(v1);
                        *(threadgroup uint2 *)bdst[b] = as_type<uint2>(h0);
                        *(threadgroup uint2 *)(bdst[b] + 4) = as_type<uint2>(h1);
                        *(threadgroup uint2 *)rdst[b] = as_type<uint2>(half4(v0 - float4(h0)));
                        *(threadgroup uint2 *)(rdst[b] + 4) = as_type<uint2>(half4(v1 - float4(h1)));
                    } else {
                        *(threadgroup uint2 *)bdst[b] = uint2(0u);
                        *(threadgroup uint2 *)(bdst[b] + 4) = uint2(0u);
                        *(threadgroup uint2 *)rdst[b] = uint2(0u);
                        *(threadgroup uint2 *)(rdst[b] + 4) = uint2(0u);
                    }
                } else {
                    qwen4_nax_stage8(bdst[b], xr[b] ? xr[b] + kb * NK : (device const XT *)0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                auto mB = tB.slice(0, 0);
                auto mAg = tA_g.slice(0, 0);
                auto mAu = tA_u.slice(0, 0);
                mm.run(mB, mAg, cG);
                mm.run(mB, mAu, cU);
                if constexpr (COMP) {
                    auto mBr = tBr.slice(0, 0);
                    mm.run(mBr, mAg, cG);
                    mm.run(mBr, mAu, cU);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#pragma unroll
        for (uint16_t i = 0; i < cG.get_capacity(); ++i) { if (cG.is_valid_element(i)) cG[i] = qwen4_silu(cG[i]) * cU[i]; }
        {
            auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
            auto mC = tC.slice(0, 0);
            cG.store(mC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = sgitg; j < n_tile; j += 4) {
            const int pair = list[t0 + j];
            const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
            device float *out = mid + ((uint64_t)t * args.n_out + slot) * args.out_rows + row0;
            device half *outh = nullptr, *outr = nullptr;
            for (uint i = tiisg; i < NR0 && row0 + i < args.out_rows; i += 32) {
                const float v = Cs[j * NR0 + i];
                out[i] = v;
                if constexpr (is_same<XT, half>::value || COMP) {
                    outh = midh + ((uint64_t)t * args.n_out + slot) * args.out_rows + row0;
                    if constexpr (COMP) outr = midr + ((uint64_t)t * args.n_out + slot) * args.out_rows + row0;
                }
                if (outh) {
                    const half h = (half)v;
                    outh[i] = h;
                    if constexpr (COMP) outr[i] = (half)(v - (float)h);
                }
            }
        }
    }
}

template <int NR1, typename XT, bool COMP>
kernel void kernel_qwen4_moe_mm_down_nax_t(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *down_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const XT      *midv,       /* [T][n_out][in_dim] */
        device float         *part,
        device const half    *midr,       /* [T][n_out][in_dim] residual of midh (COMP) */
        threadgroup char     *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NK = 32;
    constexpr int NB = NR1 * 4 / 128;
    uint rb, tile0;
    if (args.expert_major) { const uint n_rb = (args.out_rows + NR0 - 1u) / NR0; rb = tgpig.x % n_rb; tile0 = tgpig.x / n_rb; }
    else { rb = tgpig.x; tile0 = tgpig.z; }
    const uint e = qwen4_moe_mm_expert(args, tgpig.y);
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    if (!count) return;
    /* tails: with tail_base 64 the 64-token tiles keep the full tiles and the
     * 32-token kernel takes a remainder of at most 32 tokens */
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 32u ? 32u : 64u;
        if ((uint)NR1 < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != (uint)NR1) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < (uint)NR1) {
            work_count = count - remainder;
        }
    }
    threadgroup half *As = (threadgroup half *)shmem;                 /* [64][32] */
    threadgroup XT *Bs = (threadgroup XT *)(shmem + 4096);            /* [NR1][32] */
    threadgroup half *Br = (threadgroup half *)(shmem + 4096 + NR1 * 64); /* [NR1][32] residual (COMP) */
    threadgroup float *Cs = (threadgroup float *)shmem;               /* [NR1 tok][64 row] after the K loop */
    device const char *dbase = qwen4_expert_base(down_base, e, args.expert_bytes);
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * NR0;
    const uint nk = args.in_dim / NK;
    const uint type = qwen4_moe_weight_type ? qwen4_moe_weight_type : args.weight_type;
    auto tA = tensor(As, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));   /* left operand: k contiguous, one token per column */
    auto tBr = tensor(Br, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    const uint ar = tid / 2, aq = tid % 2;
    for (uint tile = tile0; tile * NR1 < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * NR1;
        const uint n_tile = min((uint)NR1, work_count - tile * NR1);
        auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
#pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0.0f; }
        device const XT *mr[NB];
        device const half *rr[NB];
        threadgroup XT *bdst[NB];
        threadgroup half *rdst[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
            const uint item = (uint)tid + (uint)b * 128u, tok = item / 4u, kq = item % 4u;
            const int pair = tok < n_tile ? list[t0 + tok] : -1;
            mr[b] = pair >= 0 ? midv + ((uint64_t)((uint)pair / args.n_slots) * args.n_out + (uint)pair % args.n_slots) * args.in_dim + kq * 8u
                              : (device const XT *)0;
            rr[b] = pair >= 0 ? midr + ((uint64_t)((uint)pair / args.n_slots) * args.n_out + (uint)pair % args.n_slots) * args.in_dim + kq * 8u
                              : (device const half *)0;
            bdst[b] = Bs + tok * NK + kq * 8u;
            if constexpr (COMP) rdst[b] = Br + tok * NK + kq * 8u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool a_row = row0 + ar < args.out_rows;
        device const char *drow = dbase + (uint64_t)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        qwen4_raw16 rd = qwen4_load_raw16(drow, 0, aq * 2, type);
        for (uint kb = 0; kb < nk; kb++) {
            {
                threadgroup half *dd = As + ar * NK + aq * 16;
                if (a_row) qwen4_dequant_raw16(rd, kb, aq * 2, type, dd);
                else for (uint i = 0; i < 16; i++) dd[i] = 0.0h;
                if (kb + 1 < nk) rd = qwen4_load_raw16(drow, kb + 1, aq * 2, type);
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if constexpr (COMP) {
                    /* stage the half operand and its residual separately */
                    *(threadgroup uint4 *)bdst[b] = mr[b] ? *(device const uint4 *)(mr[b] + kb * NK) : uint4(0u);
                    *(threadgroup uint4 *)rdst[b] = rr[b] ? *(device const uint4 *)(rr[b] + kb * NK) : uint4(0u);
                } else {
                    qwen4_nax_stage8(bdst[b], mr[b] ? mr[b] + kb * NK : (device const XT *)0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                auto mB = tB.slice(0, 0);
                auto mA = tA.slice(0, 0);
                mm.run(mB, mA, cT);
                if constexpr (COMP) {
                    auto mBr = tBr.slice(0, 0);
                    mm.run(mBr, mA, cT);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        {
            auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
            auto mC = tC.slice(0, 0);
            cT.store(mC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = sgitg; j < n_tile; j += 4) {
            const int pair = list[t0 + j];
            const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
            device float *out = part + ((uint64_t)t * args.n_out + slot) * args.out_rows + row0;
            for (uint i = tiisg; i < NR0 && row0 + i < args.out_rows; i += 32) out[i] = Cs[j * NR0 + i];
        }
    }
}

#define QWEN4_NAX_MID_SIG_HALF constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device half *, device half *, threadgroup char *, uint3, ushort, ushort, ushort
#define QWEN4_NAX_MID_SIG_FLOAT constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, device half *, device half *, threadgroup char *, uint3, ushort, ushort, ushort
#define QWEN4_NAX_DOWN_SIG_HALF constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device const half *, threadgroup char *, uint3, ushort, ushort, ushort
#define QWEN4_NAX_DOWN_SIG_FLOAT constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, device const half *, threadgroup char *, uint3, ushort, ushort, ushort
template [[host_name("kernel_qwen4_moe_mm_mid_nax")]] kernel void kernel_qwen4_moe_mm_mid_nax_t<32, half, false>(QWEN4_NAX_MID_SIG_HALF);
template [[host_name("kernel_qwen4_moe_mm_mid_nax64")]] kernel void kernel_qwen4_moe_mm_mid_nax_t<64, half, false>(QWEN4_NAX_MID_SIG_HALF);
template [[host_name("kernel_qwen4_moe_mm_down_nax")]] kernel void kernel_qwen4_moe_mm_down_nax_t<32, half, false>(QWEN4_NAX_DOWN_SIG_HALF);
template [[host_name("kernel_qwen4_moe_mm_down_nax64")]] kernel void kernel_qwen4_moe_mm_down_nax_t<64, half, false>(QWEN4_NAX_DOWN_SIG_HALF);
template [[host_name("kernel_qwen4_moe_mm_mid_naxf")]] kernel void kernel_qwen4_moe_mm_mid_nax_t<32, float, false>(QWEN4_NAX_MID_SIG_FLOAT);
template [[host_name("kernel_qwen4_moe_mm_mid_naxf64")]] kernel void kernel_qwen4_moe_mm_mid_nax_t<64, float, false>(QWEN4_NAX_MID_SIG_FLOAT);
template [[host_name("kernel_qwen4_moe_mm_down_naxf")]] kernel void kernel_qwen4_moe_mm_down_nax_t<32, float, false>(QWEN4_NAX_DOWN_SIG_FLOAT);
template [[host_name("kernel_qwen4_moe_mm_down_naxf64")]] kernel void kernel_qwen4_moe_mm_down_nax_t<64, float, false>(QWEN4_NAX_DOWN_SIG_FLOAT);
template [[host_name("kernel_qwen4_moe_mm_mid_naxc")]] kernel void kernel_qwen4_moe_mm_mid_nax_t<32, float, true>(QWEN4_NAX_MID_SIG_FLOAT);
template [[host_name("kernel_qwen4_moe_mm_mid_naxc64")]] kernel void kernel_qwen4_moe_mm_mid_nax_t<64, float, true>(QWEN4_NAX_MID_SIG_FLOAT);
template [[host_name("kernel_qwen4_moe_mm_down_naxc")]] kernel void kernel_qwen4_moe_mm_down_nax_t<32, half, true>(QWEN4_NAX_DOWN_SIG_HALF);
template [[host_name("kernel_qwen4_moe_mm_down_naxc64")]] kernel void kernel_qwen4_moe_mm_down_nax_t<64, half, true>(QWEN4_NAX_DOWN_SIG_HALF);
#undef QWEN4_NAX_MID_SIG_HALF
#undef QWEN4_NAX_MID_SIG_FLOAT
#undef QWEN4_NAX_DOWN_SIG_HALF
#undef QWEN4_NAX_DOWN_SIG_FLOAT
#endif /* DS4_METAL_HAS_TENSOR */
/* --- prefill: dense tiled GEMM for f32/f16/q8_0 weights ----------------- */

struct ds4_metal_args_qwen4_dense_mm {
    uint32_t n_tokens;
    uint32_t in_dim;
    uint32_t out_rows;
    uint32_t weight_type;   /* 0 f32, 1 f16, 8 q8_0 */
    uint32_t row_bytes;
    /* Number of k-splits.  One (or zero) writes straight to out; more makes
     * each grid slice cover a slice of k and write its own partial plane,
     * which kernel_qwen4_dense_mm_reduce then sums. */
    uint32_t n_split;
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
    /* A narrow projection gives this kernel only a handful of threadgroups,
     * each walking every k-block in turn; splitting k spreads that walk over
     * the machine instead. */
    const uint nsplit = args.n_split ? args.n_split : 1u;
    const uint kb_lo = (uint)(((uint64_t)tgpig.z * nk) / nsplit);
    const uint kb_hi = (uint)(((uint64_t)(tgpig.z + 1u) * nk) / nsplit);
    for (uint kb = kb_lo; kb < kb_hi; kb++) {
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
        if (row < args.out_rows && t < args.n_tokens) {
            out[(uint64_t)tgpig.z * args.n_tokens * args.out_rows +
                (uint64_t)t * args.out_rows + row] = Cs[sg][j][el];
        }
    }
}

/* Sum the k-split planes written above. */
kernel void kernel_qwen4_dense_mm_reduce(
        constant ds4_metal_args_qwen4_dense_mm & args,
        device const float *partials,
        device float       *out,
        uint gid [[thread_position_in_grid]]) {
    const uint n = args.n_tokens * args.out_rows;
    if (gid >= n) return;
    const uint nsplit = args.n_split ? args.n_split : 1u;
    float acc = 0.0f;
    for (uint s = 0; s < nsplit; s++) acc += partials[(uint64_t)s * n + gid];
    out[gid] = acc;
}

/* --- decode batch: Q8 GEMM on fp32 simdgroup matrices ------------------- */

#define QWEN4_BMM_ROWS 32   /* weight rows per simdgroup */
#define QWEN4_BMM_K 32      /* k per staging step: one q8_0 block per lane */

/* out[t][r] = w[r] . x[t] for a decode batch of 8 or 16 rows.  Every
 * simdgroup owns 32 weight rows and all the tokens: each lane stages one
 * q8_0 block of its row (the matvec's d * q values, as floats) into the
 * simdgroup's own threadgroup memory, then the rows multiply on fp32
 * simdgroup matrices against token tiles read straight from x.  The weights
 * are read once for the batch and x once per 32 rows; simdgroups never wait
 * on each other.  Grid (rows/128, 1, n_split): a k-split writes its plane of
 * partials for kernel_qwen4_dense_mm_reduce.  The sums are fp32 in another
 * order than the matvec's. */
template <uint NTT>   /* token tiles of 8: 1 or 2 */
kernel void kernel_qwen4_batch_mm_q8(
        constant ds4_metal_args_qwen4_dense_mm & args,
        device const char  *w,          /* [out_rows] q8_0 rows */
        device const float *x,          /* [T][in_dim] */
        device float       *out,        /* [n_split][T][out_rows] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (tgpig.x * 4u + sgitg) * QWEN4_BMM_ROWS;
    if (row0 >= args.out_rows) return;
    threadgroup float As[4][QWEN4_BMM_ROWS * QWEN4_BMM_K];
    threadgroup float *A = As[sgitg];
    /* every matrix index below is a compile-time constant: a dynamically
     * indexed simdgroup matrix leaves the registers */
    simdgroup_float8x8 C[4][NTT];
#pragma unroll
    for (uint i = 0; i < 4; i++) {
#pragma unroll
        for (uint j = 0; j < NTT; j++) C[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    device const char *row = w + (uint64_t)(row0 + tiisg) * args.row_bytes;
    const uint nk = args.in_dim / QWEN4_BMM_K;
    const uint nsplit = args.n_split ? args.n_split : 1u;
    const uint kb_lo = (uint)(((uint64_t)tgpig.z * nk) / nsplit);
    const uint kb_hi = (uint)(((uint64_t)(tgpig.z + 1u) * nk) / nsplit);
    for (uint kb = kb_lo; kb < kb_hi; kb++) {
        {
            device const char *blk = row + (uint64_t)kb * 34u;
            const float d = (float)(*(device const half *)blk);
            device const ushort *qs16 = (device const ushort *)(blk + 2);
            threadgroup float *dst = A + tiisg * QWEN4_BMM_K;
#pragma unroll
            for (uint i = 0; i < 16; i++) {
                const ushort u = qs16[i];
                dst[2 * i] = ((float)(int8_t)(u & 0xFFu)) * d;
                dst[2 * i + 1] = ((float)(int8_t)(u >> 8)) * d;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint ks = 0; ks < 4; ks++) {
            simdgroup_float8x8 b[NTT];
#pragma unroll
            for (uint j = 0; j < NTT; j++) {
                simdgroup_load(b[j], x + (uint64_t)(j * 8) * args.in_dim + kb * QWEN4_BMM_K + ks * 8, args.in_dim, 0, true);
            }
#pragma unroll
            for (uint i = 0; i < 4; i++) {
                simdgroup_float8x8 a;
                simdgroup_load(a, A + (i * 8) * QWEN4_BMM_K + ks * 8, QWEN4_BMM_K, 0, false);
#pragma unroll
                for (uint j = 0; j < NTT; j++) simdgroup_multiply_accumulate(C[i][j], a, b[j], C[i][j]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *plane = out + (uint64_t)tgpig.z * args.n_tokens * args.out_rows;
#pragma unroll
    for (uint i = 0; i < 4; i++) {
#pragma unroll
        for (uint j = 0; j < NTT; j++) {
            simdgroup_store(C[i][j], plane + (uint64_t)(j * 8) * args.out_rows + row0 + i * 8, args.out_rows, 0, true);
        }
    }
}

#define QWEN4_BATCH_MM_INSTANCE(NTT_) \
template [[host_name("kernel_qwen4_batch_mm_q8_t" #NTT_)]] \
kernel void kernel_qwen4_batch_mm_q8<NTT_>(constant ds4_metal_args_qwen4_dense_mm &, device const char *, \
        device const float *, device float *, uint3, ushort, ushort);
QWEN4_BATCH_MM_INSTANCE(1)
QWEN4_BATCH_MM_INSTANCE(2)

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

/* Separate embedding/hidden dispatches keep every row in the SIMD batch
 * active. Each walks only its E-wide half, with the original mul_mv_ext
 * float4 dots and shuffle tree; weight/concat strides remain full-width. */
template<short r1ptg, bool emb>
kernel void kernel_qwen4_mtp_project_q8(
        constant ds4_metal_args_mul_mv_ext & args,
        device const char *src0,
        device const char *src1,
        device char *dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;
    const short chpt = 4, chpb = 8;
    const short nypsg = 32 / nxpsg;
    const short tx = tiisg % nxpsg, ty = tiisg / nxpsg;
    const int i01 = tgpig.x * (nypsg * NSG) + nypsg * sgitg + ty;
    const int E = args.ne00 / 2;
    const int k0 = emb ? 0 : E;
    int out_row[r1ptg];
    device const float4 *y4[r1ptg];
    for (short ir1 = 0; ir1 < r1ptg; ir1++) {
        out_row[ir1] = emb ? ir1 * 5 : (int)tgpig.y * 5 + 1 + ir1;
        y4[ir1] = (device const float4 *)(src1 + (uint64_t)out_row[ir1] * args.nb11 +
                                         (uint64_t)k0 * sizeof(float)) + tx;
    }
    device const block_q8_0 *wrow = i01 < args.ne01 ?
        (device const block_q8_0 *)(src0 + (uint64_t)i01 * args.nb01) :
        (device const block_q8_0 *)src0;
    device const block_q8_0 *xq = wrow + k0 / 32 + tx / chpb;
    float sumf[r1ptg] = { [0 ... r1ptg - 1] = 0.0f };
    short cch = tx % chpb;
    for (int ich = tx; 4 * ich < E; ich += chpt * nxpsg) {
        float4 lx[chpt];
#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ch++) {
            dequantize_q8_0_t4(xq, cch, lx[ch]);
            cch += nxpsg;
            if (cch >= chpb) {
                xq += cch / chpb;
                cch %= chpb;
            }
        }
#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ch++) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ir1++) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch * nxpsg]);
            }
        }
#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ir1++) y4[ir1] += chpt * nxpsg;
    }
    for (short ir1 = 0; ir1 < r1ptg; ir1++) {
        if (nxpsg >= 32) sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        if (nxpsg >= 16) sumf[ir1] += simd_shuffle_down(sumf[ir1], 8);
        if (nxpsg >= 8) sumf[ir1] += simd_shuffle_down(sumf[ir1], 4);
        if (nxpsg >= 4) sumf[ir1] += simd_shuffle_down(sumf[ir1], 2);
        if (nxpsg >= 2) sumf[ir1] += simd_shuffle_down(sumf[ir1], 1);
    }
    /* The zero half contributes nothing for finite scales. Any Inf/NaN
     * scale makes its original dot NaN, regardless of the packed quants.
     * Read only those scale words, once per row across its eight lanes. */
    uint nonfinite = 0u;
    device const block_q8_0 *zero_half = wrow + (emb ? E / 32 : 0);
    for (int b = tx; b < E / 32; b += nxpsg)
        nonfinite |= (as_type<ushort>(zero_half[b].d) & 0x7c00u) == 0x7c00u;
    if (nxpsg >= 32) nonfinite |= simd_shuffle_down(nonfinite, 16);
    if (nxpsg >= 16) nonfinite |= simd_shuffle_down(nonfinite, 8);
    if (nxpsg >= 8) nonfinite |= simd_shuffle_down(nonfinite, 4);
    if (nxpsg >= 4) nonfinite |= simd_shuffle_down(nonfinite, 2);
    if (nxpsg >= 2) nonfinite |= simd_shuffle_down(nonfinite, 1);
    if (tx == 0 && i01 < args.ne01) {
        for (short ir1 = 0; ir1 < r1ptg; ir1++) {
            device float *out = (device float *)dst + (uint64_t)out_row[ir1] * args.ne0;
            out[i01] = nonfinite ? as_type<float>(0x7fc00000u) : sumf[ir1];
        }
    }
}
template [[host_name("kernel_qwen4_mtp_project_q8_emb_1")]]
kernel void kernel_qwen4_mtp_project_q8<1, true>(constant ds4_metal_args_mul_mv_ext &, device const char *,
    device const char *, device char *, uint3, ushort, ushort);
template [[host_name("kernel_qwen4_mtp_project_q8_emb_2")]]
kernel void kernel_qwen4_mtp_project_q8<2, true>(constant ds4_metal_args_mul_mv_ext &, device const char *,
    device const char *, device char *, uint3, ushort, ushort);
template [[host_name("kernel_qwen4_mtp_project_q8_emb_3")]]
kernel void kernel_qwen4_mtp_project_q8<3, true>(constant ds4_metal_args_mul_mv_ext &, device const char *,
    device const char *, device char *, uint3, ushort, ushort);
template [[host_name("kernel_qwen4_mtp_project_q8_hidden_4")]]
kernel void kernel_qwen4_mtp_project_q8<4, false>(constant ds4_metal_args_mul_mv_ext &, device const char *,
    device const char *, device char *, uint3, ushort, ushort);

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
    uint32_t snap2_tok;    /* second snapshot point for 3-row MTP verifies */
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
        device float       *snap2_state,
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
            if (tok == args.snap2_tok) {
                for (uint t = 0; t + 1 < K; t++) snap2_state[t * C + c] = state[t * C + c];
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

/* The predictor needs a token ID, not a CPU copy of the entire vocabulary.
 * First reduce independent 4096-value chunks; then merge their winners.
 * The -1e30 initial score and index-zero fallback match sample_argmax. */
struct qwen4_argmax_args { uint n, finish; };
kernel void kernel_qwen4_argmax(
        constant qwen4_argmax_args &args,
        device const float *logits,
        device uint2 *partials,
        device int *out_idx,
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint begin = args.finish ? 0u : group * 4096u;
    const uint end = args.finish ? args.n : min(begin + 4096u, args.n);
    float best = -1.0e30f;
    uint index = 0;
    for (uint i = begin + tid; i < end; i += 256u) {
        const uint2 p = args.finish ? partials[i] : uint2(as_type<uint>(logits[i]), i);
        if ((p.x & 0x7fffffffu) > 0x7f800000u) continue;
        const float v = as_type<float>(p.x);
        if (v > best || (v == best && p.y < index)) { best = v; index = p.y; }
    }
    float top = simd_max(best);
    uint winner = simd_min(best == top ? index : 0xffffffffu);
    threadgroup float scores[8];
    threadgroup uint indices[8];
    if (!lane) { scores[sg] = top; indices[sg] = winner; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sg) {
        best = lane < 8u ? scores[lane] : -1.0e30f;
        index = lane < 8u ? indices[lane] : 0u;
        top = simd_max(best);
        winner = simd_min(best == top ? index : 0xffffffffu);
        if (!lane) {
            if (args.finish) out_idx[0] = (int)winner;
            else partials[group] = uint2(as_type<uint>(top), winner);
        }
    }
}

/* Read the old residual and injection buffer; write separate next buffers.
 * Each normalization chunk writes only its own residual slice. */
template <typename W>
kernel void kernel_qwen4_hc_combine_norm(
        constant ds4_metal_args_qwen4_hc_norm & args,
        device const float *R,          /* [T][hc*E] */
        device const float *gamma,      /* [hc*E] */
        device const char  *w_inject,   /* [n_inject][hc*E] */
        device float       *xn,         /* [T][hc*E] */
        device float       *inj_part,   /* [T][hc*chunks][n_inject] */
        device float *next_R, device const float *blk, device const float *old_inj,
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
    const float weight = qwen4_hc_inject_weight(old_inj, args.n_hc, s);
    float ss = 0.0f;
    for (uint i = tid; i < E; i += nth) { float v = r[i] + weight * blk[i]; ss += v * v; }
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
        const float combined = r[i] + weight * blk[i];
        next_R[s * E + i] = combined;
        const float v = combined * inv * g[i];
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

#define QWEN4_HC_COMBINE_NORM_INSTANCE(SUFFIX, W) \
template [[host_name("kernel_qwen4_hc_combine_norm_" #SUFFIX)]] \
kernel void kernel_qwen4_hc_combine_norm<W>(constant ds4_metal_args_qwen4_hc_norm &, device const float *, \
        device const float *, device const char *, device float *, device float *, device float *, device const float *, device const float *, uint3, ushort, ushort3, ushort, ushort);
QWEN4_HC_COMBINE_NORM_INSTANCE(f16, qwen4_w_f16)

/* Disjoint output grids retain the standalone Q8 reduction trees. */
kernel void kernel_qwen4_q8_concat(
    constant ds4_metal_args_mul_mv &a, constant ds4_metal_args_mul_mv &b,
    device const char *wa, device const char *wb, device const char *x,
    device char *oa, device char *ob, threadgroup char *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint first = (a.ne01 + 1) / 2;
    if (group.x < first) kernel_mul_mv_q8_0_f32_impl<2, constant ds4_metal_args_mul_mv &>(a, wa, x, oa, shared, group, lane, sg);
    else { group.x -= first; kernel_mul_mv_q8_0_f32_impl<2, constant ds4_metal_args_mul_mv &>(b, wb, x, ob, shared, group, lane, sg); }
}

/* Eight physical SIMD groups own the eight chunks, but each chunk retains
 * the original four logical SIMD partials and ascending final reduction.
 * RMS uses exactly the original first 128 threads once per stream/token. */
kernel void kernel_qwen4_hc_norm_parallel_f16(
        constant ds4_metal_args_qwen4_hc_norm &args,
        device const float *R,
        device const float *gamma,
        device const half *inject,
        device float *xn,
        device float *inj_part,
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint stream = group.x, token = group.y, E = args.n_embd;
    if (stream >= args.n_hc || token >= args.n_tokens) return;
    const uint dim = E * args.n_hc, per = (E + 7u) / 8u;
    device const float *r = R + ((uint64_t)token * args.n_hc + stream) * E;
    device const float *g = gamma + stream * E;
    device float *o = xn + ((uint64_t)token * args.n_hc + stream) * E;
    threadgroup float red[4];
    if (tid < 128u) {
        float ss = 0.0f;
        for (uint i = tid; i < E; i += 128u) ss += r[i] * r[i];
        ss = simd_sum(ss);
        if (lane == 0u) red[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint q = 0; q < 4u; q++) total += red[q];
    const float inv = rsqrt(total / (float)E + args.eps);
    const uint begin = sg * per, end = min(E, begin + per);
    float combined[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint logical_sg = 0; logical_sg < 4u; logical_sg++) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = begin + logical_sg * 32u + lane; i < end; i += 128u) {
            const float value = r[i] * inv * g[i];
            o[i] = value;
            for (uint j = 0; j < 4u; j++) {
                if (j < args.n_inject)
                    acc[j] += (float)inject[(uint64_t)j * dim + stream * E + i] * value;
            }
        }
        for (uint j = 0; j < 4u; j++) {
            if (j >= args.n_inject) break;
            combined[j] += simd_sum(acc[j]);
        }
    }
    if (lane == 0u) for (uint j = 0; j < args.n_inject; j++)
        inj_part[((uint64_t)token * args.n_hc * 8u + stream * 8u + sg) * args.n_inject + j] = combined[j];
}

/* Eight independently accumulated value rows per SIMD group. This keeps
 * the same float4 dot and SIMD reductions as the original R4 recurrence. */
kernel void kernel_qwen4_gdn_scan_r8(
        constant ds4_metal_args_qwen4_gdn_scan & args,
        device const float *qkv,
        device const float *ga,
        device const float *gb,
        device float       *state,
        device float       *out,
        device float       *snap_state,
        device float       *snap2_state,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint R = 8;
    const uint dv0 = (tgpig.x * (ntg.x / 32) + sgitg) * R;
    const uint h = tgpig.y;
    if (dv0 >= args.head_dim || h >= args.n_v_head) return;
    const uint D = 128, dk0 = tiisg * 4;
    const uint kh = h % args.n_k_head;
    const uint conv_dim = 2 * args.n_k_head * D + args.n_v_head * D;

    float4 s[R];
    device float *srow = state + ((uint64_t)h * D + dv0) * D + dk0;
    device float *snaprow = snap_state + ((uint64_t)h * D + dv0) * D + dk0;
    device float *snap2row = snap2_state + ((uint64_t)h * D + dv0) * D + dk0;
    for (uint r = 0; r < R; r++) s[r] = *(device const float4 *)(srow + r * D);

    for (uint tok = 0; tok < args.n_tokens; tok++) {
        device const float *base = qkv + (uint64_t)tok * conv_dim;
        const float4 q = *(device const float4 *)(base + kh * D + dk0);
        const float4 k = *(device const float4 *)(base + (args.n_k_head + kh) * D + dk0);
        float v[R];
        for (uint r = 0; r < R; r++) v[r] = base[2 * args.n_k_head * D + h * D + dv0 + r];
        const float g = ga[(uint64_t)tok * args.n_v_head + h];
        const float beta = gb[(uint64_t)tok * args.n_v_head + h];
        float u[R], o[R];
        for (uint r = 0; r < R; r++) { s[r] *= g; u[r] = dot(s[r], k); }
        for (uint r = 0; r < R; r++) u[r] = simd_sum(u[r]);
        for (uint r = 0; r < R; r++) { s[r] += k * ((v[r] - u[r]) * beta); o[r] = dot(s[r], q); }
        for (uint r = 0; r < R; r++) o[r] = simd_sum(o[r]);
        if (tiisg == 0) {
            for (uint r = 0; r < R; r++) out[((uint64_t)tok * args.n_v_head + h) * D + dv0 + r] = o[r];
        }
        if (tok == args.snap_tok) {
            for (uint r = 0; r < R; r++) *(device float4 *)(snaprow + r * D) = s[r];
        }
        if (tok == args.snap2_tok) {
            for (uint r = 0; r < R; r++) *(device float4 *)(snap2row + r * D) = s[r];
        }
    }
    for (uint r = 0; r < R; r++) *(device float4 *)(srow + r * D) = s[r];
}

/* Experimental gate/up-only global fragment pack. One active expert batch
 * owns this workspace until all its mid dispatches complete. Layout is
 * [local expert][row32][K32][K8 sub][row8 group][row][K], contiguous 8x8.
 * The Q4 helper is the frozen K32 staging helper, including FP32-to-half. */
kernel void kernel_qwen4_gain5_pack_q4_expert_fragments(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const char *gate, device const char *up,
        device const int32_t *counts,
        device half *packed_gate, device half *packed_up,
        constant uint2 &range,
        uint3 grid [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint e = qwen4_moe_mm_expert(args, range.x + grid.z);
    if (grid.z >= range.y || e >= args.n_expert || !counts[e]) return;
    threadgroup half ag[1024], au[1024];
    const uint r = tid / 4u, q = tid % 4u;
    const uint row = grid.y * 32u + r, k32 = grid.x;
    const uint at = (q * 4u + r / 8u) * 64u + (r % 8u) * 8u;
    if (row < args.out_rows) {
        device const char *gb = qwen4_expert_base(gate, e, args.expert_bytes) + (uint64_t)row * args.row_bytes;
        device const char *ub = qwen4_expert_base(up, e, args.expert_bytes) + (uint64_t)row * args.row_bytes;
        const uint sb = k32 / 8u, group = k32 % 8u;
        device const uchar *g = (device const uchar *)(gb + (uint64_t)sb * 144u);
        device const uchar *u = (device const uchar *)(ub + (uint64_t)sb * 144u);
        const uint offset = 16u + (group >> 1u) * 32u + q * 8u;
        qwen4_mm_q4_header8(*(device const uint4 *)g, *(device const uint2 *)(g + offset), group, ag + at);
        qwen4_mm_q4_header8(*(device const uint4 *)u, *(device const uint2 *)(u + offset), group, au + at);
    } else for (uint i = 0; i < 8u; i++) { ag[at + i] = 0.0h; au[at + i] = 0.0h; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row_blocks = (args.out_rows + 31u) / 32u;
    const uint64_t block = ((uint64_t)grid.z * row_blocks + grid.y) * (args.in_dim / 32u) + grid.x;
    device half4 *dg = (device half4 *)(packed_gate + block * 1024u);
    device half4 *du = (device half4 *)(packed_up + block * 1024u);
    for (uint j = 0; j < 2u; j++) {
        dg[tid + j * 128u] = ((threadgroup half4 *)ag)[tid + j * 128u];
        du[tid + j * 128u] = ((threadgroup half4 *)au)[tid + j * 128u];
    }
}

/* Both operands are contiguous device fragments. Ascending K8 MMA is unchanged;
 * no threadgroup staging or loop barrier remains. Only the epilogue uses SIMD
 * storage. Quantized originals are still packed on every invocation. */
kernel void kernel_qwen4_gain5_moe_packed_ab_mid(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const half *gate_base, device const half *up_base,
        device const int32_t *lists, device const int32_t *counts,
        device const half *x, device float *mid, device half *mid_shadow,
        device const uint *offsets, constant uint2 &range,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NT = 8u, KS = 32u, TT = 64u;
    threadgroup float Cs[4u * 2u * 64u];
    static_assert(KS == 32u || KS == 64u, "supported MoE K staging widths");
    static_assert((KS * TT / 128u) % 4u == 0u, "B staging uses four-value loads");
    const uint2 block = qwen4_moe_mm_block(args, tgpig);
    if (tgpig.y >= range.y) return;
    const uint rb = block.x, e = qwen4_moe_mm_expert(args, range.x + tgpig.y);
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    if (!count) return;
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 8u ? 8u : remainder <= 16u ? 16u : remainder <= 32u ? 32u : 64u;
        if (TT < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != TT) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < TT) {
            work_count = count - remainder;
        }
    }
    const uint row_blocks = (args.out_rows + 31u) / 32u;
    const uint64_t tile_base = ((uint64_t)tgpig.y * row_blocks + rb) * (args.in_dim / 32u) * 1024u;
    device const half *gbase = gate_base + tile_base;
    device const half *ubase = up_base + tile_base;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * QWEN4_MM_ROWS;
    const uint nk = args.in_dim / KS;
    for (uint tile = block.y; tile * TT < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * TT;
        const uint n_tile = min((uint)TT, work_count - tile * TT);
        simdgroup_float8x8 Cg[NT], Cu[NT];
        for (uint nt = 0; nt < NT; nt++) {
            Cg[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            Cu[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint kb = 0; kb < nk; kb++) {
            for (uint sub = 0; sub < KS / 8; sub++) {
                simdgroup_half8x8 ag, au, b;
                simdgroup_load(ag, gbase + (uint64_t)kb * 1024u + sub * 256u + sgitg * 64u, 8u, 0, false);
                simdgroup_load(au, ubase + (uint64_t)kb * 1024u + sub * 256u + sgitg * 64u, 8u, 0, false);
                for (uint nt = 0; nt < NT; nt++) {
                    simdgroup_load(b, x + (uint64_t)(offsets[e] + t0 + (nt / 4u) * 32u) * args.in_dim + (kb * KS + sub * 8u) * 32u + (nt % 4u) * 64u, 8u, 0, false);
                    simdgroup_multiply_accumulate(Cg[nt], ag, b, Cg[nt]);
                    simdgroup_multiply_accumulate(Cu[nt], au, b, Cu[nt]);
                }
            }
        }
        for (uint nt = 0; nt < NT; nt++) {
            simdgroup_store(Cg[nt], Cs + ((uint)sgitg * 2u) * 64u, 8, 0, false);
            simdgroup_store(Cu[nt], Cs + ((uint)sgitg * 2u + 1u) * 64u, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = (uint)sgitg * 64u + (tid & 31u); idx < ((uint)sgitg + 1u) * 64u; idx += 32u) {
                const uint sg = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                const uint row = row0 + sg * 8 + r;
                if (row >= args.out_rows) continue;
                if (tok >= n_tile) continue;
                const int pair = list[t0 + tok];
                const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
                const float g = Cs[(sg * 2u) * 64u + el], u = Cs[(sg * 2u + 1u) * 64u + el];
                if constexpr (true) {
                    /* Round the completed FP32 activation, exactly where
                     * the down kernel would round its float mid operand. */
                    const float v = qwen4_silu(g) * u;
                    mid[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = v;
                    mid_shadow[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = (half)v;
                } else {
                    mid[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = qwen4_silu(g) * u;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}


kernel void kernel_qwen4_gain5_packed_pair_map(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const int32_t *lists, device const int32_t *counts,
        device const uint *offsets, device int32_t *tokens,
        uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint expert = group.x;
    const uint count = counts[expert], padded = offsets[expert + 1u] - offsets[expert];
    for (uint i = tid; i < padded; i += 128u)
        tokens[offsets[expert] + i] = i < count ? lists[(uint64_t)expert * args.list_cap + i] : -1;
}

struct qwen4_gain5_packed_mid_gather_args { uint rows, dim, slots, stride; };
kernel void kernel_qwen4_gain5_packed_mid_gather(
        constant qwen4_gain5_packed_mid_gather_args &args,
        device const int32_t *tokens, device const half *x, device half *packed,
        uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[32u * 32u];
    const uint ktiles = args.dim / 32u;
    const uint row0 = (group.x / ktiles) * 32u, k0 = (group.x % ktiles) * 32u;
    const uint t = tid / 4u, k = (tid % 4u) * 8u;
    const int pair = tokens[row0 + t];
    const int token = pair < 0 ? -1 : (pair / (int)args.slots) * (int)args.stride + pair % (int)args.slots;
    const half4 va = token >= 0 ? *(device const half4 *)(x + (uint64_t)token * args.dim + k0 + k) : half4(0.0f);
    const half4 vb = token >= 0 ? *(device const half4 *)(x + (uint64_t)token * args.dim + k0 + k + 4u) : half4(0.0f);
    const uint dst = (k / 8u) * 256u + (t / 8u) * 64u + t % 8u;
    for (uint j=0;j<4u;j++) { tile[dst + j*8u]=(half)va[j]; tile[dst + (j+4u)*8u]=(half)vb[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half *out = packed + (uint64_t)row0 * args.dim + k0 * 32u + tid * 8u;
    *(device half4 *)out = *(threadgroup half4 *)(tile + tid * 8u);
    *(device half4 *)(out + 4u) = *(threadgroup half4 *)(tile + tid * 8u + 4u);
}


/* Predecode one hot MXFP4 expert batch using the exact frozen helper. */
kernel void kernel_qwen4_gain5_pack_mxfp4_expert_fragments(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const char *down, device const int32_t *counts,
        device half *packed_down, constant uint2 &range,
        uint3 grid [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint e=qwen4_moe_mm_expert(args,range.x+grid.z);
    if(grid.z>=range.y || e>=args.n_expert || !counts[e])return;
    threadgroup half values[1024];
    const uint r=tid/4u,q=tid%4u,row=grid.y*32u+r,k32=grid.x;
    const uint at=(q*4u+r/8u)*64u+(r%8u)*8u;
    if(row<args.out_rows) {
        device const char *src=qwen4_expert_base(down,e,args.expert_bytes)+(uint64_t)row*args.row_bytes;
        qwen4_mm_stage8(src,k32,q,39u,values+at);
    } else for(uint j=0;j<8u;j++)values[at+j]=0.0h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row_blocks=(args.out_rows+31u)/32u,nk=args.in_dim/32u;
    device half *dst=packed_down+(((uint64_t)grid.z*row_blocks+grid.y)*nk+k32)*1024u+tid*8u;
    *(device half4 *)dst=*(threadgroup half4 *)(values+tid*8u);
    *(device half4 *)(dst+4u)=*(threadgroup half4 *)(values+tid*8u+4u);
}

kernel void kernel_qwen4_gain5_moe_packed_ab_down(
        constant ds4_metal_args_qwen4_moe_mm &args,
        device const half *down_base, device const int32_t *lists,
        device const int32_t *counts, device const half *midv, device float *part,
        device const uint *offsets, constant uint2 &range,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NT = 8u, KS = 32u, TT = 64u;
    static_assert(KS == 32u || KS == 64u, "supported MoE K staging widths");
    static_assert((KS * TT / 128u) % 4u == 0u, "B staging uses four-value loads");
    const uint2 block = qwen4_moe_mm_block(args, tgpig);
    if(tgpig.y>=range.y)return;
    const uint rb=block.x,e=qwen4_moe_mm_expert(args,range.x+tgpig.y);
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    if (!count) return;
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 8u ? 8u : remainder <= 16u ? 16u : remainder <= 32u ? 32u : 64u;
        if (TT < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != TT) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < TT) {
            work_count = count - remainder;
        }
    }
    threadgroup float Cs[4][64];
    const uint row_blocks=(args.out_rows+31u)/32u;
    device const half *dbase=down_base+((uint64_t)tgpig.y*row_blocks+rb)*(args.in_dim/32u)*1024u;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * QWEN4_MM_ROWS;
    const uint nk = args.in_dim / KS;
    for (uint tile = block.y; tile * TT < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * TT;
        const uint n_tile = min((uint)TT, work_count - tile * TT);
        simdgroup_float8x8 C[NT];
        for (uint nt = 0; nt < NT; nt++) C[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kb = 0; kb < nk; kb++) {
            for (uint sub = 0; sub < KS / 8; sub++) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a,dbase+(uint64_t)kb*1024u+sub*256u+sgitg*64u,8u,0,false);
                for (uint nt = 0; nt < NT; nt++) {
                    simdgroup_load(b, midv+(uint64_t)(offsets[e]+t0+(nt/4u)*32u)*args.in_dim+(kb*KS+sub*8u)*32u+(nt%4u)*64u, 8u, 0, false);
                    simdgroup_multiply_accumulate(C[nt], a, b, C[nt]);
                }
            }
        }
        for (uint nt = 0; nt < NT; nt++) {
            simdgroup_store(C[nt], Cs[sgitg], 8, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = (uint)sgitg * 64u + (tid & 31u); idx < ((uint)sgitg + 1u) * 64u; idx += 32u) {
                const uint sg = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                const uint row = row0 + sg * 8 + r;
                if (tok >= n_tile || row >= args.out_rows) continue;
                const int pair = list[t0 + tok];
                const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
                part[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = Cs[sg][el];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
