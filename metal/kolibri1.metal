// Kolibri-1 kernels: per-head QK RMSNorm + optional NeoX rope + K/V cache
// store, GQA attention over sliding-window or full spans, the sigmoid router
// (top-k on logit + expert bias, weights = sigmoid(logit) unnormalized) and
// the routed/shared MoE sum.  Transients and the K/V caches are f32, which
// keeps the graph comparable to kolibri1_forward_token_cpu on the F32
// fixture; the real artifact differs only in the Q8 weight path.  Full
// layers address KV rows by absolute position; the sliding layers keep a
// ring of window + prefill-chunk rows (cache_rows, 0 = absolute) and store
// at pos % cache_rows, so their caches cost the window, not the context.

struct ds4_metal_args_kolibri_attn_prep {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t cache_rows;  /* 0 = absolute-position rows; else ring modulus */
    uint32_t use_rope;    /* sliding layers rope, full layers do not */
    float    eps;
    float    rope_freq[64];
};

static inline float kolibri_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

static inline float kolibri_silu(float x) {
    return x * kolibri_sigmoid(x);
}

/* Kolibri rope rotates the full head dimension with pairs (i, i + head/2) at
 * base-relative frequencies; no YaRN, no multimodal positions. */
static inline void kolibri_rope_neox(thread float *x, uint n_rot, uint pos, constant float *freq) {
    const uint nh = n_rot / 2;
    for (uint i = 0; i < nh; i++) {
        const float theta = (float)pos * freq[i];
        const float c = cos(theta), s = sin(theta);
        const float x0 = x[i], x1 = x[i + nh];
        x[i] = x0 * c - x1 * s;
        x[i + nh] = x0 * s + x1 * c;
    }
}

/* Rotate pairs of the staged head row in place.  Pair p touches exactly the
 * addresses {p, p + head/2}, and those sets are disjoint across lanes, so
 * each lane can read its two values and write its two results without a
 * second barrier between them.  The caller barriers before and after. */
static inline void kolibri_rope_stage(threadgroup float *staged, uint head_dim, uint pos,
                                      constant float *freq, ushort tiisg) {
    const uint nh = head_dim / 2u;
    for (uint p = tiisg; p < nh; p += 32u) {
        const float theta = (float)pos * freq[p];
        const float c = cos(theta), s = sin(theta);
        const float x0 = staged[p], x1 = staged[p + nh];
        staged[p] = x0 * c - x1 * s;
        staged[p + nh] = x0 * s + x1 * c;
    }
}

/* Per (head slot, token) simdgroup.  Slots 0..H-1 norm and (maybe) rope one
 * query head, slots H..H+Hkv-1 one key head, and the last slot copies V raw
 * into the cache. */
kernel void kernel_kolibri_attn_prep(
        constant ds4_metal_args_kolibri_attn_prep & args,
        device const float *qproj,     /* [T][H*D] */
        device const float *kproj,     /* [T][Hkv*D] */
        device const float *vproj,     /* [T][Hkv*D] */
        device const float *g_q,       /* [D] */
        device const float *g_k,       /* [D] */
        device float       *q_out,     /* [T][H*D] */
        device float       *k_cache,   /* [cache_cap][Hkv*D] */
        device float       *v_cache,   /* [cache_cap][Hkv*D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.x, tok = tgpig.y;
    const uint H = args.n_head, Hkv = args.n_head_kv, D = args.head_dim;
    if (tok >= args.n_tokens || slot > H + Hkv) return;
    const uint pos = args.pos0 + tok;
    const uint cache_row = args.cache_rows ? (pos % args.cache_rows) : pos;
    threadgroup float staged[128];

    if (slot < H) {
        device const float *src = qproj + ((uint64_t)tok * H + slot) * D;
        float ss = 0.0f;
        for (uint i = tiisg * 4u; i < D; i += 128u)
            ss += src[i] * src[i] + src[i+1] * src[i+1] + src[i+2] * src[i+2] + src[i+3] * src[i+3];
        ss = simd_sum(ss);
        const float r = 1.0f / sqrt(ss / (float)D + args.eps);
        for (uint i = tiisg * 4u; i < D; i += 128u) {
            staged[i]   = src[i]   * r * g_q[i];
            staged[i+1] = src[i+1] * r * g_q[i+1];
            staged[i+2] = src[i+2] * r * g_q[i+2];
            staged[i+3] = src[i+3] * r * g_q[i+3];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (args.use_rope) {
            kolibri_rope_stage(staged, D, pos, args.rope_freq, tiisg);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        device float *dst = q_out + ((uint64_t)tok * H + slot) * D;
        for (uint i = tiisg * 4u; i < D; i += 128u) {
            dst[i] = staged[i];
            dst[i+1] = staged[i+1];
            dst[i+2] = staged[i+2];
            dst[i+3] = staged[i+3];
        }
        return;
    }
    if (slot < H + Hkv) {
        const uint h = slot - H;
        device const float *src = kproj + ((uint64_t)tok * Hkv + h) * D;
        float ss = 0.0f;
        for (uint i = tiisg * 4u; i < D; i += 128u)
            ss += src[i] * src[i] + src[i+1] * src[i+1] + src[i+2] * src[i+2] + src[i+3] * src[i+3];
        ss = simd_sum(ss);
        const float r = 1.0f / sqrt(ss / (float)D + args.eps);
        for (uint i = tiisg * 4u; i < D; i += 128u) {
            staged[i]   = src[i]   * r * g_k[i];
            staged[i+1] = src[i+1] * r * g_k[i+1];
            staged[i+2] = src[i+2] * r * g_k[i+2];
            staged[i+3] = src[i+3] * r * g_k[i+3];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (args.use_rope) {
            kolibri_rope_stage(staged, D, pos, args.rope_freq, tiisg);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        device float *dk = k_cache + cache_row * Hkv * D + (uint64_t)h * D;
        device float *dv = v_cache + cache_row * Hkv * D + (uint64_t)h * D;
        device const float *vs = vproj + ((uint64_t)tok * Hkv + h) * D;
        for (uint i = tiisg * 4u; i < D; i += 128u) {
            dk[i] = staged[i];
            dk[i+1] = staged[i+1];
            dk[i+2] = staged[i+2];
            dk[i+3] = staged[i+3];
            dv[i] = vs[i];
            dv[i+1] = vs[i+1];
            dv[i+2] = vs[i+2];
            dv[i+3] = vs[i+3];
        }
        return;
    }
    {
        device const float *src = vproj + (uint64_t)tok * Hkv * D;
        device float *dst = v_cache + cache_row * Hkv * D;
        for (uint i = tiisg * 4u; i < Hkv * D; i += 128u) {
            dst[i]   = src[i];
            dst[i+1] = src[i+1];
            dst[i+2] = src[i+2];
            dst[i+3] = src[i+3];
        }
    }
}

/* --- GQA attention ------------------------------------------------------- */

struct ds4_metal_args_kolibri_attn {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t window;       /* sliding-window rows; 0 = full span 0..pos */
    uint32_t cache_rows;   /* 0 = absolute-position rows; else ring modulus */
    uint32_t n_splits;     /* key ranges per (kv head, token); 1 writes out */
    uint32_t keys_per_split;
    float    scale;
};

#define KOLIBRI_ATTN_NSG 4          /* simdgroups per threadgroup, each owning a slice of the q-head group */

/* Online-softmax GQA decode tile over the row span this token attends.
 * Each simdgroup owns a slice of the kv head's query-head group; a lane owns
 * four consecutive head dims.  One split writes out directly; otherwise each
 * head leaves (m, l, acc) partials for the merge kernel.  Inactive
 * simdgroups (ng = 0) still run the loop with no work so every threadgroup
 * takes the same path. */
static inline void kolibri_attn_decode_tile(
        constant ds4_metal_args_kolibri_attn &args, uint split, uint kvh, uint tok,
        uint n, uint start, uint n_splits, uint keys_per_split, uint part_splits,
        device const float *q, device const float *k_cache, device const float *v_cache,
        device float *out, device float *part,
        ushort sgitg, ushort tiisg) {
    const uint H = args.n_head, Hkv = args.n_head_kv;
    const uint D = args.head_dim;
    const uint group = H / Hkv;
    const uint hps = (group + KOLIBRI_ATTN_NSG - 1) / KOLIBRI_ATTN_NSG;
    const uint hb = (uint)sgitg * hps;
    const uint ng = hb < group ? min(hps, group - hb) : 0u;
    const uint k0 = split * keys_per_split;
    const uint k1 = min(n, k0 + keys_per_split);

    float qv[3][4];
    float m[3], l[3], acc[3][4];
    for (uint g = 0; g < 3u; g++) {
        if (g < ng) {
            const uint h = kvh * group + hb + g;
            device const float *qh = q + ((uint64_t)tok * H + h) * D;
            for (uint j = 0; j < 4u; j++) {
                const uint d = tiisg * 4u + j;
                qv[g][j] = d < D ? qh[d] * args.scale : 0.0f;
            }
        }
        m[g] = -3.0e38f;
        l[g] = 0.0f;
        for (uint j = 0; j < 4u; j++) acc[g][j] = 0.0f;
    }
    for (uint idx = k0; idx < k1; idx++) {
        const uint p = start + idx;
        const uint row = args.cache_rows ? (p % args.cache_rows) : p;
        device const float *kr = k_cache + ((uint64_t)row * Hkv + kvh) * D;
        device const float *vr = v_cache + ((uint64_t)row * Hkv + kvh) * D;
        float kv[4], vv[4];
        for (uint j = 0; j < 4u; j++) {
            const uint d = tiisg * 4u + j;
            kv[j] = d < D ? kr[d] : 0.0f;
            vv[j] = d < D ? vr[d] : 0.0f;
        }
        for (uint g = 0; g < ng; g++) {
            float s = 0.0f;
            for (uint j = 0; j < 4u; j++) s += qv[g][j] * kv[j];
            s = simd_sum(s);
            const float m_new = max(m[g], s);
            const float corr = exp(m[g] - m_new);
            const float w = exp(s - m_new);
            l[g] = l[g] * corr + w;
            for (uint j = 0; j < 4u; j++) acc[g][j] = acc[g][j] * corr + w * vv[j];
            m[g] = m_new;
        }
    }
    if (n_splits == 1) {
        for (uint g = 0; g < ng; g++) {
            const uint h = kvh * group + hb + g;
            device float *dst = out + ((uint64_t)tok * H + h) * D;
            const float inv = l[g] > 0.0f ? 1.0f / l[g] : 0.0f;
            for (uint j = 0; j < 4u; j++) {
                const uint d = tiisg * 4u + j;
                if (d < D) dst[d] = acc[g][j] * inv;
            }
        }
        return;
    }
    for (uint g = 0; g < ng; g++) {
        const uint h = kvh * group + hb + g;
        device float *dst = part + ((((uint64_t)tok * Hkv + kvh) * part_splits + split) * group
                                    + (hb + g)) * (2u + D);
        if (tiisg == 0) { dst[0] = m[g]; dst[1] = l[g]; }
        for (uint j = 0; j < 4u; j++) {
            const uint d = tiisg * 4u + j;
            if (d < D) dst[2u + d] = acc[g][j];
        }
    }
}

kernel void kernel_kolibri_attn_decode(
        constant ds4_metal_args_kolibri_attn & args,
        device const float   *q,          /* [T][H*D] */
        device const float   *k_cache,    /* [cache_cap][Hkv*D] */
        device const float   *v_cache,    /* [cache_cap][Hkv*D] */
        device float         *out,        /* [T][H*D] */
        device float         *part,       /* [T][Hkv][n_splits][group][2+D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint split = tgpig.x;
    const uint kvh = tgpig.y;
    const uint tok = tgpig.z;
    if (split >= args.n_splits || kvh >= args.n_head_kv || tok >= args.n_tokens) return;
    const uint pos = args.pos0 + tok;
    const uint n = args.window ? min(pos + 1u, args.window) : pos + 1u;
    const uint start = args.window ? pos + 1u - n : 0u;
    kolibri_attn_decode_tile(args, split, kvh, tok, n, start, args.n_splits, args.keys_per_split,
                             args.n_splits, q, k_cache, v_cache, out, part, sgitg, tiisg);
}

/* Combine the split partials of one (head, dim-slice).  One threadgroup
 * per 32-dim slice of a head.  The split walk is software-pipelined: the
 * m/l maximum is reduced cooperatively, and the accumulation loop batches
 * four splits' loads into registers before combining them in ascending
 * split order, so every output element sees the same FP operations in the
 * same order as the serial version while the load latencies overlap. */
kernel void kernel_kolibri_attn_merge(
        constant ds4_metal_args_kolibri_attn & args,
        device const float *part,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint H = args.n_head, Hkv = args.n_head_kv;
    const uint D = args.head_dim;
    const uint n_slice = (D + 31u) / 32u;
    const uint h = tgpig.x / n_slice;
    const uint slice = tgpig.x % n_slice;
    const uint tok = tgpig.y;
    if (h >= H || tok >= args.n_tokens) return;
    const uint group = H / Hkv;
    const uint kvh = h / group, g = h % group;
    const uint64_t stride = (uint64_t)group * (2u + D);
    device const float *base = part + (((uint64_t)tok * Hkv + kvh) * args.n_splits * group + g) * (2u + D);
    const uint ns = args.n_splits;
    /* mm via strided loads per lane and a cooperative simd maximum; each
     * lane covers the 16 splits at tiisg + 32*k (up to 512 splits). */
    float mm = -3.0e38f;
    for (uint k = 0; k < 16u; k++) {
        const uint idx = tiisg + 32u * k;
        if (idx < ns) mm = max(mm, base[(uint64_t)idx * stride]);
    }
    mm = simd_max(mm);
    float ll = 0.0f;
    float o[4];
    for (uint j = 0; j < 4u; j++) o[j] = 0.0f;
    const uint d0 = slice * 32u + tiisg * 4u;
    for (uint s = 0; s < ns; s += 4u) {
        /* Batch this group's independent loads before any dependent math. */
        device const float *p0 = base + (uint64_t)s * stride;
        device const float *p1 = p0 + stride;
        device const float *p2 = p0 + 2u * stride;
        device const float *p3 = p0 + 3u * stride;
        const bool h1 = s + 1u < ns, h2 = s + 2u < ns, h3 = s + 3u < ns;
        float4 a0 = d0 + 3u < D ? *(device const float4 *)(p0 + 2u + d0) : float4(0.0f);
        float4 a1 = h1 && d0 + 3u < D ? *(device const float4 *)(p1 + 2u + d0) : float4(0.0f);
        float4 a2 = h2 && d0 + 3u < D ? *(device const float4 *)(p2 + 2u + d0) : float4(0.0f);
        float4 a3 = h3 && d0 + 3u < D ? *(device const float4 *)(p3 + 2u + d0) : float4(0.0f);
        float m0 = p0[0], l0 = p0[1];
        float m1 = h1 ? p1[0] : -3.0e38f, l1 = h1 ? p1[1] : 0.0f;
        float m2 = h2 ? p2[0] : -3.0e38f, l2 = h2 ? p2[1] : 0.0f;
        float m3 = h3 ? p3[0] : -3.0e38f, l3 = h3 ? p3[1] : 0.0f;
        if (s < ns) {
            const float c = l0 > 0.0f ? exp(m0 - mm) : 0.0f;
            ll += l0 * c;
            o[0] += a0.x * c; o[1] += a0.y * c; o[2] += a0.z * c; o[3] += a0.w * c;
        }
        if (h1) {
            const float c = l1 > 0.0f ? exp(m1 - mm) : 0.0f;
            ll += l1 * c;
            o[0] += a1.x * c; o[1] += a1.y * c; o[2] += a1.z * c; o[3] += a1.w * c;
        }
        if (h2) {
            const float c = l2 > 0.0f ? exp(m2 - mm) : 0.0f;
            ll += l2 * c;
            o[0] += a2.x * c; o[1] += a2.y * c; o[2] += a2.z * c; o[3] += a2.w * c;
        }
        if (h3) {
            const float c = l3 > 0.0f ? exp(m3 - mm) : 0.0f;
            ll += l3 * c;
            o[0] += a3.x * c; o[1] += a3.y * c; o[2] += a3.z * c; o[3] += a3.w * c;
        }
    }
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    device float *dst = out + ((uint64_t)tok * H + h) * D;
    if (d0 + 3u < D) {
        *(device float4 *)(dst + d0) = float4(o[0] * inv, o[1] * inv, o[2] * inv, o[3] * inv);
    } else {
        for (uint j = 0; j < 4u; j++) {
            const uint d = d0 + j;
            if (d < D) dst[d] = o[j] * inv;
        }
    }
}

/* --- sigmoid router ------------------------------------------------------ */

struct ds4_metal_args_kolibri_router {
    uint32_t n_tokens;
    uint32_t n_expert;
    uint32_t n_used;
    float    weight_scale;
};

#define KOLIBRI_ROUTER_MAX_EXPERT 512
#define KOLIBRI_ROUTER_MAX_USED 8
#define KOLIBRI_ROUTER_NSG 4
#define KOLIBRI_ROUTER_LANE_MAX 4   /* experts per lane: 32 * 4 simdgroups * 4 = 512 */

/* Top-k on logit + expert_bias with weights = sigmoid(logit) * scale, left
 * unnormalized (norm_topk_prob = false).  Ties keep the lower expert index,
 * matching kolibri1_route on the CPU reference. */
kernel void kernel_kolibri_router_topk(
        constant ds4_metal_args_kolibri_router & args,
        device const float *logits,     /* [T][n_expert] */
        device const float *bias,       /* [n_expert] */
        device int32_t     *selected,   /* [T][n_used] */
        device float       *weights,    /* [T][n_used] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens) return;
    const uint NE = args.n_expert;
    const uint nsg = ntg.x / 32u;
    threadgroup float cand_v[KOLIBRI_ROUTER_NSG * KOLIBRI_ROUTER_MAX_USED];
    threadgroup int   cand_i[KOLIBRI_ROUTER_NSG * KOLIBRI_ROUTER_MAX_USED];
    device const float *lg = logits + (uint64_t)tok * NE;

    const float unset = -3.402823466e+38f;
    float mine[KOLIBRI_ROUTER_LANE_MAX];
    for (uint k = 0; k < KOLIBRI_ROUTER_LANE_MAX; k++) {
        const uint e = (uint)sgitg * 32u + tiisg + 32u * nsg * k;
        mine[k] = e < NE ? lg[e] + bias[e] : unset;
    }

    /* rank the lane-owned scores, one selected expert per round */
    for (uint r = 0; r < args.n_used; r++) {
        float bv = unset;
        int bi = 0x7fffffff;
        for (uint k = 0; k < KOLIBRI_ROUTER_LANE_MAX; k++) {
            if (mine[k] > bv) { bv = mine[k]; bi = (int)((uint)sgitg * 32u + tiisg + 32u * nsg * k); }
        }
        for (uint off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_xor(bv, off);
            const int oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (tiisg == 0) {
            cand_v[sgitg * args.n_used + r] = bv;
            cand_i[sgitg * args.n_used + r] = bi;
        }
        for (uint k = 0; k < KOLIBRI_ROUTER_LANE_MAX; k++) {
            if ((int)((uint)sgitg * 32u + tiisg + 32u * nsg * k) == bi) mine[k] = unset;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const uint n_cand = nsg * args.n_used;
        float cv[KOLIBRI_ROUTER_NSG];
        int ci[KOLIBRI_ROUTER_NSG];
        for (uint j = 0; j < KOLIBRI_ROUTER_NSG; j++) {
            const uint c = tiisg + 32u * j;
            cv[j] = c < n_cand ? cand_v[c] : unset;
            ci[j] = c < n_cand ? cand_i[c] : 0x7fffffff;
        }
        for (uint r = 0; r < args.n_used; r++) {
            float bv = unset;
            int bi = 0x7fffffff;
            for (uint j = 0; j < KOLIBRI_ROUTER_NSG; j++) {
                if (cv[j] > bv || (cv[j] == bv && ci[j] < bi)) { bv = cv[j]; bi = ci[j]; }
            }
            for (uint off = 16u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_xor(bv, off);
                const int oi = simd_shuffle_xor(bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            for (uint j = 0; j < KOLIBRI_ROUTER_NSG; j++) if (ci[j] == bi) cv[j] = unset;
            if (tiisg == 0) {
                selected[(uint64_t)tok * args.n_used + r] = bi;
                weights[(uint64_t)tok * args.n_used + r] =
                    kolibri_sigmoid(lg[(uint)bi]) * args.weight_scale;
            }
        }
    }
}

/* --- routed + shared MoE sum --------------------------------------------- */

struct ds4_metal_args_kolibri_moe_sum {
    uint32_t n_tokens;
    uint32_t n_slots;
    uint32_t dim;
    uint32_t pad0;
};

/* part[t][n_slots] is the shared expert's projection: it joins the residual
 * unweighted, every routed slot through its router weight. */
kernel void kernel_kolibri_moe_sum(
        constant ds4_metal_args_kolibri_moe_sum & args,
        device const float *part,       /* [T][n_slots+1][dim] */
        device const float *weights,    /* [T][n_slots] */
        device float       *out,        /* [T][dim] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint64_t stride = args.n_slots + 1u;
    for (uint d = tgpig.x * ntg.x + tid; d < args.dim; d += ntg.x) {
        float acc = part[((uint64_t)tok * stride + args.n_slots) * args.dim + d];
        for (uint s = 0; s < args.n_slots; s++) {
            acc += weights[(uint64_t)tok * args.n_slots + s] *
                   part[((uint64_t)tok * stride + s) * args.dim + d];
        }
        out[(uint64_t)tok * args.dim + d] = acc;
    }
}

/* Prefill grouped-GEMM variant: the routed slots come from the tiled expert
 * matmuls as [T][n_slots][dim] and the shared expert (dense projections)
 * from its own [T][dim] buffer.  Same element order as kernel_kolibri_moe_sum:
 * the shared row first, then every routed slot ascending. */
kernel void kernel_kolibri_moe_sum_shared(
        constant ds4_metal_args_kolibri_moe_sum & args,
        device const float *part,       /* [T][n_slots][dim] */
        device const float *shared,     /* [T][dim] */
        device const float *weights,    /* [T][n_slots] */
        device float       *out,        /* [T][dim] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    for (uint d = tgpig.x * ntg.x + tid; d < args.dim; d += ntg.x) {
        float acc = shared[(uint64_t)tok * args.dim + d];
        for (uint s = 0; s < args.n_slots; s++) {
            acc += weights[(uint64_t)tok * args.n_slots + s] *
                   part[(((uint64_t)tok * args.n_slots) + s) * args.dim + d];
        }
        out[(uint64_t)tok * args.dim + d] = acc;
    }
}

