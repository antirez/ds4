// MiniMax-M3.1-Flash kernels: per-head norm + partial RoPE, MiniMax Sparse
// Attention (indexer block scores, per-KV-head top-k block selection,
// block-sparse GQA attention), sigmoid router and the MoE combine.  The
// routed expert matmuls reuse the qwen4 MoE kernels with the SwiGLU-OAI
// activation (qwen4_swiglu_oai); this file is appended after qwen4.metal.

#define MM3_MAX_ROT_PAIRS 32

struct ds4_metal_args_mm3_attn_prep {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_idx_head;
    uint32_t pos0;
    uint32_t cache_cap;
    float    eps;
    uint32_t pad0;
    float    rope_freq[MM3_MAX_ROT_PAIRS];
};

// Gemma-folded RMSNorm (the converter stores 1 + w) and NeoX RoPE over the
// first 64 of 128 dims.  The angle is a float32 product, as in vLLM's cos/sin
// cache; precise:: cos/sin keep it exact at million-token positions.  Lane l owns dims 4l..4l+3; the partner of dim d < 32
// is d + 32, which lives four lanes x two = eight lanes up.
static inline void mm3_norm_rope(thread float *v, device const float *g, uint pos,
                                 constant float *freq, float eps, ushort lane) {
    float ss = 0.0f;
    for (uint k = 0; k < 4; k++) ss += v[k] * v[k];
    ss = simd_sum(ss);
    const float r = rsqrt(ss / 128.0f + eps);
    for (uint k = 0; k < 4; k++) v[k] = v[k] * r * g[lane * 4 + k];
    float other[4];
    const ushort partner = lane < 8 ? lane + 8 : lane - 8;
    for (uint k = 0; k < 4; k++) other[k] = simd_shuffle(v[k], lane < 16 ? partner : lane);
    if (lane < 8) {
        for (uint k = 0; k < 4; k++) {
            const float th = (float)pos * freq[lane * 4 + k];
            v[k] = v[k] * precise::cos(th) - other[k] * precise::sin(th);
        }
    } else if (lane < 16) {
        for (uint k = 0; k < 4; k++) {
            const float th = (float)pos * freq[lane * 4 + k - 32];
            v[k] = other[k] * precise::sin(th) + v[k] * precise::cos(th);
        }
    }
}

// One simdgroup per (head slot, token).  Slots: H query heads, Hkv key heads
// (with their values), Hi indexer query heads, then the single indexer key.
// Keys and values go to the f16 caches at pos0 + tok; indexer keys to the f32
// indexer cache, already normed and rotated as the scorer consumes them.
kernel void kernel_mm3_attn_prep(
        constant ds4_metal_args_mm3_attn_prep & args,
        device const float *qp,       // [T][H*128]
        device const float *kp,       // [T][Hkv*128]
        device const float *vp,       // [T][Hkv*128]
        device const float *iqp,      // [T][Hi*128]
        device const float *ikp,      // [T][128]
        device const float *g_q,
        device const float *g_k,
        device const float *g_iq,
        device const float *g_ik,
        device float       *q_out,    // [T][H*128]
        device half        *k_cache,  // [cap][Hkv*128]
        device half        *v_cache,
        device float       *iq_out,   // [T][Hi*128]
        device float       *ik_cache, // [cap][128]
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.x, tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint H = args.n_head, Hkv = args.n_head_kv, Hi = args.n_idx_head;
    const uint pos = args.pos0 + tok;
    float v[4];
    device const float *src;
    device const float *g;
    if (slot < H) { src = qp + ((uint64_t)tok * H + slot) * 128; g = g_q; }
    else if (slot < H + Hkv) { src = kp + ((uint64_t)tok * Hkv + (slot - H)) * 128; g = g_k; }
    else if (slot < H + Hkv + Hi) { src = iqp + ((uint64_t)tok * Hi + (slot - H - Hkv)) * 128; g = g_iq; }
    else { src = ikp + (uint64_t)tok * 128; g = g_ik; }
    for (uint k = 0; k < 4; k++) v[k] = src[lane * 4 + k];
    mm3_norm_rope(v, g, pos, args.rope_freq, args.eps, lane);
    if (slot < H) {
        device float *d = q_out + ((uint64_t)tok * H + slot) * 128;
        for (uint k = 0; k < 4; k++) d[lane * 4 + k] = v[k];
    } else if (slot < H + Hkv) {
        const uint h = slot - H;
        device half *dk = k_cache + ((uint64_t)pos * Hkv + h) * 128;
        device half *dv = v_cache + ((uint64_t)pos * Hkv + h) * 128;
        device const float *vs = vp + ((uint64_t)tok * Hkv + h) * 128;
        for (uint k = 0; k < 4; k++) {
            dk[lane * 4 + k] = (half)v[k];
            dv[lane * 4 + k] = (half)vs[lane * 4 + k];
        }
    } else if (slot < H + Hkv + Hi) {
        device float *d = iq_out + ((uint64_t)tok * Hi + (slot - H - Hkv)) * 128;
        for (uint k = 0; k < 4; k++) d[lane * 4 + k] = v[k];
    } else {
        device float *d = ik_cache + (uint64_t)pos * 128;
        for (uint k = 0; k < 4; k++) d[lane * 4 + k] = v[k];
    }
}

struct ds4_metal_args_mm3_idx {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_idx_head;   // == n_head_kv
    uint32_t block;        // 128
    uint32_t top_k;        // 16
    uint32_t stride;       // score row stride (blocks)
    uint32_t pad0, pad1;
};

// score[t][h][b] = max over the causal keys j of block b of iq[t][h] . ik[j].
// One 128-thread threadgroup per (block, token), one key per thread.
kernel void kernel_mm3_idx_score(
        constant ds4_metal_args_mm3_idx & args,
        device const float *iq,        // [T][Hi*128]
        device const float *ik_cache,  // [cap][128]
        device float       *score,     // [T][Hi][stride]
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint b = tgpig.x, tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint pos = args.pos0 + tok;
    const uint Hi = args.n_idx_head;
    if (b > pos / args.block) {
        if (tid < Hi) score[((uint64_t)tok * Hi + tid) * args.stride + b] = -INFINITY;
        return;
    }
    threadgroup float q[4 * 128];
    threadgroup float red[4][4];
    for (uint i = tid; i < Hi * 128; i += 128) q[i] = iq[(uint64_t)tok * Hi * 128 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint key = b * args.block + tid;
    float s[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    if (key <= pos) {
        device const float4 *kr = (device const float4 *)(ik_cache + (uint64_t)key * 128);
        float acc[4] = { 0, 0, 0, 0 };
        for (uint d = 0; d < 32; d++) {
            const float4 kv = kr[d];
            for (uint h = 0; h < Hi; h++) {
                threadgroup const float4 *qh = (threadgroup const float4 *)(q + h * 128);
                acc[h] += dot(qh[d], kv);
            }
        }
        for (uint h = 0; h < Hi; h++) s[h] = acc[h];
    }
    for (uint h = 0; h < Hi; h++) {
        const float m = simd_max(s[h]);
        if (lane == 0) red[sg][h] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < Hi) {
        const float m = max(max(red[0][tid], red[1][tid]), max(red[2][tid], red[3][tid]));
        score[((uint64_t)tok * Hi + tid) * args.stride + b] = m;
    }
}

// sel[t][h][0..top_k) = the token's own block, then the highest-scoring
// other blocks (lower index first on ties).  Fewer visible blocks than top_k:
// every block, padded with -1.  One 256-thread threadgroup per (head, token);
// chosen scores are overwritten with -inf in place.
kernel void kernel_mm3_idx_select(
        constant ds4_metal_args_mm3_idx & args,
        device float   *score,   // [T][Hi][stride], clobbered
        device int32_t *sel,     // [T][Hi][top_k]
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint h = tgpig.x, tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint pos = args.pos0 + tok;
    const uint nb = pos / args.block + 1;
    const uint K = args.top_k;
    device int32_t *out = sel + ((uint64_t)tok * args.n_idx_head + h) * K;
    if (nb <= K) {
        for (uint i = tid; i < K; i += 256) out[i] = i < nb ? (int32_t)i : -1;
        return;
    }
    device float *sc = score + ((uint64_t)tok * args.n_idx_head + h) * args.stride;
    threadgroup float best_v[8];
    threadgroup int best_i[8];
    if (tid == 0) { out[0] = (int32_t)(nb - 1); sc[nb - 1] = -INFINITY; }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint k = 1; k < K; k++) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (uint b = tid; b < nb; b += 256) {
            const float v = sc[b];
            if (v > bv) { bv = v; bi = (int)b; }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(bv, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { best_v[sg] = bv; best_i[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = best_v[0];
            int i = best_i[0];
            for (uint g = 1; g < 8; g++) {
                if (best_v[g] > v || (best_v[g] == v && best_i[g] < i)) { v = best_v[g]; i = best_i[g]; }
            }
            if (i == 0x7fffffff) i = 0;   // all -inf: cannot happen with nb > K
            out[k] = i;
            sc[i] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

struct ds4_metal_args_mm3_attn {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t pos0;
    uint32_t block;
    uint32_t top_k;
    float    scale;
    uint32_t n_split;      // 1: final output; >1: partials, see kernel_mm3_attn_combine
};

#define MM3_ATTN_CHUNK 32
#define MM3_ATTN_PART 132  // per (token, head, split): acc[128], max, sum

// Block-sparse GQA attention.  One threadgroup per (KV head, token) with one
// simdgroup per query head of the group (16).  The token's block list is the
// indexer selection, or every block while there are at most top_k of them.
// K/V are staged 32 keys at a time; lane i scores key i, then every lane
// folds the chunk into its 4 output dims with an online softmax.  Decode has
// only Hkv threadgroups, so it splits the block list over grid z (split z
// takes blocks z, z + n_split, ...) and kernel_mm3_attn_combine merges.
kernel void kernel_mm3_attn(
        constant ds4_metal_args_mm3_attn & args,
        device const float   *q,        // [T][H*128]
        device const half    *k_cache,  // [cap][Hkv*128]
        device const half    *v_cache,
        device const int32_t *sel,      // [T][Hkv][top_k]
        device float         *out,      // [T][H*128]
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint kvh = tgpig.x, tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint H = args.n_head, Hkv = args.n_head_kv, group = H / Hkv;
    const uint nth = ntg.x;
    const uint pos = args.pos0 + tok;
    const uint nb = pos / args.block + 1;
    const uint n_list = nb <= args.top_k ? nb : args.top_k;
    device const int32_t *list = sel + ((uint64_t)tok * Hkv + kvh) * args.top_k;
    const uint head = kvh * group + sg;

    threadgroup float qs[16 * 128];
    threadgroup half ks[MM3_ATTN_CHUNK * 128];
    threadgroup half vs[MM3_ATTN_CHUNK * 128];
    for (uint i = tid; i < group * 128; i += nth)
        qs[i] = q[((uint64_t)tok * H + kvh * group) * 128 + i] * args.scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, l = 0.0f, acc[4] = { 0, 0, 0, 0 };
    for (uint li = tgpig.z; li < n_list; li += args.n_split) {
        const uint b = nb <= args.top_k ? li : (uint)list[li];
        const uint start = b * args.block;
        const uint end = min(start + args.block, pos + 1);
        for (uint key0 = start; key0 < end; key0 += MM3_ATTN_CHUNK) {
            const uint n = min((uint)MM3_ATTN_CHUNK, end - key0);
            for (uint i = tid; i < n * 128; i += nth) {
                const uint r = i / 128, d = i % 128;
                const uint64_t at = ((uint64_t)(key0 + r) * Hkv + kvh) * 128 + d;
                ks[i] = k_cache[at];
                vs[i] = v_cache[at];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg < group) {
                float s = -INFINITY;
                if (lane < n) {
                    threadgroup const float4 *qh = (threadgroup const float4 *)(qs + sg * 128);
                    threadgroup const half4 *kr = (threadgroup const half4 *)(ks + lane * 128);
                    float a = 0.0f;
                    for (uint d = 0; d < 32; d++) a += dot(qh[d], float4(kr[d]));
                    s = a;
                }
                const float mc = simd_max(s);
                const float mn = max(m, mc);
                const float p = lane < n ? exp(s - mn) : 0.0f;
                const float corr = exp(m - mn);
                l = l * corr + simd_sum(p);
                for (uint k = 0; k < 4; k++) acc[k] *= corr;
                for (uint i = 0; i < n; i++) {
                    const float pi = simd_shuffle(p, (ushort)i);
                    threadgroup const half4 *vr = (threadgroup const half4 *)(vs + i * 128);
                    const float4 vv = float4(vr[lane]);
                    acc[0] += pi * vv.x; acc[1] += pi * vv.y; acc[2] += pi * vv.z; acc[3] += pi * vv.w;
                }
                m = mn;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (sg < group && args.n_split > 1) {
        device float *o = out + (((uint64_t)tok * H + head) * args.n_split + tgpig.z) * MM3_ATTN_PART;
        for (uint k = 0; k < 4; k++) o[lane * 4 + k] = acc[k];
        if (lane == 0) { o[128] = m; o[129] = l; }
    } else if (sg < group) {
        device float *o = out + ((uint64_t)tok * H + head) * 128;
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (uint k = 0; k < 4; k++) o[lane * 4 + k] = acc[k] * inv;
    }
}

// out[t][h] from the n_split online-softmax partials of kernel_mm3_attn.
// One simdgroup per (head, token); empty splits carry max -inf and sum 0.
kernel void kernel_mm3_attn_combine(
        constant ds4_metal_args_mm3_attn & args,
        device const float *part,   // [T][H][n_split][MM3_ATTN_PART]
        device float       *out,    // [T][H*128]
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = tgpig.x, tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    device const float *p = part + ((uint64_t)tok * args.n_head + head) * args.n_split * MM3_ATTN_PART;
    float m = -INFINITY;
    for (uint z = 0; z < args.n_split; z++) m = max(m, p[z * MM3_ATTN_PART + 128]);
    float l = 0.0f, acc[4] = { 0, 0, 0, 0 };
    for (uint z = 0; z < args.n_split; z++) {
        device const float *pz = p + z * MM3_ATTN_PART;
        if (pz[129] == 0.0f) continue;
        const float c = exp(pz[128] - m);
        l += pz[129] * c;
        for (uint k = 0; k < 4; k++) acc[k] += pz[lane * 4 + k] * c;
    }
    device float *o = out + ((uint64_t)tok * args.n_head + head) * 128;
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint k = 0; k < 4; k++) o[lane * 4 + k] = acc[k] * inv;
}

struct ds4_metal_args_mm3_router {
    uint32_t n_tokens;
    uint32_t n_expert;   // 128
    uint32_t n_used;     // 4
    float    scale;      // routed_scaling_factor
};

// Sigmoid scores; the correction bias only ranks experts; the chosen scores
// are renormalized and scaled.  One thread per expert (n_expert <= 256).
kernel void kernel_mm3_router(
        constant ds4_metal_args_mm3_router & args,
        device const float *logits,    // [T][n_expert]
        device const float *bias,      // [n_expert]
        device int32_t     *selected,  // [T][n_used]
        device float       *weights,   // [T][n_used]
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens) return;
    const uint nsg = ntg.x / 32;
    const bool live = tid < args.n_expert;
    const float sc = live ? 1.0f / (1.0f + exp(-logits[(uint64_t)tok * args.n_expert + tid])) : 0.0f;
    float rank = live ? sc + bias[tid] : -INFINITY;
    threadgroup float bv_s[8];
    threadgroup int bi_s[8];
    threadgroup float chosen[16];
    threadgroup int chosen_i[16];
    for (uint k = 0; k < args.n_used; k++) {
        float bv = rank;
        int bi = live ? (int)tid : 0x7fffffff;
        for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(bv, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { bv_s[sg] = bv; bi_s[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = bv_s[0];
            int i = bi_s[0];
            for (uint g = 1; g < nsg; g++) {
                if (bv_s[g] > v || (bv_s[g] == v && bi_s[g] < i)) { v = bv_s[g]; i = bi_s[g]; }
            }
            chosen_i[k] = i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)tid == chosen_i[k]) { chosen[k] = sc; rank = -INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < args.n_used; k++) sum += chosen[k];
        for (uint k = 0; k < args.n_used; k++) {
            selected[(uint64_t)tok * args.n_used + k] = chosen_i[k];
            weights[(uint64_t)tok * args.n_used + k] = chosen[k] / sum * args.scale;
        }
    }
}

struct ds4_metal_args_mm3_moe_reduce {
    uint32_t n_tokens;
    uint32_t n_slots;
    uint32_t part_stride;  // slots per token in part
    uint32_t dim;
    uint32_t shared_src;   // 0 none, 1 part slot n_slots, 2 the shared buffer
    uint32_t pad0, pad1, pad2;
};

// x[t] += sum_s w[t][s] * part[t][s] + shared expert (unit weight).
kernel void kernel_mm3_moe_reduce(
        constant ds4_metal_args_mm3_moe_reduce & args,
        device float       *x,        // [T][dim], the residual stream
        device const float *part,     // [T][part_stride][dim]
        device const float *weights,  // [T][n_slots]
        device const float *shared,   // [T][dim] when shared_src == 2
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint d = tgpig.x * ntg.x + tid;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens || d >= args.dim) return;
    float acc = 0.0f;
    for (uint s = 0; s < args.n_slots; s++)
        acc += weights[(uint64_t)tok * args.n_slots + s] * part[((uint64_t)tok * args.part_stride + s) * args.dim + d];
    if (args.shared_src == 1) acc += part[((uint64_t)tok * args.part_stride + args.n_slots) * args.dim + d];
    else if (args.shared_src == 2) acc += shared[(uint64_t)tok * args.dim + d];
    x[(uint64_t)tok * args.dim + d] += acc;
}

struct ds4_metal_args_mm3_glu {
    uint32_t n;
    uint32_t pad0, pad1, pad2;
};

kernel void kernel_mm3_glu(
        constant ds4_metal_args_mm3_glu & args,
        device const float *gate,
        device const float *up,
        device float       *out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    out[gid] = qwen4_swiglu_oai(gate[gid], up[gid]);
}

/* MiniMax GGUFs keep routed expert matrices unscaled and store one F32
 * multiplier per expert/projection.  Gate and up must be scaled before the
 * non-linearity; down must be scaled before the routed reduction. */
static inline float2 mm3_q3_pair_dot(
        device const char *gate_row,
        device const char *up_row,
        device const float *x,
        uint in_dim,
        ushort lane) {
    float gate_acc = 0.0f, up_acc = 0.0f;
    const uint nb = in_dim / 256u;
    const uint lane0 = (uint)lane * 8u;
    /* A lane's eight values stay inside one 16-value Q3_K scale group.
     * Decode that group's metadata once per block, and reuse each activation
     * load for the gate and up rows.  The generic row-dot path decoded the
     * same scale eight times and loaded x independently for both matrices. */
    const uint group = (uint)lane / 2u;
    const uint local = group & 7u;
    const uint l0 = ((uint)lane & 1u) * 8u;
    const uint sg = group / 4u;
    const uint kmask1 = sg > 1u ? (sg > 2u ? 192u : 48u) : (sg > 0u ? 12u : 3u);
    const uint kmask2 = group >= 8u ? 0xF0u : 0x0Fu;
    const uint qbase = (group >= 8u ? 32u : 0u) + (local & 1u) * 16u;
    const uint qshift = 2u * (local / 2u);
    const uint hbase = (local & 1u) * 16u;
    const uint hbit = 1u << (group / 2u);
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar *gb = (device const uchar *)(gate_row + (uint64_t)ib * 110u);
        device const uchar *ub = (device const uchar *)(up_row + (uint64_t)ib * 110u);
        device const uchar *gsc = gb + 96u, *usc = ub + 96u;
        const uint gs2 = gsc[local], gs1 = gsc[8u + (local & 3u)];
        const uint us2 = usc[local], us1 = usc[8u + (local & 3u)];
        const uint gp = (local / 4u) & 1u ? (gs2 & kmask2) | ((gs1 & kmask1) << 2u)
                                               : (gs2 & kmask2) | ((gs1 & kmask1) << 4u);
        const uint up = (local / 4u) & 1u ? (us2 & kmask2) | ((us1 & kmask1) << 2u)
                                               : (us2 & kmask2) | ((us1 & kmask1) << 4u);
        const float gscale = (float)(*(device const half *)(gb + 108u)) *
                             ((float)(group < 8u ? gp : gp / 16u) - 32.0f);
        const float uscale = (float)(*(device const half *)(ub + 108u)) *
                             ((float)(group < 8u ? up : up / 16u) - 32.0f);
        device const uchar *gq = gb + 32u, *uq = ub + 32u;
        for (uint i = 0; i < 8u; i++) {
            const uint l = l0 + i;
            const float xv = x[ib * 256u + lane0 + i];
            const uint gqv = (gq[qbase + l] >> qshift) & 3u;
            const uint uqv = (uq[qbase + l] >> qshift) & 3u;
            gate_acc += gscale * ((float)gqv - ((gb[hbase + l] & hbit) ? 0.0f : 4.0f)) * xv;
            up_acc += uscale * ((float)uqv - ((ub[hbase + l] & hbit) ? 0.0f : 4.0f)) * xv;
        }
    }
    return float2(simd_sum(gate_acc), simd_sum(up_acc));
}

static inline float mm3_q6_row_dot(
        device const char *row,
        device const float *x,
        uint in_dim,
        ushort lane) {
    float acc = 0.0f;
    const uint nb = in_dim / 256u;
    const uint lane0 = (uint)lane * 8u;
    /* As with Q3_K, a lane's eight consecutive values remain inside one
     * Q6_K scale group and one packed quadrant.  Hoist all address/scale
     * decoding that does not vary across those eight values. */
    const uint half128 = lane0 / 128u;
    const uint p0 = lane0 & 127u;
    const uint quadrant = p0 / 32u;
    const uint l0 = p0 & 31u;
    const uint qli0 = half128 * 64u + l0 + ((quadrant & 1u) ? 32u : 0u);
    const uint qhi0 = half128 * 32u + l0;
    const uint qh_shift = 2u * quadrant;
    const uint scale_i = half128 * 8u + (l0 / 16u) + 2u * quadrant;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar *b = (device const uchar *)(row + (uint64_t)ib * 210u);
        device const uchar *ql = b, *qh = b + 128u;
        device const char *sc = (device const char *)(b + 192u);
        const float scale = (float)(*(device const half *)(b + 208u)) * (float)sc[scale_i];
        for (uint i = 0; i < 8u; i++) {
            const uint lo = quadrant >= 2u ? ql[qli0 + i] >> 4u : ql[qli0 + i] & 0xFu;
            const uint hi = (qh[qhi0 + i] >> qh_shift) & 3u;
            acc += scale * (float)((int)(lo | (hi << 4u)) - 32) * x[ib * 256u + lane0 + i];
        }
    }
    return simd_sum(acc);
}

static inline float mm3_nvfp4_row_dot(
        device const char *row,
        device const float *x,
        uint in_dim,
        ushort lane) {
    float acc = 0.0f;
    const uint nb = in_dim / 64u;
    const uint lane_group = (uint)lane / 8u;
    const uint lane0 = ((uint)lane & 7u) * 8u;
    const uint sub = lane0 / 16u;
    const uint j0 = lane0 & 15u;
    for (uint ib = lane_group; ib < nb; ib += 4u) {
        device const uchar *b = (device const uchar *)(row + (uint64_t)ib * 36u);
        const uchar e = b[sub];
        float scale = 0.0f;
        if (e != 0u && e != 0x7Fu) {
            const uint ex = (e >> 3u) & 0xFu, man = e & 7u;
            scale = ex == 0u ? ldexp((float)man, -10) :
                               ldexp(1.0f + (float)man / 8.0f, (int)ex - 8);
        }
        device const uchar *q = b + 4u + sub * 8u;
        for (uint i = 0; i < 8u; i++) {
            const uint j = j0 + i;
            const uchar packed = q[j & 7u];
            const uint qv = j < 8u ? packed & 0xFu : packed >> 4u;
            acc += scale * ds4_metal_mxfp4_values[qv] * x[ib * 64u + lane0 + i];
        }
    }
    return simd_sum(acc);
}

kernel void kernel_mm3_moe_mid_scaled(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *selected,
        device const float   *x,
        device float         *mid,
        device const float   *gate_scale,
        device const float   *up_scale,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * 2u;
    if (row0 >= args.out_rows || slot >= args.n_slots || tok >= args.n_tokens) return;
    const int32_t expert = selected[(uint64_t)tok * args.n_slots + slot];
    const uint64_t out_base = ((uint64_t)tok * args.n_slots + slot) * args.out_rows;
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0) for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++) mid[out_base + r] = 0.0f;
        return;
    }
    const uint e = (uint)expert;
    const uint64_t ebase = (uint64_t)e * args.expert_bytes;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * args.row_bytes;
        const float2 gu = args.weight_type == 11u && args.has_shared != 0u
            ? mm3_q3_pair_dot(gate_base + off, up_base + off, xt, args.in_dim, tiisg)
            : float2(qwen4_row_dot(gate_base + off, xt, args.weight_type, args.in_dim, tiisg),
                     qwen4_row_dot(up_base + off, xt, args.weight_type, args.in_dim, tiisg));
        const float g = gu.x, u = gu.y;
        if (tiisg == 0) mid[out_base + r] = qwen4_swiglu_oai(g * gate_scale[e], u * up_scale[e]);
    }
}

kernel void kernel_mm3_moe_down_scaled(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *down_base,
        device const int32_t *selected,
        device const float   *mid,
        device float         *part,
        device const float   *down_scale,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * 2u;
    if (row0 >= args.out_rows || slot >= args.n_slots || tok >= args.n_tokens) return;
    const int32_t expert = selected[(uint64_t)tok * args.n_slots + slot];
    const uint64_t pair = (uint64_t)tok * args.n_slots + slot;
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0) for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++) part[pair * args.out_rows + r] = 0.0f;
        return;
    }
    const uint e = (uint)expert;
    const uint64_t ebase = (uint64_t)e * args.expert_bytes;
    device const float *m = mid + pair * args.in_dim;
    for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++) {
        device const char *row = down_base + ebase + (uint64_t)r * args.row_bytes;
        const float v = args.weight_type == 14u && args.has_shared != 0u
            ? mm3_q6_row_dot(row, m, args.in_dim, tiisg)
            : args.weight_type == 40u && args.shared_type != 0u
                ? mm3_nvfp4_row_dot(row, m, args.in_dim, tiisg)
                : qwen4_row_dot(row, m, args.weight_type, args.in_dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v * down_scale[e];
    }
}

// Publish the decode router's choice to the CPU without waiting for the
// command buffer: both id lists and then the sequence number go to one small
// shared line the host polls.
kernel void kernel_mm3_publish(
        constant uint &seq,
        constant uint &n,
        device const int32_t *a,
        device const int32_t *b,
        device atomic_uint *line,     // [0] seq, [1..n] a, [1+n..2n] b
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    for (uint i = 0; i < n; i++) {
        atomic_store_explicit(&line[1 + i], (uint)a[i], memory_order_relaxed);
        atomic_store_explicit(&line[1 + n + i], (uint)b[i], memory_order_relaxed);
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_store_explicit(&line[0], seq, memory_order_relaxed);
}

/* Grouped routed experts.  shared_row_bytes == 0 keeps today's
 * (row block, slot, token) grid.  shared_row_bytes != 0 is the prefill grid
 * (row block, expert).  Fast Q3/Q6/NVFP4 rows are decoded once per four
 * tokens; the per-token product is the same expression as the one-token dot. */
template <uint NJ>
static inline void mm3_q3_pair_nj(
        device const char *gate_row,
        device const char *up_row,
        device const float *x,
        thread const uint *pairs,
        uint n_slots,
        uint in_dim,
        ushort lane,
        thread float *gate_out,
        thread float *up_out) {
    float gate_acc[NJ], up_acc[NJ];
#pragma unroll
    for (uint j = 0; j < NJ; j++) { gate_acc[j] = 0.0f; up_acc[j] = 0.0f; }
    const uint nb = in_dim / 256u;
    const uint lane0 = (uint)lane * 8u;
    const uint group = (uint)lane / 2u;
    const uint local = group & 7u;
    const uint l0 = ((uint)lane & 1u) * 8u;
    const uint sg = group / 4u;
    const uint kmask1 = sg > 1u ? (sg > 2u ? 192u : 48u) : (sg > 0u ? 12u : 3u);
    const uint kmask2 = group >= 8u ? 0xF0u : 0x0Fu;
    const uint qbase = (group >= 8u ? 32u : 0u) + (local & 1u) * 16u;
    const uint qshift = 2u * (local / 2u);
    const uint hbase = (local & 1u) * 16u;
    const uint hbit = 1u << (group / 2u);
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar *gb = (device const uchar *)(gate_row + (uint64_t)ib * 110u);
        device const uchar *ub = (device const uchar *)(up_row + (uint64_t)ib * 110u);
        device const uchar *gsc = gb + 96u, *usc = ub + 96u;
        const uint gs2 = gsc[local], gs1 = gsc[8u + (local & 3u)];
        const uint us2 = usc[local], us1 = usc[8u + (local & 3u)];
        const uint gp = (local / 4u) & 1u ? (gs2 & kmask2) | ((gs1 & kmask1) << 2u)
                                               : (gs2 & kmask2) | ((gs1 & kmask1) << 4u);
        const uint upv = (local / 4u) & 1u ? (us2 & kmask2) | ((us1 & kmask1) << 2u)
                                                : (us2 & kmask2) | ((us1 & kmask1) << 4u);
        const float gscale = (float)(*(device const half *)(gb + 108u)) *
                             ((float)(group < 8u ? gp : gp / 16u) - 32.0f);
        const float uscale = (float)(*(device const half *)(ub + 108u)) *
                             ((float)(group < 8u ? upv : upv / 16u) - 32.0f);
        device const uchar *gq = gb + 32u, *uq = ub + 32u;
        for (uint i = 0; i < 8u; i++) {
            const uint l = l0 + i;
            const uint gqv = (gq[qbase + l] >> qshift) & 3u;
            const uint uqv = (uq[qbase + l] >> qshift) & 3u;
            const float gh = (gb[hbase + l] & hbit) ? 0.0f : 4.0f;
            const float uh = (ub[hbase + l] & hbit) ? 0.0f : 4.0f;
#pragma unroll
            for (uint j = 0; j < NJ; j++) {
                const float xv = x[(uint64_t)(pairs[j] / n_slots) * in_dim + ib * 256u + lane0 + i];
                gate_acc[j] += gscale * ((float)gqv - gh) * xv;
                up_acc[j] += uscale * ((float)uqv - uh) * xv;
            }
        }
    }
#pragma unroll
    for (uint j = 0; j < NJ; j++) {
        gate_out[j] = simd_sum(gate_acc[j]);
        up_out[j] = simd_sum(up_acc[j]);
    }
}

template <uint NJ>
static inline void mm3_q6_nj(
        device const char *row,
        device const float *mid,
        thread const uint *pairs,
        uint in_dim,
        ushort lane,
        thread float *out) {
    float acc[NJ];
#pragma unroll
    for (uint j = 0; j < NJ; j++) acc[j] = 0.0f;
    const uint nb = in_dim / 256u;
    const uint lane0 = (uint)lane * 8u;
    const uint half128 = lane0 / 128u;
    const uint p0 = lane0 & 127u;
    const uint quadrant = p0 / 32u;
    const uint l0 = p0 & 31u;
    const uint qli0 = half128 * 64u + l0 + ((quadrant & 1u) ? 32u : 0u);
    const uint qhi0 = half128 * 32u + l0;
    const uint qh_shift = 2u * quadrant;
    const uint scale_i = half128 * 8u + (l0 / 16u) + 2u * quadrant;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar *b = (device const uchar *)(row + (uint64_t)ib * 210u);
        device const uchar *ql = b, *qh = b + 128u;
        device const char *sc = (device const char *)(b + 192u);
        const float scale = (float)(*(device const half *)(b + 208u)) * (float)sc[scale_i];
        for (uint i = 0; i < 8u; i++) {
            const uint lo = quadrant >= 2u ? ql[qli0 + i] >> 4u : ql[qli0 + i] & 0xFu;
            const uint hi = (qh[qhi0 + i] >> qh_shift) & 3u;
            const float w = scale * (float)((int)(lo | (hi << 4u)) - 32);
#pragma unroll
            for (uint j = 0; j < NJ; j++)
                acc[j] += w * mid[(uint64_t)pairs[j] * in_dim + ib * 256u + lane0 + i];
        }
    }
#pragma unroll
    for (uint j = 0; j < NJ; j++) out[j] = simd_sum(acc[j]);
}

template <uint NJ>
static inline void mm3_nvfp4_nj(
        device const char *row,
        device const float *mid,
        thread const uint *pairs,
        uint in_dim,
        ushort lane,
        thread float *out) {
    float acc[NJ];
#pragma unroll
    for (uint j = 0; j < NJ; j++) acc[j] = 0.0f;
    const uint nb = in_dim / 64u;
    const uint lane_group = (uint)lane / 8u;
    const uint lane0 = ((uint)lane & 7u) * 8u;
    const uint sub = lane0 / 16u;
    const uint j0 = lane0 & 15u;
    for (uint ib = lane_group; ib < nb; ib += 4u) {
        device const uchar *b = (device const uchar *)(row + (uint64_t)ib * 36u);
        const uchar e = b[sub];
        float scale = 0.0f;
        if (e != 0u && e != 0x7Fu) {
            const uint ex = (e >> 3u) & 0xFu, man = e & 7u;
            scale = ex == 0u ? ldexp((float)man, -10) :
                               ldexp(1.0f + (float)man / 8.0f, (int)ex - 8);
        }
        device const uchar *q = b + 4u + sub * 8u;
        for (uint i = 0; i < 8u; i++) {
            const uint qj = j0 + i;
            const uchar packed = q[qj & 7u];
            const uint qv = qj < 8u ? packed & 0xFu : packed >> 4u;
            const float w = scale * ds4_metal_mxfp4_values[qv];
#pragma unroll
            for (uint j = 0; j < NJ; j++)
                acc[j] += w * mid[(uint64_t)pairs[j] * in_dim + ib * 64u + lane0 + i];
        }
    }
#pragma unroll
    for (uint j = 0; j < NJ; j++) out[j] = simd_sum(acc[j]);
}

static inline void mm3_grouped_pairs(device const int32_t *list, uint j0, uint n, uint n_pairs,
                                     thread uint *pairs) {
    for (uint j = 0; j < 4u; j++) pairs[j] = 0u;
    for (uint j = 0; j < n; j++) {
        const uint pair = (uint)list[j0 + j];
        pairs[j] = pair < n_pairs ? pair : 0u;
    }
}

static inline void mm3_moe_mid_grouped_rows(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *gate_base,
        device const char *up_base,
        device const float *x,
        device float *mid,
        device const float *gate_scale,
        device const float *up_scale,
        device const int32_t *list,
        uint count,
        uint expert,
        uint row0,
        ushort tiisg) {
    const uint64_t ebase = (uint64_t)expert * args.expert_bytes;
    const uint n_pairs = args.n_tokens * args.n_slots;
    const bool q3 = args.weight_type == 11u && args.has_shared != 0u;
    const float gs = gate_scale[expert], us = up_scale[expert];
    for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * args.row_bytes;
        if (!q3) {
            for (uint j = 0; j < count; j++) {
                const uint pair = (uint)list[j];
                if (pair >= n_pairs) continue;
                device const float *xt = x + (uint64_t)(pair / args.n_slots) * args.in_dim;
                const float g = qwen4_row_dot(gate_base + off, xt, args.weight_type, args.in_dim, tiisg);
                const float u = qwen4_row_dot(up_base + off, xt, args.weight_type, args.in_dim, tiisg);
                if (tiisg == 0) mid[(uint64_t)pair * args.out_rows + r] = qwen4_swiglu_oai(g * gs, u * us);
            }
            continue;
        }
        for (uint j0 = 0; j0 < count; j0 += 4u) {
            const uint n = count - j0 < 4u ? count - j0 : 4u;
            uint pairs[4];
            mm3_grouped_pairs(list, j0, n, n_pairs, pairs);
            float g[4], u[4];
            if (n == 4u) mm3_q3_pair_nj<4>(gate_base + off, up_base + off, x, pairs, args.n_slots, args.in_dim, tiisg, g, u);
            else if (n == 3u) mm3_q3_pair_nj<3>(gate_base + off, up_base + off, x, pairs, args.n_slots, args.in_dim, tiisg, g, u);
            else if (n == 2u) mm3_q3_pair_nj<2>(gate_base + off, up_base + off, x, pairs, args.n_slots, args.in_dim, tiisg, g, u);
            else mm3_q3_pair_nj<1>(gate_base + off, up_base + off, x, pairs, args.n_slots, args.in_dim, tiisg, g, u);
            if (tiisg != 0) continue;
            for (uint j = 0; j < n; j++) {
                const uint pair = pairs[j];
                if (pair >= n_pairs || (uint)list[j0 + j] != pair) continue;
                mid[(uint64_t)pair * args.out_rows + r] = qwen4_swiglu_oai(g[j] * gs, u[j] * us);
            }
        }
    }
}

static inline void mm3_moe_down_grouped_rows(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *down_base,
        device const float *mid,
        device float *part,
        device const float *down_scale,
        device const int32_t *list,
        uint count,
        uint expert,
        uint row0,
        ushort tiisg) {
    const uint64_t ebase = (uint64_t)expert * args.expert_bytes;
    const uint n_pairs = args.n_tokens * args.n_slots;
    const bool q6 = args.weight_type == 14u && args.has_shared != 0u;
    const bool nv = args.weight_type == 40u && args.shared_type != 0u;
    const float ds = down_scale[expert];
    for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++) {
        device const char *row = down_base + ebase + (uint64_t)r * args.row_bytes;
        if (!q6 && !nv) {
            for (uint j = 0; j < count; j++) {
                const uint pair = (uint)list[j];
                if (pair >= n_pairs) continue;
                const float v = qwen4_row_dot(row, mid + (uint64_t)pair * args.in_dim, args.weight_type, args.in_dim, tiisg);
                if (tiisg == 0) part[(uint64_t)pair * args.out_rows + r] = v * ds;
            }
            continue;
        }
        for (uint j0 = 0; j0 < count; j0 += 4u) {
            const uint n = count - j0 < 4u ? count - j0 : 4u;
            uint pairs[4];
            mm3_grouped_pairs(list, j0, n, n_pairs, pairs);
            float v[4];
            if (q6) {
                if (n == 4u) mm3_q6_nj<4>(row, mid, pairs, args.in_dim, tiisg, v);
                else if (n == 3u) mm3_q6_nj<3>(row, mid, pairs, args.in_dim, tiisg, v);
                else if (n == 2u) mm3_q6_nj<2>(row, mid, pairs, args.in_dim, tiisg, v);
                else mm3_q6_nj<1>(row, mid, pairs, args.in_dim, tiisg, v);
            } else if (n == 4u) mm3_nvfp4_nj<4>(row, mid, pairs, args.in_dim, tiisg, v);
            else if (n == 3u) mm3_nvfp4_nj<3>(row, mid, pairs, args.in_dim, tiisg, v);
            else if (n == 2u) mm3_nvfp4_nj<2>(row, mid, pairs, args.in_dim, tiisg, v);
            else mm3_nvfp4_nj<1>(row, mid, pairs, args.in_dim, tiisg, v);
            if (tiisg != 0) continue;
            for (uint j = 0; j < n; j++) {
                const uint pair = pairs[j];
                if (pair >= n_pairs || (uint)list[j0 + j] != pair) continue;
                part[(uint64_t)pair * args.out_rows + r] = v[j] * ds;
            }
        }
    }
}

static inline uint mm3_grouped_count(constant ds4_metal_args_qwen4_moe & args,
                                     device const int32_t *counts, uint expert) {
    uint count = (uint)counts[expert];
    if (count > args.list_cap) count = args.list_cap;
    return count;
}

kernel void kernel_mm3_moe_mid_grouped(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *selected,
        device const float   *x,
        device float         *mid,
        device const float   *gate_scale,
        device const float   *up_scale,
        device const int32_t *lists,
        device const int32_t *counts,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * 2u;
    if (row0 >= args.out_rows) return;
    if (args.shared_row_bytes != 0u) {
        const uint e = tgpig.y;
        if (e >= args.n_total_expert) return;
        const uint count = mm3_grouped_count(args, counts, e);
        if (count == 0u) return;
        mm3_moe_mid_grouped_rows(args, gate_base, up_base, x, mid, gate_scale, up_scale,
                                 lists + (uint64_t)e * args.list_cap, count, e, row0, tiisg);
        return;
    }
    const uint slot = tgpig.y, tok = tgpig.z;
    if (slot >= args.n_slots || tok >= args.n_tokens) return;
    const uint pair = tok * args.n_slots + slot;
    const int32_t expert = selected[pair];
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0) {
            for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++)
                mid[(uint64_t)pair * args.out_rows + r] = 0.0f;
        }
        return;
    }
    const uint e = (uint)expert;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint count = mm3_grouped_count(args, counts, e);
    if (count == 0u || (uint)list[0] != pair) return;
    mm3_moe_mid_grouped_rows(args, gate_base, up_base, x, mid, gate_scale, up_scale,
                             list, count, e, row0, tiisg);
}

kernel void kernel_mm3_moe_down_grouped(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *down_base,
        device const int32_t *selected,
        device const float   *mid,
        device float         *part,
        device const float   *down_scale,
        device const int32_t *lists,
        device const int32_t *counts,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * 2u;
    if (row0 >= args.out_rows) return;
    if (args.shared_row_bytes != 0u) {
        const uint e = tgpig.y;
        if (e >= args.n_total_expert) return;
        const uint count = mm3_grouped_count(args, counts, e);
        if (count == 0u) return;
        mm3_moe_down_grouped_rows(args, down_base, mid, part, down_scale,
                                  lists + (uint64_t)e * args.list_cap, count, e, row0, tiisg);
        return;
    }
    const uint slot = tgpig.y, tok = tgpig.z;
    if (slot >= args.n_slots || tok >= args.n_tokens) return;
    const uint pair = tok * args.n_slots + slot;
    const int32_t expert = selected[pair];
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0) {
            for (uint r = row0; r < row0 + 2u && r < args.out_rows; r++)
                part[(uint64_t)pair * args.out_rows + r] = 0.0f;
        }
        return;
    }
    const uint e = (uint)expert;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint count = mm3_grouped_count(args, counts, e);
    if (count == 0u || (uint)list[0] != pair) return;
    mm3_moe_down_grouped_rows(args, down_base, mid, part, down_scale,
                              list, count, e, row0, tiisg);
}
