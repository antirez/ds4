// Laguna-specific primitives. The C graph owns model semantics and scheduling;
// these kernels only cover operations that are not represented by the shared
// DeepSeek/GLM Metal API.

struct ds4_metal_args_laguna_norm_rope {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t cache_row;   /* used only by the fused rope+store variant */
};

// Simdgroup-resident norm+rope for head_dim 128: each lane owns dims
// {lane, lane+32, lane+64, lane+96}, so the square sum reduces with one
// simd_sum (no barrier), every lane recomputes the cheap rsqrt, and both
// elements of every NeoX rotary pair land in the same lane for both Laguna
// rotary widths (64 and 128).  The head row is read and written exactly
// once, in registers, with no threadgroup memory at all.
static inline float4 laguna_head_rms_norm_rope_neox_simd(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *row,
        device const float *weight,
        ushort lane,
        uint token) {
    const uint d0 = lane;
    const uint d1 = lane + 32u;
    const uint d2 = lane + 64u;
    const uint d3 = lane + 96u;
    float4 v = float4(row[d0], row[d1], row[d2], row[d3]);
    const float ss = simd_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    const float inv = rsqrt(ss / 128.0f + args.eps);
    v *= inv * float4(weight[d0], weight[d1], weight[d2], weight[d3]);

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)args.n_rot,
                            (int)args.n_ctx_orig,
                            args.freq_base,
                            args.beta_fast,
                            args.beta_slow,
                            corr_dims);
    }
    const float inv_ndims = -1.0f / (float)args.n_rot;
    const float pos = (float)(args.pos0 + token);
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float log2_base = log2(args.freq_base);
#define DS4_LAGUNA_ROPE_THETA(rel) \
        (pos * exp2(inv_ndims * (float)(rel) * log2_base))
#else
#define DS4_LAGUNA_ROPE_THETA(rel) \
        (pos * pow(args.freq_base, inv_ndims * (float)(rel)))
#endif
    if (args.n_rot == 128u) {
        float cos_theta;
        float sin_theta;
        int rel_i0 = (int)(2u * lane);
        rope_yarn(DS4_LAGUNA_ROPE_THETA(rel_i0),
                  args.freq_scale, corr_dims, rel_i0,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        const float x0 = v.x;
        const float x1 = v.z;
        v.x = x0 * cos_theta - x1 * sin_theta;
        v.z = x0 * sin_theta + x1 * cos_theta;
        rel_i0 = (int)(2u * (lane + 32u));
        rope_yarn(DS4_LAGUNA_ROPE_THETA(rel_i0),
                  args.freq_scale, corr_dims, rel_i0,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        const float y0 = v.y;
        const float y1 = v.w;
        v.y = y0 * cos_theta - y1 * sin_theta;
        v.w = y0 * sin_theta + y1 * cos_theta;
    } else {
        /* n_rot == 64: dims 64..127 stay unrotated. */
        float cos_theta;
        float sin_theta;
        const int rel_i0 = (int)(2u * lane);
        rope_yarn(DS4_LAGUNA_ROPE_THETA(rel_i0),
                  args.freq_scale, corr_dims, rel_i0,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        const float x0 = v.x;
        const float x1 = v.y;
        v.x = x0 * cos_theta - x1 * sin_theta;
        v.y = x0 * sin_theta + x1 * cos_theta;
    }
#undef DS4_LAGUNA_ROPE_THETA
    row[d0] = v.x;
    row[d1] = v.y;
    row[d2] = v.z;
    row[d3] = v.w;
    return v;
}

// Barrier-free variants for the Laguna head geometry (head_dim 128,
// n_rot 64 or 128): four heads per threadgroup, one simdgroup each.
kernel void kernel_laguna_head_rms_norm_rope_simd(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *x,
        device const float *weight,
        ushort lane [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x * 4u + sgitg;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.head_dim != 128u ||
        (args.n_rot != 64u && args.n_rot != 128u)) {
        return;
    }
    device float *row = x +
        ((uint64_t)token * args.n_head + head) * args.head_dim;
    laguna_head_rms_norm_rope_neox_simd(args, row, weight, lane, token);
}

kernel void kernel_laguna_qk_head_rms_norm_rope_simd(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *q,
        device float       *k,
        device const float *q_weight,
        device const float *k_weight,
        constant uint      &n_q_head,
        ushort lane [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint combined_head = tgpig.x * 4u + sgitg;
    const uint token = tgpig.y;
    if (combined_head >= args.n_head || token >= args.n_tokens ||
        n_q_head >= args.n_head || args.head_dim != 128u ||
        (args.n_rot != 64u && args.n_rot != 128u)) {
        return;
    }
    const bool is_q = combined_head < n_q_head;
    const uint tensor_head = is_q ? combined_head : combined_head - n_q_head;
    const uint tensor_n_head = is_q ? n_q_head : args.n_head - n_q_head;
    device float *row = (is_q ? q : k) +
        ((uint64_t)token * tensor_n_head + tensor_head) * args.head_dim;
    laguna_head_rms_norm_rope_neox_simd(
        args, row, is_q ? q_weight : k_weight, lane, token);
}

// Decode-only twin of the Q/K kernel above that also commits the roped K row
// and the untouched V row to the attention ring as f16, removing the separate
// store dispatch from the per-token chain.  Four heads per threadgroup, one
// barrier-free simdgroup each; K heads convert their freshly roped registers
// straight to the cache row.
kernel void kernel_laguna_qk_head_rms_norm_rope_store_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *q,
        device float       *k,
        device const float *q_weight,
        device const float *k_weight,
        constant uint      &n_q_head,
        device const float *v,
        device half        *key_cache,
        device half        *value_cache,
        ushort lane [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint combined_head = tgpig.x * 4u + sgitg;
    if (combined_head >= args.n_head || args.n_tokens != 1u ||
        n_q_head >= args.n_head || args.head_dim != 128u ||
        (args.n_rot != 64u && args.n_rot != 128u)) {
        return;
    }

    const bool is_q = combined_head < n_q_head;
    const uint tensor_head = is_q ? combined_head : combined_head - n_q_head;
    const uint tensor_n_head = is_q ? n_q_head : args.n_head - n_q_head;
    device float *row = (is_q ? q : k) +
        (uint64_t)tensor_head * args.head_dim;
    const float4 roped = laguna_head_rms_norm_rope_neox_simd(
        args, row, is_q ? q_weight : k_weight, lane, 0u);
    if (is_q) return;

    const uint width = tensor_n_head * args.head_dim;
    const uint64_t dst =
        (uint64_t)args.cache_row * width +
        (uint64_t)tensor_head * args.head_dim;
    device const float *v_row = v + (uint64_t)tensor_head * args.head_dim;
    key_cache[dst + lane]       = (half)roped.x;
    key_cache[dst + lane + 32u] = (half)roped.y;
    key_cache[dst + lane + 64u] = (half)roped.z;
    key_cache[dst + lane + 96u] = (half)roped.w;
    value_cache[dst + lane]       = (half)v_row[lane];
    value_cache[dst + lane + 32u] = (half)v_row[lane + 32u];
    value_cache[dst + lane + 64u] = (half)v_row[lane + 64u];
    value_cache[dst + lane + 96u] = (half)v_row[lane + 96u];
}

// Rows twin of the fused rope+store kernel: verifier rows occupy
// consecutive cache slots (the caller guarantees the block does not wrap
// the sliding ring), so row r lands at cache_row + r.  Norm/RoPE values
// and the stored K/V bytes are identical to the unfused pair.
kernel void kernel_laguna_qk_head_rms_norm_rope_store_rows_neox(
        constant ds4_metal_args_laguna_norm_rope &args,
        device float       *q,
        device float       *k,
        device const float *q_weight,
        device const float *k_weight,
        constant uint      &n_q_head,
        device const float *v,
        device half        *key_cache,
        device half        *value_cache,
        ushort lane [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint combined_head = tgpig.x * 4u + sgitg;
    const uint token = tgpig.y;
    if (combined_head >= args.n_head || token >= args.n_tokens ||
        n_q_head >= args.n_head || args.head_dim != 128u ||
        (args.n_rot != 64u && args.n_rot != 128u)) {
        return;
    }

    const bool is_q = combined_head < n_q_head;
    const uint tensor_head = is_q ? combined_head : combined_head - n_q_head;
    const uint tensor_n_head = is_q ? n_q_head : args.n_head - n_q_head;
    device float *row = (is_q ? q : k) +
        ((uint64_t)token * tensor_n_head + tensor_head) * args.head_dim;
    const float4 roped = laguna_head_rms_norm_rope_neox_simd(
        args, row, is_q ? q_weight : k_weight, lane, token);
    if (is_q) return;

    const uint width = tensor_n_head * args.head_dim;
    const uint64_t dst =
        (uint64_t)(args.cache_row + token) * width +
        (uint64_t)tensor_head * args.head_dim;
    device const float *v_row = v +
        ((uint64_t)token * tensor_n_head + tensor_head) * args.head_dim;
    key_cache[dst + lane]       = (half)roped.x;
    key_cache[dst + lane + 32u] = (half)roped.y;
    key_cache[dst + lane + 64u] = (half)roped.z;
    key_cache[dst + lane + 96u] = (half)roped.w;
    value_cache[dst + lane]       = (half)v_row[lane];
    value_cache[dst + lane + 32u] = (half)v_row[lane + 32u];
    value_cache[dst + lane + 64u] = (half)v_row[lane + 64u];
    value_cache[dst + lane + 96u] = (half)v_row[lane + 96u];
}

struct ds4_metal_args_laguna_kv_store {
    uint32_t cache_cap;
    uint32_t cache_row;
    uint32_t n_head_kv;
    uint32_t head_dim;
};

kernel void kernel_laguna_store_kv_f16(
        constant ds4_metal_args_laguna_kv_store &args,
        device const float *k,
        device const float *v,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    if (gid >= width || args.cache_row >= args.cache_cap) return;
    const uint64_t dst = (uint64_t)args.cache_row * width + gid;
    key_cache[dst] = (half)k[gid];
    value_cache[dst] = (half)v[gid];
}

struct ds4_metal_args_laguna_prefill_attention {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    float    scale;
    uint32_t pad0;
};

// Before the sliding window wraps, verifier rows occupy distinct cache slots.
// Store the complete speculative block at once; each query still limits its
// key count, so later rows cannot become visible to earlier queries.
kernel void kernel_laguna_store_kv_rows_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *k,
        device const float *v,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    const uint token = gid / width;
    const uint col = gid - token * width;
    const uint cache_row = (args.pos0 + token) % args.cache_cap;
    const uint64_t dst = (uint64_t)cache_row * width + col;
    key_cache[dst] = (half)k[gid];
    value_cache[dst] = (half)v[gid];
}

// Stage the current chunk as f16 before attention. This preserves the same KV
// precision as decode without overwriting sliding-window rows that early
// queries in the chunk still need.
kernel void kernel_laguna_stage_kv_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *k,
        device const float *v,
        device half *staged_key,
        device half *staged_value,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    staged_key[gid] = (half)k[gid];
    staged_value[gid] = (half)v[gid];
}

// One SIMD group owns one query head. Queries in a prefill chunk execute in
// parallel, while each query visits keys in causal order. Keys from the current
// chunk come from the staging buffer; older keys come from the persistent ring.
kernel void kernel_laguna_attention_prefill_gqa_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh = q +
        ((uint64_t)token * args.n_head + head) * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    for (uint key_pos = key_start; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial += qh[d] * key_value;
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc = acc * old_scale + value * value_scale;
        max_score = next_max;
    }

    const float inv_sum = score_sum > 0.0f ? 1.0f / score_sum : 0.0f;
    const float gate_value = gate[(uint64_t)token * args.n_head + head];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out +
        ((uint64_t)token * args.n_head + head) * args.head_dim;
    oh[lane]       = acc.x * inv_sum * gate_scale;
    oh[lane + 32u] = acc.y * inv_sum * gate_scale;
    oh[lane + 64u] = acc.z * inv_sum * gate_scale;
    oh[lane + 96u] = acc.w * inv_sum * gate_scale;
}

// Three adjacent query heads sharing one KV head execute in one SIMD group.
// The per-head reduction and causal key order are unchanged, while each K/V
// value is fetched only once for the group.
kernel void kernel_laguna_attention_prefill_gqa3_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 3u;
    const uint head1 = head0 + 1u;
    const uint head2 = head0 + 2u;
    const uint token = tgpig.y;
    if (head2 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head2 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh0 = q +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    /* Four keys per pass: one vector simd_sum yields four scores, so the
     * per-key reduction and exp latency chains overlap.  The softmax result
     * is unchanged; only the running-max rescale happens per quad. */
    uint key_pos = key_start;
    for (; key_pos + 4u <= query_pos + 1u; key_pos += 4u) {
        uint64_t kb[4];
        bool cur[4];
        FOR_UNROLL (short kk = 0; kk < 4; kk++) {
            const uint kp = key_pos + (uint)kk;
            cur[kk] = kp >= args.pos0;
            const uint source_row = cur[kk] ?
                kp - args.pos0 : kp % args.cache_cap;
            kb[kk] = (uint64_t)source_row * cache_width +
                     (uint64_t)kv_head * args.head_dim;
        }

        float4 p0 = float4(0.0f);
        float4 p1 = float4(0.0f);
        float4 p2 = float4(0.0f);
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float q0 = qh0[d];
            const float q1 = qh1[d];
            const float q2 = qh2[d];
            FOR_UNROLL (short kk = 0; kk < 4; kk++) {
                const float key_value = cur[kk] ?
                    (float)staged_key[kb[kk] + d] :
                    (float)key_cache[kb[kk] + d];
                p0[kk] += q0 * key_value;
                p1[kk] += q1 * key_value;
                p2[kk] += q2 * key_value;
            }
        }
        const float4 s0 = simd_sum(p0) * args.scale;
        const float4 s1 = simd_sum(p1) * args.scale;
        const float4 s2 = simd_sum(p2) * args.scale;

        const float next_max0 =
            max(max0, max(max(s0.x, s0.y), max(s0.z, s0.w)));
        const float next_max1 =
            max(max1, max(max(s1.x, s1.y), max(s1.z, s1.w)));
        const float next_max2 =
            max(max2, max(max(s2.x, s2.y), max(s2.z, s2.w)));
        const float old_scale0 = max0 == -INFINITY ?
            0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ?
            0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ?
            0.0f : exp(max2 - next_max2);
        const float4 e0 = exp(s0 - next_max0);
        const float4 e1 = exp(s1 - next_max1);
        const float4 e2 = exp(s2 - next_max2);
        sum0 = sum0 * old_scale0 + (e0.x + e0.y) + (e0.z + e0.w);
        sum1 = sum1 * old_scale1 + (e1.x + e1.y) + (e1.z + e1.w);
        sum2 = sum2 * old_scale2 + (e2.x + e2.y) + (e2.z + e2.w);
        acc0 = acc0 * old_scale0;
        acc1 = acc1 * old_scale1;
        acc2 = acc2 * old_scale2;
        const uint d0 = lane;
        FOR_UNROLL (short kk = 0; kk < 4; kk++) {
            const float4 value = cur[kk] ?
                float4((float)staged_value[kb[kk] + d0],
                       (float)staged_value[kb[kk] + d0 + 32u],
                       (float)staged_value[kb[kk] + d0 + 64u],
                       (float)staged_value[kb[kk] + d0 + 96u]) :
                float4((float)value_cache[kb[kk] + d0],
                       (float)value_cache[kb[kk] + d0 + 32u],
                       (float)value_cache[kb[kk] + d0 + 64u],
                       (float)value_cache[kb[kk] + d0 + 96u]);
            acc0 += value * e0[kk];
            acc1 += value * e1[kk];
            acc2 += value * e2[kk];
        }
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }
    for (; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial0 += qh0[d] * key_value;
            partial1 += qh1[d] * key_value;
            partial2 += qh2[d] * key_value;
        }
        const float score0 = simd_sum(partial0) * args.scale;
        const float score1 = simd_sum(partial1) * args.scale;
        const float score2 = simd_sum(partial2) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float old_scale0 = max0 == -INFINITY ?
            0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ?
            0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ?
            0.0f : exp(max2 - next_max2);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }

    const float inv_sum0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inv_sum1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
    const float inv_sum2 = sum2 > 0.0f ? 1.0f / sum2 : 0.0f;
    const float gate_value0 = gate[(uint64_t)token * args.n_head + head0];
    const float gate_value1 = gate[(uint64_t)token * args.n_head + head1];
    const float gate_value2 = gate[(uint64_t)token * args.n_head + head2];
    const float gate_scale0 = gate_value0 > 20.0f ?
        gate_value0 : log(1.0f + exp(gate_value0));
    const float gate_scale1 = gate_value1 > 20.0f ?
        gate_value1 : log(1.0f + exp(gate_value1));
    const float gate_scale2 = gate_value2 > 20.0f ?
        gate_value2 : log(1.0f + exp(gate_value2));
    device float *oh0 = out +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device float *oh1 = oh0 + args.head_dim;
    device float *oh2 = oh1 + args.head_dim;
    oh0[lane]       = acc0.x * inv_sum0 * gate_scale0;
    oh0[lane + 32u] = acc0.y * inv_sum0 * gate_scale0;
    oh0[lane + 64u] = acc0.z * inv_sum0 * gate_scale0;
    oh0[lane + 96u] = acc0.w * inv_sum0 * gate_scale0;
    oh1[lane]       = acc1.x * inv_sum1 * gate_scale1;
    oh1[lane + 32u] = acc1.y * inv_sum1 * gate_scale1;
    oh1[lane + 64u] = acc1.z * inv_sum1 * gate_scale1;
    oh1[lane + 96u] = acc1.w * inv_sum1 * gate_scale1;
    oh2[lane]       = acc2.x * inv_sum2 * gate_scale2;
    oh2[lane + 32u] = acc2.y * inv_sum2 * gate_scale2;
    oh2[lane + 64u] = acc2.z * inv_sum2 * gate_scale2;
    oh2[lane + 96u] = acc2.w * inv_sum2 * gate_scale2;
}

// Global Laguna layers have six query heads per KV head. Keeping all six in
// one SIMD group preserves each head's reduction order while halving K/V
// traffic relative to the three-head kernel.
kernel void kernel_laguna_attention_prefill_gqa6_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort lane [[thread_index_in_simdgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 6u;
    const uint head5 = head0 + 5u;
    const uint token = tgpig.y;
    if (head5 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        args.cache_cap == 0u) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head5 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint query_pos = args.pos0 + token;
    const uint key_count = min(query_pos + 1u, args.cache_cap);
    const uint key_start = query_pos + 1u - key_count;
    device const float *qh0 = q +
        ((uint64_t)token * args.n_head + head0) * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;
    device const float *qh3 = qh2 + args.head_dim;
    device const float *qh4 = qh3 + args.head_dim;
    device const float *qh5 = qh4 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);
    float4 acc4 = float4(0.0f);
    float4 acc5 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float max3 = -INFINITY;
    float max4 = -INFINITY;
    float max5 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    /* Two keys per pass: vector simd_sums overlap the reduction and exp
     * chains while keeping the six-head register footprint in bounds. */
    uint key_pos = key_start;
    for (; key_pos + 2u <= query_pos + 1u; key_pos += 2u) {
        uint64_t kb[2];
        bool cur[2];
        FOR_UNROLL (short kk = 0; kk < 2; kk++) {
            const uint kp = key_pos + (uint)kk;
            cur[kk] = kp >= args.pos0;
            const uint source_row = cur[kk] ?
                kp - args.pos0 : kp % args.cache_cap;
            kb[kk] = (uint64_t)source_row * cache_width +
                     (uint64_t)kv_head * args.head_dim;
        }

        float2 p0 = float2(0.0f);
        float2 p1 = float2(0.0f);
        float2 p2 = float2(0.0f);
        float2 p3 = float2(0.0f);
        float2 p4 = float2(0.0f);
        float2 p5 = float2(0.0f);
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float2 key_value = float2(
                cur[0] ? (float)staged_key[kb[0] + d]
                       : (float)key_cache[kb[0] + d],
                cur[1] ? (float)staged_key[kb[1] + d]
                       : (float)key_cache[kb[1] + d]);
            p0 += qh0[d] * key_value;
            p1 += qh1[d] * key_value;
            p2 += qh2[d] * key_value;
            p3 += qh3[d] * key_value;
            p4 += qh4[d] * key_value;
            p5 += qh5[d] * key_value;
        }
        const float2 s0 = simd_sum(p0) * args.scale;
        const float2 s1 = simd_sum(p1) * args.scale;
        const float2 s2 = simd_sum(p2) * args.scale;
        const float2 s3 = simd_sum(p3) * args.scale;
        const float2 s4 = simd_sum(p4) * args.scale;
        const float2 s5 = simd_sum(p5) * args.scale;

        const float next_max0 = max(max0, max(s0.x, s0.y));
        const float next_max1 = max(max1, max(s1.x, s1.y));
        const float next_max2 = max(max2, max(s2.x, s2.y));
        const float next_max3 = max(max3, max(s3.x, s3.y));
        const float next_max4 = max(max4, max(s4.x, s4.y));
        const float next_max5 = max(max5, max(s5.x, s5.y));
        const float old_scale0 = max0 == -INFINITY ?
            0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ?
            0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ?
            0.0f : exp(max2 - next_max2);
        const float old_scale3 = max3 == -INFINITY ?
            0.0f : exp(max3 - next_max3);
        const float old_scale4 = max4 == -INFINITY ?
            0.0f : exp(max4 - next_max4);
        const float old_scale5 = max5 == -INFINITY ?
            0.0f : exp(max5 - next_max5);
        const float2 e0 = exp(s0 - next_max0);
        const float2 e1 = exp(s1 - next_max1);
        const float2 e2 = exp(s2 - next_max2);
        const float2 e3 = exp(s3 - next_max3);
        const float2 e4 = exp(s4 - next_max4);
        const float2 e5 = exp(s5 - next_max5);
        sum0 = sum0 * old_scale0 + e0.x + e0.y;
        sum1 = sum1 * old_scale1 + e1.x + e1.y;
        sum2 = sum2 * old_scale2 + e2.x + e2.y;
        sum3 = sum3 * old_scale3 + e3.x + e3.y;
        sum4 = sum4 * old_scale4 + e4.x + e4.y;
        sum5 = sum5 * old_scale5 + e5.x + e5.y;
        acc0 = acc0 * old_scale0;
        acc1 = acc1 * old_scale1;
        acc2 = acc2 * old_scale2;
        acc3 = acc3 * old_scale3;
        acc4 = acc4 * old_scale4;
        acc5 = acc5 * old_scale5;
        const uint dv = lane;
        FOR_UNROLL (short kk = 0; kk < 2; kk++) {
            const float4 value = cur[kk] ?
                float4((float)staged_value[kb[kk] + dv],
                       (float)staged_value[kb[kk] + dv + 32u],
                       (float)staged_value[kb[kk] + dv + 64u],
                       (float)staged_value[kb[kk] + dv + 96u]) :
                float4((float)value_cache[kb[kk] + dv],
                       (float)value_cache[kb[kk] + dv + 32u],
                       (float)value_cache[kb[kk] + dv + 64u],
                       (float)value_cache[kb[kk] + dv + 96u]);
            acc0 += value * e0[kk];
            acc1 += value * e1[kk];
            acc2 += value * e2[kk];
            acc3 += value * e3[kk];
            acc4 += value * e4[kk];
            acc5 += value * e5[kk];
        }
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
        max3 = next_max3;
        max4 = next_max4;
        max5 = next_max5;
    }
    for (; key_pos <= query_pos; key_pos++) {
        const bool current = key_pos >= args.pos0;
        const uint source_row = current ?
            key_pos - args.pos0 : key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)source_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        float partial4 = 0.0f;
        float partial5 = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            const float key_value = current ?
                (float)staged_key[kv_base + d] :
                (float)key_cache[kv_base + d];
            partial0 += qh0[d] * key_value;
            partial1 += qh1[d] * key_value;
            partial2 += qh2[d] * key_value;
            partial3 += qh3[d] * key_value;
            partial4 += qh4[d] * key_value;
            partial5 += qh5[d] * key_value;
        }
        const float score0 = simd_sum(partial0) * args.scale;
        const float score1 = simd_sum(partial1) * args.scale;
        const float score2 = simd_sum(partial2) * args.scale;
        const float score3 = simd_sum(partial3) * args.scale;
        const float score4 = simd_sum(partial4) * args.scale;
        const float score5 = simd_sum(partial5) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float next_max3 = max(max3, score3);
        const float next_max4 = max(max4, score4);
        const float next_max5 = max(max5, score5);
        const float old_scale0 = max0 == -INFINITY ?
            0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ?
            0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ?
            0.0f : exp(max2 - next_max2);
        const float old_scale3 = max3 == -INFINITY ?
            0.0f : exp(max3 - next_max3);
        const float old_scale4 = max4 == -INFINITY ?
            0.0f : exp(max4 - next_max4);
        const float old_scale5 = max5 == -INFINITY ?
            0.0f : exp(max5 - next_max5);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        const float value_scale3 = exp(score3 - next_max3);
        const float value_scale4 = exp(score4 - next_max4);
        const float value_scale5 = exp(score5 - next_max5);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        sum3 = sum3 * old_scale3 + value_scale3;
        sum4 = sum4 * old_scale4 + value_scale4;
        sum5 = sum5 * old_scale5 + value_scale5;
        const uint d0 = lane;
        const float4 value = current ?
            float4((float)staged_value[kv_base + d0],
                   (float)staged_value[kv_base + d0 + 32u],
                   (float)staged_value[kv_base + d0 + 64u],
                   (float)staged_value[kv_base + d0 + 96u]) :
            float4((float)value_cache[kv_base + d0],
                   (float)value_cache[kv_base + d0 + 32u],
                   (float)value_cache[kv_base + d0 + 64u],
                   (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        acc3 = acc3 * old_scale3 + value * value_scale3;
        acc4 = acc4 * old_scale4 + value * value_scale4;
        acc5 = acc5 * old_scale5 + value * value_scale5;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
        max3 = next_max3;
        max4 = next_max4;
        max5 = next_max5;
    }

    const float inv_sum0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
    const float inv_sum1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
    const float inv_sum2 = sum2 > 0.0f ? 1.0f / sum2 : 0.0f;
    const float inv_sum3 = sum3 > 0.0f ? 1.0f / sum3 : 0.0f;
    const float inv_sum4 = sum4 > 0.0f ? 1.0f / sum4 : 0.0f;
    const float inv_sum5 = sum5 > 0.0f ? 1.0f / sum5 : 0.0f;
    const uint64_t gate_base = (uint64_t)token * args.n_head + head0;
    const float gate_value0 = gate[gate_base];
    const float gate_value1 = gate[gate_base + 1u];
    const float gate_value2 = gate[gate_base + 2u];
    const float gate_value3 = gate[gate_base + 3u];
    const float gate_value4 = gate[gate_base + 4u];
    const float gate_value5 = gate[gate_base + 5u];
    const float gate_scale0 = gate_value0 > 20.0f ?
        gate_value0 : log(1.0f + exp(gate_value0));
    const float gate_scale1 = gate_value1 > 20.0f ?
        gate_value1 : log(1.0f + exp(gate_value1));
    const float gate_scale2 = gate_value2 > 20.0f ?
        gate_value2 : log(1.0f + exp(gate_value2));
    const float gate_scale3 = gate_value3 > 20.0f ?
        gate_value3 : log(1.0f + exp(gate_value3));
    const float gate_scale4 = gate_value4 > 20.0f ?
        gate_value4 : log(1.0f + exp(gate_value4));
    const float gate_scale5 = gate_value5 > 20.0f ?
        gate_value5 : log(1.0f + exp(gate_value5));
    device float *oh0 = out + gate_base * args.head_dim;
    device float *oh1 = oh0 + args.head_dim;
    device float *oh2 = oh1 + args.head_dim;
    device float *oh3 = oh2 + args.head_dim;
    device float *oh4 = oh3 + args.head_dim;
    device float *oh5 = oh4 + args.head_dim;
    oh0[lane]       = acc0.x * inv_sum0 * gate_scale0;
    oh0[lane + 32u] = acc0.y * inv_sum0 * gate_scale0;
    oh0[lane + 64u] = acc0.z * inv_sum0 * gate_scale0;
    oh0[lane + 96u] = acc0.w * inv_sum0 * gate_scale0;
    oh1[lane]       = acc1.x * inv_sum1 * gate_scale1;
    oh1[lane + 32u] = acc1.y * inv_sum1 * gate_scale1;
    oh1[lane + 64u] = acc1.z * inv_sum1 * gate_scale1;
    oh1[lane + 96u] = acc1.w * inv_sum1 * gate_scale1;
    oh2[lane]       = acc2.x * inv_sum2 * gate_scale2;
    oh2[lane + 32u] = acc2.y * inv_sum2 * gate_scale2;
    oh2[lane + 64u] = acc2.z * inv_sum2 * gate_scale2;
    oh2[lane + 96u] = acc2.w * inv_sum2 * gate_scale2;
    oh3[lane]       = acc3.x * inv_sum3 * gate_scale3;
    oh3[lane + 32u] = acc3.y * inv_sum3 * gate_scale3;
    oh3[lane + 64u] = acc3.z * inv_sum3 * gate_scale3;
    oh3[lane + 96u] = acc3.w * inv_sum3 * gate_scale3;
    oh4[lane]       = acc4.x * inv_sum4 * gate_scale4;
    oh4[lane + 32u] = acc4.y * inv_sum4 * gate_scale4;
    oh4[lane + 64u] = acc4.z * inv_sum4 * gate_scale4;
    oh4[lane + 96u] = acc4.w * inv_sum4 * gate_scale4;
    oh5[lane]       = acc5.x * inv_sum5 * gate_scale5;
    oh5[lane + 32u] = acc5.y * inv_sum5 * gate_scale5;
    oh5[lane + 64u] = acc5.z * inv_sum5 * gate_scale5;
    oh5[lane + 96u] = acc5.w * inv_sum5 * gate_scale5;
}

kernel void kernel_laguna_commit_kv_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const half *staged_key,
        device const half *staged_value,
        device half *key_cache,
        device half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint width = args.n_head_kv * args.head_dim;
    const uint values = args.n_tokens * width;
    if (gid >= values) return;
    const uint token = gid / width;
    const uint col = gid - token * width;
    const uint cache_row = (args.pos0 + token) % args.cache_cap;
    const uint64_t dst = (uint64_t)cache_row * width + col;
    key_cache[dst] = staged_key[gid];
    value_cache[dst] = staged_value[gid];
}

struct ds4_metal_args_laguna_attention {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t cache_cap;
    uint32_t key_start;
    uint32_t key_count;
    float    scale;
    uint32_t pad0;
};

struct ds4_metal_args_laguna_gqa3_decode {
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t key_count0;
    uint32_t n_tokens;
    uint32_t nsg;
    uint32_t nwg;
    float    scale;
};

// Global layers split keys across 32 workgroups. Evaluate three query heads
// sharing one KV head together so each K/V row is loaded once instead of three
// times. DFlash verifier rows use the second grid dimension while retaining
// the exact one-row arithmetic and established gated reduction.
kernel void kernel_laguna_attention_decode_gqa3_split_f16(
        constant ds4_metal_args_laguna_gqa3_decode &args,
        device const float *q,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *tmp,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head0 = tgpig.x * 3u;
    const uint head2 = head0 + 2u;
    const uint token = tgpig.y;
    const uint iwg = tgpig.z;
    const uint simd_group = (uint)simd_group_u;
    const uint key_count = args.key_count0 + token;
    if (head2 >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u ||
        key_count == 0u ||
        args.nsg == 0u || simd_group >= args.nsg || iwg >= args.nwg) {
        return;
    }

    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if (head2 / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t query_base =
        (uint64_t)token * args.n_head * args.head_dim;
    device const float *qh0 = q + query_base +
                              (uint64_t)head0 * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    const uint first = iwg * args.nsg + simd_group;
    const uint stride = args.nwg * args.nsg;
    for (uint i = first; i < key_count; i += stride) {
        const uint64_t kv_base =
            (uint64_t)i * cache_width +
            (uint64_t)kv_head * args.head_dim;
        const uint d0 = lane;
        const float4 key = float4((float)key_cache[kv_base + d0],
                                  (float)key_cache[kv_base + d0 + 32u],
                                  (float)key_cache[kv_base + d0 + 64u],
                                  (float)key_cache[kv_base + d0 + 96u]);
        const float score0 = simd_sum(dot(float4(qh0[d0],
                                                   qh0[d0 + 32u],
                                                   qh0[d0 + 64u],
                                                   qh0[d0 + 96u]), key)) * args.scale;
        const float score1 = simd_sum(dot(float4(qh1[d0],
                                                   qh1[d0 + 32u],
                                                   qh1[d0 + 64u],
                                                   qh1[d0 + 96u]), key)) * args.scale;
        const float score2 = simd_sum(dot(float4(qh2[d0],
                                                   qh2[d0 + 32u],
                                                   qh2[d0 + 64u],
                                                   qh2[d0 + 96u]), key)) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float old_scale0 = max0 == -INFINITY ? 0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ? 0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ? 0.0f : exp(max2 - next_max2);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        const float4 value = float4((float)value_cache[kv_base + d0],
                                    (float)value_cache[kv_base + d0 + 32u],
                                    (float)value_cache[kv_base + d0 + 64u],
                                    (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + 3u * args.nsg;
    threadgroup float *partial_value = partial_sum + 3u * args.nsg;
    const uint slots[3] = {simd_group,
                           args.nsg + simd_group,
                           2u * args.nsg + simd_group};
    const float maxima[3] = {max0, max1, max2};
    const float sums[3] = {sum0, sum1, sum2};
    const float4 values[3] = {acc0, acc1, acc2};
    for (uint h = 0u; h < 3u; h++) {
        const uint slot = slots[h];
        if (lane == 0u) {
            partial_max[slot] = maxima[h];
            partial_sum[slot] = sums[h];
        }
        const uint base = slot * args.head_dim + lane;
        partial_value[base] = values[h].x;
        partial_value[base + 32u] = values[h].y;
        partial_value[base + 64u] = values[h].z;
        partial_value[base + 96u] = values[h].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;
    const uint nrows = args.n_tokens * args.n_head;
    device float *stats =
        tmp + (uint64_t)nrows * args.head_dim * args.nwg;
    for (uint h = 0u; h < 3u; h++) {
        const uint slot_base = h * args.nsg;
        float global_max = partial_max[slot_base];
        for (uint sg = 1u; sg < args.nsg; sg++) {
            global_max = max(global_max, partial_max[slot_base + sg]);
        }
        float merged_sum = 0.0f;
        float4 merged = float4(0.0f);
        for (uint sg = 0u; sg < args.nsg; sg++) {
            const uint slot = slot_base + sg;
            const float weight = partial_sum[slot] > 0.0f ?
                exp(partial_max[slot] - global_max) : 0.0f;
            merged_sum += partial_sum[slot] * weight;
            const uint base = slot * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }

        const uint row = token * args.n_head + head0 + h;
        const uint64_t row_base =
            (uint64_t)row * args.head_dim * args.nwg;
        const uint dims[4] = {lane, lane + 32u, lane + 64u, lane + 96u};
        const float outputs[4] = {merged.x, merged.y, merged.z, merged.w};
        for (uint j = 0u; j < 4u; j++) {
            const uint d = dims[j];
            tmp[row_base + (d / 4u) * args.nwg * 4u + iwg * 4u + d % 4u] =
                outputs[j];
        }
        if (lane == 0u) {
            const uint64_t stat = (uint64_t)row * 2u * args.nwg + 2u * iwg;
            stats[stat] = merged_sum;
            stats[stat + 1u] = global_max;
        }
    }
}

// One threadgroup owns one query head. Short histories retain the original
// single-SIMD online reduction. Longer histories are striped over eight SIMD
// groups, then their independently normalized partials are merged in
// threadgroup memory. This keeps decode latency from growing serially with the
// absolute context position while preserving the short-context arithmetic.
kernel void kernel_laguna_attention_decode_gqa_f16(
        constant ds4_metal_args_laguna_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *out,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint head [[threadgroup_position_in_grid]]) {
    constexpr uint split_simd_groups = 8u;
    constexpr uint split_threshold = 256u;
    if (head >= args.n_head || args.n_head_kv == 0u ||
        args.head_dim != 128u || args.key_count == 0u) {
        return;
    }
    const uint simd_group = (uint)simd_group_u;
    const bool split = args.key_count > split_threshold;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    device const float *qh = q + (uint64_t)head * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    const uint key_first = split ? simd_group : 0u;
    const uint key_stride = split ? split_simd_groups : 1u;
    for (uint i = key_first; i < args.key_count; i += key_stride) {
        const uint key_pos = args.key_start + i;
        const uint cache_row = key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)cache_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            partial += qh[d] * (float)key_cache[kv_base + d];
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        acc.x = acc.x * old_scale +
            value_scale * (float)value_cache[kv_base + d0];
        acc.y = acc.y * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 32u];
        acc.z = acc.z * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 64u];
        acc.w = acc.w * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 96u];
        max_score = next_max;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + split_simd_groups;
    threadgroup float *partial_value = partial_sum + split_simd_groups;
    if (lane == 0u) {
        partial_max[simd_group] = max_score;
        partial_sum[simd_group] = score_sum;
    }
    const uint value_base = simd_group * args.head_dim;
    partial_value[value_base + lane] = acc.x;
    partial_value[value_base + lane + 32u] = acc.y;
    partial_value[value_base + lane + 64u] = acc.z;
    partial_value[value_base + lane + 96u] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;

    float4 merged = acc;
    float merged_sum = score_sum;
    if (split) {
        float global_max = partial_max[0];
        for (uint sg = 1u; sg < split_simd_groups; sg++) {
            global_max = max(global_max, partial_max[sg]);
        }
        merged = float4(0.0f);
        merged_sum = 0.0f;
        for (uint sg = 0u; sg < split_simd_groups; sg++) {
            const float weight = partial_sum[sg] > 0.0f ?
                exp(partial_max[sg] - global_max) : 0.0f;
            merged_sum += partial_sum[sg] * weight;
            const uint base = sg * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }
    }

    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    const float gate_value = gate[head];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out + (uint64_t)head * args.head_dim;
    oh[lane]       = merged.x * inv_sum * gate_scale;
    oh[lane + 32u] = merged.y * inv_sum * gate_scale;
    oh[lane + 64u] = merged.z * inv_sum * gate_scale;
    oh[lane + 96u] = merged.w * inv_sum * gate_scale;
}

// Batch independent pre-wrap verifier rows in the grid's Y dimension. This is
// deliberately the same per-head reduction as decode; only command encoding
// and dispatch are shared across rows.
kernel void kernel_laguna_attention_decode_rows_gqa_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *out,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint split_simd_groups = 8u;
    constexpr uint split_threshold = 256u;
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u) {
        return;
    }
    const uint key_count = min(args.pos0 + token + 1u, args.cache_cap);
    if (key_count == 0u) return;
    const uint key_start = args.pos0 + token + 1u - key_count;
    const uint simd_group = (uint)simd_group_u;
    const bool split = key_count > split_threshold;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device const float *qh = q + row * args.head_dim;

    float4 acc = float4(0.0f);
    float max_score = -INFINITY;
    float score_sum = 0.0f;
    const uint key_first = split ? simd_group : 0u;
    const uint key_stride = split ? split_simd_groups : 1u;
    for (uint i = key_first; i < key_count; i += key_stride) {
        const uint key_pos = key_start + i;
        const uint cache_row = key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)cache_row * cache_width +
            (uint64_t)kv_head * args.head_dim;

        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32u) {
            partial += qh[d] * (float)key_cache[kv_base + d];
        }
        const float score = simd_sum(partial) * args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ?
            0.0f : exp(max_score - next_max);
        const float value_scale = exp(score - next_max);
        score_sum = score_sum * old_scale + value_scale;
        const uint d0 = lane;
        acc.x = acc.x * old_scale +
            value_scale * (float)value_cache[kv_base + d0];
        acc.y = acc.y * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 32u];
        acc.z = acc.z * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 64u];
        acc.w = acc.w * old_scale +
            value_scale * (float)value_cache[kv_base + d0 + 96u];
        max_score = next_max;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + split_simd_groups;
    threadgroup float *partial_value = partial_sum + split_simd_groups;
    if (lane == 0u) {
        partial_max[simd_group] = max_score;
        partial_sum[simd_group] = score_sum;
    }
    const uint value_base = simd_group * args.head_dim;
    partial_value[value_base + lane] = acc.x;
    partial_value[value_base + lane + 32u] = acc.y;
    partial_value[value_base + lane + 64u] = acc.z;
    partial_value[value_base + lane + 96u] = acc.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;

    float4 merged = acc;
    float merged_sum = score_sum;
    if (split) {
        float global_max = partial_max[0];
        for (uint sg = 1u; sg < split_simd_groups; sg++) {
            global_max = max(global_max, partial_max[sg]);
        }
        merged = float4(0.0f);
        merged_sum = 0.0f;
        for (uint sg = 0u; sg < split_simd_groups; sg++) {
            const float weight = partial_sum[sg] > 0.0f ?
                exp(partial_max[sg] - global_max) : 0.0f;
            merged_sum += partial_sum[sg] * weight;
            const uint base = sg * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }
    }

    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    const float gate_value = gate[row];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    device float *oh = out + row * args.head_dim;
    oh[lane]       = merged.x * inv_sum * gate_scale;
    oh[lane + 32u] = merged.y * inv_sum * gate_scale;
    oh[lane + 64u] = merged.z * inv_sum * gate_scale;
    oh[lane + 96u] = merged.w * inv_sum * gate_scale;
}

// Sliding-window verifier rows evaluated three query heads per threadgroup so
// each K/V row is loaded once instead of three times (the global-layer gqa3
// split kernel established this reduction shape).  Eight SIMD groups stripe
// the key range; their independently normalized partials merge in threadgroup
// memory exactly like kernel_laguna_attention_decode_rows_gqa_f16.
kernel void kernel_laguna_attention_decode_rows_gqa3_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device float       *out,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint nsg = 8u;
    const uint head0 = tgpig.x * 3u;
    const uint token = tgpig.y;
    if (head0 + 2u >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u) {
        return;
    }
    const uint key_count = min(args.pos0 + token + 1u, args.cache_cap);
    if (key_count == 0u) return;
    const uint key_start = args.pos0 + token + 1u - key_count;
    const uint simd_group = (uint)simd_group_u;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if ((head0 + 2u) / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t row0 = (uint64_t)token * args.n_head + head0;
    device const float *qh0 = q + row0 * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (uint i = simd_group; i < key_count; i += nsg) {
        const uint key_pos = key_start + i;
        const uint cache_row = key_pos % args.cache_cap;
        const uint64_t kv_base =
            (uint64_t)cache_row * cache_width +
            (uint64_t)kv_head * args.head_dim;
        const uint d0 = lane;
        const float4 key = float4((float)key_cache[kv_base + d0],
                                  (float)key_cache[kv_base + d0 + 32u],
                                  (float)key_cache[kv_base + d0 + 64u],
                                  (float)key_cache[kv_base + d0 + 96u]);
        const float score0 = simd_sum(dot(float4(qh0[d0],
                                                   qh0[d0 + 32u],
                                                   qh0[d0 + 64u],
                                                   qh0[d0 + 96u]), key)) * args.scale;
        const float score1 = simd_sum(dot(float4(qh1[d0],
                                                   qh1[d0 + 32u],
                                                   qh1[d0 + 64u],
                                                   qh1[d0 + 96u]), key)) * args.scale;
        const float score2 = simd_sum(dot(float4(qh2[d0],
                                                   qh2[d0 + 32u],
                                                   qh2[d0 + 64u],
                                                   qh2[d0 + 96u]), key)) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float old_scale0 = max0 == -INFINITY ? 0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ? 0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ? 0.0f : exp(max2 - next_max2);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        const float4 value = float4((float)value_cache[kv_base + d0],
                                    (float)value_cache[kv_base + d0 + 32u],
                                    (float)value_cache[kv_base + d0 + 64u],
                                    (float)value_cache[kv_base + d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + 3u * nsg;
    threadgroup float *partial_value = partial_sum + 3u * nsg;
    const uint slots[3] = {simd_group,
                           nsg + simd_group,
                           2u * nsg + simd_group};
    const float maxima[3] = {max0, max1, max2};
    const float sums[3] = {sum0, sum1, sum2};
    const float4 values[3] = {acc0, acc1, acc2};
    for (uint h = 0u; h < 3u; h++) {
        const uint slot = slots[h];
        if (lane == 0u) {
            partial_max[slot] = maxima[h];
            partial_sum[slot] = sums[h];
        }
        const uint base = slot * args.head_dim + lane;
        partial_value[base] = values[h].x;
        partial_value[base + 32u] = values[h].y;
        partial_value[base + 64u] = values[h].z;
        partial_value[base + 96u] = values[h].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;
    for (uint h = 0u; h < 3u; h++) {
        const uint slot_base = h * nsg;
        float global_max = partial_max[slot_base];
        for (uint sg = 1u; sg < nsg; sg++) {
            global_max = max(global_max, partial_max[slot_base + sg]);
        }
        float merged_sum = 0.0f;
        float4 merged = float4(0.0f);
        for (uint sg = 0u; sg < nsg; sg++) {
            const uint slot = slot_base + sg;
            const float weight = partial_sum[slot] > 0.0f ?
                exp(partial_max[slot] - global_max) : 0.0f;
            merged_sum += partial_sum[slot] * weight;
            const uint base = slot * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }

        const uint64_t row = row0 + h;
        const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
        const float gate_value = gate[row];
        const float gate_scale = gate_value > 20.0f ?
            gate_value : log(1.0f + exp(gate_value));
        device float *oh = out + row * args.head_dim;
        oh[lane]       = merged.x * inv_sum * gate_scale;
        oh[lane + 32u] = merged.y * inv_sum * gate_scale;
        oh[lane + 64u] = merged.z * inv_sum * gate_scale;
        oh[lane + 96u] = merged.w * inv_sum * gate_scale;
    }
}

// Post-wrap twin of the gqa3 verifier-rows kernel: the block's own keys come
// from the staging buffer instead of the ring, so storing the block can wait
// until after attention and never overwrites slots still visible to earlier
// rows.  Key visit order per row is unchanged (ring segment then staged
// segment continue one striding sequence), so each row stays bit-identical
// to the pre-wrap kernel's reduction.
kernel void kernel_laguna_attention_decode_rows_gqa3_staged_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        threadgroup float  *scratch [[threadgroup(0)]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint nsg = 8u;
    const uint head0 = tgpig.x * 3u;
    const uint token = tgpig.y;
    if (head0 + 2u >= args.n_head || token >= args.n_tokens ||
        args.n_head_kv == 0u || args.head_dim != 128u) {
        return;
    }
    const uint key_count = min(args.pos0 + token + 1u, args.cache_cap);
    if (key_count == 0u) return;
    const uint key_start = args.pos0 + token + 1u - key_count;
    const uint ring_len = key_count - (token + 1u);
    const uint simd_group = (uint)simd_group_u;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head0 / heads_per_kv;
    if ((head0 + 2u) / heads_per_kv != kv_head) return;
    const uint cache_width = args.n_head_kv * args.head_dim;
    const uint64_t row0 = (uint64_t)token * args.n_head + head0;
    device const float *qh0 = q + row0 * args.head_dim;
    device const float *qh1 = qh0 + args.head_dim;
    device const float *qh2 = qh1 + args.head_dim;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float max2 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (uint i = simd_group; i < key_count; i += nsg) {
        device const half *kp;
        device const half *vp;
        if (i < ring_len) {
            const uint cache_row = (key_start + i) % args.cache_cap;
            const uint64_t kv_base =
                (uint64_t)cache_row * cache_width +
                (uint64_t)kv_head * args.head_dim;
            kp = key_cache + kv_base;
            vp = value_cache + kv_base;
        } else {
            const uint64_t kv_base =
                (uint64_t)(i - ring_len) * cache_width +
                (uint64_t)kv_head * args.head_dim;
            kp = staged_key + kv_base;
            vp = staged_value + kv_base;
        }
        const uint d0 = lane;
        const float4 key = float4((float)kp[d0],
                                  (float)kp[d0 + 32u],
                                  (float)kp[d0 + 64u],
                                  (float)kp[d0 + 96u]);
        const float score0 = simd_sum(dot(float4(qh0[d0],
                                                   qh0[d0 + 32u],
                                                   qh0[d0 + 64u],
                                                   qh0[d0 + 96u]), key)) * args.scale;
        const float score1 = simd_sum(dot(float4(qh1[d0],
                                                   qh1[d0 + 32u],
                                                   qh1[d0 + 64u],
                                                   qh1[d0 + 96u]), key)) * args.scale;
        const float score2 = simd_sum(dot(float4(qh2[d0],
                                                   qh2[d0 + 32u],
                                                   qh2[d0 + 64u],
                                                   qh2[d0 + 96u]), key)) * args.scale;
        const float next_max0 = max(max0, score0);
        const float next_max1 = max(max1, score1);
        const float next_max2 = max(max2, score2);
        const float old_scale0 = max0 == -INFINITY ? 0.0f : exp(max0 - next_max0);
        const float old_scale1 = max1 == -INFINITY ? 0.0f : exp(max1 - next_max1);
        const float old_scale2 = max2 == -INFINITY ? 0.0f : exp(max2 - next_max2);
        const float value_scale0 = exp(score0 - next_max0);
        const float value_scale1 = exp(score1 - next_max1);
        const float value_scale2 = exp(score2 - next_max2);
        sum0 = sum0 * old_scale0 + value_scale0;
        sum1 = sum1 * old_scale1 + value_scale1;
        sum2 = sum2 * old_scale2 + value_scale2;
        const float4 value = float4((float)vp[d0],
                                    (float)vp[d0 + 32u],
                                    (float)vp[d0 + 64u],
                                    (float)vp[d0 + 96u]);
        acc0 = acc0 * old_scale0 + value * value_scale0;
        acc1 = acc1 * old_scale1 + value * value_scale1;
        acc2 = acc2 * old_scale2 + value * value_scale2;
        max0 = next_max0;
        max1 = next_max1;
        max2 = next_max2;
    }

    threadgroup float *partial_max = scratch;
    threadgroup float *partial_sum = partial_max + 3u * nsg;
    threadgroup float *partial_value = partial_sum + 3u * nsg;
    const uint slots[3] = {simd_group,
                           nsg + simd_group,
                           2u * nsg + simd_group};
    const float maxima[3] = {max0, max1, max2};
    const float sums[3] = {sum0, sum1, sum2};
    const float4 values[3] = {acc0, acc1, acc2};
    for (uint h = 0u; h < 3u; h++) {
        const uint slot = slots[h];
        if (lane == 0u) {
            partial_max[slot] = maxima[h];
            partial_sum[slot] = sums[h];
        }
        const uint base = slot * args.head_dim + lane;
        partial_value[base] = values[h].x;
        partial_value[base + 32u] = values[h].y;
        partial_value[base + 64u] = values[h].z;
        partial_value[base + 96u] = values[h].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group != 0u) return;
    for (uint h = 0u; h < 3u; h++) {
        const uint slot_base = h * nsg;
        float global_max = partial_max[slot_base];
        for (uint sg = 1u; sg < nsg; sg++) {
            global_max = max(global_max, partial_max[slot_base + sg]);
        }
        float merged_sum = 0.0f;
        float4 merged = float4(0.0f);
        for (uint sg = 0u; sg < nsg; sg++) {
            const uint slot = slot_base + sg;
            const float weight = partial_sum[slot] > 0.0f ?
                exp(partial_max[slot] - global_max) : 0.0f;
            merged_sum += partial_sum[slot] * weight;
            const uint base = slot * args.head_dim + lane;
            merged.x += partial_value[base] * weight;
            merged.y += partial_value[base + 32u] * weight;
            merged.z += partial_value[base + 64u] * weight;
            merged.w += partial_value[base + 96u] * weight;
        }

        const uint64_t row = row0 + h;
        const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
        const float gate_value = gate[row];
        const float gate_scale = gate_value > 20.0f ?
            gate_value : log(1.0f + exp(gate_value));
        device float *oh = out + row * args.head_dim;
        oh[lane]       = merged.x * inv_sum * gate_scale;
        oh[lane + 32u] = merged.y * inv_sum * gate_scale;
        oh[lane + 64u] = merged.z * inv_sum * gate_scale;
        oh[lane + 96u] = merged.w * inv_sum * gate_scale;
    }
}

// Laguna's split-K decode attention consumes the generic FlashAttention
// partial layout, but every reduced head is immediately multiplied by its
// learned gate. Apply that epilogue before the final store so decode does not
// write and reread the complete head buffer in a second dispatch.
kernel void kernel_laguna_flash_attn_reduce_gate_f32(
        constant ds4_metal_args_flash_attn_ext_vec_reduce &args,
        device const char  *htmp,
        device       char  *dst,
        device const float *gate,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t row = tgpig;
    device const float *stats = (device const float *)htmp +
        (uint64_t)args.nrows * DV * NWG;

    float sum = stats[row * (2 * NWG) + 2 * lane];
    float max_value = stats[row * (2 * NWG) + 2 * lane + 1];
    const float global_max = simd_max(max_value);
    const float scale = exp(max_value - global_max);
    sum = simd_sum(sum * scale);
    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;

    const float gate_value = gate[row];
    const float gate_scale = gate_value > 20.0f ?
        gate_value : log(1.0f + exp(gate_value));
    const short DV4 = DV / 4;
    device const float4 *partials =
        (device const float4 *)htmp + row * DV4 * NWG;
    device volatile float4 *out =
        (device volatile float4 *)dst + row * DV4;

    for (short i = simd_group; i < DV4; i += NWG) {
        const float4 value = simd_sum(partials[i * NWG + lane] * scale);
        if (lane == 0) {
            // Keep the same F32 materialization boundary as the unfused
            // reducer followed by the gate kernel. The volatile round trip
            // prevents fast-math from reassociating the two multiplies.
            out[i] = value * inv_sum;
            out[i] = out[i] * gate_scale;
        }
    }

#undef NWG
#undef DV
}

struct ds4_metal_args_laguna_q6_matmul {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tokens;
    uint32_t pad0;
    uint64_t row_bytes;
};

// Dense Q6_K projection used by Laguna's down projections and output head.
// The quantized arithmetic follows DwarfStar's existing Q6_K routed-down
// implementation, but addresses a single dense matrix directly.
kernel void kernel_laguna_q6_K_matmul_f32(
        constant ds4_metal_args_laguna_q6_matmul &args,
        device const char  *weight,
        device const float *x,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_simd = 2u;
    constexpr uint simd_groups = 2u;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;
    constexpr uint qk_k = 256u;

    const uint row0 = (tgpig.x * simd_groups + simd_group) * rows_per_simd;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int n_blocks = (int)(args.in_dim / qk_k);
    const short tid = (short)(lane / 2u);
    const short ix = (short)(lane & 1u);
    const short ip = (short)(tid / 8);
    const short il = (short)(tid % 8);
    const short l0 = (short)(4 * il);
    const short is = (short)(8 * ip + l0 / 16);
    const short y_offset = (short)(128 * ip + l0);
    const short q_offset_l = (short)(64 * ip + l0);
    const short q_offset_h = (short)(32 * ip + l0);
    device const float *input = x + (uint64_t)token * args.in_dim;
    float sums[rows_per_simd] = {0.0f, 0.0f};
    float yl[16];

    for (int ib = ix; ib < n_blocks; ib += 2) {
        device const float *y = input + (uint64_t)ib * qk_k + y_offset;
        for (short l = 0; l < 4; l++) {
            yl[4 * l + 0] = y[l + 0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }

        for (uint r = 0u; r < rows_per_simd && row0 + r < args.out_dim; r++) {
            device const block_q6_K *block =
                (device const block_q6_K *)(weight +
                    (uint64_t)(row0 + r) * args.row_bytes) + ib;
            device const uchar *q1 = block->ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = block->qh + q_offset_h;
            device const char *sc = block->scales + is;
            float4 part = float4(0.0f);
            for (short l = 0; l < 4; l++) {
                const uint h = (uint)qh[l];
                part[0] += yl[4 * l + 0] *
                    (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                part[1] += yl[4 * l + 1] *
                    (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                part[2] += yl[4 * l + 2] *
                    (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                part[3] += yl[4 * l + 3] *
                    (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
            }
            sums[r] += (float)block->d *
                (part[0] * (float)sc[0] + part[1] * (float)sc[2] +
                 part[2] * (float)sc[4] + part[3] * (float)sc[6]);
        }
    }

    for (uint r = 0u; r < rows_per_simd && row0 + r < args.out_dim; r++) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + r] = sum;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
/*
 * Register-resident FlashAttention for Laguna prefill on the M5 tensor
 * units.  The fragment machinery is a compact port of MLX's steel NAX
 * attention support (Copyright © 2025 Apple Inc., MIT license): 16x16
 * fragments live in registers, matmuls run through
 * mpp::tensor_ops::matmul2d<16,32,16> per simdgroup, and the online
 * softmax reduces rows with two lane shuffles.  No threadgroup memory is
 * used anywhere.
 *
 * Laguna specifics on top of the steel kernel: keys arrive as up to three
 * linear segments (the full-attention ring is linear, the sliding-window
 * ring splits at its wrap point, and the current chunk lives in the staging
 * buffers), masking applies causality plus the 512-token window in absolute
 * positions, and the learned per-head gate multiplies the normalized output
 * in the epilogue.
 */

#define STEEL_CONST static constant constexpr const

namespace ds4nax {

constant constexpr float kNegInf = -3.402823466e38f;

struct Frag {
    STEEL_CONST short kFragRows = 16;
    STEEL_CONST short kFragCols = 16;
    STEEL_CONST short kElemsPerFrag = 8;
    STEEL_CONST short kElemRows = 2;
    STEEL_CONST short kElemCols = 4;
    STEEL_CONST short kElemRowsJump = 8;

    template <typename U>
    using frag_t = metal::vec<U, kElemsPerFrag>;

    METAL_FUNC static short2 get_coord() {
        const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
        const short qid = lane >> 2;
        const short fm = ((qid & 4) | ((lane >> 1) & 3));
        const short fn = ((qid & 2) | (lane & 1)) * 4;
        return short2{fn, fm};
    }

    template <typename T, typename U>
    METAL_FUNC static void load(
            thread frag_t<T> &dst,
            device const U *src,
            const int ld,
            const int off_x,
            const int off_y) {
        const short2 sc = get_coord();
        src += (sc.y + off_x) * ld + sc.x + off_y;
        FOR_UNROLL (short i = 0; i < kElemRows; i++) {
            FOR_UNROLL (short j = 0; j < kElemCols; j++) {
                dst[i * kElemCols + j] =
                    static_cast<T>(src[i * kElemRowsJump * ld + j]);
            }
        }
    }

    template <typename T, typename U>
    METAL_FUNC static void load_rows(
            thread frag_t<T> &dst,
            device const U *src,
            const int ld,
            const int lim_x,
            const int off_x,
            const int off_y) {
        const short2 sc = get_coord();
        src += (sc.y + off_x) * ld + sc.x + off_y;
        const int lx = lim_x - off_x - sc.y;
        FOR_UNROLL (short i = 0; i < kElemRows; i++) {
            if (i * kElemRowsJump < lx) {
                FOR_UNROLL (short j = 0; j < kElemCols; j++) {
                    dst[i * kElemCols + j] =
                        static_cast<T>(src[i * kElemRowsJump * ld + j]);
                }
            } else {
                FOR_UNROLL (short j = 0; j < kElemCols; j++) {
                    dst[i * kElemCols + j] = T(0);
                }
            }
        }
    }

    template <typename T, typename U>
    METAL_FUNC static void store_rows(
            const thread frag_t<T> &src,
            device U *dst,
            const int ld,
            const int lim_x,
            const int off_x,
            const int off_y) {
        const short2 sc = get_coord();
        dst += (sc.y + off_x) * ld + sc.x + off_y;
        const int lx = lim_x - off_x - sc.y;
        FOR_UNROLL (short i = 0; i < kElemRows; i++) {
            if (i * kElemRowsJump < lx) {
                FOR_UNROLL (short j = 0; j < kElemCols; j++) {
                    dst[i * kElemRowsJump * ld + j] =
                        static_cast<U>(src[i * kElemCols + j]);
                }
            }
        }
    }

    template <typename Op, typename T>
    METAL_FUNC static void row_reduce(
            const thread frag_t<T> &inp,
            thread T *reduced) {
        FOR_UNROLL (short i = 0; i < kElemRows; i++) {
            T tr = Op::apply(
                Op::apply(inp[i * kElemCols + 0], inp[i * kElemCols + 1]),
                Op::apply(inp[i * kElemCols + 2], inp[i * kElemCols + 3]));
            T qr = simd_shuffle_xor(tr, ushort(1));
            qr = Op::apply(tr, qr);
            T sr = simd_shuffle_xor(qr, ushort(8));
            sr = Op::apply(qr, sr);
            reduced[i] = Op::apply(reduced[i], sr);
        }
    }

    template <typename Op, typename T>
    METAL_FUNC static void row_bin_op(
            thread frag_t<T> &inp,
            const thread T *rows) {
        FOR_UNROLL (short i = 0; i < kElemRows; i++) {
            FOR_UNROLL (short j = 0; j < kElemCols; j++) {
                inp[i * kElemCols + j] =
                    Op::apply(inp[i * kElemCols + j], rows[i]);
            }
        }
    }

    /* C[16x32] += A[16x16] x (B0|B1)[16x32], optionally B transposed. */
    template <typename CT, typename AT, typename BT, bool tb>
    METAL_FUNC static void mma(
            thread frag_t<CT> &c0,
            thread frag_t<CT> &c1,
            const thread frag_t<AT> &a,
            const thread frag_t<BT> &b0,
            const thread frag_t<BT> &b1,
            metal::bool_constant<tb>) {
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 16, false, tb, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        auto ca = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
        auto cb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
        auto cc = op.template get_destination_cooperative_tensor<
            decltype(ca), decltype(cb), CT>();
        FOR_UNROLL (short i = 0; i < kElemsPerFrag; i++) {
            ca[i] = a[i];
        }
        FOR_UNROLL (short i = 0; i < kElemsPerFrag; i++) {
            cb[i] = b0[i];
            cb[kElemsPerFrag + i] = b1[i];
        }
        FOR_UNROLL (short i = 0; i < kElemsPerFrag; i++) {
            cc[i] = c0[i];
            cc[kElemsPerFrag + i] = c1[i];
        }
        op.run(ca, cb, cc);
        FOR_UNROLL (short i = 0; i < kElemsPerFrag; i++) {
            c0[i] = cc[i];
            c1[i] = cc[kElemsPerFrag + i];
        }
    }
};

struct MaxOp {
    template <typename T>
    METAL_FUNC static T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
    template <typename T>
    METAL_FUNC static T apply(T x, T y) { return x + y; }
};
struct MulOp {
    template <typename T>
    METAL_FUNC static T apply(T x, T y) { return x * y; }
};
struct ExpSubOp {
    template <typename T>
    METAL_FUNC static T apply(T x, T y) { return metal::fast::exp2(x - y); }
};

} // namespace ds4nax

/* One threadgroup covers 64 query rows of one head: four simdgroups of 16
 * rows each, fully warp-autonomous except for two scheduling barriers that
 * every warp reaches the same number of times (loop bounds derive from the
 * threadgroup's query range; per-row precision comes from masking). */
kernel void kernel_laguna_attention_prefill_nax_f16(
        constant ds4_metal_args_laguna_prefill_attention &args,
        device const float *q,
        device const float *gate,
        device const half  *key_cache,
        device const half  *value_cache,
        device const half  *staged_key,
        device const half  *staged_value,
        device float       *out,
        ushort simd_group_id [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    using Frag = ds4nax::Frag;
    constexpr short BQ = 64;
    constexpr short BK = 32;
    constexpr short TD = 8;   /* 128 / 16 head-dim fragments */
    constexpr short TK = 2;   /* 32 / 16 key fragments per block */
    using frag_f = Frag::frag_t<float>;
    using frag_h = Frag::frag_t<half>;

    const uint head = tgpig.y;
    if (head >= args.n_head || args.head_dim != 128u) return;
    const uint heads_per_kv = args.n_head / args.n_head_kv;
    const uint kv_head = head / heads_per_kv;
    const int kv_ld = (int)(args.n_head_kv * args.head_dim);
    const int q_ld = (int)(args.n_head * args.head_dim);

    const int tile_row0 = (int)tgpig.x * BQ + 16 * (int)simd_group_id;
    device const float *Qw = q + (uint64_t)tile_row0 * q_ld +
                             (uint64_t)head * args.head_dim;
    const int rows_lim = (int)args.n_tokens - tile_row0;
    const int q_abs0 = (int)args.pos0 + tile_row0;

    /* Threadgroup-uniform query range for loop bounds. */
    const int tg_q_lo = (int)args.pos0 + (int)tgpig.x * BQ;
    const int tg_q_hi = (int)args.pos0 +
        metal::min((int)tgpig.x * BQ + BQ, (int)args.n_tokens) - 1;
    const int window = args.cache_cap == 512u ? 512 : 0x40000000;

    const float scale2 = args.scale * 1.44269504089f;

    frag_f Ofrag[TD];
    FOR_UNROLL (short i = 0; i < TD; i++) {
        Ofrag[i] = frag_f(0.0f);
    }
    float2 max_score = float2(ds4nax::kNegInf);
    float2 sum_score = float2(0.0f);

    /* Key/value segments in absolute-position order. */
    device const half *seg_k[3];
    device const half *seg_v[3];
    int seg_abs[3];
    int seg_len[3];
    short n_segs = 0;
    if (args.cache_cap != 512u) {
        /* Full attention: the ring never wraps during prefill. */
        if (args.pos0 != 0u) {
            seg_k[n_segs] = key_cache + (uint64_t)kv_head * args.head_dim;
            seg_v[n_segs] = value_cache + (uint64_t)kv_head * args.head_dim;
            seg_abs[n_segs] = 0;
            seg_len[n_segs] = (int)args.pos0;
            n_segs++;
        }
    } else {
        const int h = metal::min((int)args.pos0, 512);
        if (h > 0) {
            const int base_abs = (int)args.pos0 - h;
            const int slot0 = base_abs % 512;
            const int part1 = metal::min(h, 512 - slot0);
            seg_k[n_segs] = key_cache + (uint64_t)slot0 * kv_ld +
                            (uint64_t)kv_head * args.head_dim;
            seg_v[n_segs] = value_cache + (uint64_t)slot0 * kv_ld +
                            (uint64_t)kv_head * args.head_dim;
            seg_abs[n_segs] = base_abs;
            seg_len[n_segs] = part1;
            n_segs++;
            if (h > part1) {
                seg_k[n_segs] = key_cache + (uint64_t)kv_head * args.head_dim;
                seg_v[n_segs] = value_cache + (uint64_t)kv_head * args.head_dim;
                seg_abs[n_segs] = base_abs + part1;
                seg_len[n_segs] = h - part1;
                n_segs++;
            }
        }
    }
    seg_k[n_segs] = staged_key + (uint64_t)kv_head * args.head_dim;
    seg_v[n_segs] = staged_value + (uint64_t)kv_head * args.head_dim;
    seg_abs[n_segs] = (int)args.pos0;
    seg_len[n_segs] = (int)args.n_tokens;
    n_segs++;

    const short2 sc = Frag::get_coord();
    const short sm = sc.y;
    const short sn = sc.x;

    for (short seg = 0; seg < n_segs; seg++) {
        const int s_abs = seg_abs[seg];
        const int s_len = seg_len[seg];
        /* Block culling against the threadgroup's query range. */
        int kb_start = 0;
        if (window != 0x40000000) {
            kb_start = metal::max(0, (tg_q_lo - window + 1 - s_abs) / BK);
        }
        int kb_lim = (tg_q_hi - s_abs) / BK + 1;
        kb_lim = metal::min(kb_lim, (s_len + BK - 1) / BK);
        if (kb_lim <= kb_start) continue;

        device const half *Kw = seg_k[seg] + (uint64_t)kb_start * BK * kv_ld;
        device const half *Vw = seg_v[seg] + (uint64_t)kb_start * BK * kv_ld;

        for (int kb = kb_start; kb < kb_lim; kb++) {
            const int col_abs0 = s_abs + kb * BK;
            const int col_in_seg0 = kb * BK;
            const bool tail_k = col_in_seg0 + BK > s_len;

            /* S = Q @ K^T in the exp2 domain. */
            frag_f Sfrag[TK];
            FOR_UNROLL (short ik = 0; ik < TK; ik++) {
                Sfrag[ik] = frag_f(0.0f);
            }
            FOR_UNROLL (short id = 0; id < TD; id++) {
                frag_h Qf;
                frag_h K0;
                frag_h K1;
                if (rows_lim < 16) {
                    Frag::load_rows(Qf, Qw, q_ld, rows_lim, 0, id * 16);
                } else {
                    Frag::load(Qf, Qw, q_ld, 0, id * 16);
                }
                if (tail_k) {
                    Frag::load_rows(K0, Kw, kv_ld, s_len - col_in_seg0,
                                    0, id * 16);
                    Frag::load_rows(K1, Kw, kv_ld, s_len - col_in_seg0,
                                    16, id * 16);
                } else {
                    Frag::load(K0, Kw, kv_ld, 0, id * 16);
                    Frag::load(K1, Kw, kv_ld, 16, id * 16);
                }
                Frag::mma(Sfrag[0], Sfrag[1], Qf, K0, K1,
                          metal::bool_constant<true>{});
            }
            FOR_UNROLL (short ik = 0; ik < TK; ik++) {
                FOR_UNROLL (short ii = 0; ii < Frag::kElemsPerFrag; ii++) {
                    Sfrag[ik][ii] *= scale2;
                }
            }

            /* Causality, sliding window, and segment tail in one mask. */
            const bool needs_causal = col_abs0 + BK - 1 > tg_q_lo;
            const bool needs_window = window != 0x40000000 &&
                col_abs0 < tg_q_hi - window + 1;
            if (needs_causal || needs_window || tail_k) {
                FOR_UNROLL (short ik = 0; ik < TK; ik++) {
                    FOR_UNROLL (short ii = 0; ii < Frag::kElemRows; ii++) {
                        const int r = q_abs0 + sm + ii * Frag::kElemRowsJump;
                        FOR_UNROLL (short jj = 0; jj < Frag::kElemCols; jj++) {
                            const int c = col_abs0 + ik * 16 + sn + jj;
                            const int cs = col_in_seg0 + ik * 16 + sn + jj;
                            const bool valid =
                                c <= r && r - c < window && cs < s_len;
                            const short loc = ii * Frag::kElemCols + jj;
                            Sfrag[ik][loc] =
                                valid ? Sfrag[ik][loc] : ds4nax::kNegInf;
                        }
                    }
                }
            }

            /* Online softmax. */
            float2 new_max = max_score;
            FOR_UNROLL (short ik = 0; ik < TK; ik++) {
                Frag::row_reduce<ds4nax::MaxOp>(Sfrag[ik],
                                                (thread float *)&new_max);
            }
            FOR_UNROLL (short ik = 0; ik < TK; ik++) {
                Frag::row_bin_op<ds4nax::ExpSubOp>(Sfrag[ik],
                                                   (thread float *)&new_max);
            }
            float2 factor;
            FOR_UNROLL (short i = 0; i < 2; i++) {
                factor[i] = metal::fast::exp2(max_score[i] - new_max[i]);
                max_score[i] = new_max[i];
                sum_score[i] *= factor[i];
            }
            FOR_UNROLL (short ik = 0; ik < TK; ik++) {
                Frag::row_reduce<ds4nax::SumOp>(Sfrag[ik],
                                                (thread float *)&sum_score);
            }
            FOR_UNROLL (short id = 0; id < TD; id++) {
                Frag::row_bin_op<ds4nax::MulOp>(Ofrag[id],
                                                (thread float *)&factor);
            }

            simdgroup_barrier(mem_flags::mem_none);

            /* O += P @ V. */
            FOR_UNROLL (short id = 0; id < TD; id += 2) {
                if (id == 4) {
                    threadgroup_barrier(mem_flags::mem_none);
                }
                FOR_UNROLL (short ik = 0; ik < TK; ik++) {
                    frag_h V0;
                    frag_h V1;
                    if (tail_k) {
                        Frag::load_rows(V0, Vw, kv_ld, s_len - col_in_seg0,
                                        ik * 16, id * 16);
                        Frag::load_rows(V1, Vw, kv_ld, s_len - col_in_seg0,
                                        ik * 16, id * 16 + 16);
                    } else {
                        Frag::load(V0, Vw, kv_ld, ik * 16, id * 16);
                        Frag::load(V1, Vw, kv_ld, ik * 16, id * 16 + 16);
                    }
                    Frag::mma(Ofrag[id], Ofrag[id + 1], Sfrag[ik], V0, V1,
                              metal::bool_constant<false>{});
                }
            }

            Kw += BK * kv_ld;
            Vw += BK * kv_ld;
        }
    }

    threadgroup_barrier(mem_flags::mem_none);

    /* Normalize and apply the learned per-head softplus gate. */
    float2 rcp;
    FOR_UNROLL (short i = 0; i < 2; i++) {
        const int t = tile_row0 + sm + i * Frag::kElemRowsJump;
        float gate_scale = 1.0f;
        if (t < (int)args.n_tokens) {
            const float gv = gate[(uint64_t)t * args.n_head + head];
            gate_scale = gv > 20.0f ? gv : metal::log(1.0f + metal::exp(gv));
        }
        rcp[i] = sum_score[i] > 0.0f ? gate_scale / sum_score[i] : 0.0f;
    }
    FOR_UNROLL (short id = 0; id < TD; id++) {
        Frag::row_bin_op<ds4nax::MulOp>(Ofrag[id], (thread float *)&rcp);
    }

    device float *Ow = out + (uint64_t)tile_row0 * q_ld +
                       (uint64_t)head * args.head_dim;
    FOR_UNROLL (short id = 0; id < TD; id++) {
        Frag::store_rows(Ofrag[id], Ow, q_ld, rows_lim, 0, id * 16);
    }
}
#endif
