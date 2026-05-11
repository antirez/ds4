struct ds4_metal_args_dsv4_topk_mask {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexer_weighted_sum {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
};

struct ds4_metal_args_dsv4_softmax_pool {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexed_attention {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_indexer_scores_fused {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_router_select_one {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
};

struct ds4_metal_args_dsv4_directional_steering_project {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
};

// Optional directional steering projection.
//
// Each threadgroup owns one 4096-wide token row, computes
// dot(row, direction[layer]), then subtracts scale * direction * dot in-place.
// Positive scales remove a concept direction; negative scales amplify it.  The
// kernel is not used unless a steering file and nonzero scale are provided.
kernel void kernel_dsv4_directional_steering_project_f32(
        constant ds4_metal_args_dsv4_directional_steering_project & args,
        device float *x,
        device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.rows || args.width == 0) return;

    device float *xr = x + (uint64_t)row * args.width;
    device const float *dir = directions + (uint64_t)args.layer * args.width;
    const uint nth = args.n_threads;

    float sum = 0.0f;
    for (uint i = tid; i < args.width; i += nth) {
        sum += xr[i] * dir[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float coeff = args.scale * scratch[0];
    for (uint i = tid; i < args.width; i += nth) {
        xr[i] -= coeff * dir[i];
    }
}

// Decode-only DS4 ratio-4 indexer score builder.  One threadgroup owns one
// compressed row for the current token, stages that 128-wide row once, then
// walks the 64 indexer heads in four-head groups.  This avoids materializing the
// intermediate [compressed rows x heads] score matrix used by the generic
// matvec + weighted-sum path.
kernel void kernel_dsv4_indexer_score_one_direct(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_comp || args.n_head != 64u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;        // [128]
    threadgroup float *psum = ktg + 128u;   // [4]

    if (tid < 128u) {
        device const float *krow = (device const float *)(index_comp +
            (uint64_t)row * args.index_row_stride);
        ktg[tid] = krow[tid];
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 64u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * args.q_head_stride);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            device const float *w = (device const float *)weights;
            psum[sg] = max(s, 0.0f) * (w[head] * args.scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device float *dst = (device float *)scores;
        dst[row] = acc;
    }
}

// Decode router post-processing for one token. The selected expert ids are
// already known; this gathers their probabilities, normalizes by the selected
// sum, clamps the denominator like the reference path, and applies DS4's 1.5
// expert-weight scale in one tiny dispatch.
kernel void kernel_dsv4_router_weights_one(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 6) return;

    device const float *p = (device const float *)probs;
    device const int   *s = (device const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, 6.103515625e-5f);

    device float *w = (device float *)weights;
    w[tid] = p[s[tid]] / sum * 1.5f;
}

// Decode router selection for one token after the existing
// sqrt(softplus(logit)) probability kernel has run. Bias affects only top-k
// selection. Route-weight normalization deliberately stays in the old one-token
// kernel: even tiny denominator-order changes here are amplified by 43 MoE
// layers, so this kernel only replaces the selection work.
kernel void kernel_dsv4_router_finalize_one(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256) return;

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Fills the dense compressed-attention mask with -inf. The selected top-k rows
// are enabled by kernel_dsv4_topk_mask_scatter in a second ordered dispatch.
kernel void kernel_dsv4_topk_mask(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    (void)topk;
    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = -INFINITY;
}

// Enables the selected compressed rows in the dense mask. This replaces the
// old O(n_comp * n_tokens * top_k) membership test with O(top_k * n_tokens)
// writes while preserving exactly the same 0/-inf mask consumed by attention.
kernel void kernel_dsv4_topk_mask_scatter(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne00 * args.ne01;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ik = gid % args.ne00;
    const int64_t it = gid / args.ne00;
    const int32_t idx = *((device const int32_t *) (topk + ik*args.nb00 + it*args.nb01));
    if (idx >= 0 && (int64_t)idx < args.ne0) {
        *((device float *) (dst + (int64_t)idx*args.nb0 + it*args.nb1)) = 0.0f;
    }
}

// Sorts each token's selected compressed rows by row id. The indexer selects by
// score, but attention scans compressed K/V in cache order in the dense graph.
// Sorting preserves that order while still letting the indexed attention kernel
// touch only the selected rows.
kernel void kernel_dsv4_sort_i32_rows_asc(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * src,
        device       char * dst,
        threadgroup int32_t * row_tmp [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint top_k = (uint)args.ne00;
    if (row >= (uint)args.ne01 || tid >= top_k) {
        return;
    }

    row_tmp[tid] = *((device const int32_t *) (src + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= top_k; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid && other < top_k) {
                const int32_t a = row_tmp[tid];
                const int32_t b = row_tmp[other];
                const bool up = (tid & k) == 0;
                if ((up && a > b) || (!up && a < b)) {
                    row_tmp[tid] = b;
                    row_tmp[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    *((device int32_t *) (dst + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01)) = row_tmp[tid];
}

static inline void dsv4_attend_f32_row_as_f16(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    device const float4 *kv4 = (device const float4 *)(kv + (uint64_t)row * row_stride);
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16(
        threadgroup const float4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16_at(
        threadgroup const float4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_f32_row_as_f16(kv4 + row_in_tg * 128u,
                                      q0, q1, q2, q3,
                                      scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
}

static inline void dsv4_attend_sink(
        float score,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;
    M = new_m;
}

// DS4 ratio-4 indexed mixed attention. It replaces the dense top-k mask path:
// the threadgroup covers one token and eight heads. Top-k rows and local raw
// rows are the same for all heads of a token, so K/V is staged once in
// threadgroup memory and reused by the eight simdgroups. It keeps the DS4 F16
// attention rounding by casting Q/K/V to half before the dot/value update.
kernel void kernel_dsv4_indexed_mixed_attention_heads8(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            const uint logical = pos - first_raw_pos;
            const uint row = (args.raw_start + logical) % args.raw_cap;
            device const float4 *src = (device const float4 *)(raw_kv +
                (uint64_t)row * args.raw_row_stride);
            if (tid < 128) kv_shared[tid] = src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_f32_row_as_f16(kv_shared,
                                              q0, q1, q2, q3,
                                              args.scale,
                                              lane,
                                              M, S,
                                              o0, o1, o2, o3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    for (uint i = 0; i < args.top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) {
            continue;
        }
        if ((uint)idx >= visible) {
            break;
        }
        device const float4 *src = (device const float4 *)(comp_kv +
            (uint64_t)(uint)idx * args.comp_row_stride);
        if (tid < 128) kv_shared[tid] = src[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dsv4_attend_shared_f32_row_as_f16(kv_shared,
                                          q0, q1, q2, q3,
                                          args.scale,
                                          lane,
                                          M, S,
                                          o0, o1, o2, o3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

// Decode specialization of kernel_dsv4_indexed_mixed_attention_heads8.
// Generation attends one token at a time, so the ratio-4 indexed path spends a
// visible amount of time repeatedly staging the same K/V row for the eight
// heads in a group. This diagnostic variant stages four selected rows at once.
kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb4(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos0 = first; pos0 <= last; pos0 += 4u) {
            const uint n_rows = min(4u, last - pos0 + 1u);
            for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                const uint r = off >> 7;
                const uint c = off & 127u;
                const uint logical = pos0 + r - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                kv_shared[off] = src[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_f32_row_as_f16_at(kv_shared,
                                                     r,
                                                     q0, q1, q2, q3,
                                                     args.scale,
                                                     lane,
                                                     M, S,
                                                     o0, o1, o2, o3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    bool stop = false;
    for (uint i = 0; i < args.top_k && !stop; i += 4u) {
        uint rows[4];
        uint n_rows = 0;
        for (uint j = 0; j < 4u && i + j < args.top_k; j++) {
            const int32_t idx = row_topk[i + j];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                stop = true;
                break;
            }
            rows[n_rows++] = (uint)idx;
        }
        if (n_rows == 0) {
            continue;
        }
        for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
            const uint r = off >> 7;
            const uint c = off & 127u;
            device const float4 *src = (device const float4 *)(comp_kv +
                (uint64_t)rows[r] * args.comp_row_stride);
            kv_shared[off] = src[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_f32_row_as_f16_at(kv_shared,
                                                 r,
                                                 q0, q1, q2, q3,
                                                 args.scale,
                                                 lane,
                                                 M, S,
                                                 o0, o1, o2, o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

// Decode specialization of kernel_dsv4_indexed_mixed_attention_heads8.
// Generation attends one token at a time, so the ratio-4 indexed path spends a
// visible amount of time repeatedly staging the same K/V row for the eight
// heads in a group. This variant stages sixteen selected rows at once and then
// consumes them sequentially, preserving the row order and online softmax math
// while cutting threadgroup barriers in the long top-k scan.
kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos0 = first; pos0 <= last; pos0 += 16u) {
            const uint n_rows = min(16u, last - pos0 + 1u);
            for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                const uint r = off >> 7;
                const uint c = off & 127u;
                const uint logical = pos0 + r - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                kv_shared[off] = src[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_f32_row_as_f16_at(kv_shared,
                                                     r,
                                                     q0, q1, q2, q3,
                                                     args.scale,
                                                     lane,
                                                     M, S,
                                                     o0, o1, o2, o3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    bool stop = false;
    for (uint i = 0; i < args.top_k && !stop; i += 16u) {
        uint rows[16];
        uint n_rows = 0;
        for (uint j = 0; j < 16u && i + j < args.top_k; j++) {
            const int32_t idx = row_topk[i + j];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                stop = true;
                break;
            }
            rows[n_rows++] = (uint)idx;
        }
        if (n_rows == 0) {
            continue;
        }
        for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
            const uint r = off >> 7;
            const uint c = off & 127u;
            device const float4 *src = (device const float4 *)(comp_kv +
                (uint64_t)rows[r] * args.comp_row_stride);
            kv_shared[off] = src[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_f32_row_as_f16_at(kv_shared,
                                                 r,
                                                 q0, q1, q2, q3,
                                                 args.scale,
                                                 lane,
                                                 M, S,
                                                 o0, o1, o2, o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

static inline float dsv4_indexer_dot128_shared_q(
        float4 c0,
        float4 c1,
        float4 c2,
        float4 c3,
        threadgroup const float4 *q4,
        ushort lane) {
    float sum = 0.0f;
    if (lane < 8) {
        const ushort ib = lane >> 1;
        const ushort il = lane & 1;
        const ushort base = ib*8 + il*4;
        sum += dot(c0, q4[base + 0]);
        sum += dot(c1, q4[base + 1]);
        sum += dot(c2, q4[base + 2]);
        sum += dot(c3, q4[base + 3]);
    }
    return simd_sum(sum);
}

// Tiled prefill score builder for the sparse-compressed attention indexer.
//
// The kernel covers an 8-token by 32-compressed-row rectangle: K is copied into
// threadgroup memory once, then reused for all 64 indexer heads, while simdgroup
// matrix multiply computes each 8x8 score subtile.
//
// It still writes the exact score matrix consumed by top-k:
//
//     score[t,c] = sum_h relu(dot(Q[t,h], K[c])) * W[t,h] * scale
//
// Causal masking is applied on store so invisible compressed rows become -inf.
kernel void kernel_dsv4_indexer_scores_tiled_f32(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup float *qtg = shared;             // [8][128]
    threadgroup float *ktg = qtg + TM*D;         // [32][128]
    threadgroup float *dot = ktg + TN*D;         // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        float v = 0.0f;
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = row[d];
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            float v = 0.0f;
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = qrow[d];
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_dsv4_indexer_scores_tiled(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    // Q/K are staged as half but the dot accumulator and final score remain
    // float. This is the one intentional precision tradeoff in the indexer:
    // the indexer only ranks compressed rows for top-k selection, and long
    // context profiling shows this score matrix dominates the prefill slope.
    threadgroup half *qtg = (threadgroup half *)shared; // [8][128]
    threadgroup half *ktg = qtg + TM*D;                 // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    // Stage compressed index rows once. Edge columns are zeroed so the matrix
    // loads below can stay regular; guarded stores discard them.
    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(row[d]);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        // Stage Q for the eight-token tile. Each 8x8 matrix load below reads a
        // contiguous depth block from this layout.
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

// Collapses per-head indexer scores into one score per compressed row using the
// learned head weights. Negative head scores are clipped exactly as DS4 expects.
kernel void kernel_dsv4_indexer_weighted_sum(
        constant ds4_metal_args_dsv4_indexer_weighted_sum & args,
        device const char * scores,
        device const char * weights,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    float acc = 0.0f;
    for (int64_t ih = 0; ih < args.ne02; ++ih) {
        const float s = *((device const float *) (scores  + ic*args.nb00 + it*args.nb01 + ih*args.nb02));
        const float w = *((device const float *) (weights + ih*args.nb10 + it*args.nb11));
        acc += max(s, 0.0f) * (w * args.scale);
    }

    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = acc;
}

// Fused softmax-weighted pooling of compressed KV rows. It is used when several
// compressor rows are present; the one-row case deliberately follows the
// unfused softmax/mul/sum graph in Objective-C to keep identical reductions.
kernel void kernel_dsv4_softmax_pool(
        constant ds4_metal_args_dsv4_softmax_pool & args,
        device const char * kv,
        device const char * score,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t id = gid % args.ne0;
    const int64_t ic = gid / args.ne0;

    float max_s = -INFINITY;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        max_s = max(max_s, s);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        const float w = exp(s - max_s);
        const float v = *((device const float *) (kv + ir*args.nb00 + id*args.nb01 + ic*args.nb02));
        sum += w;
        acc += v*w;
    }

    *((device float *) (dst + id*args.nb0 + ic*args.nb1)) = acc/sum;
}
