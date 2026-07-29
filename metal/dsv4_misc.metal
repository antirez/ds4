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

struct ds4_metal_args_dsv4_softmax_pool_ratio4_direct {
    int64_t  n_rows;
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t pad;
};

struct ds4_metal_args_dsv4_compressor_score_ape {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_tokens;
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
    uint32_t comp_kv_f16;
    uint32_t pad0;
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

struct ds4_metal_args_glm_router_select_one {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t pad0;
};

struct ds4_metal_args_glm_kv_lora_rms_norm {
    uint32_t n_tokens;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    float    eps;
};

struct ds4_metal_args_glm_k_b_project {
    uint32_t n_tokens;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t n_head;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_glm_build_kv_cache {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

struct ds4_metal_args_glm_store_compact_kv {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_rope;
    uint32_t cache_f16;
    uint32_t pad1;
};

struct ds4_metal_args_glm_qkv_norm_store_compact_kv {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t q_n;
    uint32_t q_n4;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t kv_lora_n4;
    uint32_t qk_rope;
    uint32_t cache_f16;
    float    eps;
    uint32_t pad0;
};

struct ds4_metal_args_glm_store_indexer_k {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    float    pad1;
};

struct ds4_metal_args_glm_attention_full {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_len;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t qk_dim;
    uint32_t value_dim;
    uint32_t pad0;
    uint32_t cache_f16;
    uint32_t pad1;
    uint32_t pad2;
    float    scale;
};

struct ds4_metal_args_glm_fill_selected_range {
    uint32_t n_selected;
};

struct ds4_metal_args_glm_fill_selected_range_batch {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_selected;
    uint32_t pad_row;
};

struct ds4_metal_args_glm_indexer_rope_tail {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t rot_offset;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

struct ds4_metal_args_glm_indexer_score_one {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t cache_f16;
    float    scale;
};

struct ds4_metal_args_glm_indexer_scores_batch {
    uint32_t n_rows;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t cache_f16;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t score_token_stride;
    float    scale;
};

struct ds4_metal_args_glm_qk_lowrank {
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_glm_qk_lowrank_batch {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    /* First head this dispatch computes: under tensor-parallel head split
     * each rank covers a contiguous half of the heads; buffers and weights
     * keep full-model layout and are indexed by absolute head. */
    uint32_t head_base;
};

struct ds4_metal_args_glm_attention_indexed_decode {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
};

struct ds4_metal_args_glm_attention_indexed_decode_split {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t block_rows;
    uint32_t n_blocks;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
};

struct ds4_metal_args_glm_attention_indexed_batch {
    uint32_t n_tokens;
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t value_type;
    uint32_t pos0;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t head_base;
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

static inline float ds4_glm_router_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = exp(x);
        return e / (1.0f + e);
    }
}

static inline bool ds4_glm_router_better(
        threadgroup const float *scores,
        int32_t                  a,
        int32_t                  b) {
    const float sa = scores[(uint)a];
    const float sb = scores[(uint)b];
    return sa > sb || (sa == sb && a < b);
}

static float glm_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static void glm_rope_yarn(
        float theta_extrap,
        float freq_scale,
        float corr_dims[2],
        int   i0,
        float ext_factor,
        float mscale,
        thread float *cos_theta,
        thread float *sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = glm_rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

static float glm_rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void glm_rope_yarn_corr_dims(
        int   n_dims,
        int   n_ctx_orig,
        float freq_base,
        float beta_fast,
        float beta_slow,
        float dims[2]) {
    dims[0] = max(0.0f,
                  floor(glm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f,
                  ceil(glm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

kernel void kernel_glm_kv_lora_rms_norm(
        constant ds4_metal_args_glm_kv_lora_rms_norm & args,
        device const char *src,
        device const char *weight,
        device       char *dst,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    if (row >= args.n_tokens) return;

    const uint tid = tid_u;
    const uint nth = ntg_u.x;
    device const float *x = (device const float *)(src + (uint64_t)row * args.kv_raw_dim * sizeof(float));
    device const float *w = (device const float *)weight;
    device float *out = (device float *)(dst + (uint64_t)row * args.kv_lora_dim * sizeof(float));

    float ss = 0.0f;
    for (uint i = tid; i < args.kv_lora_dim; i += nth) {
        const float v = x[i];
        ss += v * v;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(scratch[0] / (float)args.kv_lora_dim + args.eps);
    for (uint i = tid; i < args.kv_lora_dim; i += nth) {
        out[i] = x[i] * inv * w[i];
    }
}

static inline float glm_quant_weight_at(
        uint weight_type,
        device const char *row,
        uint col);

kernel void kernel_glm_k_b_project_q8_0(
        constant ds4_metal_args_glm_k_b_project & args,
        device const char *weight,
        device const char *kv_norm,
        device       char *dst,
        threadgroup float *kv_scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = (uint)ntg_u.x * (uint)ntg_u.y;
    device const float *kv =
        (device const float *)(kv_norm + (uint64_t)token * args.kv_lora_dim * sizeof(float));
    device float *out =
        (device float *)(dst +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        kv_scratch[j] = kv[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint block = (uint)sgitg;
    const uint q = (block << 5) + (uint)tiisg;
    if (q < args.qk_nope) {
        float acc = 0.0f;
        for (uint j = 0; j < args.kv_lora_dim; j++) {
            device const char *row =
                weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
            acc += glm_quant_weight_at(args.weight_type, row, q) * kv_scratch[j];
        }
        out[q] = acc;
    }
}

kernel void kernel_glm_store_compact_kv(
        constant ds4_metal_args_glm_store_compact_kv & args,
        device const char *kv_norm,
        device const char *kv_raw,
        device       char *kv_lora_cache,
        device       char *k_rope_cache,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint part = tgpig.y;
    if (token >= args.n_tokens || part > 1u) return;

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;

    const uint nth = ntg_u.x;
    if (part == 0) {
        device const float *src =
            (device const float *)(kv_norm +
                (uint64_t)token * args.kv_lora_dim * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(kv_lora_cache +
                    (uint64_t)pos * args.kv_lora_dim * sizeof(half));
            for (uint i = tid; i < args.kv_lora_dim; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(kv_lora_cache +
                    (uint64_t)pos * args.kv_lora_dim * sizeof(float));
            for (uint i = tid; i < args.kv_lora_dim; i += nth) {
                dst[i] = src[i];
            }
        }
    } else {
        device const float *src =
            (device const float *)(kv_raw +
                ((uint64_t)token * args.kv_raw_dim + args.kv_lora_dim) * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(half));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(float));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = src[i];
            }
        }
    }
}

kernel void kernel_glm_qkv_norm_store_compact_kv(
        constant ds4_metal_args_glm_qkv_norm_store_compact_kv & args,
        device const char *q_src,
        device const char *q_weight,
        device       char *q_dst,
        device const char *kv_raw,
        device const char *kv_weight,
        device       char *kv_lora_cache,
        device       char *k_rope_cache,
        threadgroup float *shmem_f32 [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint part = tgpig.y;
    if (token >= args.n_tokens || part > 2u) return;

    const uint nth = ntg_u.x;
    if (part == 2u) {
        const uint pos = args.pos0 + token;
        if (pos >= args.cache_cap) return;
        device const float *src =
            (device const float *)(kv_raw +
                ((uint64_t)token * args.kv_raw_dim + args.kv_lora_dim) * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(half));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(float));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = src[i];
            }
        }
        return;
    }

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const bool kv_task = part != 0u;
    const uint n = kv_task ? args.kv_lora_dim : args.q_n;
    const uint n4 = kv_task ? args.kv_lora_n4 : args.q_n4;
    device const float4 *x =
        kv_task
            ? (device const float4 *)(kv_raw +
                (uint64_t)token * args.kv_raw_dim * sizeof(float))
            : (device const float4 *)(q_src +
                (uint64_t)token * args.q_n * sizeof(float));
    device const float4 *w =
        kv_task ? (device const float4 *)kv_weight
                : (device const float4 *)q_weight;

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += nth) {
        const float4 v = x[i];
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

#ifdef DS4_METAL_NORM_RSQRT_DISABLE
    const float scale = 1.0f / sqrt(sumf / float(n) + args.eps);
#else
    const float scale = rsqrt(sumf / float(n) + args.eps);
#endif

    if (!kv_task) {
        device float4 *y =
            (device float4 *)(q_dst +
                (uint64_t)token * args.q_n * sizeof(float));
        for (uint i = tid; i < n4; i += nth) {
            y[i] = (x[i] * scale) * w[i];
        }
        return;
    }

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;
    device const float *x1 =
        (device const float *)(kv_raw +
            (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *w1 = (device const float *)kv_weight;
    if (args.cache_f16 != 0u) {
        device half *dst =
            (device half *)(kv_lora_cache +
                (uint64_t)pos * args.kv_lora_dim * sizeof(half));
        for (uint i = tid; i < args.kv_lora_dim; i += nth) {
            dst[i] = (half)((x1[i] * scale) * w1[i]);
        }
    } else {
        device float *dst =
            (device float *)(kv_lora_cache +
                (uint64_t)pos * args.kv_lora_dim * sizeof(float));
        for (uint i = tid; i < args.kv_lora_dim; i += nth) {
            dst[i] = (x1[i] * scale) * w1[i];
        }
    }
}

kernel void kernel_glm_store_indexer_k(
        constant ds4_metal_args_glm_store_indexer_k & args,
        device const char *raw_k,
        device const char *weight,
        device const char *bias,
        device       char *indexer_key_cache,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    if (token >= args.n_tokens) return;

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;

    const uint nth = ntg_u.x;
    const uint head_dim = args.head_dim;
    const uint rot_dim = args.rot_dim;

    device const float *src =
        (device const float *)(raw_k + (uint64_t)token * head_dim * sizeof(float));
    device const float *w = (device const float *)weight;
    device const float *b = (device const float *)bias;

    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += nth) {
        sum += src[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = scratch[0] / (float)head_dim;

    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += nth) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(scratch[0] / (float)head_dim + args.eps);

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)rot_dim,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)rot_dim;

    if (args.cache_f16 != 0u) {
        device half *dst =
            (device half *)(indexer_key_cache +
                (uint64_t)pos * head_dim * sizeof(half));
        for (uint i = tid; i < head_dim; i += nth) {
            if (i < rot_dim) {
                if ((i & 1u) != 0u) continue;
                const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
                const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
                const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
                float cos_theta;
                float sin_theta;
                glm_rope_yarn(theta,
                              args.freq_scale,
                              corr_dims,
                              (int)rel_i0,
                              args.ext_factor,
                              args.attn_factor,
                              &cos_theta,
                              &sin_theta);
                const float x0 = (src[i] - mean) * inv * w[i] + b[i];
                const uint j = i + 1u;
                const float x1 = (src[j] - mean) * inv * w[j] + b[j];
                dst[i] = (half)(x0 * cos_theta - x1 * sin_theta);
                dst[j] = (half)(x0 * sin_theta + x1 * cos_theta);
            } else if (i >= rot_dim) {
                const float x = (src[i] - mean) * inv * w[i] + b[i];
                dst[i] = (half)x;
            }
        }
    } else {
        device float *dst =
            (device float *)(indexer_key_cache +
                (uint64_t)pos * head_dim * sizeof(float));
        for (uint i = tid; i < head_dim; i += nth) {
            if (i < rot_dim) {
                if ((i & 1u) != 0u) continue;
                const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
                const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
                const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
                float cos_theta;
                float sin_theta;
                glm_rope_yarn(theta,
                              args.freq_scale,
                              corr_dims,
                              (int)rel_i0,
                              args.ext_factor,
                              args.attn_factor,
                              &cos_theta,
                              &sin_theta);
                const float x0 = (src[i] - mean) * inv * w[i] + b[i];
                const uint j = i + 1u;
                const float x1 = (src[j] - mean) * inv * w[j] + b[j];
                dst[i] = x0 * cos_theta - x1 * sin_theta;
                dst[j] = x0 * sin_theta + x1 * cos_theta;
            } else if (i >= rot_dim) {
                const float x = (src[i] - mean) * inv * w[i] + b[i];
                dst[i] = x;
            }
        }
    }
}

static inline void glm_dense_cache_store_f32_or_f16(
        device char *base,
        uint64_t index,
        uint cache_f16,
        float x) {
    if (cache_f16 != 0u) {
        ((device half *)base)[index] = (half)x;
    } else {
        ((device float *)base)[index] = x;
    }
}

static inline float glm_dense_cache_load_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        return (float)((device const half *)base)[index];
    }
    return ((device const float *)base)[index];
}

static inline float4 glm_dense_cache_load4_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        device const half *h = (device const half *)base;
        return float4((float)h[index + 0u],
                      (float)h[index + 1u],
                      (float)h[index + 2u],
                      (float)h[index + 3u]);
    }
    return ((device const float4 *)base)[index >> 2u];
}

kernel void kernel_glm_build_kv_cache(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *kn =
        (device const float *)(k_nope +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
    device const float *val =
        (device const float *)(value +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;

    for (uint i = tid; i < args.qk_nope; i += nth) {
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, kn[i]);
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)args.qk_rope;
    for (uint r = tid * 2u; r < args.qk_rope; r += nth * 2u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        glm_dense_cache_store_f32_or_f16(key_cache,
                                         kbase + dst0,
                                         args.cache_f16,
                                         x0 * cos_theta - x1 * sin_theta);
        glm_dense_cache_store_f32_or_f16(key_cache,
                                         kbase + dst0 + 1u,
                                         args.cache_f16,
                                         x0 * sin_theta + x1 * cos_theta);
    }

    for (uint i = tid; i < args.value_dim; i += nth) {
        glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, val[i]);
    }
}

kernel void kernel_glm_build_kv_cache_decode_group4(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint group_head0 = tgpig.y * 4u;
    if (token >= args.n_tokens || group_head0 >= args.n_head) return;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    const uint lane = tid & 63u;
    const uint slot = tid >> 6;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));

    const uint head = group_head0 + slot;
    if (slot < 4u && head < args.n_head) {
        device const float *kn =
            (device const float *)(k_nope +
                ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
        device const float *val =
            (device const float *)(value +
                ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
        const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;

        for (uint i = lane; i < args.qk_nope; i += 64u) {
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, kn[i]);
        }
        for (uint i = lane; i < args.value_dim; i += 64u) {
            glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, val[i]);
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)args.qk_rope;
    for (uint r = tid * 2u; r < args.qk_rope; r += 512u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        const float y0 = x0 * cos_theta - x1 * sin_theta;
        const float y1 = x0 * sin_theta + x1 * cos_theta;
        for (uint h = group_head0; h < min(group_head0 + 4u, args.n_head); h++) {
            const uint64_t kbase = ((uint64_t)pos * args.n_head + h) * qk_dim;
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0, args.cache_f16, y0);
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0 + 1u, args.cache_f16, y1);
        }
    }
}

kernel void kernel_glm_build_kv_cache_flash(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        device       char *key_f16,
        device       char *value_f16,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *kn =
        (device const float *)(k_nope +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
    device const float *val =
        (device const float *)(value +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;
    device half *kdst_f16 =
        (device half *)(key_f16 +
            ((uint64_t)head * args.n_tokens + token) * qk_dim * sizeof(half));
    device half *vdst_f16 =
        (device half *)(value_f16 +
            ((uint64_t)head * args.n_tokens + token) * args.value_dim * sizeof(half));

    for (uint i = tid; i < args.qk_nope; i += nth) {
        const float x = kn[i];
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, x);
        kdst_f16[i] = (half)x;
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)args.qk_rope;
    for (uint r = tid * 2u; r < args.qk_rope; r += nth * 2u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        const float y0 = x0 * cos_theta - x1 * sin_theta;
        const float y1 = x0 * sin_theta + x1 * cos_theta;
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0, args.cache_f16, y0);
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0 + 1u, args.cache_f16, y1);
        kdst_f16[dst0] = (half)y0;
        kdst_f16[dst0 + 1u] = (half)y1;
    }

    for (uint i = tid; i < args.value_dim; i += nth) {
        const float x = val[i];
        glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, x);
        vdst_f16[i] = (half)x;
    }
}

kernel void kernel_glm_attention_full(
        constant ds4_metal_args_glm_attention_full & args,
        device const char *q,
        device const char *key_cache,
        device const char *value_cache,
        device       char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk4 = args.qk_dim / 4u;
    const uint visible = min(args.cache_len, args.pos0 + token + 1u);
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;

    device const float4 *q4 = (device const float4 *)(q +
        ((uint64_t)token * args.n_head + head) * args.qk_dim * sizeof(float));

    if (args.pad0 == 2u) {
        for (uint s = tid; s < visible; s += nth) {
            const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
            float dotv = 0.0f;
            for (uint i = 0; i < qk4; i++) {
                dotv += dot(q4[i],
                            glm_dense_cache_load4_f32_or_f16(key_cache,
                                                             kbase + 4u * (uint64_t)i,
                                                             args.cache_f16));
            }
            scores[s] = dotv * args.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float max_score = -INFINITY;
            for (uint s = 0; s < visible; s++) {
                max_score = max(max_score, scores[s]);
            }
            float sum = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const float w = exp(scores[s] - max_score);
                scores[s] = w;
                sum += w;
            }
            red[0] = max(sum, 1.0e-20f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float denom = red[0];
        device float *out = (device float *)(heads +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            float acc = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
                acc += scores[s] *
                       glm_dense_cache_load_f32_or_f16(value_cache,
                                                       vbase + d,
                                                       args.cache_f16);
            }
            out[d] = acc / denom;
        }
        return;
    }

    if (args.pad0 == 1u) {
        if (tid == 0u) {
            float max_score = -INFINITY;
            for (uint s = 0; s < visible; s++) {
                const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
                float dotv = 0.0f;
                for (uint i = 0; i < qk4; i++) {
                    dotv += dot(q4[i],
                                glm_dense_cache_load4_f32_or_f16(key_cache,
                                                                 kbase + 4u * (uint64_t)i,
                                                                 args.cache_f16));
                }
                const float score = dotv * args.scale;
                scores[s] = score;
                max_score = max(max_score, score);
            }
            float sum = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const float w = exp(scores[s] - max_score);
                scores[s] = w;
                sum += w;
            }
            red[0] = max(sum, 1.0e-20f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float denom = red[0];
        device float *out = (device float *)(heads +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            float acc = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
                acc += scores[s] *
                       glm_dense_cache_load_f32_or_f16(value_cache,
                                                       vbase + d,
                                                       args.cache_f16);
            }
            out[d] = acc / denom;
        }
        return;
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < visible; s += nth) {
        const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
        float dotv = 0.0f;
        for (uint i = 0; i < qk4; i++) {
            dotv += dot(q4[i],
                        glm_dense_cache_load4_f32_or_f16(key_cache,
                                                         kbase + 4u * (uint64_t)i,
                                                         args.cache_f16));
        }
        const float score = dotv * args.scale;
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];

    float local_sum = 0.0f;
    for (uint s = tid; s < visible; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out = (device float *)(heads +
        ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    for (uint d = tid; d < args.value_dim; d += nth) {
        float acc = 0.0f;
        for (uint s = 0; s < visible; s++) {
            const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
            acc += scores[s] *
                   glm_dense_cache_load_f32_or_f16(value_cache,
                                                   vbase + d,
                                                   args.cache_f16);
        }
        out[d] = acc / denom;
    }
}

kernel void kernel_glm_fill_selected_range(
        constant ds4_metal_args_glm_fill_selected_range & args,
        device uint32_t *selected,
        uint gid [[thread_position_in_grid]]) {
    if (gid < args.n_selected) selected[gid] = gid;
}

kernel void kernel_glm_fill_selected_range_batch(
        constant ds4_metal_args_glm_fill_selected_range_batch & args,
        device uint32_t *selected,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.n_selected;
    if (gid >= total || args.n_selected == 0u) return;
    const uint token = gid / args.n_selected;
    const uint slot = gid - token * args.n_selected;
    const uint visible = args.pos0 + token + 1u;
    selected[gid] = slot < visible ? slot : args.pad_row;
}

kernel void kernel_glm_indexer_rope_tail_f32(
        constant ds4_metal_args_glm_indexer_rope_tail & args,
        device char *x,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    if (args.rot_dim == 0u || args.rot_offset > args.head_dim ||
        args.rot_dim > args.head_dim - args.rot_offset || (args.rot_dim & 1u) != 0u) return;

    const uint nth = ntg_u.x;
    const uint pos = args.pos0 + token;
    device float *row =
        (device float *)(x +
            ((uint64_t)token * args.n_head + head) * args.head_dim * sizeof(float));
    row += args.rot_offset;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.rot_dim,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)args.rot_dim;
    for (uint i = tid * 2u; i < args.rot_dim; i += nth * 2u) {
        const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)rel_i0,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint j = i + 1u;
        const float x0 = row[i];
        const float x1 = row[j];
        row[i] = x0 * cos_theta - x1 * sin_theta;
        row[j] = x0 * sin_theta + x1 * cos_theta;
    }
}

static inline float glm_cache_load_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        return (float)((device const half *)base)[index];
    }
    return ((device const float *)base)[index];
}

static inline float glm_cache_load_f16_only(
        device const char *base,
        uint64_t index) {
    return (float)((device const half *)base)[index];
}

static inline float2 glm_cache_load_rotated_rope_pair(
        device const char *base,
        uint64_t           rope_base,
        uint               r,
        uint               row,
        uint               qk_rope,
        uint               cache_f16,
        float              freq_base,
        float              freq_scale,
        float              ext_factor,
        float              attn_factor,
        float              corr0,
        float              corr1) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = theta_base * exp2(inv_ndims * (float)r * log2(freq_base));
#else
    const float theta = theta_base * pow(freq_base, inv_ndims * (float)r);
#endif
    float corr_dims[2] = {corr0, corr1};
    float cos_theta;
    float sin_theta;
    glm_rope_yarn(theta,
                  freq_scale,
                  corr_dims,
                  (int)r,
                  ext_factor,
                  attn_factor,
                  &cos_theta,
                  &sin_theta);
    const float x0 = glm_cache_load_f32_or_f16(base, rope_base + r, cache_f16);
    const float x1 = glm_cache_load_f32_or_f16(base, rope_base + r + 1u, cache_f16);
    return float2(x0 * cos_theta - x1 * sin_theta,
                  x0 * sin_theta + x1 * cos_theta);
}

static inline float2 glm_cache_load_rotated_rope_pair_f16_only(
        device const char *base,
        uint64_t           rope_base,
        uint               r,
        uint               row,
        uint               qk_rope,
        float              freq_base,
        float              freq_scale,
        float              ext_factor,
        float              attn_factor,
        float              corr0,
        float              corr1) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = theta_base * exp2(inv_ndims * (float)r * log2(freq_base));
#else
    const float theta = theta_base * pow(freq_base, inv_ndims * (float)r);
#endif
    float corr_dims[2] = {corr0, corr1};
    float cos_theta;
    float sin_theta;
    glm_rope_yarn(theta,
                  freq_scale,
                  corr_dims,
                  (int)r,
                  ext_factor,
                  attn_factor,
                  &cos_theta,
                  &sin_theta);
    const float x0 = glm_cache_load_f16_only(base, rope_base + r);
    const float x1 = glm_cache_load_f16_only(base, rope_base + r + 1u);
    return float2(x0 * cos_theta - x1 * sin_theta,
                  x0 * sin_theta + x1 * cos_theta);
}

static inline float glm_q8_0_weight_at(
        device const char *row,
        uint col) {
    const uint block = col >> 5;
    const uint qi = col & 31u;
    device const char *block_base = row + (uint64_t)block * 34u;
    const float d = (float)(*((device const half *)block_base));
    device const int8_t *qs = (device const int8_t *)(block_base + 2u);
    return d * (float)qs[qi];
}

static inline float glm_q8_0_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    const uint n_blocks = (n_cols + 31u) >> 5;
    for (uint block = 0; block < n_blocks; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        const uint count = min(32u, n_cols - base);
        for (uint qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static inline float glm_q8_0_dot_row_tg_f32_512(
        device const char *row,
        threadgroup const float *x) {
    float acc = 0.0f;
    for (uint block = 0; block < 16u; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static inline float glm_q8_0_dot_row_tg_f32_fast(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    if (n_cols == 512u) {
        return glm_q8_0_dot_row_tg_f32_512(row, x);
    }
    return glm_q8_0_dot_row_tg_f32(row, x, n_cols);
}

static inline float glm_q8_0_dot_row_dev_f32(
        device const char *row,
        device const float *x,
        uint n_cols) {
    float acc = 0.0f;
    const uint n_blocks = (n_cols + 31u) >> 5;
    for (uint block = 0; block < n_blocks; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        const uint count = min(32u, n_cols - base);
        for (uint qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

#define DS4_METAL_GGUF_Q4_0 2u
#define DS4_METAL_GGUF_Q8_0 8u
#define DS4_METAL_GGUF_Q4_K 12u

static inline uchar2 glm_q4_K_scale_min(int j, int k, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0x0f) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

static inline float glm_q4_0_weight_at(device const char *row, uint col) {
    const uint block = col >> 5;
    const uint qi = col & 31u;
    device const char *block_base = row + (uint64_t)block * 18u;
    const float d = (float)(*((device const half *)block_base));
    device const uchar *qs = (device const uchar *)(block_base + 2u);
    /* ggml Q4_0: elems 0..15 = low nibbles of qs[0..15], 16..31 = high. */
    const uchar packed = qs[qi & 15u];
    const uchar q = (qi < 16u) ? (packed & 0x0f) : (packed >> 4);
    return d * ((float)q - 8.0f);
}

static inline float glm_q4_K_weight_at(device const char *row, uint col) {
    const uint block = col >> 8u;
    const uint idx = col & 255u;
    device const char *block_base = row + (uint64_t)block * 144u;
    const float d = (float)(*((device const half *)(block_base + 0u)));
    const float dmin = (float)(*((device const half *)(block_base + 2u)));
    device const uchar *scales = (device const uchar *)(block_base + 4u);
    device const uchar *qs = (device const uchar *)(block_base + 16u);
    const uint group = idx >> 5u;
    const uint l = idx & 31u;
    const uchar2 sm = glm_q4_K_scale_min((int)group, 0, scales);
    const uint byte_off = (group >> 1u) * 32u + l;
    const uint shift = (group & 1u) * 4u;
    const uint q = ((uint)qs[byte_off] >> shift) & 0x0fu;
    return d * (float)sm.x * (float)q - dmin * (float)sm.y;
}

static inline float glm_quant_weight_at(
        uint weight_type,
        device const char *row,
        uint col) {
    if (weight_type == DS4_METAL_GGUF_Q4_0) return glm_q4_0_weight_at(row, col);
    if (weight_type == DS4_METAL_GGUF_Q4_K) return glm_q4_K_weight_at(row, col);
    return glm_q8_0_weight_at(row, col);
}

static inline float glm_q4_0_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_q4_0_weight_at(row, col) * x[col];
    }
    return acc;
}

static inline float glm_q4_K_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_q4_K_weight_at(row, col) * x[col];
    }
    return acc;
}

static inline float glm_quant_dot_row_tg_f32(
        uint weight_type,
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    if (weight_type == DS4_METAL_GGUF_Q4_0) return glm_q4_0_dot_row_tg_f32(row, x, n_cols);
    if (weight_type == DS4_METAL_GGUF_Q4_K) return glm_q4_K_dot_row_tg_f32(row, x, n_cols);
    return glm_q8_0_dot_row_tg_f32_fast(row, x, n_cols);
}

/* Per-lane Q4_K row dot: lane l covers elements (g*32 + l) of every
 * 32-group so the 144-byte superblocks are read with coalesced per-lane
 * bytes; callers simd_sum the result. x lives in threadgroup memory. */
static inline float glm_q4_K_dot_row_lane_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols,
        ushort lane) {
    float acc = 0.0f;
    const uint nblocks = n_cols >> 8u;
    for (uint b = 0; b < nblocks; b++) {
        device const char *block_base = row + (uint64_t)b * 144u;
        const float d = (float)(*((device const half *)(block_base + 0u)));
        const float dmin = (float)(*((device const half *)(block_base + 2u)));
        device const uchar *scales = (device const uchar *)(block_base + 4u);
        device const uchar *qs = (device const uchar *)(block_base + 16u);
        threadgroup const float *xb = x + (b << 8u);
        FOR_UNROLL (uint g = 0; g < 8u; g++) {
            const uchar2 sm = glm_q4_K_scale_min((int)g, 0, scales);
            const uint byte_off = (g >> 1u) * 32u + lane;
            const uint shift = (g & 1u) * 4u;
            const uint q = ((uint)qs[byte_off] >> shift) & 0x0fu;
            const float xv = xb[(g << 5u) + lane];
            acc += (d * (float)sm.x * (float)q - dmin * (float)sm.y) * xv;
        }
    }
    return acc;
}

static inline float glm_quant_dot_row_dev_f32(
        uint weight_type,
        device const char *row,
        device const float *x,
        uint n_cols) {
    if (weight_type == DS4_METAL_GGUF_Q8_0) return glm_q8_0_dot_row_dev_f32(row, x, n_cols);
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_quant_weight_at(weight_type, row, col) * x[col];
    }
    return acc;
}

kernel void kernel_glm_indexer_score_one(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint row = tgpig.x;
    if (row >= args.n_rows) return;
    const uint nth = ntg_u.x;
    float score = 0.0f;
    for (uint h = 0; h < args.n_head; h++) {
        float partial = 0.0f;
        device const float *qh =
            (device const float *)(q + (uint64_t)h * args.head_dim * sizeof(float));
        for (uint d = tid; d < args.head_dim; d += nth) {
            const float k = glm_cache_load_f32_or_f16(indexer_key_cache,
                                                      (uint64_t)row * args.head_dim + d,
                                                      args.cache_f16);
            partial += qh[d] * k;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            score += max(scratch[0] * args.scale, 0.0f) * weights[h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) scores[row] = score;
}

kernel void kernel_glm_indexer_score_one_direct(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_rows || args.n_head != 32u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;
    threadgroup float *psum = ktg + 128u;

    if (tid < 128u) {
        ktg[tid] = glm_cache_load_f32_or_f16(indexer_key_cache,
                                             (uint64_t)row * 128u + tid,
                                             args.cache_f16);
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 32u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * 128u * sizeof(float));
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            psum[sg] = max(s * args.scale, 0.0f) * weights[head];
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
        scores[row] = acc;
    }
}

kernel void kernel_glm_indexer_scores_batch(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= args.n_rows || token >= args.n_tokens) return;

    device float *dst = (device float *)(scores +
        (uint64_t)token * args.score_token_stride) + row;
    const uint visible = min(args.pos0 + token + 1u, args.n_rows);
    if (row >= visible) {
        if (tid == 0) *dst = -INFINITY;
        return;
    }

    const uint nth = ntg_u.x;
    float score = 0.0f;
    for (uint h = 0; h < args.n_head; h++) {
        float partial = 0.0f;
        device const float *qh = (device const float *)(q +
            (uint64_t)token * args.q_token_stride +
            (uint64_t)h     * args.q_head_stride);
        for (uint d = tid; d < args.head_dim; d += nth) {
            const float k = glm_cache_load_f32_or_f16(indexer_key_cache,
                                                      (uint64_t)row * args.head_dim + d,
                                                      args.cache_f16);
            partial += qh[d] * k;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token * args.weights_token_stride);
            score += max(scratch[0] * args.scale, 0.0f) * w[h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) *dst = score;
}

kernel void kernel_glm_indexer_scores_tiled_f32(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint row_base = tgpig.x * TN;
    const uint token_base = tgpig.y * TM;

    threadgroup float *qtg = shared;
    threadgroup float *ktg = qtg + TM*D;
    threadgroup float *dot = ktg + TN*D;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        min(args.pos0 + last_token, args.n_rows) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        float v = 0.0f;
        if (row < args.n_rows) {
            v = glm_cache_load_f32_or_f16(indexer_key_cache,
                                          (uint64_t)row * args.head_dim + d,
                                          args.cache_f16);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint tr = i / D;
            const uint d = i - tr*D;
            const uint token = token_base + tr;
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

        if (token0 < args.n_tokens && row0 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[token_row0*TN + col0];
            acc0 += max(s * args.scale, 0.0f) * w[head];
        }
        if (token1 < args.n_tokens && row1 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[token_row1*TN + col1];
            acc1 += max(s * args.scale, 0.0f) * w[head];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = min(args.pos0 + token0 + 1u, args.n_rows);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = min(args.pos0 + token1 + 1u, args.n_rows);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_glm_indexer_scores_tiled(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint row_base = tgpig.x * TN;
    const uint token_base = tgpig.y * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        min(args.pos0 + last_token, args.n_rows) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint tr = i / D;
            const uint d = i - tr*D;
            const uint token = token_base + tr;
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

        if (token0 < args.n_tokens && row0 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[token_row0*TN + col0];
            acc0 += max(s * args.scale, 0.0f) * w[head];
        }
        if (token1 < args.n_tokens && row1 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[token_row1*TN + col1];
            acc1 += max(s * args.scale, 0.0f) * w[head];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = min(args.pos0 + token0 + 1u, args.n_rows);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = min(args.pos0 + token1 + 1u, args.n_rows);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_glm_qk_lowrank_q8_0(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head) return;
    const uint nth = ntg_u.x;
    device const float *qh =
        (device const float *)(q + (uint64_t)head * args.qk_dim * sizeof(float));
    device float *out =
        (device float *)(qk_low + (uint64_t)head * args.kv_lora_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
        out[j] = glm_quant_dot_row_dev_f32(args.weight_type, row, qh, args.qk_nope);
    }
}

kernel void kernel_glm_qk_lowrank_q8_0_glm52(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint n_head = 64u;
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 192u;
    constexpr uint qk_dim = 256u;
    constexpr uint row_bytes = 204u;

    const uint head = tgpig.x;
    if (head >= n_head ||
        args.n_head != n_head ||
        args.kv_lora_dim != kv_lora_dim ||
        args.qk_nope != qk_nope ||
        args.qk_dim != qk_dim ||
        args.row_bytes != row_bytes ||
        args.weight_type != DS4_METAL_GGUF_Q8_0) {
        return;
    }
    const uint nth = ntg_u.x;
    device const float *qh =
        (device const float *)(q + (uint64_t)head * qk_dim * sizeof(float));
    for (uint d = tid; d < qk_nope; d += nth) {
        x[d] = qh[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(qk_low + (uint64_t)head * kv_lora_dim * sizeof(float));
    for (uint j = tid; j < kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc = 0.0f;
        for (uint block = 0; block < 6u; block++) {
            device const char *block_base = row + (uint64_t)block * 34u;
            const float d = (float)(*((device const half *)block_base));
            device const int8_t *qs = (device const int8_t *)(block_base + 2u);
            const uint base = block << 5;
            FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
                const uint col = base + qi;
                acc += d * (float)qs[qi] * x[col];
            }
        }
        out[j] = acc;
    }
}

// Coalesced GLM 5.2 decode qk-low: one simdgroup per pair of output rows,
// lanes split the 192-wide dot so the 204-byte Q8 rows are read with
// consecutive per-lane bytes. The thread-per-row variant above issues
// strided scalar byte loads from only 64 threadgroups and measures ~7.5x
// off the weight-bandwidth floor.
kernel void kernel_glm_qk_lowrank_q8_0_glm52_sg(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint n_head = 64u;
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 192u;
    constexpr uint qk_dim = 256u;
    constexpr uint NR = 2u;

    const uint head = tgpig.x;
    const uint wt = args.weight_type;
    if (head >= n_head ||
        args.n_head != n_head ||
        args.kv_lora_dim != kv_lora_dim ||
        args.qk_nope != qk_nope ||
        args.qk_dim != qk_dim ||
        !((wt == DS4_METAL_GGUF_Q8_0 && args.row_bytes == 204u) ||
          (wt == DS4_METAL_GGUF_Q4_0 && args.row_bytes == 108u))) {
        return;
    }
    const uint row_bytes = args.row_bytes;

    const uint nsg = ntg_u.y;
    const uint row0 = (tgpig.y * nsg + (uint)sgitg) * NR;
    if (row0 >= kv_lora_dim) return;

    device const float *qh =
        (device const float *)(q + (uint64_t)head * qk_dim * sizeof(float));
    float qv[6];
    FOR_UNROLL (uint b = 0; b < 6u; b++) {
        qv[b] = qh[(b << 5) + tiisg];
    }

    device float *out =
        (device float *)(qk_low + (uint64_t)head * kv_lora_dim * sizeof(float));
    for (uint r = 0; r < NR; r++) {
        const uint j = row0 + r;
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc = 0.0f;
        if (wt == DS4_METAL_GGUF_Q8_0) {
            FOR_UNROLL (uint b = 0; b < 6u; b++) {
                device const char *block_base = row + (uint64_t)b * 34u;
                const float d = (float)(*((device const half *)block_base));
                device const int8_t *qs = (device const int8_t *)(block_base + 2u);
                acc += d * (float)qs[tiisg] * qv[b];
            }
        } else {
            /* Q4_0: 18B blocks; elems 0..15 = low nibbles, 16..31 = high. */
            FOR_UNROLL (uint b = 0; b < 6u; b++) {
                device const char *block_base = row + (uint64_t)b * 18u;
                const float d = (float)(*((device const half *)block_base));
                device const uint8_t *qs = (device const uint8_t *)(block_base + 2u);
                const uint byte = qs[tiisg & 15u];
                const float v = (float)((tiisg < 16u) ? (byte & 0xFu) : (byte >> 4)) - 8.0f;
                acc += d * v * qv[b];
            }
        }
        const float sum = simd_sum(acc);
        if (tiisg == 0) {
            out[j] = sum;
        }
    }
}

kernel void kernel_glm_qk_lowrank_q8_0_batch(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x + args.head_base;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_dim;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head * qk_dim * sizeof(float));
    device float *out =
        (device float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
        out[j] = glm_quant_dot_row_dev_f32(args.weight_type, row, qh, args.qk_nope);
    }
}

kernel void kernel_glm_qk_lowrank_q8_0_batch_glm52_t4(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint n_head = 64u;
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 192u;
    constexpr uint qk_dim = 256u;
    constexpr uint tile_tokens = 4u;
    constexpr uint row_bytes = 204u;

    const uint head = tgpig.x + args.head_base;
    const uint token0 = tgpig.y * tile_tokens;
    const uint nth = ntg_u.x;
    const uint64_t q_token_stride = (uint64_t)n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)n_head * kv_lora_dim * sizeof(float);

    for (uint t = 0; t < tile_tokens; t++) {
        const uint token = token0 + t;
        threadgroup float *xt = x + t * qk_nope;
        if (token < args.n_tokens) {
            device const float *qh =
                (device const float *)(q +
                    (uint64_t)token * q_token_stride +
                    (uint64_t)head * qk_dim * sizeof(float));
            for (uint d = tid; d < qk_nope; d += nth) {
                xt[d] = qh[d];
            }
        } else {
            for (uint d = tid; d < qk_nope; d += nth) {
                xt[d] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (uint block = 0; block < 6u; block++) {
            device const char *block_base = row + (uint64_t)block * 34u;
            const float d = (float)(*((device const half *)block_base));
            device const int8_t *qs = (device const int8_t *)(block_base + 2u);
            const uint base = block << 5;
            FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
                const uint col = base + qi;
                const float wq = d * (float)qs[qi];
                acc0 += wq * x[col];
                acc1 += wq * x[qk_nope + col];
                acc2 += wq * x[2u * qk_nope + col];
                acc3 += wq * x[3u * qk_nope + col];
            }
        }

        if (token0 < args.n_tokens) {
            device float *out0 =
                (device float *)(qk_low +
                    (uint64_t)token0 * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out0[j] = acc0;
        }
        if (token0 + 1u < args.n_tokens) {
            device float *out1 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 1u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out1[j] = acc1;
        }
        if (token0 + 2u < args.n_tokens) {
            device float *out2 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 2u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out2[j] = acc2;
        }
        if (token0 + 3u < args.n_tokens) {
            device float *out3 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 3u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out3[j] = acc3;
        }
    }
}

kernel void kernel_glm_value_project_q8_0(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head) return;
    const uint nth = ntg_u.x;
    device const float *src =
        (device const float *)(lora + (uint64_t)head * args.kv_lora_dim * sizeof(float));
    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        x[j] = src[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.qk_dim * sizeof(float));
    for (uint d = tid; d < args.qk_dim; d += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.qk_dim + d) * args.row_bytes;
        out[d] = glm_quant_dot_row_tg_f32(args.weight_type, row, x, args.kv_lora_dim);
    }
}

kernel void kernel_glm_value_project_q8_0_batch_heads(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x + args.head_base;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    const uint nth = ntg_u.x;
    const uint value_dim = args.qk_dim;
    const uint64_t lora_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride =
        (uint64_t)args.n_head * value_dim * sizeof(float);
    device const float *src =
        (device const float *)(lora +
            (uint64_t)token * lora_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));
    device float *out =
        (device float *)(heads +
            (uint64_t)token * heads_token_stride +
            (uint64_t)head * value_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        x[j] = src[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < value_dim; d += nth) {
        device const char *row =
            weight + ((uint64_t)head * value_dim + d) * args.row_bytes;
        out[d] = glm_quant_dot_row_tg_f32(args.weight_type, row, x, args.kv_lora_dim);
    }
}

kernel void kernel_glm_value_project_q8_0_batch_heads_mma(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR0 = 64u;
    constexpr uint NR1 = 32u;
    constexpr uint NK = 32u;
    constexpr uint NL0 = 2u;
    constexpr uint NL1 = 4u;

    const uint token0 = tgpig.x * NR1;
    const uint value0 = tgpig.y * NR0;
    const uint head = tgpig.z + args.head_base;
    if (head >= args.n_head || token0 >= args.n_tokens || value0 >= args.qk_dim) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096u);

    const uint nr0 = min(NR0, args.qk_dim - value0);
    const uint nr1 = min(NR1, args.n_tokens - token0);

    const uint lr0 = min((uint)tid / NL0, nr0 - 1u);
    const uint lr1 = min((uint)tid / NL1, nr1 - 1u);
    const uint il0 = (uint)tid & 1u;
    const uint iy = 8u * ((uint)tid & (NL1 - 1u));

    const uint64_t lora_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride =
        (uint64_t)args.n_head * args.qk_dim * sizeof(float);
    const uint64_t head_lora_base =
        (uint64_t)head * args.kv_lora_dim * sizeof(float);
    const uint64_t head_out_base =
        (uint64_t)head * args.qk_dim * sizeof(float);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.kv_lora_dim; loop_k += NK) {
        const uint value = value0 + lr0;
        const uint block = loop_k >> 5;
        device const char *row =
            weight + ((uint64_t)head * args.qk_dim + value) * args.row_bytes;
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < 16u; i++) {
            const uint k = loop_k + 16u * il0 + i;
            const uint sx = 2u * il0 + i / 8u;
            const uint sy = ((uint)tid / NL0) / 8u;
            const uint lx = ((uint)tid / NL0) & 7u;
            const uint ly = i & 7u;
            const uint ib = 8u * sx + sy;
            const half v = (value < args.qk_dim && k < args.kv_lora_dim) ?
                half(d * (float)qs[16u * il0 + i]) :
                half(0.0f);
            *(sa + 64u * ib + 8u * ly + lx) = v;
        }

        const uint token = token0 + lr1;
        device const float *y =
            (device const float *)(lora +
                (uint64_t)token * lora_token_stride +
                head_lora_base +
                (uint64_t)loop_k * sizeof(float) +
                (uint64_t)iy * sizeof(float));
        for (uint i = 0; i < 8u; i++) {
            const uint k = loop_k + iy + i;
            const uint sx = ((uint)tid) & (NL1 - 1u);
            const uint sy = ((uint)tid / NL1) / 8u;
            const uint lx = i;
            const uint ly = ((uint)tid / NL1) & 7u;
            const uint ib = 4u * sx + sy;
            const half v = (token < args.n_tokens && k < args.kv_lora_dim) ?
                half(y[i]) :
                half(0.0f);
            *(sb + 64u * ib + 8u * ly + lx) = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4u * 64u * ((uint)sg & 1u);
        threadgroup const half *lsmb = sb + 2u * 64u * ((uint)sg >> 1);

        for (uint ik = 0; ik < NK / 8u; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 4u; i++) {
                simdgroup_load(ma[i], lsma + 64u * i, 8u, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 2u; i++) {
                simdgroup_load(mb[i], lsmb + 64u * i, 8u, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 8u; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i & 3u], mc[i]);
            }

            lsma += 8u * 64u;
            lsmb += 4u * 64u;
        }
    }

    if (nr0 == NR0 && nr1 == NR1) {
        device float *dst =
            (device float *)(heads +
                (uint64_t)(token0 + 16u * ((uint)sg >> 1)) * heads_token_stride +
                head_out_base +
                (uint64_t)(value0 + 32u * ((uint)sg & 1u)) * sizeof(float));
        for (uint i = 0; i < 8u; i++) {
            simdgroup_store(mc[i],
                            dst + 8u * (i & 3u) + 8u * (heads_token_stride / sizeof(float)) * (i / 4u),
                            heads_token_stride / sizeof(float),
                            0,
                            false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float *tmp = (threadgroup float *)shmem;
        for (uint i = 0; i < 8u; i++) {
            simdgroup_store(mc[i],
                            tmp + 32u * ((uint)sg & 1u) +
                                  16u * ((uint)sg >> 1) * NR0 +
                                  8u * (i & 3u) + 8u * NR0 * (i / 4u),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            for (uint t = tid; t < nr1; t += 128u) {
                device float *dst =
                    (device float *)(heads +
                        (uint64_t)(token0 + t) * heads_token_stride +
                        head_out_base +
                        (uint64_t)value0 * sizeof(float));
                threadgroup const float *src = tmp + t * NR0;
                for (uint v = 0; v < nr0; v++) {
                    dst[v] = src[v];
                }
            }
        }
    }
}

template <bool assume_valid_rows, bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_decode_split_group8_partial_impl(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *partial_lora,
        device char *partial_ms,
        threadgroup half4 *scratch [[threadgroup(0)]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group;
    const uint block = tgpig.y;
    if (args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        args.qk_rope != 64u ||
        args.block_rows == 0u ||
        block >= args.n_blocks) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint block_start = block * args.block_rows;
    const uint block_end = min(args.n_selected, block_start + args.block_rows);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q + (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = block_start; base < block_end; base += stage_rows) {
        const uint rows = min(stage_rows, block_end - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                const float2 y0 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                const float2 y1 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r + 2u,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
            } else {
                rope_shared[off] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        device float4 *out4 =
            (device float4 *)(partial_lora +
                ((uint64_t)block * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
        if (lane == 0u) {
            device float *ms =
                (device float *)(partial_ms +
                    ((uint64_t)block * args.n_head + head) * 2u * sizeof(float));
            ms[0] = M;
            ms[1] = S;
        }
    }
}

typedef decltype(kernel_glm_attention_indexed_decode_split_group8_partial_impl<false, false>)
        glm_attention_indexed_decode_split_group8_partial_t;

template [[host_name("kernel_glm_attention_indexed_decode_split_group8_partial")]]
kernel glm_attention_indexed_decode_split_group8_partial_t
kernel_glm_attention_indexed_decode_split_group8_partial_impl<false, false>;

template [[host_name("kernel_glm_attention_indexed_decode_split_group8_partial_valid_fullheads")]]
kernel glm_attention_indexed_decode_split_group8_partial_t
kernel_glm_attention_indexed_decode_split_group8_partial_impl<true, true>;

template<uint FIXED_BLOCKS>
static void kernel_glm_attention_indexed_decode_split_group8_reduce_impl(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint n_blocks = FIXED_BLOCKS != 0u ? FIXED_BLOCKS : args.n_blocks;
    if (head >= args.n_head ||
        args.n_selected == 0u ||
        args.kv_lora_dim != 512u ||
        n_blocks == 0u ||
        n_blocks > 64u ||
        (FIXED_BLOCKS != 0u && args.n_blocks != FIXED_BLOCKS)) {
        return;
    }

    const uint nth = ntg_u.x;
    threadgroup float *red = scratch;
    threadgroup float *block_scale = scratch + 256u;
    threadgroup float *lora_sum = scratch + 320u;

    float local_m = -FLT_MAX / 2.0f;
    if (tid < n_blocks) {
        device const float *ms =
            (device const float *)(partial_ms +
                ((uint64_t)tid * args.n_head + head) * 2u * sizeof(float));
        local_m = ms[1] > 0.0f ? ms[0] : -FLT_MAX / 2.0f;
    }
    red[tid] = local_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_m = red[0];

    float local_denom = 0.0f;
    if (tid < n_blocks) {
        device const float *ms =
            (device const float *)(partial_ms +
                ((uint64_t)tid * args.n_head + head) * 2u * sizeof(float));
        const float s = ms[1];
        const float e = s > 0.0f ? exp(ms[0] - max_m) : 0.0f;
        block_scale[tid] = e;
        local_denom = s * e;
    }
    red[tid] = local_denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint b = 0u; b < n_blocks; b++) {
            device const float *src =
                (device const float *)(partial_lora +
                    ((uint64_t)b * args.n_head + head) *
                        args.kv_lora_dim * sizeof(float));
            acc += src[j] * block_scale[b];
        }
        lora_sum[j] = acc / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_decode_split_group8_reduce(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    kernel_glm_attention_indexed_decode_split_group8_reduce_impl<0>(
            args, partial_lora, partial_ms, value_weight, heads, scratch,
            tid, ntg_u, tgpig);
}

kernel void kernel_glm_attention_indexed_decode_split_group8_reduce16(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    kernel_glm_attention_indexed_decode_split_group8_reduce_impl<16>(
            args, partial_lora, partial_ms, value_weight, heads, scratch,
            tid, ntg_u, tgpig);
}

kernel void kernel_glm_attention_indexed_decode(
        constant ds4_metal_args_glm_attention_indexed_decode & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;
    threadgroup float *lora_sum = scores + args.n_selected;

    device const float *qh =
        (device const float *)(q + (uint64_t)head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low + (uint64_t)head * args.kv_lora_dim * sizeof(float));

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    if (args.cache_f16 != 0u) {
        float local_max = -INFINITY;
        for (uint s = tid; s < args.n_selected; s += nth) {
            const uint row = selected[s];
            float score = -INFINITY;
            if (row < args.cache_cap) {
                float dotv = 0.0f;
                const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
                uint j = 0;
                for (; j + 3u < args.kv_lora_dim; j += 4u) {
                    device const half4 *kv4 =
                        (device const half4 *)((device const half *)kv_lora_cache + lora_base + j);
                    device const float4 *low4 =
                        (device const float4 *)(low + j);
                    const float4 kv = (float4)(*kv4);
                    const float4 qv = *low4;
                    dotv += qv.x * kv.x + qv.y * kv.y +
                            qv.z * kv.z + qv.w * kv.w;
                }
                if (j < args.kv_lora_dim) {
                    for (; j < args.kv_lora_dim; j++) {
                        const float kv = glm_cache_load_f16_only(kv_lora_cache,
                                                                 lora_base + j);
                        dotv += low[j] * kv;
                    }
                }
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                for (uint r = 0; r < args.qk_rope; r += 2u) {
                    const float2 y = glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                                                rope_base,
                                                                                r,
                                                                                row,
                                                                                args.qk_rope,
                                                                                args.freq_base,
                                                                                args.freq_scale,
                                                                                args.ext_factor,
                                                                                args.attn_factor,
                                                                                corr_dims[0],
                                                                                corr_dims[1]);
                    dotv += qh[args.qk_nope + r] * y.x +
                            qh[args.qk_nope + r + 1u] * y.y;
                }
                score = dotv * args.scale;
            }
            scores[s] = score;
            local_max = max(local_max, score);
        }
        red[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] = max(red[tid], red[tid + step]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float max_score = red[0];

        float local_sum = 0.0f;
        for (uint s = tid; s < args.n_selected; s += nth) {
            const float w = exp(scores[s] - max_score);
            scores[s] = w;
            local_sum += w;
        }
        red[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float denom = max(red[0], 1.0e-20f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j0 = tid * 2u; j0 < args.kv_lora_dim; j0 += nth * 2u) {
            const uint j1 = j0 + 1u;
            const bool use_j1 = j1 < args.kv_lora_dim;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint s = 0; s < args.n_selected; s++) {
                const uint row = selected[s];
                if (row < args.cache_cap) {
                    const uint64_t row_base = (uint64_t)row * args.kv_lora_dim;
                    const float w = scores[s];
                    if (use_j1) {
                        device const half2 *kv2 =
                            (device const half2 *)((device const half *)kv_lora_cache + row_base + j0);
                        const float2 kv = (float2)(*kv2);
                        acc0 += w * kv.x;
                        acc1 += w * kv.y;
                    } else {
                        const float kv0 = glm_cache_load_f16_only(kv_lora_cache,
                                                                  row_base + j0);
                        acc0 += w * kv0;
                    }
                }
            }
            lora_sum[j0] = acc0 / denom;
            if (use_j1) lora_sum[j1] = acc1 / denom;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float *out =
            (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
        return;
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = selected[s];
        float score = -INFINITY;
        if (row < args.cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dotv += low[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dotv += qh[args.qk_nope + r] * y.x +
                        qh[args.qk_nope + r + 1u] * y.y;
            }
            score = dotv * args.scale;
        }
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];

    float local_sum = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j0 = tid; j0 < args.kv_lora_dim; j0 += nth * 2u) {
        const uint j1 = j0 + nth;
        const bool use_j1 = j1 < args.kv_lora_dim;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = selected[s];
            if (row < args.cache_cap) {
                const uint64_t row_base = (uint64_t)row * args.kv_lora_dim;
                const float w = scores[s];
                const float kv0 = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                            row_base + j0,
                                                            args.cache_f16);
                acc0 += w * kv0;
                if (use_j1) {
                    const float kv1 = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                                row_base + j1,
                                                                args.cache_f16);
                    acc1 += w * kv1;
                }
            }
        }
        lora_sum[j0] = acc0 / denom;
        if (use_j1) lora_sum[j1] = acc1 / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;
    threadgroup float *lora_sum = scores + args.n_selected;

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = token_selected[s];
        float score = -INFINITY;
        if (row < args.cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dotv += low[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dotv += qh[args.qk_nope + r] * y.x +
                        qh[args.qk_nope + r + 1u] * y.y;
            }
            score = dotv * args.scale;
        }
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];

    float local_sum = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = token_selected[s];
            if (row < args.cache_cap) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           (uint64_t)row * args.kv_lora_dim + j,
                                                           args.cache_f16);
                acc += scores[s] * kv;
            }
        }
        lora_sum[j] = acc / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads +
            (uint64_t)token * heads_token_stride +
            (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch_group2(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.y;
    if (token >= args.n_tokens || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint head0 = tgpig.x * 2u;
    const uint head1 = head0 + 1u;
    const bool valid0 = head0 < args.n_head;
    const bool valid1 = head1 < args.n_head;
    if (!valid0 && !valid1) return;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    threadgroup float *red0 = scratch;
    threadgroup float *red1 = red0 + 256u;
    threadgroup float *scores0 = red1 + 256u;
    threadgroup float *scores1 = scores0 + args.n_selected;
    threadgroup float *lora0 = scores1 + args.n_selected;
    threadgroup float *lora1 = lora0 + args.kv_lora_dim;

    device const float *qh0 =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head0 * qk_dim * sizeof(float));
    device const float *qh1 =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)(valid1 ? head1 : head0) * qk_dim * sizeof(float));
    device const float *low0 =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head0 * args.kv_lora_dim * sizeof(float));
    device const float *low1 =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)(valid1 ? head1 : head0) * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = token_selected[s];
        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (row < args.cache_cap) {
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dot0 += low0[j] * kv;
                if (valid1) dot1 += low1[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dot0 += qh0[args.qk_nope + r] * y.x +
                        qh0[args.qk_nope + r + 1u] * y.y;
                if (valid1) {
                    dot1 += qh1[args.qk_nope + r] * y.x +
                            qh1[args.qk_nope + r + 1u] * y.y;
                }
            }
            score0 = dot0 * args.scale;
            if (valid1) score1 = dot1 * args.scale;
        }
        scores0[s] = score0;
        scores1[s] = score1;
        local_max0 = max(local_max0, score0);
        local_max1 = max(local_max1, score1);
    }
    red0[tid] = local_max0;
    red1[tid] = local_max1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red0[tid] = max(red0[tid], red0[tid + step]);
            red1[tid] = max(red1[tid], red1[tid + step]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score0 = red0[0];
    const float max_score1 = red1[0];

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w0 = (max_score0 > -INFINITY) ? exp(scores0[s] - max_score0) : 0.0f;
        const float w1 = (valid1 && max_score1 > -INFINITY) ? exp(scores1[s] - max_score1) : 0.0f;
        scores0[s] = w0;
        scores1[s] = w1;
        local_sum0 += w0;
        local_sum1 += w1;
    }
    red0[tid] = local_sum0;
    red1[tid] = local_sum1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red0[tid] += red0[tid + step];
            red1[tid] += red1[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom0 = max(red0[0], 1.0e-20f);
    const float denom1 = max(red1[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = token_selected[s];
            if (row < args.cache_cap) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           (uint64_t)row * args.kv_lora_dim + j,
                                                           args.cache_f16);
                acc0 += scores0[s] * kv;
                if (valid1) acc1 += scores1[s] * kv;
            }
        }
        lora0[j] = acc0 / denom0;
        if (valid1) lora1[j] = acc1 / denom1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < args.value_dim; d += nth) {
        device float *out0 =
            (device float *)(heads +
                (uint64_t)token * heads_token_stride +
                (uint64_t)head0 * args.value_dim * sizeof(float));
        device const char *row0 =
            value_weight + ((uint64_t)head0 * args.value_dim + d) * args.value_row_bytes;
        out0[d] = glm_quant_dot_row_tg_f32(args.value_type, row0, lora0, args.kv_lora_dim);

        if (valid1) {
            device float *out1 =
                (device float *)(heads +
                    (uint64_t)token * heads_token_stride +
                    (uint64_t)head1 * args.value_dim * sizeof(float));
            device const char *row1 =
                value_weight + ((uint64_t)head1 * args.value_dim + d) * args.value_row_bytes;
            out1[d] = glm_quant_dot_row_tg_f32(args.value_type, row1, lora1, args.kv_lora_dim);
        }
    }
}

template <bool assume_valid_rows, bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        args.qk_rope != 64u) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                const float2 y0 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                const float2 y1 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r + 2u,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
            } else {
                rope_shared[off] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_impl<false, false>)
        glm_attention_indexed_batch_lora_group8_vec_t;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<false, false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_valid")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<true, false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<true, true>;

template <bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        args.qk_rope != 64u) {
        return;
    }

    const uint visible = min(args.n_selected, args.pos0 + token + 1u);
    if (visible == 0u) return;

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < visible; base += stage_rows) {
        const uint rows = min(stage_rows, visible - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = base + rr;
            device const half4 *src =
                (device const half4 *)((device const half *)kv_lora_cache +
                    (uint64_t)row * args.kv_lora_dim);
            kv_shared[off] = src[vv];
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = base + rr;
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            const float2 y0 =
                glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                          rope_base,
                                                          r,
                                                          row,
                                                          args.qk_rope,
                                                          args.freq_base,
                                                          args.freq_scale,
                                                          args.ext_factor,
                                                          args.attn_factor,
                                                          corr_dims[0],
                                                          corr_dims[1]);
            const float2 y1 =
                glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                          rope_base,
                                                          r + 2u,
                                                          row,
                                                          args.qk_rope,
                                                          args.freq_base,
                                                          args.freq_scale,
                                                          args.ext_factor,
                                                          args.attn_factor,
                                                          corr_dims[0],
                                                          corr_dims[1]);
            rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score = valid_head ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<false>)
        glm_attention_indexed_batch_lora_group8_vec_causal_t;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_causal")]]
kernel glm_attention_indexed_batch_lora_group8_vec_causal_t
kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_causal_t
kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<true>;

kernel void kernel_glm_attention_indexed_batch_group8(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.y;
    if (token >= args.n_tokens || args.n_selected == 0u) return;

    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 8u;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    const bool valid_head = head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    threadgroup half *kv_shared = (threadgroup half *)scratch;
    threadgroup half *rope_shared = kv_shared + stage_rows * args.kv_lora_dim;
    threadgroup float *lora_sums =
        (threadgroup float *)(rope_shared + stage_rows * args.qk_rope);
    threadgroup float *head_lora = lora_sums + head_in_group * args.kv_lora_dim;

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    constexpr uint max_low_cache = 16u;
    constexpr uint max_qrope_cache = 4u;
    const bool use_low_cache = args.kv_lora_dim <= max_low_cache * 32u;
    const bool use_qrope_cache = args.qk_rope <= max_qrope_cache * 32u;
    half low_cache[max_low_cache];
    half qrope_cache[max_qrope_cache];
    for (uint k = 0u; k < max_low_cache; k++) {
        const uint j = lane + k * 32u;
        low_cache[k] = (valid_head && use_low_cache && j < args.kv_lora_dim) ?
            (half)low[j] : (half)0.0f;
    }
    for (uint k = 0u; k < max_qrope_cache; k++) {
        const uint r = lane + k * 32u;
        qrope_cache[k] = (valid_head && use_qrope_cache && r < args.qk_rope) ?
            (half)qh[args.qk_nope + r] : (half)0.0f;
    }

    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora[j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;
    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        const uint kv_count = rows * args.kv_lora_dim;
        const uint rope_pairs = args.qk_rope >> 1;
        const uint rope_count = rows * rope_pairs;

        for (uint idx = tid; idx < kv_count; idx += 256u) {
            const uint rr = idx / args.kv_lora_dim;
            const uint j = idx - rr * args.kv_lora_dim;
            const uint row = token_selected[base + rr];
            kv_shared[idx] = (row < args.cache_cap)
                ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                  (uint64_t)row * args.kv_lora_dim + j,
                                                  args.cache_f16)
                : (half)0.0f;
        }
        for (uint idx = tid; idx < rope_count; idx += 256u) {
            const uint rr = idx / rope_pairs;
            const uint pair = idx - rr * rope_pairs;
            const uint r = pair * 2u;
            const uint row = token_selected[base + rr];
            threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
            if (row < args.cache_cap) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   (uint64_t)row * args.qk_rope,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                rope_row[r] = (half)y.x;
                rope_row[r + 1u] = (half)y.y;
            } else {
                rope_row[r] = (half)0.0f;
                rope_row[r + 1u] = (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = row < args.cache_cap;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;
                if (use_low_cache) {
                    for (uint k = 0u; k < max_low_cache; k++) {
                        const uint j = lane + k * 32u;
                        if (j < args.kv_lora_dim) {
                            partial += (float)(low_cache[k] * kv_row[j]);
                        }
                    }
                } else {
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        partial += low[j] * (float)kv_row[j];
                    }
                }
                if (use_qrope_cache) {
                    for (uint k = 0u; k < max_qrope_cache; k++) {
                        const uint r = lane + k * 32u;
                        if (r < args.qk_rope) {
                            partial += (float)(qrope_cache[k] * rope_row[r]);
                        }
                    }
                } else {
                    for (uint r = lane; r < args.qk_rope; r += 32u) {
                        partial += qh[args.qk_nope + r] * (float)rope_row[r];
                    }
                }
            }

            const float sum = simd_sum(partial);
            const float score = (valid_head && valid_row) ? sum * args.scale : -INFINITY;
            if (valid_head && valid_row) {
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                const float old_m = M;
                const float new_m = max(M, score);
                const float old_scale = (old_m == -INFINITY) ? 0.0f : exp(old_m - new_m);
                const float row_scale = exp(score - new_m);
                S = S * old_scale + row_scale;
                for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                    head_lora[j] = head_lora[j] * old_scale + row_scale * (float)kv_row[j];
                }
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_s = (valid_head && S > 0.0f) ? 1.0f / S : 0.0f;
    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora[j] *= inv_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_head) {
        if (args.value_type == 1u) {
            const uint64_t offset =
                (uint64_t)token *
                    ((uint64_t)args.n_head * args.kv_lora_dim * sizeof(float)) +
                (uint64_t)head * args.kv_lora_dim * sizeof(float);
            device float *out =
                (device float *)(heads + offset);
            for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                out[j] = head_lora[j];
            }
            return;
        }
        device float *out =
            (device float *)(heads +
                (uint64_t)token * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = lane; d < args.value_dim; d += 32u) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch_q2_group4(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup uint *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    const uint token0 = tgpig.y * 2u;
    if (token0 >= args.n_tokens || args.n_selected == 0u) return;

    constexpr uint group_heads = 4u;
    constexpr uint stage_rows = 4u;
    constexpr uint group_threads = 128u;
    const uint token1 = token0 + 1u;
    const bool valid1 = token1 < args.n_tokens;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    const bool valid_head = head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    const uint bit_words = (args.cache_cap + 31u) >> 5;
    threadgroup atomic_uint *member_bits = (threadgroup atomic_uint *)scratch;
    threadgroup half *kv_shared = (threadgroup half *)(scratch + bit_words);
    threadgroup half *rope_shared = kv_shared + stage_rows * args.kv_lora_dim;
    threadgroup float *lora_sums =
        (threadgroup float *)(rope_shared + stage_rows * args.qk_rope);
    threadgroup float *head_lora0 = lora_sums + head_in_group * args.kv_lora_dim;
    threadgroup float *head_lora1 =
        lora_sums + (group_heads + head_in_group) * args.kv_lora_dim;

    const uint safe_token1 = valid1 ? token1 : token0;
    device const float *qh0 =
        (device const float *)(q +
            (uint64_t)token0 * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *qh1 =
        (device const float *)(q +
            (uint64_t)safe_token1 * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *low0 =
        (device const float *)(qk_low +
            (uint64_t)token0 * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const float *low1 =
        (device const float *)(qk_low +
            (uint64_t)safe_token1 * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *selected0 = selected + (uint64_t)token0 * args.n_selected;
    device const uint32_t *selected1 = selected + (uint64_t)safe_token1 * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    constexpr uint max_low_cache = 16u;
    constexpr uint max_qrope_cache = 4u;
    const bool use_low_cache = args.kv_lora_dim <= max_low_cache * 32u;
    const bool use_qrope_cache = args.qk_rope <= max_qrope_cache * 32u;
    half low_cache0[max_low_cache];
    half low_cache1[max_low_cache];
    half qrope_cache0[max_qrope_cache];
    half qrope_cache1[max_qrope_cache];
    for (uint k = 0u; k < max_low_cache; k++) {
        const uint j = lane + k * 32u;
        low_cache0[k] = (valid_head && use_low_cache && j < args.kv_lora_dim) ?
            (half)low0[j] : (half)0.0f;
        low_cache1[k] = (valid_head && valid1 && use_low_cache && j < args.kv_lora_dim) ?
            (half)low1[j] : (half)0.0f;
    }
    for (uint k = 0u; k < max_qrope_cache; k++) {
        const uint r = lane + k * 32u;
        qrope_cache0[k] = (valid_head && use_qrope_cache && r < args.qk_rope) ?
            (half)qh0[args.qk_nope + r] : (half)0.0f;
        qrope_cache1[k] = (valid_head && valid1 && use_qrope_cache && r < args.qk_rope) ?
            (half)qh1[args.qk_nope + r] : (half)0.0f;
    }

    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora0[j] = 0.0f;
        if (valid1) head_lora1[j] = 0.0f;
    }
    for (uint i = tid; i < bit_words; i += group_threads) {
        atomic_store_explicit(member_bits + i, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tid; s < args.n_selected; s += group_threads) {
        const uint row = selected0[s];
        if (row < args.cache_cap) {
            const uint mask = 1u << (row & 31u);
            atomic_fetch_or_explicit(member_bits + (row >> 5), mask, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M0 = -INFINITY;
    float S0 = 0.0f;
    float M1 = -INFINITY;
    float S1 = 0.0f;

    if (valid1) {
        for (uint base = 0u; base < args.n_selected; base += stage_rows) {
            const uint rows = min(stage_rows, args.n_selected - base);
            const uint kv_count = rows * args.kv_lora_dim;
            const uint rope_pairs = args.qk_rope >> 1;
            const uint rope_count = rows * rope_pairs;

            for (uint idx = tid; idx < kv_count; idx += 256u) {
                const uint rr = idx / args.kv_lora_dim;
                const uint j = idx - rr * args.kv_lora_dim;
                const uint row = selected1[base + rr];
                kv_shared[idx] = (row < args.cache_cap)
                    ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                      (uint64_t)row * args.kv_lora_dim + j,
                                                      args.cache_f16)
                    : (half)0.0f;
            }
            for (uint idx = tid; idx < rope_count; idx += 256u) {
                const uint rr = idx / rope_pairs;
                const uint pair = idx - rr * rope_pairs;
                const uint r = pair * 2u;
                const uint row = selected1[base + rr];
                threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
                if (row < args.cache_cap) {
                    const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                       (uint64_t)row * args.qk_rope,
                                                                       r,
                                                                       row,
                                                                       args.qk_rope,
                                                                       args.cache_f16,
                                                                       args.freq_base,
                                                                       args.freq_scale,
                                                                       args.ext_factor,
                                                                       args.attn_factor,
                                                                       corr_dims[0],
                                                                       corr_dims[1]);
                    rope_row[r] = (half)y.x;
                    rope_row[r + 1u] = (half)y.y;
                } else {
                    rope_row[r] = (half)0.0f;
                    rope_row[r + 1u] = (half)0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint rr = 0u; rr < rows; rr++) {
                const uint row = selected1[base + rr];
                const bool valid_row = row < args.cache_cap;
                const bool in_token0 = valid_row &&
                    ((atomic_load_explicit(member_bits + (row >> 5),
                                           memory_order_relaxed) &
                      (1u << (row & 31u))) != 0u);
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;

                float partial0 = 0.0f;
                float partial1 = 0.0f;
                if (valid_head && valid_row) {
                    if (use_low_cache) {
                        for (uint k = 0u; k < max_low_cache; k++) {
                            const uint j = lane + k * 32u;
                            if (j < args.kv_lora_dim) {
                                const half kv = kv_row[j];
                                if (in_token0) partial0 += (float)(low_cache0[k] * kv);
                                partial1 += (float)(low_cache1[k] * kv);
                            }
                        }
                    } else {
                        for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                            const float kv = (float)kv_row[j];
                            if (in_token0) partial0 += low0[j] * kv;
                            partial1 += low1[j] * kv;
                        }
                    }
                    if (use_qrope_cache) {
                        for (uint k = 0u; k < max_qrope_cache; k++) {
                            const uint r = lane + k * 32u;
                            if (r < args.qk_rope) {
                                const half kv = rope_row[r];
                                if (in_token0) partial0 += (float)(qrope_cache0[k] * kv);
                                partial1 += (float)(qrope_cache1[k] * kv);
                            }
                        }
                    } else {
                        for (uint r = lane; r < args.qk_rope; r += 32u) {
                            const float kv = (float)rope_row[r];
                            if (in_token0) partial0 += qh0[args.qk_nope + r] * kv;
                            partial1 += qh1[args.qk_nope + r] * kv;
                        }
                    }
                }

                const float sum0 = simd_sum(partial0);
                const float sum1 = simd_sum(partial1);
                const float score0 = (valid_head && in_token0) ? sum0 * args.scale : -INFINITY;
                const float score1 = (valid_head && valid_row) ? sum1 * args.scale : -INFINITY;
                if (valid_head && in_token0) {
                    const float new_m = max(M0, score0);
                    const float old_scale = (M0 == -INFINITY) ? 0.0f : exp(M0 - new_m);
                    const float row_scale = exp(score0 - new_m);
                    S0 = S0 * old_scale + row_scale;
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        head_lora0[j] = head_lora0[j] * old_scale + row_scale * (float)kv_row[j];
                    }
                    M0 = new_m;
                }
                if (valid_head && valid_row) {
                    const float new_m = max(M1, score1);
                    const float old_scale = (M1 == -INFINITY) ? 0.0f : exp(M1 - new_m);
                    const float row_scale = exp(score1 - new_m);
                    S1 = S1 * old_scale + row_scale;
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        head_lora1[j] = head_lora1[j] * old_scale + row_scale * (float)kv_row[j];
                    }
                    M1 = new_m;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint i = tid; i < bit_words; i += group_threads) {
            atomic_store_explicit(member_bits + i, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = tid; s < args.n_selected; s += group_threads) {
            const uint row = selected1[s];
            if (row < args.cache_cap) {
                const uint mask = 1u << (row & 31u);
                atomic_fetch_or_explicit(member_bits + (row >> 5), mask, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        const uint kv_count = rows * args.kv_lora_dim;
        const uint rope_pairs = args.qk_rope >> 1;
        const uint rope_count = rows * rope_pairs;

        for (uint idx = tid; idx < kv_count; idx += 256u) {
            const uint rr = idx / args.kv_lora_dim;
            const uint j = idx - rr * args.kv_lora_dim;
            const uint row = selected0[base + rr];
            kv_shared[idx] = (row < args.cache_cap)
                ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                  (uint64_t)row * args.kv_lora_dim + j,
                                                  args.cache_f16)
                : (half)0.0f;
        }
        for (uint idx = tid; idx < rope_count; idx += 256u) {
            const uint rr = idx / rope_pairs;
            const uint pair = idx - rr * rope_pairs;
            const uint r = pair * 2u;
            const uint row = selected0[base + rr];
            threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
            if (row < args.cache_cap) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   (uint64_t)row * args.qk_rope,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                rope_row[r] = (half)y.x;
                rope_row[r + 1u] = (half)y.y;
            } else {
                rope_row[r] = (half)0.0f;
                rope_row[r + 1u] = (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = selected0[base + rr];
            const bool valid_row = row < args.cache_cap;
            const bool in_token1 = valid1 && valid_row &&
                ((atomic_load_explicit(member_bits + (row >> 5),
                                       memory_order_relaxed) &
                  (1u << (row & 31u))) != 0u);
            const bool take0 = valid_row && !in_token1;
            threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
            threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;

            float partial0 = 0.0f;
            if (valid_head && take0) {
                if (use_low_cache) {
                    for (uint k = 0u; k < max_low_cache; k++) {
                        const uint j = lane + k * 32u;
                        if (j < args.kv_lora_dim) {
                            partial0 += (float)(low_cache0[k] * kv_row[j]);
                        }
                    }
                } else {
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        partial0 += low0[j] * (float)kv_row[j];
                    }
                }
                if (use_qrope_cache) {
                    for (uint k = 0u; k < max_qrope_cache; k++) {
                        const uint r = lane + k * 32u;
                        if (r < args.qk_rope) {
                            partial0 += (float)(qrope_cache0[k] * rope_row[r]);
                        }
                    }
                } else {
                    for (uint r = lane; r < args.qk_rope; r += 32u) {
                        partial0 += qh0[args.qk_nope + r] * (float)rope_row[r];
                    }
                }
            }

            const float sum0 = simd_sum(partial0);
            const float score0 = (valid_head && take0) ? sum0 * args.scale : -INFINITY;
            if (valid_head && take0) {
                const float new_m = max(M0, score0);
                const float old_scale = (M0 == -INFINITY) ? 0.0f : exp(M0 - new_m);
                const float row_scale = exp(score0 - new_m);
                S0 = S0 * old_scale + row_scale;
                for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                    head_lora0[j] = head_lora0[j] * old_scale + row_scale * (float)kv_row[j];
                }
                M0 = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_s0 = (valid_head && S0 > 0.0f) ? 1.0f / S0 : 0.0f;
    const float inv_s1 = (valid_head && valid1 && S1 > 0.0f) ? 1.0f / S1 : 0.0f;
    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora0[j] *= inv_s0;
        if (valid1) head_lora1[j] *= inv_s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_head) {
        device float *out0 =
            (device float *)(heads +
                (uint64_t)token0 * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        device float *out1 =
            (device float *)(heads +
                (uint64_t)safe_token1 * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = lane; d < args.value_dim; d += 32u) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out0[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora0, args.kv_lora_dim);
            if (valid1) {
                out1[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora1, args.kv_lora_dim);
            }
        }
    }
}

// GLM-5.2 decode router for one token. Selection uses sigmoid(logit)+bias,
// while route weights are normalized from the unbiased sigmoid probabilities.
kernel void kernel_glm_router_select_one(
        constant ds4_metal_args_glm_router_select_one & args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        threadgroup float *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    device const float *token_logits = logits + (uint64_t)token * args.n_expert;
    device int32_t *token_selected = selected + (uint64_t)token * args.n_expert_used;
    device float *token_weights = weights + (uint64_t)token * args.n_expert_used;
    device float *token_probs = probs + (uint64_t)token * args.n_expert;

    const uint n_expert = min(args.n_expert, 256u);
    const bool active = tid < n_expert;
    const float p = active ? ds4_glm_router_sigmoid(token_logits[tid]) : 0.0f;
    if (active) token_probs[tid] = p;
    sel_scores[tid] = active ? p + bias[tid] : -INFINITY;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid) {
                const int32_t a = idx[tid];
                const int32_t b = idx[other];
                const bool descending = (tid & k) == 0;
                const bool swap = descending
                    ? ds4_glm_router_better(sel_scores, b, a)
                    : ds4_glm_router_better(sel_scores, a, b);
                if (swap) {
                    idx[tid] = b;
                    idx[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint k_used = min(args.n_expert_used, n_expert);
    if (tid < k_used) {
        token_selected[tid] = idx[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < k_used) {
        float sum = 0.0f;
        for (uint i = 0; i < k_used; i++) {
            sum += token_probs[(uint)token_selected[i]];
        }
        sum = max(sum, 6.103515625e-5f);
        token_weights[tid] = token_probs[(uint)token_selected[tid]] / sum * args.expert_weight_scale;
    }
}

// Batched Flash-router weight finalization after selection is already known.
// Six active lanes deliberately match kernel_sum_rows_f32_f32's reduction
// topology. The denominator and divided weights cross threadgroup storage
// boundaries so division cannot be reassociated with the final scale.
kernel void kernel_dsv4_router_weights_batch(
        constant float &scale,
        device const float *probs,
        device const int32_t *selected,
        device float *weights,
        threadgroup volatile float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (tid >= 6) return;

    threadgroup volatile float *sum_scratch = scratch;
    threadgroup volatile float *denom_scratch = scratch + 32;
    threadgroup volatile float *div_scratch = scratch + 33;
    const uint out_index = row * 6u + (uint)tid;
    const int32_t expert = selected[out_index];
    const float p = probs[row * 256u + (uint)expert];

    // Keep this sequence identical to kernel_sum_rows_f32_f32 for width 6.
    if (sgitg == 0) {
        sum_scratch[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    sumf += p;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_scratch[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sum_scratch[tiisg];
    sumf = simd_sum(sumf);

    if (tid == 0) {
        denom_scratch[0] = clamp(sumf, 6.103515625e-5f, INFINITY);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    div_scratch[tid] = p / denom_scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    weights[out_index] = div_scratch[tid] * scale;
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

// M3 decode specialization for the non-hash one-token router. Scores and ids
// stay in registers. Intra-SIMD bitonic stages use shuffle-xor; the six stages
// that cross 32-lane SIMD groups exchange through alternating threadgroup
// banks. The next bank's publish barrier proves every prior-bank read finished;
// by the time a bank is reused two cross stages later, no reader can remain.
kernel void kernel_dsv4_router_finalize_one_simd(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
}

// M3 decode specialization that extends the register/TG SIMD selection above
// through the existing six-value serial weight normalization. The selected ids
// cross the same device-memory boundary as the standalone weight kernel;
// volatile TG stores pin its left-fold and scaled-reciprocal rounding points.
kernel void kernel_dsv4_router_finalize_weights_one_simd(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        device const int32_t *s = selected;
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; i++) {
            norm_scratch[0] = norm_scratch[0] + probs[s[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        device const int32_t *s = selected;
        weights[tid] = probs[s[tid]] * norm_scratch[1];
    }
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
        uint tid [[thread_position_in_threadgroup]],
        uint n_threads [[threads_per_threadgroup]]) {
    const uint top_k = (uint)args.ne00;
    if (row >= (uint)args.ne01 || tid >= n_threads) {
        return;
    }

    for (uint i = tid; i < top_k; i += n_threads) {
        row_tmp[i] = *((device const int32_t *) (src + (uint64_t)i*args.nb00 + (uint64_t)row*args.nb01));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= top_k; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < top_k; i += n_threads) {
                const uint other = i ^ j;
                if (other > i && other < top_k) {
                    const int32_t a = row_tmp[i];
                    const int32_t b = row_tmp[other];
                    const bool up = (i & k) == 0;
                    if ((up && a > b) || (!up && a < b)) {
                        row_tmp[i] = b;
                        row_tmp[other] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = tid; i < top_k; i += n_threads) {
        *((device int32_t *) (dst + (uint64_t)i*args.nb00 + (uint64_t)row*args.nb01)) = row_tmp[i];
    }
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

static inline void dsv4_attend_shared_h4_row(
        threadgroup const half4 *kv4,
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
    const half4 k0 = kv4[lane +  0];
    const half4 k1 = kv4[lane + 32];
    const half4 k2 = kv4[lane + 64];
    const half4 k3 = kv4[lane + 96];

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

static inline void dsv4_attend_shared_h4_row_at(
        threadgroup const half4 *kv4,
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
    dsv4_attend_shared_h4_row(kv4 + row_in_tg * 128u,
                              q0, q1, q2, q3,
                              scale,
                              lane,
                              M, S,
                              o0, o1, o2, o3);
}

static inline half4 dsv4_load_cache_h4(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        uint col,
        bool f16_rows) {
    device const char *base = kv + (uint64_t)row * row_stride;
    if (f16_rows) {
        return ((device const half4 *)base)[col];
    }
    return (half4)((device const float4 *)base)[col];
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
        threadgroup half4 *kv_shared [[threadgroup(0)]],
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
            if (tid < 128) kv_shared[tid] = (half4)src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_h4_row(kv_shared,
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
        if (tid < 128) {
            kv_shared[tid] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                (uint)idx,
                                                tid,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dsv4_attend_shared_h4_row(kv_shared,
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
        threadgroup half4 *kv_shared [[threadgroup(0)]],
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
                kv_shared[off] = (half4)src[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_h4_row_at(kv_shared,
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
            kv_shared[off] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                rows[r],
                                                c,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_h4_row_at(kv_shared,
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

#ifdef DS4_METAL_HAS_TENSOR
// Retained full-512 prefill indexer score path.  This is the part of sparse
// compressed attention that maps cleanly to TensorOps: a regular token by
// compressed-row dot tile.  The kernel intentionally leaves top-k selection and
// indexed attention semantics unchanged; all 512 selected rows remain available
// to the later attention kernel.
//
// Each matmul processes a pair of heads (TQ = 2 x TM q rows): the per-element
// dot is still a 128-deep reduction in 32-wide k-steps, so scores are
// bit-identical to single-head tiles while the run count halves.  The q tile
// is double-buffered, so the next k-step's stage overlaps the current
// cooperative matmul and each pair needs 5 barriers instead of 10.  q and k
// staging use one float4/half4 per lane (each thread covers one row of 8/32
// consecutive elements), which is the same half(float) conversion per element
// as the scalar form.
kernel void kernel_dsv4_indexer_scores_nax(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup half *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    constexpr int TM = 16;
    constexpr int TQ = 32;
    constexpr int TN = 32;
    constexpr int NK = 32;
    constexpr int D  = 128;
    constexpr int NUM_THREADS = 128;

    // The 16-token x 32-row tile was the winning NAX shape in local sweeps.  A
    // wider 64-row compressed tile increased setup/cache pressure and was
    // slower despite doing more work per dispatch.
    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup half  *qtg = shared;               // 2 x [TQ][NK]
    threadgroup half  *ktg = qtg + 2*TQ*NK;        // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [TQ][TN], column-major

    const uint last_token = min(t0 + (uint)TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += NUM_THREADS) {
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

    {
        // One compressed row per 4 threads, 32 consecutive floats per thread.
        const uint cc = tid / 4;
        const uint comp = c0 + cc;
        device const float *krow = nullptr;
        if (comp < args.n_comp) {
            krow = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
        }
        const uint d0 = (tid % 4) * 32;
        FOR_UNROLL (uint j = 0; j < 8; j++) {
            const float4 kv = krow ? *(device const float4 *)(krow + d0 + 4*j)
                                   : float4(0.0f);
            *(threadgroup half4 *)(ktg + cc*D + d0 + 4*j) = half4(kv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        acc[j] = 0.0f;
    }

    auto tq0 = tensor(qtg,          dextents<int32_t, 2>(NK, TQ));
    auto tq1 = tensor(qtg + TQ*NK,  dextents<int32_t, 2>(NK, TQ));
    auto tk = tensor(ktg, dextents<int32_t, 2>(D, TN));
    auto td = tensor(dot, dextents<int32_t, 2>(TQ, TN), array<int, 2>({1, TQ}));

    matmul2d<
        matmul2d_descriptor(TN, TQ, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    // One q row per 4 threads, 8 consecutive floats per thread.  Row r covers
    // head (r / TM) of the pair and token row (r % TM).
    const uint q_r = tid / 4;
    const uint q_k4 = (tid % 4) * 8;
    const uint q_hl = q_r / TM;
    const uint q_tr = q_r % TM;
    const uint q_token = t0 + q_tr;
    device const char *q_row_base = nullptr;
    if (q_token < args.n_tokens) {
        q_row_base = q + (uint64_t)q_token * args.q_token_stride;
    }

    auto stage_q = [&](const uint head0, const uint loop_k, threadgroup half *buf) {
        const uint head = head0 + q_hl;
        half4 v0 = half4(0.0f);
        half4 v1 = half4(0.0f);
        if (q_row_base && head < args.n_head) {
            device const float4 *src4 = (device const float4 *)
                (q_row_base + (uint64_t)head * args.q_head_stride +
                 (uint64_t)(loop_k + q_k4) * sizeof(float));
            v0 = half4(src4[0]);
            v1 = half4(src4[1]);
        }
        *(threadgroup half4 *)(buf + q_r*NK + q_k4)     = v0;
        *(threadgroup half4 *)(buf + q_r*NK + q_k4 + 4) = v1;
    };

    for (uint head0 = 0; head0 < args.n_head; head0 += 2) {
        auto ct = mm.template get_destination_cooperative_tensor<decltype(tk), decltype(tq0), float>();
        #pragma unroll
        for (uint16_t i = 0; i < ct.get_capacity(); i++) {
            if (ct.is_valid_element(i)) {
                ct[i] = 0.0f;
            }
        }

        stage_q(head0, 0, qtg);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint qsel = 0;
        FOR_UNROLL (uint i = 0; i < 4; i++) {
            auto mk = tk.slice(i*NK, 0);
            auto mq = (qsel ? tq1 : tq0).slice(0, 0);
            mm.run(mk, mq, ct);
            if (i < 3) {
                qsel ^= 1u;
                stage_q(head0, (i + 1)*NK, qsel ? qtg + TQ*NK : qtg);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        ct.store(td);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            const uint linear = (uint)tid + j*NUM_THREADS;
            if (linear < TM*TN) {
                const uint r = linear / TN;
                const uint cc = linear - r*TN;
                const uint token = t0 + r;
                if (token < args.n_tokens) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token * args.weights_token_stride);
                    acc[j] += max(dot[cc*TQ + r], 0.0f) * (w[head0] * args.scale);
                    if (head0 + 1 < args.n_head) {
                        acc[j] += max(dot[cc*TQ + TM + r], 0.0f) * (w[head0 + 1] * args.scale);
                    }
                }
            }
        }
        // No barrier here: the next pair's q stage and these dot reads touch
        // different buffers, and the next q-stage barrier separates the next
        // ct.store from these reads.
    }

    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        const uint linear = (uint)tid + j*NUM_THREADS;
        if (linear >= TM*TN) {
            continue;
        }
        const uint r = linear / TN;
        const uint cc = linear - r*TN;
        const uint token = t0 + r;
        const uint comp = c0 + cc;
        if (token < args.n_tokens && comp < args.n_comp) {
            const uint visible = min((args.pos0 + token + 1u) / args.ratio, args.n_comp);
            device float *dst = (device float *)(scores +
                (uint64_t)token * args.score_token_stride) + comp;
            *dst = comp < visible ? acc[j] : -INFINITY;
        }
    }
}
#endif

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

// Adds the periodic compressor APE directly to projected scores. The legacy
// path materializes one repeated APE segment per period and then performs this
// same single F32 add; these kernels remove only that intermediate copy graph.
kernel void kernel_dsv4_compressor_score_ape_f32(
        constant ds4_metal_args_dsv4_compressor_score_ape & args,
        device const float *score,
        device const float *ape,
        device       float *dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t)args.n_tokens * args.width;
    if ((uint64_t)gid >= total) return;

    const uint token = gid / args.width;
    const uint col = gid - token*args.width;
    const uint ape_row = (uint)(((uint64_t)args.pos0 + token) % args.ratio);
    dst[gid] = score[gid] + ape[(uint64_t)ape_row*args.width + col];
}

kernel void kernel_dsv4_compressor_score_ape_f16(
        constant ds4_metal_args_dsv4_compressor_score_ape & args,
        device const float *score,
        device const half  *ape,
        device       float *dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t)args.n_tokens * args.width;
    if ((uint64_t)gid >= total) return;

    const uint token = gid / args.width;
    const uint col = gid - token*args.width;
    const uint ape_row = (uint)(((uint64_t)args.pos0 + token) % args.ratio);
    dst[gid] = score[gid] + float(ape[(uint64_t)ape_row*args.width + col]);
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



// Tensor-parallel keep-alive: a few threadgroups of FMAs dispatched
// back-to-back on a side queue while TP decode runs.  The per-layer gate
// stalls make the real workload look idle to the GPU power manager, which
// otherwise halves the clocks within a second (~2x decode regression);
// this holds them up for negligible bandwidth and a few watts.
kernel void kernel_dsv4_tp_keepalive(
        device float * out,
        constant uint & iters,
        uint tid [[thread_position_in_grid]]) {
    float a = out[tid];
    const float b = 1.000001f;
    for (uint i = 0; i < iters; i++) {
        a = fma(a, b, 0.000001f);
        a = fma(a, b, -0.000001f);
    }
    out[tid] = a;
}

// Tensor-parallel gate flag: publishes a sequence number to a slab slot the
// CPU service thread spin-reads, replacing the much slower shared-event
// signal for the GPU->CPU direction.  Ordering against the partial-output
// kernels comes from the buffer hazard on the shared slab.
kernel void kernel_dsv4_tp_flag_set(
        device atomic_uint & flag,
        constant uint & value,
        uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        atomic_store_explicit(&flag, value, memory_order_relaxed);
    }
}

// Ratio-4 compressor pooling without materializing the [n_comp, 8, head_dim]
// KV and score packs. The row mapping and both reduction loops deliberately
// match kernel_dsv4_softmax_pool so the arithmetic order is unchanged.
kernel void kernel_dsv4_softmax_pool_ratio4_direct(
        constant ds4_metal_args_dsv4_softmax_pool_ratio4_direct & args,
        device const float * kv,
        device const float * score,
        device const float * state_kv,
        device const float * state_score,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.head_dim * args.n_comp;
    if ((uint64_t)gid >= n || args.head_dim == 0u) {
        return;
    }

    const uint64_t id = gid % args.head_dim;
    const uint64_t ic = gid / args.head_dim;
    const uint64_t input_row_stride = 2ull * args.head_dim;

    float max_s = -INFINITY;
    float sum = 0.0f;
    float acc = 0.0f;
    if (ic != 0u) {
        const int64_t token_base = (int64_t)ic * 4 - 4;
        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            const uint64_t token = (uint64_t)(token_base + ir);
            const uint64_t src = token * input_row_stride +
                                 ((uint64_t)ir >> 2u) * args.head_dim + id;
            const float s = score[src];
            max_s = max(max_s, s);
        }

        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            const uint64_t token = (uint64_t)(token_base + ir);
            const uint64_t src = token * input_row_stride +
                                 ((uint64_t)ir >> 2u) * args.head_dim + id;
            const float s = score[src];
            const float w = exp(s - max_s);
            const float v = kv[src];
            sum += w;
            acc += v*w;
        }
    } else {
        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            float s;
            if (ir >= 4) {
                const uint64_t src = (uint64_t)(ir - 4) * input_row_stride +
                                     args.head_dim + id;
                s = score[src];
            } else if (args.replay != 0u) {
                s = state_score[(uint64_t)ir * input_row_stride + id];
            } else {
                s = -INFINITY;
            }
            max_s = max(max_s, s);
        }

        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            float s;
            float v;
            if (ir >= 4) {
                const uint64_t src = (uint64_t)(ir - 4) * input_row_stride +
                                     args.head_dim + id;
                s = score[src];
                v = kv[src];
            } else if (args.replay != 0u) {
                const uint64_t src = (uint64_t)ir * input_row_stride + id;
                s = state_score[src];
                v = state_kv[src];
            } else {
                s = -INFINITY;
                v = 0.0f;
            }
            const float w = exp(s - max_s);
            sum += w;
            acc += v*w;
        }
    }

    dst[ic * args.head_dim + id] = acc/sum;
}
