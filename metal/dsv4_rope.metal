struct ds4_metal_args_dsv4_rope_tail {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_dims;
    int32_t  mode;
    int32_t  n_ctx_orig;
    int32_t  inverse;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    bool     src2;
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

// Applies DeepSeek V4's partial RoPE: the no-position prefix is copied and only
// the rotated tail is transformed. This is used for Q/K after their projections
// and before writing/reading the attention KV state.
kernel void kernel_dsv4_rope_tail_f32(
        constant ds4_metal_args_dsv4_rope_tail & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint  tid   [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    device const int32_t * pos = (device const int32_t *) src1;

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;
    const bool is_neox = args.mode == 2;

    for (int i0 = tid; i0 < args.ne00; i0 += ntg.x) {
        device const char * src_base = src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01;
        device       char * dst_base = dst  + i3*args.nb3  + i2*args.nb2  + i1*args.nb1;

        if (i0 < n_nope) {
            *((device float *) (dst_base + i0*args.nb0)) = *((device const float *) (src_base + i0*args.nb00));
            continue;
        }

        const int r = i0 - n_nope;
        if (is_neox) {
            const int n_half = args.n_dims/2;
            if (r >= n_half) {
                continue;
            }

            const int ic = r;
            const int rel_i0 = 2*ic;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
            // Equivalent to pow(freq_base, k) but expressed through IEEE-754
            // primitives that have tighter precision guarantees than Metal's pow().
            const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
            const float theta = theta_base * pow(args.freq_base, inv_ndims*rel_i0);
#endif
            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, rel_i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *((device const float *) (src_base + j0*args.nb00));
            const float x1 = *((device const float *) (src_base + j1*args.nb00));

            *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
            *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
        } else {
            if ((r & 1) != 0) {
                continue;
            }

            const int ic = r/2;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
            const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
            const float theta = theta_base * pow(args.freq_base, inv_ndims*r);
#endif
            const float freq_factor = args.src2 ? ((device const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *((device const float *) (src_base + j0*args.nb00));
            const float x1 = *((device const float *) (src_base + j1*args.nb00));

            *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
            *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
        }
    }
}
