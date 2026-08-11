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

struct ds4_metal_args_dsv4_rope_affine_pair {
    uint64_t row_bytes;
    uint64_t token_bytes;
    int32_t head_dim;
    int32_t n_dims;
    int32_t n_ctx_orig;
    int32_t inverse;
    uint32_t pos0;
    uint32_t pos_step;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
};

struct ds4_metal_args_dsv4_head_norm_rope {
    int32_t n_head;
    int32_t head_dim;
    int32_t head_dim4;
    int32_t n_dims;
    int32_t n_ctx_orig;
    int32_t pos0;
    int32_t inverse;
    float eps;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
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

// DS4 only calls the Metal RoPE helper in-place and uses the adjacent-pair
// layout (mode 0). The generic kernel above still copies the no-position
// prefix, even though source and destination alias, and dispatches enough
// lanes for the full head. This specialization maps lanes directly to the
// rotated pairs and deliberately never reads or writes the unchanged prefix.
// Keep the arithmetic below in the same order as the mode-0 branch above so
// the optimized and reference paths remain bit-identical.
kernel void kernel_dsv4_rope_tail_f32_inplace_pair(
        constant ds4_metal_args_dsv4_rope_tail & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint  tid   [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    if (args.mode != 0) {
        return;
    }

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
    device const char * src_base = src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01;
    device       char * dst_base = dst  + i3*args.nb3  + i2*args.nb2  + i1*args.nb1;

    // Keep each pair on the same SIMD lane as the generic in-place dispatch.
    // n_nope is 32-aligned for the supported DS4 shapes, so logical tail index
    // r ran on lane r%32 in the reference kernel. Compacting pairs would move
    // them to lane (r/2)%32 and can perturb fast-math code generation.
    for (int r = tid; r < args.n_dims; r += ntg.x) {
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

// The Q/K RoPE calls use the same position and scaling parameters for every
// head. Group four heads into one threadgroup so one 64-thread cohort computes
// the 32 adjacent-pair coefficients and the other cohorts reuse them. Keeping
// r on the same r%32 SIMD lane as the per-head specialization preserves the
// fast-math instruction mapping; only the redundant coefficient work changes.
kernel void kernel_dsv4_rope_tail_f32_inplace_pair_shared4(
        constant ds4_metal_args_dsv4_rope_tail & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        uint  tid   [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    if (args.mode != 0 || args.n_dims != 64) {
        return;
    }

    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    const uint cohort = tid >> 6;
    const int r = (int)(tid & 63u);
    threadgroup float cos_shared[32];
    threadgroup float sin_shared[32];

    device const int32_t * pos = (device const int32_t *) src1;
    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const int i2 = tgpig[1];
    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    if (cohort == 0 && (r & 1) == 0) {
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
        cos_shared[ic] = cos_theta;
        sin_shared[ic] = sin_theta;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int i1 = (int)tgpig[0]*4 + (int)cohort;
    if (i1 >= args.ne01 || (r & 1) != 0) {
        return;
    }

    const int i3 = tgpig[2];
    device const char * src_base = src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01;
    device       char * dst_base = dst  + i3*args.nb3  + i2*args.nb2  + i1*args.nb1;
    const int j0 = n_nope + r;
    const int j1 = j0 + 1;
    const float x0 = *((device const float *) (src_base + j0*args.nb00));
    const float x1 = *((device const float *) (src_base + j1*args.nb00));
    const float cos_theta = cos_shared[r/2];
    const float sin_theta = sin_shared[r/2];

    *((device float *) (dst_base + j0*args.nb0)) = x0*cos_theta - x1*sin_theta;
    *((device float *) (dst_base + j1*args.nb0)) = x0*sin_theta + x1*cos_theta;
}

// Fuses the per-head RMSNorm and partial Q RoPE while retaining the standalone
// norm reduction tree and the mode-0 RoPE lane mapping.
kernel void kernel_dsv4_head_rms_norm_rope_tail_f32(
        constant ds4_metal_args_dsv4_head_norm_rope & args,
        device char * xraw,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const uint head = tgpig.x;
    const uint tok = tgpig.y;
    device float4 * x4 = (device float4 *)xraw +
        ((uint64_t)tok * (uint64_t)args.n_head + head) *
        (uint64_t)args.head_dim4;

    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.head_dim4; i00 += ntg.x) {
        sumf += dot(x4[i00], x4[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = simd_sum(shmem_f32[tiisg]);
    const float scale = 1.0f / sqrt(sumf / args.head_dim + args.eps);
    const int n_nope = args.head_dim - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base,
                        args.beta_fast, args.beta_slow, corr_dims);
    const float theta_base = (float)(args.pos0 + (int)tok);
    const float inv_ndims = -1.0f / args.n_dims;
    device float * xs = (device float *)x4;

    for (int i0 = tpitg.x; i0 < args.head_dim; i0 += ntg.x) {
        if (i0 < n_nope) {
            xs[i0] = xs[i0] * scale;
            continue;
        }
        const int r = i0 - n_nope;
        if ((r & 1) != 0) {
            continue;
        }
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta =
            theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta =
            theta_base * pow(args.freq_base, inv_ndims * r);
#endif
        float cos_theta;
        float sin_theta;
        rope_yarn(theta, args.freq_scale, corr_dims, r,
                  args.ext_factor, args.attn_factor,
                  &cos_theta, &sin_theta);
        if (args.inverse) {
            sin_theta = -sin_theta;
        }

        const float x0 = xs[i0] * scale;
        const float x1 = xs[i0 + 1] * scale;
        xs[i0] = x0 * cos_theta - x1 * sin_theta;
        xs[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

// DS4 positions are always affine within one RoPE dispatch. This variant
// reconstructs the same wrapped int32 position in-kernel, avoiding the host
// position array and its buffer binding while preserving the pair lane mapping
// and all floating-point operations of the specialization above.

/* Shared, deliberately noinline so that every caller gets bit-identical
 * trigonometric codegen. The header note about tiny trig codegen changes
 * flipping sampled tokens is exactly why this body must be compiled once and
 * shared rather than inlined separately into each kernel. */
static __attribute__((noinline)) void ds4_rope_tail_pair_affine_row(
        constant ds4_metal_args_dsv4_rope_affine_pair & args,
        device const char * src_base,
        device char * dst_base,
        int n_nope,
        uint raw_pos,
        uint tid,
        uint nthreads) {
    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

        const float theta_base = (float)as_type<int>(raw_pos);
    const float inv_ndims = -1.f/args.n_dims;

    for (int r = tid; r < args.n_dims; r += nthreads) {
        if ((r & 1) != 0) {
            continue;
        }
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims*r);
#endif
        const float freq_factor = 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
        if (args.inverse) {
            sin_theta = -sin_theta;
        }

        const int j0 = n_nope + r;
        const int j1 = j0 + 1;
        const float x0 = *((device const float *) (src_base + j0*sizeof(float)));
        const float x1 = *((device const float *) (src_base + j1*sizeof(float)));

        *((device float *) (dst_base + j0*sizeof(float))) = x0*cos_theta - x1*sin_theta;
        *((device float *) (dst_base + j1*sizeof(float))) = x0*sin_theta + x1*cos_theta;
    }}

kernel void kernel_dsv4_rope_tail_f32_inplace_pair_affine(
        constant ds4_metal_args_dsv4_rope_affine_pair & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device       char * dst  [[buffer(4)]],
        uint  tid   [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int n_nope = args.head_dim - args.n_dims;
    if (n_nope < 0) {
        return;
    }
    const uint raw_pos = args.pos0 + (uint)i2 * args.pos_step;
    device const char * src_base =
        src0 + (uint64_t)i2*args.token_bytes + (uint64_t)i1*args.row_bytes;
    device char * dst_base =
        dst + (uint64_t)i2*args.token_bytes + (uint64_t)i1*args.row_bytes;
    ds4_rope_tail_pair_affine_row(args, src_base, dst_base, n_nope, raw_pos, tid, ntg.x);

}

// Decode-only fusion of the KV RoPE tail with the FP8/raw finalizer. Both were
// already single 64-thread threadgroups on the same row, back to back, so the
// pair cost two dispatches (~12.4 us) to touch 2 KB. The RoPE body below is a
// verbatim copy of kernel_dsv4_rope_tail_f32_inplace_pair_affine specialised to
// the decode grid (one head, one token, so i1 = i2 = 0) and the finalizer body
// is a verbatim copy of kernel_dsv4_kv_fp8_store_f32. The barrier between them
// is required because RoPE writes element pairs across lanes while the raw copy
// reads them per lane. Arithmetic, order and rounding are unchanged; the header
// warning above about trigonometric codegen still applies, so this kernel is
// gated and verified against full-vocabulary logits before promotion.
kernel void kernel_dsv4_kv_rope_fp8_store_f32(
        constant ds4_metal_args_dsv4_kv_fp8_store & args,
        constant ds4_metal_args_dsv4_rope_affine_pair & rope,
        device        float * kv,
        device        float * raw_cache,
        threadgroup   float * scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]]) {
    {
        const int rope_n_nope = rope.head_dim - rope.n_dims;
        if (rope_n_nope < 0) {
            return;
        }
        ds4_rope_tail_pair_affine_row(rope,
                                      (device const char *)kv,
                                      (device char *)kv,
                                      rope_n_nope,
                                      rope.pos0,
                                      tid,
                                      64u);
    }
    /* The RoPE helper writes device-memory pairs that different lanes read
     * below. A threadgroup-only fence does not make those cross-lane device
     * writes visible. */
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    {

    const int head_dim = args.head_dim;
    const int n_rot = args.n_rot;
    const int n_nope = head_dim - n_rot;
    if (head_dim <= 0 || n_rot < 0 || n_nope < 0 || tid >= 64) {
        return;
    }

    device float * raw = raw_cache + (int64_t)args.raw_row * head_dim;

    for (int off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + (int)tid < n_nope) {
            v = kv[off + tid];
            scratch[tid] = abs(v);
        } else {
            scratch[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float amax = max(scratch[0], 1.0e-4f);
        const float fp8_scale = exp2(ceil(log2(amax / 448.0f)));
        if (off + (int)tid < n_nope) {
            const float q = dsv4_e4m3fn_dequant(clamp(v / fp8_scale, -448.0f, 448.0f)) * fp8_scale;
            kv[off + tid] = q;
            // Diagnostic only: skip the FP16 round-trip that normally matches the
            // half-typed FlashAttention KV buffer's precision. With this enabled the
            // indexer will see higher-precision raw values than FlashAttention does,
            // which is informative but not a production-ready setting.
#ifdef DS4_METAL_KV_RAW_F32
            raw[off + tid] = q;
#else
            raw[off + tid] = (float)((half)q);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = n_nope + tid; i < head_dim; i += 64) {
#ifdef DS4_METAL_KV_RAW_F32
        raw[i] = kv[i];
#else
        raw[i] = (float)((half)kv[i]);
#endif
    }
    }
}

/* Decode-only sibling of kernel_flash_attn_ext_vec_reduce that also applies the
 * inverse RoPE tail to the row it just produced, removing a whole dispatch per
 * layer. Each threadgroup owns one head's entire 512-float row, so the RoPE is
 * an intra-threadgroup dependency: reduce, barrier, rotate. Both halves call the
 * same shared noinline helpers the standalone kernels use, so the arithmetic and
 * its codegen are identical to running the two dispatches back to back. */
kernel void kernel_flash_attn_ext_vec_reduce_rope(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        constant ds4_metal_args_dsv4_rope_affine_pair & rope,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    ds4_flash_attn_vec_reduce_row(args, htmp, dst, tgpig, tiisg, sgitg,
                                  (short)FC_flash_attn_ext_vec_reduce_NWG,
                                  (short)FC_flash_attn_ext_vec_reduce_DV);

    threadgroup_barrier(mem_flags::mem_device);

    const int n_nope = rope.head_dim - rope.n_dims;
    if (n_nope < 0) {
        return;
    }
    device char * row = dst + (uint64_t)tgpig * rope.row_bytes;
    ds4_rope_tail_pair_affine_row(rope,
                                  (device const char *)row,
                                  row,
                                  n_nope,
                                  rope.pos0,
                                  tiitg,
                                  (uint)(32 * FC_flash_attn_ext_vec_reduce_NWG));
}

struct ds4_metal_args_dsv4_comp_finalize {
    ds4_metal_args_dsv4_rope_affine_pair rope;
    float    rms_eps;
    uint32_t pad0;
};

/* Decode-only emit-path fusion.  Every ratio-th token, each layer finalizes
 * one freshly pooled compressor row per compressor: RMS norm, RoPE tail, and
 * then the FP8 round-trip + F16 commit copy (attention, 512 floats) or the
 * Hadamard+FP4 QAT (indexer, 128 floats).  Those were seven single-row
 * dispatches; this kernel is one dispatch with two threadgroups.
 *
 * Each phase reproduces its standalone kernel bit-exactly:
 *  - norm: kernel_rms_norm_mul_f32_4's tree (float4 lanes, simd_sum, zero-
 *    padded 32-slot cross-simdgroup reduce); 512 uses 128 virtual threads on
 *    simdgroups 0-3, 128 uses 32 virtual threads on simdgroup 0.
 *  - rope: ds4_rope_tail_pair_affine_row verbatim (lanes 0-63, nthreads=64).
 *  - fp8: kernel_dsv4_fp8_kv_quantize_f32's 64-lane shmem max tree and
 *    round-trip, src==dst so the verbatim tail copy is a no-op and dropped.
 *  - commit: per-element f32->f16 conversion (value-wise exact).
 *  - qat: kernel_dsv4_indexer_hadamard_fp4_f32's butterfly and per-32 amax
 *    tree on lanes 0-127.
 * Threads outside a phase's virtual width still execute every barrier, so
 * threadgroup barriers stay uniform across the 256-thread threadgroup. */
kernel void kernel_dsv4_comp_row_finalize_f32(
        constant ds4_metal_args_dsv4_comp_finalize & args [[buffer(0)]],
        device       float * attn_row     [[buffer(1)]],
        device const float * attn_norm_w  [[buffer(2)]],
        device       char  * attn_cache   [[buffer(3)]],
        device       float * index_row    [[buffer(4)]],
        device const float * index_norm_w [[buffer(5)]],
        device       float * attn_state_kv    [[buffer(6)]],
        device       float * attn_state_score [[buffer(7)]],
        device       float * index_state_kv    [[buffer(8)]],
        device       float * index_state_score [[buffer(9)]],
        threadgroup  float * shmem        [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constant ds4_metal_args_dsv4_rope_affine_pair & rope_args = args.rope;

    if (tgpig == 0) {
        /* -------- attention compressor row (512 floats) -------- */
        {
            device float4 * y4 = (device float4 *)attn_row;
            device const float4 * x4 = (device const float4 *)attn_row;
            device const float4 * w4 = (device const float4 *)attn_norm_w;
            if (sgitg == 0) {
                shmem[tiisg] = 0.0f;
            }
            float sumf = 0.0f;
            if (tiitg < 128) {
                sumf = dot(x4[tiitg], x4[tiitg]);
            }
            sumf = simd_sum(sumf);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tiitg < 128 && tiisg == 0) {
                shmem[sgitg] = sumf;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = 0.0f;
            if (tiitg < 128) {
                total = simd_sum(shmem[tiisg]);
            }
            const float mean  = total / 512.0f;
            const float scale = 1.0f/sqrt(mean + args.rms_eps);
            if (tiitg < 128) {
                y4[tiitg] = (x4[tiitg]*scale)*w4[tiitg];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        ds4_rope_tail_pair_affine_row(rope_args,
                                      (device const char *)attn_row,
                                      (device char *)attn_row,
                                      512 - rope_args.n_dims,
                                      rope_args.pos0,
                                      tiitg,
                                      64u);
        threadgroup_barrier(mem_flags::mem_device);
        for (int off = 0; off < 512 - rope_args.n_dims; off += 64) {
            float v = 0.0f;
            if (tiitg < 64) {
                v = attn_row[off + tiitg];
                shmem[tiitg] = abs(v);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 32; stride > 0; stride >>= 1) {
                if (tiitg < stride) {
                    shmem[tiitg] = max(shmem[tiitg], shmem[tiitg + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float amax = max(shmem[0], 1.0e-4f);
            const float scale = exp2(ceil(log2(amax / 448.0f)));
            if (tiitg < 64) {
                const float q = dsv4_e4m3fn_dequant(clamp(v / scale, -448.0f, 448.0f)) * scale;
                attn_row[off + tiitg] = q;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (tiitg < 128) {
            device const float4 * x4 = (device const float4 *)attn_row;
            device half4 * o4 = (device half4 *)attn_cache;
            const float4 v = x4[tiitg];
            o4[tiitg] = half4(v);
        }
        return;
    }

    if (tgpig >= 2u) {
        /* Ratio-4 state shifts for both compressors (elementwise row move,
         * so the flat gid mapping is bit-exact): 4*1024 attention elements
         * then 4*256 indexer elements. */
        const uint gid = (tgpig - 2u) * 256u + tiitg;
        const uint n0 = 4u * 1024u;
        if (gid < n0) {
            attn_state_kv[gid] = attn_state_kv[n0 + gid];
            attn_state_score[gid] = attn_state_score[n0 + gid];
            return;
        }
        const uint gid1 = gid - n0;
        const uint n1 = 4u * 256u;
        if (gid1 >= n1) return;
        index_state_kv[gid1] = index_state_kv[n1 + gid1];
        index_state_score[gid1] = index_state_score[n1 + gid1];
        return;
    }

    /* -------- indexer compressor row (128 floats) -------- */
    {
        device float4 * y4 = (device float4 *)index_row;
        device const float4 * x4 = (device const float4 *)index_row;
        device const float4 * w4 = (device const float4 *)index_norm_w;
        if (sgitg == 0) {
            shmem[tiisg] = 0.0f;
        }
        float sumf = 0.0f;
        if (tiitg < 32) {
            sumf = dot(x4[tiitg], x4[tiitg]);
        }
        sumf = simd_sum(sumf);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            shmem[sgitg] = sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        if (tiitg < 32) {
            total = simd_sum(shmem[tiisg]);
        }
        const float mean  = total / 128.0f;
        const float scale = 1.0f/sqrt(mean + args.rms_eps);
        if (tiitg < 32) {
            y4[tiitg] = (x4[tiitg]*scale)*w4[tiitg];
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    ds4_rope_tail_pair_affine_row(rope_args,
                                  (device const char *)index_row,
                                  (device char *)index_row,
                                  128 - rope_args.n_dims,
                                  rope_args.pos0,
                                  tiitg,
                                  64u);
    threadgroup_barrier(mem_flags::mem_device);
    {
        threadgroup float *vals = shmem;
        threadgroup float *absbuf = shmem + 128;
        if (tiitg < 128) {
            vals[tiitg] = index_row[tiitg];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < 128u; stride <<= 1u) {
            if (tiitg < 128 && (tiitg & stride) == 0u) {
                const uint base = (tiitg & ~(2u * stride - 1u)) + (tiitg & (stride - 1u));
                const float a = vals[base];
                const float b = vals[base + stride];
                vals[base] = a + b;
                vals[base + stride] = a - b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float v = 0.0f;
        if (tiitg < 128) {
            v = vals[tiitg] * 0.08838834764831845f;
            absbuf[tiitg] = abs(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint block = tiitg >> 5u;
        const uint lane = tiitg & 31u;
        const uint block_base = block * 32u;
        for (uint stride = 16u; stride > 0u; stride >>= 1u) {
            if (tiitg < 128 && lane < stride) {
                absbuf[block_base + lane] = max(absbuf[block_base + lane],
                                                absbuf[block_base + lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tiitg < 128) {
            const float amax = max(absbuf[block_base], 7.052966104933725e-38f);
            const float scale = exp2(ceil(log2(amax / 6.0f)));
            index_row[tiitg] = dsv4_e2m1fn_dequant(clamp(v / scale, -6.0f, 6.0f)) * scale;
        }
    }
}

struct ds4_metal_args_dsv4_comp_rows_update {
    ds4_metal_args_dsv4_rope_affine_pair rope;
    float    rms_eps;
    uint32_t head_dim;         /* 512 (attention) or 128 (indexer)          */
    uint32_t width;            /* state/batch row width: coff * head_dim    */
    uint32_t ratio;            /* 4 or 128                                  */
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t row_stride;       /* batch kv/score row stride in floats       */
    uint32_t ape_type;         /* 0 = F32, 1 = F16                          */
    uint32_t comp_row0;        /* first cache row this call appends         */
    uint32_t cache_row_stride; /* cache row stride in elements              */
    uint32_t out_f16;          /* 1: finalize in work row + commit to half  */
    uint32_t quant_mode;       /* 0 = FP8 E4M3 tail, 1 = Hadamard+FP4       */
    uint32_t capture_slots;    /* speculative prefix slots to snapshot      */
    uint32_t state_rows;       /* coff * ratio (8 or 128)                   */
};

/* Batch-path sibling of the decode finalize fusion above.  The speculative
 * verify path (n_tokens 2..6) and unaligned prefill chunks used to run a host
 * loop per token: one store dispatch, and on every ratio-th token a pool +
 * norm + rope + quantize + commit + shift chain plus two prefix-capture blits.
 * This kernel replays that whole per-layer loop in ONE single-threadgroup
 * dispatch.  One threadgroup because every token depends on the state the
 * previous token wrote (the compressor is a recurrence): sequencing inside one
 * threadgroup with device barriers is the only way to keep the chain on-GPU
 * without a dispatch per step.
 *
 * Each phase reproduces its standalone kernel bit-exactly:
 *  - store:  kernel_dsv4_compressor_store_one (elementwise, one lane per col).
 *  - pool r4:  ds4_dsv4_compressor_exact_pool_ratio4_col, the exact body the
 *    standalone exact-pool kernel runs, one column per simdgroup; its device
 *    volatile softmax/product round-trips through the SAME host scratch
 *    buffers, so the compiler cannot keep the normalized softmax in registers
 *    and the dispatch-boundary store/load semantics are preserved.
 *  - pool r128:  the GGML soft_max_f32_4(ne00=128, nth=32) -> mul ->
 *    sum_rows(nth=128) chain, one score column per simdgroup for the softmax
 *    and eight 128-thread cohorts for the sum_rows two-stage simd_sum tree.
 *    The transpose staging copy of the fallback is a bit-preserving load and
 *    is replaced by direct gathers; softmax and product still materialize in
 *    device scratch with volatile reloads at each former dispatch boundary.
 *  - norm/rope/fp8/commit/qat:  verbatim kernel_dsv4_comp_row_finalize_f32
 *    phases (same virtual widths; threads outside a phase's width still
 *    execute every barrier so barriers stay uniform); the rope tail uses the
 *    shared noinline helper with comp_pos = pos + 1 - ratio.
 *  - shift:  kernel_dsv4_ratio4_shift_f32 (elementwise row move).
 *  - capture:  elementwise copy of both state tensors into prefix slot t,
 *    replacing the two per-token blit encoders.
 * All loop bounds derive from args (no arrays sized by n_tokens), so the same
 * kernel serves verify chunks and long unaligned prefill chunks alike. */
[[max_total_threads_per_threadgroup(1024)]]
kernel void kernel_dsv4_comp_rows_update_f32(
        constant ds4_metal_args_dsv4_comp_rows_update & args [[buffer(0)]],
        device const float * batch_kv     [[buffer(1)]],
        device const float * batch_sc     [[buffer(2)]],
        device const char  * ape          [[buffer(3)]],
        device       float * state_kv     [[buffer(4)]],
        device       float * state_score  [[buffer(5)]],
        device const float * norm_w       [[buffer(6)]],
        device       float * pool_softmax [[buffer(7)]],
        device       float * pool_product [[buffer(8)]],
        device       float * work         [[buffer(9)]],
        device       half  * cache_f16    [[buffer(10)]],
        device       float * prefix_kv    [[buffer(11)]],
        device       float * prefix_score [[buffer(12)]],
        threadgroup  float * shmem        [[threadgroup(0)]],
        uint  tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint nth = 1024u;
    if (args.head_dim == 0u || args.width == 0u || args.ratio == 0u ||
        (args.head_dim & 31u) != 0u) {
        return;
    }

    uint emit_index = 0u;
    for (uint t = 0u; t < args.n_tokens; ++t) {
        const uint pos = args.pos0 + t;
        const uint pos_mod = pos % args.ratio;
        const bool emit = ((pos + 1u) % args.ratio) == 0u;

        /* ---- store: verbatim kernel_dsv4_compressor_store_one ---- */
        {
            const uint dst_row = args.ratio == 4u ? args.ratio + pos_mod
                                                  : pos_mod;
            device const float * kv_row =
                batch_kv + (uint64_t)t * args.row_stride;
            device const float * sc_row =
                batch_sc + (uint64_t)t * args.row_stride;
            for (uint gid = tiitg; gid < args.width; gid += nth) {
                const uint dst = dst_row * args.width + gid;
                const uint ape_i = pos_mod * args.width + gid;
                float ape_v;
                if (args.ape_type == 1u) {
                    ape_v = (float)(((device const half *)ape)[ape_i]);
                } else {
                    ape_v = ((device const float *)ape)[ape_i];
                }
                state_kv[dst] = kv_row[gid];
                state_score[dst] = sc_row[gid] + ape_v;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (emit) {
            device float * row = args.out_f16 != 0u
                ? work
                : work + (uint64_t)(args.comp_row0 + emit_index) *
                         args.cache_row_stride;

            /* ---- pool ---- */
            if (args.ratio == 4u) {
                /* One state column per simdgroup: each 32-lane simdgroup
                 * replays the standalone exact-pool kernel's 32-thread
                 * threadgroup, with a private 32-float scratch slice. */
                for (uint base = 0u; base < args.head_dim; base += 32u) {
                    ds4_dsv4_compressor_exact_pool_ratio4_col(
                            state_kv,
                            state_score,
                            pool_softmax,
                            pool_product,
                            row,
                            shmem + (uint)sgitg * 32u,
                            args.head_dim,
                            /* host dispatches replay=1, n_comp=1 */
                            1.0f,
                            0.0f,
                            base + (uint)sgitg,
                            (uint)tiisg);
                }
            } else {
                /* Ratio-128 pool: the fallback runs the generic GGML chain on
                 * a [head_dim][128] transposed scratch.  Reproduce each of its
                 * three dispatches. */
                const uint n_rows = args.ratio; /* 128 */
                /* (a) kernel_soft_max_f32_4(ne00=128, nth=32): one scratch row
                 * (= one state column) per simdgroup, one float4 per lane.
                 * No mask/sink sources, scale 1.0 -- keep the literal
                 * `x*scale + (float4)(0.0f)` arithmetic of the kernel. */
                const float sm_scale = 1.0f;
                for (uint base = 0u; base < args.head_dim; base += 32u) {
                    const uint col = base + (uint)sgitg;
                    float4 score4;
                    for (uint j = 0u; j < 4u; ++j) {
                        score4[j] = state_score[
                            (uint64_t)(4u * (uint)tiisg + j) * args.width + col];
                    }
                    device float4 * pdst4 = (device float4 *)
                        (pool_softmax + (uint64_t)col * n_rows);

                    float4 lmax4 = -INFINITY;
                    for (int i00 = (int)tiisg; i00 < (int)(n_rows / 4u);
                         i00 += 32) {
                        lmax4 = fmax(lmax4, score4 * sm_scale +
                                            (float4)(0.0f));
                    }
                    const float lmax =
                        MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));
                    const float max_val = simd_max(lmax);

                    float4 lsum4 = 0.0f;
                    for (int i00 = (int)tiisg; i00 < (int)(n_rows / 4u);
                         i00 += 32) {
                        const float4 exp_score4 =
                            exp((score4 * sm_scale + (float4)(0.0f)) - max_val);
                        lsum4 += exp_score4;
                        pdst4[i00] = exp_score4;
                    }
                    const float lsum =
                        lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];
                    threadgroup_barrier(mem_flags::mem_none);
                    const float sum = simd_sum(lsum);
                    const float inv_sum = 1.0f / sum;
                    for (int i00 = (int)tiisg; i00 < (int)(n_rows / 4u);
                         i00 += 32) {
                        pdst4[i00] *= inv_sum;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
                /* (b) elementwise multiply into the product scratch.  The
                 * fallback's mul reads the softmax from device after its own
                 * dispatch boundary: reload through volatile.  The KV operand
                 * was a bit-preserving transpose copy; gather it directly. */
                {
                    device volatile const float * reloaded_softmax =
                        (device volatile const float *)pool_softmax;
                    const uint total = args.head_dim * n_rows;
                    for (uint idx = tiitg; idx < total; idx += nth) {
                        const uint d = idx / n_rows;
                        const uint r = idx % n_rows;
                        float value = state_kv[(uint64_t)r * args.width + d];
                        value *= reloaded_softmax[idx];
                        pool_product[idx] = value;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
                /* (c) kernel_sum_rows_f32_f32(ne00=128, nth=128): eight
                 * 128-thread cohorts (four simdgroups each) replay the
                 * two-stage simd_sum tree with a private 32-float scratch. */
                {
                    device volatile const float * reloaded_product =
                        (device volatile const float *)pool_product;
                    const uint cohort = tiitg >> 7u;      /* 0..7  */
                    const uint ltid = tiitg & 127u;       /* 0..127 */
                    const uint lsg = (uint)sgitg & 3u;    /* 0..3  */
                    threadgroup float * sh = shmem + cohort * 32u;
                    for (uint base = 0u; base < args.head_dim; base += 8u) {
                        const uint d = base + cohort;
                        /* Each pass reuses the cohort scratch that simdgroups
                         * 1..3 of the cohort read at the end of the previous
                         * pass; the standalone kernel had one dispatch per
                         * row, so this ordering came for free there. */
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (lsg == 0u) {
                            sh[tiisg] = 0.0f;
                        }
                        float sumf = 0.0f;
                        sumf += reloaded_product[(uint64_t)d * n_rows + ltid];
                        sumf = simd_sum(sumf);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tiisg == 0) {
                            sh[lsg] = sumf;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        sumf = sh[tiisg];
                        sumf = simd_sum(sumf);
                        if (ltid == 0u) {
                            row[d] = sumf;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_device);

            /* ---- norm: verbatim comp_row_finalize phases ---- */
            if (args.head_dim == 512u) {
                device float4 * y4 = (device float4 *)row;
                device const float4 * x4 = (device const float4 *)row;
                device const float4 * w4 = (device const float4 *)norm_w;
                if (sgitg == 0) {
                    shmem[tiisg] = 0.0f;
                }
                float sumf = 0.0f;
                if (tiitg < 128) {
                    sumf = dot(x4[tiitg], x4[tiitg]);
                }
                sumf = simd_sum(sumf);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tiitg < 128 && tiisg == 0) {
                    shmem[sgitg] = sumf;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float total = 0.0f;
                if (tiitg < 128) {
                    total = simd_sum(shmem[tiisg]);
                }
                const float mean  = total / 512.0f;
                const float scale = 1.0f/sqrt(mean + args.rms_eps);
                if (tiitg < 128) {
                    y4[tiitg] = (x4[tiitg]*scale)*w4[tiitg];
                }
            } else {
                device float4 * y4 = (device float4 *)row;
                device const float4 * x4 = (device const float4 *)row;
                device const float4 * w4 = (device const float4 *)norm_w;
                if (sgitg == 0) {
                    shmem[tiisg] = 0.0f;
                }
                float sumf = 0.0f;
                if (tiitg < 32) {
                    sumf = dot(x4[tiitg], x4[tiitg]);
                }
                sumf = simd_sum(sumf);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tiisg == 0) {
                    shmem[sgitg] = sumf;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float total = 0.0f;
                if (tiitg < 32) {
                    total = simd_sum(shmem[tiisg]);
                }
                const float mean  = total / 128.0f;
                const float scale = 1.0f/sqrt(mean + args.rms_eps);
                if (tiitg < 32) {
                    y4[tiitg] = (x4[tiitg]*scale)*w4[tiitg];
                }
            }
            threadgroup_barrier(mem_flags::mem_device);

            /* ---- rope: shared noinline helper, comp_pos per emit ---- */
            ds4_rope_tail_pair_affine_row(args.rope,
                                          (device const char *)row,
                                          (device char *)row,
                                          (int)args.head_dim - args.rope.n_dims,
                                          pos + 1u - args.ratio,
                                          tiitg,
                                          64u);
            threadgroup_barrier(mem_flags::mem_device);

            /* ---- quantize ---- */
            if (args.quant_mode == 0u) {
                /* FP8 E4M3 round-trip on [0, head_dim - n_rot); the host gate
                 * guarantees a multiple of 64 so every block is full. */
                const int n_nope = (int)args.head_dim - args.rope.n_dims;
                for (int off = 0; off < n_nope; off += 64) {
                    float v = 0.0f;
                    if (tiitg < 64) {
                        v = row[off + tiitg];
                        shmem[tiitg] = abs(v);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = 32; stride > 0; stride >>= 1) {
                        if (tiitg < stride) {
                            shmem[tiitg] = max(shmem[tiitg],
                                               shmem[tiitg + stride]);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    const float amax = max(shmem[0], 1.0e-4f);
                    const float scale = exp2(ceil(log2(amax / 448.0f)));
                    if (tiitg < 64) {
                        const float q = dsv4_e4m3fn_dequant(
                            clamp(v / scale, -448.0f, 448.0f)) * scale;
                        row[off + tiitg] = q;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else {
                /* Hadamard + FP4: verbatim indexer QAT phase (128 lanes). */
                threadgroup float *vals = shmem;
                threadgroup float *absbuf = shmem + 128;
                if (tiitg < 128) {
                    vals[tiitg] = row[tiitg];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = 1u; stride < 128u; stride <<= 1u) {
                    if (tiitg < 128 && (tiitg & stride) == 0u) {
                        const uint base = (tiitg & ~(2u * stride - 1u)) +
                                          (tiitg & (stride - 1u));
                        const float a = vals[base];
                        const float b = vals[base + stride];
                        vals[base] = a + b;
                        vals[base + stride] = a - b;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float v = 0.0f;
                if (tiitg < 128) {
                    v = vals[tiitg] * 0.08838834764831845f;
                    absbuf[tiitg] = abs(v);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint block = tiitg >> 5u;
                const uint lane = tiitg & 31u;
                const uint block_base = block * 32u;
                for (uint stride = 16u; stride > 0u; stride >>= 1u) {
                    if (tiitg < 128 && lane < stride) {
                        absbuf[block_base + lane] =
                            max(absbuf[block_base + lane],
                                absbuf[block_base + lane + stride]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tiitg < 128) {
                    const float amax = max(absbuf[block_base],
                                           7.052966104933725e-38f);
                    const float scale = exp2(ceil(log2(amax / 6.0f)));
                    row[tiitg] = dsv4_e2m1fn_dequant(
                        clamp(v / scale, -6.0f, 6.0f)) * scale;
                }
            }
            threadgroup_barrier(mem_flags::mem_device);

            /* ---- commit: per-element f32->f16 (value-wise exact) ---- */
            if (args.out_f16 != 0u && tiitg < 128 && args.head_dim == 512u) {
                device const float4 * x4 = (device const float4 *)row;
                device half4 * o4 = (device half4 *)
                    (cache_f16 + (uint64_t)(args.comp_row0 + emit_index) *
                                 args.cache_row_stride);
                o4[tiitg] = half4(x4[tiitg]);
            }

            /* ---- shift: verbatim kernel_dsv4_ratio4_shift_f32 ---- */
            if (args.ratio == 4u) {
                const uint n0 = 4u * args.width;
                for (uint gid = tiitg; gid < n0; gid += nth) {
                    state_kv[gid] = state_kv[n0 + gid];
                    state_score[gid] = state_score[n0 + gid];
                }
            }
            emit_index++;
        }
        threadgroup_barrier(mem_flags::mem_device);

        /* ---- speculative prefix capture (replaces the two blits) ---- */
        if (args.capture_slots != 0u && t + 1u < args.n_tokens &&
            t < args.capture_slots) {
            const uint64_t n = (uint64_t)args.state_rows * args.width;
            device float * dkv = prefix_kv + (uint64_t)t * n;
            device float * dsc = prefix_score + (uint64_t)t * n;
            for (uint64_t i = tiitg; i < n; i += nth) {
                dkv[i] = state_kv[i];
                dsc[i] = state_score[i];
            }
            /* The next token's store overwrites the state these reads use. */
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
}

// Host-visible packed FlashAttention + exact inverse-RoPE decode kernel.
kernel void kernel_dsv4_flash_attn_vec_packed32_reduce_rope_f16_dk512_dv512(
        constant ds4_metal_args_flash_attn_ext_vec & args [[buffer(0)]],
        device const char * q                         [[buffer(1)]],
        device const char * k                         [[buffer(2)]],
        device const char * v                         [[buffer(3)]],
        device const char * mask                      [[buffer(4)]],
        device const char * sinks                     [[buffer(5)]],
        device const char * pad                       [[buffer(6)]],
        device       char * dst                       [[buffer(7)]],
        constant ds4_metal_args_dsv4_rope_affine_pair & rope
                                                        [[buffer(8)]],
        threadgroup char * shmem [[threadgroup(0)]],
        uint head       [[threadgroup_position_in_grid]],
        ushort tiitg    [[thread_index_in_threadgroup]],
        ushort tiisg    [[thread_index_in_simdgroup]],
        ushort sgitg    [[simdgroup_index_in_threadgroup]]) {
    /* Uniform specialization guard; host applies the same eligibility gate. */
    if (!FC_flash_attn_ext_vec_has_mask ||
        !FC_flash_attn_ext_vec_has_sinks ||
         FC_flash_attn_ext_vec_has_bias ||
         FC_flash_attn_ext_vec_has_scap ||
         FC_flash_attn_ext_vec_nsg != 1 ||
         FC_flash_attn_ext_vec_nwg != 32 ||
         FC_flash_attn_ext_vec_ns10 != 512 ||
         FC_flash_attn_ext_vec_ns20 != 512 ||
         args.ne01 != 1 || args.ne02 != 64 || args.ne03 != 1 ||
         args.ne_12_2 != 1 || args.ne_12_3 != 1 ||
         args.ne31 != 1 || args.ne32 != 1 || args.ne33 != 1 ||
         args.ne11 <= 0 || args.ne11 > 1024 || head >= (uint)args.ne02 ||
         args.nb02 != 2048 || args.nb11 != 1024 || args.nb21 != 1024 ||
         rope.head_dim != 512 || rope.n_dims != 64 ||
         rope.row_bytes != 2048 || rope.inverse == 0) {
        return;
    }

    ds4_flash_attn_vec_packed8_reduce_f16_512(
        args, q, k, v, mask, sinks, pad, dst, shmem,
        head, tiisg, sgitg);

    /* Same producer/consumer boundary as the current reduce+RoPE kernel. */
    threadgroup_barrier(mem_flags::mem_device);

    const int n_nope = rope.head_dim - rope.n_dims;
    device char * row = dst + (uint64_t)head * rope.row_bytes;
    ds4_rope_tail_pair_affine_row(rope,
                                  (device const char *)row,
                                  row,
                                  n_nope,
                                  rope.pos0,
                                  tiitg,
                                  32u * 32u);
}
