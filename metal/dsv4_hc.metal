struct ds4_metal_args_dsv4_hc_split_sinkhorn {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_weighted_sum {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum_norm {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float    eps;
    float    norm_eps;
};

struct ds4_metal_args_dsv4_hc_expand {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_add0;
    uint64_t nb_add1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t  has_add;
};

// Numerically stable sigmoid for the standalone split/sinkhorn path. The naive
// form 1/(1+exp(-z)) overflows for large negative z (exp(-z) blows up);
// replacing it with the 0.5*(tanh(z/2)+1) identity keeps the value bounded in
// [0, 1] across the entire float range. Gated by DS4_METAL_HC_STABLE so we can
// A/B vs the historical form on M5 Max where the faster ALU is more likely to
// push HC mixer inputs into the unstable regime.
//
// Do not automatically use these helpers in the fused HC decode kernels below:
// routing the fused vector sites through the tanh form produced non-finite
// logits on M5 Max, while the historical inline exp form remains finite and is
// the decode throughput baseline.
#ifdef DS4_METAL_HC_STABLE
static inline float  ds4_hc_sigmoid(float  z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
// 2 * sigmoid(z) == 1 + tanh(z/2).
static inline float  ds4_hc_twice_sigmoid(float  z) { return 1.0f + tanh(0.5f * z); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 1.0f + tanh(0.5f * z); }
#else
static inline float  ds4_hc_sigmoid(float  z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float  ds4_hc_twice_sigmoid(float  z) { return 2.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 2.0f / (1.0f + exp(-z)); }
#endif

// Splits an HC mixer row into pre weights, post gates, and the HC-to-HC
// combination matrix. The 4-channel path is specialized because DS4 Flash uses
// HC=4 in normal inference, while the scalar fallback keeps diagnostics usable.
kernel void kernel_dsv4_hc_split_sinkhorn(
        constant ds4_metal_args_dsv4_hc_split_sinkhorn & args,
        device  const float * mixes,
        device  const float * scale,
        device  const float * base,
        device        float * dst,
        uint tid [[thread_position_in_grid]]) {
    if ((int64_t) tid >= args.n_rows) {
        return;
    }

    constexpr int HC_MAX = 16;
    const int HC = args.n_hc;
    if (HC <= 0 || HC > HC_MAX) {
        return;
    }

    device const float * mix = mixes + ((int64_t) tid)*args.mix_hc;
    device       float * out = dst    + ((int64_t) tid)*args.mix_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (HC == 4) {
        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        *((device float4 *) out) = ds4_hc_sigmoid(pre_z) + epsv;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = ds4_hc_twice_sigmoid(post_z);

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
        return;
    }

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = ds4_hc_sigmoid(z) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = ds4_hc_twice_sigmoid(z);
    }

    float c[HC_MAX*HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const int off = 2*HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = max(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const float v = exp(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc*HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc*HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC*HC; ++i) {
        out[2*HC + i] = c[i];
    }
}

// Decode-side fusion of HC split and pre-weighted HC reduction. One threadgroup
// handles one token row: lane 0 computes the HC=4 mixer split once, stores the
// post/comb data for the following HC expand, and all lanes reuse the pre
// weights from threadgroup memory to produce the embedding row.
kernel void kernel_dsv4_hc_split_weighted_sum(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        threadgroup   float * pre_shmem [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if ((int64_t) row >= args.n_rows || args.n_hc != 4) {
        return;
    }

    device const float * mix = (device const float *) (mixes + (uint64_t)row*args.nb_mix1);
    device       float * out = (device       float *) (split + (uint64_t)row*args.nb_split1);

    if (tid == 0) {
        const float epsv       = args.eps;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *) out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int64_t d = tid; d < args.n_embd; d += ntg) {
        float acc = 0.0f;
        acc += *((device const float *) (x + d*args.nb_x0 + 0*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[0];
        acc += *((device const float *) (x + d*args.nb_x0 + 1*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[1];
        acc += *((device const float *) (x + d*args.nb_x0 + 2*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[2];
        acc += *((device const float *) (x + d*args.nb_x0 + 3*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[3];
        *((device float *) (dst + d*args.nb0 + (uint64_t)row*args.nb1)) = acc;
    }
}

// Decode HC-pre plus the following RMSNorm.  DS4 always uses HC=4 and a
// 4096-wide sublayer row.  The normal release path computes HC coefficients,
// collapses four residual streams into that row, then immediately launches a
// weighted RMSNorm over the row.  This kernel keeps the HC split math identical
// to kernel_dsv4_hc_split_weighted_sum, stores the HC-pre row for diagnostics,
// and reuses the just-collapsed values from threadgroup memory for the RMSNorm
// reduction.  The reduction mirrors kernel_rms_norm_mul_f32_4's 1024-thread
// float4 shape for a 4096-wide row.
kernel void kernel_dsv4_hc_split_weighted_sum_norm4(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        device  const char  * norm_weight,
        device        char  * norm_dst,
        threadgroup   float * shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if ((int64_t)row >= args.n_rows || args.n_hc != 4 || args.n_embd != 4096) {
        return;
    }

    threadgroup float4 *row_shmem = (threadgroup float4 *)shared;
    threadgroup float *pre_shmem = shared + 4096;
    threadgroup float *sum_shmem = pre_shmem + 4;

    device const float *mix = (device const float *)(mixes + (uint64_t)row * args.nb_mix1);
    device float *out = (device float *)(split + (uint64_t)row * args.nb_split1);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = args.eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf = 0.0f;
    const uint n4 = 1024u;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x1 = (device const float4 *)(x + 1 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x2 = (device const float4 *)(x + 2 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x3 = (device const float4 *)(x + 3 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        const float4 v = x0[i] * pre_shmem[0] +
                         x1[i] * pre_shmem[1] +
                         x2[i] * pre_shmem[2] +
                         x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    const float norm_scale = rsqrt(sumf / 4096.0f + args.norm_eps);

    device float4 *dst4 = (device float4 *)(dst + (uint64_t)row * args.nb1);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(norm_dst + (uint64_t)row * args.nb_norm1);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}

// Expands an embedding-sized block back into HC channels after attention/FFN.
// The post gate scales the current block, while the Sinkhorn combination matrix
// mixes residual HC channels from the previous state.
kernel void kernel_dsv4_hc_expand(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }
    const float post_v  = *((device const float *) (post      + dst_hc*args.nb_post0 + t*args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((device const float *) (comb     + dst_hc*args.nb_comb0 + src_hc*args.nb_comb1 + t*args.nb_comb2));
        const float res_v  = *((device const float *) (residual + d*args.nb_res0 + src_hc*args.nb_res1 + t*args.nb_res2));
        acc += comb_v * res_v;
    }

    *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
}

// HC=4 specialization of the post/expand step. One thread computes all four
// destination HC streams for one token/dimension, reusing the same block output
// and residual HC values while preserving the per-stream accumulation order.
kernel void kernel_dsv4_hc_expand4(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }

    const float r0 = *((device const float *) (residual + d*args.nb_res0 + 0*args.nb_res1 + t*args.nb_res2));
    const float r1 = *((device const float *) (residual + d*args.nb_res0 + 1*args.nb_res1 + t*args.nb_res2));
    const float r2 = *((device const float *) (residual + d*args.nb_res0 + 2*args.nb_res1 + t*args.nb_res2));
    const float r3 = *((device const float *) (residual + d*args.nb_res0 + 3*args.nb_res1 + t*args.nb_res2));

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * *((device const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1));

        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)) * r0;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)) * r1;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)) * r2;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)) * r3;

        *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
    }
}

// Decode-time FFN tail fusion:
//
//     shared_out = shared_mid @ Wshared_down
//     after_ffn_hc = HCPost(routed_out + shared_out, residual_hc, split)
//
// The Q8_0 dot reduction is intentionally copied from the normal matvec shape
// so the shared expert result is bit-identical.  The only specialization is
// that DS4 decode has one token and HC=4, so the thread that finishes each
// shared-down output row can immediately expand it into the four HC streams.
kernel void kernel_dsv4_shared_down_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * shared_mid,
        device        char * shared_out,
        device  const char * routed_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(shared_mid);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float shared_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(shared_out + (uint64_t)d * sizeof(float))) = shared_v;

            float block_v = *((device const float *)(routed_out + (uint64_t)d * hc.nb_block0));
            block_v += shared_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// Decode-time attention output tail fusion:
//
//     attn_out = attn_low @ Wob
//     after_attn_hc = HCPost(attn_out, residual_hc, split)
//
// This is the no-add sibling of the shared-down/FFN fusion above.  It preserves
// the exact Q8_0 matvec reduction, stores `attn_out` for diagnostics, and then
// writes the four HC streams for the same embedding dimension.
kernel void kernel_dsv4_q8_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * input,
        device        char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(input);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float block_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = block_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// Reduces HC channels to a normal embedding row with the learned pre weights.
// This is the input adapter before the attention block and before the FFN block.
kernel void kernel_dsv4_hc_weighted_sum(
        constant ds4_metal_args_dsv4_hc_weighted_sum & args,
        device  const char * x,
        device  const char * weights,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((device const float *) (x       + d*args.nb_x0 + h*args.nb_x1 + t*args.nb_x2));
        const float wv = *((device const float *) (weights + h*args.nb_w0 + t*args.nb_w1));
        acc += xv * wv;
    }

    *((device float *) (dst + d*args.nb0 + t*args.nb1)) = acc;
}
