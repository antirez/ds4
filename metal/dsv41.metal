// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

/* Decode-time V4.1 hyper-connection glue, one token row, HC=4.
 *
 * The released graph runs each half-layer's HC work as separate dispatches:
 * split/sinkhorn, the pre-weighted collapse of the four streams, a BF16
 * rounding pass, the weighted RMSNorm and another BF16 pass (and, after the
 * sublayer, the post/comb expand plus its BF16 pass).  DeepSeek's production
 * decode does the whole thing in one "Mega-mHC" kernel.  These two kernels
 * are that fusion for ds4's graph: every reduction keeps the standalone
 * kernel's thread mapping and accumulation order, and every rounding point
 * is the same dsv41_bf16, so the outputs are byte-identical (checked by
 * tests/test_deepseek41_metal --hc-fuse). */
struct ds4_metal_args_dsv41_hc {
    uint  n_embd;
    uint  sinkhorn_iters;
    float hc_eps;
    float norm_eps;
    uint  copy_pre;
    uint  has_add;
};

static inline float4 dsv41_bf16x4(float4 v) {
    return float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
}

/* kernel_dsv4_hc_split_sinkhorn's HC == 4 body, verbatim, on one lane. */
static inline void dsv41_hc_split4(device const float *mix, device const float *scale,
                                   device const float *base, device float *out,
                                   uint sinkhorn_iters, float epsv) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    const float4 pre_z = *((device const float4 *)mix) * pre_scale + *((device const float4 *)base);
    *((device float4 *)out) = ds4_hc_sigmoid(pre_z) + epsv;

    const float4 post_z = *((device const float4 *)(mix + 4)) * post_scale + *((device const float4 *)(base + 4));
    *((device float4 *)(out + 4)) = ds4_hc_twice_sigmoid(post_z);

    float4 r0 = *((device const float4 *)(mix +  8)) * comb_scale + *((device const float4 *)(base +  8));
    float4 r1 = *((device const float4 *)(mix + 12)) * comb_scale + *((device const float4 *)(base + 12));
    float4 r2 = *((device const float4 *)(mix + 16)) * comb_scale + *((device const float4 *)(base + 16));
    float4 r3 = *((device const float4 *)(mix + 20)) * comb_scale + *((device const float4 *)(base + 20));

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

    for (uint iter = 1; iter < sinkhorn_iters; ++iter) {
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

    *((device float4 *)(out +  8)) = r0;
    *((device float4 *)(out + 12)) = r1;
    *((device float4 *)(out + 16)) = r2;
    *((device float4 *)(out + 20)) = r3;
}

/* split(mix) -> split_out; x = bf16(sum_h pre[h] * residual[h]); norm =
 * bf16(rmsnorm(x) * weight).  `pre` is the coefficient row the previous
 * sublayer produced (V4.1's single-pass mHC), not this split's.  One
 * threadgroup of ds4_gpu_rms_norm_threads(n_embd) threads: the collapse
 * keeps kernel_dsv4_hc_weighted_sum's per-stream order and the norm keeps
 * kernel_rms_norm_mul_f32_4's loop, simd tree and 1/sqrt scale. */
kernel void kernel_dsv41_hc_collapse_norm4(
        constant ds4_metal_args_dsv41_hc &args,
        device const float *mix,
        device const float *scale,
        device const float *base,
        device const float *pre,
        device const float4 *residual,
        device       float *split_out,
        device       float4 *x_out,
        device const float4 *norm_weight,
        device       float4 *norm_out,
        threadgroup  float *shared [[threadgroup(0)]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    const uint n4 = args.n_embd >> 2;
    threadgroup float4 *row = (threadgroup float4 *)shared;
    threadgroup float *sums = shared + args.n_embd;

    if (tid == 0 && args.sinkhorn_iters) dsv41_hc_split4(mix, scale, base, split_out, args.sinkhorn_iters, args.hc_eps);
    if (sgitg == 0) sums[tiisg] = 0.0f;

    const float4 p = *((device const float4 *)pre);
    device const float4 *x0 = residual;
    device const float4 *x1 = residual + n4;
    device const float4 *x2 = residual + 2u * n4;
    device const float4 *x3 = residual + 3u * n4;
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        float4 v = 0.0f;
        v += x0[i] * p.x;
        v += x1[i] * p.y;
        v += x2[i] * p.z;
        v += x3[i] * p.w;
        v = dsv41_bf16x4(v);
        row[i] = v;
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) sums[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = simd_sum(sums[tiisg]);

    const float mean = sumf / (float)args.n_embd;
    const float norm_scale = 1.0f / sqrt(mean + args.norm_eps);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row[i];
        x_out[i] = v;
        norm_out[i] = dsv41_bf16x4((v * norm_scale) * norm_weight[i]);
    }
}

/* out[h] = bf16(post[h] * block + sum_s comb[h, s] * residual[s]) for the
 * four streams, kernel_dsv4_hc_expand4's index arithmetic and accumulation
 * order; copy_pre also carries split[0..3] into `pre` for the next layer.
 * has_add: block = bf16(block_in + add) first (the routed + shared sum and
 * its rounding), written back to block_sum. */
kernel void kernel_dsv41_hc_expand4_bf16(
        constant ds4_metal_args_dsv41_hc &args,
        device const float *block_in,
        device const float *residual,
        device const float *split,
        device       float *out,
        device       float *pre_out,
        device const float *add,
        device       float *block_sum,
        uint d [[thread_position_in_grid]]) {
    if (d >= args.n_embd) return;
    device const float *post = split + 4;
    device const float *comb = split + 8;

    float block_v = block_in[d];
    if (args.has_add) {
        block_v += add[d];
        block_v = dsv41_bf16(block_v);
        block_sum[d] = block_v;
    }
    const float r0 = residual[d];
    const float r1 = residual[d + args.n_embd];
    const float r2 = residual[d + 2u * args.n_embd];
    const float r3 = residual[d + 3u * args.n_embd];

    for (uint dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * post[dst_hc];
        acc += comb[dst_hc + 0u * 4u] * r0;
        acc += comb[dst_hc + 1u * 4u] * r1;
        acc += comb[dst_hc + 2u * 4u] * r2;
        acc += comb[dst_hc + 3u * 4u] * r3;
        out[d + dst_hc * args.n_embd] = dsv41_bf16(acc);
    }
    if (args.copy_pre && d < 4) pre_out[d] = split[d];
}

/* Single-row matvecs that round to BF16 on the store: the V4.1 graph rounds
 * almost every projection output, as a separate dispatch over the result.
 * Bodies are kernel_mul_mv_q8_0_f32_impl and kernel_mul_mv_t_t_4_impl
 * <half, half4, float, float4> with helper_mv_reduce_and_write's tree; only
 * the store changes (tests/test_deepseek41_metal --attn-fuse). */
template<short NR0>
static inline void dsv41_mv_reduce_write_bf16(
        device float * dst_f32, float sumf[NR0], const int r0, const int ne01,
        ushort tiisg, ushort sgitg, threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;
    threadgroup float * shmem_f32[NR0];
    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;
        if (sgitg == 0) shmem_f32[row][tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) dst_f32[r0 + row] = dsv41_bf16(tot);
    }
}

kernel void kernel_dsv41_mul_mv_q8_0_f32_bf16(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = N_R0_Q8_0;
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }
    float sumf[NR0] = { 0.f };
    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);
    const int ib0 = sgitg*NQ + ix;
    float yl[NQ];
    device const float * yb = y + ib0*QK8_0 + il*NQ;
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) sumq += qs[i] * yl[i];
            sumf[row] += sumq*ax[row][ib].d;
        }
        yb += NSG*NQ*QK8_0;
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    dsv41_mv_reduce_write_bf16<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<short NR0>
void dsv41_mul_mv_f16_f32_4_bf16_impl(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;
    const int nb = args.ne00/NB;
    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float  * y  = (device const float  *) (src1 + offset1);
    device const float4 * y4 = (device const float4 *) (src1 + offset1);
    device const half  * ax [NR0];
    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
        ax [row] = (device const half  *) ((device char *) src0 + offset0);
        ax4[row] = (device const half4 *) ((device char *) src0 + offset0);
    }
    float sumf[NR0] = { 0.f };
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;
    float4 yl4[NF4];
    device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;
    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) yl4[i] = yb4[i];
        for (short row = 0; row < NR0; row++) {
            device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) sumq += dot(float4(xb4[i]), float4(yl4[i]));
            sumf[row] += sumq;
        }
        yb4 += NSG*NF*NW/4;
    }
    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) sumf[row] += ax[row][i] * y[i];
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    dsv41_mv_reduce_write_bf16<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

kernel void kernel_dsv41_mul_mv_f16_f32_4_bf16(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    switch (args.nr0) {
        case 2: dsv41_mul_mv_f16_f32_4_bf16_impl<2>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: dsv41_mul_mv_f16_f32_4_bf16_impl<4>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}

/* BF16 rounding of a whole row followed by the adjacent-pair RoPE on its
 * last 64 values (kernel_dsv41_rope), one simdgroup per row: what the graph
 * ran as a rounding pass over all heads plus the inverse RoPE. */
kernel void kernel_dsv41_bf16_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = ((ulong)group.y * args.heads + group.x) * args.width;
    for (uint i = lane; i < args.width; i += 32u) x[row + i] = dsv41_bf16(x[row + i]);
    simdgroup_barrier(mem_flags::mem_device);
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = row + args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

/* The attention projections' normalization and the KV tail in one dispatch:
 * threadgroup y=0 is kernel_rms_norm_mul_f32_4 over the q LoRA row plus its
 * BF16 rounding; y=1 the same over the 512-wide KV row, then the RoPE on
 * its last 64 values, the FP8 (E8M0-scaled) block quantization of
 * kernel_dsv41_quantize mode 1 and the store into the raw window slot.
 * The threadgroup is the q row's norm thread count; the KV lanes beyond its
 * own count hold no elements, so their zero partials leave the standalone
 * reduction tree unchanged. */
struct ds4_metal_args_dsv41_qkv_tail {
    uint  q_n;
    uint  kv_n;
    float eps;
    uint  pos;
    float frequencies[32];
};

kernel void kernel_dsv41_qkv_norm_kv_tail(
        constant ds4_metal_args_dsv41_qkv_tail &args,
        device float4 *q,
        device const float4 *q_weight,
        device float4 *kv,
        device const float4 *kv_weight,
        device float *window_row,
        threadgroup float *shared [[threadgroup(0)]],
        ushort task [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    const bool kv_task = task != 0;
    const uint n = kv_task ? args.kv_n : args.q_n;
    const uint n4 = n >> 2;
    device float4 *x = kv_task ? kv : q;
    device const float4 *w = kv_task ? kv_weight : q_weight;
    threadgroup float4 *row = (threadgroup float4 *)shared;
    threadgroup float *sums = shared + 4u * n4;

    if (sgitg == 0) sums[tiisg] = 0.0f;
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = x[i];
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) sums[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = simd_sum(sums[tiisg]);
    const float mean = sumf / (float)n;
    const float scale = 1.0f / sqrt(mean + args.eps);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = dsv41_bf16x4((x[i] * scale) * w[i]);
        row[i] = v;
        if (!kv_task) x[i] = v;
    }
    if (!kv_task) return;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *rowf = (threadgroup float *)row;
    if (sgitg == 0) {
        const float theta = float(args.pos) * args.frequencies[tiisg];
        const float c = precise::cos(theta);
        const float s = precise::sin(theta);
        const uint i = n - 64u + 2u * tiisg;
        const float re = rowf[i], im = rowf[i + 1u];
        rowf[i] = dsv41_bf16(re * c - im * s);
        rowf[i + 1u] = dsv41_bf16(re * s + im * c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nsg = (ntg + 31u) / 32u;
    for (uint block = sgitg; block < n / 32u; block += nsg) {
        const uint column = block * 32u + tiisg;
        const float value = dsv41_bf16(rowf[column]);
        const float amax = simd_max(abs(value));
        const float qscale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        const float result = copysign(dsv4_e4m3fn_dequant(abs(value) / qscale), value) * qscale;
        const float out = dsv41_bf16(result);
        ((device float *)kv)[column] = out;
        window_row[column] = out;
    }
}

/* Decode-time V4.1 MoE glue, one token row.  Same discipline as the HC
 * kernels above: every reduction keeps the standalone kernel's shape and
 * every BF16 rounding point is kept (tests/test_deepseek41_metal --moe-fuse). */

struct ds4_metal_args_dsv41_router {
    uint  n_embd;
    uint  n_expert;
    uint  n_used;
    uint  has_bias;
    float scale;
    uint  fused_matvec;
};

/* Router logits (F32 matvec, kernel_mul_mv_f32_f32_4 with nsg=8, nr0=2) and
 * the generic select sequence the V4.1 graph ran as ten dispatches:
 * probs = sqrt(softplus(logits)); score = probs + bias; top-k in the
 * canonical (score desc, idx asc) order of kernel_argsort_f32_i32_desc_canon;
 * weights = (probs[sel] / max(sum, 2^-14)) * scale, where sum is
 * kernel_sum_rows_f32_f32's six-lane simd reduction.
 * fused_matvec: the last-arriving threadgroup selects (DeepSeek's Mega-Gate
 * shape); the host enables it on M5 only, like V4's fused router: on the
 * M3 Ultra the other groups' logits were not reliably visible to the last
 * one (nondeterministic probs).  Otherwise one threadgroup selects from
 * logits produced by the standalone matvec dispatch. */
kernel void kernel_dsv41_router_select(
        constant ds4_metal_args_dsv41_router &args,
        device const float *weight,
        device const float *x,
        device float *logits,
        device float *probs,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device atomic_uint *completion,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (args.fused_matvec) {
        constexpr short NSG = 8;
        constexpr short NR0 = 2;
        constexpr short NB  = 32;
        constexpr short NF  = 16;
        constexpr short NF4 = NF/4;
        constexpr short NW  = N_SIMDWIDTH;
        const int nb = (int)args.n_embd/NB;
        const int r0 = tgpig.x*NR0;
        device const float4 *y4 = (device const float4 *)x;
        device const float4 *ax4[NR0];
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            ax4[row] = (device const float4 *)(weight + (uint64_t)(r0 + row)*args.n_embd);
        }
        float sumf[NR0] = {0.f};
        const short ix = tiisg/(NW/NF);
        const short il = tiisg%(NW/NF);
        const int ib0 = sgitg*NF + ix;
        device const float4 *yb4 = y4 + (ib0*NB + il*NF)/4;
        for (int ib = ib0; ib < nb; ib += NSG*NF) {
            float4 yl4[NF4];
            FOR_UNROLL (short i = 0; i < NF4; ++i) yl4[i] = yb4[i];
            FOR_UNROLL (short row = 0; row < NR0; ++row) {
                device const float4 *xb4 = ax4[row] + (ib*NB + il*NF)/4;
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NF4; ++i) sumq += dot(float4(xb4[i]), float4(yl4[i]));
                sumf[row] += sumq;
            }
            yb4 += NSG*NF*NW/4;
        }
        for (int i = nb*NB + sgitg*NW + tiisg; i < (int)args.n_embd; i += NW*NSG) {
            FOR_UNROLL (short row = 0; row < NR0; ++row) {
                sumf[row] += ((device const float *)ax4[row])[i] * x[i];
            }
        }
        helper_mv_reduce_and_write<NR0>(logits, sumf, r0, (int)args.n_expert,
                                        tiisg, sgitg, (threadgroup char *)scratch);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (tid == 0) {
            const uint old = atomic_fetch_add_explicit(completion, 1u, memory_order_relaxed);
            scratch[0] = old + 1u == (args.n_expert + 1u) / 2u ? 1.0f : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (scratch[0] == 0.0f) return;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }

    constexpr uint SLOTS = 512;
    threadgroup float *score_tg = scratch;                          /* SLOTS */
    threadgroup float *red_s = scratch + SLOTS;                     /* 8 */
    threadgroup int32_t *red_i = (threadgroup int32_t *)(scratch + SLOTS + 8);   /* 8 */
    threadgroup int32_t *sel_tg = (threadgroup int32_t *)(scratch + SLOTS + 16); /* 8 */
    threadgroup volatile float *stage = scratch + SLOTS + 24;      /* 8 */

    device volatile const float *lg = (device volatile const float *)logits;
    for (uint i = tid; i < SLOTS; i += 256u) {
        float s = -INFINITY;
        if (i < args.n_expert) {
            const float v = lg[i];
            const float sp = select(log(1.0f + exp(v)), v, v > 20.0f);
            const float p = sqrt(sp);
            probs[i] = p;
            s = args.has_bias ? p + bias[i] : p;
        }
        score_tg[i] = s;
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    float s0 = score_tg[tid], s1 = score_tg[tid + 256u];
    for (uint r = 0; r < args.n_used; r++) {
        float bs = s0;
        int32_t bi = (int32_t)tid;
        if (s1 > bs) { bs = s1; bi = (int32_t)tid + 256; }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ps = simd_shuffle_xor(bs, o);
            const int32_t pi = simd_shuffle_xor(bi, o);
            if (ps > bs || (ps == bs && pi < bi)) { bs = ps; bi = pi; }
        }
        if (tiisg == 0) { red_s[sgitg] = bs; red_i[sgitg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float best = red_s[0];
            int32_t best_i = red_i[0];
            for (uint g = 1; g < 8; g++) {
                if (red_s[g] > best || (red_s[g] == best && red_i[g] < best_i)) {
                    best = red_s[g]; best_i = red_i[g];
                }
            }
            sel_tg[r] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int32_t sel = sel_tg[r];
        if ((int32_t)tid == sel) s0 = -INFINITY;
        if ((int32_t)tid + 256 == sel) s1 = -INFINITY;
    }

    /* kernel_sum_rows_f32_f32 runs six threads: lanes 0..5 hold one weight
     * each and the simdgroup reduces them; then (w / clamp(sum)) * scale as
     * two separately rounded ops (the div-row and mul-scalar kernels). */
    const bool lane = sgitg == 0 && tiisg < args.n_used;
    device volatile const float *pr = (device volatile const float *)probs;
    const float w = lane ? pr[sel_tg[tiisg]] : 0.0f;
    float sum = simd_sum(w);
    if (sgitg == 0 && tiisg == 0) {
        sum = clamp(sum, 6.103515625e-5f, INFINITY);
        stage[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < args.n_used) {
        stage[1 + tid] = w / stage[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < args.n_used) {
        selected[tid] = sel_tg[tid];
        weights[tid] = stage[1 + tid] * args.scale;
    }
    if (args.fused_matvec) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (tid == 0) atomic_store_explicit(completion, 0u, memory_order_relaxed);
    }
}

/* Shared expert gate/up (Q8_0, kernel_mul_mv_q8_0_f32's walk and reduction
 * tree at the dispatch's nsg) with V4.1's rounding: gate and up are rounded
 * to BF16 before SwiGLU and the product after it. */
kernel void kernel_dsv41_shared_gate_up_swiglu_q8_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_mid,
        constant     float &clamp_value,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = N_R0_Q8_0;
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    device const float *y = (device const float *)src1;

    device const block_q8_0 *ag[NR0];
    device const block_q8_0 *au[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01;
        ag[row] = (device const block_q8_0 *)(src0_gate + offset0);
        au[row] = (device const block_q8_0 *)(src0_up   + offset0);
    }

    float sumg[NR0] = { 0.f };
    float sumu[NR0] = { 0.f };
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) yl[i] = yb[i];
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const int8_t *qg = ag[row][ib].qs + il * NQ;
            device const int8_t *qu = au[row][ib].qs + il * NQ;
            float sg = 0.f;
            float su = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sg += qg[i] * yl[i];
                su += qu[i] * yl[i];
            }
            sumg[row] += sg * ag[row][ib].d;
            sumu[row] += su * au[row][ib].d;
        }
        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32 = (threadgroup float *)shmem;
    threadgroup float *sh_gate[NR0];
    threadgroup float *sh_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sh_gate[row] = shmem_f32 + NW * row;
        sh_up[row]   = shmem_f32 + NW * (NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *mid_f32 = (device float *)dst_mid;
    FOR_UNROLL (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        const float gate = simd_sum(sh_gate[row][tiisg]);
        const float up = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            float g = dsv41_bf16(gate);
            float u = dsv41_bf16(up);
            if (clamp_value > 1.0e-6f) {
                g = min(g, clamp_value);
                u = clamp(u, -clamp_value, clamp_value);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[r0 + row] = dsv41_bf16(silu * u);
        }
    }
}

