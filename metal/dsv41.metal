// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
// Construct scaled IEEE bits instead of approximating exp2/pow. An exponent
// overflow or discarded significand bit must not silently round source weights.
static uint dsv41_native_bits(uint magnitude, int power, uint scale, uint sign) {
    if (scale == 255u) return 0x7fc00000u;
    if (!magnitude) return sign;
    uint top = 31u - clz(magnitude);
    int exponent = int(top) + power + int(scale);
    uint significand = magnitude << (23u - top);
    if (exponent >= 255) return 0x7fc00000u;
    if (exponent <= 0) {
        uint shift = uint(1 - exponent);
        if (shift >= 32u || (significand & ((1u << shift) - 1u))) return 0x7fc00000u;
        return sign | (significand >> shift);
    }
    return sign | (uint(exponent) << 23u) | (significand & 0x7fffffu);
}

// Widen once per prefill layer. Preserve exact F32 bits, including signed
// zero, and flag any value that BF16 cannot represent without rounding.
kernel void kernel_dsv41_native_expand_mxfp4_bf16(
        constant ulong &count [[buffer(0)]], device const uchar *packed [[buffer(1)]],
        device ushort *out [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const ulong i = (ulong(group.y) * 65535u + group.x) * 256u + tid;
    if (i >= count) return;
    const ulong block = (i / 32u) * 17u;
    const uint code = (packed[block + 1u + (i % 16u)] >> ((i % 32u >= 16u) ? 4u : 0u)) & 15u;
    constexpr uint magnitude[8] = {0u, 1u, 2u, 3u, 4u, 6u, 8u, 12u};
    const uint bits = dsv41_native_bits(magnitude[code & 7u], -1, packed[block], (code & 8u) << 28u);
    if (bits & 0xffffu) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    out[i] = ushort(bits >> 16u);
}


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

// Compare in the scaled domain: Metal division can move an exact midpoint
// by an ULP even in precise mode. Products of E4M3 and E2M1 fit exactly in F32.
static float dsv41_fp4_scaled(float value, float scale) {
    const float ax=abs(value);
    int best=0;float distance=ax;
    for(int i=1;i<8;i++) {
        const float d=abs(ax-scale*dsv4_e2m1fn_values[i]);
        if(d<distance || (d==distance && !(i&1) && (best&1))) {best=i;distance=d;}
    }
    return copysign(scale*dsv4_e2m1fn_values[best],value);
}

static float dsv41_fp4_e4m3_scale(float amax) {
    const float ax=max(amax,0.01171875f);
    int lo=0,hi=126;
    while(lo<hi) {
        const int mid=(lo+hi+1)>>1;
        if(6.f*dsv4_e4m3fn_value(mid)<=ax)lo=mid;else hi=mid-1;
    }
    int best=lo;
    if(best<126) {
        const float d0=abs(ax-6.f*dsv4_e4m3fn_value(best));
        const float d1=abs(ax-6.f*dsv4_e4m3fn_value(best+1));
        if(d1<d0 || (d1==d0 && !((best+1)&1) && (best&1)))best++;
    }
    return dsv4_e4m3fn_value(best);
}

template<bool NATIVE>
kernel void kernel_dsv41_quantize_impl(
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
            ? (NATIVE ? dsv41_fp4_e4m3_scale(amax) : dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f))
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = NATIVE ? dsv41_fp4_scaled(value, scale) :
            copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

typedef decltype(kernel_dsv41_quantize_impl<false>) dsv41_quantize_kernel;
template [[host_name("kernel_dsv41_quantize")]] kernel dsv41_quantize_kernel kernel_dsv41_quantize_impl<false>;
template [[host_name("kernel_dsv41_native_quantize")]] kernel dsv41_quantize_kernel kernel_dsv41_quantize_impl<true>;

struct ds4_metal_args_dsv41_native_rope_quantize {
    uint width, start, inverse, mode;
    float frequencies[32];
};

// Preserve the native cache rotation's contraction order explicitly. A
// different FMA can cross a BF16 midpoint before FP8/FP4 quantization.
kernel void kernel_dsv41_native_rope_quantize(
        constant ds4_metal_args_dsv41_native_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? x[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float theta = float(args.start) * args.frequencies[(column - rotary) >> 1u];
        const float c = precise::cos(theta);
        const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv41_fp4_e4m3_scale(amax)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = dsv41_fp4_scaled(value, scale);
    }
    if (valid) dst[column] = dsv41_bf16(result);
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
// The released FP8 GEMM resets its dot accumulator every 32 values, then
// adds scaled block results in K order. Power-of-two scales commute with
// these normal F32 products; dense storage contains their exact values.
kernel void kernel_dsv41_native_fp8_mv(
        constant uint4 &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0.f;
    for (uint col = lane; col < a.x; col += 32u) {
        ulong i = ulong(row) * a.x + col;
        float weight = a.z ? as_type<float>(uint(reinterpret_cast<device const ushort *>(w)[i]) << 16u) :
                            reinterpret_cast<device const float *>(w)[i];
        sum += simd_sum(weight * x[col]);
    }
    if (!lane) out[row] = dsv41_bf16(sum);
}

// Four independent tokens reuse each weight load. Each dot still reduces
// 32 products at a time and adds those blocks in the original K order.
kernel void kernel_dsv41_native_fp8_mm(
        constant uint4 &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x, first = group.y * 4u;
    float sums[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint col = lane; col < a.x; col += 32u) {
        const ulong i = ulong(row) * a.x + col;
        const float weight = a.z ? as_type<float>(uint(reinterpret_cast<device const ushort *>(w)[i]) << 16u) :
                                  reinterpret_cast<device const float *>(w)[i];
        for (uint t = 0; t < 4u; t++) {
            const float value = first + t < a.w ? x[ulong(first + t) * a.x + col] : 0.f;
            sums[t] += simd_sum(weight * value);
        }
    }
    if (!lane) for (uint t = 0; t < 4u && first + t < a.w; t++)
        out[ulong(first + t) * a.y + row] = dsv41_bf16(sums[t]);
}

// Native MXFP4 is decoded into F32 registers. In particular, the weights are
// never converted through a half-precision tile or an integer quantizer.
kernel void kernel_dsv41_native_mxfp4_mv(
        constant uint4 &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device const int *selected [[buffer(3)]],
        device float *out [[buffer(4)]], constant uint &separate_inputs [[buffer(5)]],
        constant uint &streamed [[buffer(6)]], constant ulong *addresses [[buffer(7)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint cols = a.x, rows = a.y, slot = group.y, row = group.x;
    if (row >= rows || slot >= a.z) return;
    uint expert = uint(selected[slot]);
    if (expert >= a.w) { if (!lane) out[slot * rows + row] = as_type<float>(0x7fc00000u); return; }
    if (streamed == 2u && addresses[slot] == 0u) return;
    if (streamed == 2u) w = reinterpret_cast<device const uchar *>(addresses[slot]);
    ulong base = (ulong(streamed == 2u ? 0u : streamed == 1u ? slot : expert) * rows + row) * (cols / 32u) * 17u;
    float sum = 0.f;
    constexpr uint magnitude[8] = {0u, 1u, 2u, 3u, 4u, 6u, 8u, 12u};
    for (uint col = lane; col < cols; col += 32u) {
        float weight;
        if (streamed == 3u) {
            const ulong i = (ulong(expert) * rows + row) * cols + col;
            weight = as_type<float>(uint(reinterpret_cast<device const ushort *>(w)[i]) << 16u);
        } else {
            ulong block = base + ulong(col / 32u) * 17u;
            uint code = (w[block + 1u + (col % 16u)] >> ((col % 32u >= 16u) ? 4u : 0u)) & 15u;
            weight = as_type<float>(dsv41_native_bits(magnitude[code & 7u], -1, w[block], (code & 8u) << 28u));
        }
        sum += simd_sum(weight * x[(separate_inputs ? slot * cols : 0u) + col]);
    }
    if (!lane) out[slot * rows + row] = dsv41_bf16(sum);
}

// Tiles list up to four token/slot pairs selecting the same expert. Results
// go back to their original slots, preserving the later expert-ID sum.
kernel void kernel_dsv41_native_mxfp4_mm(
        constant uint4 &a [[buffer(0)]], device const ushort *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device const int *selected [[buffer(3)]],
        device float *out [[buffer(4)]], constant uint &separate_inputs [[buffer(5)]],
        device const int4 *tiles [[buffer(7)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const int4 slots = tiles[group.y];
    const uint expert = uint(selected[slots[0]]), row = group.x;
    float sums[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint col = lane; col < a.x; col += 32u) {
        const ulong i = (ulong(expert) * a.y + row) * a.x + col;
        const float weight = as_type<float>(uint(w[i]) << 16u);
        for (uint t = 0; t < 4u; t++) {
            const uint input = separate_inputs ? uint(max(slots[t], 0)) : uint(max(slots[t], 0)) / a.z;
            const float value = slots[t] >= 0 ? x[ulong(input) * a.x + col] : 0.f;
            sums[t] += simd_sum(weight * value);
        }
    }
    if (!lane) for (uint t = 0; t < 4u; t++)
        if (slots[t] >= 0) out[ulong(slots[t]) * a.y + row] = dsv41_bf16(sums[t]);
}

struct ds4_native_mid_args { uint width, count; float clamp; };
kernel void kernel_dsv41_native_mid(constant ds4_native_mid_args &a [[buffer(0)]],
        device const float *gate [[buffer(1)]], device const float *up [[buffer(2)]],
        device const float *weights [[buffer(3)]], device float *mid [[buffer(4)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= a.width * a.count) return;
    float g = min(gate[i], a.clamp), u = clamp(up[i], -a.clamp, a.clamp);
    mid[i] = dsv41_bf16(precise::divide(g, 1.f + precise::exp(-g)) * u * weights[i / a.width]);
}

kernel void kernel_dsv41_native_sum(constant uint4 &a [[buffer(0)]],
        device const float *experts [[buffer(1)]], device const int *selected [[buffer(2)]],
        device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.x * a.z) return;
    const uint token = i / a.x, column = i % a.x;
    selected += token * a.y;
    experts += ulong(token) * a.y * a.x;
    float sum = 0.f;
    uint used = 0u;
    // The upstream MoE visits experts by ID, rather than router score order.
    for (uint k = 0; k < a.y; k++) {
        int best = 0x7fffffff; uint slot = 0u;
        for (uint j = 0; j < a.y; j++)
            if (!(used & (1u << j)) && selected[j] < best) { best = selected[j]; slot = j; }
        sum += experts[slot * a.x + column]; used |= 1u << slot;
    }
    out[i] = sum;
}

// Match the released sparse_attn kernel's 64-slot online softmax, including
// invalid ring slots and the BF16 probability cast before the value product.
kernel void kernel_dsv41_native_attention(constant uint4 &a [[buffer(0)]],
        device const float *q [[buffer(1)]], device const float *raw [[buffer(2)]],
        device const float *compressed [[buffer(3)]], device const int *selected [[buffer(4)]],
        device const float *sink [[buffer(5)]], device float *out [[buffer(6)]],
        uint head [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_position_in_threadgroup]]) {
    const uint pos=a.x, count=a.y, top=a.z, dim=a.w;
    const uint nraw = pos == 0u ? 1u : 128u;
    // Eight SIMD groups keep the original 32-lane dot reduction, but compute
    // different keys concurrently. Value dimensions are independent, so each
    // thread owns two of the 512 dimensions and still visits keys in order.
    threadgroup float scores[64], probabilities[64], state[3];
    threadgroup uint indices[64], valid[64];
    float acc0=0.f, acc1=0.f;
    float maximum=-1e30f, total=0.f;
    for (uint base=0;base<nraw+top;base+=64u) {
        for (uint k=sg;k<64;k+=8u) {
            uint slot=base+k, index=0u; bool good=slot<nraw+top;
            if (slot<nraw) {
                index=pos==0u ? 0u : (pos+1u+slot)%128u;
                good=good && (pos>=127u || index<=pos);
            } else if (good) {
                uint ci=count<=top ? slot-nraw : uint(selected[slot-nraw]);
                good=ci<count; index=128u+ci;
            }
            float dot=0.f;
            if (good) for (uint d=lane;d<dim;d+=32u)
                dot += q[head*dim+d] * (index<128u ? raw[index*dim+d] : compressed[(index-128u)*dim+d]);
            dot=simd_sum(dot)*0.04419417382415922f;
            if (!lane) {
                scores[k]=good ? dot : -INFINITY; indices[k]=index; valid[k]=good;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            float next=maximum;
            for (uint k=0;k<64;k++) next=max(next,scores[k]);
            state[0]=precise::exp(maximum-next); state[1]=next;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<64u) {
            float p=precise::exp(scores[tid]-state[1]);
            scores[tid]=p; probabilities[tid]=dsv41_bf16(p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc0*=state[0]; acc1*=state[0];
        for (uint k=0;k<64;k++) {
            if (valid[k]) {
                uint index=indices[k]; float p=probabilities[k];
                device const float *value=index<128u ? raw+index*dim : compressed+(index-128u)*dim;
                acc0 += p*value[tid]; acc1 += p*value[tid+256u];
            }
        }
        // The release reduces each 64-slot block before its online update.
        // A sequential denominator sum can cross a BF16 rounding midpoint.
        for (uint stride=1u;stride<64u;stride*=2u) {
            const uint k=tid*2u*stride;
            if (k<64u) scores[k]+=scores[k+stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (!tid) { total=total*state[0]+scores[0]; maximum=state[1]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) state[2]=total+precise::exp(sink[head]-maximum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[head*dim+tid]=dsv41_bf16(precise::divide(acc0,state[2]));
    out[head*dim+tid+256u]=dsv41_bf16(precise::divide(acc1,state[2]));
}

// Correctness path: keep the release's softplus and explicit normalization
// boundaries. The expert calculation is sensitive to a last-bit weight change.
kernel void kernel_dsv41_native_router(constant uint2 &a [[buffer(0)]],
        device const float *logits [[buffer(1)]], device const float *bias [[buffer(2)]],
        device float *probs [[buffer(3)]], device int *selected [[buffer(4)]],
        device float *weights [[buffer(5)]], constant float &scale [[buffer(6)]],
        uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[1024], best_scores[8];
    threadgroup int best_ids[8];
    for (uint i=tid;i<a.x;i+=ntg) {
        float x=logits[i];
        float ex=precise::exp(x);
        float u=1.f+ex;
        float sp=x>20.f ? x : precise::log(u)-precise::divide((u-1.f)-ex,u);
        probs[i]=precise::sqrt(sp);
        scores[i]=probs[i]+bias[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint j=0;j<a.y;j++) {
        float best=-INFINITY; int bi=-1;
        for (uint i=tid;i<a.x;i+=ntg) {
            float score=scores[i];
            if (score>best) {best=score;bi=int(i);}
        }
        for (uint offset=16u;offset;offset>>=1u) {
            const float other=simd_shuffle_down(best,offset);
            const int oi=simd_shuffle_down(bi,offset);
            if (oi>=0 && (other>best || (other==best && (bi<0 || oi<bi)))) {
                best=other;bi=oi;
            }
        }
        if (!lane) {best_scores[sg]=best;best_ids[sg]=bi;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            for (uint k=1;k<8u;k++) {
                const float other=best_scores[k];const int oi=best_ids[k];
                if (oi>=0 && (other>best || (other==best && (bi<0 || oi<bi)))) {
                    best=other;bi=oi;
                }
            }
            selected[j]=bi;
            if (bi>=0) scores[bi]=-INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid) return;
    float total=0.f;
    for (uint j=0;j<a.y;j++) total+=probs[selected[j]];
    for (uint j=0;j<a.y;j++) {
        float quotient=precise::divide(probs[selected[j]],total+1e-20f);
        weights[j]=quotient*scale;
    }
}

kernel void kernel_dsv41_native_hc(constant uint2 &a [[buffer(0)]],
        device const float *fn [[buffer(1)]], device const float *x [[buffer(2)]],
        device float *out [[buffer(3)]], constant float &eps [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row=group.x;
    x += ulong(group.y)*a.x; out += ulong(group.y)*a.y;
    float dot=0.f, squares=0.f;
    for(uint i=lane;i<a.x;i+=32u) { dot+=fn[ulong(row)*a.x+i]*x[i];squares+=x[i]*x[i]; }
    dot=simd_sum(dot);squares=simd_sum(squares);
    if(!lane) out[row]=dot*precise::rsqrt(squares/float(a.x)+eps);
}

// Keep softmax normalization before the two weighted products, as in the
// source Compressor. Dividing their combined numerator changes BF16 ties.
kernel void kernel_dsv41_native_pool2(constant ds4_metal_args_dsv41_pool &args,
        device float *out, device const float *kv, device const float *scores,
        device const float *previous_kv, device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if(index.x>=args.width || index.y>=args.pairs)return;
    const long a=long(index.y)*2-args.tail;
    const ulong b=ulong(a+1)*args.width+index.x;
    const float ka=a<0 ? previous_kv[index.x] : kv[ulong(a)*args.width+index.x];
    const float sa=a<0 ? previous_scores[index.x] : scores[ulong(a)*args.width+index.x];
    const float sb=scores[b], peak=max(sa,sb);
    const float ea=precise::exp(sa-peak),eb=precise::exp(sb-peak),den=ea+eb;
    const float wa=precise::divide(ea,den),wb=precise::divide(eb,den);
    const float first=ka*wa,second=kv[b]*wb;
    out[ulong(index.y)*args.width+index.x]=dsv41_bf16(first+second);
}

// Native HC keeps the release's individual division boundaries. Reciprocal
// multiplication in the optimized HC kernel changes the cached BF16 input.
kernel void kernel_dsv41_native_hc_input(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm &a [[buffer(0)]],
        device const float *mix [[buffer(1)]], device const float *scale [[buffer(2)]],
        device const float *base [[buffer(3)]], device const float *x [[buffer(4)]],
        device float *split [[buffer(5)]], device float *collapsed [[buffer(6)]],
        device const float *norm_weight [[buffer(7)]], device float *norm [[buffer(8)]],
        device const float *pre [[buffer(9)]], threadgroup float *shared [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float *row=shared+4,*sums=row+a.n_embd;
    if(tid<32u)sums[tid]=0.f;
    if(!tid) {
        for(uint i=0;i<4;i++) {
            float p=mix[i]*scale[0]+base[i];
            float q=mix[i+4]*scale[1]+base[i+4];
            split[i]=precise::divide(1.f,1.f+precise::exp(-p))+a.eps;
            split[i+4]=2.f*precise::divide(1.f,1.f+precise::exp(-q));
        }
        float comb[16];
        for(uint i=0;i<16;i++)comb[i]=mix[i+8]*scale[2]+base[i+8];
        for(uint r=0;r<4;r++) {
            float peak=max(max(comb[r*4],comb[r*4+1]),max(comb[r*4+2],comb[r*4+3]));
            for(uint j=0;j<4;j++)comb[r*4+j]=precise::exp(comb[r*4+j]-peak);
            float sum=(comb[r*4]+comb[r*4+1])+(comb[r*4+2]+comb[r*4+3]);
            for(uint j=0;j<4;j++)comb[r*4+j]=precise::divide(comb[r*4+j],sum)+a.eps;
        }
        for(uint j=0;j<4;j++) {
            float sum=((comb[j]+comb[4+j])+comb[8+j])+comb[12+j]+a.eps;
            for(uint r=0;r<4;r++)comb[r*4+j]=precise::divide(comb[r*4+j],sum);
        }
        for(uint iter=1;iter<uint(a.sinkhorn_iters);iter++) {
            for(uint r=0;r<4;r++) {
                float sum=((comb[r*4]+comb[r*4+1])+comb[r*4+2])+comb[r*4+3]+a.eps;
                for(uint j=0;j<4;j++)comb[r*4+j]=precise::divide(comb[r*4+j],sum);
            }
            for(uint j=0;j<4;j++) {
                float sum=((comb[j]+comb[4+j])+comb[8+j])+comb[12+j]+a.eps;
                for(uint r=0;r<4;r++)comb[r*4+j]=precise::divide(comb[r*4+j],sum);
            }
        }
        for(uint i=0;i<16;i++)split[i+8]=comb[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float square=0.f;
    for(uint i=tid;i<uint(a.n_embd);i+=ntg) {
        #pragma METAL fp contract(off)
        float v=0.f;
        for(uint h=0;h<4;h++)v+=x[h*a.n_embd+i]*pre[h];
        v=dsv41_bf16(v);row[i]=v;collapsed[i]=v;square+=v*v;
    }
    square=simd_sum(square);
    if(!lane)sums[sg]=square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    square=simd_sum(sums[lane]);
    const float rstd=precise::rsqrt(precise::divide(square,float(a.n_embd))+a.norm_eps);
    for(uint i=tid;i<uint(a.n_embd);i+=ntg)norm[i]=dsv41_bf16((row[i]*rstd)*norm_weight[i]);
}

// hc_pre materializes the four F32 products before reducing them.
kernel void kernel_dsv41_native_hc_weighted_sum(
        constant ds4_metal_args_dsv4_hc_weighted_sum &a,
        device const char *x, device const char *weights, device char *out,
        uint gid [[thread_position_in_grid]]) {
    #pragma METAL fp contract(off)
    if (long(gid) >= a.n_embd * a.n_tokens) return;
    const long d = long(gid) % a.n_embd, t = long(gid) / a.n_embd;
    float sum = 0.f;
    for (long h = 0; h < a.n_hc; h++) {
        float v = *reinterpret_cast<device const float *>(x + d*a.nb_x0 + h*a.nb_x1 + t*a.nb_x2);
        float w = *reinterpret_cast<device const float *>(weights + h*a.nb_w0 + t*a.nb_w1);
        sum += v*w;
    }
    *reinterpret_cast<device float *>(out + d*a.nb0 + t*a.nb1) = sum;
}

// The release sums the residual products before adding the post-scaled block.
// Including the block in that reduction changes BF16 midpoint decisions.
kernel void kernel_dsv41_native_hc_expand4(
        constant ds4_metal_args_dsv4_hc_expand &a,
        device const char *block, device const char *residual,
        device const char *post, device const char *comb,
        device const char *add, device char *out,
        uint gid [[thread_position_in_grid]]) {
    // hc_post materializes each F32 product before its sum. Safe math still
    // permits statement-level FMA, which crosses those source boundaries.
    #pragma METAL fp contract(off)
    if (long(gid) >= a.n_embd * a.n_tokens) return;
    const long d = long(gid) % a.n_embd, t = long(gid) / a.n_embd;
    float v = *reinterpret_cast<device const float *>(block + d*a.nb_block0 + t*a.nb_block1);
    if (a.has_add) v += *reinterpret_cast<device const float *>(add + d*a.nb_add0 + t*a.nb_add1);
    for (long dst = 0; dst < 4; dst++) {
        float sum = 0.f;
        for (long src = 0; src < 4; src++) {
            float c = *reinterpret_cast<device const float *>(comb + dst*a.nb_comb0 + src*a.nb_comb1 + t*a.nb_comb2);
            float r = *reinterpret_cast<device const float *>(residual + d*a.nb_res0 + src*a.nb_res1 + t*a.nb_res2);
            sum += c*r;
        }
        float p = *reinterpret_cast<device const float *>(post + dst*a.nb_post0 + t*a.nb_post1);
        float result = v*p + sum;
        *reinterpret_cast<device float *>(out + d*a.nb0 + dst*a.nb1 + t*a.nb2) = result;
    }
}
