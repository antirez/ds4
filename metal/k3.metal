// Checkpoint-native kernels for Kimi K3.

static inline float k3_bf16_to_f32(ushort v) {
    return as_type<float>((uint)v << 16);
}

static inline ushort k3_f32_to_bf16_rne(float value) {
    uint bits = as_type<uint>(value);
    const uint rounding = 0x7fffu + ((bits >> 16) & 1u);
    return (ushort)((bits + rounding) >> 16);
}

static inline float k3_round_bf16(float value) {
    return k3_bf16_to_f32(k3_f32_to_bf16_rne(value));
}

static inline float k3_e8m0_to_f32(uchar value) {
    if (value == 255u) return NAN;
    if (value == 0u) return as_type<float>(0x00400000u);
    return as_type<float>((uint)value << 23);
}

static inline float k3_e4m3fn_to_f32(uchar value) {
    const uint sign = value >> 7;
    const uint exponent = (value >> 3) & 15u;
    const uint mantissa = value & 7u;
    if (exponent == 15u && mantissa == 7u) return NAN;
    if (exponent == 0u) {
        const float result = (float)mantissa * 0x1.0p-9f;
        return sign ? -result : result;
    }
    const uint bits =
        (sign << 31) | ((exponent + 120u) << 23) | (mantissa << 20);
    return as_type<float>(bits);
}

constant float k3_e2m1_table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
};

static inline uchar k3_f32_to_e4m3fn(float value) {
    const uint raw = as_type<uint>(value);
    const uint sign = raw >> 31;
    uint absolute = raw & 0x7fffffffu;
    absolute = min(absolute, 0x43e00000u); // Saturate at 448.
    uint code = 0u;

    if (absolute < 0x3c800000u) {
        code = (uint)clamp(
            rint(as_type<float>(absolute) * 512.0f), 0.0f, 8.0f);
    } else {
        uint exponent = (absolute >> 23) - 120u;
        uint mantissa = (absolute >> 20) & 7u;
        const uint remainder = absolute & 0xfffffu;
        if (remainder > 0x80000u ||
            (remainder == 0x80000u && (mantissa & 1u))) {
            mantissa++;
            if (mantissa == 8u) {
                exponent++;
                mantissa = 0u;
            }
        }
        code = min((exponent << 3) | mantissa, 126u);
    }
    return (uchar)(code | (sign && code != 0u ? 0x80u : 0u));
}

static inline float k3_e2m1_to_f32(uchar nibble) {
    return k3_e2m1_table[nibble & 15u];
}

struct k3_matmul_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

kernel void kernel_k3_embedding_bf16(
        constant k3_matmul_args &args,
        device const ushort     *weights,
        device const int        *tokens,
        device float            *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= args.in_dim || row >= args.n_rows) return;
    const int token = tokens[row];
    out[(ulong)row * args.in_dim + d] =
        token >= 0 && (uint)token < args.out_dim
            ? k3_bf16_to_f32(weights[(ulong)(uint)token * args.in_dim + d])
            : 0.0f;
}

kernel void kernel_k3_expand_bf16(
        device const ushort *in,
        device float        *out,
        uint gid [[thread_position_in_grid]]) {
    out[gid] = k3_bf16_to_f32(in[gid]);
}

/*
 * Each simdgroup owns one output row and keeps the reduction entirely in
 * registers. The host tunes the number of rows in a threadgroup.
 */
kernel void kernel_k3_mul_mv_bf16_f32(
        constant k3_matmul_args &args,
        device const ushort     *weights,
        device const float      *x,
        device float            *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    for (uint k = lane; k < args.in_dim; k += 32u) {
        sum = fma(k3_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

/*
 * Preserve each lane's original reduction order while issuing four strided
 * weight/input loads before the dependent FMA chain. This exposes memory-level
 * parallelism without changing the final simdgroup reduction topology.
 */
kernel void kernel_k3_mul_mv_bf16_f32_unroll4(
        constant k3_matmul_args &args,
        device const ushort     *weights,
        device const float      *x,
        device float            *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 96u < args.in_dim; k += 128u) {
        const ushort w0 = w[k];
        const ushort w1 = w[k + 32u];
        const ushort w2 = w[k + 64u];
        const ushort w3 = w[k + 96u];
        const float x0 = xr[k];
        const float x1 = xr[k + 32u];
        const float x2 = xr[k + 64u];
        const float x3 = xr[k + 96u];
        sum = fma(k3_bf16_to_f32(w0), x0, sum);
        sum = fma(k3_bf16_to_f32(w1), x1, sum);
        sum = fma(k3_bf16_to_f32(w2), x2, sum);
        sum = fma(k3_bf16_to_f32(w3), x3, sum);
    }
    for (; k < args.in_dim; k += 32u) {
        sum = fma(k3_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

kernel void kernel_k3_mul_mv_bf16_f32_unroll8(
        constant k3_matmul_args &args,
        device const ushort     *weights,
        device const float      *x,
        device float            *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 224u < args.in_dim; k += 256u) {
        const ushort w0 = w[k];
        const ushort w1 = w[k + 32u];
        const ushort w2 = w[k + 64u];
        const ushort w3 = w[k + 96u];
        const ushort w4 = w[k + 128u];
        const ushort w5 = w[k + 160u];
        const ushort w6 = w[k + 192u];
        const ushort w7 = w[k + 224u];
        const float x0 = xr[k];
        const float x1 = xr[k + 32u];
        const float x2 = xr[k + 64u];
        const float x3 = xr[k + 96u];
        const float x4 = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        sum = fma(k3_bf16_to_f32(w0), x0, sum);
        sum = fma(k3_bf16_to_f32(w1), x1, sum);
        sum = fma(k3_bf16_to_f32(w2), x2, sum);
        sum = fma(k3_bf16_to_f32(w3), x3, sum);
        sum = fma(k3_bf16_to_f32(w4), x4, sum);
        sum = fma(k3_bf16_to_f32(w5), x5, sum);
        sum = fma(k3_bf16_to_f32(w6), x6, sum);
        sum = fma(k3_bf16_to_f32(w7), x7, sum);
    }
    for (; k < args.in_dim; k += 32u) {
        sum = fma(k3_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

kernel void kernel_k3_mul_mv_bf16_f32_unroll16(
        constant k3_matmul_args &args,
        device const ushort     *weights,
        device const float      *x,
        device float            *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 480u < args.in_dim; k += 512u) {
        const ushort w0 = w[k];
        const ushort w1 = w[k + 32u];
        const ushort w2 = w[k + 64u];
        const ushort w3 = w[k + 96u];
        const ushort w4 = w[k + 128u];
        const ushort w5 = w[k + 160u];
        const ushort w6 = w[k + 192u];
        const ushort w7 = w[k + 224u];
        const ushort w8 = w[k + 256u];
        const ushort w9 = w[k + 288u];
        const ushort w10 = w[k + 320u];
        const ushort w11 = w[k + 352u];
        const ushort w12 = w[k + 384u];
        const ushort w13 = w[k + 416u];
        const ushort w14 = w[k + 448u];
        const ushort w15 = w[k + 480u];
        const float x0 = xr[k];
        const float x1 = xr[k + 32u];
        const float x2 = xr[k + 64u];
        const float x3 = xr[k + 96u];
        const float x4 = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        const float x8 = xr[k + 256u];
        const float x9 = xr[k + 288u];
        const float x10 = xr[k + 320u];
        const float x11 = xr[k + 352u];
        const float x12 = xr[k + 384u];
        const float x13 = xr[k + 416u];
        const float x14 = xr[k + 448u];
        const float x15 = xr[k + 480u];
        sum = fma(k3_bf16_to_f32(w0), x0, sum);
        sum = fma(k3_bf16_to_f32(w1), x1, sum);
        sum = fma(k3_bf16_to_f32(w2), x2, sum);
        sum = fma(k3_bf16_to_f32(w3), x3, sum);
        sum = fma(k3_bf16_to_f32(w4), x4, sum);
        sum = fma(k3_bf16_to_f32(w5), x5, sum);
        sum = fma(k3_bf16_to_f32(w6), x6, sum);
        sum = fma(k3_bf16_to_f32(w7), x7, sum);
        sum = fma(k3_bf16_to_f32(w8), x8, sum);
        sum = fma(k3_bf16_to_f32(w9), x9, sum);
        sum = fma(k3_bf16_to_f32(w10), x10, sum);
        sum = fma(k3_bf16_to_f32(w11), x11, sum);
        sum = fma(k3_bf16_to_f32(w12), x12, sum);
        sum = fma(k3_bf16_to_f32(w13), x13, sum);
        sum = fma(k3_bf16_to_f32(w14), x14, sum);
        sum = fma(k3_bf16_to_f32(w15), x15, sum);
    }
    for (; k < args.in_dim; k += 32u) {
        sum = fma(k3_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

/*
 * KDA's Q/K/V/output-gate projections share input and shape. A third grid
 * dimension selects the matrix, reducing four independent dispatches to one
 * without changing any row's reduction order.
 */
kernel void kernel_k3_mul_mv4_bf16_f32_unroll8(
        constant k3_matmul_args &args,
        device const ushort     *weights0,
        device const ushort     *weights1,
        device const ushort     *weights2,
        device const ushort     *weights3,
        device const float      *x,
        device float            *out0,
        device float            *out1,
        device float            *out2,
        device float            *out3,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint matrix = tgpig.z;
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (matrix >= 4u || out_row >= args.out_dim ||
        token >= args.n_rows) {
        return;
    }

    device const ushort *weights = weights0;
    device float *out = out0;
    if (matrix == 1u) {
        weights = weights1;
        out = out1;
    } else if (matrix == 2u) {
        weights = weights2;
        out = out2;
    } else if (matrix == 3u) {
        weights = weights3;
        out = out3;
    }

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 224u < args.in_dim; k += 256u) {
        const ushort w0 = w[k];
        const ushort w1 = w[k + 32u];
        const ushort w2 = w[k + 64u];
        const ushort w3 = w[k + 96u];
        const ushort w4 = w[k + 128u];
        const ushort w5 = w[k + 160u];
        const ushort w6 = w[k + 192u];
        const ushort w7 = w[k + 224u];
        const float x0 = xr[k];
        const float x1 = xr[k + 32u];
        const float x2 = xr[k + 64u];
        const float x3 = xr[k + 96u];
        const float x4 = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        sum = fma(k3_bf16_to_f32(w0), x0, sum);
        sum = fma(k3_bf16_to_f32(w1), x1, sum);
        sum = fma(k3_bf16_to_f32(w2), x2, sum);
        sum = fma(k3_bf16_to_f32(w3), x3, sum);
        sum = fma(k3_bf16_to_f32(w4), x4, sum);
        sum = fma(k3_bf16_to_f32(w5), x5, sum);
        sum = fma(k3_bf16_to_f32(w6), x6, sum);
        sum = fma(k3_bf16_to_f32(w7), x7, sum);
    }
    for (; k < args.in_dim; k += 32u) {
        sum = fma(k3_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        out[(ulong)token * args.out_dim + out_row] = sum;
    }
}

kernel void kernel_k3_mul_mv4_bf16_f32_unroll16(
        constant k3_matmul_args &args,
        device const ushort     *weights0,
        device const ushort     *weights1,
        device const ushort     *weights2,
        device const ushort     *weights3,
        device const float      *x,
        device float            *out0,
        device float            *out1,
        device float            *out2,
        device float            *out3,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint matrix = tgpig.z;
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (matrix >= 4u || out_row >= args.out_dim ||
        token >= args.n_rows) {
        return;
    }

    device const ushort *weights = weights0;
    device float *out = out0;
    if (matrix == 1u) {
        weights = weights1;
        out = out1;
    } else if (matrix == 2u) {
        weights = weights2;
        out = out2;
    } else if (matrix == 3u) {
        weights = weights3;
        out = out3;
    }

    device const ushort *w = weights + (ulong)out_row * args.in_dim;
    device const float *xr = x + (ulong)token * args.in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 480u < args.in_dim; k += 512u) {
        const ushort w0 = w[k];
        const ushort w1 = w[k + 32u];
        const ushort w2 = w[k + 64u];
        const ushort w3 = w[k + 96u];
        const ushort w4 = w[k + 128u];
        const ushort w5 = w[k + 160u];
        const ushort w6 = w[k + 192u];
        const ushort w7 = w[k + 224u];
        const ushort w8 = w[k + 256u];
        const ushort w9 = w[k + 288u];
        const ushort w10 = w[k + 320u];
        const ushort w11 = w[k + 352u];
        const ushort w12 = w[k + 384u];
        const ushort w13 = w[k + 416u];
        const ushort w14 = w[k + 448u];
        const ushort w15 = w[k + 480u];
        const float x0 = xr[k];
        const float x1 = xr[k + 32u];
        const float x2 = xr[k + 64u];
        const float x3 = xr[k + 96u];
        const float x4 = xr[k + 128u];
        const float x5 = xr[k + 160u];
        const float x6 = xr[k + 192u];
        const float x7 = xr[k + 224u];
        const float x8 = xr[k + 256u];
        const float x9 = xr[k + 288u];
        const float x10 = xr[k + 320u];
        const float x11 = xr[k + 352u];
        const float x12 = xr[k + 384u];
        const float x13 = xr[k + 416u];
        const float x14 = xr[k + 448u];
        const float x15 = xr[k + 480u];
        sum = fma(k3_bf16_to_f32(w0), x0, sum);
        sum = fma(k3_bf16_to_f32(w1), x1, sum);
        sum = fma(k3_bf16_to_f32(w2), x2, sum);
        sum = fma(k3_bf16_to_f32(w3), x3, sum);
        sum = fma(k3_bf16_to_f32(w4), x4, sum);
        sum = fma(k3_bf16_to_f32(w5), x5, sum);
        sum = fma(k3_bf16_to_f32(w6), x6, sum);
        sum = fma(k3_bf16_to_f32(w7), x7, sum);
        sum = fma(k3_bf16_to_f32(w8), x8, sum);
        sum = fma(k3_bf16_to_f32(w9), x9, sum);
        sum = fma(k3_bf16_to_f32(w10), x10, sum);
        sum = fma(k3_bf16_to_f32(w11), x11, sum);
        sum = fma(k3_bf16_to_f32(w12), x12, sum);
        sum = fma(k3_bf16_to_f32(w13), x13, sum);
        sum = fma(k3_bf16_to_f32(w14), x14, sum);
        sum = fma(k3_bf16_to_f32(w15), x15, sum);
    }
    for (; k < args.in_dim; k += 32u) {
        sum = fma(k3_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        out[(ulong)token * args.out_dim + out_row] = sum;
    }
}

struct k3_bf16_block16 {
    ushort v[16];
};

template <typename type4x4>
void k3_dequantize_bf16(
        device const k3_bf16_block16 *src,
        short il,
        thread type4x4 &reg) {
    (void)il;
    float4x4 tmp;
    for (short i = 0; i < 16; i++) {
        tmp[i / 4][i % 4] = k3_bf16_to_f32(src->v[i]);
    }
    reg = (type4x4)tmp;
}

typedef decltype(kernel_mul_mm<
        half, half4x4, simdgroup_half8x8,
        half, half2x4, simdgroup_half8x8,
        k3_bf16_block16, 1, k3_dequantize_bf16,
        float, float4x4, float, float2x4>) k3_mul_mm_bf16_t;

template [[host_name("kernel_k3_mul_mm_bf16_f32")]]
kernel k3_mul_mm_bf16_t kernel_mul_mm<
        half, half4x4, simdgroup_half8x8,
        half, half2x4, simdgroup_half8x8,
        k3_bf16_block16, 1, k3_dequantize_bf16,
        half, half4x4, float, float2x4>;

struct k3_norm_args {
    uint width;
    uint n_rows;
    float eps;
};

struct k3_strided_norm_args {
    uint width;
    uint n_rows;
    uint row_stride;
    float eps;
};

kernel void kernel_k3_rms_norm_bf16(
        constant k3_norm_args &args,
        device const float    *x,
        device const ushort   *weight,
        device float          *out,
        threadgroup float     *partials [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.n_rows) return;
    device const float *xr = x + (ulong)row * args.width;
    device float *yr = out + (ulong)row * args.width;
    float sum = 0.0f;
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        sum = fma(xr[d], xr[d], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partials[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < nsg ? partials[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)args.width + args.eps);
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        yr[d] = xr[d] * scale * k3_bf16_to_f32(weight[d]);
    }
}

kernel void kernel_k3_rms_norm_bf16_strided(
        constant k3_strided_norm_args &args,
        device const float            *x,
        device const ushort           *weight,
        device float                  *out,
        threadgroup float             *partials [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.n_rows) return;
    device const float *xr = x + (ulong)row * args.row_stride;
    device float *yr = out + (ulong)row * args.row_stride;
    float sum = 0.0f;
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        sum = fma(xr[d], xr[d], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partials[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < nsg ? partials[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)args.width + args.eps);
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        yr[d] = xr[d] * scale * k3_bf16_to_f32(weight[d]);
    }
}

struct k3_situ_args {
    uint width;
    uint n_rows;
    float beta;
    float linear_beta;
};

kernel void kernel_k3_situ(
        constant k3_situ_args &args,
        device const float    *gate,
        device const float    *up,
        device float          *out,
        uint gid [[thread_position_in_grid]]) {
    const ulong count = (ulong)args.width * args.n_rows;
    if ((ulong)gid >= count) return;
    const float g = gate[gid];
    const float u = up[gid];
    const float bounded_gate =
        args.beta * tanh(g / args.beta) / (1.0f + exp(-g));
    const float bounded_up = args.linear_beta > 0.0f
        ? args.linear_beta * tanh(u / args.linear_beta)
        : u;
    out[gid] = bounded_gate * bounded_up;
}

struct k3_mxfp_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
    float beta;
    float linear_beta;
    float multiplier;
    uint accumulate;
};

kernel void kernel_k3_mxfp8_quantize(
        constant k3_mxfp_args &args,
        device const float    *x,
        device uchar          *values,
        device uchar          *scales,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint group = tgpig.x;
    const uint row = tgpig.y;
    if (row >= args.n_rows || group >= args.in_dim / 32u) return;
    const ulong index = (ulong)row * args.in_dim + group * 32u + lane;
    const float v = x[index];
    float amax = simd_max(fabs(v));
    amax = max(amax, 0x1.0p-126f);
    uint scale_value = 0u;
    if (lane == 0u) {
        const uint amax_bits = as_type<uint>(amax);
        int biased;
        if ((amax_bits & 0x7f800000u) == 0x7f800000u) {
            biased = 254;
        } else {
            biased = (int)(amax_bits >> 23) - 8;
            biased += (amax_bits & 0x7fffffu) > 0x600000u;
        }
        scale_value = (uint)clamp(biased, 0, 254);
        scales[(ulong)row * (args.in_dim / 32u) + group] =
            (uchar)scale_value;
    }
    const uchar scale = (uchar)simd_broadcast_first(scale_value);
    const uchar inverse_scale = (uchar)(254u - scale);
    values[index] =
        k3_f32_to_e4m3fn(v * k3_e8m0_to_f32(inverse_scale));
}

static inline float2 k3_mxfp4_pair_dot(
        device const uchar *packed_a,
        device const uchar *scale_a,
        device const uchar *packed_b,
        device const uchar *scale_b,
        device const uchar *x_values,
        device const uchar *x_scales,
        uint in_dim,
        ushort lane) {
    const uint groups = in_dim / 32u;
    float2 sum = 0.0f;
    for (uint group = 0; group < groups; group++) {
        const uint k = group * 32u + lane;
        const uchar pa = packed_a[k >> 1];
        const uchar pb = packed_b[k >> 1];
        const uchar nibble_a = (k & 1u) ? pa >> 4 : pa & 15u;
        const uchar nibble_b = (k & 1u) ? pb >> 4 : pb & 15u;
        const float xv = k3_e4m3fn_to_f32(x_values[k]);
        const float xs = k3_e8m0_to_f32(x_scales[group]);
        sum.x = fma(k3_e2m1_to_f32(nibble_a) *
                    k3_e8m0_to_f32(scale_a[group]) * xs, xv, sum.x);
        sum.y = fma(k3_e2m1_to_f32(nibble_b) *
                    k3_e8m0_to_f32(scale_b[group]) * xs, xv, sum.y);
    }
    return simd_sum(sum);
}

kernel void kernel_k3_mxfp4_pair_situ(
        constant k3_mxfp_args &args,
        device const uchar    *gate_packed,
        device const uchar    *gate_scales,
        device const uchar    *up_packed,
        device const uchar    *up_scales,
        device const uchar    *x_values,
        device const uchar    *x_scales,
        device float          *gate,
        device float          *up,
        device float          *mid,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output = tgpig.x * 4u + sg;
    const uint row = tgpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    const uint packed_stride = args.in_dim / 2u;
    const uint scale_stride = args.in_dim / 32u;
    const ulong wo = (ulong)output * packed_stride;
    const ulong so = (ulong)output * scale_stride;
    const ulong xv = (ulong)row * args.in_dim;
    const ulong xs = (ulong)row * scale_stride;
    const float2 result = k3_mxfp4_pair_dot(
        gate_packed + wo, gate_scales + so,
        up_packed + wo, up_scales + so,
        x_values + xv, x_scales + xs, args.in_dim, lane);
    if (lane == 0u) {
        const ulong dst = (ulong)row * args.out_dim + output;
        const float g = result.x;
        const float u = result.y;
        gate[dst] = g;
        up[dst] = u;
        mid[dst] =
            args.beta * tanh(g / args.beta) / (1.0f + exp(-g)) *
            (args.linear_beta > 0.0f
                 ? args.linear_beta * tanh(u / args.linear_beta)
                 : u);
    }
}

kernel void kernel_k3_mxfp4_down(
        constant k3_mxfp_args &args,
        device const uchar    *packed,
        device const uchar    *weight_scales,
        device const uchar    *x_values,
        device const uchar    *x_scales,
        device float          *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output = tgpig.x * 4u + sg;
    const uint row = tgpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    const uint groups = args.in_dim / 32u;
    const uint packed_stride = args.in_dim / 2u;
    device const uchar *wr = packed + (ulong)output * packed_stride;
    device const uchar *ws = weight_scales + (ulong)output * groups;
    device const uchar *xv = x_values + (ulong)row * args.in_dim;
    device const uchar *xs = x_scales + (ulong)row * groups;

    float sum = 0.0f;
    for (uint group = 0; group < groups; group++) {
        const uint k = group * 32u + lane;
        const uchar p = wr[k >> 1];
        const uchar nibble = (k & 1u) ? p >> 4 : p & 15u;
        sum = fma(k3_e2m1_to_f32(nibble) *
                  k3_e8m0_to_f32(ws[group]) *
                  k3_e8m0_to_f32(xs[group]),
                  k3_e4m3fn_to_f32(xv[k]), sum);
    }
    sum = simd_sum(sum) * args.multiplier;
    if (lane == 0u) {
        const ulong dst = (ulong)row * args.out_dim + output;
        if (args.accumulate) out[dst] += sum;
        else out[dst] = sum;
    }
}

static inline float2 k3_gguf_mxfp4_pair_dot(
        device const uchar *gate,
        device const uchar *up,
        device const uchar *x_values,
        device const uchar *x_scales,
        uint groups,
        ushort lane) {
    float2 sum = 0.0f;
    for (uint group = 0; group < groups; group++) {
        device const uchar *gb = gate + (ulong)group * 17u;
        device const uchar *ub = up + (ulong)group * 17u;
        const uint packed_index = 1u + ((uint)lane & 15u);
        const uint shift = lane < 16u ? 0u : 4u;
        const uchar gq = (gb[packed_index] >> shift) & 15u;
        const uchar uq = (ub[packed_index] >> shift) & 15u;
        const float xv =
            k3_e4m3fn_to_f32(x_values[group * 32u + lane]);
        const float xs = k3_e8m0_to_f32(x_scales[group]);
        sum.x = fma(k3_e2m1_to_f32(gq) *
                    k3_e8m0_to_f32(gb[0]) * xs, xv, sum.x);
        sum.y = fma(k3_e2m1_to_f32(uq) *
                    k3_e8m0_to_f32(ub[0]) * xs, xv, sum.y);
    }
    return simd_sum(sum);
}

kernel void kernel_k3_gguf_mxfp4_pair_situ(
        constant k3_mxfp_args &args,
        device const uchar    *gate,
        device const uchar    *up,
        device const uchar    *x_values,
        device const uchar    *x_scales,
        device float          *mid,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output = tgpig.x * 4u + sg;
    const uint row = tgpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    const uint groups = args.in_dim / 32u;
    const uint row_bytes = groups * 17u;
    const float2 result = k3_gguf_mxfp4_pair_dot(
        gate + (ulong)output * row_bytes,
        up + (ulong)output * row_bytes,
        x_values + (ulong)row * args.in_dim,
        x_scales + (ulong)row * groups,
        groups,
        lane);
    if (lane == 0u) {
        const float g = result.x;
        const float u = result.y;
        mid[(ulong)row * args.out_dim + output] =
            args.beta * tanh(g / args.beta) / (1.0f + exp(-g)) *
            (args.linear_beta > 0.0f
                 ? args.linear_beta * tanh(u / args.linear_beta)
                 : u);
    }
}

kernel void kernel_k3_gguf_mxfp4_down(
        constant k3_mxfp_args &args,
        device const uchar    *weights,
        device const uchar    *x_values,
        device const uchar    *x_scales,
        device float          *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint output = tgpig.x * 4u + sg;
    const uint row = tgpig.y;
    if (output >= args.out_dim || row >= args.n_rows) return;
    const uint groups = args.in_dim / 32u;
    const uint row_bytes = groups * 17u;
    device const uchar *wr = weights + (ulong)output * row_bytes;
    device const uchar *xv = x_values + (ulong)row * args.in_dim;
    device const uchar *xs = x_scales + (ulong)row * groups;
    float sum = 0.0f;
    for (uint group = 0; group < groups; group++) {
        device const uchar *block = wr + (ulong)group * 17u;
        const uint shift = lane < 16u ? 0u : 4u;
        const uchar q =
            (block[1u + ((uint)lane & 15u)] >> shift) & 15u;
        sum = fma(k3_e2m1_to_f32(q) *
                  k3_e8m0_to_f32(block[0]) *
                  k3_e8m0_to_f32(xs[group]),
                  k3_e4m3fn_to_f32(xv[group * 32u + lane]),
                  sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        out[(ulong)row * args.out_dim + output] = sum;
    }
}

struct k3_rows_args {
    uint n_rows;
    uint width;
};

kernel void kernel_k3_gather_rows(
        constant k3_rows_args &args,
        device const float    *x,
        device const int      *row_ids,
        device float          *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= args.width || row >= args.n_rows) return;
    const int source = row_ids[row];
    out[(ulong)row * args.width + d] =
        source >= 0 ? x[(ulong)(uint)source * args.width + d] : 0.0f;
}

kernel void kernel_k3_scatter_add_rows(
        constant k3_rows_args &args,
        device const float    *x,
        device const int      *row_ids,
        device const float    *row_weights,
        device float          *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= args.width || row >= args.n_rows) return;
    const int destination = row_ids[row];
    if (destination >= 0) {
        out[(ulong)(uint)destination * args.width + d] +=
            x[(ulong)row * args.width + d] * row_weights[row];
    }
}

struct k3_router_args {
    uint n_rows;
    uint n_experts;
    uint top_k;
};

kernel void kernel_k3_router_topk(
        constant k3_router_args &args,
        device const float      *logits,
        device const float      *correction_bias,
        device int              *selected,
        device float            *weights,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= args.n_rows || tid != 0u) return;
    device const float *lr = logits + (ulong)row * args.n_experts;
    device int *ids = selected + (ulong)row * args.top_k;
    device float *wr = weights + (ulong)row * args.top_k;
    float score[16];

    for (uint j = 0; j < args.top_k; j++) {
        ids[j] = -1;
        score[j] = -INFINITY;
        wr[j] = 0.0f;
    }
    for (uint expert = 0; expert < args.n_experts; expert++) {
        const float raw = 1.0f / (1.0f + exp(-lr[expert]));
        const float candidate = raw + correction_bias[expert];
        uint slot = args.top_k;
        for (uint j = 0; j < args.top_k; j++) {
            if (candidate > score[j] ||
                (candidate == score[j] && (int)expert < ids[j])) {
                slot = j;
                break;
            }
        }
        if (slot < args.top_k) {
            for (uint j = args.top_k - 1u; j > slot; j--) {
                score[j] = score[j - 1u];
                ids[j] = ids[j - 1u];
            }
            score[slot] = candidate;
            ids[slot] = (int)expert;
        }
    }
    float total = 0.0f;
    for (uint j = 0; j < args.top_k; j++) {
        const float raw = 1.0f / (1.0f + exp(-lr[ids[j]]));
        wr[j] = raw;
        total += raw;
    }
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    for (uint j = 0; j < args.top_k; j++) wr[j] *= inv;
}

/*
 * Parallelize only the independent sigmoid+bias evaluations. Lane zero keeps
 * the original expert-order insertion and normalization loops, preserving
 * tie-breaking and accumulation order.
 */
kernel void kernel_k3_router_topk_parallel_score(
        constant k3_router_args &args,
        device const float      *logits,
        device const float      *correction_bias,
        device int              *selected,
        device float            *weights,
        threadgroup float       *candidates [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= args.n_rows) return;
    device const float *lr = logits + (ulong)row * args.n_experts;
    for (uint expert = tid; expert < args.n_experts; expert += 32u) {
        const float raw = 1.0f / (1.0f + exp(-lr[expert]));
        candidates[expert] = raw + correction_bias[expert];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;

    device int *ids = selected + (ulong)row * args.top_k;
    device float *wr = weights + (ulong)row * args.top_k;
    float score[16];
    for (uint j = 0; j < args.top_k; j++) {
        ids[j] = -1;
        score[j] = -INFINITY;
        wr[j] = 0.0f;
    }
    for (uint expert = 0; expert < args.n_experts; expert++) {
        const float candidate = candidates[expert];
        uint slot = args.top_k;
        for (uint j = 0; j < args.top_k; j++) {
            if (candidate > score[j] ||
                (candidate == score[j] && (int)expert < ids[j])) {
                slot = j;
                break;
            }
        }
        if (slot < args.top_k) {
            for (uint j = args.top_k - 1u; j > slot; j--) {
                score[j] = score[j - 1u];
                ids[j] = ids[j - 1u];
            }
            score[slot] = candidate;
            ids[slot] = (int)expert;
        }
    }
    float total = 0.0f;
    for (uint j = 0; j < args.top_k; j++) {
        const float raw = 1.0f / (1.0f + exp(-lr[ids[j]]));
        wr[j] = raw;
        total += raw;
    }
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    for (uint j = 0; j < args.top_k; j++) wr[j] *= inv;
}

/*
 * Give every expert its exact position in descending-score, ascending-ID
 * order. Only the first top_k ranks write outputs, so writes are unique and
 * the lane-zero normalization retains the reference accumulation order.
 */
kernel void kernel_k3_router_topk_rank(
        constant k3_router_args &args,
        device const float      *logits,
        device const float      *correction_bias,
        device int              *selected,
        device float            *weights,
        threadgroup float       *candidates [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        uint3 tptg [[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    if (row >= args.n_rows) return;
    device const float *lr = logits + (ulong)row * args.n_experts;
    for (uint expert = tid; expert < args.n_experts; expert += tptg.x) {
        const float raw = 1.0f / (1.0f + exp(-lr[expert]));
        candidates[expert] = raw + correction_bias[expert];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device int *ids = selected + (ulong)row * args.top_k;
    for (uint expert = tid; expert < args.n_experts; expert += tptg.x) {
        const float candidate = candidates[expert];
        uint rank = 0u;
        for (uint other = 0; other < args.n_experts; other++) {
            const float score = candidates[other];
            rank += score > candidate ||
                    (score == candidate && other < expert);
            if (rank >= args.top_k) break;
        }
        if (rank < args.top_k) ids[rank] = (int)expert;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;

    device float *wr = weights + (ulong)row * args.top_k;
    float total = 0.0f;
    for (uint j = 0; j < args.top_k; j++) {
        const float raw = 1.0f / (1.0f + exp(-lr[ids[j]]));
        wr[j] = raw;
        total += raw;
    }
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    for (uint j = 0; j < args.top_k; j++) wr[j] *= inv;
}

struct k3_router_pair {
    float score;
    float raw;
    uint expert;
};

static inline bool k3_router_pair_before(
        k3_router_pair a,
        k3_router_pair b) {
    return a.score > b.score ||
           (a.score == b.score && a.expert < b.expert);
}

/*
 * Sort one power-of-two threadgroup in descending-score, ascending-ID order.
 * K3 has 896 experts, so 128 sentinel lanes bring the network to 1024.
 */
kernel void kernel_k3_router_topk_bitonic(
        constant k3_router_args &args,
        device const float      *logits,
        device const float      *correction_bias,
        device int              *selected,
        device float            *weights,
        threadgroup k3_router_pair *pairs [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= args.n_rows) return;
    device const float *lr = logits + (ulong)row * args.n_experts;
    if (tid < args.n_experts) {
        const float raw = 1.0f / (1.0f + exp(-lr[tid]));
        pairs[tid] =
            k3_router_pair{raw + correction_bias[tid], raw, tid};
    } else {
        pairs[tid] = k3_router_pair{-INFINITY, 0.0f, (uint)tid};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 2u; width <= 1024u; width <<= 1u) {
        for (uint stride = width >> 1u; stride != 0u; stride >>= 1u) {
            const uint other = (uint)tid ^ stride;
            if (other > tid) {
                const k3_router_pair a = pairs[tid];
                const k3_router_pair b = pairs[other];
                const bool descending = ((uint)tid & width) == 0u;
                const bool swap = descending
                    ? k3_router_pair_before(b, a)
                    : k3_router_pair_before(a, b);
                if (swap) {
                    pairs[tid] = b;
                    pairs[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid != 0u) return;

    device int *ids = selected + (ulong)row * args.top_k;
    device float *wr = weights + (ulong)row * args.top_k;
    float total = 0.0f;
    for (uint j = 0; j < args.top_k; j++) {
        ids[j] = (int)pairs[j].expert;
        const float raw = pairs[j].raw;
        wr[j] = raw;
        total += raw;
    }
    const float inv = total > 0.0f ? 1.0f / total : 0.0f;
    for (uint j = 0; j < args.top_k; j++) wr[j] *= inv;
}

struct k3_attn_res_args {
    uint width;
    uint n_rows;
    uint block_capacity;
    uint num_blocks;
    int block_write_idx;
    uint has_delta;
    uint preserve_prefix;
    uint apply_output_norm;
    float eps;
    float output_norm_eps;
};

kernel void kernel_k3_attn_res_prepare(
        constant k3_attn_res_args &args,
        device float              *prefix,
        device const float        *delta,
        device float              *blocks,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= args.width || row >= args.n_rows) return;
    device float *pr = prefix + (ulong)row * args.width;
    if (args.has_delta) {
        device const float *dr = delta + (ulong)row * args.width;
        pr[d] = k3_round_bf16(pr[d] + dr[d]);
    } else if (!args.preserve_prefix) {
        pr[d] = k3_round_bf16(pr[d]);
    }
    if (args.block_write_idx >= 0) {
        device float *br = blocks +
            (ulong)row * args.block_capacity * args.width;
        br[(ulong)(uint)args.block_write_idx * args.width + d] = pr[d];
    }
}

kernel void kernel_k3_attn_res_score(
        constant k3_attn_res_args &args,
        device const float        *prefix,
        device const float        *blocks,
        device const ushort       *norm_weight,
        device const ushort       *proj_weight,
        device float              *out,
        threadgroup float         *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint row = tgpig.x;
    const uint source = tgpig.y;
    const uint sources = args.num_blocks + 1u;
    if (row >= args.n_rows || source >= sources) return;
    device const float *pr = prefix + (ulong)row * args.width;
    device const float *br = blocks +
        (ulong)row * args.block_capacity * args.width;
    device const float *v = source == args.num_blocks
        ? pr
        : br + (ulong)source * args.width;
    float sumsq = 0.0f;
    float dotp = 0.0f;
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        const float value = v[d];
        sumsq = fma(value, value, sumsq);
        dotp = fma(value,
                   k3_bf16_to_f32(norm_weight[d]) *
                   k3_bf16_to_f32(proj_weight[d]),
                   dotp);
    }
    sumsq = simd_sum(sumsq);
    dotp = simd_sum(dotp);
    if (lane == 0u) {
        scratch[sg] = sumsq;
        scratch[8u + sg] = dotp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sumsq_total = 0.0f;
        float dotp_total = 0.0f;
        for (uint part = 0; part < nsg; part++) {
            sumsq_total += scratch[part];
            dotp_total += scratch[8u + part];
        }
        out[(ulong)row * args.width + source] =
            dotp_total *
            rsqrt(sumsq_total / (float)args.width + args.eps);
    }
}

kernel void kernel_k3_attn_res_mix(
        constant k3_attn_res_args &args,
        device const float        *prefix,
        device const float        *blocks,
        device const ushort       *output_norm_weight,
        device float              *out,
        threadgroup float         *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.n_rows) return;
    const uint sources = args.num_blocks + 1u;
    device const float *pr = prefix + (ulong)row * args.width;
    device const float *br = blocks +
        (ulong)row * args.block_capacity * args.width;
    device float *yr = out + (ulong)row * args.width;
    if (tid == 0u) {
        float max_score = -INFINITY;
        for (uint source = 0; source < sources; source++) {
            const float score = yr[source];
            scratch[source] = score;
            max_score = max(max_score, score);
        }
        float denominator = 0.0f;
        for (uint source = 0; source < sources; source++) {
            const float w = exp(scratch[source] - max_score);
            scratch[16u + source] = w;
            denominator += w;
        }
        for (uint source = 0; source < sources; source++) {
            scratch[16u + source] /= denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sumsq = 0.0f;
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        float mixed = 0.0f;
        for (uint source = 0; source < sources; source++) {
            device const float *v = source == args.num_blocks
                ? pr
                : br + (ulong)source * args.width;
            mixed = fma(scratch[16u + source], v[d], mixed);
        }
        yr[d] = mixed;
        local_sumsq = fma(mixed, mixed, local_sumsq);
    }
    if (!args.apply_output_norm) return;
    local_sumsq = simd_sum(local_sumsq);
    if (lane == 0u) scratch[sg] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < nsg ? scratch[lane] : 0.0f;
    total = simd_sum(total);
    const float scale =
        rsqrt(total / (float)args.width + args.output_norm_eps);
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        yr[d] *= scale * k3_bf16_to_f32(output_norm_weight[d]);
    }
}

kernel void kernel_k3_attn_res(
        constant k3_attn_res_args &args,
        device float              *prefix,
        device const float        *delta,
        device float              *blocks,
        device const ushort       *norm_weight,
        device const ushort       *proj_weight,
        device const ushort       *output_norm_weight,
        device float              *out,
        threadgroup float         *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.n_rows) return;
    const uint sources = args.num_blocks + 1u;
    device float *pr = prefix + (ulong)row * args.width;
    device float *br = blocks +
        (ulong)row * args.block_capacity * args.width;
    device float *yr = out + (ulong)row * args.width;

    if (args.has_delta) {
        device const float *dr = delta + (ulong)row * args.width;
        for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
            pr[d] = k3_round_bf16(pr[d] + dr[d]);
        }
    } else if (!args.preserve_prefix) {
        for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
            pr[d] = k3_round_bf16(pr[d]);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (args.block_write_idx >= 0) {
        device float *dst = br + (ulong)(uint)args.block_write_idx * args.width;
        for (uint d = tid; d < args.width; d += (uint)nsg * 32u) dst[d] = pr[d];
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint source = 0; source < sources; source++) {
        device const float *v = source == args.num_blocks
            ? pr
            : br + (ulong)source * args.width;
        float sumsq = 0.0f;
        float dotp = 0.0f;
        for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
            const float value = v[d];
            sumsq = fma(value, value, sumsq);
            dotp = fma(value,
                       k3_bf16_to_f32(norm_weight[d]) *
                       k3_bf16_to_f32(proj_weight[d]),
                       dotp);
        }
        sumsq = simd_sum(sumsq);
        dotp = simd_sum(dotp);
        if (lane == 0u) {
            scratch[source * 16u + sg] = sumsq;
            scratch[source * 16u + 8u + sg] = dotp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u && lane < sources) {
        float sumsq = 0.0f;
        float dotp = 0.0f;
        for (uint part = 0; part < nsg; part++) {
            sumsq += scratch[lane * 16u + part];
            dotp += scratch[lane * 16u + 8u + part];
        }
        scratch[144u + lane] =
            dotp * rsqrt(sumsq / (float)args.width + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float max_score = -INFINITY;
        for (uint source = 0; source < sources; source++) {
            max_score = max(max_score, scratch[144u + source]);
        }
        float denominator = 0.0f;
        for (uint source = 0; source < sources; source++) {
            const float w = exp(scratch[144u + source] - max_score);
            scratch[160u + source] = w;
            denominator += w;
        }
        for (uint source = 0; source < sources; source++) {
            scratch[160u + source] /= denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sumsq = 0.0f;
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        float mixed = 0.0f;
        for (uint source = 0; source < sources; source++) {
            device const float *v = source == args.num_blocks
                ? pr
                : br + (ulong)source * args.width;
            mixed = fma(scratch[160u + source], v[d], mixed);
        }
        yr[d] = mixed;
        local_sumsq = fma(mixed, mixed, local_sumsq);
    }

    if (!args.apply_output_norm) return;
    local_sumsq = simd_sum(local_sumsq);
    if (lane == 0u) scratch[sg] = local_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < nsg ? scratch[lane] : 0.0f;
    total = simd_sum(total);
    const float scale =
        rsqrt(total / (float)args.width + args.output_norm_eps);
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        yr[d] *= scale * k3_bf16_to_f32(output_norm_weight[d]);
    }
}

struct k3_kda_args {
    uint n_heads;
    uint n_rows;
    float lower_bound;
    float norm_eps;
};

/*
 * One threadgroup owns one (sequence, head). Four simdgroups update four
 * value rows concurrently; every lane owns four adjacent key columns.
 */
kernel void kernel_k3_kda_decode(
        constant k3_kda_args &args,
        device const float   *q_in,
        device const float   *k_in,
        device const float   *v_in,
        device const float   *raw_gate,
        device const float   *raw_beta,
        device const float   *output_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device const float   *output_norm,
        device float         *conv_state,
        device float         *state,
        device float         *out,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *sd = sk + D;
    threadgroup float *sv = sd + D;
    threadgroup float *so = sv + D;
    threadgroup float *reduce_q = so + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    threadgroup float *reduce_o = reduce_k + 4u;
    threadgroup float *beta_shared = reduce_o + 4u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    const ulong conv_row_stride = 3ul * HISTORY * projection;

    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float *q_state = conv_state +
            (ulong)row * conv_row_stride;
        device float *k_state = q_state + HISTORY * projection;
        device float *v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float q_sumsq = sq[tid] * sq[tid];
    float k_sumsq = sk[tid] * sk[tid];
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head =
        ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += 4u) {
        device float4 *hptr =
            (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup |
                       mem_flags::mem_device);

    float o_sumsq = so[tid] * so[tid];
    o_sumsq = simd_sum(o_sumsq);
    if (lane == 0u) reduce_o[sg] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = simd_sum(o_total);
    const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
    if (tid < D) {
        const ulong index = input_base + tid;
        const float gate =
            1.0f / (1.0f + exp(-output_gate[index]));
        out[index] = so[tid] * o_scale * output_norm[tid] * gate;
    }
}

kernel void kernel_k3_kda_prefill_prepare(
        constant k3_kda_args &args,
        device float         *q,
        device float         *k,
        device float         *v,
        device float         *raw_gate,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    if (head >= args.n_heads) return;
    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *reduce_q = sk + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    device float *q_state = conv_state;
    device float *k_state = q_state + HISTORY * projection;
    device float *v_state = k_state + HISTORY * projection;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong index = (ulong)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);
        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        v[index] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[index] + dt_bias[channel];
        raw_gate[index] = exp(args.lower_bound *
            (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        q[index] = sq[tid] * rsqrt(q_total + 1.0e-6f) *
                   0x1.6a09e6p-4f;
        k[index] = sk[tid] * rsqrt(k_total + 1.0e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);
    }
}

kernel void kernel_k3_kda_prefill_recurrence(
        constant k3_kda_args &args,
        device const float   *q,
        device const float   *k,
        device const float   *v,
        device const float   *decay,
        device const float   *raw_beta,
        device float         *state,
        device float         *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value = tgpig.y * 4u + sg;
    if (head >= args.n_heads || value >= D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    device float4 *state_ptr = (device float4 *)(
        state + ((ulong)head * D + value) * D + k0);
    float4 h = *state_ptr;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 =
            *((device const float4 *)(decay + base + k0));
        h *= decay4;
        const float hk = simd_sum(dot(h, k4));
        const float beta = 1.0f /
            (1.0f + exp(-raw_beta[(ulong)token * args.n_heads + head]));
        const float delta_v = (v[base + value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        const float result = simd_sum(dot(h, q4));
        if (lane == 0u) out[base + value] = result;
    }
    *state_ptr = h;
}

kernel void kernel_k3_kda_prefill_output(
        constant k3_kda_args &args,
        device float         *out,
        device const float   *output_gate,
        device const float   *output_norm,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_rows || head >= args.n_heads) return;
    const uint projection = args.n_heads * D;
    const ulong base = (ulong)token * projection + head * D;
    const float raw = out[base + tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 4u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    out[base + tid] = raw * scale * output_norm[tid] /
        (1.0f + exp(-output_gate[base + tid]));
}

struct k3_mla_args {
    uint n_heads;
    uint position;
    uint cache_capacity;
    uint max_partials;
    uint n_partials;
};

kernel void kernel_k3_mla_store_cache(
        constant k3_mla_args &args,
        device const float   *normalized_kv,
        device ushort        *cache,
        uint d [[thread_position_in_grid]]) {
    if (d >= 576u || args.position >= args.cache_capacity) return;
    cache[(ulong)args.position * 576u + d] =
        k3_f32_to_bf16_rne(normalized_kv[d]);
}

kernel void kernel_k3_mla_absorb_q(
        constant k3_mla_args &args,
        device const float   *q,
        device const ushort  *kv_b,
        device float         *q_latent,
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (head >= args.n_heads) return;
    device const float *qh = q + (ulong)head * 192u;
    for (uint latent = tid; latent < 512u; latent += 128u) {
        float sum = 0.0f;
        for (uint d = 0; d < 128u; d += 16u) {
            const ulong weight =
                ((ulong)head * 256u + d) * 512u + latent;
            const float q0 = qh[d];
            const float q1 = qh[d + 1u];
            const float q2 = qh[d + 2u];
            const float q3 = qh[d + 3u];
            const float q4 = qh[d + 4u];
            const float q5 = qh[d + 5u];
            const float q6 = qh[d + 6u];
            const float q7 = qh[d + 7u];
            const float q8 = qh[d + 8u];
            const float q9 = qh[d + 9u];
            const float q10 = qh[d + 10u];
            const float q11 = qh[d + 11u];
            const float q12 = qh[d + 12u];
            const float q13 = qh[d + 13u];
            const float q14 = qh[d + 14u];
            const float q15 = qh[d + 15u];
            const ushort w0 = kv_b[weight];
            const ushort w1 = kv_b[weight + 512u];
            const ushort w2 = kv_b[weight + 1024u];
            const ushort w3 = kv_b[weight + 1536u];
            const ushort w4 = kv_b[weight + 2048u];
            const ushort w5 = kv_b[weight + 2560u];
            const ushort w6 = kv_b[weight + 3072u];
            const ushort w7 = kv_b[weight + 3584u];
            const ushort w8 = kv_b[weight + 4096u];
            const ushort w9 = kv_b[weight + 4608u];
            const ushort w10 = kv_b[weight + 5120u];
            const ushort w11 = kv_b[weight + 5632u];
            const ushort w12 = kv_b[weight + 6144u];
            const ushort w13 = kv_b[weight + 6656u];
            const ushort w14 = kv_b[weight + 7168u];
            const ushort w15 = kv_b[weight + 7680u];
            sum = fma(q0, k3_bf16_to_f32(w0), sum);
            sum = fma(q1, k3_bf16_to_f32(w1), sum);
            sum = fma(q2, k3_bf16_to_f32(w2), sum);
            sum = fma(q3, k3_bf16_to_f32(w3), sum);
            sum = fma(q4, k3_bf16_to_f32(w4), sum);
            sum = fma(q5, k3_bf16_to_f32(w5), sum);
            sum = fma(q6, k3_bf16_to_f32(w6), sum);
            sum = fma(q7, k3_bf16_to_f32(w7), sum);
            sum = fma(q8, k3_bf16_to_f32(w8), sum);
            sum = fma(q9, k3_bf16_to_f32(w9), sum);
            sum = fma(q10, k3_bf16_to_f32(w10), sum);
            sum = fma(q11, k3_bf16_to_f32(w11), sum);
            sum = fma(q12, k3_bf16_to_f32(w12), sum);
            sum = fma(q13, k3_bf16_to_f32(w13), sum);
            sum = fma(q14, k3_bf16_to_f32(w14), sum);
            sum = fma(q15, k3_bf16_to_f32(w15), sum);
        }
        q_latent[(ulong)head * 512u + latent] = sum;
    }
}

/*
 * Each simdgroup owns one head and one 1024-token context interval. A lane
 * carries 16 latent accumulator elements, so the online softmax needs no
 * threadgroup synchronization.
 */
kernel void kernel_k3_mla_attention_partials(
        constant k3_mla_args &args,
        device const float   *q,
        device const float   *q_latent,
        device const ushort  *cache,
        device float         *partial_values,
        device float2        *partial_stats,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BLOCK_T = 1024u;
    const uint head = tgpig.x * 4u + sg;
    const uint part = tgpig.y;
    if (head >= args.n_heads || part >= args.n_partials) return;
    const uint begin = part * BLOCK_T;
    const uint end = min(begin + BLOCK_T, args.position + 1u);
    device const float *ql = q_latent + (ulong)head * 512u;
    device const float *qpe = q + (ulong)head * 192u + 128u;
    float accumulator[16];
    for (uint j = 0; j < 16u; j++) accumulator[j] = 0.0f;
    float running_max = -INFINITY;
    float denominator = 0.0f;

    for (uint token = begin; token < end; token++) {
        device const ushort *kv = cache + (ulong)token * 576u;
        float kv_values[16];
        float score = 0.0f;
        for (uint j = 0; j < 16u; j++) {
            const uint latent = lane + j * 32u;
            kv_values[j] = k3_bf16_to_f32(kv[latent]);
            score = fma(ql[latent], kv_values[j], score);
        }
        score = fma(qpe[lane],
                    k3_bf16_to_f32(kv[512u + lane]), score);
        score = fma(qpe[32u + lane],
                    k3_bf16_to_f32(kv[544u + lane]), score);
        score = simd_sum(score) * rsqrt(192.0f);

        const float new_max = max(running_max, score);
        const float old_scale = exp(running_max - new_max);
        const float token_scale = exp(score - new_max);
        denominator = denominator * old_scale + token_scale;
        for (uint j = 0; j < 16u; j++) {
            accumulator[j] = accumulator[j] * old_scale +
                token_scale * kv_values[j];
        }
        running_max = new_max;
    }

    const ulong base =
        ((ulong)head * args.max_partials + part) * 512u;
    for (uint j = 0; j < 16u; j++) {
        partial_values[base + lane + j * 32u] = accumulator[j];
    }
    if (lane == 0u) {
        partial_stats[(ulong)head * args.max_partials + part] =
            float2(running_max, denominator);
    }
}

kernel void kernel_k3_mla_reduce_partials(
        constant k3_mla_args &args,
        device const float   *partial_values,
        device const float2  *partial_stats,
        device float         *latent_out,
        threadgroup float2   *shared,
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (head >= args.n_heads) return;
    if (tid == 0u) {
        float global_max = -INFINITY;
        for (uint part = 0; part < args.n_partials; part++) {
            global_max = max(
                global_max,
                partial_stats[(ulong)head * args.max_partials + part].x);
        }
        float denominator = 0.0f;
        for (uint part = 0; part < args.n_partials; part++) {
            const float2 stats =
                partial_stats[(ulong)head * args.max_partials + part];
            denominator += stats.y * exp(stats.x - global_max);
        }
        shared[0] = float2(global_max, denominator);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float global_max = shared[0].x;
    const float inv_denom = 1.0f / shared[0].y;
    for (uint latent = tid; latent < 512u; latent += 128u) {
        float sum = 0.0f;
        for (uint part = 0; part < args.n_partials; part++) {
            const float2 stats =
                partial_stats[(ulong)head * args.max_partials + part];
            const ulong index =
                ((ulong)head * args.max_partials + part) * 512u + latent;
            sum = fma(partial_values[index],
                      exp(stats.x - global_max), sum);
        }
        latent_out[(ulong)head * 512u + latent] = sum * inv_denom;
    }
}

kernel void kernel_k3_mla_value_up_gate(
        constant k3_mla_args &args,
        device const float   *latent,
        device const ushort  *kv_b,
        device const float   *output_gate,
        device float         *out,
        uint head [[threadgroup_position_in_grid]],
        ushort value [[thread_index_in_threadgroup]]) {
    if (head >= args.n_heads || value >= 128u) return;
    device const float *lh = latent + (ulong)head * 512u;
    const ulong weight_row =
        ((ulong)head * 256u + 128u + value) * 512u;
    float sum = 0.0f;
    for (uint k = 0; k < 512u; k += 16u) {
        const float x0 = lh[k];
        const float x1 = lh[k + 1u];
        const float x2 = lh[k + 2u];
        const float x3 = lh[k + 3u];
        const float x4 = lh[k + 4u];
        const float x5 = lh[k + 5u];
        const float x6 = lh[k + 6u];
        const float x7 = lh[k + 7u];
        const float x8 = lh[k + 8u];
        const float x9 = lh[k + 9u];
        const float x10 = lh[k + 10u];
        const float x11 = lh[k + 11u];
        const float x12 = lh[k + 12u];
        const float x13 = lh[k + 13u];
        const float x14 = lh[k + 14u];
        const float x15 = lh[k + 15u];
        const ushort w0 = kv_b[weight_row + k];
        const ushort w1 = kv_b[weight_row + k + 1u];
        const ushort w2 = kv_b[weight_row + k + 2u];
        const ushort w3 = kv_b[weight_row + k + 3u];
        const ushort w4 = kv_b[weight_row + k + 4u];
        const ushort w5 = kv_b[weight_row + k + 5u];
        const ushort w6 = kv_b[weight_row + k + 6u];
        const ushort w7 = kv_b[weight_row + k + 7u];
        const ushort w8 = kv_b[weight_row + k + 8u];
        const ushort w9 = kv_b[weight_row + k + 9u];
        const ushort w10 = kv_b[weight_row + k + 10u];
        const ushort w11 = kv_b[weight_row + k + 11u];
        const ushort w12 = kv_b[weight_row + k + 12u];
        const ushort w13 = kv_b[weight_row + k + 13u];
        const ushort w14 = kv_b[weight_row + k + 14u];
        const ushort w15 = kv_b[weight_row + k + 15u];
        sum = fma(x0, k3_bf16_to_f32(w0), sum);
        sum = fma(x1, k3_bf16_to_f32(w1), sum);
        sum = fma(x2, k3_bf16_to_f32(w2), sum);
        sum = fma(x3, k3_bf16_to_f32(w3), sum);
        sum = fma(x4, k3_bf16_to_f32(w4), sum);
        sum = fma(x5, k3_bf16_to_f32(w5), sum);
        sum = fma(x6, k3_bf16_to_f32(w6), sum);
        sum = fma(x7, k3_bf16_to_f32(w7), sum);
        sum = fma(x8, k3_bf16_to_f32(w8), sum);
        sum = fma(x9, k3_bf16_to_f32(w9), sum);
        sum = fma(x10, k3_bf16_to_f32(w10), sum);
        sum = fma(x11, k3_bf16_to_f32(w11), sum);
        sum = fma(x12, k3_bf16_to_f32(w12), sum);
        sum = fma(x13, k3_bf16_to_f32(w13), sum);
        sum = fma(x14, k3_bf16_to_f32(w14), sum);
        sum = fma(x15, k3_bf16_to_f32(w15), sum);
    }
    const ulong output = (ulong)head * 128u + value;
    out[output] = sum /
        (1.0f + exp(-output_gate[output]));
}

struct k3_mla_prefill_args {
    uint n_heads;
    uint start_position;
    uint n_tokens;
    uint cache_capacity;
};

kernel void kernel_k3_mla_prefill_store_cache(
        constant k3_mla_prefill_args &args,
        device const float           *normalized_kv,
        device ushort                *cache,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    const uint position = args.start_position + row;
    if (d >= 576u || row >= args.n_tokens ||
        position >= args.cache_capacity) {
        return;
    }
    cache[(ulong)position * 576u + d] =
        k3_f32_to_bf16_rne(normalized_kv[(ulong)row * 576u + d]);
}

kernel void kernel_k3_mla_prefill_absorb_q(
        constant k3_mla_prefill_args &args,
        device const float           *q,
        device const ushort          *kv_b,
        device float                 *q_latent,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_tokens || head >= args.n_heads) return;
    device const float *qh =
        q + ((ulong)row * args.n_heads + head) * 192u;
    device float *ql =
        q_latent + ((ulong)row * args.n_heads + head) * 512u;
    for (uint latent = tid; latent < 512u; latent += 128u) {
        float sum = 0.0f;
        for (uint d = 0; d < 128u; d++) {
            const ulong weight =
                ((ulong)head * 256u + d) * 512u + latent;
            sum = fma(qh[d], k3_bf16_to_f32(kv_b[weight]), sum);
        }
        ql[latent] = sum;
    }
}

kernel void kernel_k3_mla_prefill_attention(
        constant k3_mla_prefill_args &args,
        device const float           *q,
        device float                 *q_latent,
        device const ushort          *cache,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint head = tgpig.x * 4u + sg;
    const uint row = tgpig.y;
    if (head >= args.n_heads || row >= args.n_tokens) return;
    const uint last_key = args.start_position + row;
    device const float *ql =
        q_latent + ((ulong)row * args.n_heads + head) * 512u;
    device const float *qpe =
        q + ((ulong)row * args.n_heads + head) * 192u + 128u;
    float accumulator[16];
    for (uint j = 0; j < 16u; j++) accumulator[j] = 0.0f;
    float running_max = -INFINITY;
    float denominator = 0.0f;

    for (uint token = 0; token <= last_key; token++) {
        device const ushort *kv = cache + (ulong)token * 576u;
        float score = 0.0f;
        for (uint j = 0; j < 16u; j++) {
            const uint latent = lane + j * 32u;
            score = fma(ql[latent], k3_bf16_to_f32(kv[latent]), score);
        }
        score = fma(qpe[lane],
                    k3_bf16_to_f32(kv[512u + lane]), score);
        score = fma(qpe[32u + lane],
                    k3_bf16_to_f32(kv[544u + lane]), score);
        score = simd_sum(score) * rsqrt(192.0f);
        const float new_max = max(running_max, score);
        const float old_scale = exp(running_max - new_max);
        const float token_scale = exp(score - new_max);
        denominator = denominator * old_scale + token_scale;
        for (uint j = 0; j < 16u; j++) {
            const uint latent = lane + j * 32u;
            accumulator[j] = accumulator[j] * old_scale +
                token_scale * k3_bf16_to_f32(kv[latent]);
        }
        running_max = new_max;
    }
    const float inv_denom = 1.0f / denominator;
    for (uint j = 0; j < 16u; j++) {
        q_latent[
            ((ulong)row * args.n_heads + head) * 512u +
            lane + j * 32u] = accumulator[j] * inv_denom;
    }
}

kernel void kernel_k3_mla_prefill_value_up_gate(
        constant k3_mla_prefill_args &args,
        device const float           *latent,
        device const ushort          *kv_b,
        device const float           *output_gate,
        device float                 *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort value [[thread_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_tokens || head >= args.n_heads || value >= 128u) return;
    device const float *lh =
        latent + ((ulong)row * args.n_heads + head) * 512u;
    const ulong weight_row =
        ((ulong)head * 256u + 128u + value) * 512u;
    float sum = 0.0f;
    for (uint k = 0; k < 512u; k++) {
        sum = fma(lh[k], k3_bf16_to_f32(kv_b[weight_row + k]), sum);
    }
    const ulong output =
        ((ulong)row * args.n_heads + head) * 128u + value;
    out[output] = sum /
        (1.0f + exp(-output_gate[output]));
}
