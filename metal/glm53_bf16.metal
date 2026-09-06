// BF16 model-weight kernels used by GLM-5.3 Flash.

static inline float glm53_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

struct glm53_bf16_matmul_args {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

kernel void kernel_glm53_embedding_bf16(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const int                *tokens,
        device float                    *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= args.in_dim || row >= args.n_rows) return;
    const int token = tokens[row];
    out[(ulong)row * args.in_dim + d] =
        token >= 0 && (uint)token < args.out_dim
            ? glm53_bf16_to_f32(weights[(ulong)(uint)token * args.in_dim + d])
            : 0.0f;
}

static inline void glm53_mul_mv_bf16_f32_row(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        uint2                            tgpig,
        ushort                           lane,
        ushort                           sg,
        ushort                           nsg) {
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
        sum = fma(glm53_bf16_to_f32(w0), x0, sum);
        sum = fma(glm53_bf16_to_f32(w1), x1, sum);
        sum = fma(glm53_bf16_to_f32(w2), x2, sum);
        sum = fma(glm53_bf16_to_f32(w3), x3, sum);
        sum = fma(glm53_bf16_to_f32(w4), x4, sum);
        sum = fma(glm53_bf16_to_f32(w5), x5, sum);
        sum = fma(glm53_bf16_to_f32(w6), x6, sum);
        sum = fma(glm53_bf16_to_f32(w7), x7, sum);
    }
    for (; k < args.in_dim; k += 32u) {
        sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[(ulong)token * args.out_dim + out_row] = sum;
}

/* One simdgroup owns one output row. Eight independent loads expose enough
 * memory-level parallelism for decode without changing the reduction tree. */
kernel void kernel_glm53_mul_mv_bf16_f32(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    glm53_mul_mv_bf16_f32_row(args, weights, x, out,
                              tgpig, lane, sg, nsg);
}

kernel void kernel_glm53_mul_mv_bf16_f32_qkv(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights_q,
        device const ushort             *weights_k,
        device const ushort             *weights_v,
        device const float              *x,
        device float                    *out_q,
        device float                    *out_k,
        device float                    *out_v,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const ushort *weights = tgpig.z == 0u ? weights_q :
                                     (tgpig.z == 1u ? weights_k : weights_v);
    device float *out = tgpig.z == 0u ? out_q :
                            (tgpig.z == 1u ? out_k : out_v);
    glm53_mul_mv_bf16_f32_row(args, weights, x, out,
                              tgpig.xy, lane, sg, nsg);
}

struct glm53_bf16_block16 {
    ushort v[16];
};

template <typename type4x4>
void glm53_dequantize_bf16(
        device const glm53_bf16_block16 *src,
        short il,
        thread type4x4 &reg) {
    (void)il;
    float4x4 values;
    for (short i = 0; i < 16; i++) {
        values[i / 4][i % 4] = glm53_bf16_to_f32(src->v[i]);
    }
    reg = (type4x4)values;
}

typedef decltype(kernel_mul_mm<
        half, half4x4, simdgroup_half8x8,
        half, half2x4, simdgroup_half8x8,
        glm53_bf16_block16, 1, glm53_dequantize_bf16,
        float, float4x4, float, float2x4>) glm53_mul_mm_bf16_t;

template [[host_name("kernel_glm53_mul_mm_bf16_f32")]]
kernel glm53_mul_mm_bf16_t kernel_mul_mm<
        half, half4x4, simdgroup_half8x8,
        half, half2x4, simdgroup_half8x8,
        glm53_bf16_block16, 1, glm53_dequantize_bf16,
        half, half4x4, float, float2x4>;
