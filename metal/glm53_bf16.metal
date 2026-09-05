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

/* The accumulation, split out unchanged so an epilogue kernel can use the sum
 * before it is stored.  Callers must range-check out_row and token first. */
static inline float glm53_mul_mv_bf16_f32_row_sum(
        uint                             in_dim,
        device const ushort             *weights,
        device const float              *x,
        uint                             out_row,
        uint                             token,
        ushort                           lane) {
    device const ushort *w = weights + (ulong)out_row * in_dim;
    device const float *xr = x + (ulong)token * in_dim;
    float sum = 0.0f;
    uint k = lane;
    for (; k + 224u < in_dim; k += 256u) {
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
    for (; k < in_dim; k += 32u) {
        sum = fma(glm53_bf16_to_f32(w[k]), xr[k], sum);
    }
    return simd_sum(sum);
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
    const float sum =
        glm53_mul_mv_bf16_f32_row_sum(args.in_dim, weights, x, out_row, token, lane);
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

/*
 * BF16 matvec with the HC expansion folded into its epilogue.
 *
 * The simdgroup that finishes output row d already holds that row's value in
 * lane 0, so it can expand it into the four HC streams there instead of
 * writing it out and having a second dispatch read it straight back.  This is
 * the shape kernel_dsv4_q8_hc_expand4_q8_0 already uses for DeepSeek, in BF16.
 *
 * Decode only: one token, HC = 4.  The arithmetic and the operand order match
 * kernel_dsv4_hc_expand4 exactly, including that comb is indexed [j][h].
 */
kernel void kernel_glm53_mul_mv_bf16_f32_hc_expand4(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights,
        device const float              *x,
        device float                    *out,
        device const float              *residual,
        device const float              *post,
        device const float              *comb,
        device float                    *hc_out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (out_row >= args.out_dim || token >= args.n_rows) return;
    const float sum =
        glm53_mul_mv_bf16_f32_row_sum(args.in_dim, weights, x, out_row, token, lane);
    if (lane != 0u) return;
    out[(ulong)token * args.out_dim + out_row] = sum;

    const uint n = args.out_dim;
    const float r0 = residual[0u * n + out_row];
    const float r1 = residual[1u * n + out_row];
    const float r2 = residual[2u * n + out_row];
    const float r3 = residual[3u * n + out_row];
    for (uint h = 0u; h < 4u; ++h) {
        float acc = sum * post[h];
        acc += comb[0u * 4u + h] * r0;
        acc += comb[1u * 4u + h] * r1;
        acc += comb[2u * 4u + h] * r2;
        acc += comb[3u * 4u + h] * r3;
        hc_out[h * n + out_row] = acc;
    }
}

struct glm53_bf16_trio_args {
    uint in_dim;
    uint out_dim_ab;
    uint out_dim_c;
    uint n_rows;
};

/*
 * Three matvecs over one shared input in a single dispatch, where the third
 * has a shorter output than the first two.  GLM 5.3's KDA gate chain is
 * exactly that shape: f_a and g_a are [4096 -> 128] and beta is [4096 -> 64],
 * all reading attn_norm.  The pair kernel could not carry beta because it
 * assumes one output width for every slot.
 *
 * The grid is sized for the wider pair, so the beta slot's upper threadgroups
 * exit on the bounds check.
 */
kernel void kernel_glm53_mul_mv_bf16_f32_trio(
        constant glm53_bf16_trio_args &args,
        device const ushort            *weights_a,
        device const ushort            *weights_b,
        device const ushort            *weights_c,
        device const float             *x,
        device float                   *out_a,
        device float                   *out_b,
        device float                   *out_c,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint slot = tgpig.z;
    device const ushort *w = slot == 0u ? weights_a
                           : (slot == 1u ? weights_b : weights_c);
    device float *out = slot == 0u ? out_a : (slot == 1u ? out_b : out_c);
    const uint out_dim = slot == 2u ? args.out_dim_c : args.out_dim_ab;
    const uint out_row = tgpig.x * (uint)nsg + sg;
    const uint token = tgpig.y;
    if (out_row >= out_dim || token >= args.n_rows) return;
    const float sum =
        glm53_mul_mv_bf16_f32_row_sum(args.in_dim, w, x, out_row, token, lane);
    if (lane == 0u) out[(ulong)token * out_dim + out_row] = sum;
}

/* Flatten the six independent KDA projections into one grid, avoiding empty
 * groups for the three short outputs. Each row retains the existing dot. */
kernel void kernel_glm53_kda_inputs_bf16(
        device const ushort *weights_q,
        device const ushort *weights_k,
        device const ushort *weights_v,
        device const ushort *weights_f,
        device const ushort *weights_g,
        device const ushort *weights_beta,
        device const float *x,
        device float *out_q,
        device float *out_k,
        device float *out_v,
        device float *out_f,
        device float *out_g,
        device float *out_beta,
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    uint row = group * 4u + sg;
    if (row >= 24896u) return;
    device const ushort *weights;
    device float *out;
    if (row < 8192u) {
        weights = weights_q; out = out_q;
    } else if (row < 16384u) {
        row -= 8192u; weights = weights_k; out = out_k;
    } else if (row < 24576u) {
        row -= 16384u; weights = weights_v; out = out_v;
    } else if (row < 24704u) {
        row -= 24576u; weights = weights_f; out = out_f;
    } else if (row < 24832u) {
        row -= 24704u; weights = weights_g; out = out_g;
    } else {
        row -= 24832u; weights = weights_beta; out = out_beta;
    }
    const float sum = glm53_mul_mv_bf16_f32_row_sum(4096u, weights, x, row, 0u, lane);
    if (lane == 0u) out[row] = sum;
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

/*
 * Two independent matvecs of the same shape in one dispatch, selected by
 * tgpig.z, exactly as the qkv variant above selects three.  The inputs are
 * separate pointers rather than one shared row, which lets this serve both
 * halves of the GLM 5.3 KDA gate chain: f_a/g_a read the same attn_norm row,
 * while f_b/g_b read the two different low-rank vectors those produce.
 */
kernel void kernel_glm53_mul_mv_bf16_f32_pair(
        constant glm53_bf16_matmul_args &args,
        device const ushort             *weights_a,
        device const ushort             *weights_b,
        device const float              *x_a,
        device const float              *x_b,
        device float                    *out_a,
        device float                    *out_b,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    device const ushort *weights = tgpig.z == 0u ? weights_a : weights_b;
    device const float  *x       = tgpig.z == 0u ? x_a : x_b;
    device float        *out     = tgpig.z == 0u ? out_a : out_b;
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
