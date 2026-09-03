// DeepSeek V4 Flash Vision-Exp operations not covered by shared BF16 matmuls.

struct deepseek4_vision_rows_args {
    uint width;
    uint rows;
};

struct deepseek4_vision_qkv_args {
    uint rows;
    uint grid_width;
};

struct deepseek4_vision_align_args {
    uint grid_height;
    uint grid_width;
    uint output_rows;
};

static inline float deepseek4_vision_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x00007fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float deepseek4_vision_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * exp(-a * a));
}

kernel void kernel_deepseek4_vision_qkv_rope(
        constant deepseek4_vision_qkv_args &args,
        device const float                 *qkv,
        device const ushort                *bias,
        device float                       *q,
        device float                       *k,
        device float                       *v,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong qkv_base = (ulong)row * 3072u + (ulong)head * 64u;
    const ulong out_base = (ulong)row * 1024u + (ulong)head * 64u;
    const float q0 = deepseek4_vision_round_bf16(
        qkv[qkv_base + lane] +
        glm53_bf16_to_f32(bias[(ulong)head * 64u + lane]));
    const float q1 = deepseek4_vision_round_bf16(
        qkv[qkv_base + lane + 32u] +
        glm53_bf16_to_f32(bias[(ulong)head * 64u + lane + 32u]));
    const float k0 = deepseek4_vision_round_bf16(
        qkv[qkv_base + 1024u + lane] +
        glm53_bf16_to_f32(bias[1024u + (ulong)head * 64u + lane]));
    const float k1 = deepseek4_vision_round_bf16(
        qkv[qkv_base + 1024u + lane + 32u] +
        glm53_bf16_to_f32(bias[1024u + (ulong)head * 64u + lane + 32u]));
    const uint y = row / args.grid_width;
    const uint x = row - y * args.grid_width;
    const uint pos = lane < 16u ? y : x;
    const uint freq_index = lane & 15u;
    const float inv_freq = powr(10000.0f, -(float)freq_index / 16.0f);
    const float angle = (float)pos * inv_freq;
    const float cs = cos(angle);
    const float sn = sin(angle);
    q[out_base + lane] = deepseek4_vision_round_bf16(q0 * cs - q1 * sn);
    q[out_base + lane + 32u] =
        deepseek4_vision_round_bf16(q1 * cs + q0 * sn);
    k[out_base + lane] = deepseek4_vision_round_bf16(k0 * cs - k1 * sn);
    k[out_base + lane + 32u] =
        deepseek4_vision_round_bf16(k1 * cs + k0 * sn);
    v[out_base + lane] = deepseek4_vision_round_bf16(
        qkv[qkv_base + 2048u + lane] +
        glm53_bf16_to_f32(bias[2048u + (ulong)head * 64u + lane]));
    v[out_base + lane + 32u] = deepseek4_vision_round_bf16(
        qkv[qkv_base + 2048u + lane + 32u] +
        glm53_bf16_to_f32(
            bias[2048u + (ulong)head * 64u + lane + 32u]));
}

kernel void kernel_deepseek4_vision_round_bf16(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = deepseek4_vision_round_bf16(x[off]);
}

kernel void kernel_deepseek4_vision_add_residual(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        device const float                 *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = deepseek4_vision_round_bf16(x[off] + residual[off]);
}

kernel void kernel_deepseek4_vision_swiglu_split(
        constant deepseek4_vision_rows_args &args,
        device const float                 *gate_up,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong source = (ulong)gid.y * args.width * 2u + gid.x;
    const float gate = gate_up[source];
    const float up = gate_up[source + args.width];
    out[(ulong)gid.y * args.width + gid.x] = deepseek4_vision_round_bf16(
        (gate / (1.0f + exp(-gate))) * up);
}

/* Match F.unfold(x, 3, stride=3): channels are outermost, followed by the
 * row-major 3x3 neighborhood. Values outside the ViT grid are zero. */
kernel void kernel_deepseek4_vision_aligner_reorder(
        constant deepseek4_vision_align_args &args,
        device const float                  *x,
        device float                        *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= 9216u || row >= args.output_rows) return;
    const uint output_width = (args.grid_width + 2u) / 3u;
    const uint block_y = row / output_width;
    const uint block_x = row - block_y * output_width;
    const uint channel = d / 9u;
    const uint within = d - channel * 9u;
    const uint source_y = block_y * 3u + within / 3u;
    const uint source_x = block_x * 3u + within % 3u;
    float value = 0.0f;
    if (source_y < args.grid_height && source_x < args.grid_width) {
        const ulong source_row = (ulong)source_y * args.grid_width + source_x;
        value = x[source_row * 1024u + channel];
    }
    out[(ulong)row * 9216u + d] = value;
}

kernel void kernel_deepseek4_vision_gelu_bias(
        constant deepseek4_vision_rows_args &args,
        device const float                 *x,
        device const ushort                *bias,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    const float value = deepseek4_vision_round_bf16(
        x[off] + glm53_bf16_to_f32(bias[gid.x]));
    out[off] = deepseek4_vision_round_bf16(
        0.5f * value *
        (1.0f + deepseek4_vision_erf(value * 0.7071067811865475f)));
}
