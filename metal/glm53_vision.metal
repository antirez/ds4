// GLM-5.3 Flash vision operations not covered by the shared BF16 matmuls.

struct glm53_vision_rows_args {
    uint width;
    uint rows;
    float eps;
};

struct glm53_vision_qkv_args {
    uint rows;
    uint grid_h;
    uint grid_w;
    float eps;
};

struct glm53_vision_attention_args {
    uint rows;
    float scale;
};

struct glm53_vision_scatter_args {
    uint dst_row;
    uint image_row;
    uint rows;
    uint total_rows;
    uint width;
    uint hc;
};

static inline float glm53_vision_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * exp(-a * a));
}

kernel void kernel_glm53_vision_add_bias(
        constant glm53_vision_rows_args &args,
        device float                    *x,
        device const ushort             *bias,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    x[(ulong)gid.y * args.width + gid.x] += glm53_bf16_to_f32(bias[gid.x]);
}

kernel void kernel_glm53_vision_rms_bf16(
        constant glm53_vision_rows_args &args,
        device const float              *x,
        device const ushort             *weight,
        device float                    *out,
        threadgroup float               *partial,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    device const float *xr = x + (ulong)row * args.width;
    device float *yr = out + (ulong)row * args.width;
    float sum = 0.0f;
    for (uint d = tid; d < args.width; d += 256u) sum = fma(xr[d], xr[d], sum);
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = lane < nsg ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = rsqrt(v / (float)args.width + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    for (uint d = tid; d < args.width; d += 256u) {
        yr[d] = xr[d] * inv * glm53_bf16_to_f32(weight[d]);
    }
}

kernel void kernel_glm53_vision_qkv_rope(
        constant glm53_vision_qkv_args &args,
        device const float             *qkv,
        device const ushort            *bias,
        device const ushort            *q_weight,
        device const ushort            *k_weight,
        device float                   *q,
        device float                   *k,
        device float                   *v,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong qkv_base = (ulong)row * 3072u + (ulong)head * 64u;
    const ulong out_base = (ulong)row * 1024u + (ulong)head * 64u;
    float q0 = qkv[qkv_base + lane] + glm53_bf16_to_f32(bias[(ulong)head * 64u + lane]);
    float q1 = qkv[qkv_base + lane + 32u] +
               glm53_bf16_to_f32(bias[(ulong)head * 64u + lane + 32u]);
    float k0 = qkv[qkv_base + 1024u + lane] +
               glm53_bf16_to_f32(bias[1024u + (ulong)head * 64u + lane]);
    float k1 = qkv[qkv_base + 1024u + lane + 32u] +
               glm53_bf16_to_f32(bias[1024u + (ulong)head * 64u + lane + 32u]);
    const float qsum = simd_sum(fma(q0, q0, q1 * q1));
    const float ksum = simd_sum(fma(k0, k0, k1 * k1));
    const float qinv = rsqrt(qsum / 64.0f + args.eps);
    const float kinv = rsqrt(ksum / 64.0f + args.eps);
    q0 *= qinv * glm53_bf16_to_f32(q_weight[lane]);
    q1 *= qinv * glm53_bf16_to_f32(q_weight[lane + 32u]);
    k0 *= kinv * glm53_bf16_to_f32(k_weight[lane]);
    k1 *= kinv * glm53_bf16_to_f32(k_weight[lane + 32u]);

    const uint merge_w = args.grid_w / 2u;
    const uint group_index = row / 4u;
    const uint within = row & 3u;
    const uint py = (group_index / merge_w) * 2u + within / 2u;
    const uint px = (group_index % merge_w) * 2u + within % 2u;
    const uint freq_index = lane & 15u;
    const uint pos = lane < 16u ? py : px;
    const float inv_freq = powr(10000.0f, -(float)freq_index / 16.0f);
    const float angle = (float)pos * inv_freq;
    const float cs = cos(angle);
    const float sn = sin(angle);
    q[out_base + lane] = q0 * cs - q1 * sn;
    q[out_base + lane + 32u] = q1 * cs + q0 * sn;
    k[out_base + lane] = k0 * cs - k1 * sn;
    k[out_base + lane + 32u] = k1 * cs + k0 * sn;
    v[out_base + lane] = qkv[qkv_base + 2048u + lane] +
                         glm53_bf16_to_f32(bias[2048u + (ulong)head * 64u + lane]);
    v[out_base + lane + 32u] = qkv[qkv_base + 2048u + lane + 32u] +
                               glm53_bf16_to_f32(bias[2048u + (ulong)head * 64u + lane + 32u]);
}

/* A simdgroup owns one query/head and keeps its 64 output values in registers.
 * This is quadratic in compute, as the model graph requires, but linear in
 * memory and never materializes the attention matrix. */
kernel void kernel_glm53_vision_attention(
        constant glm53_vision_attention_args &args,
        device const float                   *q,
        device const float                   *k,
        device const float                   *v,
        device float                         *out,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong base = (ulong)row * 1024u + (ulong)head * 64u;
    const float q0 = q[base + lane];
    const float q1 = q[base + lane + 32u];
    float acc0 = 0.0f, acc1 = 0.0f;
    float max_score = -INFINITY;
    float denom = 0.0f;
    for (uint key_row = 0; key_row < args.rows; key_row++) {
        const ulong kb = (ulong)key_row * 1024u + (ulong)head * 64u;
        float score = simd_sum(q0 * k[kb + lane] + q1 * k[kb + lane + 32u]);
        score *= args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ? 0.0f : exp(max_score - next_max);
        const float new_scale = exp(score - next_max);
        denom = denom * old_scale + new_scale;
        acc0 = acc0 * old_scale + new_scale * v[kb + lane];
        acc1 = acc1 * old_scale + new_scale * v[kb + lane + 32u];
        max_score = next_max;
    }
    out[base + lane] = acc0 / denom;
    out[base + lane + 32u] = acc1 / denom;
}

kernel void kernel_glm53_vision_bias_residual(
        constant glm53_vision_rows_args &args,
        device float                    *x,
        device const ushort             *bias,
        device const float              *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] += glm53_bf16_to_f32(bias[gid.x]) + residual[off];
}

kernel void kernel_glm53_vision_swiglu_bias(
        constant glm53_vision_rows_args &args,
        device const float              *gate,
        device const ushort             *gate_bias,
        device const float              *up,
        device const ushort             *up_bias,
        device float                    *out,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    const float g = min(gate[off] + glm53_bf16_to_f32(gate_bias[gid.x]), 10.0f);
    const float u = clamp(up[off] + glm53_bf16_to_f32(up_bias[gid.x]), -10.0f, 10.0f);
    out[off] = (g / (1.0f + exp(-g))) * u;
}

kernel void kernel_glm53_vision_downsample_reorder(
        device const float *x,
        device float       *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= 4096u) return;
    const uint channel = d / 4u;
    const uint within = d & 3u;
    out[(ulong)row * 4096u + d] = x[((ulong)row * 4u + within) * 1024u + channel];
}

kernel void kernel_glm53_vision_layernorm_gelu(
        constant glm53_vision_rows_args &args,
        device const float              *x,
        device const ushort             *weight,
        device const ushort             *bias,
        device float                    *out,
        threadgroup float               *partial,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    device const float *xr = x + (ulong)row * args.width;
    device float *yr = out + (ulong)row * args.width;
    float sum = 0.0f;
    for (uint d = tid; d < args.width; d += 256u) sum += xr[d];
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = lane < nsg ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = v / (float)args.width;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0];
    float var = 0.0f;
    for (uint d = tid; d < args.width; d += 256u) {
        const float centered = xr[d] - mean;
        var = fma(centered, centered, var);
    }
    var = simd_sum(var);
    if (lane == 0u) partial[sg] = var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = lane < nsg ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = rsqrt(v / (float)args.width + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    const float inv_sqrt2 = 0.7071067811865475f;
    for (uint d = tid; d < args.width; d += 256u) {
        float v = (xr[d] - mean) * inv * glm53_bf16_to_f32(weight[d]) +
                  glm53_bf16_to_f32(bias[d]);
        yr[d] = 0.5f * v * (1.0f + glm53_vision_erf(v * inv_sqrt2));
    }
}

kernel void kernel_glm53_vision_scatter_hc(
        constant glm53_vision_scatter_args &args,
        device float                       *hc,
        device const float                 *image,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint linear_row = gid.y;
    if (d >= args.width || linear_row >= args.rows * args.hc) return;
    const uint image_delta = linear_row / args.hc;
    const uint hc_index = linear_row % args.hc;
    const ulong dst = ((ulong)(args.dst_row + image_delta) * args.hc + hc_index) *
                      args.width + d;
    const ulong src = (ulong)(args.image_row + image_delta) * args.width + d;
    if (args.dst_row + image_delta < args.total_rows) hc[dst] = image[src];
}
