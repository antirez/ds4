#if !defined(DS4_NO_GPU) && defined(__APPLE__)

struct qwen4_vision_rows_args {
    uint width;
    uint rows;
    float eps;
};

struct qwen4_vision_grid_args {
    uint rows;
    uint grid_h;
    uint grid_w;
};

struct qwen4_vision_attention_args {
    uint rows;
    float scale;
};

struct qwen4_vision_pad_args {
    uint source_width;
    uint padded_width;
    uint rows;
};

kernel void kernel_qwen4_vision_layernorm_bf16(
        constant qwen4_vision_rows_args &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const ushort *weight [[buffer(2)]],
        device const ushort *bias [[buffer(3)]],
        device float *out [[buffer(4)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    const ulong base = (ulong)row * args.width;
    float sum = 0.0f;
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u)
        sum += x[base + d];
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float value = lane < nsg ? partial[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0u) partial[0] = value / (float)args.width;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0];
    float variance = 0.0f;
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        const float centered = x[base + d] - mean;
        variance = fma(centered, centered, variance);
    }
    variance = simd_sum(variance);
    if (lane == 0u) partial[sg] = variance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float value = lane < nsg ? partial[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0u)
            partial[0] = rsqrt(value / (float)args.width + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    for (uint d = tid; d < args.width; d += (uint)nsg * 32u) {
        out[base + d] = (x[base + d] - mean) * inv *
            qwen4_bf16_to_f32(weight[d]) + qwen4_bf16_to_f32(bias[d]);
    }
}

kernel void kernel_qwen4_vision_patch_position(
        constant qwen4_vision_grid_args &args [[buffer(0)]],
        device float *x [[buffer(1)]],
        device const ushort *bias [[buffer(2)]],
        device const ushort *positions [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= 1152u || row >= args.rows) return;
    const uint merge_w = args.grid_w / 2u;
    const uint block = row / 4u;
    const uint within = row & 3u;
    const uint py = (block / merge_w) * 2u + within / 2u;
    const uint px = (block % merge_w) * 2u + within % 2u;
    const float sy = args.grid_h > 1u
        ? (float)py * 47.0f / (float)(args.grid_h - 1u) : 0.0f;
    const float sx = args.grid_w > 1u
        ? (float)px * 47.0f / (float)(args.grid_w - 1u) : 0.0f;
    const uint y0 = min((uint)floor(sy), 47u);
    const uint x0 = min((uint)floor(sx), 47u);
    const uint y1 = min(y0 + 1u, 47u);
    const uint x1 = min(x0 + 1u, 47u);
    const float fy = sy - (float)y0;
    const float fx = sx - (float)x0;
    const float p00 = qwen4_bf16_to_f32(positions[((ulong)y0 * 48u + x0) * 1152u + d]);
    const float p01 = qwen4_bf16_to_f32(positions[((ulong)y0 * 48u + x1) * 1152u + d]);
    const float p10 = qwen4_bf16_to_f32(positions[((ulong)y1 * 48u + x0) * 1152u + d]);
    const float p11 = qwen4_bf16_to_f32(positions[((ulong)y1 * 48u + x1) * 1152u + d]);
    const float top = mix(p00, p01, fx);
    const float bottom = mix(p10, p11, fx);
    x[(ulong)row * 1152u + d] += qwen4_bf16_to_f32(bias[d]) +
        mix(top, bottom, fy);
}

kernel void kernel_qwen4_vision_qkv_rope(
        constant qwen4_vision_grid_args &args [[buffer(0)]],
        device const float *qkv [[buffer(1)]],
        device const ushort *bias [[buffer(2)]],
        device float *q [[buffer(3)]],
        device float *k [[buffer(4)]],
        device float *v [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint linear = gid.y;
    const uint row = linear / 16u;
    const uint head = linear % 16u;
    if (d >= 72u || row >= args.rows) return;
    const ulong row_base = (ulong)row * 3456u;
    const ulong head_base = (ulong)head * 72u;
    const ulong out_base = ((ulong)row * 16u + head) * 72u;
    const uint pair = d < 36u ? d + 36u : d - 36u;
    const uint freq = d < 36u ? d : d - 36u;
    const uint merge_w = args.grid_w / 2u;
    const uint block = row / 4u;
    const uint within = row & 3u;
    const uint py = (block / merge_w) * 2u + within / 2u;
    const uint px = (block % merge_w) * 2u + within % 2u;
    const uint position = freq < 18u ? py : px;
    const uint axis_freq = freq < 18u ? freq : freq - 18u;
    const float angle = (float)position * powr(10000.0f,
        -2.0f * (float)axis_freq / 36.0f);
    const float cs = cos(angle);
    const float sn = sin(angle);
    float q0 = qkv[row_base + head_base + d] +
        qwen4_bf16_to_f32(bias[head_base + d]);
    const float qp = qkv[row_base + head_base + pair] +
        qwen4_bf16_to_f32(bias[head_base + pair]);
    float k0 = qkv[row_base + 1152u + head_base + d] +
        qwen4_bf16_to_f32(bias[1152u + head_base + d]);
    const float kp = qkv[row_base + 1152u + head_base + pair] +
        qwen4_bf16_to_f32(bias[1152u + head_base + pair]);
    q0 = d < 36u ? fma(-qp, sn, q0 * cs) : fma(qp, sn, q0 * cs);
    k0 = d < 36u ? fma(-kp, sn, k0 * cs) : fma(kp, sn, k0 * cs);
    q[out_base + d] = q0;
    k[out_base + d] = k0;
    v[out_base + d] = qkv[row_base + 2304u + head_base + d] +
        qwen4_bf16_to_f32(bias[2304u + head_base + d]);
}

/* One simdgroup owns one query/head.  The 72 channels are striped over the
 * 32 lanes and attention uses online softmax, so no quadratic score matrix
 * is allocated. */
kernel void kernel_qwen4_vision_attention(
        constant qwen4_vision_attention_args &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong qbase = ((ulong)row * 16u + head) * 72u;
    float qv[3] = {0.0f, 0.0f, 0.0f};
    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (uint i = 0; i < 3u; i++) {
        const uint d = (uint)lane + i * 32u;
        if (d < 72u) qv[i] = q[qbase + d];
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint key_row = 0; key_row < args.rows; key_row++) {
        const ulong base = ((ulong)key_row * 16u + head) * 72u;
        float dot = 0.0f;
        for (uint i = 0; i < 3u; i++) {
            const uint d = (uint)lane + i * 32u;
            if (d < 72u) dot = fma(qv[i], k[base + d], dot);
        }
        dot = simd_sum(dot) * args.scale;
        const float next = max(maximum, dot);
        const float old_scale = maximum == -INFINITY ? 0.0f : exp(maximum - next);
        const float new_scale = exp(dot - next);
        denominator = denominator * old_scale + new_scale;
        for (uint i = 0; i < 3u; i++) {
            const uint d = (uint)lane + i * 32u;
            if (d < 72u)
                acc[i] = acc[i] * old_scale + new_scale * v[base + d];
        }
        maximum = next;
    }
    for (uint i = 0; i < 3u; i++) {
        const uint d = (uint)lane + i * 32u;
        if (d < 72u) out[qbase + d] = acc[i] / denominator;
    }
}

kernel void kernel_qwen4_vision_bias_residual(
        constant qwen4_vision_rows_args &args [[buffer(0)]],
        device float *x [[buffer(1)]],
        device const ushort *bias [[buffer(2)]],
        device const float *residual [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong at = (ulong)gid.y * args.width + gid.x;
    x[at] += qwen4_bf16_to_f32(bias[gid.x]) + residual[at];
}

kernel void kernel_qwen4_vision_bias_gelu_tanh(
        constant qwen4_vision_rows_args &args [[buffer(0)]],
        device float *x [[buffer(1)]],
        device const ushort *bias [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong at = (ulong)gid.y * args.width + gid.x;
    const float value = x[at] + qwen4_bf16_to_f32(bias[gid.x]);
    const float inner = 0.7978845608028654f *
        fma(0.044715f * value * value, value, value);
    /* The default fast intrinsic can return NaN for the large negative tail
     * reached by real Qwen FC1 activations.  The precise intrinsic remains
     * finite and matches gelu_pytorch_tanh across that range. */
    x[at] = 0.5f * value * (1.0f + precise::tanh(inner));
}

kernel void kernel_qwen4_vision_pad_rows(
        constant qwen4_vision_pad_args &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device float *out [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.padded_width || gid.y >= args.rows) return;
    const ulong dst = (ulong)gid.y * args.padded_width + gid.x;
    out[dst] = gid.x < args.source_width
        ? x[(ulong)gid.y * args.source_width + gid.x]
        : 0.0f;
}

/* Metal does not expose erf in the runtime compiler used on M3.  This
 * single-precision approximation has a maximum absolute error below 1.5e-7,
 * well below the BF16 sidecar's quantization floor. */
static inline float qwen4_vision_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * exp(-a * a));
}

kernel void kernel_qwen4_vision_bias_gelu_exact(
        constant qwen4_vision_rows_args &args [[buffer(0)]],
        device float *x [[buffer(1)]],
        device const ushort *bias [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong at = (ulong)gid.y * args.width + gid.x;
    const float value = x[at] + qwen4_bf16_to_f32(bias[gid.x]);
    x[at] = 0.5f * value *
        (1.0f + qwen4_vision_erf(value * 0.7071067811865475f));
}

kernel void kernel_qwen4_vision_add_bias(
        constant qwen4_vision_rows_args &args [[buffer(0)]],
        device float *x [[buffer(1)]],
        device const ushort *bias [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    x[(ulong)gid.y * args.width + gid.x] += qwen4_bf16_to_f32(bias[gid.x]);
}

kernel void kernel_qwen4_vision_merger_norm_pack(
        constant qwen4_vision_rows_args &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const ushort *weight [[buffer(2)]],
        device const ushort *bias [[buffer(3)]],
        device float *out [[buffer(4)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    const ulong source = (ulong)row * 1152u;
    const ulong target = (ulong)(row / 4u) * 4608u +
                         (ulong)(row & 3u) * 1152u;
    float sum = 0.0f;
    for (uint d = tid; d < 1152u; d += (uint)nsg * 32u)
        sum += x[source + d];
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float value = lane < nsg ? partial[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0u) partial[0] = value / 1152.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0];
    float variance = 0.0f;
    for (uint d = tid; d < 1152u; d += (uint)nsg * 32u) {
        const float centered = x[source + d] - mean;
        variance = fma(centered, centered, variance);
    }
    variance = simd_sum(variance);
    if (lane == 0u) partial[sg] = variance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float value = lane < nsg ? partial[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0u) partial[0] = rsqrt(value / 1152.0f + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    for (uint d = tid; d < 1152u; d += (uint)nsg * 32u) {
        out[target + d] = (x[source + d] - mean) * inv *
            qwen4_bf16_to_f32(weight[d]) + qwen4_bf16_to_f32(bias[d]);
    }
}

#endif
