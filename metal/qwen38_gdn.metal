// Qwen 3.8 Gated DeltaNet kernels, derived from the GLM 5.3 KDA kernels.
//
// Differences from KDA: 16 QK heads are shared by 48 value heads (each QK
// head serves qk_ratio = 3 consecutive value heads), the state decay is one
// scalar per value head computed as exp(-exp(A_log) * softplus(a + dt_bias))
// instead of a per-channel vector. The sigmoid output gate, the q/k L2
// normalization (eps 1e-6), the 1/sqrt(128) query scale, the SiLU on the
// conv outputs and the delta-rule update are identical to KDA.

struct qwen38_gdn_args {
    uint n_qk_heads;   /* 16 */
    uint n_v_heads;    /* 48 */
    uint n_rows;
    float norm_eps;
};

/*
 * Phase 1: causal conv + SiLU + q/k L2 norm + per-head decay, walking the
 * tokens sequentially so the rolling conv state stays exact. One threadgroup
 * per QK head group (128 threads); each thread owns one q channel, one k
 * channel and qk_ratio v channels. Overwrites q/k/v in place with their
 * normalized/activated values and replaces raw_alpha with the decay factor.
 */
kernel void kernel_qwen38_gdn_prepare(
        constant qwen38_gdn_args &args,
        device float         *q,
        device float         *k,
        device float         *v,
        device float         *raw_alpha,
        device const float   *q_conv,
        device const float   *k_conv,
        device const float   *v_conv,
        device const float   *a_log,
        device const float   *dt_bias,
        device float         *conv_state,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    if (group >= args.n_qk_heads) return;
    const uint ratio = args.n_v_heads / args.n_qk_heads;
    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *reduce_q = sk + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    const uint qk_projection = args.n_qk_heads * D;
    const uint v_projection = args.n_v_heads * D;
    const uint qk_channel = group * D + tid;
    device float *q_state = conv_state;
    device float *k_state = q_state + HISTORY * qk_projection;
    device float *v_state = k_state + HISTORY * qk_projection;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong qk_index = (ulong)token * qk_projection + qk_channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * qk_projection + qk_channel],
                        q_conv[(ulong)qk_channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * qk_projection + qk_channel],
                        k_conv[(ulong)qk_channel * 4u + w], k_acc);
        }
        const float q_new = q[qk_index];
        const float k_new = k[qk_index];
        q_acc = fma(q_new, q_conv[(ulong)qk_channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)qk_channel * 4u + 3u], k_acc);
        q_state[qk_channel] = q_state[qk_projection + qk_channel];
        q_state[qk_projection + qk_channel] = q_state[2ul * qk_projection + qk_channel];
        q_state[2ul * qk_projection + qk_channel] = q_new;
        k_state[qk_channel] = k_state[qk_projection + qk_channel];
        k_state[qk_projection + qk_channel] = k_state[2ul * qk_projection + qk_channel];
        k_state[2ul * qk_projection + qk_channel] = k_new;

        for (uint i = 0; i < ratio; i++) {
            const uint v_channel = (group * ratio + i) * D + tid;
            const ulong v_index = (ulong)token * v_projection + v_channel;
            float v_acc = 0.0f;
            for (uint w = 0; w < HISTORY; w++) {
                v_acc = fma(v_state[(ulong)w * v_projection + v_channel],
                            v_conv[(ulong)v_channel * 4u + w], v_acc);
            }
            const float v_new = v[v_index];
            v_acc = fma(v_new, v_conv[(ulong)v_channel * 4u + 3u], v_acc);
            v_state[v_channel] = v_state[v_projection + v_channel];
            v_state[v_projection + v_channel] = v_state[2ul * v_projection + v_channel];
            v_state[2ul * v_projection + v_channel] = v_new;
            v[v_index] = v_acc / (1.0f + exp(-v_acc));
        }

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        if (tid < ratio) {
            const uint head = group * ratio + tid;
            const float a_raw = raw_alpha[(ulong)token * args.n_v_heads + head] +
                                dt_bias[head];
            /* softplus with the usual overflow guard */
            const float sp = a_raw > 20.0f ? a_raw : log(1.0f + exp(a_raw));
            raw_alpha[(ulong)token * args.n_v_heads + head] =
                exp(-exp(a_log[head]) * sp);
        }
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
        /* 0x1.6a09e6p-4f = sqrt(2)/16 = 1/sqrt(128): the post-L2 query scale. */
        q[qk_index] = sq[tid] * rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
        k[qk_index] = sk[tid] * rsqrt(k_total + 1.0e-6f);
        threadgroup_barrier(mem_flags::mem_threadgroup |
                           mem_flags::mem_device);
    }
}

/*
 * Phase 2: the gated delta rule. One threadgroup per (value head, block of
 * four value rows); four simdgroups update four value rows concurrently and
 * every lane owns four adjacent key columns. q/k are read from the head's
 * QK group, the decay is the per-head scalar prepared in phase 1.
 */
kernel void kernel_qwen38_gdn_recurrence(
        constant qwen38_gdn_args &args,
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
    if (head >= args.n_v_heads || value >= D) return;
    const uint ratio = args.n_v_heads / args.n_qk_heads;
    const uint qk_projection = args.n_qk_heads * D;
    const uint v_projection = args.n_v_heads * D;
    const uint group = head / ratio;
    const uint k0 = lane * 4u;
    device float4 *state_ptr = (device float4 *)(
        state + ((ulong)head * D + value) * D + k0);
    float4 h = *state_ptr;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong qk_base = (ulong)token * qk_projection + group * D;
        const ulong v_base = (ulong)token * v_projection + head * D;
        const float4 q4 = *((device const float4 *)(q + qk_base + k0));
        const float4 k4 = *((device const float4 *)(k + qk_base + k0));
        h *= decay[(ulong)token * args.n_v_heads + head];
        const float hk = simd_sum(dot(h, k4));
        const float beta = 1.0f /
            (1.0f + exp(-raw_beta[(ulong)token * args.n_v_heads + head]));
        const float delta_v = (v[v_base + value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        const float result = simd_sum(dot(h, q4));
        if (lane == 0u) out[v_base + value] = result;
    }
    *state_ptr = h;
}

/*
 * Phase 3: per-head-dim RMSNorm with a sigmoid output gate
 * (config.output_gate_type; the one numerical difference from Qwen3.5's
 * GDN, whose gate is SiLU). One threadgroup per (token, value head).
 */
kernel void kernel_qwen38_gdn_output(
        constant qwen38_gdn_args &args,
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
    if (token >= args.n_rows || head >= args.n_v_heads) return;
    const uint v_projection = args.n_v_heads * D;
    const ulong base = (ulong)token * v_projection + head * D;
    const float raw = out[base + tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 4u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    const float gate = output_gate[base + tid];
    out[base + tid] = raw * scale * output_norm[tid] /
        (1.0f + exp(-gate));
}
