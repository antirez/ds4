// Kimi Delta Attention kernels, adapted from the kimi-k3 branch.

struct glm53_kda_args {
    uint n_heads;
    uint n_rows;
    float lower_bound;
    float norm_eps;
};

/*
 * One threadgroup owns one (sequence, head). Four SIMDgroups own disjoint
 * value rows; every lane owns four adjacent key columns.
 */
template<uint VALUES>
kernel void kernel_glm53_kda_decode_values(
        constant glm53_kda_args &args,
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
    threadgroup float *a_decay_shared = beta_shared + 1u;

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
    }
    if (tid == 0u) {
        beta_shared[0] =
            1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
        /* head is uniform over the threadgroup, so exp(a_log[head]) is a
         * single value; every one of the D channels used to recompute it. */
        a_decay_shared[0] = exp(a_log[head]);
    }
    /* Only threadgroup memory is shared between threads here: the conv-state
     * writes above are each thread's own channel and no thread reads another's,
     * so the barrier does not need device scope. */
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        sd[tid] = exp(args.lower_bound *
                      (1.0f / (1.0f + exp(-a_decay_shared[0] * gate))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head =
        ((ulong)row * args.n_heads + head) * D * D;

    /* Interleave independent value rows to hide load/reduction latency.
     * Every row retains its original SIMDgroup, dot/FMA sequence, and stores;
     * normalization still uses the same four-SIMDgroup reduction tree. */
    if (VALUES == 1u) {
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
    } else {
        for (uint value = sg; value < D; value += 4u * VALUES) {
            device float4 *hptr[VALUES];
            float4 h[VALUES];
            float hk[VALUES];
            FOR_UNROLL (uint i = 0; i < VALUES; i++) {
                hptr[i] = (device float4 *)(state + state_head + (ulong)(value + i * 4u) * D + k0);
                h[i] = *hptr[i] * decay4;
            }
            FOR_UNROLL (uint i = 0; i < VALUES; i++) {
                hk[i] = dot(h[i], k4);
                hk[i] = simd_sum(hk[i]);
            }
            FOR_UNROLL (uint i = 0; i < VALUES; i++) {
                const float delta_v = (sv[value + i * 4u] - hk[i]) * beta_shared[0];
                h[i] = fma(k4, float4(delta_v), h[i]);
                *hptr[i] = h[i];
            }
            FOR_UNROLL (uint i = 0; i < VALUES; i++) {
                const float hq = simd_sum(dot(h[i], q4));
                if (lane == 0u) so[value + i * 4u] = hq;
            }
        }
    }
    /* Likewise: the state writes above are not re-read in this kernel, only
     * so[] crosses simdgroups. */
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

typedef decltype(kernel_glm53_kda_decode_values<1u>) glm53_kda_decode_values_t;
template [[host_name("kernel_glm53_kda_decode")]]
kernel glm53_kda_decode_values_t kernel_glm53_kda_decode_values<1u>;
template [[host_name("kernel_glm53_kda_decode_v4")]]
kernel glm53_kda_decode_values_t kernel_glm53_kda_decode_values<4u>;

kernel void kernel_glm53_kda_prefill_prepare(
        constant glm53_kda_args &args,
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

/*
 * Boundary rows for the blocked prepare kernel below.
 *
 * A block starts its causal convolution window on the raw q/k/v of the three
 * rows before it, and the block before it overwrites exactly those rows with
 * its normalized outputs. Block 0 also needs an immutable copy of the incoming
 * conv state: the last block may overwrite that state before block 0 starts.
 */
struct glm53_kda_blocked_args {
    uint n_heads;
    uint n_rows;
    uint block_rows;
    uint n_blocks;
    uint block_base;
    float lower_bound;
    float norm_eps;
};

kernel void kernel_glm53_kda_prefill_conv_halo(
        constant glm53_kda_blocked_args &args,
        device const float *q,
        device const float *k,
        device const float *v,
        device float       *halo,
        device const float *conv_state,
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    constexpr uint NTH = 256u;   /* matches the dispatch */
    const uint projection = args.n_heads * D;
    const uint block = tgpig.x;
    const uint w = tgpig.y;
    if (block >= args.n_blocks || w >= HISTORY) return;
    const ulong plane = (ulong)args.n_blocks * HISTORY * projection;
    const ulong slot = ((ulong)block * HISTORY + w) * projection;
    for (uint c = tid; c < projection; c += NTH) {
        if (block == 0u) {
            const ulong index = (ulong)w * projection + c;
            halo[slot + c] = conv_state[index];
            halo[plane + slot + c] = conv_state[HISTORY * projection + index];
            halo[2u * plane + slot + c] = conv_state[2u * HISTORY * projection + index];
        } else {
            const uint token = block * args.block_rows + w - HISTORY;
            const ulong index = (ulong)token * projection + c;
            halo[slot + c] = q[index];
            halo[plane + slot + c] = k[index];
            halo[2u * plane + slot + c] = v[index];
        }
    }
}

/*
 * Token-parallel form of kernel_glm53_kda_prefill_prepare.
 *
 * The serial kernel runs one threadgroup per head -- 64 of them on an 80-core
 * GPU -- and walks all the chunk's tokens inside it, so it costs 3.4 ms of a
 * 2048-token layer at about 5% occupancy.  Nothing in it is actually
 * sequential: the causal convolution reads the raw q/k/v of t-3..t, which are
 * all known before the kernel starts.
 *
 * One threadgroup now owns (block of block_rows tokens, head) and keeps the
 * three-row convolution history in registers instead of re-reading and
 * re-writing the device conv state every token.  Its first three rows come
 * from the immutable halo above, including the incoming state for block 0.
 * The last block leaves the outgoing conv state exactly where the serial kernel
 * left it.
 *
 * Every value keeps the serial kernel's expression and order: the same
 * four-term fma chain in the same order, the same silu, the same
 * 4-simdgroup RMS reduction, the same decay-gate expression.
 */
kernel void kernel_glm53_kda_prefill_prepare_blocked(
        constant glm53_kda_blocked_args &args,
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
        device const float   *halo,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint block = tgpig.x + args.block_base;
    const uint head = tgpig.y;
    if (block >= args.n_blocks || head >= args.n_heads) return;
    threadgroup float *sq = scratch;
    threadgroup float *sk = sq + D;
    threadgroup float *reduce_q = sk + D;
    threadgroup float *reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const uint token0 = block * args.block_rows;
    const uint token_end = token0 + min(args.block_rows, args.n_rows - token0);
    if (token0 >= token_end) return;

    float hq[HISTORY];
    float hk[HISTORY];
    float hv[HISTORY];
    const ulong plane = (ulong)args.n_blocks * HISTORY * projection;
    const ulong slot = (ulong)block * HISTORY * projection + channel;
    for (uint w = 0; w < HISTORY; w++) {
        hq[w] = halo[slot + w * projection];
        hk[w] = halo[plane + slot + w * projection];
        hv[w] = halo[2u * plane + slot + w * projection];
    }

    for (uint token = token0; token < token_end; token++) {
        const ulong index = (ulong)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(hq[w], q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(hk[w], k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(hv[w], v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);
        hq[0] = hq[1]; hq[1] = hq[2]; hq[2] = q_new;
        hk[0] = hk[1]; hk[1] = hk[2]; hk[2] = k_new;
        hv[0] = hv[1]; hv[1] = hv[2]; hv[2] = v_new;

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

    /* The serial kernel leaves the conv state holding the raw q/k/v of the
     * last three rows it processed, which is what this block's history is
     * once its last token has shifted through. */
    if (token_end == args.n_rows) {
        device float *q_state = conv_state;
        device float *k_state = q_state + HISTORY * projection;
        device float *v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_state[(ulong)w * projection + channel] = hq[w];
            k_state[(ulong)w * projection + channel] = hk[w];
            v_state[(ulong)w * projection + channel] = hv[w];
        }
    }
}

kernel void kernel_glm53_kda_prefill_recurrence(
        constant glm53_kda_args &args,
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

/* Each value row is an independent recurrence. Carrying two or four rows in
 * a SIMDgroup reuses q/k/decay and beta across them, while keeping the serial
 * token order, dot reductions and FMA expression of the reference above. */
template <uint VALUES>
kernel void kernel_glm53_kda_prefill_recurrence_values(
        constant glm53_kda_args &args,
        device const float *q,
        device const float *k,
        device const float *v,
        device const float *decay,
        device const float *raw_beta,
        device float *state,
        device float *out,
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value0 = (tgpig.y * 4u + sg) * VALUES;
    if (head >= args.n_heads || value0 + VALUES > D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    float4 h[VALUES];
    FOR_UNROLL (uint i = 0; i < VALUES; i++) {
        h[i] = *((device float4 *)(state + ((ulong)head * D + value0 + i) * D + k0));
    }
    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 = *((device const float4 *)(decay + base + k0));
        const float beta = 1.0f /
            (1.0f + exp(-raw_beta[(ulong)token * args.n_heads + head]));
        FOR_UNROLL (uint i = 0; i < VALUES; i++) {
            h[i] *= decay4;
            const float hk = simd_sum(dot(h[i], k4));
            const float delta_v = (v[base + value0 + i] - hk) * beta;
            h[i] = fma(k4, float4(delta_v), h[i]);
            const float result = simd_sum(dot(h[i], q4));
            if (lane == 0u) out[base + value0 + i] = result;
        }
    }
    FOR_UNROLL (uint i = 0; i < VALUES; i++) {
        *((device float4 *)(state + ((ulong)head * D + value0 + i) * D + k0)) = h[i];
    }
}

typedef decltype(kernel_glm53_kda_prefill_recurrence_values<2u>) glm53_kda_recurrence_values_t;
template [[host_name("kernel_glm53_kda_prefill_recurrence_v2")]]
kernel glm53_kda_recurrence_values_t kernel_glm53_kda_prefill_recurrence_values<2u>;
template [[host_name("kernel_glm53_kda_prefill_recurrence_v4")]]
kernel glm53_kda_recurrence_values_t kernel_glm53_kda_prefill_recurrence_values<4u>;

kernel void kernel_glm53_kda_prefill_output(
        constant glm53_kda_args &args,
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
