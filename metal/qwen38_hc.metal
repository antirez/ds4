// Qwen 3.8 lowrank gated-residual (hyper-connection) and PLE kernels.
//
// The gated residual around each sub-block computes, over 4 streams of
// 2560 (hc_hidden = 10240):
//   normed = grouped_rmsnorm(h)                      (weights carry the +1)
//   mix    = sigmoid(up(silu(down(normed) / 4)))     (matmuls done outside)
//   mixed  = mean_s(mix[s] * normed[s])              -> sub-block input
//   inject = 2 * sigmoid(inject_proj(normed) / 4)    (matmul done outside)
//   h      = h + y (x) inject                        (y: sub-block output)
// The PLE layer gates a shared value vector per stream with a signed-sqrt
// scaled dot of per-stream keys and queries, then adds a dilated depthwise
// conv (kernel 4, dilation 3) of the normed gated value.

struct qwen38_hc_args {
    uint n_streams;    /* 4 */
    uint n_embd;       /* 2560 */
    uint n_rows;
    float norm_eps;
};

/*
 * Grouped RMSNorm: each (row, stream) group of n_embd values is normalized
 * independently. One threadgroup per (row, stream), 256 threads; each
 * thread owns n_embd/256 elements.
 */
kernel void kernel_qwen38_hc_group_norm(
        constant qwen38_hc_args &args,
        device const float   *x,
        device const float   *weight,
        device float         *out,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint stream = tgpig.y;
    if (row >= args.n_rows || stream >= args.n_streams) return;
    const uint per_thread = args.n_embd / 256u;
    const ulong base = ((ulong)row * args.n_streams + stream) * args.n_embd;
    const uint w_base = stream * args.n_embd;

    float sumsq = 0.0f;
    for (uint i = 0; i < per_thread; i++) {
        const float v = x[base + tid * per_thread + i];
        sumsq = fma(v, v, sumsq);
    }
    sumsq = simd_sum(sumsq);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 8u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)args.n_embd + args.norm_eps);
    for (uint i = 0; i < per_thread; i++) {
        const uint d = tid * per_thread + i;
        out[base + d] = x[base + d] * scale * weight[w_base + d];
    }
}

/* x = silu(x / n_streams), elementwise over n_rows * width values. */
kernel void kernel_qwen38_hc_mix_silu(
        constant qwen38_hc_args &args,
        device float         *x,
        constant uint        &width,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_rows * width) return;
    const float v = x[gid] / (float)args.n_streams;
    x[gid] = v / (1.0f + exp(-v));
}

/*
 * mixed[row, d] = mean over streams of sigmoid(mix[row, s, d]) *
 * normed[row, s, d]. Grid-stride over rows * n_embd.
 */
kernel void kernel_qwen38_hc_combine(
        constant qwen38_hc_args &args,
        device const float   *mix,
        device const float   *normed,
        device float         *mixed,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_rows * args.n_embd) return;
    const uint row = gid / args.n_embd;
    const uint d = gid % args.n_embd;
    const ulong base = (ulong)row * args.n_streams * args.n_embd + d;
    float acc = 0.0f;
    for (uint s = 0; s < args.n_streams; s++) {
        const ulong index = base + (ulong)s * args.n_embd;
        acc += normed[index] / (1.0f + exp(-mix[index]));
    }
    mixed[gid] = acc / (float)args.n_streams;
}

/*
 * h[row, s, d] += y[row, d] * 2 * sigmoid(inject[row, s] / n_streams).
 * Grid-stride over rows * n_streams * n_embd.
 */
kernel void kernel_qwen38_hc_inject(
        constant qwen38_hc_args &args,
        device const float   *y,
        device const float   *inject,
        device float         *h,
        uint gid [[thread_position_in_grid]]) {
    const uint stream_width = args.n_streams * args.n_embd;
    if (gid >= args.n_rows * stream_width) return;
    const uint row = gid / stream_width;
    const uint s = (gid % stream_width) / args.n_embd;
    const uint d = gid % args.n_embd;
    const float raw = inject[(ulong)row * args.n_streams + s] /
                      (float)args.n_streams;
    const float w = 2.0f / (1.0f + exp(-raw));
    h[gid] += y[(ulong)row * args.n_embd + d] * w;
}

/*
 * PLE stream gate: for each (row, stream), gate = signed-sqrt of
 * (key_normed[s] . query_normed[s]) / sqrt(n_embd), clamped at 1e-6 in
 * magnitude before the sqrt; gated[row, s, d] = sigmoid(gate) * value[row, d].
 * One threadgroup per (row, stream), 256 threads.
 */
kernel void kernel_qwen38_ple_gate(
        constant qwen38_hc_args &args,
        device const float   *key_normed,
        device const float   *query_normed,
        device const float   *value,
        device float         *gated,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint stream = tgpig.y;
    if (row >= args.n_rows || stream >= args.n_streams) return;
    const uint per_thread = args.n_embd / 256u;
    const ulong base = ((ulong)row * args.n_streams + stream) * args.n_embd;

    float dot = 0.0f;
    for (uint i = 0; i < per_thread; i++) {
        const uint d = tid * per_thread + i;
        dot = fma(key_normed[base + d], query_normed[base + d], dot);
    }
    dot = simd_sum(dot);
    if (lane == 0u) partial[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 8u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    total /= sqrt((float)args.n_embd);
    const float magnitude = sqrt(max(fabs(total), 1.0e-6f));
    const float gate = total < 0.0f ? -magnitude : magnitude;
    const float g = 1.0f / (1.0f + exp(-gate));
    for (uint i = 0; i < per_thread; i++) {
        const uint d = tid * per_thread + i;
        gated[base + d] = g * value[(ulong)row * args.n_embd + d];
    }
}

static bool qwen38_router_better(threadgroup const float *scores,
                                 int32_t a, int32_t b) {
    const float sa = scores[a];
    const float sb = scores[b];
    if (sa != sb) return sa > sb;
    return a < b;
}

/*
 * Qwen 3.8 router: softmax probabilities renormalized over the selected
 * top-k (selection order equals logit order, so the sort works on raw
 * logits and only the selected entries are exponentiated). One threadgroup
 * per token, sort_width threads.
 */
kernel void kernel_qwen38_router_select(
        constant qwen38_hc_args &args,
        device const float   *logits,
        device int32_t       *selected,
        device float         *weights,
        constant uint        &n_expert,
        constant uint        &n_expert_used,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint sort_width = n_expert > 256u ? 512u : 256u;
    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + sort_width);
    if (token >= args.n_rows) return;
    device const float *token_logits = logits + (uint64_t)token * n_expert;
    device int32_t *token_selected = selected + (uint64_t)token * n_expert_used;
    device float *token_weights = weights + (uint64_t)token * n_expert_used;

    const uint limit = min(n_expert, 512u);
    const bool active = tid < limit;
    sel_scores[tid] = active ? token_logits[tid] : -INFINITY;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= sort_width; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid) {
                const int32_t a = idx[tid];
                const int32_t b = idx[other];
                const bool descending = (tid & k) == 0;
                const bool swap = descending
                    ? qwen38_router_better(sel_scores, b, a)
                    : qwen38_router_better(sel_scores, a, b);
                if (swap) {
                    idx[tid] = b;
                    idx[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint k_used = min(n_expert_used, limit);
    if (tid < k_used) {
        token_selected[tid] = idx[tid];
        /* The top logit sits at idx[0] after the sort; renormalizing over
         * the selected set cancels the full softmax denominator. */
        const float top = sel_scores[idx[0]];
        float sum = 0.0f;
        for (uint i = 0; i < k_used; i++) {
            sum += exp(sel_scores[idx[i]] - top);
        }
        token_weights[tid] = exp(sel_scores[idx[tid]] - top) / sum;
    }
}

/*
 * PLE dilated depthwise conv (kernel 4, dilation 3) + SiLU, walking rows
 * sequentially with a rolling 9-slot state of the *normed* gated values,
 * added on top of the raw gated values:
 *   out[row, c] = gated[row, c] + silu(sum_w conv[c][w] * hist(w))
 * where hist taps are at time offsets -9, -6, -3, 0. One threadgroup per
 * channel block of 256; state layout [9][hc_hidden].
 */
kernel void kernel_qwen38_ple_conv(
        constant qwen38_hc_args &args,
        device const float   *gated_normed,
        device const float   *conv_weight,
        device float         *state,
        device float         *gated_out,
        uint block [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint TAPS = 4u;
    constexpr uint DILATION = 3u;
    constexpr uint STATE = (TAPS - 1u) * DILATION; /* 9 */
    const uint hc_hidden = args.n_streams * args.n_embd;
    const uint channel = block * 256u + tid;
    if (channel >= hc_hidden) return;

    for (uint row = 0; row < args.n_rows; row++) {
        const ulong index = (ulong)row * hc_hidden + channel;
        const float current = gated_normed[index];
        float acc = current * conv_weight[(ulong)channel * TAPS + (TAPS - 1u)];
        for (uint w = 0; w < TAPS - 1u; w++) {
            /* Tap w reads the value DILATION*(TAPS-1-w) steps back: state
             * slot 0 is the oldest. */
            const uint slot = w * DILATION;
            acc = fma(state[(ulong)slot * hc_hidden + channel],
                      conv_weight[(ulong)channel * TAPS + w], acc);
        }
        /* Shift the rolling state by one row and append the current value. */
        for (uint s = 0; s + 1u < STATE; s++) {
            state[(ulong)s * hc_hidden + channel] =
                state[(ulong)(s + 1u) * hc_hidden + channel];
        }
        state[(ulong)(STATE - 1u) * hc_hidden + channel] = current;
        gated_out[index] += acc / (1.0f + exp(-acc));
    }
}

/*
 * out[row, d] = a[row, d] + b[row, d] * sigmoid(s[row]); used to add the
 * scalar-gated shared expert onto the routed MoE output.
 */
kernel void kernel_qwen38_add_sigmoid_rows(
        constant qwen38_hc_args &args,
        device float         *out,
        device const float   *b,
        device const float   *s,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_rows * args.n_embd) return;
    const uint row = gid / args.n_embd;
    const float g = 1.0f / (1.0f + exp(-s[row]));
    out[gid] += b[gid] * g;
}
