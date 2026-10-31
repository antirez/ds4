/* Muse Spark Open helpers.  Activations are f32; K/V cache rows are f16.
 * The attention and expert matmuls reuse the Qwen4 kernels, while these
 * kernels implement Muse-specific normalization, gating and routing. */

static inline float muse_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

struct ds4_metal_args_muse_attn_prep {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t use_rope;
    float rope_base;
    float eps;
    float q_scale;
    uint32_t pad0;
};

/* Gate is not read here. Sigmoid consumes the gate projection directly, so
 * this kernel has no hazard on it. The old body stored gateproj into
 * gate_out with a plain copy. */
kernel void kernel_muse_attn_prep(
        constant ds4_metal_args_muse_attn_prep &args,
        device const float *qproj,
        device const float *kproj,
        device const float *vproj,
        device const float *q_norm,
        device const float *k_norm,
        device float *q_out,
        device half *k_cache,
        device half *v_cache,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.x;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint D = args.head_dim;
    const uint npt = D / 32u;
    const uint pos = args.pos0 + tok;
    threadgroup float row[64];

    if (slot < args.n_head) {
        device const float *src = qproj + ((uint64_t)tok * args.n_head + slot) * D;
        float ss = 0.0f;
        for (uint i = 0; i < npt; i++) {
            const float v = src[lane * npt + i];
            row[lane * npt + i] = v;
            ss += v * v;
        }
        ss = simd_sum(ss);
        const float r = rsqrt(ss / (float)D + args.eps);
        for (uint i = 0; i < npt; i++) row[lane * npt + i] *= r * q_norm[lane * npt + i] * args.q_scale;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (args.use_rope && lane < args.n_rot / 2u) {
            const uint i = 2u * lane;
            const uint j = i + 1u;
            const float theta = (float)pos * pow(args.rope_base, -2.0f * (float)lane / (float)args.n_rot);
            const float c = cos(theta), s = sin(theta);
            const float x0 = row[i], x1 = row[j];
            row[i] = x0 * c - x1 * s;
            row[j] = x0 * s + x1 * c;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device float *dst = q_out + ((uint64_t)tok * args.n_head + slot) * D;
        for (uint i = 0; i < npt; i++) {
            const uint j = lane * npt + i;
            dst[j] = row[j];
        }
        return;
    }

    const uint h = slot - args.n_head;
    if (h >= args.n_head_kv || pos >= args.cache_cap) return;
    device const float *src = kproj + ((uint64_t)tok * args.n_head_kv + h) * D;
    float ss = 0.0f;
    for (uint i = 0; i < npt; i++) {
        const float v = src[lane * npt + i];
        row[lane * npt + i] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float r = rsqrt(ss / (float)D + args.eps);
    for (uint i = 0; i < npt; i++) row[lane * npt + i] *= r * k_norm[lane * npt + i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (args.use_rope && lane < args.n_rot / 2u) {
        const uint i = 2u * lane;
        const uint j = i + 1u;
        const float theta = (float)pos * pow(args.rope_base, -2.0f * (float)lane / (float)args.n_rot);
        const float c = cos(theta), s = sin(theta);
        const float x0 = row[i], x1 = row[j];
        row[i] = x0 * c - x1 * s;
        row[j] = x0 * s + x1 * c;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device half *kd = k_cache + ((uint64_t)pos * args.n_head_kv + h) * D;
    device half *vd = v_cache + ((uint64_t)pos * args.n_head_kv + h) * D;
    device const float *vs = vproj + ((uint64_t)tok * args.n_head_kv + h) * D;
    for (uint i = 0; i < npt; i++) {
        const uint j = lane * npt + i;
        kd[j] = (half)row[j];
        vd[j] = (half)vs[j];
    }
}

struct ds4_metal_args_muse_rows {
    uint32_t n_tokens;
    uint32_t dim;
    float alpha;
    float beta;
};

kernel void kernel_muse_sigmoid_mul(
        constant ds4_metal_args_muse_rows &args,
        device const float *x,
        device const float *gate,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.n_tokens * args.dim;
    if ((uint64_t)gid < n) out[gid] = x[gid] * muse_sigmoid(gate[gid]);
}

kernel void kernel_muse_residual_blend(
        constant ds4_metal_args_muse_rows &args,
        device const float *a,
        device const float *b,
        device const float *alpha,
        device const float *beta,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.n_tokens * args.dim;
    if ((uint64_t)gid < n) {
        const uint col = gid % args.dim;
        out[gid] = a[gid] * alpha[col] + b[gid] * beta[col];
    }
}

kernel void kernel_muse_weight_mul(
        constant ds4_metal_args_muse_rows &args,
        device const float *x,
        device const float *weight,
        device float *out,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.n_tokens * args.dim;
    if ((uint64_t)gid < n) out[gid] = x[gid] * weight[gid % args.dim];
}

/* Muse post-FFN normalization is RMSNorm(weight * x), rather than the usual
 * weight * RMSNorm(x). For the fixed 8192-wide residual, each of 1024 threads
 * owns two float4s, keeping the weighted values live across the reduction.
 * This fuses the old elementwise multiply and plain RMSNorm without changing
 * their reduction or multiply order. */
kernel void kernel_muse_weighted_rms_8192(
        constant ds4_metal_args_muse_rows &args,
        device const float *x,
        device const float *weight,
        device float *out,
        threadgroup float *shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens || args.dim != 8192u) return;
    if (sgitg == 0) shmem_f32[tiisg] = 0.0f;
    device const float4 *xr = (device const float4 *)(x + (uint64_t)tok * args.dim);
    device const float4 *wr = (device const float4 *)weight;
    device float4 *yr = (device float4 *)(out + (uint64_t)tok * args.dim);
    const uint i0 = (uint)tpitg.x;
    const uint i1 = i0 + ntg.x;
    const float4 v0 = xr[i0] * wr[i0];
    const float4 v1 = xr[i1] * wr[i1];
    float sumf = dot(v0, v0);
    sumf += dot(v1, v1);
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) shmem_f32[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = simd_sum(shmem_f32[tiisg]);
    const float scale = 1.0f / sqrt(sumf / (float)args.dim + args.alpha);
    yr[i0] = v0 * scale;
    yr[i1] = v1 * scale;
}

struct ds4_metal_args_muse_expert_finalize {
    uint32_t n_tokens;
    uint32_t n_slots;
    uint32_t dim;
    float eps;
};

/* Weight-first RMSNorm for every routed expert row, followed by the fp32
 * top-k weighted sum. This is Muse's post-expert finalize contract. */
kernel void kernel_muse_expert_finalize(
        constant ds4_metal_args_muse_expert_finalize &args,
        device const float *part,
        device const float *route_weight,
        device const float *norm_weight,
        device float *out,
        uint tok [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (tok >= args.n_tokens || args.n_slots > 8u || (args.dim & 3u)) return;
    threadgroup float partial[8][32];
    threadgroup float inv[8];
    for (uint i = tid; i < 8u * 32u; i += 256u)
        partial[i / 32u][i % 32u] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n4 = args.dim / 4u;
    float sums[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint s = 0; s < args.n_slots; s++) {
        device const float4 *x4 = (device const float4 *)(part +
                ((ulong)tok * args.n_slots + s) * args.dim);
        device const float4 *w4 = (device const float4 *)norm_weight;
        float sum = 0.0f;
        for (uint d4 = tid; d4 < n4; d4 += 256u) {
            const float4 v = x4[d4] * w4[d4];
            sum += dot(v, v);
        }
        sums[s] = simd_sum(sum);
        if (lane == 0u) partial[s][sg] = sums[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < args.n_slots; s++) {
        float sum = lane < nsg ? partial[s][lane] : 0.0f;
        sum = simd_sum(sum);
        if (tid == 0u) inv[s] = rsqrt(sum / (float)args.dim + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *o4 = (device float4 *)(out + (ulong)tok * args.dim);
    for (uint d4 = tid; d4 < n4; d4 += 256u) {
        float4 acc = float4(0.0f);
        const float4 nw = w4[d4];
        for (uint s = 0; s < args.n_slots; s++) {
            device const float4 *x4 = (device const float4 *)(part +
                    ((ulong)tok * args.n_slots + s) * args.dim);
            acc += route_weight[(ulong)tok * args.n_slots + s] *
                   (x4[d4] * nw * inv[s]);
        }
        o4[d4] = acc;
    }
}

struct ds4_metal_args_muse_router {
    uint32_t n_tokens;
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t pad0;
};

kernel void kernel_muse_router_topk(
        constant ds4_metal_args_muse_router &args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        uint tok [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (tok >= args.n_tokens) return;
    device const float *lg = logits + (uint64_t)tok * args.n_expert;
    const uint nsg = ntg / 32u;
    threadgroup float group_v[16];
    threadgroup int group_i[16];
    threadgroup int winners[16];
    threadgroup float winner_w[16];

    /* One lane owns at most two experts for the supported <=512 shape.  The
     * old kernel evaluated the same values serially in one lane; parallel
     * repeated argmax keeps its value ordering and lowest-index tie break. */
    float mine_v[2] = {-INFINITY, -INFINITY};
    int mine_i[2] = {0x7fffffff, 0x7fffffff};
    for (uint j = 0; j < 2u; j++) {
        const uint e = (uint)tid + j * (uint)ntg;
        if (e < args.n_expert) {
            const float w = muse_sigmoid(lg[e]);
            mine_v[j] = w + bias[e];
            mine_i[j] = (int)e;
        }
    }

    for (uint r = 0; r < args.n_used; r++) {
        float best_v = mine_v[0];
        int best_i = mine_i[0];
        if (mine_v[1] > best_v || (mine_v[1] == best_v && mine_i[1] < best_i)) {
            best_v = mine_v[1];
            best_i = mine_i[1];
        }
        const float simd_v = simd_max(best_v);
        const int simd_i = simd_min(best_v == simd_v ? best_i : 0x7fffffff);
        if (lane == 0) {
            group_v[sg] = simd_v;
            group_i[sg] = simd_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            best_v = lane < nsg ? group_v[lane] : -INFINITY;
            best_i = lane < nsg ? group_i[lane] : 0x7fffffff;
            const float top_v = simd_max(best_v);
            const int top_i = simd_min(best_v == top_v ? best_i : 0x7fffffff);
            if (lane == 0) {
                winners[r] = top_i;
                winner_w[r] = muse_sigmoid(lg[top_i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 2u; j++) {
            if (mine_i[j] == winners[r]) mine_v[j] = -INFINITY;
        }
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < args.n_used; k++) sum += winner_w[k];
        const float inv = 1.0f / max(sum, 6.103515625e-5f);
        for (uint k = 0; k < args.n_used; k++) {
            selected[(uint64_t)tok * args.n_used + k] = winners[k];
            weights[(uint64_t)tok * args.n_used + k] = winner_w[k] * inv;
        }
    }
}

struct ds4_metal_args_muse_swa {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t window;
    uint32_t stride;
};

kernel void kernel_muse_swa_indices(
        constant ds4_metal_args_muse_swa &args,
        device int32_t *selected,
        device uint32_t *counts,
        uint tok [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if (tok >= args.n_tokens) return;
    const uint end = args.pos0 + tok + 1u;
    const uint start = end > args.window ? end - args.window : 0u;
    const uint n = end - start;
    for (uint i = tid; i < n; i += ntg) selected[(uint64_t)tok * args.stride + i] = (int32_t)(start + i);
    if (tid == 0) counts[tok] = n;
}

struct ds4_metal_args_muse_logits {
    uint32_t count;
    float scale;
    float softcap;
    uint32_t pad0;
};

kernel void kernel_muse_logits(
        constant ds4_metal_args_muse_logits &args,
        device float *x,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.count) return;
    float v = x[gid] * args.scale;
    if (args.softcap > 0.0f) v = args.softcap * tanh(v / args.softcap);
    x[gid] = v;
}
