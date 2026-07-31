// DFlash-specific data movement. The target and draft graphs otherwise reuse
// Laguna's dense, normalization, RoPE, attention, and FFN kernels.

struct ds4_metal_args_dflash_capture {
    uint32_t n_rows;
    uint32_t n_embd;
    uint32_t n_aux;
    uint32_t aux_index;
    uint32_t src_row0;
    uint32_t dst_row0;
};

kernel void kernel_dflash_capture_rows(
        constant ds4_metal_args_dflash_capture &args,
        device const float *src,
        device       float *features,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t count = (uint64_t)args.n_rows * args.n_embd;
    if ((uint64_t)gid >= count || args.aux_index >= args.n_aux) return;

    const uint row = gid / args.n_embd;
    const uint col = gid - row * args.n_embd;
    float value =
        src[(uint64_t)(args.src_row0 + row) * args.n_embd + col];
    const uint value_bits = as_type<uint>(value);
    if ((value_bits & 0x7f800000u) == 0x7f800000u) {
        value = (value_bits & 0x007fffffu) != 0u ? 0.0f :
            ((value_bits & 0x80000000u) != 0u ? -65504.0f : 65504.0f);
    }
    const uint64_t dst =
        (uint64_t)(args.dst_row0 + row) * args.n_aux * args.n_embd +
        (uint64_t)args.aux_index * args.n_embd + col;
    features[dst] = value;
}

struct ds4_metal_args_dflash_aux_norm {
    uint32_t n_rows;
    uint32_t n_embd;
    uint32_t n_aux;
    float eps;
};

kernel void kernel_dflash_aux_rms_norm(
        constant ds4_metal_args_dflash_aux_norm &args,
        device       float *features,
        device const float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    const uint aux = tgpig.y;
    if (row >= args.n_rows || aux >= args.n_aux || args.n_embd == 0u) return;

    device float *x = features +
        ((uint64_t)row * args.n_aux + aux) * args.n_embd;
    float sum = 0.0f;
    for (uint col = tid; col < args.n_embd; col += ntg.x) {
        const float value = x[col];
        sum += value * value;
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = ntg.x >> 1u; step != 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale =
        rsqrt(scratch[0] / (float)args.n_embd + args.eps);
    device const float *weight = weights + (uint64_t)aux * args.n_embd;
    for (uint col = tid; col < args.n_embd; col += ntg.x) {
        x[col] = x[col] * scale * weight[col];
    }
}

struct ds4_metal_args_dflash_commit_kv {
    uint32_t n_rows;
    uint32_t pos0;
    uint32_t cache_cap;
    uint32_t width;
};

kernel void kernel_dflash_commit_kv_f16(
        constant ds4_metal_args_dflash_commit_kv &args,
        device const float *k,
        device const float *v,
        device        half *key_cache,
        device        half *value_cache,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t count = (uint64_t)args.n_rows * args.width;
    if ((uint64_t)gid >= count || args.cache_cap == 0u) return;

    const uint row = gid / args.width;
    const uint col = gid - row * args.width;
    const uint cache_row = (args.pos0 + row) % args.cache_cap;
    const uint64_t src = (uint64_t)row * args.width + col;
    const uint64_t dst = (uint64_t)cache_row * args.width + col;
    key_cache[dst] = (half)k[src];
    value_cache[dst] = (half)v[src];
}

struct ds4_metal_args_dflash_probabilities {
    uint32_t n_rows;
    uint32_t n_vocab;
};

kernel void kernel_dflash_probabilities(
        constant ds4_metal_args_dflash_probabilities &args,
        device const float   *logits,
        device const int32_t *argmax,
        device       float   *probabilities,
        device       int32_t *top2,
        device       float   *probabilities2,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    if (row >= args.n_rows || args.n_vocab == 0u) return;
    const int32_t top = argmax[row];
    if (top < 0 || (uint32_t)top >= args.n_vocab) {
        if (tid == 0u) {
            probabilities[row] = 0.0f;
            top2[row] = -1;
            probabilities2[row] = 0.0f;
        }
        return;
    }

    device const float *row_logits =
        logits + (uint64_t)row * args.n_vocab;
    const float top_logit = row_logits[top];
    float sum = 0.0f;
    float m2 = -INFINITY;
    uint i2 = 0xFFFFFFFFu;
    if (isfinite(top_logit)) {
        for (uint col = tid; col < args.n_vocab; col += ntg.x) {
            const float value = row_logits[col];
            const float term = exp(value - top_logit);
            if (isfinite(term)) sum += term;
            if (col != (uint32_t)top && value > m2) {
                m2 = value;
                i2 = col;
            }
        }
    }
    scratch[tid] = sum;
    threadgroup float *max2 = scratch + ntg.x;
    threadgroup uint *idx2 = (threadgroup uint *)(max2 + ntg.x);
    max2[tid] = m2;
    idx2[tid] = i2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = ntg.x >> 1u; step != 0u; step >>= 1u) {
        if (tid < step) {
            scratch[tid] += scratch[tid + step];
            if (max2[tid + step] > max2[tid]) {
                max2[tid] = max2[tid + step];
                idx2[tid] = idx2[tid + step];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        const float inv_sum =
            scratch[0] > 0.0f ? 1.0f / scratch[0] : 0.0f;
        probabilities[row] = inv_sum;
        top2[row] = idx2[0] == 0xFFFFFFFFu ? -1 : (int32_t)idx2[0];
        probabilities2[row] = isfinite(max2[0]) ?
            exp(max2[0] - top_logit) * inv_sum : 0.0f;
    }
}
