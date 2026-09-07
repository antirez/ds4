// Qwen 3.8 QSA attention kernels.
//
// GQA: 24 query heads over 2 KV heads at head dim 256, with a per-head-dim
// (1+w baked) RMSNorm on q and k, a 64-dim neox-style RoPE prefix (theta from
// args), softmax scale 1/16, and a per-head sigmoid output gate provided as a
// separate buffer (the converter splits the fused [query|gate] projection
// into attn_q and attn_gate).
//
// Sparsity comes from the indexer: every query row attends to the tokens of
// its top-512 pooled key blocks plus the incomplete tail block, at most
// 2048 + 3 positions. The selection reaches the attention kernel as one bit
// per position and row, and the walk stays in ascending position order
// skipping cleared bits: below the 2048-token budget every bit is set and the
// result is bit-identical to the dense walk (mask_words == 0), which is what
// the release path uses there since it also skips the scoring work.

struct qwen38_qsa_args {
    uint n_q_heads;    /* 24 */
    uint n_kv_heads;   /* 2 */
    uint head_dim;     /* 256 */
    uint rope_dim;     /* 64 */
    uint n_rows;       /* query rows in this pass */
    uint pos0;         /* absolute position of the first query row */
    uint cache_cap;    /* KV cache capacity in tokens */
    uint mask_words;   /* uint32 words per row of the selection mask; 0 = dense */
    float rope_freq_base;
    float norm_eps;
    float attn_scale;  /* 1/sqrt(head_dim) */
};

// Indexer geometry: 4 query heads and one shared key head at dim 128, keys
// pooled 4 at a time. Raw (pre-norm, pre-RoPE) key rows are cached per token
// so a pool can be rebuilt from any chunk boundary; the pooled cache holds one
// normalized, position-rotated row per complete block.
struct qwen38_indexer_args {
    uint n_heads;      /* 4 */
    uint head_dim;     /* 128 */
    uint rope_dim;     /* 64 */
    uint n_rows;
    uint pos0;
    uint cache_cap;    /* raw cache capacity in tokens */
    uint pool_size;    /* 4 */
    float rope_freq_base;
    float norm_eps;
};

/* RMSNorm over head_dim followed by neox RoPE of the first rope_dim dims at
 * `pos`, for one head vector held one value per thread. `partial` needs
 * nsg + rope_dim floats. Barriers are uniform: every thread of the head calls. */
static inline float qwen38_norm_rope_value(
        float raw, device const float *norm_w, uint tid, uint D, uint nsg,
        uint rope_dim, uint pos, float freq_base, float eps,
        threadgroup float *partial, ushort lane, ushort sg) {
    const float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < nsg ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float normed = raw * rsqrt(total / (float)D + eps) * norm_w[tid];
    threadgroup float *rope_vals = partial + nsg;
    if (tid < rope_dim) rope_vals[tid] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = normed;
    const uint half_rope = rope_dim / 2u;
    if (tid < rope_dim) {
        const uint pair = tid < half_rope ? tid : tid - half_rope;
        const float inv_freq = pow(freq_base, -(float)(2u * pair) / (float)rope_dim);
        const float angle = (float)pos * inv_freq;
        const float c = cos(angle);
        const float sn = sin(angle);
        result = tid < half_rope
            ? rope_vals[tid] * c - rope_vals[tid + half_rope] * sn
            : rope_vals[tid] * c + rope_vals[tid - half_rope] * sn;
    }
    return result;
}

/*
 * Indexer queries: RMSNorm + RoPE in place. q layout per row: n_heads x 128.
 * One threadgroup per (row, head), head_dim threads.
 */
kernel void kernel_qwen38_indexer_prepare_q(
        constant qwen38_indexer_args &args,
        device float         *q,
        device const float   *q_norm,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    const uint D = args.head_dim;
    if (row >= args.n_rows || head >= args.n_heads || tid >= D) return;
    device float *hq = q + ((ulong)row * args.n_heads + head) * D;
    hq[tid] = qwen38_norm_rope_value(hq[tid], q_norm, tid, D, D / 32u,
                                     args.rope_dim, args.pos0 + row,
                                     args.rope_freq_base, args.norm_eps,
                                     partial, lane, sg);
}

/* Append the raw indexer key rows to the per-token cache: [cache_cap][128]. */
kernel void kernel_qwen38_indexer_store_k(
        constant qwen38_indexer_args &args,
        device const float   *k,
        device half          *raw_cache,
        uint2 gid [[thread_position_in_grid]]) {
    const uint row = gid.y;
    const uint d = gid.x;
    if (row >= args.n_rows || d >= args.head_dim) return;
    const uint pos = args.pos0 + row;
    if (pos >= args.cache_cap) return;
    raw_cache[(ulong)pos * args.head_dim + d] =
        (half)k[(ulong)row * args.head_dim + d];
}

struct qwen38_indexer_expand_args {
    uint n_rows;
    uint pos0;
    uint top_k;        /* selected blocks per row, as the top-k produced them */
    uint index_topk;   /* token budget: block slots per row */
    uint pool_size;
    uint mask_words;   /* uint32 words per row of the output mask */
};

/*
 * Block selection to a per-row position mask (zeroed by the caller): the
 * tokens of every selected block that is complete for this row, then the
 * row's incomplete tail block. One thread per (row, slot) over
 * index_topk + pool_size - 1 slots. The top-k ranks the row's blocks against
 * the whole pass's, so for early rows it also returns blocks the row cannot
 * see yet (their scores were -inf); those are dropped here, exactly as the
 * reference selects only among num_complete_blocks, so the tail never
 * duplicates a block token.
 */
kernel void kernel_qwen38_indexer_expand(
        constant qwen38_indexer_expand_args &args,
        device const uint    *pool_selected,
        device atomic_uint   *mask,
        uint gid [[thread_position_in_grid]]) {
    const uint slots = args.index_topk + args.pool_size - 1u;
    const uint total = args.n_rows * slots;
    if (gid >= total) return;
    const uint row = gid / slots;
    const uint slot = gid - row * slots;
    const uint visible = args.pos0 + row + 1u;
    const uint complete = visible / args.pool_size;
    uint pos = 0xffffffffu;
    if (slot < args.index_topk) {
        const uint pool_slot = slot / args.pool_size;
        if (pool_slot < args.top_k) {
            const uint pool = pool_selected[(ulong)row * args.top_k + pool_slot];
            if (pool < complete) pos = pool * args.pool_size + slot % args.pool_size;
        }
    } else {
        const uint tail_slot = slot - args.index_topk;
        if (tail_slot < visible % args.pool_size) pos = complete * args.pool_size + tail_slot;
    }
    if (pos == 0xffffffffu) return;
    atomic_fetch_or_explicit(mask + (ulong)row * args.mask_words + (pos >> 5u),
                             1u << (pos & 31u), memory_order_relaxed);
}

/*
 * Rebuild every complete pool that overlaps [pos0, pos0 + n_rows): mean of
 * the pool's raw keys in FP32, RMSNorm with the key norm, RoPE at the pool's
 * first position (the reference rotates pooled keys with the block start).
 * Pooled cache layout: [cache_cap / pool_size][128] halves. One threadgroup
 * per pool, head_dim threads. Reading the raw cache rather than this pass's
 * rows is what lets a pool straddle two prefill chunks or a session rewind.
 */
kernel void kernel_qwen38_indexer_pool(
        constant qwen38_indexer_args &args,
        device const half    *raw_cache,
        device const float   *k_norm,
        device half          *pool_cache,
        threadgroup float    *partial [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint D = args.head_dim;
    if (tid >= D) return;
    const uint pool = args.pos0 / args.pool_size + tgpig;
    const uint pool_start = pool * args.pool_size;
    const uint input_end = args.pos0 + args.n_rows;
    if (pool_start + args.pool_size > input_end ||
        pool_start + args.pool_size > args.cache_cap) return;
    float sum = 0.0f;
    for (uint r = 0; r < args.pool_size; r++) {
        sum += (float)raw_cache[(ulong)(pool_start + r) * D + tid];
    }
    const float mean = sum / (float)args.pool_size;
    pool_cache[(ulong)pool * D + tid] = (half)qwen38_norm_rope_value(
        mean, k_norm, tid, D, D / 32u, args.rope_dim, pool_start,
        args.rope_freq_base, args.norm_eps, partial, lane, sg);
}

/*
 * Normalize (RMS over head_dim, weight already includes the +1) and RoPE
 * the query rows in place. q layout per row: n_q_heads x 256. One
 * threadgroup per (row, q head), 256 threads.
 */
kernel void kernel_qwen38_qsa_prepare_q(
        constant qwen38_qsa_args &args,
        device float         *q,
        device const float   *q_norm,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    const uint D = args.head_dim;
    if (row >= args.n_rows || head >= args.n_q_heads || tid >= D) return;
    device float *hq = q + ((ulong)row * args.n_q_heads + head) * D;

    const float raw = hq[tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 8u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    const float normed = raw * scale * q_norm[tid];

    /* neox-style RoPE on the first rope_dim dims: pair (i, i+rope_dim/2).
     * The pair partner's normalized value moves through shared memory, and
     * the barriers stay uniform across the threadgroup. */
    threadgroup float *rope_vals = partial + 8u;
    if (tid < args.rope_dim) rope_vals[tid] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = normed;
    const uint half_rope = args.rope_dim / 2u;
    if (tid < args.rope_dim) {
        const uint pair = tid < half_rope ? tid : tid - half_rope;
        const float inv_freq = pow(args.rope_freq_base,
                                   -(float)(2u * pair) / (float)args.rope_dim);
        const float angle = (float)(args.pos0 + row) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        if (tid < half_rope) {
            result = rope_vals[tid] * c - rope_vals[tid + half_rope] * s;
        } else {
            result = rope_vals[tid] * c + rope_vals[tid - half_rope] * s;
        }
    }
    hq[tid] = result;
}

/*
 * Normalize + RoPE the new key rows and append key/value rows to the f16
 * caches. One threadgroup per (row, kv head), 256 threads. k/v layouts per
 * row: n_kv_heads x 256. Cache layouts: [cache_cap][n_kv_heads][256] halves.
 */
kernel void kernel_qwen38_qsa_store_kv(
        constant qwen38_qsa_args &args,
        device const float   *k,
        device const float   *v,
        device const float   *k_norm,
        device half          *k_cache,
        device half          *v_cache,
        threadgroup float    *partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    const uint D = args.head_dim;
    if (row >= args.n_rows || head >= args.n_kv_heads || tid >= D) return;
    const uint pos = args.pos0 + row;
    if (pos >= args.cache_cap) return;
    device const float *hk = k + ((ulong)row * args.n_kv_heads + head) * D;
    device const float *hv = v + ((ulong)row * args.n_kv_heads + head) * D;
    const ulong cache_index = ((ulong)pos * args.n_kv_heads + head) * D + tid;

    const float raw = hk[tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 8u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    const float normed = raw * scale * k_norm[tid];

    threadgroup float *rope_vals = partial + 8u;
    if (tid < args.rope_dim) rope_vals[tid] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float stored = normed;
    const uint half_rope = args.rope_dim / 2u;
    if (tid < args.rope_dim) {
        const uint pair = tid < half_rope ? tid : tid - half_rope;
        const float inv_freq = pow(args.rope_freq_base,
                                   -(float)(2u * pair) / (float)args.rope_dim);
        const float angle = (float)pos * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        if (tid < half_rope) {
            stored = rope_vals[tid] * c - rope_vals[tid + half_rope] * s;
        } else {
            stored = rope_vals[tid] * c + rope_vals[tid - half_rope] * s;
        }
    }
    k_cache[cache_index] = (half)stored;
    v_cache[cache_index] = (half)hv[tid];
}

/*
 * Dense causal GQA attention with the fused sigmoid output gate. One
 * threadgroup per (row, q head), 256 threads (8 simdgroups); lanes stream
 * the cache positions with an online softmax, each thread owning one output
 * dimension for the value accumulation is instead handled per-position:
 * every simdgroup takes one position at a time (8 in flight), computes the
 * q.k dot with 8 lanes x 32... — kept simple: each simdgroup walks
 * positions strided by 8, accumulating unnormalized numerator/denominator
 * with the running-max trick, then the partials merge in shared memory.
 * Output row layout matches the value projection: n_q_heads x 256.
 */
kernel void kernel_qwen38_qsa_attn(
        constant qwen38_qsa_args &args,
        device const float   *q,
        device const float   *out_gate,
        device const half    *k_cache,
        device const half    *v_cache,
        device float         *out,
        device const uint    *sel_mask,
        threadgroup float    *scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    const uint D = args.head_dim;
    if (row >= args.n_rows || head >= args.n_q_heads) return;
    const uint kv_head = head / (args.n_q_heads / args.n_kv_heads);
    const uint n_pos = args.pos0 + row + 1u; /* causal: positions 0..pos */
    /* Every simdgroup walks positions sg, sg+8, ... in order; with a mask it
     * skips the cleared ones. The position is uniform across the simdgroup,
     * so the simd_sum below stays convergent either way. */
    device const uint *mask = sel_mask + (ulong)row * args.mask_words;

    /* Shared: q vector (D), per-simdgroup [max, denom] (2*8), and the
     * per-simdgroup output accumulators (8*D). */
    threadgroup float *sq = scratch;
    threadgroup float *sg_max = sq + D;
    threadgroup float *sg_den = sg_max + 8u;
    threadgroup float *sg_out = sg_den + 8u;

    device const float *hq = q + ((ulong)row * args.n_q_heads + head) * D;
    if (tid < D) {
        sq[tid] = hq[tid];
        for (uint g = 0; g < 8u; g++) sg_out[g * D + tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Each simdgroup owns positions sg, sg+8, sg+16, ... Every lane holds
     * 8 query components (256 = 32 lanes x 8). */
    float q_frag[8];
    for (uint j = 0; j < 8u; j++) q_frag[j] = sq[lane * 8u + j];

    float run_max = -INFINITY;
    float run_den = 0.0f;
    for (uint pos = sg; pos < n_pos; pos += 8u) {
        if (args.mask_words != 0u &&
            ((mask[pos >> 5u] >> (pos & 31u)) & 1u) == 0u) continue;
        device const half *kp = k_cache +
            ((ulong)pos * args.n_kv_heads + kv_head) * D + lane * 8u;
        float dot8 = 0.0f;
        for (uint j = 0; j < 8u; j++) dot8 = fma(q_frag[j], (float)kp[j], dot8);
        const float score = simd_sum(dot8) * args.attn_scale;
        const float new_max = max(run_max, score);
        const float correction = exp(run_max - new_max);
        const float weight = exp(score - new_max);
        run_den = run_den * correction + weight;
        run_max = new_max;
        device const half *vp = v_cache +
            ((ulong)pos * args.n_kv_heads + kv_head) * D + lane * 8u;
        for (uint j = 0; j < 8u; j++) {
            const uint d = lane * 8u + j;
            sg_out[sg * (uint)D + d] =
                sg_out[sg * (uint)D + d] * correction + weight * (float)vp[j];
        }
    }
    if (lane == 0u) {
        sg_max[sg] = run_max;
        sg_den[sg] = run_den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Merge the eight simdgroup partials and apply the sigmoid gate. */
    if (tid < D) {
        float best = -INFINITY;
        for (uint g = 0; g < 8u; g++) best = max(best, sg_max[g]);
        float den = 0.0f;
        float num = 0.0f;
        for (uint g = 0; g < 8u; g++) {
            if (sg_max[g] == -INFINITY) continue;
            const float f = exp(sg_max[g] - best);
            den += sg_den[g] * f;
            num += sg_out[g * D + tid] * f;
        }
        const float gate =
            out_gate[((ulong)row * args.n_q_heads + head) * D + tid];
        out[((ulong)row * args.n_q_heads + head) * D + tid] =
            (num / den) / (1.0f + exp(-gate));
    }
}
