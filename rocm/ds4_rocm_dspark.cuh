/* DSpark speculative decoding kernels for ROCm.
 *
 * These two entry points were previously stubbed out in
 * ds4_rocm_unavailable.cu, so every DSpark proposal attempt failed at the
 * first stage: metal_graph_eval_dspark_stage_block() saw the noncausal
 * attention return 0, stage_chain_done stayed at 0, and the proposer emitted
 * no draft at all. DSpark therefore loaded and ran on ROCm but never proposed
 * a token, costing the drafting overhead for nothing.
 *
 * Both kernels are direct ports of the CUDA versions in ds4_cuda.cu. They use
 * only plain FP32 math, shared memory and __syncthreads(), so no HIP-specific
 * restructuring was needed. Differences from the CUDA originals:
 *   - the sinks/weight rows resolve through cuda_model_range_ptr(), matching
 *     the other ROCm attention launchers, instead of the multi-tier
 *     cuda_resolve_weight_ptr();
 *   - the markov wrapper drops the CUDA device-tier save/restore, since the
 *     ROCm backend is single-device.
 */

/* Non-causal attention over the DSpark draft block: every draft row attends to
 * the whole support window, so there is no causal mask and no position term. */
__global__ static void dspark_attention_noncausal_raw_batch_heads_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t tok = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (tok >= n_tokens || h >= n_head) return;
    extern __shared__ float sh_scores[]; /* n_raw floats */
    const float *qh = q + ((uint64_t)tok * n_head + h) * head_dim;
    const float scale = rsqrtf((float)head_dim);
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        const uint32_t row = (raw_start + r) % raw_cap;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        sh_scores[r] = dot * scale;
    }
    __syncthreads();
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        local_max = fmaxf(local_max, sh_scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        sh_scores[r] = expf(sh_scores[r] - max_s);
        den_local += sh_scores[r];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)tok * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            acc += raw_kv[(uint64_t)row * head_dim + d] * sh_scores[r];
        }
        oh[d] = acc / denom;
    }
}

extern "C" int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t              n_tokens,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (!heads || !q || !raw_kv || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || n_head == 0 || head_dim == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float)) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
            "dspark_attn_sinks");
    if (!sinks) return 0;
    const size_t shmem = (size_t)n_raw * sizeof(float);
    if (shmem > 32768) return 0; /* draft blocks are tiny; guard anyway */
    dim3 grid(n_tokens, n_head, 1);
    dspark_attention_noncausal_raw_batch_heads_kernel<<<grid, 256, shmem>>>(
            (float *)heads->ptr,
            sinks,
            (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_tokens, n_raw, raw_cap, raw_start, n_head, head_dim);
    return cuda_ok(cudaGetLastError(),
                   "dspark attention noncausal raw batch heads launch");
}

/* Markov correction over the base logits: adds a low-rank q8-style term keyed
 * on the previous token, then argmaxes. The packed row layout is 34 bytes per
 * 32-value block, an fp16 scale followed by 32 int8 weights. */
__global__ static void dspark_markov_argmax_kernel(
        unsigned long long *out_key,
        const float *logits,
        const unsigned char *w1_row,
        const unsigned char *w2,
        uint32_t vocab,
        uint32_t rank_blocks) {
    __shared__ float state[256];
    const uint32_t tid = threadIdx.x;
    if (tid < rank_blocks * 32u) {
        const uint32_t b = tid >> 5, k = tid & 31u;
        const unsigned char *blk = w1_row + (uint64_t)b * 34u;
        const float d = __half2float(*(const __half *)blk);
        state[tid] = d * (float)((const int8_t *)(blk + 2))[k];
    }
    __syncthreads();

    float best_v = -INFINITY;
    uint32_t best_i = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vocab;
         i += gridDim.x * blockDim.x) {
        const unsigned char *row = w2 + (uint64_t)i * rank_blocks * 34u;
        float acc = 0.0f;
        for (uint32_t b = 0; b < rank_blocks; b++) {
            const unsigned char *blk = row + (uint64_t)b * 34u;
            const float d = __half2float(*(const __half *)blk);
            const int8_t *q = (const int8_t *)(blk + 2);
            float s = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < 32u; k++) s += (float)q[k] * state[b * 32u + k];
            acc += d * s;
        }
        const float v = logits[i] + acc;
        if (topk_score_better(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
        }
    }

    __shared__ float vals[256];
    __shared__ uint32_t idxs[256];
    vals[tid] = best_v;
    idxs[tid] = best_i;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            if (topk_score_better(vals[tid + stride], idxs[tid + stride],
                                  vals[tid], idxs[tid])) {
                vals[tid] = vals[tid + stride];
                idxs[tid] = idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        /* Monotonic float key; ~idx in the low bits makes ties resolve to
         * the smaller index under atomicMax (matches topk_score_better). */
        const unsigned int f = __float_as_uint(vals[0]);
        const unsigned int fkey = (f & 0x80000000u) ? ~f : (f | 0x80000000u);
        const unsigned long long key =
            ((unsigned long long)fkey << 32) | (unsigned int)(~idxs[0]);
        atomicMax(out_key, key);
    }
}

extern "C" int ds4_gpu_dspark_markov_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits_row,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              w1_offset,
        uint64_t              w2_offset,
        uint32_t              prev_token,
        uint32_t              vocab,
        uint32_t              rank) {
    if (!out_idx || !logits_row || !model_map || vocab == 0 ||
        rank == 0 || (rank & 31u) != 0u || rank > 256u ||
        out_idx->bytes < sizeof(unsigned long long) ||
        logits_row->bytes < (uint64_t)vocab * sizeof(float)) {
        return 0;
    }
    const uint32_t rank_blocks = rank / 32u;
    const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;
    if (w1_offset > model_size ||
        (uint64_t)prev_token * row_bytes + row_bytes > model_size - w1_offset ||
        w2_offset > model_size ||
        (uint64_t)vocab * row_bytes > model_size - w2_offset) {
        return 0;
    }
    const unsigned char *w1_row = (const unsigned char *)cuda_model_range_ptr(
            model_map, w1_offset + (uint64_t)prev_token * row_bytes,
            row_bytes, "markov_w1_row");
    const unsigned char *w2 = (const unsigned char *)cuda_model_range_ptr(
            model_map, w2_offset, (uint64_t)vocab * row_bytes, "markov_w2");
    if (!w1_row || !w2) return 0;
    if (cudaMemsetAsync(out_idx->ptr, 0,
                        sizeof(unsigned long long)) != cudaSuccess) {
        return 0;
    }
    dspark_markov_argmax_kernel<<<128, 256>>>(
            (unsigned long long *)out_idx->ptr,
            (const float *)logits_row->ptr,
            w1_row, w2, vocab, rank_blocks);
    return cuda_ok(cudaGetLastError(), "dspark markov argmax launch");
}
