__global__ static void attention_visual_pack_mixed_kv_kernel(
        float       *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t     n_raw,
        uint32_t     raw_cap,
        uint32_t     raw_start,
        uint32_t     n_comp,
        uint32_t     head_dim) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)(n_raw + n_comp) * head_dim;
    if (gid >= count) return;
    const uint32_t d = gid % head_dim;
    const uint32_t row = gid / head_dim;
    if (row < n_raw) {
        const uint32_t physical = (raw_start + row) % raw_cap;
        dst[gid] = raw_kv[(uint64_t)physical * head_dim + d];
    } else {
        dst[gid] = comp_kv[(uint64_t)(row - n_raw) * head_dim + d];
    }
}

__global__ static void attention_visual_mixed_softmax_kernel(
        float          *scores,
        const float    *sinks,
        const float    *comp_mask,
        const uint32_t *raw_bounds,
        uint32_t        use_comp_mask,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        first_raw_pos,
        uint32_t        n_raw,
        uint32_t        n_comp,
        uint32_t        ratio,
        uint32_t        n_keys) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    const uint32_t raw_lo = raw_bounds[2u * t];
    const uint32_t raw_hi = raw_bounds[2u * t + 1u];
    const uint32_t qpos = pos0 + t;
    uint32_t visible_comp = ratio ? (qpos + 1u) / ratio : 0u;
    if (visible_comp > n_comp) visible_comp = n_comp;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float score = -INFINITY;
        if (k < n_raw) {
            const uint32_t kpos = first_raw_pos + k;
            if (kpos >= raw_lo && kpos <= raw_hi) score = row[k];
        } else {
            const uint32_t c = k - n_raw;
            if (c < visible_comp) {
                const float add = use_comp_mask
                    ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) score = row[k] + add;
            }
        }
        row[k] = score;
        local_max = fmaxf(local_max, score);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x],
                                          partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0u) max_s = partial[0];
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        const float probability = isfinite(row[k])
            ? expf(row[k] - max_s) : 0.0f;
        row[k] = probability;
        sum += probability;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u)
        denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x)
        row[k] /= denom;
}

__global__ static void router_select_visual_parallel_kernel(
        int32_t       *selected,
        float         *weights,
        float         *probs,
        const float   *bias,
        const float   *visual_bias,
        const int32_t *hash,
        const float   *logits,
        const int32_t *tokens,
        uint32_t       hash_rows,
        uint32_t       vocab_size,
        uint32_t       n_tokens,
        int            has_bias,
        int            hash_mode) {
    const uint32_t t = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs + (uint64_t)t * 256u;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[256];

    const float p = ds4_precise_sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0u) return;
    const int32_t token = tokens[t];
    const bool image = token >= 0 && (uint32_t)token >= vocab_size;
    if (hash_mode && !image) {
        const uint32_t row = token >= 0 && (uint32_t)token < hash_rows
            ? (uint32_t)token : 0u;
        const int32_t *src = hash + (uint64_t)row * 6u;
        for (uint32_t j = 0; j < 6u; j++) sel[j] = src[j];
    } else {
        for (uint32_t j = 0; j < 6u; j++) sel[j] = -1;
        for (uint32_t e = 0; e < 256u; e++) {
            const float score = sprob[e] +
                (image ? visual_bias[e] : (has_bias ? bias[e] : 0.0f));
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t old = sel[j];
                const float old_score = old < 0 ? -INFINITY :
                    sprob[old] + (image ? visual_bias[old]
                                        : (has_bias ? bias[old] : 0.0f));
                if (old < 0 || score > old_score) {
                    for (uint32_t k = 5u; k > j; k--) sel[k] = sel[k - 1u];
                    sel[j] = (int32_t)e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < 6u; j++) {
        const int32_t e = sel[j];
        const float v = e >= 0 && e < 256 ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
}

extern "C" int ds4_gpu_attention_visual_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask,
        uint32_t use_comp_mask, const int32_t *tokens, uint32_t vocab_size,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    if (!heads || !model_map || !q || !raw_kv || !tokens || vocab_size == 0 ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && (!comp_kv || ratio == 0)) ||
        (use_comp_mask && !comp_mask) || comp_kv_f16 || !g_cublas_ready ||
        !cuda_model_range_fits(model_size, sinks_offset,
                               (uint64_t)n_head * sizeof(float)) ||
        !cuda_tensor_has_elems3(heads, n_tokens, n_head, head_dim,
                                sizeof(float)) ||
        !cuda_tensor_has_elems3(q, n_tokens, n_head, head_dim,
                                sizeof(float)) ||
        !cuda_tensor_has_elems2(raw_kv, raw_cap, head_dim, sizeof(float)) ||
        (n_comp && !cuda_tensor_has_elems2(comp_kv, n_comp, head_dim,
                                           sizeof(float))) ||
        (use_comp_mask && !cuda_tensor_has_elems2(comp_mask, n_tokens, n_comp,
                                                   sizeof(float))) ||
        pos0 > UINT32_MAX - n_tokens || n_raw > pos0 + n_tokens) return 0;

    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
            "visual_attn_sinks");
    if (!sinks) return 0;

    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    std::vector<uint32_t> bounds((size_t)n_tokens * 2u);
    if (!ds4_deepseek4_attention_bounds(
            (const int *)tokens, n_tokens, vocab_size,
            pos0, n_raw, window, bounds.data())) return 0;

    if (n_raw > UINT32_MAX - n_comp) return 0;
    const uint32_t n_keys = n_raw + n_comp;
    const uint64_t kv_count = (uint64_t)n_keys * head_dim;
    const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
    const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
    if (kv_count > UINT64_MAX / sizeof(float) ||
        score_count > UINT64_MAX / sizeof(float) ||
        out_count > UINT64_MAX / sizeof(float)) return 0;
    const uint64_t kv_bytes = kv_count * sizeof(float);
    const uint64_t score_offset = (kv_bytes + 255u) & ~UINT64_C(255);
    const uint64_t score_bytes = score_count * sizeof(float);
    if (score_offset > UINT64_MAX - score_bytes - 255u) return 0;
    const uint64_t out_offset =
        (score_offset + score_bytes + 255u) & ~UINT64_C(255);
    const uint64_t out_bytes = out_count * sizeof(float);
    if (out_offset > UINT64_MAX - out_bytes - 255u) return 0;
    const uint64_t bounds_offset =
        (out_offset + out_bytes + 255u) & ~UINT64_C(255);
    const uint64_t bounds_bytes =
        (uint64_t)n_tokens * 2u * sizeof(uint32_t);
    if (bounds_offset > UINT64_MAX - bounds_bytes) return 0;
    float *tmp = (float *)cuda_tmp_alloc(
            bounds_offset + bounds_bytes, "visual mixed attention");
    if (!tmp) return 0;
    float *kv = tmp;
    float *scores = (float *)((char *)tmp + score_offset);
    float *out_tmp = (float *)((char *)tmp + out_offset);
    uint32_t *device_bounds = (uint32_t *)((char *)tmp + bounds_offset);
    if (!cuda_ok(cudaMemcpy(device_bounds, bounds.data(),
                            (size_t)bounds_bytes, cudaMemcpyHostToDevice),
                 "visual attention bounds upload")) return 0;

    attention_visual_pack_mixed_kv_kernel<<<
        (kv_count + 255u) / 256u, 256u>>>(
            kv, (const float *)raw_kv->ptr,
            n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
            n_raw, raw_cap, raw_start, n_comp, head_dim);
    if (!cuda_ok(cudaGetLastError(), "visual attention KV pack launch"))
        return 0;

    const float alpha = rsqrtf((float)head_dim);
    const float beta = 0.0f;
    cublasStatus_t status = cublasSgemmStridedBatched(
            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)n_keys, (int)n_tokens, (int)head_dim,
            &alpha, kv, (int)head_dim, 0,
            (const float *)q->ptr, (int)(n_head * head_dim),
            (long long)head_dim, &beta, scores, (int)n_keys,
            (long long)n_keys * n_tokens, (int)n_head);
    if (!cublas_ok(status, "visual attention score gemm")) return 0;
    const dim3 score_grid(n_tokens, n_head, 1u);
    attention_visual_mixed_softmax_kernel<<<score_grid, 256u>>>(
            scores, sinks,
            use_comp_mask ? (const float *)comp_mask->ptr : NULL,
            device_bounds, use_comp_mask, n_tokens, pos0, first_raw_pos,
            n_raw, n_comp, ratio, n_keys);
    if (!cuda_ok(cudaGetLastError(), "visual attention softmax launch"))
        return 0;

    const float one = 1.0f;
    status = cublasSgemmStridedBatched(
            g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)head_dim, (int)n_tokens, (int)n_keys,
            &one, kv, (int)head_dim, 0, scores, (int)n_keys,
            (long long)n_keys * n_tokens, &beta, out_tmp, (int)head_dim,
            (long long)head_dim * n_tokens, (int)n_head);
    if (!cublas_ok(status, "visual attention value gemm")) return 0;
    const uint64_t values = (uint64_t)n_tokens * n_head * head_dim;
    attention_prefill_unpack_heads_kernel<<<
        (values + 255u) / 256u, 256u>>>(
            (float *)heads->ptr, out_tmp, n_tokens, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "visual attention unpack launch");
}

extern "C" int ds4_gpu_router_select_batch_visual_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
        uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows,
        bool has_bias, bool hash_mode, const void *vision_map,
        uint64_t vision_size, uint64_t visual_bias_offset,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens,
        uint32_t vocab_size, uint32_t n_expert, uint32_t n_expert_used,
        float expert_weight_scale, uint32_t n_tokens) {
    if (!selected || !weights || !probs || !logits || !tokens ||
        !model_map || !vision_map || n_tokens == 0 || vocab_size == 0 ||
        n_expert != 256u || n_expert_used != 6u ||
        fabsf(expert_weight_scale - 1.5f) > 1.0e-6f ||
        !cuda_tensor_has_elems2(logits, n_tokens, 256u, sizeof(float)) ||
        !cuda_tensor_has_elems2(probs, n_tokens, 256u, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, 6u, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, 6u, sizeof(float)) ||
        !cuda_tensor_has_i32(tokens, n_tokens)) return 0;

    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (!cuda_model_range_fits(model_size, bias_offset,
                                   256u * sizeof(float))) return 0;
        bias = (const float *)cuda_model_range_ptr(
                model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        if (hash_rows == 0u) return 0;
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (!cuda_model_range_fits(model_size, hash_offset, hash_bytes))
            return 0;
        hash = (const int32_t *)cuda_model_range_ptr(
                model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    if (!cuda_model_range_fits(vision_size, visual_bias_offset,
                               256u * sizeof(float))) return 0;
    const float *visual_bias = (const float *)cuda_model_range_ptr(
            vision_map, visual_bias_offset, 256u * sizeof(float),
            "visual_router_bias");
    if (!visual_bias) return 0;

    router_select_visual_parallel_kernel<<<n_tokens, 256>>>(
            (int32_t *)selected->ptr, (float *)weights->ptr,
            (float *)probs->ptr, bias, visual_bias, hash,
            (const float *)logits->ptr, (const int32_t *)tokens->ptr,
            hash_rows, vocab_size, n_tokens,
            has_bias && !hash_mode, hash_mode);
    return cuda_ok(cudaGetLastError(), "visual router select launch");
}
