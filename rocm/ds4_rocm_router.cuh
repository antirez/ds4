__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -10.0f) return ds4_precise_expf(x);
    return ds4_precise_log1pf(ds4_precise_expf(x));
}

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

template <uint32_t N_EXPERT>
__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        float expert_weight_scale,
        int has_bias,
        int hash_mode) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * N_EXPERT;
    float *prob = probs + (uint64_t)t * N_EXPERT;
    int32_t *sel = selected + (uint64_t)t * DS4_ROCM_N_EXPERT_USED;
    float *w = weights + (uint64_t)t * DS4_ROCM_N_EXPERT_USED;
    __shared__ float sprob[4][N_EXPERT];
    float local_prob[N_EXPERT / 32u];
    float local_score[N_EXPERT / 32u];

    #pragma unroll
    for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = ds4_precise_sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t] : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * DS4_ROCM_N_EXPERT_USED;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT_USED; j++) {
                const int32_t e = row[j];
                const int valid = e >= 0 && e < (int32_t)N_EXPERT;
                sel[j] = valid ? e : 0;
                const float v = valid ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT_USED; j++) w[j] = w[j] / sum * expert_weight_scale;
        }
        return;
    }

    float out_prob[DS4_ROCM_N_EXPERT_USED] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[DS4_ROCM_N_EXPERT_USED] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < DS4_ROCM_N_EXPERT_USED; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            const float other_score = __shfl_xor(best_score, mask, 32);
            const float other_prob = __shfl_xor(best_prob, mask, 32);
            const uint32_t other_idx = __shfl_xor(best_idx, mask, 32);
#else
            const float other_score = __shfl_xor_sync(FULL_WARP_MASK, best_score, mask);
            const float other_prob = __shfl_xor_sync(FULL_WARP_MASK, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(FULL_WARP_MASK, best_idx, mask);
#endif
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT_USED; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < DS4_ROCM_N_EXPERT_USED; j++) w[j] = w[j] / sum * expert_weight_scale;
    }
}

extern "C" int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    const uint32_t active_n_expert = n_expert != 0u ? n_expert : DS4_ROCM_N_EXPERT;
    const float active_scale = expert_weight_scale != 0.0f ? expert_weight_scale : DS4_ROCM_EXPERT_WEIGHT_SCALE;
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u ||
        (active_n_expert != DS4_ROCM_N_EXPERT && active_n_expert != DS4_ROCM_MAX_N_EXPERT) ||
        (n_expert_used != 0u && n_expert_used != DS4_ROCM_N_EXPERT_USED) ||
        !(active_scale > 0.0f) ||
        !cuda_tensor_has_f32(logits, active_n_expert) ||
        !cuda_tensor_has_f32(probs, active_n_expert) ||
        !cuda_tensor_has_i32(selected, DS4_ROCM_N_EXPERT_USED) ||
        !cuda_tensor_has_f32(weights, DS4_ROCM_N_EXPERT_USED)) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (ok && has_bias && !hash_mode) {
        if (!cuda_model_range_fits(model_size, bias_offset, active_n_expert * sizeof(float))) ok = 0;
        else bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, active_n_expert * sizeof(float), "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        if (hash_rows == 0u) ok = 0;
        else {
            const uint64_t hash_bytes = (uint64_t)hash_rows * DS4_ROCM_N_EXPERT_USED * sizeof(int32_t);
            if (!cuda_model_range_fits(model_size, hash_offset, hash_bytes)) ok = 0;
            else hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
            if (!hash) ok = 0;
        }
    }
    if (ok) {
        dim3 block(32, 4, 1);
        if (active_n_expert == DS4_ROCM_MAX_N_EXPERT) {
            router_select_warp_topk_kernel<DS4_ROCM_MAX_N_EXPERT><<<1, block>>>(
                    (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                    active_scale, has_bias && !hash_mode, hash_mode);
        } else {
            router_select_warp_topk_kernel<DS4_ROCM_N_EXPERT><<<1, block>>>(
                    (int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                    bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                    active_scale, has_bias && !hash_mode, hash_mode);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}
extern "C" int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    const uint32_t active_n_expert = n_expert != 0u ? n_expert : DS4_ROCM_N_EXPERT;
    const float active_scale = expert_weight_scale != 0.0f ? expert_weight_scale : DS4_ROCM_EXPERT_WEIGHT_SCALE;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        (active_n_expert != DS4_ROCM_N_EXPERT && active_n_expert != DS4_ROCM_MAX_N_EXPERT) ||
        (n_expert_used != 0u && n_expert_used != DS4_ROCM_N_EXPERT_USED) ||
        !(active_scale > 0.0f) ||
        !cuda_tensor_has_i32(tokens, n_tokens) ||
        !cuda_tensor_has_elems2(logits, n_tokens, active_n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems2(probs, n_tokens, active_n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, DS4_ROCM_N_EXPERT_USED, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, DS4_ROCM_N_EXPERT_USED, sizeof(float))) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (!cuda_model_range_fits(model_size, bias_offset, active_n_expert * sizeof(float))) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, active_n_expert * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        if (hash_rows == 0u) return 0;
        const uint64_t hash_bytes = (uint64_t)hash_rows * DS4_ROCM_N_EXPERT_USED * sizeof(int32_t);
        if (!cuda_model_range_fits(model_size, hash_offset, hash_bytes)) return 0;
        hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    dim3 block(32, 4, 1);
    if (active_n_expert == DS4_ROCM_MAX_N_EXPERT) {
        router_select_warp_topk_kernel<DS4_ROCM_MAX_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                hash,
                (const float *)logits->ptr,
                (const int32_t *)tokens->ptr,
                0,
                hash_rows,
                n_tokens,
                active_scale,
                has_bias && !hash_mode,
                hash_mode);
    } else {
        router_select_warp_topk_kernel<DS4_ROCM_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                hash,
                (const float *)logits->ptr,
                (const int32_t *)tokens->ptr,
                0,
                hash_rows,
                n_tokens,
                active_scale,
                has_bias && !hash_mode,
                hash_mode);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}
