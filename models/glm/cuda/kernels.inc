__device__ __forceinline__ static float glm_rope_yarn_corr_factor_dev(
        int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2.0f * (float)M_PI)) /
           (2.0f * logf(base));
}
__device__ __forceinline__ static float glm_rope_yarn_ramp_dev(
        float low, float high, int i0) {
    const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}
__device__ __forceinline__ static void glm_rope_yarn_dev(
        float theta_extrap, float freq_scale, const float corr_dims[2],
        int i0, float ext_factor, float mscale,
        float *cos_theta, float *sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = glm_rope_yarn_ramp_dev(corr_dims[0], corr_dims[1], i0) *
                         ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static int cuda_current_tier(void) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    return dev;
}

/* ===== GLM 5.2 stubs (to be implemented; fail loudly) ===== */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
__global__ static void add3_kernel(float *out, const float *a,
                                   const float *b, const float *c,
                                   uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i] + c[i];
}

extern "C" int ds4_gpu_add3_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t                n) {
    if (!out || !a || !b || !c || n == 0 ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float) ||
        c->bytes < (uint64_t)n * sizeof(float)) {
        return 0;
    }
    add3_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr, (const float *)a->ptr,
            (const float *)b->ptr, (const float *)c->ptr, n);
    return cuda_ok(cudaGetLastError(), "add3 launch");
}

/* Fused decode residual: sum_out = a + b; norm_out = rmsnorm(sum) * w.
 * Single row, one block (two-pass over n with a shared reduction). */
__global__ static void glm_add_rms_norm_weight_kernel(
        float *norm_out,
        float *sum_out,
        const float *a,
        const float *b,
        const float *w,
        uint32_t n,
        float eps) {
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    __shared__ float sh[32];
    float sumsq = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = a[i] + b[i];
        sum_out[i] = v;
        sumsq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sumsq += __shfl_xor_sync(0xffffffffu, sumsq, off);
    }
    if ((tid & 31u) == 0u) sh[tid >> 5] = sumsq;
    __syncthreads();
    if (tid < 32u) {
        sumsq = (tid < (nth + 31u) / 32u) ? sh[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sumsq += __shfl_xor_sync(0xffffffffu, sumsq, off);
        }
        if (tid == 0u) sh[0] = sumsq;
    }
    __syncthreads();
    const float scale = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += nth) {
        norm_out[i] = (sum_out[i] * scale) * w[i];
    }
}

extern "C" int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    if (!norm_out || !sum_out || !a || !b || !model_map || n == 0 ||
        norm_out->bytes < (uint64_t)n * sizeof(float) ||
        sum_out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float) ||
        weight_offset > model_size ||
        (uint64_t)n * sizeof(float) > model_size - weight_offset) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *w = (const float *)cuda_resolve_weight_ptr(
            model_map, weight_offset, (uint64_t)n * sizeof(float),
            logical_tier, "rms_weight");
    if (!w) return 0;
    glm_add_rms_norm_weight_kernel<<<1, 1024>>>(
            (float *)norm_out->ptr, (float *)sum_out->ptr,
            (const float *)a->ptr, (const float *)b->ptr, w, n, eps);
    return cuda_ok(cudaGetLastError(), "add rms norm weight");
}

extern "C" bool ds4_gpu_commands_active(void) {
    return false;
}

__global__ static void glm_embed_token_q8_0_kernel(
        float *out,
        const unsigned char *w,
        uint32_t token,
        uint32_t n_embd) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const uint64_t row_blocks = n_embd / 32u;
    const unsigned char *blk =
        w + ((uint64_t)token * row_blocks + (d >> 5)) * 34u;
    const float scale = __half2float(*(const __half *)blk);
    out[d] = scale * (float)((const int8_t *)(blk + 2))[d & 31u];
}

extern "C" int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          weight_type,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (!out || !model_map || n_embd == 0 || (n_embd & 31u) != 0u ||
        token >= n_vocab) {
        return 0;
    }
    if (weight_type != 8u) {   /* DS4_TENSOR_Q8_0 */
        fprintf(stderr, "ds4: embed_token_quant: unsupported type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)n_embd / 32u) * 34u;
    if (weight_offset > model_size ||
        (uint64_t)n_vocab * row_bytes > model_size - weight_offset ||
        out->bytes < (uint64_t)n_embd * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const unsigned char *w = (const unsigned char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, (uint64_t)n_vocab * row_bytes,
            logical_tier, "glm_token_embd");
    if (!w) return 0;
    glm_embed_token_q8_0_kernel<<<(n_embd + 255) / 256, 256>>>(
            (float *)out->ptr, w, token, n_embd);
    return cuda_ok(cudaGetLastError(), "glm embed token launch");
}

__global__ static void glm_embed_tokens_q8_0_kernel(
        float *out,
        const int32_t *tokens,
        const unsigned char *w,
        uint32_t n_tokens,
        uint32_t n_embd) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    uint32_t t = gid / n_embd;
    uint32_t d = gid - (uint64_t)t * n_embd;
    int32_t tok = tokens[t];
    const uint64_t row_blocks = n_embd / 32u;
    const unsigned char *blk = w + ((uint64_t)tok * row_blocks + (d >> 5)) * 34u;
    const float scale = __half2float(*(const __half *)blk);
    out[gid] = scale * (float)((const int8_t *)(blk + 2))[d & 31u];
}

extern "C" int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (!out || !tokens || !model_map || n_tokens == 0 || n_embd == 0 ||
        (n_embd & 31u) != 0u) {
        return 0;
    }
    if (weight_type != 8u) {   /* DS4_TENSOR_Q8_0 */
        fprintf(stderr, "ds4: embed_tokens_quant: unsupported type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)n_embd / 32u) * 34u;
    if (weight_offset > model_size ||
        (uint64_t)n_vocab * row_bytes > model_size - weight_offset ||
        out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const unsigned char *w = (const unsigned char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, (uint64_t)n_vocab * row_bytes,
            logical_tier, "glm_token_embd");
    if (!w) return 0;
    uint64_t n = (uint64_t)n_tokens * n_embd;
    glm_embed_tokens_q8_0_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const int32_t *)tokens->ptr,
            w, n_tokens, n_embd);
    return cuda_ok(cudaGetLastError(), "glm embed tokens launch");
}

extern "C" int ds4_gpu_flush_encoder(void) {
    /* Metal encoder flush: CUDA kernels are already queued in stream
     * order, nothing to split. */
    return 1;
}

extern "C" int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_flash_staged_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_flash_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_full_tensor\n");
    return 0;
}

template <typename CT>
__device__ __forceinline__ static float2 glm_cache_rope_pair_f16_dev(
        const CT *rope_cache, uint64_t rope_base, uint32_t r,
        uint32_t row, uint32_t qk_rope, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, const float corr_dims[2]) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
    const float theta = theta_base * powf(freq_base, inv_ndims * (float)r);
    float ct, st;
    glm_rope_yarn_dev(theta, freq_scale, corr_dims, (int)r,
                      ext_factor, attn_factor, &ct, &st);
    const float x0 = (float)rope_cache[rope_base + r];
    const float x1 = (float)rope_cache[rope_base + r + 1u];
    return make_float2(x0 * ct - x1 * st, x0 * st + x1 * ct);
}

/* Scalar-correct MLA attention: one warp per head, grid
 * (ceil(n_head/8), n_tokens). The row loop handles either a contiguous
 * causal range or an explicit selected-row list and mirrors the Metal online
 * softmax so numerics stay comparable. */
template <typename CT, bool selected_rows>
__global__ static void glm_attention_lora_causal_kernel(
        float *lora_out,
        const float *q,
        const float *qk_low,
        const CT *kv_lora_cache,
        const CT *k_rope_cache,
        const uint32_t *selected,
        uint32_t cache_cap,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float scale) {
    const uint32_t token = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t head = blockIdx.x * 8u + warp;
    if (token >= n_tokens || head >= n_head || kv_lora_dim != 512u ||
        qk_rope != 64u) {
        return;
    }
    const uint32_t visible = selected_rows
        ? n_selected : min(n_selected, pos0 + token + 1u);
    if (visible == 0u) return;

    const uint32_t qk_dim = qk_nope + qk_rope;
    const float *qh = q + (uint64_t)token * n_head * qk_dim +
                      (uint64_t)head * qk_dim;
    const float4 *low4 = (const float4 *)(qk_low +
        (uint64_t)token * n_head * kv_lora_dim +
        (uint64_t)head * kv_lora_dim);

    float4 low0 = low4[lane];
    float4 low1 = low4[lane + 32u];
    float4 low2 = low4[lane + 64u];
    float4 low3 = low4[lane + 96u];
    float4 qrope = make_float4(0.f, 0.f, 0.f, 0.f);
    const uint32_t rope_vecs = qk_rope >> 2;   /* 16 */
    if (lane < rope_vecs) {
        qrope = *((const float4 *)(qh + qk_nope + lane * 4u));
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)qk_rope, (int)n_ctx_orig,
                                                 beta_fast, freq_base)));
        corr_dims[1] = fminf((float)qk_rope - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)qk_rope, (int)n_ctx_orig,
                                                beta_slow, freq_base)));
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = make_float4(0.f,0.f,0.f,0.f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t ri = 0u; ri < visible; ri++) {
        const uint32_t row = selected_rows
            ? selected[(uint64_t)token * n_selected + ri] : ri;
        if (row >= cache_cap) continue;
        const CT *kvrow = kv_lora_cache + (uint64_t)row * kv_lora_dim;
        float partial = 0.0f;
        {
            const float4 k0 = make_float4(
                (float)(kvrow[lane*4u+0u]), (float)(kvrow[lane*4u+1u]),
                (float)(kvrow[lane*4u+2u]), (float)(kvrow[lane*4u+3u]));
            const float4 k1 = make_float4(
                (float)(kvrow[(lane+32u)*4u+0u]), (float)(kvrow[(lane+32u)*4u+1u]),
                (float)(kvrow[(lane+32u)*4u+2u]), (float)(kvrow[(lane+32u)*4u+3u]));
            const float4 k2 = make_float4(
                (float)(kvrow[(lane+64u)*4u+0u]), (float)(kvrow[(lane+64u)*4u+1u]),
                (float)(kvrow[(lane+64u)*4u+2u]), (float)(kvrow[(lane+64u)*4u+3u]));
            const float4 k3 = make_float4(
                (float)(kvrow[(lane+96u)*4u+0u]), (float)(kvrow[(lane+96u)*4u+1u]),
                (float)(kvrow[(lane+96u)*4u+2u]), (float)(kvrow[(lane+96u)*4u+3u]));
            partial += low0.x*k0.x + low0.y*k0.y + low0.z*k0.z + low0.w*k0.w;
            partial += low1.x*k1.x + low1.y*k1.y + low1.z*k1.z + low1.w*k1.w;
            partial += low2.x*k2.x + low2.y*k2.y + low2.z*k2.z + low2.w*k2.w;
            partial += low3.x*k3.x + low3.y*k3.y + low3.z*k3.z + low3.w*k3.w;
            if (lane < rope_vecs) {
                const uint64_t rope_base = (uint64_t)row * qk_rope;
                const uint32_t r = lane * 4u;
                const float2 y0 = glm_cache_rope_pair_f16_dev(
                        k_rope_cache, rope_base, r, row, qk_rope, freq_base,
                        freq_scale, ext_factor, attn_factor, corr_dims);
                const float2 y1 = glm_cache_rope_pair_f16_dev(
                        k_rope_cache, rope_base, r + 2u, row, qk_rope,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        corr_dims);
                partial += qrope.x*y0.x + qrope.y*y0.y +
                           qrope.z*y1.x + qrope.w*y1.y;
            }
            for (uint32_t off = 16u; off > 0u; off >>= 1u) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale;
            const float new_m = fmaxf(M, score);
            const float old_scale = expf(M - new_m);
            const float row_scale = expf(score - new_m);
            o0.x = o0.x*old_scale + k0.x*row_scale; o0.y = o0.y*old_scale + k0.y*row_scale;
            o0.z = o0.z*old_scale + k0.z*row_scale; o0.w = o0.w*old_scale + k0.w*row_scale;
            o1.x = o1.x*old_scale + k1.x*row_scale; o1.y = o1.y*old_scale + k1.y*row_scale;
            o1.z = o1.z*old_scale + k1.z*row_scale; o1.w = o1.w*old_scale + k1.w*row_scale;
            o2.x = o2.x*old_scale + k2.x*row_scale; o2.y = o2.y*old_scale + k2.y*row_scale;
            o2.z = o2.z*old_scale + k2.z*row_scale; o2.w = o2.w*old_scale + k2.w*row_scale;
            o3.x = o3.x*old_scale + k3.x*row_scale; o3.y = o3.y*old_scale + k3.y*row_scale;
            o3.z = o3.z*old_scale + k3.z*row_scale; o3.w = o3.w*old_scale + k3.w*row_scale;
            S = S*old_scale + row_scale;
            M = new_m;
        }
    }

    const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
    float4 *out4 = (float4 *)(lora_out +
        ((uint64_t)token * n_head + head) * kv_lora_dim);
    o0.x*=inv_s; o0.y*=inv_s; o0.z*=inv_s; o0.w*=inv_s;
    o1.x*=inv_s; o1.y*=inv_s; o1.z*=inv_s; o1.w*=inv_s;
    o2.x*=inv_s; o2.y*=inv_s; o2.z*=inv_s; o2.w*=inv_s;
    o3.x*=inv_s; o3.y*=inv_s; o3.z*=inv_s; o3.w*=inv_s;
    out4[lane] = o0;
    out4[lane + 32u] = o1;
    out4[lane + 64u] = o2;
    out4[lane + 96u] = o3;
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        n_tokens == 0 || n_head == 0 || kv_lora_dim != 512u ||
        qk_rope != 64u) {
        fprintf(stderr, "ds4: glm attn lora causal: unsupported config "
                "(n_tok=%u head=%u lora=%u rope=%u f16=%d)\n",
                n_tokens, n_head, kv_lora_dim, qk_rope, (int)cache_f16);
        return 0;
    }
    const float scale = 1.0f / sqrtf((float)(qk_nope + qk_rope));
    dim3 grid((n_head + 7u) / 8u, n_tokens, 1);
    if (cache_f16) {
        glm_attention_lora_causal_kernel<__half, false><<<grid, 256>>>(
                (float *)lora_out->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const __half *)kv_lora_cache->ptr,
                (const __half *)k_rope_cache->ptr,
                NULL, cache_cap,
                n_tokens, pos0, n_selected, n_head, kv_lora_dim, qk_nope,
                qk_rope, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, scale);
    } else {
        glm_attention_lora_causal_kernel<float, false><<<grid, 256>>>(
                (float *)lora_out->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const float *)kv_lora_cache->ptr,
                (const float *)k_rope_cache->ptr,
                NULL, cache_cap,
                n_tokens, pos0, n_selected, n_head, kv_lora_dim, qk_nope,
                qk_rope, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, scale);
    }
    return cuda_ok(cudaGetLastError(), "glm attn lora causal launch");
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    const uint64_t cache_elem = cache_f16 ? sizeof(__half) : sizeof(float);
    const uint64_t qk_dim = (uint64_t)qk_nope + qk_rope;
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !selected || n_tokens == 0u || n_selected == 0u || n_head == 0u ||
        kv_lora_dim != 512u || qk_rope != 64u || cache_cap == 0u ||
        selected->bytes < (uint64_t)n_tokens * n_selected * sizeof(uint32_t) ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        qk_low->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * cache_elem ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * cache_elem ||
        lora_out->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float)) {
        fprintf(stderr, "ds4: glm attn lora selected: unsupported config "
                "(n_tok=%u selected=%u head=%u lora=%u rope=%u f16=%d)\n",
                n_tokens, n_selected, n_head, kv_lora_dim, qk_rope,
                (int)cache_f16);
        return 0;
    }
    const float scale = 1.0f / sqrtf((float)(qk_nope + qk_rope));
    dim3 grid((n_head + 7u) / 8u, n_tokens, 1);
    if (cache_f16) {
        glm_attention_lora_causal_kernel<__half, true><<<grid, 256>>>(
                (float *)lora_out->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const __half *)kv_lora_cache->ptr,
                (const __half *)k_rope_cache->ptr,
                (const uint32_t *)selected->ptr, cache_cap,
                n_tokens, 0u, n_selected, n_head, kv_lora_dim, qk_nope,
                qk_rope, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, scale);
    } else {
        glm_attention_lora_causal_kernel<float, true><<<grid, 256>>>(
                (float *)lora_out->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const float *)kv_lora_cache->ptr,
                (const float *)k_rope_cache->ptr,
                (const uint32_t *)selected->ptr, cache_cap,
                n_tokens, 0u, n_selected, n_head, kv_lora_dim, qk_nope,
                qk_rope, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, scale);
    }
    return cuda_ok(cudaGetLastError(), "glm attn lora selected launch");
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_indexed_batch_typed_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor\n");
    return 0;
}

__device__ __forceinline__ static float glm_q8_0_dot_row_dev(
        const char *row, const float *x, uint32_t n_cols) {
    float acc = 0.0f;
    const uint32_t nb = n_cols >> 5;
    for (uint32_t b = 0; b < nb; b++) {
        const char *blk = row + (uint64_t)b * 34u;
        const float d = __half2float(*(const __half *)blk);
        const int8_t *q = (const int8_t *)(blk + 2);
        float s = 0.0f;
        #pragma unroll 8
        for (uint32_t k = 0; k < 32u; k++) s += (float)q[k] * x[b * 32u + k];
        acc += d * s;
    }
    return acc;
}

template <typename CT>
__device__ __forceinline__ static float2 glm_cache_value_pair_dev(
        const CT *p) {
    return make_float2((float)p[0], (float)p[1]);
}

template <>
__device__ __forceinline__ float2 glm_cache_value_pair_dev<__half>(
        const __half *p) {
    return __half22float2(*(const __half2 *)p);
}

template <>
__device__ __forceinline__ float2 glm_cache_value_pair_dev<float>(
        const float *p) {
    return *(const float2 *)p;
}

/* Exact staged decode attention. The original fused kernel owns one block per
 * head, which leaves more than half of an L40S idle. These stages preserve the
 * fused kernel's arithmetic order for every score, softmax lane, lora output,
 * and value-projection row while exposing independent rows/dimensions as
 * separate blocks. */
template <typename CT, bool RANGE_TOK2 = false>
__global__ static void glm_attention_decode_weights_staged_kernel(
        float *weights,
        float *denom,
        const float *q,
        const float *qk_low,
        const CT *kv_lora_cache,
        const CT *k_rope_cache,
        const uint32_t *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        float scale,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        bool score_vec2) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = RANGE_TOK2 ? blockIdx.y : 0u;
    const uint32_t row_count = n_selected + (RANGE_TOK2 ? token : 0u);
    const uint32_t score_stride = n_selected + (RANGE_TOK2 ? 1u : 0u);
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (head >= n_head || row_count == 0u) return;
    const uint32_t qk_dim = qk_nope + qk_rope;
    extern __shared__ float glm_dec_stage_sh[];
    float *red = glm_dec_stage_sh;
    float *scores = glm_dec_stage_sh + 256u;
    const float *qh = q +
        ((uint64_t)token * n_head + head) * qk_dim;
    const float *low = qk_low +
        ((uint64_t)token * n_head + head) * kv_lora_dim;

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)qk_rope,
                    (int)n_ctx_orig, beta_fast, freq_base)));
        corr_dims[1] = fminf((float)qk_rope - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)qk_rope,
                    (int)n_ctx_orig, beta_slow, freq_base)));
    }

    float local_max = -FLT_MAX;
    for (uint32_t s = tid; s < row_count; s += nth) {
        const uint32_t row = RANGE_TOK2 ? s : selected[s];
        float score = -FLT_MAX;
        if (row < cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            if (score_vec2) {
                for (uint32_t j = 0; j < kv_lora_dim; j += 2u) {
                    const float2 x = *(const float2 *)(low + j);
                    const float2 y = glm_cache_value_pair_dev(
                            kv_lora_cache + lora_base + j);
                    dotv += x.x * y.x;
                    dotv += x.y * y.y;
                }
            } else {
                for (uint32_t j = 0; j < kv_lora_dim; j++) {
                    dotv += low[j] * (float)kv_lora_cache[lora_base + j];
                }
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float2 y = glm_cache_rope_pair_f16_dev<CT>(
                        k_rope_cache, rope_base, r, row, qk_rope,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        corr_dims);
                dotv += qh[qk_nope + r] * y.x +
                        qh[qk_nope + r + 1u] * y.y;
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    red[tid] = local_max;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }
    const float max_score = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t s = tid; s < row_count; s += nth) {
        const float w = expf(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    const uint64_t head_index = (uint64_t)token * n_head + head;
    if (tid == 0u) denom[head_index] = fmaxf(red[0], 1.0e-20f);
    float *head_weights = weights + head_index * score_stride;
    for (uint32_t s = tid; s < row_count; s += nth) {
        head_weights[s] = scores[s];
    }
}

template <typename CT, bool RANGE_TOK2 = false>
__global__ static void glm_attention_decode_lora_staged_kernel(
        float *lora_sum,
        const float *scores,
        const float *denom,
        const CT *kv_lora_cache,
        const uint32_t *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim) {
    const uint32_t head = blockIdx.y;
    const uint32_t token = RANGE_TOK2 ? blockIdx.z : 0u;
    const uint32_t row_count = n_selected + (RANGE_TOK2 ? token : 0u);
    const uint32_t score_stride = n_selected + (RANGE_TOK2 ? 1u : 0u);
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = pair * 2u;
    if (head >= n_head || j >= kv_lora_dim) return;
    const uint64_t head_index = (uint64_t)token * n_head + head;
    const float *head_scores = scores + head_index * score_stride;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t s = 0; s < row_count; s++) {
        const uint32_t row = RANGE_TOK2 ? s : selected[s];
        if (row < cache_cap) {
            const float2 v = glm_cache_value_pair_dev(
                    kv_lora_cache + (uint64_t)row * kv_lora_dim + j);
            const float w = head_scores[s];
            acc0 += w * v.x;
            acc1 += w * v.y;
        }
    }
    float *out = lora_sum + head_index * kv_lora_dim + j;
    out[0] = acc0 / denom[head_index];
    out[1] = acc1 / denom[head_index];
}

template <bool TOK2 = false>
__global__ static void glm_attention_decode_value_staged_kernel(
        float *heads,
        const float *lora_sum,
        const char *value_weight,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint32_t value_row_bytes) {
    const uint32_t head = blockIdx.y;
    const uint32_t token = TOK2 ? blockIdx.z : 0u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t nwarps = blockDim.x >> 5;
    const uint32_t out_warp = blockIdx.x * nwarps + warp;
    const uint32_t total_warps = gridDim.x * nwarps;
    if (head >= n_head) return;
    const float *low = lora_sum +
        ((uint64_t)token * n_head + head) * kv_lora_dim;
    float *out = heads +
        ((uint64_t)token * n_head + head) * value_dim;
    const uint32_t nblk = kv_lora_dim >> 5;
    for (uint32_t d = out_warp; d < value_dim; d += total_warps) {
        const char *row = value_weight +
            ((uint64_t)head * value_dim + d) * value_row_bytes;
        float acc = 0.0f;
        for (uint32_t blk = lane >> 1; blk < nblk; blk += 16u) {
            const char *b = row + (uint64_t)blk * 34u;
            const float dscale = __half2float(*(const __half *)b);
            const int8_t *q = (const int8_t *)(b + 2) + (lane & 1u) * 16u;
            const float *xs = low + blk * 32u + (lane & 1u) * 16u;
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; k++) s += (float)q[k] * xs[k];
            acc += dscale * s;
        }
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0u) out[d] = acc;
    }
}

/* Single-token indexed MLA decode attention, one block per head.
 * Fuses score (qk_low . kv_lora + q_rope . rope(k_rope@row)), softmax over
 * the indexer-selected rows, the weighted kv_lora sum, and the per-head
 * value projection (q8_0). Dynamic shared: red[256] + scores[n_selected] +
 * lora_sum[kv_lora_dim]. */
template <typename CT, bool RANGE_TOK2 = false>
__global__ static void glm_attention_indexed_decode_kernel(
        float *heads,
        const float *q,
        const float *qk_low,
        const CT *kv_lora_cache,
        const CT *k_rope_cache,
        const char *value_weight,
        const uint32_t *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t value_row_bytes,
        bool lora_vec2,
        bool score_vec2,
        float scale,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = RANGE_TOK2 ? blockIdx.y : 0u;
    const uint32_t row_count = n_selected + (RANGE_TOK2 ? token : 0u);
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (head >= n_head || row_count == 0u) return;
    const uint32_t qk_dim = qk_nope + qk_rope;

    extern __shared__ float glm_dec_sh[];
    float *red = glm_dec_sh;
    float *scores = glm_dec_sh + 256u;
    float *lora_sum = scores + row_count;

    const float *qh = q +
        ((uint64_t)token * n_head + head) * qk_dim;
    const float *low = qk_low +
        ((uint64_t)token * n_head + head) * kv_lora_dim;

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)qk_rope,
                    (int)n_ctx_orig, beta_fast, freq_base)));
        corr_dims[1] = fminf((float)qk_rope - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)qk_rope,
                    (int)n_ctx_orig, beta_slow, freq_base)));
    }

    float local_max = -FLT_MAX;
    for (uint32_t s = tid; s < row_count; s += nth) {
        const uint32_t row = RANGE_TOK2 ? s : selected[s];
        float score = -FLT_MAX;
        if (row < cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            if (score_vec2) {
                for (uint32_t j = 0; j < kv_lora_dim; j += 2u) {
                    const float2 x = *(const float2 *)(low + j);
                    const float2 y = glm_cache_value_pair_dev(
                            kv_lora_cache + lora_base + j);
                    dotv += x.x * y.x;
                    dotv += x.y * y.y;
                }
            } else {
                for (uint32_t j = 0; j < kv_lora_dim; j++) {
                    dotv += low[j] * (float)kv_lora_cache[lora_base + j];
                }
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float2 y = glm_cache_rope_pair_f16_dev<CT>(
                        k_rope_cache, rope_base, r, row, qk_rope,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        corr_dims);
                dotv += qh[qk_nope + r] * y.x + qh[qk_nope + r + 1u] * y.y;
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    red[tid] = local_max;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }
    const float max_score = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t s = tid; s < row_count; s += nth) {
        const float w = expf(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    const float denom = fmaxf(red[0], 1.0e-20f);
    __syncthreads();

    if (lora_vec2) {
        for (uint32_t j = tid * 2u; j < kv_lora_dim; j += nth * 2u) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint32_t s = 0; s < row_count; s++) {
                const uint32_t row = RANGE_TOK2 ? s : selected[s];
                if (row < cache_cap) {
                    const float2 v = glm_cache_value_pair_dev(
                            kv_lora_cache + (uint64_t)row * kv_lora_dim + j);
                    const float w = scores[s];
                    acc0 += w * v.x;
                    acc1 += w * v.y;
                }
            }
            lora_sum[j] = acc0 / denom;
            lora_sum[j + 1u] = acc1 / denom;
        }
    } else {
        for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < row_count; s++) {
                const uint32_t row = RANGE_TOK2 ? s : selected[s];
                if (row < cache_cap) {
                    acc += scores[s] *
                           (float)kv_lora_cache[(uint64_t)row * kv_lora_dim + j];
                }
            }
            lora_sum[j] = acc / denom;
        }
    }
    __syncthreads();

    float *out = heads +
        ((uint64_t)token * n_head + head) * value_dim;
    /* Warp-cooperative value projection: one warp per output dim, two
     * lanes per q8_0 block (16 cols each). */
    const uint32_t nwarps = nth >> 5;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t nblk = kv_lora_dim >> 5;
    for (uint32_t d = warp; d < value_dim; d += nwarps) {
        const char *row = value_weight +
            ((uint64_t)head * value_dim + d) * value_row_bytes;
        float acc = 0.0f;
        for (uint32_t blk = lane >> 1; blk < nblk; blk += 16u) {
            const char *b = row + (uint64_t)blk * 34u;
            const float dscale = __half2float(*(const __half *)b);
            const int8_t *q = (const int8_t *)(b + 2) + (lane & 1u) * 16u;
            const float *xs = lora_sum + blk * 32u + (lane & 1u) * 16u;
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; k++) s += (float)q[k] * xs[k];
            acc += dscale * s;
        }
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0u) out[d] = acc;
    }
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!heads || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !model_map || !selected ||
        n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 || (kv_lora_dim & 31u) != 0u ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0u ||
        value_dim == 0) {
        return 0;
    }
    if (value_weight_type != 8u) {   /* DS4_TENSOR_Q8_0 */
        fprintf(stderr,
            "ds4: glm indexed decode attention: unsupported value type %u\n",
            value_weight_type);
        return 0;
    }
    const uint64_t value_row_bytes = ((uint64_t)kv_lora_dim / 32u) * 34u;
    const uint64_t value_weight_bytes =
        (uint64_t)n_head * value_dim * value_row_bytes;
    if (value_weight_offset > model_size ||
        value_weight_bytes > model_size - value_weight_offset) {
        return 0;
    }
    const uint64_t cache_elem = cache_f16 ? 2u : 4u;
    if (heads->bytes < (uint64_t)n_head * value_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * qk_dim * sizeof(float) ||
        qk_low->bytes < (uint64_t)n_head * kv_lora_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * cache_elem ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * cache_elem ||
        selected->bytes < (uint64_t)n_selected * sizeof(uint32_t)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const char *vw = cuda_resolve_weight_ptr(model_map, value_weight_offset,
            value_weight_bytes, logical_tier, "glm_v_b_decode");
    if (!vw) return 0;
    const float scale = 1.0f / sqrtf((float)qk_dim);
    const bool score_vec2 = getenv("DS4_GLM_ATTN_NO_SCORE_VEC2") == NULL;
    const bool range_tok2 =
        g_glm_mtp_verify_mode &&
        getenv("DS4_GLM_MTP_NO_ATTN_TOK2") == NULL &&
        n_selected < cache_cap &&
        heads->bytes >= 2u * (uint64_t)n_head * value_dim * sizeof(float) &&
        q->bytes >= 2u * (uint64_t)n_head * qk_dim * sizeof(float) &&
        qk_low->bytes >=
            2u * (uint64_t)n_head * kv_lora_dim * sizeof(float);
    if (range_tok2 && n_selected < 512u) {
        const bool lora_vec2 =
            getenv("DS4_GLM_ATTN_NO_LORA_VEC2") == NULL;
        const uint32_t shmem =
            (256u + n_selected + 1u + kv_lora_dim) *
            (uint32_t)sizeof(float);
        const dim3 grid(n_head, 2u, 1u);
        if (cache_f16) {
            glm_attention_indexed_decode_kernel<__half, true>
                    <<<grid, 256, shmem>>>(
                    (float *)heads->ptr, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const __half *)kv_lora_cache->ptr,
                    (const __half *)k_rope_cache->ptr,
                    vw, (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, value_dim,
                    (uint32_t)value_row_bytes,
                    lora_vec2, score_vec2, scale, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow);
        } else {
            glm_attention_indexed_decode_kernel<float, true>
                    <<<grid, 256, shmem>>>(
                    (float *)heads->ptr, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const float *)kv_lora_cache->ptr,
                    (const float *)k_rope_cache->ptr,
                    vw, (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, value_dim,
                    (uint32_t)value_row_bytes,
                    lora_vec2, score_vec2, scale, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow);
        }
        return cuda_ok(cudaGetLastError(),
                       "glm indexed decode attention tok2 range");
    }
    if (n_selected >= 512u &&
        getenv("DS4_GLM_ATTN_NO_STAGED_DECODE") == NULL) {
        const uint32_t token_count = range_tok2 ? 2u : 1u;
        const uint32_t score_stride = n_selected + (range_tok2 ? 1u : 0u);
        const uint64_t head_count = (uint64_t)token_count * n_head;
        if (head_count > UINT64_MAX / score_stride ||
            head_count > UINT64_MAX / kv_lora_dim) {
            return 0;
        }
        const uint64_t score_count = head_count * score_stride;
        const uint64_t lora_count = head_count * kv_lora_dim;
        if (score_count > UINT64_MAX - head_count - lora_count ||
            score_count + head_count + lora_count >
                UINT64_MAX / sizeof(float)) {
            return 0;
        }
        const uint64_t scratch_bytes =
            (score_count + head_count + lora_count) * sizeof(float);
        float *scratch = (float *)cuda_tmp_alloc_on(
                ds4_tensor_device_idx(heads), scratch_bytes,
                "glm staged decode attention");
        if (!scratch) return 0;
        float *softmax_denom = scratch + score_count;
        float *lora_sum = softmax_denom + head_count;
        const uint32_t weight_shmem =
            (256u + score_stride) * (uint32_t)sizeof(float);
        const dim3 weight_grid(n_head, token_count, 1u);
        if (cache_f16 && range_tok2) {
            glm_attention_decode_weights_staged_kernel<__half, true>
                    <<<weight_grid, 256, weight_shmem>>>(
                    scratch, softmax_denom, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const __half *)kv_lora_cache->ptr,
                    (const __half *)k_rope_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, scale, n_ctx_orig, freq_base,
                    freq_scale, ext_factor, attn_factor, beta_fast,
                    beta_slow, score_vec2);
        } else if (cache_f16) {
            glm_attention_decode_weights_staged_kernel<__half>
                    <<<weight_grid, 256, weight_shmem>>>(
                    scratch, softmax_denom, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const __half *)kv_lora_cache->ptr,
                    (const __half *)k_rope_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, scale, n_ctx_orig, freq_base,
                    freq_scale, ext_factor, attn_factor, beta_fast,
                    beta_slow, score_vec2);
        } else if (range_tok2) {
            glm_attention_decode_weights_staged_kernel<float, true>
                    <<<weight_grid, 256, weight_shmem>>>(
                    scratch, softmax_denom, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const float *)kv_lora_cache->ptr,
                    (const float *)k_rope_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, scale, n_ctx_orig, freq_base,
                    freq_scale, ext_factor, attn_factor, beta_fast,
                    beta_slow, score_vec2);
        } else {
            glm_attention_decode_weights_staged_kernel<float>
                    <<<weight_grid, 256, weight_shmem>>>(
                    scratch, softmax_denom, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const float *)kv_lora_cache->ptr,
                    (const float *)k_rope_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, scale, n_ctx_orig, freq_base,
                    freq_scale, ext_factor, attn_factor, beta_fast,
                    beta_slow, score_vec2);
        }
        if (!cuda_ok(cudaGetLastError(),
                     "glm staged decode weights launch")) {
            return 0;
        }
        dim3 lora_grid((kv_lora_dim / 2u + 63u) / 64u,
                       n_head, token_count);
        if (cache_f16 && range_tok2) {
            glm_attention_decode_lora_staged_kernel<__half, true>
                    <<<lora_grid, 64>>>(
                    lora_sum, scratch, softmax_denom,
                    (const __half *)kv_lora_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim);
        } else if (cache_f16) {
            glm_attention_decode_lora_staged_kernel<__half>
                    <<<lora_grid, 64>>>(
                    lora_sum, scratch, softmax_denom,
                    (const __half *)kv_lora_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim);
        } else if (range_tok2) {
            glm_attention_decode_lora_staged_kernel<float, true>
                    <<<lora_grid, 64>>>(
                    lora_sum, scratch, softmax_denom,
                    (const float *)kv_lora_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim);
        } else {
            glm_attention_decode_lora_staged_kernel<float>
                    <<<lora_grid, 64>>>(
                    lora_sum, scratch, softmax_denom,
                    (const float *)kv_lora_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim);
        }
        if (!cuda_ok(cudaGetLastError(),
                     "glm staged decode lora launch")) {
            return 0;
        }
        dim3 value_grid((value_dim + 127u) / 128u,
                        n_head, token_count);
        if (range_tok2) {
            glm_attention_decode_value_staged_kernel<true>
                    <<<value_grid, 128>>>(
                    (float *)heads->ptr, lora_sum, vw,
                    n_head, kv_lora_dim, value_dim,
                    (uint32_t)value_row_bytes);
        } else {
            glm_attention_decode_value_staged_kernel<false>
                    <<<value_grid, 128>>>(
                    (float *)heads->ptr, lora_sum, vw,
                    n_head, kv_lora_dim, value_dim,
                    (uint32_t)value_row_bytes);
        }
        return cuda_ok(cudaGetLastError(),
                       "glm staged decode value launch");
    }
    const bool lora_vec2 = getenv("DS4_GLM_ATTN_NO_LORA_VEC2") == NULL;
    const uint32_t shmem =
        (256u + n_selected + kv_lora_dim) * (uint32_t)sizeof(float);
    if (cache_f16) {
        glm_attention_indexed_decode_kernel<__half><<<n_head, 256, shmem>>>(
                (float *)heads->ptr, (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const __half *)kv_lora_cache->ptr,
                (const __half *)k_rope_cache->ptr,
                vw, (const uint32_t *)selected->ptr,
                n_selected, cache_cap, n_head, kv_lora_dim,
                qk_nope, qk_rope, value_dim, (uint32_t)value_row_bytes,
                lora_vec2, score_vec2, scale, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    } else {
        glm_attention_indexed_decode_kernel<float><<<n_head, 256, shmem>>>(
                (float *)heads->ptr, (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const float *)kv_lora_cache->ptr,
                (const float *)k_rope_cache->ptr,
                vw, (const uint32_t *)selected->ptr,
                n_selected, cache_cap, n_head, kv_lora_dim,
                qk_nope, qk_rope, value_dim, (uint32_t)value_row_bytes,
                lora_vec2, score_vec2, scale, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }
    return cuda_ok(cudaGetLastError(), "glm indexed decode attention");
}

extern "C" int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_build_kv_cache_flash_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_build_kv_cache_tensor\n");
    return 0;
}

__global__ static void glm_fill_selected_range_batch_kernel(
        uint32_t *selected,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t pad_row) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n_tokens * n_selected;
    if (gid >= total || n_selected == 0u) return;
    const uint32_t token = gid / n_selected;
    const uint32_t slot = gid - token * n_selected;
    const uint32_t visible = pos0 + token + 1u;
    selected[gid] = slot < visible ? slot : pad_row;
}

extern "C" int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row) {
    if (!selected || n_tokens == 0 || n_selected == 0 ||
        selected->bytes < (uint64_t)n_tokens * n_selected * sizeof(uint32_t)) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_tokens * n_selected;
    glm_fill_selected_range_batch_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
            (uint32_t *)selected->ptr, n_tokens, pos0, n_selected, pad_row);
    return cuda_ok(cudaGetLastError(), "glm fill selected batch launch");
}

__global__ static void glm_fill_selected_range_kernel(
        uint32_t *selected, uint32_t n_selected) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_selected) selected[gid] = gid;
}

extern "C" int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected) {
    if (!selected || n_selected == 0 ||
        selected->bytes < (uint64_t)n_selected * sizeof(uint32_t)) {
        return 0;
    }
    glm_fill_selected_range_kernel<<<(n_selected + 255) / 256, 256>>>(
            (uint32_t *)selected->ptr, n_selected);
    return cuda_ok(cudaGetLastError(), "glm fill selected launch");
}

static int glm_rope_tail_offset_launch(
        ds4_gpu_tensor *x,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t rot_offset, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, const char *what);

extern "C" int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow) {
    return glm_rope_tail_offset_launch(x, n_tokens, n_head, head_dim,
            rot_dim, 0, pos0, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, "glm indexer rope tail");
}

template <typename CT>
__global__ static void glm_indexer_scores_f32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const CT *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        float scale,
        bool causal) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (row >= n_rows || token >= n_tokens || tid >= 128u) return;
    if (causal && row >= min(n_rows, pos0 + token + 1u)) {
        if (tid == 0u) scores[(uint64_t)token * n_rows + row] = -INFINITY;
        return;
    }

    __shared__ float partial[128];
    float total = 0.0f;
    const CT *krow = indexer_key_cache + (uint64_t)row * head_dim;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q +
            ((uint64_t)token * n_head + h) * head_dim;
        float dot = tid < head_dim ? qh[tid] * (float)krow[tid] : 0.0f;
        partial[tid] = dot;
        __syncthreads();
        for (uint32_t stride = 64u; stride > 0u; stride >>= 1u) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            total += fmaxf(partial[0], 0.0f) *
                weights[(uint64_t)token * n_head + h];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        scores[(uint64_t)token * n_rows + row] = total * scale;
    }
}

/* 16-token x 128-row indexer tile. Q and cached K are staged as fp16,
 * matching the model's compact-cache precision; each head's MMA result and
 * the weighted head reduction remain fp32. */
template <typename CT>
__global__ static void glm_indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const CT *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        float scale,
        bool causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t row0 = blockIdx.x * 128u;
    const uint32_t token0 = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(token0 + 16u, n_tokens);
        const uint32_t max_visible = last_token > token0
            ? min(pos0 + last_token, n_rows) : 0u;
        if (row0 >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t token = token0 + (i >> 7u);
                const uint32_t row = row0 + (i & 127u);
                if (token < n_tokens && row < n_rows) {
                    scores[(uint64_t)token * n_rows + row] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half q_sh[16 * 128];
    __shared__ __half k_sh[128 * 128];
    __shared__ float dot_sh[8 * 16 * 16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t r = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t row = row0 + r;
        const float v = row < n_rows
            ? (float)indexer_key_cache[(uint64_t)row * head_dim + d]
            : 0.0f;
        k_sh[d + r * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t tr = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = token0 + tr;
            const float v = token < n_tokens
                ? q[((uint64_t)token * n_head + h) * head_dim + d]
                : 0.0f;
            q_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __half, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dot_frag;
        wmma::fill_fragment(dot_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(q_frag, q_sh + k0, 128);
            wmma::load_matrix_sync(k_frag,
                                   k_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(dot_frag, q_frag, k_frag, dot_frag);
        }
        wmma::store_matrix_sync(dot_sh + warp * 16u * 16u,
                                dot_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token = token0 + (local0 >> 4u);
        const float w = token < n_tokens
            ? weights[(uint64_t)token * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u;
             i += 256u, slot++) {
            const uint32_t row = row0 + (i >> 8u) * 16u + (i & 15u);
            if (token < n_tokens && row < n_rows) {
                acc[slot] += fmaxf(dot_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u;
         i += 256u, slot++) {
        const uint32_t local = i & 255u;
        const uint32_t token = token0 + (local >> 4u);
        const uint32_t row = row0 + (i >> 8u) * 16u + (local & 15u);
        if (token < n_tokens && row < n_rows) {
            float out = acc[slot] * scale;
            if (causal && row >= pos0 + token + 1u) out = -INFINITY;
            scores[(uint64_t)token * n_rows + row] = out;
        }
    }
#endif
}

static int glm_indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16,
        bool                  causal) {
    const uint64_t cache_elem = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!scores || !q || !weights || !indexer_key_cache || n_rows == 0u ||
        n_tokens == 0u || n_head == 0u || head_dim != 128u ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        indexer_key_cache->bytes < (uint64_t)n_rows * head_dim * cache_elem ||
        scores->bytes < (uint64_t)n_tokens * n_rows * sizeof(float)) {
        return 0;
    }
    if (!g_quality_mode) {
        dim3 grid((n_rows + 127u) / 128u,
                  (n_tokens + 15u) / 16u, 1);
        if (cache_f16) {
            glm_indexer_scores_wmma128_kernel<__half><<<grid, 256>>>(
                    (float *)scores->ptr, (const float *)q->ptr,
                    (const float *)weights->ptr,
                    (const __half *)indexer_key_cache->ptr,
                    n_rows, n_tokens, pos0, n_head, head_dim, scale, causal);
        } else {
            glm_indexer_scores_wmma128_kernel<float><<<grid, 256>>>(
                    (float *)scores->ptr, (const float *)q->ptr,
                    (const float *)weights->ptr,
                    (const float *)indexer_key_cache->ptr,
                    n_rows, n_tokens, pos0, n_head, head_dim, scale, causal);
        }
        return cuda_ok(cudaGetLastError(), "glm indexer scores wmma launch");
    }

    dim3 grid(n_rows, n_tokens, 1);
    if (cache_f16) {
        glm_indexer_scores_f32_kernel<__half><<<grid, 128>>>(
                (float *)scores->ptr, (const float *)q->ptr,
                (const float *)weights->ptr,
                (const __half *)indexer_key_cache->ptr,
                n_rows, n_tokens, pos0, n_head, head_dim, scale, causal);
    } else {
        glm_indexer_scores_f32_kernel<float><<<grid, 128>>>(
                (float *)scores->ptr, (const float *)q->ptr,
                (const float *)weights->ptr,
                (const float *)indexer_key_cache->ptr,
                n_rows, n_tokens, pos0, n_head, head_dim, scale, causal);
    }
    return cuda_ok(cudaGetLastError(), "glm indexer scores f32 launch");
}

extern "C" int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    return glm_indexer_scores_launch(scores, q, weights, indexer_key_cache,
                                     n_rows, 1u, 0u, n_head, head_dim,
                                     scale, cache_f16, false);
}

extern "C" int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    return glm_indexer_scores_launch(scores, q, weights, indexer_key_cache,
                                     n_rows, n_tokens, pos0, n_head, head_dim,
                                     scale, cache_f16, true);
}

extern "C" int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_k_b_project_typed_tensor\n");
    return 0;
}

__global__ static void glm_kv_lora_rms_norm_kernel(
        float *dst,
        const float *src,
        const float *w,
        uint32_t n_tokens,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const float *x = src + (uint64_t)row * kv_raw_dim;
    float *out = dst + (uint64_t)row * kv_lora_dim;
    __shared__ float scratch[256];
    float ss = 0.0f;
    for (uint32_t i = tid; i < kv_lora_dim; i += nth) {
        const float v = x[i];
        ss += v * v;
    }
    scratch[tid] = ss;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / (float)kv_lora_dim + eps);
    for (uint32_t i = tid; i < kv_lora_dim; i += nth) {
        out[i] = x[i] * inv * w[i];
    }
}

extern "C" int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps) {
    if (!dst || !src || !model_map || n_tokens == 0 ||
        kv_lora_dim == 0 || kv_lora_dim > kv_raw_dim) {
        return 0;
    }
    const uint64_t wb = (uint64_t)kv_lora_dim * sizeof(float);
    if (weight_offset > model_size || wb > model_size - weight_offset ||
        src->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float) ||
        dst->bytes < (uint64_t)n_tokens * kv_lora_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *w = (const float *)cuda_resolve_weight_ptr(
            model_map, weight_offset, wb, logical_tier, "glm_kv_lora_norm");
    if (!w) return 0;
    glm_kv_lora_rms_norm_kernel<<<n_tokens, 256>>>(
            (float *)dst->ptr, (const float *)src->ptr, w,
            n_tokens, kv_raw_dim, kv_lora_dim, eps);
    return cuda_ok(cudaGetLastError(), "glm kv lora rms norm launch");
}



__global__ static void glm_qk_lowrank_q8_0_batch_kernel(
        float *qk_low,
        const char *weight,
        const float *q,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_dim,
        uint64_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const float *qh = q + (uint64_t)token * n_head * qk_dim +
                      (uint64_t)head * qk_dim;
    float *out = qk_low + (uint64_t)token * n_head * kv_lora_dim +
                 (uint64_t)head * kv_lora_dim;
    for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
        const char *row = weight +
            ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        out[j] = glm_q8_0_dot_row_dev(row, qh, qk_nope);
    }
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    if (!qk_low || !q || !model_map || n_tokens == 0 || n_head == 0 ||
        kv_lora_dim == 0 || qk_nope == 0 || (qk_nope & 31u) != 0u) {
        return 0;
    }
    if (weight_type != 8u) {
        fprintf(stderr, "ds4: glm qk_lowrank: unsupported type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)qk_nope / 32u) * 34u;
    const uint64_t wbytes = (uint64_t)n_head * kv_lora_dim * row_bytes;
    if (weight_offset > model_size || wbytes > model_size - weight_offset ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        qk_low->bytes <
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const char *w = (const char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, wbytes, logical_tier, "glm_k_b_qk");
    if (!w) return 0;
    if (g_q8_dequant_gemm_enabled && g_cublas_ready && n_tokens >= 128u) {
        /* Per-head strided-batched GEMM over a dequantized k_b: the
         * per-(token,head) warp kernel was ~65ms/layer at 820 tokens.
         * Scratch (executing device): [w_f16][q_f16][out_f32]. */
        const uint64_t wh_bytes =
            (uint64_t)n_head * kv_lora_dim * qk_nope * sizeof(__half);
        const uint64_t xh_off = (wh_bytes + 255u) & ~255ull;
        const uint64_t xh_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(__half);
        const uint64_t oo_off = (xh_off + xh_bytes + 255u) & ~255ull;
        const uint64_t oo_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        void *tmp = cuda_tmp_alloc_on(logical_tier, oo_off + oo_bytes,
                                      "glm qk_low gemm");
        if (tmp) {
            __half *wh = (__half *)tmp;
            __half *xh = (__half *)((char *)tmp + xh_off);
            float *oo = (float *)((char *)tmp + oo_off);
            const uint64_t total_blocks =
                (uint64_t)n_head * kv_lora_dim * (qk_nope / 32u);
            q8_0_dequant_f16_kernel<<<(unsigned)((total_blocks * 2u + 255u) / 256u), 256>>>(
                    wh, (const unsigned char *)w, total_blocks,
                    qk_nope / 32u, qk_nope);
            const uint64_t xn = (uint64_t)n_tokens * n_head * qk_dim;
            f32_to_f16_kernel<<<(xn + 255u) / 256u, 256>>>(
                    xh, (const float *)q->ptr, xn);
            if (cuda_ok(cudaGetLastError(), "glm qk_low gemm staging")) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmStridedBatchedEx(
                        cuda_cublas_for_tier(logical_tier),
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)kv_lora_dim, (int)n_tokens, (int)qk_nope,
                        &alpha,
                        wh, CUDA_R_16F, (int)qk_nope,
                        (long long)((uint64_t)kv_lora_dim * qk_nope),
                        xh, CUDA_R_16F, (int)(n_head * qk_dim),
                        (long long)qk_dim,
                        &beta,
                        oo, CUDA_R_32F, (int)(n_head * kv_lora_dim),
                        (long long)kv_lora_dim,
                        (int)n_head,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                if (st == CUBLAS_STATUS_SUCCESS &&
                    cuda_ok(cudaMemcpyAsync(qk_low->ptr, oo, oo_bytes,
                            cudaMemcpyDeviceToDevice, 0),
                            "glm qk_low gemm out copy")) {
                    return 1;
                }
                fprintf(stderr,
                        "ds4: glm qk_low gemm failed (status %d); native path\n",
                        (int)st);
            }
        }
    }
    dim3 grid(n_head, n_tokens, 1);
    glm_qk_lowrank_q8_0_batch_kernel<<<grid, 128>>>(
            (float *)qk_low->ptr, w, (const float *)q->ptr,
            n_tokens, n_head, kv_lora_dim, qk_nope, qk_dim, row_bytes);
    return cuda_ok(cudaGetLastError(), "glm qk lowrank batch launch");
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    return ds4_gpu_glm_qk_lowrank_typed_batch_tensor(qk_low, q, model_map,
                                                     model_size,
                                                     weight_offset,
                                                     weight_type, 1u, n_head,
                                                     kv_lora_dim, qk_nope,
                                                     qk_dim);
}

/* Fused decode-path QKV norm + compact-KV store, one block per
 * (token, part): part 0 rms-norms q into q_out, part 1 rms-norms
 * kv_raw[:kv_lora_dim] into the kv_lora ring, part 2 copies the
 * UNROTATED rope tail into the k_rope ring (roped at attention read). */
__global__ static void glm_qkv_norm_store_compact_kv_kernel(
        float *q_dst,
        const float *q_src,
        const float *q_w,
        uint32_t q_n,
        const float *kv_raw,
        const float *kv_w,
        char *kv_lora_cache,
        char *k_rope_cache,
        uint32_t pos0,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope,
        int cache_f16,
        float eps) {
    const uint32_t token = blockIdx.x;
    const uint32_t part = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const uint32_t pos = pos0 + token;

    if (part == 2u) {
        if (pos >= cache_cap) return;
        const float *src = kv_raw + (uint64_t)token * kv_raw_dim + kv_lora_dim;
        if (cache_f16) {
            __half *dst = (__half *)k_rope_cache + (uint64_t)pos * qk_rope;
            for (uint32_t i = tid; i < qk_rope; i += nth) {
                dst[i] = __float2half(src[i]);
            }
        } else {
            float *dst = (float *)k_rope_cache + (uint64_t)pos * qk_rope;
            for (uint32_t i = tid; i < qk_rope; i += nth) {
                dst[i] = src[i];
            }
        }
        return;
    }

    const bool kv_task = part != 0u;
    const uint32_t n = kv_task ? kv_lora_dim : q_n;
    const float *x = kv_task ? kv_raw + (uint64_t)token * kv_raw_dim
                             : q_src + (uint64_t)token * q_n;
    const float *w = kv_task ? kv_w : q_w;

    __shared__ float sh[32];
    float sumf = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = x[i];
        sumf += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sumf += __shfl_xor_sync(0xffffffffu, sumf, off);
    }
    if ((tid & 31u) == 0u) sh[tid >> 5] = sumf;
    __syncthreads();
    if (tid < 32u) {
        sumf = (tid < (nth + 31u) / 32u) ? sh[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sumf += __shfl_xor_sync(0xffffffffu, sumf, off);
        }
        if (tid == 0u) sh[0] = sumf;
    }
    __syncthreads();
    const float scale = rsqrtf(sh[0] / (float)n + eps);

    if (!kv_task) {
        float *y = q_dst + (uint64_t)token * q_n;
        for (uint32_t i = tid; i < n; i += nth) {
            y[i] = (x[i] * scale) * w[i];
        }
        return;
    }

    if (pos >= cache_cap) return;
    if (cache_f16) {
        __half *dst = (__half *)kv_lora_cache + (uint64_t)pos * kv_lora_dim;
        for (uint32_t i = tid; i < kv_lora_dim; i += nth) {
            dst[i] = __float2half((x[i] * scale) * w[i]);
        }
    } else {
        float *dst = (float *)kv_lora_cache + (uint64_t)pos * kv_lora_dim;
        for (uint32_t i = tid; i < kv_lora_dim; i += nth) {
            dst[i] = (x[i] * scale) * w[i];
        }
    }
}

extern "C" int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps) {
    if (!q_out || !q || !kv_lora_cache || !k_rope_cache || !kv_raw ||
        !model_map || n_tokens == 0 || q_n == 0 || kv_lora_dim == 0 ||
        qk_rope == 0 || kv_raw_dim < kv_lora_dim + qk_rope ||
        q->bytes < (uint64_t)n_tokens * q_n * sizeof(float) ||
        q_out->bytes < (uint64_t)n_tokens * q_n * sizeof(float) ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float)) {
        return 0;
    }
    if (q_weight_offset > model_size ||
        (uint64_t)q_n * sizeof(float) > model_size - q_weight_offset ||
        kv_weight_offset > model_size ||
        (uint64_t)kv_lora_dim * sizeof(float) > model_size - kv_weight_offset) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *q_w = (const float *)cuda_resolve_weight_ptr(
            model_map, q_weight_offset, (uint64_t)q_n * sizeof(float),
            logical_tier, "glm_q_norm");
    const float *kv_w = (const float *)cuda_resolve_weight_ptr(
            model_map, kv_weight_offset,
            (uint64_t)kv_lora_dim * sizeof(float),
            logical_tier, "glm_kv_norm");
    if (!q_w || !kv_w) return 0;
    dim3 grid(n_tokens, 3, 1);
    glm_qkv_norm_store_compact_kv_kernel<<<grid, 256>>>(
            (float *)q_out->ptr,
            (const float *)q->ptr,
            q_w,
            q_n,
            (const float *)kv_raw->ptr,
            kv_w,
            (char *)kv_lora_cache->ptr,
            (char *)k_rope_cache->ptr,
            pos0, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope,
            cache_f16 ? 1 : 0, eps);
    return cuda_ok(cudaGetLastError(), "glm qkv norm store compact kv");
}

/* In-place interleaved-pair yarn rope on a [n_tokens][n_head][head_dim]
 * f32 tensor, rotating rot_dim dims starting at rot_offset. Shared by the
 * attention q tail (offset = head_dim - rot_dim) and the DSA indexer
 * (offset = 0). Grid (n_head, n_tokens). */
__global__ static void glm_rope_tail_offset_kernel(
        float *x,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t rot_offset,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (rot_dim == 0u || (rot_dim & 1u) != 0u ||
        rot_offset > head_dim || rot_dim > head_dim - rot_offset) return;

    const uint32_t pos = pos0 + token;
    float *row = x + ((uint64_t)token * n_head + head) * head_dim + rot_offset;

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)rot_dim,
                    (int)n_ctx_orig, beta_fast, freq_base)));
        corr_dims[1] = fminf((float)rot_dim - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)rot_dim,
                    (int)n_ctx_orig, beta_slow, freq_base)));
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)rot_dim;
    for (uint32_t i = tid * 2u; i < rot_dim; i += nth * 2u) {
        const float theta =
            theta_base * powf(freq_base, inv_ndims * (float)i);
        float ct, st;
        glm_rope_yarn_dev(theta, freq_scale, corr_dims, (int)i,
                          ext_factor, attn_factor, &ct, &st);
        const float x0 = row[i];
        const float x1 = row[i + 1u];
        row[i] = x0 * ct - x1 * st;
        row[i + 1u] = x0 * st + x1 * ct;
    }
}

static int glm_rope_tail_offset_launch(
        ds4_gpu_tensor *x,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t rot_offset, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, const char *what) {
    if (!x || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        rot_dim == 0 || (rot_dim & 1u) != 0u || rot_offset > head_dim ||
        rot_dim > head_dim - rot_offset ||
        x->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    dim3 grid(n_head, n_tokens, 1);
    glm_rope_tail_offset_kernel<<<grid, 32>>>(
            (float *)x->ptr, n_head, head_dim, rot_dim, rot_offset,
            pos0, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), what);
}

extern "C" int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow) {
    if (rot_dim > head_dim) return 0;
    return glm_rope_tail_offset_launch(x, n_tokens, n_head, head_dim,
            rot_dim, head_dim - rot_dim, pos0, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, "glm rope tail");
}

extern "C" int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor\n");
    return 0;
}

/* Scalar-correct GLM routed MoE (q2_K experts): per (token, slot) block
 * quantizes nothing - dots q2_K rows against a q8_K-quantized activation
 * staged in shared memory. Grid: (n_tokens, n_expert). Mid buffer holds
 * silu(gate)*up per slot; out accumulates expert_weight-scaled down rows.
 */
__global__ static void glm_routed_moe_batch_q2K_gateup_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t n_tokens,
        uint32_t mid_token_stride) {
    const uint32_t tok = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (tok >= n_tokens || slot >= n_expert) return;
    const int32_t expert = selected[(uint64_t)tok * n_expert + slot];
    if (expert < 0) return;
    const cuda_block_q8_K *xrow = xq + (uint64_t)tok * xq_blocks;
    float *mrow = mid + (uint64_t)tok * mid_token_stride +
                  (uint64_t)slot * expert_mid_dim;
    for (uint32_t r = threadIdx.x; r < expert_mid_dim; r += blockDim.x) {
        const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base +
            (uint64_t)expert * gate_expert_bytes + (uint64_t)r * gate_row_bytes);
        const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base +
            (uint64_t)expert * up_expert_bytes + (uint64_t)r * up_row_bytes);
        float g = 0.0f, u = 0.0f;
        for (uint32_t b = 0; b < xq_blocks; b++) {
            g += dev_dot_q2_K_q8_K_block(gr + b, xrow + b);
            u += dev_dot_q2_K_q8_K_block(ur + b, xrow + b);
        }
        mrow[r] = (g / (1.0f + expf(-g))) * u;   /* silu(g)*u */
    }
}

__global__ static void glm_routed_moe_batch_q2K_down_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.y;
    if (tok >= n_tokens) return;
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < n_expert; slot++) {
        const int32_t expert = selected[(uint64_t)tok * n_expert + slot];
        if (expert < 0) continue;
        const float w = weights[(uint64_t)tok * n_expert + slot];
        const cuda_block_q2_K *dr = (const cuda_block_q2_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)r * down_row_bytes);
        const cuda_block_q8_K *mrow = midq +
            ((uint64_t)tok * n_expert + slot) * midq_blocks;
        float s = 0.0f;
        for (uint32_t b = 0; b < midq_blocks; b++) {
            s += dev_dot_q2_K_q8_K_block(dr + b, mrow + b);
        }
        acc += w * s;
    }
    out[(uint64_t)tok * out_dim + r] = acc;
}

/* Warp-per-row routed MoE (q2_K x q8_K). Each block stages the token's
 * q8_K activation row in shared memory; one warp produces one mid row
 * (gate dot + up dot + silu*mul fused). Grid:
 * (expert_mid_dim/warps, n_expert, n_tokens). */
__global__ static void glm_routed_moe_gateup_warp_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.z;
    const uint32_t slot = blockIdx.y;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (tok >= n_tokens || slot >= n_expert) return;
    const int32_t expert = selected[(uint64_t)tok * n_expert + slot];

    extern __shared__ unsigned int glm_moe_sh_u32[];
    {
        const unsigned int *src =
            (const unsigned int *)(xq + (uint64_t)tok * xq_blocks);
        const uint32_t words = xq_blocks * (uint32_t)sizeof(cuda_block_q8_K) / 4u;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
            glm_moe_sh_u32[i] = src[i];
        }
    }
    __syncthreads();
    if (expert < 0) return;
    const cuda_block_q8_K *xrow = (const cuda_block_q8_K *)glm_moe_sh_u32;

    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= expert_mid_dim) return;
    const char *gr = gate_base + (uint64_t)expert * gate_expert_bytes +
                     (uint64_t)r * gate_row_bytes;
    const char *ur = up_base + (uint64_t)expert * up_expert_bytes +
                     (uint64_t)r * up_row_bytes;
    float g = 0.0f, u = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        g += dev_dot_q2_K_q8_K_block(
                (const cuda_block_q2_K *)(gr + (uint64_t)b * 84u), xrow + b);
        u += dev_dot_q2_K_q8_K_block(
                (const cuda_block_q2_K *)(ur + (uint64_t)b * 84u), xrow + b);
    }
    for (int off = 16; off > 0; off >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, off);
        u += __shfl_down_sync(0xffffffffu, u, off);
    }
    if (lane == 0u) {
        mid[((uint64_t)tok * n_expert + slot) * expert_mid_dim + r] =
            (g / (1.0f + expf(-g))) * u;
    }
}

/* Exact two-token gate/up with adjacent-token expert reuse. Token 0 owns an
 * expert present in both rows; token 1 only launches work for unmatched
 * experts. Each token keeps the native lane assignment and warp reduction. */
__global__ static void glm_routed_moe_gateup_tok2_reuse_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert) {
    const uint32_t owner = blockIdx.y;
    const uint32_t tok = owner / n_expert;
    const uint32_t slot = owner - tok * n_expert;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (tok >= 2u || slot >= n_expert) return;

    const int32_t expert = selected[owner];
    if (expert < 0) return;
    int32_t mate_slot = -1;
    for (uint32_t s = 0; s < n_expert; s++) {
        if (selected[(uint64_t)(1u - tok) * n_expert + s] == expert) {
            mate_slot = (int32_t)s;
            break;
        }
    }
    if (tok == 1u && mate_slot >= 0) return;

    const uint32_t np = tok == 0u && mate_slot >= 0 ? 2u : 1u;
    const uint32_t pair0 = owner;
    const uint32_t pair1 = n_expert + (uint32_t)mate_slot;
    extern __shared__ unsigned int glm_moe_tok2_sh_u32[];
    const uint32_t words_per_row =
        xq_blocks * (uint32_t)sizeof(cuda_block_q8_K) / 4u;
    const unsigned int *src0 = (const unsigned int *)(
        xq + (uint64_t)tok * xq_blocks);
    for (uint32_t i = threadIdx.x; i < words_per_row;
         i += blockDim.x) {
        glm_moe_tok2_sh_u32[i] = src0[i];
    }
    if (np == 2u) {
        const unsigned int *src1 =
            (const unsigned int *)(xq + xq_blocks);
        for (uint32_t i = threadIdx.x; i < words_per_row;
             i += blockDim.x) {
            glm_moe_tok2_sh_u32[words_per_row + i] = src1[i];
        }
    }
    __syncthreads();

    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= expert_mid_dim) return;
    const char *gr = gate_base + (uint64_t)expert * gate_expert_bytes +
                     (uint64_t)r * gate_row_bytes;
    const char *ur = up_base + (uint64_t)expert * up_expert_bytes +
                     (uint64_t)r * up_row_bytes;
    const cuda_block_q8_K *x0 =
        (const cuda_block_q8_K *)glm_moe_tok2_sh_u32;
    const cuda_block_q8_K *x1 = np == 2u
        ? x0 + xq_blocks
        : NULL;
    float g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        dev_dot_q2_K_q8_K_block8(
                (const cuda_block_q2_K *)(gr + (uint64_t)b * 84u),
                x0 + b, np == 2u ? x1 + b : NULL,
                NULL, NULL, NULL, NULL, NULL, NULL, np, g);
        dev_dot_q2_K_q8_K_block8(
                (const cuda_block_q2_K *)(ur + (uint64_t)b * 84u),
                x0 + b, np == 2u ? x1 + b : NULL,
                NULL, NULL, NULL, NULL, NULL, NULL, np, u);
    }
    for (uint32_t p = 0; p < np; p++) {
        for (int off = 16; off > 0; off >>= 1) {
            g[p] += __shfl_down_sync(0xffffffffu, g[p], off);
            u[p] += __shfl_down_sync(0xffffffffu, u[p], off);
        }
        if (lane == 0u) {
            const uint32_t pair = p == 0u ? pair0 : pair1;
            mid[(uint64_t)pair * expert_mid_dim + r] =
                (g[p] / (1.0f + expf(-g[p]))) * u[p];
        }
    }
}

/* Warp-per-output-row down projection: stages all n_expert quantized mid
 * rows for the token in shared memory, each warp accumulates one out row
 * across every selected expert. Grid: (out_dim/warps, n_tokens). */
__global__ static void glm_routed_moe_down_warp_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.y;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (tok >= n_tokens) return;

    extern __shared__ unsigned int glm_moe_sh_u32[];
    {
        const unsigned int *src = (const unsigned int *)
                (midq + (uint64_t)tok * n_expert * midq_blocks);
        const uint32_t words = n_expert * midq_blocks *
                               (uint32_t)sizeof(cuda_block_q8_K) / 4u;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
            glm_moe_sh_u32[i] = src[i];
        }
    }
    __syncthreads();
    const cuda_block_q8_K *msh = (const cuda_block_q8_K *)glm_moe_sh_u32;

    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= out_dim) return;
    const uint32_t units = n_expert * midq_blocks;
    float acc = 0.0f;
    for (uint32_t idx = lane; idx < units; idx += 32u) {
        const uint32_t slot = idx / midq_blocks;
        const uint32_t b = idx - slot * midq_blocks;
        const int32_t expert = selected[(uint64_t)tok * n_expert + slot];
        if (expert < 0) continue;
        const float w = weights[(uint64_t)tok * n_expert + slot];
        const cuda_block_q2_K *dr = (const cuda_block_q2_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)r * down_row_bytes);
        acc += w * dev_dot_q2_K_q8_K_block(dr + b, msh + slot * midq_blocks + b);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0u) out[(uint64_t)tok * out_dim + r] = acc;
}

/* Expert-major routed MoE for prefill: build per-expert token lists,
 * then walk rows expert-by-expert so weights stream once per layer and
 * activations hit L2. pair = tok * n_expert + slot indexes selected/
 * weights/mid rows directly. */
__global__ static void glm_moe_expert_map_kernel(
        int32_t *counts,
        int32_t *lists,
        const int32_t *selected,
        uint32_t n_pairs,
        uint32_t n_total_expert,
        uint32_t cap,
        uint32_t pair_base) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const uint32_t pair = pair_base + p;
    const int32_t e = selected[pair];
    if (e < 0 || (uint32_t)e >= n_total_expert) return;
    const int32_t idx = atomicAdd(&counts[e], 1);
    lists[(uint64_t)e * cap + idx] = (int32_t)pair;
}

__global__ static void glm_moe_build_expert_tiles8_kernel(
        uint32_t *tile_total,
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const int32_t *counts,
        uint32_t n_total_expert) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    uint32_t total = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        const uint32_t count = counts[e] > 0 ? (uint32_t)counts[e] : 0u;
        const uint32_t ntiles = (count + 7u) / 8u;
        for (uint32_t t = 0; t < ntiles; t++) {
            tile_experts[total] = e;
            tile_starts[total] = t * 8u;
            total++;
        }
    }
    *tile_total = total;
}

/* Expert-tiled Q2_K gate/up for GLM prefill. One warp keeps the same
 * block-to-lane assignment and reduction tree as the token-major W32
 * kernel, but evaluates eight pairs against each loaded expert row. */
__global__ static void glm_routed_moe_gateup_expert_tile8_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *counts,
        const int32_t *lists,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t cap) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    const uint32_t count = counts[expert] > 0 ? (uint32_t)counts[expert] : 0u;

    __shared__ uint32_t pair[8];
    __shared__ uint32_t tok[8];
    __shared__ uint32_t np;
    if (threadIdx.x == 0u) {
        uint32_t n = count - local_start;
        if (n > 8u) n = 8u;
        np = n;
        for (uint32_t p = 0; p < n; p++) {
            const uint32_t pr =
                (uint32_t)lists[(uint64_t)expert * cap + local_start + p];
            pair[p] = pr;
            tok[p] = pr / n_expert;
        }
    }
    __syncthreads();

    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= expert_mid_dim) return;
    const char *gr = gate_base + (uint64_t)expert * gate_expert_bytes +
                     (uint64_t)r * gate_row_bytes;
    const char *ur = up_base + (uint64_t)expert * up_expert_bytes +
                     (uint64_t)r * up_row_bytes;
    float g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        const cuda_block_q8_K *x0 = np > 0u ? xq + (uint64_t)tok[0] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x1 = np > 1u ? xq + (uint64_t)tok[1] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x2 = np > 2u ? xq + (uint64_t)tok[2] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x3 = np > 3u ? xq + (uint64_t)tok[3] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x4 = np > 4u ? xq + (uint64_t)tok[4] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x5 = np > 5u ? xq + (uint64_t)tok[5] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x6 = np > 6u ? xq + (uint64_t)tok[6] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x7 = np > 7u ? xq + (uint64_t)tok[7] * xq_blocks + b : NULL;
        dev_dot_q2_K_q8_K_block8(
                (const cuda_block_q2_K *)(gr + (uint64_t)b * 84u),
                x0, x1, x2, x3, x4, x5, x6, x7, np, g);
        dev_dot_q2_K_q8_K_block8(
                (const cuda_block_q2_K *)(ur + (uint64_t)b * 84u),
                x0, x1, x2, x3, x4, x5, x6, x7, np, u);
    }
    for (uint32_t p = 0; p < np; p++) {
        for (int off = 16; off > 0; off >>= 1) {
            g[p] += __shfl_down_sync(0xffffffffu, g[p], off);
            u[p] += __shfl_down_sync(0xffffffffu, u[p], off);
        }
        if (lane == 0u) {
            mid[(uint64_t)pair[p] * expert_mid_dim + r] =
                (g[p] / (1.0f + expf(-g[p]))) * u[p];
        }
    }
}

__global__ static void glm_routed_moe_gateup_expert_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *counts,
        const int32_t *lists,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t cap) {
    const uint32_t e = blockIdx.y;
    const int32_t nt = counts[e];
    if (nt == 0) return;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= expert_mid_dim) return;
    const char *gr = gate_base + (uint64_t)e * gate_expert_bytes +
                     (uint64_t)r * gate_row_bytes;
    const char *ur = up_base + (uint64_t)e * up_expert_bytes +
                     (uint64_t)r * up_row_bytes;
    const int32_t *lst = lists + (uint64_t)e * cap;
    for (int32_t i = 0; i < nt; i++) {
        const uint32_t pair = (uint32_t)lst[i];
        const cuda_block_q8_K *xrow = xq + (uint64_t)(pair / n_expert) * xq_blocks;
        float g = 0.0f, u = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            g += dev_dot_q2_K_q8_K_block(
                    (const cuda_block_q2_K *)(gr + (uint64_t)b * 84u), xrow + b);
            u += dev_dot_q2_K_q8_K_block(
                    (const cuda_block_q2_K *)(ur + (uint64_t)b * 84u), xrow + b);
        }
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_down_sync(0xffffffffu, g, off);
            u += __shfl_down_sync(0xffffffffu, u, off);
        }
        if (lane == 0u) {
            mid[(uint64_t)pair * expert_mid_dim + r] =
                (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ static void glm_routed_moe_down_expert_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *counts,
        const int32_t *lists,
        const float *weights,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t cap) {
    const uint32_t e = blockIdx.y;
    const int32_t nt = counts[e];
    if (nt == 0) return;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= out_dim) return;
    const char *dr = down_base + (uint64_t)e * down_expert_bytes +
                     (uint64_t)r * down_row_bytes;
    const int32_t *lst = lists + (uint64_t)e * cap;
    for (int32_t i = 0; i < nt; i++) {
        const uint32_t pair = (uint32_t)lst[i];
        const cuda_block_q8_K *mrow = midq + (uint64_t)pair * midq_blocks;
        float s = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 32u) {
            s += dev_dot_q2_K_q8_K_block(
                    (const cuda_block_q2_K *)(dr + (uint64_t)b * 84u), mrow + b);
        }
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off);
        }
        if (lane == 0u) {
            atomicAdd(&out[(uint64_t)(pair / n_expert) * out_dim + r],
                      weights[pair] * s);
        }
    }
}

/* Expert-tiled down projection with an exact token-major reduction. The
 * first kernel reuses each Q2_K row across eight routed pairs, but materializes
 * each block dot. The second kernel applies the router weight and consumes the
 * dots with the same lane assignment and warp tree as the native kernel. */
__global__ static void glm_routed_moe_down_expert_tile8_terms_kernel(
        float *terms,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *counts,
        const int32_t *lists,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t cap,
        uint32_t pair_base) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    const uint32_t count = counts[expert] > 0 ? (uint32_t)counts[expert] : 0u;

    __shared__ uint32_t pair[8];
    __shared__ uint32_t np;
    __shared__ cuda_block_q8_K mq[8][8];
    if (threadIdx.x == 0u) {
        uint32_t n = count - local_start;
        if (n > 8u) n = 8u;
        np = n;
        for (uint32_t p = 0; p < n; p++) {
            pair[p] = (uint32_t)lists[
                (uint64_t)expert * cap + local_start + p];
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
        const uint32_t p = i / midq_blocks;
        const uint32_t b = i - p * midq_blocks;
        mq[p][b] = midq[(uint64_t)pair[p] * midq_blocks + b];
    }
    __syncthreads();
    if (row >= out_dim || lane >= midq_blocks) return;

    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base +
        (uint64_t)expert * down_expert_bytes +
        (uint64_t)row * down_row_bytes);
    for (uint32_t p = 0; p < np; p++) {
        const uint32_t pr = pair[p];
        const float dot = dev_dot_q2_K_q8_K_block(wr + lane, &mq[p][lane]);
        terms[((uint64_t)(pr - pair_base) * out_dim + row) *
              midq_blocks + lane] =
            dot;
    }
}

__global__ static void glm_routed_moe_down_terms_reduce_kernel(
        float *out,
        const float *terms,
        const int32_t *selected,
        const float *weights,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + warp;
    if (tok >= n_tokens || row >= out_dim) return;
    const uint32_t units = n_expert * midq_blocks;
    float acc = 0.0f;
    for (uint32_t idx = lane; idx < units; idx += 32u) {
        const uint32_t slot = idx / midq_blocks;
        const uint32_t b = idx - slot * midq_blocks;
        const uint32_t pr = tok * n_expert + slot;
        if (selected[pr] >= 0) {
            acc += weights[pr] *
                terms[((uint64_t)pr * out_dim + row) * midq_blocks + b];
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

static int glm_routed_moe_finish_batch(
        ds4_gpu_tensor *out,
        float *out_work,
        uint64_t out_bytes,
        const char *what) {
    if (!cuda_ok(cudaGetLastError(), what)) return 0;
    if (out_work == (float *)out->ptr) return 1;
    return cuda_ok(cudaMemcpyAsync(out->ptr, out_work, out_bytes,
                                   cudaMemcpyDeviceToDevice, 0),
                   "glm routed moe local output copy");
}

extern "C" int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride) {
    (void)layer_index; (void)n_total_expert;
    if (!out || !mid || !x || !selected || !weights || !model_map ||
        n_tokens == 0 || n_expert == 0 ||
        (expert_in_dim & 255u) != 0u || (expert_mid_dim & 255u) != 0u) {
        return 0;
    }
    if (gate_type != 10u || up_type != 10u || down_type != 10u) {
        fprintf(stderr, "ds4: glm routed moe: unsupported types %u/%u/%u\n",
                gate_type, up_type, down_type);
        return 0;
    }
    if (mid_token_stride != n_expert * expert_mid_dim) {
        fprintf(stderr,
            "ds4: glm routed moe: mid stride %u != %u (packed rows expected)\n",
            mid_token_stride, n_expert * expert_mid_dim);
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const char *gw = (const char *)cuda_resolve_weight_ptr(model_map,
            gate_offset, (uint64_t)256 * gate_expert_bytes, logical_tier,
            "glm_gate_exps");
    const char *uw = (const char *)cuda_resolve_weight_ptr(model_map,
            up_offset, (uint64_t)256 * up_expert_bytes, logical_tier,
            "glm_up_exps");
    const char *dw = (const char *)cuda_resolve_weight_ptr(model_map,
            down_offset, (uint64_t)256 * down_expert_bytes, logical_tier,
            "glm_down_exps");
    if (!gw || !uw || !dw) return 0;

    /* Stage 1: quantize x rows to q8_K (existing kernel). */
    const uint32_t xq_blocks = expert_in_dim / 256u;
    const uint32_t midq_blocks = expert_mid_dim / 256u;
    static ds4_gpu_tensor *xq_scratch[DS4_MAX_GPUS] = {0};
    static ds4_gpu_tensor *midq_scratch[DS4_MAX_GPUS] = {0};
    int dev = logical_tier;
    const int scratch_tier = getenv("DS4_GLM_MOE_SCRATCH_TIER0") ? 0 : dev;
    const uint64_t xq_bytes = (uint64_t)n_tokens * xq_blocks *
                              sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = (uint64_t)n_tokens * n_expert * midq_blocks *
                                sizeof(cuda_block_q8_K);
    if (!xq_scratch[dev] || xq_scratch[dev]->bytes < xq_bytes) {
        if (xq_scratch[dev]) ds4_gpu_tensor_free(xq_scratch[dev]);
        xq_scratch[dev] = ds4_gpu_tensor_alloc_ptr_on(scratch_tier, xq_bytes);
    }
    if (!midq_scratch[dev] || midq_scratch[dev]->bytes < midq_bytes) {
        if (midq_scratch[dev]) ds4_gpu_tensor_free(midq_scratch[dev]);
        midq_scratch[dev] = ds4_gpu_tensor_alloc_ptr_on(scratch_tier, midq_bytes);
    }
    if (!xq_scratch[dev] || !midq_scratch[dev]) return 0;

    static ds4_gpu_tensor *mid_local[DS4_MAX_GPUS] = {0};
    static ds4_gpu_tensor *out_local[DS4_MAX_GPUS] = {0};
    const uint64_t mid_work_bytes =
        (uint64_t)n_tokens * mid_token_stride * sizeof(float);
    const uint64_t out_work_bytes =
        (uint64_t)n_tokens * out_dim * sizeof(float);
    float *mid_work = (float *)mid->ptr;
    float *out_work = (float *)out->ptr;
    const bool use_local_batch_io =
        n_tokens >= 128u && !getenv("DS4_GLM_MOE_NO_LOCAL_BATCH_IO");
    if (use_local_batch_io && ds4_tensor_device_idx(mid) != dev) {
        if (!mid_local[dev] || mid_local[dev]->bytes < mid_work_bytes) {
            if (mid_local[dev]) ds4_gpu_tensor_free(mid_local[dev]);
            mid_local[dev] =
                ds4_gpu_tensor_alloc_ptr_on(dev, mid_work_bytes);
        }
        if (!mid_local[dev]) return 0;
        mid_work = (float *)mid_local[dev]->ptr;
    }
    if (use_local_batch_io && ds4_tensor_device_idx(out) != dev) {
        if (!out_local[dev] || out_local[dev]->bytes < out_work_bytes) {
            if (out_local[dev]) ds4_gpu_tensor_free(out_local[dev]);
            out_local[dev] =
                ds4_gpu_tensor_alloc_ptr_on(dev, out_work_bytes);
        }
        if (!out_local[dev]) return 0;
        out_work = (float *)out_local[dev]->ptr;
    }
    {
        dim3 gq(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<gq, 256>>>(
                (cuda_block_q8_K *)xq_scratch[dev]->ptr,
                (const float *)x->ptr, expert_in_dim, n_tokens);
    }

    static ds4_gpu_tensor *map_scratch[DS4_MAX_GPUS] = {0};
    static ds4_gpu_tensor *down_terms_scratch[DS4_MAX_GPUS] = {0};
    const bool use_expert_tile8 =
        n_tokens >= 128u && !getenv("DS4_GLM_MOE_NO_EXPERT_TILE8");
    const bool use_expert_major =
        n_tokens >= 16u && getenv("DS4_GLM_MOE_EXPERT_MAJOR");
    if (use_expert_tile8 || use_expert_major) {
        const uint32_t cap = n_tokens;
        const uint32_t n_pairs = n_tokens * n_expert;
        const uint64_t counts_bytes = 256u * sizeof(int32_t);
        const uint64_t lists_off = (counts_bytes + 255u) & ~255ull;
        const uint64_t lists_bytes =
            (uint64_t)256u * cap * sizeof(int32_t);
        const uint32_t tile_capacity =
            (n_pairs + 7u) / 8u + 256u;
        const uint64_t tile_total_off =
            (lists_off + lists_bytes + 255u) & ~255ull;
        const uint64_t tile_experts_off =
            (tile_total_off + sizeof(uint32_t) + 255u) & ~255ull;
        const uint64_t tile_starts_off =
            tile_experts_off + (uint64_t)tile_capacity * sizeof(uint32_t);
        const uint64_t map_bytes = use_expert_tile8
            ? tile_starts_off + (uint64_t)tile_capacity * sizeof(uint32_t)
            : lists_off + lists_bytes;
        if (!map_scratch[dev] || map_scratch[dev]->bytes < map_bytes) {
            if (map_scratch[dev]) ds4_gpu_tensor_free(map_scratch[dev]);
            map_scratch[dev] = ds4_gpu_tensor_alloc_ptr_on(dev, map_bytes);
        }
        if (map_scratch[dev]) {
            int32_t *counts = (int32_t *)map_scratch[dev]->ptr;
            int32_t *lists = (int32_t *)((char *)map_scratch[dev]->ptr + lists_off);
            cudaMemsetAsync(counts, 0, counts_bytes);
            glm_moe_expert_map_kernel<<<(n_pairs + 255u) / 256u, 256>>>(
                    counts, lists, (const int32_t *)selected->ptr,
                    n_pairs, 256u, cap, 0u);
            if (use_expert_tile8) {
                uint32_t *tile_total = (uint32_t *)(
                        (char *)map_scratch[dev]->ptr + tile_total_off);
                uint32_t *tile_experts = (uint32_t *)(
                        (char *)map_scratch[dev]->ptr + tile_experts_off);
                uint32_t *tile_starts = (uint32_t *)(
                        (char *)map_scratch[dev]->ptr + tile_starts_off);
                glm_moe_build_expert_tiles8_kernel<<<1, 1>>>(
                        tile_total, tile_experts, tile_starts,
                        counts, 256u);
                dim3 ge1((expert_mid_dim + 7u) / 8u,
                          tile_capacity, 1);
                glm_routed_moe_gateup_expert_tile8_kernel<<<ge1, 256>>>(
                        mid_work, gw, uw,
                        (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                        counts, lists,
                        tile_total, tile_experts, tile_starts,
                        gate_expert_bytes, gate_row_bytes,
                        up_expert_bytes, up_row_bytes,
                        xq_blocks, expert_mid_dim, n_expert, cap);
                q8_K_quantize_kernel<<<
                        dim3(midq_blocks, n_tokens * n_expert, 1), 256>>>(
                        (cuda_block_q8_K *)midq_scratch[dev]->ptr,
                        mid_work,
                        expert_mid_dim, n_tokens * n_expert);
                if (getenv("DS4_GLM_MOE_NO_DOWN_TILE8_EXACT") == NULL) {
                    const uint32_t max_chunk_tokens = 512u;
                    const uint32_t scratch_tokens =
                        n_tokens < max_chunk_tokens ? n_tokens : max_chunk_tokens;
                    const uint64_t term_count =
                        (uint64_t)scratch_tokens * n_expert *
                        out_dim * midq_blocks;
                    const uint64_t term_bytes = term_count * sizeof(float);
                    if (!down_terms_scratch[dev] ||
                        down_terms_scratch[dev]->bytes < term_bytes) {
                        if (down_terms_scratch[dev]) {
                            ds4_gpu_tensor_free(down_terms_scratch[dev]);
                        }
                        down_terms_scratch[dev] =
                            ds4_gpu_tensor_alloc_ptr_on(dev, term_bytes);
                    }
                    if (down_terms_scratch[dev]) {
                        for (uint32_t token0 = 0; token0 < n_tokens;
                             token0 += max_chunk_tokens) {
                            uint32_t chunk_tokens = n_tokens - token0;
                            if (chunk_tokens > max_chunk_tokens) {
                                chunk_tokens = max_chunk_tokens;
                            }
                            const uint32_t pair_base = token0 * n_expert;
                            const uint32_t chunk_pairs =
                                chunk_tokens * n_expert;
                            uint32_t chunk_tile_capacity = tile_capacity;
                            if (n_tokens > max_chunk_tokens) {
                                chunk_tile_capacity =
                                    (chunk_pairs + 7u) / 8u + 256u;
                                cudaMemsetAsync(counts, 0, counts_bytes);
                                glm_moe_expert_map_kernel<<<
                                        (chunk_pairs + 255u) / 256u, 256>>>(
                                        counts, lists,
                                        (const int32_t *)selected->ptr,
                                        chunk_pairs, 256u, cap, pair_base);
                                glm_moe_build_expert_tiles8_kernel<<<1, 1>>>(
                                        tile_total, tile_experts, tile_starts,
                                        counts, 256u);
                            }
                            dim3 gd1((out_dim + 31u) / 32u,
                                     chunk_tile_capacity, 1);
                            glm_routed_moe_down_expert_tile8_terms_kernel<<<
                                    gd1, 256>>>(
                                    (float *)down_terms_scratch[dev]->ptr,
                                    dw,
                                    (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                                    counts, lists,
                                    tile_total, tile_experts, tile_starts,
                                    down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, cap,
                                    pair_base);
                            dim3 gd2((out_dim + 7u) / 8u,
                                     chunk_tokens, 1);
                            glm_routed_moe_down_terms_reduce_kernel<<<
                                    gd2, 256>>>(
                                    out_work + (uint64_t)token0 * out_dim,
                                    (const float *)down_terms_scratch[dev]->ptr,
                                    (const int32_t *)selected->ptr + pair_base,
                                    (const float *)weights->ptr + pair_base,
                                    midq_blocks, out_dim, n_expert,
                                    chunk_tokens);
                        }
                        return glm_routed_moe_finish_batch(
                                out, out_work, out_work_bytes,
                                "glm routed moe exact down tile8");
                    }
                }
                const uint32_t warps = 8u;
                dim3 ge2((out_dim + warps - 1u) / warps,
                          n_tokens, 1);
                const uint32_t sh2 = n_expert * midq_blocks *
                                     (uint32_t)sizeof(cuda_block_q8_K);
                glm_routed_moe_down_warp_kernel<<<
                        ge2, warps * 32u, sh2>>>(
                        out_work, dw,
                        (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, n_tokens);
                return glm_routed_moe_finish_batch(
                        out, out_work, out_work_bytes,
                        "glm routed moe expert tile8");
            }
            dim3 ge1((expert_mid_dim + 7u) / 8u, 256u, 1);
            glm_routed_moe_gateup_expert_kernel<<<ge1, 256>>>(
                    mid_work, gw, uw,
                    (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                    counts, lists,
                    gate_expert_bytes, gate_row_bytes,
                    up_expert_bytes, up_row_bytes,
                    xq_blocks, expert_mid_dim, n_expert, cap);
            q8_K_quantize_kernel<<<dim3(midq_blocks, n_tokens * n_expert, 1), 256>>>(
                    (cuda_block_q8_K *)midq_scratch[dev]->ptr,
                    mid_work, expert_mid_dim, n_tokens * n_expert);
            cudaMemsetAsync(out_work, 0,
                            (uint64_t)n_tokens * out_dim * sizeof(float));
            dim3 ge2((out_dim + 7u) / 8u, 256u, 1);
            glm_routed_moe_down_expert_kernel<<<ge2, 256>>>(
                    out_work, dw,
                    (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                    counts, lists, (const float *)weights->ptr,
                    down_expert_bytes, down_row_bytes,
                    midq_blocks, out_dim, n_expert, cap);
            return glm_routed_moe_finish_batch(
                    out, out_work, out_work_bytes,
                    "glm routed moe expert-major");
        }
    }
    if (n_tokens == 2u &&
        g_glm_mtp_verify_mode &&
        getenv("DS4_GLM_MTP_NO_MOE_TOK2") == NULL) {
        const uint32_t warps = 8u;
        dim3 g1((expert_mid_dim + warps - 1u) / warps,
                2u * n_expert, 1u);
        const uint32_t sh1 = 2u * xq_blocks *
                             (uint32_t)sizeof(cuda_block_q8_K);
        glm_routed_moe_gateup_tok2_reuse_kernel<<<
                g1, warps * 32u, sh1>>>(
                mid_work, gw, uw,
                (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                gate_expert_bytes, gate_row_bytes,
                up_expert_bytes, up_row_bytes,
                xq_blocks, expert_mid_dim, n_expert);
    } else if (getenv("DS4_GLM_MOE_SCALAR")) {
        dim3 g1(n_tokens, n_expert, 1);
        glm_routed_moe_batch_q2K_gateup_kernel<<<g1, 128>>>(
                mid_work, gw, uw,
                (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes,
                xq_blocks, expert_mid_dim, n_expert, n_tokens, mid_token_stride);
    } else {
        const uint32_t warps = 8u;
        dim3 g1((expert_mid_dim + warps - 1u) / warps, n_expert, n_tokens);
        const uint32_t sh1 = xq_blocks * (uint32_t)sizeof(cuda_block_q8_K);
        glm_routed_moe_gateup_warp_kernel<<<g1, warps * 32u, sh1>>>(
                mid_work, gw, uw,
                (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes,
                xq_blocks, expert_mid_dim, n_expert, n_tokens);
    }

    {
        dim3 gq(midq_blocks, n_tokens * n_expert, 1);
        q8_K_quantize_kernel<<<gq, 256>>>(
                (cuda_block_q8_K *)midq_scratch[dev]->ptr,
                mid_work, expert_mid_dim, n_tokens * n_expert);
    }

    if (getenv("DS4_GLM_MOE_SCALAR")) {
        dim3 g2((out_dim + 127u) / 128u, n_tokens, 1);
        glm_routed_moe_batch_q2K_down_kernel<<<g2, 128>>>(
                out_work, dw,
                (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                n_expert, n_tokens);
    } else {
        const uint32_t warps = 8u;
        dim3 g2((out_dim + warps - 1u) / warps, n_tokens, 1);
        const uint32_t sh2 = n_expert * midq_blocks *
                             (uint32_t)sizeof(cuda_block_q8_K);
        glm_routed_moe_down_warp_kernel<<<g2, warps * 32u, sh2>>>(
                out_work, dw,
                (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                n_expert, n_tokens);
    }
    return glm_routed_moe_finish_batch(
            out, out_work, out_work_bytes,
            "glm routed moe batch launch");
}

extern "C" int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident) {
    (void)force_resident;
    return ds4_gpu_glm_routed_moe_batch_tensor(out, mid,
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_type, up_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            up_expert_bytes, up_row_bytes,
            down_expert_bytes, down_row_bytes,
            expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, n_total_expert, n_expert, layer_index,
            x, 1, n_expert * expert_mid_dim);
}

/* Parallel router select: 256 threads compute sigmoid probs, then top-k
 * via k rounds of shared-memory argmax over probs+bias (value desc, index
 * asc tie-break — matches the CPU topk_desc). One block per token. */
__global__ static void glm_router_select_parallel_kernel(
        int32_t *selected,
        float *weights_out,
        float *probs_out,
        const float *bias,
        const float *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tokens) return;
    const float *lg = logits + (uint64_t)tok * n_expert;
    float *probs = probs_out + (uint64_t)tok * n_expert;
    int32_t *sel = selected + (uint64_t)tok * n_expert_used;
    float *w = weights_out + (uint64_t)tok * n_expert_used;

    __shared__ float sh_v[256];
    __shared__ int sh_i[256];
    __shared__ float sh_sel_v[256];
    __shared__ float sh_sum;

    float my_v = -1e30f;
    if (tid < n_expert) {
        const float p = 1.0f / (1.0f + expf(-lg[tid]));
        probs[tid] = p;
        my_v = p + bias[tid];
    }
    if (tid == 0u) sh_sum = 0.0f;
    sh_sel_v[tid] = my_v;
    __syncthreads();

    for (uint32_t k2 = 0; k2 < n_expert_used; k2++) {
        sh_v[tid] = sh_sel_v[tid];
        sh_i[tid] = (int)tid;
        __syncthreads();
        for (uint32_t step = 128u; step > 0u; step >>= 1u) {
            if (tid < step) {
                const float ov = sh_v[tid + step];
                const int oi = sh_i[tid + step];
                if (ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid])) {
                    sh_v[tid] = ov;
                    sh_i[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0u) {
            const int best = sh_i[0];
            sel[k2] = best;
            const float p = probs[best];
            w[k2] = p;
            sh_sum += p;
            sh_sel_v[best] = -1e30f;
        }
        __syncthreads();
    }
    if (tid == 0u) {
        float sum = sh_sum;
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for (uint32_t k2 = 0; k2 < n_expert_used; k2++) {
            w[k2] = w[k2] / sum * expert_weight_scale;
        }
    }
}

__global__ static void glm_router_select_batch_kernel(
        int32_t *selected,
        float *weights_out,
        float *probs_out,
        const float *bias,
        const float *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.x;
    if (tok >= n_tokens || threadIdx.x != 0u) return;
    const float *lg = logits + (uint64_t)tok * n_expert;
    float *probs = probs_out + (uint64_t)tok * n_expert;
    int32_t *sel = selected + (uint64_t)tok * n_expert_used;
    float *w = weights_out + (uint64_t)tok * n_expert_used;

    for (uint32_t i = 0; i < n_expert; i++) {
        const float p = 1.0f / (1.0f + expf(-lg[i]));
        probs[i] = p;
    }
    /* top-k over probs+bias, ties by smaller index (matches CPU topk_desc) */
    bool taken[384];
    for (uint32_t i = 0; i < n_expert; i++) taken[i] = false;
    float sum = 0.0f;
    for (uint32_t k2 = 0; k2 < n_expert_used; k2++) {
        int best = -1; float bv = -1e30f;
        for (uint32_t i = 0; i < n_expert; i++) {
            if (taken[i]) continue;
            const float v = probs[i] + bias[i];
            if (v > bv) { bv = v; best = (int)i; }
        }
        taken[best] = true;
        sel[k2] = best;
        w[k2] = probs[best];
        sum += probs[best];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t k2 = 0; k2 < n_expert_used; k2++) {
        w[k2] = w[k2] / sum * expert_weight_scale;
    }
}

extern "C" int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert == 0 || n_expert > 384u || n_expert_used == 0 ||
        n_tokens == 0) {
        return 0;
    }
    const uint64_t bb = (uint64_t)n_expert * sizeof(float);
    if (bias_offset > model_size || bb > model_size - bias_offset ||
        logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *bias = (const float *)cuda_resolve_weight_ptr(
            model_map, bias_offset, bb, logical_tier, "glm_exp_probs_b");
    if (!bias) return 0;
    if (n_expert <= 256u && !getenv("DS4_GLM_ROUTER_SCALAR")) {
        glm_router_select_parallel_kernel<<<n_tokens, 256>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                (const float *)logits->ptr,
                n_expert, n_expert_used, expert_weight_scale, n_tokens);
    } else glm_router_select_batch_kernel<<<n_tokens, 32>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            (float *)probs->ptr,
            bias,
            (const float *)logits->ptr,
            n_expert, n_expert_used, expert_weight_scale, n_tokens);
    return cuda_ok(cudaGetLastError(), "glm router select batch launch");
}

extern "C" int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale) {
    return ds4_gpu_glm_router_select_batch_tensor(selected, weights, probs,
                                                  model_map, model_size,
                                                  bias_offset, logits,
                                                  n_expert, n_expert_used,
                                                  expert_weight_scale, 1u);
}

__global__ static void glm_store_compact_kv_kernel(
        char *kv_lora_cache,
        char *k_rope_cache,
        const float *kv_norm,
        const float *kv_raw,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope,
        uint32_t cache_f16) {
    const uint32_t token = blockIdx.x;
    const uint32_t part = blockIdx.y;
    if (token >= n_tokens || part > 1u) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (part == 0u) {
        const float *src = kv_norm + (uint64_t)token * kv_lora_dim;
        if (cache_f16) {
            __half *dst = (__half *)(kv_lora_cache +
                (uint64_t)pos * kv_lora_dim * sizeof(__half));
            for (uint32_t i = tid; i < kv_lora_dim; i += nth)
                dst[i] = __float2half(src[i]);
        } else {
            float *dst = (float *)(kv_lora_cache +
                (uint64_t)pos * kv_lora_dim * sizeof(float));
            for (uint32_t i = tid; i < kv_lora_dim; i += nth)
                dst[i] = src[i];
        }
    } else {
        const float *src = kv_raw +
            (uint64_t)token * kv_raw_dim + kv_lora_dim;
        if (cache_f16) {
            __half *dst = (__half *)(k_rope_cache +
                (uint64_t)pos * qk_rope * sizeof(__half));
            for (uint32_t i = tid; i < qk_rope; i += nth)
                dst[i] = __float2half(src[i]);
        } else {
            float *dst = (float *)(k_rope_cache +
                (uint64_t)pos * qk_rope * sizeof(float));
            for (uint32_t i = tid; i < qk_rope; i += nth)
                dst[i] = src[i];
        }
    }
}

extern "C" int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16) {
    if (!kv_lora_cache || !k_rope_cache || !kv_norm || !kv_raw ||
        n_tokens == 0 || kv_lora_dim == 0 || qk_rope == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim + qk_rope) {
        return 0;
    }
    const uint64_t es = cache_f16 ? sizeof(__half) : sizeof(float);
    if (kv_norm->bytes < (uint64_t)n_tokens * kv_lora_dim * sizeof(float) ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * es ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * es) {
        return 0;
    }
    dim3 grid(n_tokens, 2, 1);
    glm_store_compact_kv_kernel<<<grid, 128>>>(
            (char *)kv_lora_cache->ptr,
            (char *)k_rope_cache->ptr,
            (const float *)kv_norm->ptr,
            (const float *)kv_raw->ptr,
            pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope,
            cache_f16 ? 1u : 0u);
    return cuda_ok(cudaGetLastError(), "glm store compact kv launch");
}







__global__ static void glm_store_indexer_k_kernel(
        char *cache,
        const float *raw_k,
        const float *w,
        const float *b,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t n_ctx_orig,
        float eps,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t cache_f16) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const float *src = raw_k + (uint64_t)token * head_dim;

    __shared__ float scratch[256];
    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += nth) sum += src[i];
    scratch[tid] = sum;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float mean = scratch[0] / (float)head_dim;
    __syncthreads();
    float ss = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += nth) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    scratch[tid] = ss;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)rot_dim, (int)n_ctx_orig,
                                                 beta_fast, freq_base)));
        corr_dims[1] = fminf((float)rot_dim - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)rot_dim, (int)n_ctx_orig,
                                                beta_slow, freq_base)));
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)rot_dim;

    for (uint32_t i = tid; i < head_dim; i += nth) {
        float v0, v1; bool pair = false;
        if (i < rot_dim) {
            if ((i & 1u) != 0u) continue;
            const float theta = theta_base * powf(freq_base, inv_ndims * (float)i);
            float ct, st;
            glm_rope_yarn_dev(theta, freq_scale, corr_dims, (int)i,
                              ext_factor, attn_factor, &ct, &st);
            const float x0 = (src[i] - mean) * inv * w[i] + b[i];
            const float x1 = (src[i + 1u] - mean) * inv * w[i + 1u] + b[i + 1u];
            v0 = x0 * ct - x1 * st;
            v1 = x0 * st + x1 * ct;
            pair = true;
        } else {
            v0 = (src[i] - mean) * inv * w[i] + b[i];
        }
        if (cache_f16) {
            __half *dst = (__half *)(cache + (uint64_t)pos * head_dim * sizeof(__half));
            dst[i] = __float2half(v0);
            if (pair) dst[i + 1u] = __float2half(v1);
        } else {
            float *dst = (float *)(cache + (uint64_t)pos * head_dim * sizeof(float));
            dst[i] = v0;
            if (pair) dst[i + 1u] = v1;
        }
    }
}

extern "C" int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    if (!indexer_key_cache || !raw_k || !model_map || n_tokens == 0 ||
        head_dim == 0 || head_dim > 256u || (rot_dim & 1u) != 0u) {
        return 0;
    }
    const uint64_t wb = (uint64_t)head_dim * sizeof(float);
    if (weight_offset > model_size || wb > model_size - weight_offset ||
        bias_offset > model_size || wb > model_size - bias_offset ||
        raw_k->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        indexer_key_cache->bytes <
            (uint64_t)cache_cap * head_dim *
                (cache_f16 ? sizeof(__half) : sizeof(float))) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *w = (const float *)cuda_resolve_weight_ptr(
            model_map, weight_offset, wb, logical_tier, "glm_indexer_k_norm");
    const float *b = (const float *)cuda_resolve_weight_ptr(
            model_map, bias_offset, wb, logical_tier, "glm_indexer_k_norm_b");
    if (!w || !b) return 0;
    glm_store_indexer_k_kernel<<<n_tokens, 128>>>(
            (char *)indexer_key_cache->ptr,
            (const float *)raw_k->ptr,
            w, b, pos0, n_tokens, cache_cap, head_dim, rot_dim, n_ctx_orig,
            eps, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, cache_f16 ? 1u : 0u);
    return cuda_ok(cudaGetLastError(), "glm store indexer k launch");
}

extern "C" int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table || !selected || n_selected == 0 ||
        selected->bytes < (uint64_t)n_selected * sizeof(int32_t)) {
        return 0;
    }
    std::vector<int32_t> ids;
    try {
        ids.resize(n_selected);
    } catch (...) {
        return 0;
    }
    if (!cuda_ok(cudaMemcpy(ids.data(), selected->ptr,
                            (size_t)n_selected * sizeof(int32_t),
                            cudaMemcpyDeviceToHost),
                 "GLM streaming selected-id read")) {
        return 0;
    }
    return cuda_stream_selected_cache_begin_load(table, ids.data(), n_selected);
}

__global__ static void glm_value_project_q8_0_batch_heads_kernel(
        float *heads,
        const char *weight,
        const float *lora,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint64_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    extern __shared__ float xsh[];
    const float *src = lora + (uint64_t)token * n_head * kv_lora_dim +
                       (uint64_t)head * kv_lora_dim;
    float *out = heads + (uint64_t)token * n_head * value_dim +
                 (uint64_t)head * value_dim;
    for (uint32_t j = tid; j < kv_lora_dim; j += nth) xsh[j] = src[j];
    __syncthreads();
    for (uint32_t d = tid; d < value_dim; d += nth) {
        const char *row = weight + ((uint64_t)head * value_dim + d) * row_bytes;
        out[d] = glm_q8_0_dot_row_dev(row, xsh, kv_lora_dim);
    }
}

/* Reuse each Q8 row across a token tile while retaining the scalar kernel's
 * block/k accumulation order independently for every output token. */
template <uint32_t token_tile>
__global__ static void glm_value_project_q8_0_batch_heads_tiled_kernel(
        float *heads,
        const char *weight,
        const float *lora,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint64_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token0 = blockIdx.y * token_tile;
    if (head >= n_head || token0 >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    extern __shared__ float xsh[];

#pragma unroll
    for (uint32_t t = 0; t < token_tile; t++) {
        const uint32_t token = token0 + t;
        if (token >= n_tokens) break;
        const float *src = lora + (uint64_t)token * n_head * kv_lora_dim +
                           (uint64_t)head * kv_lora_dim;
        for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
            xsh[(uint64_t)t * kv_lora_dim + j] = src[j];
        }
    }
    __syncthreads();

    for (uint32_t od = tid; od < value_dim; od += nth) {
        const char *row = weight +
            ((uint64_t)head * value_dim + od) * row_bytes;
        float acc[token_tile] = { 0.0f };
        const uint32_t nb = kv_lora_dim >> 5;
        for (uint32_t b = 0; b < nb; b++) {
            const char *blk = row + (uint64_t)b * 34u;
            const float d = __half2float(*(const __half *)blk);
            const int8_t *q = (const int8_t *)(blk + 2);
            float s[token_tile] = { 0.0f };
#pragma unroll 8
            for (uint32_t k = 0; k < 32u; k++) {
                const float w = (float)q[k];
#pragma unroll
                for (uint32_t t = 0; t < token_tile; t++) {
                    if (token0 + t < n_tokens) {
                        s[t] += w * xsh[(uint64_t)t * kv_lora_dim +
                                        b * 32u + k];
                    }
                }
            }
#pragma unroll
            for (uint32_t t = 0; t < token_tile; t++) {
                if (token0 + t < n_tokens) acc[t] += d * s[t];
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < token_tile; t++) {
            const uint32_t token = token0 + t;
            if (token < n_tokens) {
                heads[(uint64_t)token * n_head * value_dim +
                      (uint64_t)head * value_dim + od] = acc[t];
            }
        }
    }
}

extern "C" int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim) {
    if (!heads || !lora || !model_map || n_tokens == 0 || n_head == 0 ||
        kv_lora_dim == 0 || (kv_lora_dim & 31u) != 0u || value_dim == 0) {
        return 0;
    }
    if (weight_type != 8u) {
        fprintf(stderr, "ds4: glm value project: unsupported type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)kv_lora_dim / 32u) * 34u;
    const uint64_t wbytes = (uint64_t)n_head * value_dim * row_bytes;
    if (weight_offset > model_size || wbytes > model_size - weight_offset ||
        lora->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const char *w = (const char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, wbytes, logical_tier, "glm_v_b");
    if (!w) return 0;
    if (n_tokens >= 16u && getenv("DS4_GLM_VALUE_NO_TILE16") == NULL) {
        dim3 grid(n_head, (n_tokens + 15u) / 16u, 1);
        const size_t shmem = 16ull * kv_lora_dim * sizeof(float);
        glm_value_project_q8_0_batch_heads_tiled_kernel<16><<<grid, 128, shmem>>>(
                (float *)heads->ptr, w, (const float *)lora->ptr,
                n_tokens, n_head, kv_lora_dim, value_dim, row_bytes);
        return cuda_ok(cudaGetLastError(), "glm value project tile16 launch");
    }
    dim3 grid(n_head, n_tokens, 1);
    const size_t shmem = (size_t)kv_lora_dim * sizeof(float);
    glm_value_project_q8_0_batch_heads_kernel<<<grid, 128, shmem>>>(
            (float *)heads->ptr, w, (const float *)lora->ptr,
            n_tokens, n_head, kv_lora_dim, value_dim, row_bytes);
    return cuda_ok(cudaGetLastError(), "glm value project launch");
}

/* Decode-time (n_tok small) quant matvec. The Metal "mpp/model-view"
 * variant is a bandwidth-tuned matvec; on CUDA the generic quant matmul
 * already dispatches per type, so delegate. Revisit in the perf pass. */
extern "C" int ds4_gpu_matmul_quant_tensor(ds4_gpu_tensor *out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok);

extern "C" int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       weight_offset, weight_type,
                                       in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_matmul_quant_rows_scalar_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    switch (weight_type) {
    case 8u:   /* Q8_0 */
        return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                          weight_offset, in_dim, out_dim,
                                          x, n_tok);
    case 1u:   /* F16 */
        return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                         weight_offset, in_dim, out_dim,
                                         x, n_tok);
    default:
        fprintf(stderr, "ds4: matmul_quant: unsupported type %u\n",
                weight_type);
        return 0;
    }
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    /* GLM graph memory guard: on this backend the model weights are
     * distributed across all devices by the multi-tier placement, so the
     * relevant budget is the aggregate VRAM. */
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return 0;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return (uint64_t)total_b * (uint64_t)n;
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected) {
    (void)selected;
    (void)n_selected;
    return 1;
}

extern "C" void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled) {
    (void)enabled;   /* SSD streaming is not used on the CUDA backend */
}

extern "C" void ds4_gpu_set_glm_mtp_verify_mode(bool enabled) {
    g_glm_mtp_verify_mode = enabled;
}

extern "C" int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) {
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size || sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL) {
        for (uint32_t i = 0; i < count; i++) {
            (void)cuda_model_prefetch_range(model_map, model_size,
                                            offsets[i], sizes[i]);
        }
    }
    return 1;
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok, float clamp);

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate, up, mid,
            model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, 1, clamp);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    (void)gate; (void)up; (void)mid; (void)model_map; (void)model_size;
    (void)gate_offset; (void)up_offset; (void)in_dim; (void)out_dim;
    (void)x; (void)n_tok; (void)clamp;
    return 0;
}

/* Fused single-token shared-expert gate+up+swiglu: one warp per output
 * row computes both q8_0 dots against a shared-staged f32 x and writes
 * silu(gate)*up directly. Falls back to the split path for n_tok > 1. */
__global__ static void glm_shared_gate_up_swiglu_one_kernel(
        float *mid,
        const char *gw,
        const char *uw,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        float clamp) {
    extern __shared__ float glm_sgu_sh[];
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        glm_sgu_sh[i] = x[i];
    }
    __syncthreads();
    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= out_dim) return;
    const uint32_t nblk = in_dim >> 5;
    const uint64_t row_bytes = (uint64_t)nblk * 34u;
    const char *grow = gw + (uint64_t)r * row_bytes;
    const char *urow = uw + (uint64_t)r * row_bytes;
    float g = 0.0f, u = 0.0f;
    for (uint32_t blk = lane; blk < nblk; blk += 32u) {
        const char *gb = grow + (uint64_t)blk * 34u;
        const char *ub = urow + (uint64_t)blk * 34u;
        const float gd = __half2float(*(const __half *)gb);
        const float ud = __half2float(*(const __half *)ub);
        const int8_t *gq = (const int8_t *)(gb + 2);
        const int8_t *uq = (const int8_t *)(ub + 2);
        const float *xs = glm_sgu_sh + blk * 32u;
        float gs = 0.0f, us = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < 32; k++) {
            gs += (float)gq[k] * xs[k];
            us += (float)uq[k] * xs[k];
        }
        g += gd * gs;
        u += ud * us;
    }
    for (int off = 16; off > 0; off >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, off);
        u += __shfl_down_sync(0xffffffffu, u, off);
    }
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        mid[r] = (g / (1.0f + expf(-g))) * u;
    }
}

/* Two-token verifier variant of the decode kernel above. Each warp loads a
 * gate/up weight row once, while each token keeps the decode kernel's block
 * order and warp reduction tree independently. */
__global__ static void glm_shared_gate_up_swiglu_tok2_exact_kernel(
        float *mid,
        const char *gw,
        const char *uw,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        float clamp) {
    extern __shared__ float glm_sgu2_sh[];
    float *x0 = glm_sgu2_sh;
    float *x1 = glm_sgu2_sh + in_dim;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        x0[i] = x[i];
        x1[i] = x[in_dim + i];
    }
    __syncthreads();
    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= out_dim) return;
    const uint32_t nblk = in_dim >> 5;
    const uint64_t row_bytes = (uint64_t)nblk * 34u;
    const char *grow = gw + (uint64_t)r * row_bytes;
    const char *urow = uw + (uint64_t)r * row_bytes;
    float g0 = 0.0f, u0 = 0.0f;
    float g1 = 0.0f, u1 = 0.0f;
    for (uint32_t blk = lane; blk < nblk; blk += 32u) {
        const char *gb = grow + (uint64_t)blk * 34u;
        const char *ub = urow + (uint64_t)blk * 34u;
        const float gd = __half2float(*(const __half *)gb);
        const float ud = __half2float(*(const __half *)ub);
        const int8_t *gq = (const int8_t *)(gb + 2);
        const int8_t *uq = (const int8_t *)(ub + 2);
        const float *xs0 = x0 + blk * 32u;
        const float *xs1 = x1 + blk * 32u;
        float gs0 = 0.0f, us0 = 0.0f;
        float gs1 = 0.0f, us1 = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < 32; k++) {
            const float gk = (float)gq[k];
            const float uk = (float)uq[k];
            gs0 += gk * xs0[k];
            us0 += uk * xs0[k];
            gs1 += gk * xs1[k];
            us1 += uk * xs1[k];
        }
        g0 += gd * gs0;
        u0 += ud * us0;
        g1 += gd * gs1;
        u1 += ud * us1;
    }
    for (int off = 16; off > 0; off >>= 1) {
        g0 += __shfl_down_sync(0xffffffffu, g0, off);
        u0 += __shfl_down_sync(0xffffffffu, u0, off);
        g1 += __shfl_down_sync(0xffffffffu, g1, off);
        u1 += __shfl_down_sync(0xffffffffu, u1, off);
    }
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (g0 > clamp) g0 = clamp;
            if (u0 > clamp) u0 = clamp;
            if (u0 < -clamp) u0 = -clamp;
            if (g1 > clamp) g1 = clamp;
            if (u1 > clamp) u1 = clamp;
            if (u1 < -clamp) u1 = -clamp;
        }
        mid[r] = (g0 / (1.0f + expf(-g0))) * u0;
        mid[out_dim + r] = (g1 / (1.0f + expf(-g1))) * u1;
    }
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (!gate || !up || !mid || !x || n_tok == 0) return 0;
    if (n_tok == 2 && (in_dim & 31u) == 0u &&
        g_glm_mtp_verify_mode &&
        getenv("DS4_GLM_MTP_NO_SHARED_TOK2") == NULL &&
        mid->bytes >= 2u * out_dim * sizeof(float) &&
        x->bytes >= 2u * in_dim * sizeof(float)) {
        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t wb = out_dim * row_bytes;
        if (gate_offset <= model_size && wb <= model_size - gate_offset &&
            up_offset <= model_size && wb <= model_size - up_offset) {
            const int logical_tier = cuda_current_tier();
            const char *gw = cuda_resolve_weight_ptr(model_map, gate_offset,
                    wb, logical_tier, "glm_shared_gate");
            const char *uw = cuda_resolve_weight_ptr(model_map, up_offset,
                    wb, logical_tier, "glm_shared_up");
            if (gw && uw) {
                const uint32_t warps = 8u;
                const uint32_t sh = 2u * (uint32_t)in_dim * sizeof(float);
                glm_shared_gate_up_swiglu_tok2_exact_kernel
                        <<<(unsigned)((out_dim + warps - 1u) / warps),
                           warps * 32u, sh>>>(
                        (float *)mid->ptr, gw, uw, (const float *)x->ptr,
                        (uint32_t)in_dim, (uint32_t)out_dim, clamp);
                return cuda_ok(cudaGetLastError(),
                               "glm shared swiglu tok2 exact");
            }
        }
    }
    if (n_tok == 1 && (in_dim & 31u) == 0u &&
        !getenv("DS4_GLM_SHARED_SPLIT") &&
        mid->bytes >= out_dim * sizeof(float) &&
        x->bytes >= in_dim * sizeof(float)) {
        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t wb = out_dim * row_bytes;
        if (gate_offset <= model_size && wb <= model_size - gate_offset &&
            up_offset <= model_size && wb <= model_size - up_offset) {
            const int logical_tier = cuda_current_tier();
            const char *gw = cuda_resolve_weight_ptr(model_map, gate_offset,
                    wb, logical_tier, "glm_shared_gate");
            const char *uw = cuda_resolve_weight_ptr(model_map, up_offset,
                    wb, logical_tier, "glm_shared_up");
            if (gw && uw) {
                const uint32_t warps = 8u;
                const uint32_t sh = (uint32_t)in_dim * sizeof(float);
                glm_shared_gate_up_swiglu_one_kernel
                        <<<(unsigned)((out_dim + warps - 1u) / warps),
                           warps * 32u, sh>>>(
                        (float *)mid->ptr, gw, uw, (const float *)x->ptr,
                        (uint32_t)in_dim, (uint32_t)out_dim, clamp);
                return cuda_ok(cudaGetLastError(), "glm shared swiglu one");
            }
        }
    }
    if (!ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size, gate_offset,
                                    in_dim, out_dim, x, n_tok) ||
        !ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size, up_offset,
                                    in_dim, out_dim, x, n_tok)) {
        return 0;
    }
    return ds4_gpu_swiglu_tensor(mid, gate, up,
                                 (uint32_t)(out_dim * n_tok), clamp, 1.0f);
}

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    static ds4_gpu_tensor *gu_scratch[DS4_MAX_GPUS][2] = {{0}};
    const int dev = cuda_current_tier();
    const uint64_t need = out_dim * sizeof(float);
    for (int i = 0; i < 2; i++) {
        if (!gu_scratch[dev][i] || gu_scratch[dev][i]->bytes < need) {
            if (gu_scratch[dev][i]) ds4_gpu_tensor_free(gu_scratch[dev][i]);
            gu_scratch[dev][i] = ds4_gpu_tensor_alloc(need);
        }
        if (!gu_scratch[dev][i]) return 0;
    }
    return ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
            gu_scratch[dev][0], gu_scratch[dev][1], mid,
            model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, 1, clamp);
}

extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (event_value) *event_value = 1;
    return cuda_ok(cudaDeviceSynchronize(), "selected readback signal");
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    return cuda_stream_selected_cache_begin_load(table, selected_ids,
                                                 n_selected);
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return 0;
}

extern "C" int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count) {
    if (!dst || !src) return 0;
    if (count == 0) return 1;
    if (count > UINT64_MAX / sizeof(float) ||
        count > UINT64_MAX / sizeof(__half)) {
        return 0;
    }
    const uint64_t src_bytes = count * sizeof(float);
    const uint64_t dst_bytes = count * sizeof(__half);
    if (src_offset > src->bytes || src_bytes > src->bytes - src_offset ||
        dst_offset > dst->bytes || dst_bytes > dst->bytes - dst_offset ||
        ds4_tensor_device_idx(dst) != ds4_tensor_device_idx(src)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(dst);
    if (ds4_gpu_set_current_device(tier) != 0) return 0;
    const uint64_t blocks = (count + 255u) / 256u;
    if (blocks > UINT32_MAX) return 0;
    f32_to_f16_kernel<<<(unsigned)blocks, 256>>>(
            (__half *)((char *)dst->ptr + dst_offset),
            (const float *)((const char *)src->ptr + src_offset),
            count);
    return cuda_ok(cudaGetLastError(), "tensor f32-to-f16 copy launch");
}

extern "C" int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                             uint64_t offset,
                                             void *data,
                                             uint64_t bytes,
                                             uint64_t event_value,
                                             const char *label) {
    (void)event_value;
    if (!tensor || !data || offset > tensor->bytes ||
        bytes > tensor->bytes - offset) {
        return 0;
    }
    if (!cuda_ok(cudaDeviceSynchronize(),
                 label ? label : "selected readback wait")) {
        return 0;
    }
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset,
                              (size_t)bytes, cudaMemcpyDeviceToHost),
                   "selected tensor read");
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                               const ds4_gpu_tensor *out_t,
                               ds4_gpu_tensor *in_t,
                               uint64_t bytes) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_tp_big_gate_encode\n");
    return 0;
}

extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_tp_gate_encode\n");
    return 0;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    (void)enabled;   /* Mac network-TP head split: no-op on CUDA */
}

extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    (void)event_value;
    return cuda_ok(cudaDeviceSynchronize(),
                   label ? label : "selected readback wait");
}

/* Compatibility surface shared with the canonical Metal/ROCm graph. CUDA
 * either delegates to its equivalent primitive or reports an unavailable
 * optional fast path so the graph can use its established fallback. */
extern "C" int ds4_gpu_commit_and_wait_selected_readback(
        uint64_t event_value, const char *label) {
    (void)event_value;
    return cuda_ok(cudaDeviceSynchronize(),
                   label ? label : "selected readback wait");
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    const int ok = ds4_gpu_set_model_fd(fd);
    if (ok) g_model_fd_host_base = model_map;
    return ok;
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    return 0;
}

extern "C" int ds4_gpu_preload_q4_expert_tables(
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
        uint32_t n_total_expert) {
    (void)model_map; (void)model_size;
    (void)gate_offset; (void)up_offset; (void)down_offset;
    (void)gate_expert_bytes; (void)down_expert_bytes;
    (void)n_total_expert;
    return 1;
}

extern "C" void ds4_gpu_set_glm_model(bool enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    cuda_stream_selected_cache_invalidate();
    if (!g_ssd_streaming_mode) cuda_stream_selected_cache_release();
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    (void)experts;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_stream_selected_cache.valid ?
        g_stream_selected_cache.compact_count : 0;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_selected_cache_release();
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    (void)table; (void)selected_ids; (void)n_selected;
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids,
        uint32_t n_tokens,
        uint32_t n_selected) {
    if (n_tokens == 0 || n_selected == 0 ||
        (uint64_t)n_tokens * n_selected > UINT32_MAX) {
        return 0;
    }
    return cuda_stream_selected_cache_begin_load(
            table, selected_ids, n_tokens * n_selected);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *expert_ids,
        const uint32_t *expert_priorities,
        uint32_t n_experts) {
    (void)table; (void)expert_ids; (void)expert_priorities; (void)n_experts;
    return 1;
}

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t n_vocab) {
    return ds4_gpu_indexer_topk_tensor(out_idx, logits, n_vocab, 1u, 1u);
}

extern "C" int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_vocab,
        uint32_t token,
        uint32_t n_embd) {
    return ds4_gpu_embed_token_quant_tensor(out, model_map, model_size,
                                            weight_offset, 8u, n_vocab,
                                            token, n_embd);
}

extern "C" int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *tokens,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd) {
    return ds4_gpu_embed_tokens_quant_tensor(out, tokens, model_map,
                                             model_size, weight_offset, 8u,
                                             n_vocab, n_tokens, n_embd);
}

extern "C" int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *kv_norm,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_tokens,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t n_head) {
    return ds4_gpu_glm_k_b_project_typed_tensor(
            out, kv_norm, model_map, model_size, weight_offset, 8u,
            n_tokens, kv_lora_dim, qk_nope, n_head);
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t full_in_dim,
        uint64_t k_off,
        uint64_t k_cnt,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!x || x_elem_off > x->bytes / sizeof(float) ||
        k_cnt > x->bytes / sizeof(float) - x_elem_off) {
        return 0;
    }
    ds4_gpu_tensor x_slice = *x;
    x_slice.ptr = (char *)x->ptr + x_elem_off * sizeof(float);
    x_slice.bytes = k_cnt * sizeof(float);
    x_slice.owner = 0;
    return ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            out, model_map, model_size, weight_offset,
            full_in_dim, out_dim, k_off, k_cnt, &x_slice, 1u);
}

extern "C" int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t weight_type,
        uint64_t full_in_dim,
        uint64_t k_off,
        uint64_t k_cnt,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_matmul_q8_0_kslice_tensor(
            out, model_map, model_size, weight_offset,
            full_in_dim, k_off, k_cnt, out_dim, x, x_elem_off);
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

extern "C" int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *q_half,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps) {
    (void)out; (void)q_half; (void)model_map; (void)model_size;
    (void)weight_offset; (void)in_dim; (void)out_dim; (void)x;
    (void)n_tok; (void)n_head; (void)head_dim; (void)n_rot; (void)pos0;
    (void)n_ctx_orig; (void)inverse; (void)freq_base; (void)freq_scale;
    (void)ext_factor; (void)attn_factor; (void)beta_fast; (void)beta_slow;
    (void)eps;
    return 0;
}

extern "C" int ds4_gpu_attention_prefill_raw_heads_range_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t q_row0, uint32_t n_q,
        uint32_t n_kv, uint32_t window, uint32_t n_head,
        uint32_t head_dim) {
    (void)heads; (void)model_map; (void)model_size; (void)sinks_offset;
    (void)q; (void)raw_kv; (void)q_row0; (void)n_q; (void)n_kv;
    (void)window; (void)n_head; (void)head_dim;
    return 0;
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, uint32_t q_row0, uint32_t n_q,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    (void)heads; (void)model_map; (void)model_size; (void)sinks_offset;
    (void)q; (void)raw_kv; (void)comp_kv; (void)comp_kv_f16;
    (void)q_row0; (void)n_q; (void)n_tokens; (void)n_comp; (void)window;
    (void)ratio; (void)n_head; (void)head_dim;
    return 0;
}

extern "C" int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor *out_h, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)out_h; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset, uint32_t out_b_type,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)out; (void)low; (void)group_tmp; (void)low_tmp;
    (void)model_map; (void)model_size; (void)out_a_offset;
    (void)out_b_offset; (void)out_b_type; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t group0, uint32_t group_cnt,
        const ds4_gpu_tensor *heads) {
    (void)low; (void)model_map; (void)model_size; (void)out_a_offset;
    (void)group_dim; (void)rank; (void)group0; (void)group_cnt;
    (void)heads;
    return 0;
}

extern "C" int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out_h; (void)residual_hc; (void)split;
    (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add_h;
    (void)residual_hc; (void)split; (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    (void)suspend;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" uint64_t ds4_gpu_tp_big_gate_kick(
        uint32_t layer, uint32_t rows, const ds4_gpu_tensor *out_t,
        ds4_gpu_tensor *in_t, uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_tp_big_gate_wait(uint64_t seq) {
    (void)seq;
    return 0;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    return 0;
}
#pragma GCC diagnostic pop
