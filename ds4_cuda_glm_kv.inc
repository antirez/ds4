/* ============================================================================
 * GLM-5.2 KV-write path kernels, ported from metal/dsv4_misc.metal
 * (kernel_glm_kv_lora_rms_norm, kernel_glm_k_b_project_q8_0,
 *  kernel_glm_store_compact_kv, kernel_glm_qkv_norm_store_compact_kv,
 *  kernel_glm_store_indexer_k, kernel_glm_build_kv_cache[_flash]).
 *
 * This file is #included near the end of ds4_cuda.cu and relies on helpers
 * defined there: cuda_ok(), cuda_model_range_ptr(), rope_yarn_ramp_dev(),
 * struct ds4_gpu_tensor.
 * ============================================================================ */

/* ---------------------------------------------------------------------------
 * Device helpers.
 * ------------------------------------------------------------------------- */

__device__ static void glm_cache_store_f32_or_f16_dev(
        char *base,
        uint64_t index,
        uint32_t cache_f16,
        float x) {
    if (cache_f16 != 0u) {
        ((__half *)base)[index] = __float2half(x);
    } else {
        ((float *)base)[index] = x;
    }
}

/* YaRN correction dims, Metal glm_rope_yarn_corr_dims equivalent. */
__device__ static void glm_rope_yarn_corr_dims_dev(
        uint32_t n_dims,
        uint32_t n_ctx_orig,
        float freq_base,
        float beta_fast,
        float beta_slow,
        float *corr0,
        float *corr1) {
    const float denom = 2.0f * logf(freq_base);
    const float c0 = floorf((float)n_dims *
            logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
    const float c1 = ceilf((float)n_dims *
            logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
    *corr0 = fmaxf(0.0f, c0);
    *corr1 = fminf((float)n_dims - 1.0f, c1);
}

/* Metal glm_rope_yarn equivalent (reuses rope_yarn_ramp_dev from ds4_cuda.cu). */
__device__ static void glm_rope_yarn_dev(
        float theta_extrap,
        float freq_scale,
        float corr0,
        float corr1,
        int i0,
        float ext_factor,
        float mscale,
        float *cos_theta,
        float *sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

/* ---------------------------------------------------------------------------
 * kernel_glm_kv_lora_rms_norm: RMS norm of the first kv_lora_dim elements of
 * each kv_raw row (row stride kv_raw_dim) into a dense [n_tokens, kv_lora_dim]
 * output, scaled by a norm weight.
 * ------------------------------------------------------------------------- */

__global__ static void glm_kv_lora_rms_norm_kernel(
        float *out,
        const float *kv_raw,
        const float *w,
        uint32_t n_tokens,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_tokens) return;
    const float *x = kv_raw + (uint64_t)row * kv_raw_dim;
    float *o = out + (uint64_t)row * kv_lora_dim;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)kv_lora_dim + eps);
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        o[i] = x[i] * inv * w[i];
    }
}

extern "C" int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_raw,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps) {
    if (!out || !kv_raw || !model_map ||
        n_tokens == 0 || kv_raw_dim == 0 || kv_lora_dim == 0 ||
        kv_lora_dim > kv_raw_dim || (kv_lora_dim & 3u) != 0 ||
        !isfinite(eps) || eps < 0.0f) {
        return 0;
    }
    const uint64_t raw_bytes = (uint64_t)n_tokens * kv_raw_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * kv_lora_dim * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)kv_lora_dim * sizeof(float);
    if (kv_raw->bytes < raw_bytes || out->bytes < out_bytes) return 0;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset,
                                                         weight_bytes,
                                                         "glm kv_lora rms weight");
    if (!w) return 0;
    glm_kv_lora_rms_norm_kernel<<<n_tokens, 256>>>(
            (float *)out->ptr, (const float *)kv_raw->ptr, w,
            n_tokens, kv_raw_dim, kv_lora_dim, eps);
    return cuda_ok(cudaGetLastError(), "glm kv_lora rms norm launch");
}

/* ---------------------------------------------------------------------------
 * kernel_glm_k_b_project_q8_0: per (token, head) projection of the normalized
 * kv_lora row through a q8_0 weight laid out as [n_head * kv_lora_dim] rows of
 * qk_nope elements each (34-byte blocks of 32).  out[token][head][qk_nope].
 * Block shape mirrors Metal: (32 lanes, q_blocks) with the kv row staged in
 * shared memory.
 * ------------------------------------------------------------------------- */

__global__ static void glm_k_b_project_q8_0_kernel(
        float *out,
        const unsigned char *weight,
        const float *kv_norm,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t row_bytes) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;

    extern __shared__ float glm_kb_kv_scratch[];
    const uint32_t nth = blockDim.x * blockDim.y;
    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const float *kv = kv_norm + (uint64_t)token * kv_lora_dim;
    float *o = out + ((uint64_t)token * n_head + head) * qk_nope;

    for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
        glm_kb_kv_scratch[j] = kv[j];
    }
    __syncthreads();

    const uint32_t block = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t q = (block << 5) + lane;
    if (q < qk_nope) {
        float acc = 0.0f;
        const unsigned char *hbase = weight + (uint64_t)head * kv_lora_dim * row_bytes;
        for (uint32_t j = 0; j < kv_lora_dim; j++) {
            const unsigned char *row = hbase + (uint64_t)j * row_bytes;
            const __half *dptr = (const __half *)(row + (uint64_t)block * 34u);
            const int8_t *qs = (const int8_t *)(row + (uint64_t)block * 34u + 2u);
            acc += __half2float(*dptr) * (float)qs[lane] * glm_kb_kv_scratch[j];
        }
        o[q] = acc;
    }
}

extern "C" int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head) {
    if (!out || !kv_norm || !model_map ||
        n_tokens == 0 || kv_lora_dim == 0 || qk_nope == 0 || n_head == 0) {
        return 0;
    }
    const uint64_t row_bytes = (((uint64_t)qk_nope + 31u) / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)n_head * kv_lora_dim * row_bytes;
    const uint64_t kv_bytes = (uint64_t)n_tokens * kv_lora_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tokens * n_head * qk_nope * sizeof(float);
    if (kv_norm->bytes < kv_bytes || out->bytes < out_bytes) return 0;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const uint32_t q_blocks = (qk_nope + 31u) / 32u;
    if (q_blocks > 8u) return 0; /* mirrors the Metal tiled-kernel limit */
    const unsigned char *w = (const unsigned char *)
        cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "glm k_b q8_0");
    if (!w) return 0;
    dim3 grid(n_tokens, n_head, 1);
    dim3 block(32, q_blocks, 1);
    glm_k_b_project_q8_0_kernel<<<grid, block, kv_lora_dim * sizeof(float)>>>(
            (float *)out->ptr, w, (const float *)kv_norm->ptr,
            n_tokens, n_head, kv_lora_dim, qk_nope, (uint32_t)row_bytes);
    return cuda_ok(cudaGetLastError(), "glm k_b project launch");
}

/* ---------------------------------------------------------------------------
 * kernel_glm_store_compact_kv: copies the normalized kv_lora row and the raw
 * rope tail into the compact caches at pos0+token.  blockIdx.y selects the
 * part (0 = kv_lora, 1 = rope tail), as in the Metal dispatch.
 * ------------------------------------------------------------------------- */

__global__ static void glm_store_compact_kv_kernel(
        const float *kv_norm,
        const float *kv_raw,
        char *kv_lora_cache,
        char *k_rope_cache,
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

    if (part == 0u) {
        const float *src = kv_norm + (uint64_t)token * kv_lora_dim;
        for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
            glm_cache_store_f32_or_f16_dev(kv_lora_cache,
                                           (uint64_t)pos * kv_lora_dim + i,
                                           cache_f16, src[i]);
        }
    } else {
        const float *src = kv_raw + (uint64_t)token * kv_raw_dim + kv_lora_dim;
        for (uint32_t i = threadIdx.x; i < qk_rope; i += blockDim.x) {
            glm_cache_store_f32_or_f16_dev(k_rope_cache,
                                           (uint64_t)pos * qk_rope + i,
                                           cache_f16, src[i]);
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
        n_tokens == 0 || cache_cap == 0 ||
        kv_raw_dim == 0 || kv_lora_dim == 0 || qk_rope == 0 ||
        kv_lora_dim > kv_raw_dim ||
        qk_rope > kv_raw_dim - kv_lora_dim ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    if (kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * cache_elem_bytes ||
        kv_norm->bytes < (uint64_t)n_tokens * kv_lora_dim * sizeof(float) ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float)) {
        return 0;
    }
    dim3 grid(n_tokens, 2, 1);
    glm_store_compact_kv_kernel<<<grid, 256>>>(
            (const float *)kv_norm->ptr, (const float *)kv_raw->ptr,
            (char *)kv_lora_cache->ptr, (char *)k_rope_cache->ptr,
            pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope,
            cache_f16 ? 1u : 0u);
    return cuda_ok(cudaGetLastError(), "glm store compact kv launch");
}

/* ---------------------------------------------------------------------------
 * kernel_glm_qkv_norm_store_compact_kv: fused RMS norm of the q row (part 0)
 * and of the kv_lora prefix of the raw row (part 1, stored to the compact
 * cache), plus a rope-tail copy (part 2).  blockIdx.y selects the part.
 * ------------------------------------------------------------------------- */

__global__ static void glm_qkv_norm_store_compact_kv_kernel(
        float *q_dst,
        const float *q_src,
        const float *q_weight,
        const float *kv_raw,
        const float *kv_weight,
        char *kv_lora_cache,
        char *k_rope_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t q_n,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope,
        uint32_t cache_f16,
        float eps) {
    const uint32_t token = blockIdx.x;
    const uint32_t part = blockIdx.y;
    if (token >= n_tokens || part > 2u) return;

    if (part == 2u) {
        const uint32_t pos = pos0 + token;
        if (pos >= cache_cap) return;
        const float *src = kv_raw + (uint64_t)token * kv_raw_dim + kv_lora_dim;
        for (uint32_t i = threadIdx.x; i < qk_rope; i += blockDim.x) {
            glm_cache_store_f32_or_f16_dev(k_rope_cache,
                                           (uint64_t)pos * qk_rope + i,
                                           cache_f16, src[i]);
        }
        return;
    }

    const bool kv_task = part != 0u;
    const uint32_t n = kv_task ? kv_lora_dim : q_n;
    const float *x = kv_task ? kv_raw + (uint64_t)token * kv_raw_dim
                             : q_src + (uint64_t)token * q_n;
    const float *w = kv_task ? kv_weight : q_weight;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);

    if (!kv_task) {
        float *y = q_dst + (uint64_t)token * q_n;
        for (uint32_t i = threadIdx.x; i < q_n; i += blockDim.x) {
            y[i] = (x[i] * scale) * w[i];
        }
        return;
    }

    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        glm_cache_store_f32_or_f16_dev(kv_lora_cache,
                                       (uint64_t)pos * kv_lora_dim + i,
                                       cache_f16, (x[i] * scale) * w[i]);
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
        !model_map || n_tokens == 0 || cache_cap == 0 ||
        q_n == 0 || kv_raw_dim == 0 || kv_lora_dim == 0 || qk_rope == 0 ||
        (q_n & 3u) != 0 || (kv_lora_dim & 3u) != 0 ||
        kv_lora_dim > kv_raw_dim ||
        qk_rope > kv_raw_dim - kv_lora_dim ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    const uint64_t q_row_bytes = (uint64_t)q_n * sizeof(float);
    const uint64_t kv_weight_bytes = (uint64_t)kv_lora_dim * sizeof(float);
    if (q->bytes < q_row_bytes * n_tokens ||
        q_out->bytes < q_row_bytes * n_tokens ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * cache_elem_bytes ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float)) {
        return 0;
    }
    if (q_weight_offset > model_size || q_row_bytes > model_size - q_weight_offset ||
        kv_weight_offset > model_size || kv_weight_bytes > model_size - kv_weight_offset) {
        return 0;
    }
    const float *q_w = (const float *)cuda_model_range_ptr(model_map, q_weight_offset,
                                                           q_row_bytes,
                                                           "glm q rms weight");
    const float *kv_w = (const float *)cuda_model_range_ptr(model_map, kv_weight_offset,
                                                            kv_weight_bytes,
                                                            "glm kv rms weight");
    if (!q_w || !kv_w) return 0;
    dim3 grid(n_tokens, 3, 1);
    glm_qkv_norm_store_compact_kv_kernel<<<grid, 256>>>(
            (float *)q_out->ptr, (const float *)q->ptr, q_w,
            (const float *)kv_raw->ptr, kv_w,
            (char *)kv_lora_cache->ptr, (char *)k_rope_cache->ptr,
            pos0, n_tokens, cache_cap, q_n, kv_raw_dim, kv_lora_dim, qk_rope,
            cache_f16 ? 1u : 0u, eps);
    return cuda_ok(cudaGetLastError(), "glm qkv norm compact store launch");
}

/* ---------------------------------------------------------------------------
 * kernel_glm_store_indexer_k: LayerNorm (mean/variance, weight + bias) over
 * the raw indexer K row, partial rope over the first rot_dim dims in adjacent
 * pairs, stored to the indexer key cache at pos0+token.
 * ------------------------------------------------------------------------- */

__global__ static void glm_store_indexer_k_kernel(
        const float *raw_k,
        const float *w,
        const float *b,
        char *indexer_key_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t n_ctx_orig,
        uint32_t cache_f16,
        float eps,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;

    const float *src = raw_k + (uint64_t)token * head_dim;
    __shared__ float partial[256];

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sum += src[i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = partial[0] / (float)head_dim;
    __syncthreads();

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)head_dim + eps);

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims_dev(rot_dim, n_ctx_orig, freq_base,
                                    beta_fast, beta_slow, &corr0, &corr1);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)rot_dim;

    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        if (i < rot_dim) {
            if ((i & 1u) != 0u) continue;
            const float theta = theta_base * powf(freq_base, inv_ndims * (float)i);
            float cos_theta, sin_theta;
            glm_rope_yarn_dev(theta, freq_scale, corr0, corr1, (int)i,
                              ext_factor, attn_factor, &cos_theta, &sin_theta);
            const float x0 = (src[i] - mean) * inv * w[i] + b[i];
            const uint32_t j = i + 1u;
            const float x1 = (src[j] - mean) * inv * w[j] + b[j];
            glm_cache_store_f32_or_f16_dev(indexer_key_cache,
                                           (uint64_t)pos * head_dim + i,
                                           cache_f16,
                                           x0 * cos_theta - x1 * sin_theta);
            glm_cache_store_f32_or_f16_dev(indexer_key_cache,
                                           (uint64_t)pos * head_dim + j,
                                           cache_f16,
                                           x0 * sin_theta + x1 * cos_theta);
        } else {
            const float x = (src[i] - mean) * inv * w[i] + b[i];
            glm_cache_store_f32_or_f16_dev(indexer_key_cache,
                                           (uint64_t)pos * head_dim + i,
                                           cache_f16, x);
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
    if (!indexer_key_cache || !raw_k || !model_map ||
        n_tokens == 0 || cache_cap == 0 ||
        head_dim == 0 || rot_dim == 0 ||
        rot_dim > head_dim || (rot_dim & 1u) != 0 ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (indexer_key_cache->bytes < (uint64_t)cache_cap * head_dim * cache_elem_bytes ||
        raw_k->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) {
        return 0;
    }
    if (weight_offset > model_size || norm_bytes > model_size - weight_offset ||
        bias_offset > model_size || norm_bytes > model_size - bias_offset) {
        return 0;
    }
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset,
                                                         norm_bytes,
                                                         "glm indexer k norm weight");
    const float *b = (const float *)cuda_model_range_ptr(model_map, bias_offset,
                                                         norm_bytes,
                                                         "glm indexer k norm bias");
    if (!w || !b) return 0;
    glm_store_indexer_k_kernel<<<n_tokens, 256>>>(
            (const float *)raw_k->ptr, w, b, (char *)indexer_key_cache->ptr,
            pos0, n_tokens, cache_cap, head_dim, rot_dim, n_ctx_orig,
            cache_f16 ? 1u : 0u, eps,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "glm store indexer k launch");
}

/* ---------------------------------------------------------------------------
 * kernel_glm_build_kv_cache[_flash]: per (token, head) dense KV cache build.
 * Copies the per-head k_nope, ropes the shared kv_raw rope tail into the key
 * row after qk_nope, and copies the per-head value row.  The flash variant
 * additionally stages f16 copies of K and V laid out [head][token][dim] for
 * the staged flash-attention prefill kernel.
 *
 * ponytail: the Metal decode_group4 variant is a perf-only specialization
 * with identical math; the generic kernel covers decode too. Add it back if
 * decode KV-build shows up in profiles.
 * ------------------------------------------------------------------------- */

__global__ static void glm_build_kv_cache_kernel(
        const float *kv_raw,
        const float *k_nope,
        const float *value,
        char *key_cache,
        char *value_cache,
        __half *key_f16,    /* optional staging, may be NULL */
        __half *value_f16,  /* optional staging, may be NULL */
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        uint32_t cache_f16,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;

    const uint32_t nth = blockDim.x;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint32_t pos = pos0 + token;
    const float *raw = kv_raw + (uint64_t)token * kv_raw_dim;
    const float *kn = k_nope + ((uint64_t)token * n_head + head) * qk_nope;
    const float *val = value + ((uint64_t)token * n_head + head) * value_dim;
    const uint64_t kbase = ((uint64_t)pos * n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * n_head + head) * value_dim;
    __half *kdst_f16 = key_f16 ?
        key_f16 + ((uint64_t)head * n_tokens + token) * qk_dim : NULL;
    __half *vdst_f16 = value_f16 ?
        value_f16 + ((uint64_t)head * n_tokens + token) * value_dim : NULL;

    for (uint32_t i = threadIdx.x; i < qk_nope; i += nth) {
        const float x = kn[i];
        glm_cache_store_f32_or_f16_dev(key_cache, kbase + i, cache_f16, x);
        if (kdst_f16) kdst_f16[i] = __float2half(x);
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims_dev(qk_rope, n_ctx_orig, freq_base,
                                    beta_fast, beta_slow, &corr0, &corr1);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)qk_rope;
    for (uint32_t r = threadIdx.x * 2u; r < qk_rope; r += nth * 2u) {
        const float theta = theta_base * powf(freq_base, inv_ndims * (float)r);
        float cos_theta, sin_theta;
        glm_rope_yarn_dev(theta, freq_scale, corr0, corr1, (int)r,
                          ext_factor, attn_factor, &cos_theta, &sin_theta);
        const uint32_t src0 = kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint32_t dst0 = qk_nope + r;
        const float y0 = x0 * cos_theta - x1 * sin_theta;
        const float y1 = x0 * sin_theta + x1 * cos_theta;
        glm_cache_store_f32_or_f16_dev(key_cache, kbase + dst0, cache_f16, y0);
        glm_cache_store_f32_or_f16_dev(key_cache, kbase + dst0 + 1u, cache_f16, y1);
        if (kdst_f16) {
            kdst_f16[dst0] = __float2half(y0);
            kdst_f16[dst0 + 1u] = __float2half(y1);
        }
    }

    for (uint32_t i = threadIdx.x; i < value_dim; i += nth) {
        const float x = val[i];
        glm_cache_store_f32_or_f16_dev(value_cache, vbase + i, cache_f16, x);
        if (vdst_f16) vdst_f16[i] = __float2half(x);
    }
}

/* Shared host-side validation for both build variants. */
static int glm_build_kv_cache_args_ok(
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        bool cache_f16) {
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!key_cache || !value_cache || !kv_raw || !k_nope || !value ||
        n_tokens == 0 || cache_cap == 0 || n_head == 0 ||
        kv_raw_dim == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || value_dim == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        qk_dim < qk_nope || (qk_rope & 1u) != 0 ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0 ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    if (key_cache->bytes < (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes ||
        value_cache->bytes < (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float) ||
        k_nope->bytes < (uint64_t)n_tokens * n_head * qk_nope * sizeof(float) ||
        value->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float)) {
        return 0;
    }
    return 1;
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
    if (!glm_build_kv_cache_args_ok(key_cache, value_cache, kv_raw, k_nope, value,
                                    pos0, n_tokens, cache_cap, n_head,
                                    kv_raw_dim, kv_lora_dim, qk_nope, qk_rope,
                                    value_dim, freq_base, freq_scale, ext_factor,
                                    attn_factor, beta_fast, beta_slow, cache_f16)) {
        return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    glm_build_kv_cache_kernel<<<grid, 256>>>(
            (const float *)kv_raw->ptr, (const float *)k_nope->ptr,
            (const float *)value->ptr,
            (char *)key_cache->ptr, (char *)value_cache->ptr,
            NULL, NULL,
            pos0, n_tokens, n_head, kv_raw_dim, kv_lora_dim,
            qk_nope, qk_rope, value_dim, n_ctx_orig, cache_f16 ? 1u : 0u,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "glm build kv cache launch");
}

/* Persistent f16 K/V staging scratch for the staged flash-attention prefill
 * path (CUDA mirror of Metal's g_flash_attn_kv_buffer).  Layout: keys at
 * offset 0 as [n_head][n_tokens][qk_dim] halves, values right after the keys
 * as [n_head][n_tokens][value_dim] halves.
 * TODO(port): ds4_gpu_glm_attention_flash_staged_tensor (attention cluster)
 * must consume this same buffer; it is intentionally non-static so the
 * attention .inc can reference it. */
void *g_glm_flash_attn_kv_buffer = NULL;
uint64_t g_glm_flash_attn_kv_bytes = 0;

static int glm_flash_attn_kv_scratch_ensure(uint64_t bytes) {
    if (g_glm_flash_attn_kv_bytes >= bytes) return 1;
    if (g_glm_flash_attn_kv_buffer) {
        (void)cudaFree(g_glm_flash_attn_kv_buffer);
        g_glm_flash_attn_kv_buffer = NULL;
        g_glm_flash_attn_kv_bytes = 0;
    }
    void *ptr = NULL;
    if (!cuda_ok(cudaMalloc(&ptr, (size_t)bytes), "glm flash kv scratch alloc")) {
        return 0;
    }
    g_glm_flash_attn_kv_buffer = ptr;
    g_glm_flash_attn_kv_bytes = bytes;
    return 1;
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
    if (pos0 != 0) return 0; /* staged flash path is prefill-from-zero only */
    if (!glm_build_kv_cache_args_ok(key_cache, value_cache, kv_raw, k_nope, value,
                                    pos0, n_tokens, cache_cap, n_head,
                                    kv_raw_dim, kv_lora_dim, qk_nope, qk_rope,
                                    value_dim, freq_base, freq_scale, ext_factor,
                                    attn_factor, beta_fast, beta_slow, cache_f16)) {
        return 0;
    }
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint64_t key_f16_bytes =
        (uint64_t)n_tokens * n_head * qk_dim * sizeof(uint16_t);
    const uint64_t value_f16_bytes =
        (uint64_t)n_tokens * n_head * value_dim * sizeof(uint16_t);
    if (!glm_flash_attn_kv_scratch_ensure(key_f16_bytes + value_f16_bytes)) return 0;
    __half *key_f16 = (__half *)g_glm_flash_attn_kv_buffer;
    __half *value_f16 = (__half *)((char *)g_glm_flash_attn_kv_buffer + key_f16_bytes);
    dim3 grid(n_tokens, n_head, 1);
    glm_build_kv_cache_kernel<<<grid, 256>>>(
            (const float *)kv_raw->ptr, (const float *)k_nope->ptr,
            (const float *)value->ptr,
            (char *)key_cache->ptr, (char *)value_cache->ptr,
            key_f16, value_f16,
            pos0, n_tokens, n_head, kv_raw_dim, kv_lora_dim,
            qk_nope, qk_rope, value_dim, n_ctx_orig, cache_f16 ? 1u : 0u,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "glm build kv cache flash launch");
}
