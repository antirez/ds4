/* ds4_cuda_glm_attn.inc - CUDA port of the GLM-5.2 attention core.
 *
 * Included near the end of ds4_cuda.cu: relies on cuda_ok(),
 * cuda_model_range_ptr(), struct ds4_gpu_tensor {ptr,bytes} and the standard
 * CUDA headers that file already includes.
 *
 * Ported from the Metal launchers in ds4_metal.m and the shaders in
 * metal/dsv4_misc.metal (kernel_glm_attention_indexed_decode,
 * kernel_glm_attention_indexed_decode_split_group8_*,
 * kernel_glm_attention_indexed_batch, the *_lora_group8_vec[_causal] family,
 * kernel_glm_attention_full and kernel_glm_fill_selected_range[_batch]).
 *
 * ponytail: the Metal backend ships several tuned variants per entry point
 * (group2/group8/q2_group4/vec/fullheads).  Each CUDA entry point below is a
 * single kernel implementing the same math; add tuned variants only if
 * profiling demands it.
 */

#include <float.h>

/* ------------------------------------------------------------------------
 * Device helpers (glm_attn_ prefix to stay clash-free inside ds4_cuda.cu).
 * ------------------------------------------------------------------------ */

__device__ static float glm_attn_cache_load(
        const char *base, uint64_t index, uint32_t cache_f16) {
    if (cache_f16) return __half2float(((const __half *)base)[index]);
    return ((const float *)base)[index];
}

/* Port of glm_rope_yarn_corr_dims() (metal/dsv4_misc.metal:514). */
__device__ static void glm_attn_rope_corr_dims(
        int n_dims, int n_ctx_orig, float freq_base,
        float beta_fast, float beta_slow, float *c0, float *c1) {
    const float denom = 2.0f * logf(freq_base);
    const float f_fast = (float)n_dims *
        logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom;
    const float f_slow = (float)n_dims *
        logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom;
    *c0 = fmaxf(0.0f, floorf(f_fast));
    *c1 = fminf((float)n_dims - 1.0f, ceilf(f_slow));
}

/* Port of glm_rope_yarn() (metal/dsv4_misc.metal:490). */
__device__ static void glm_attn_rope_yarn(
        float theta_extrap, float freq_scale, float c0, float c1, int i0,
        float ext_factor, float mscale, float *cos_theta, float *sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = ((float)(i0 / 2) - c0) / fmaxf(0.001f, c1 - c0);
        const float ramp_mix =
            (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

/* Port of glm_cache_load_rotated_rope_pair() (metal/dsv4_misc.metal:1455):
 * loads K-rope cache pair (r, r+1) of `row` and rotates it by the row
 * position at attention time. */
__device__ static float2 glm_attn_rope_pair(
        const char *base, uint64_t rope_base, uint32_t r, uint32_t row,
        uint32_t qk_rope, uint32_t cache_f16, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float c0, float c1) {
    const float inv_ndims = -1.0f / (float)qk_rope;
    const float theta =
        (float)row * powf(freq_base, inv_ndims * (float)r);
    float cos_theta, sin_theta;
    glm_attn_rope_yarn(theta, freq_scale, c0, c1, (int)r,
                       ext_factor, attn_factor, &cos_theta, &sin_theta);
    const float x0 = glm_attn_cache_load(base, rope_base + r, cache_f16);
    const float x1 = glm_attn_cache_load(base, rope_base + r + 1u, cache_f16);
    return make_float2(x0 * cos_theta - x1 * sin_theta,
                       x0 * sin_theta + x1 * cos_theta);
}

/* Port of glm_q8_0_dot_row_tg_f32() (metal/dsv4_misc.metal:1539). `x` may
 * point to shared or global memory. */
__device__ static float glm_attn_q8_dot_row(
        const char *row, const float *x, uint32_t n_cols) {
    float acc = 0.0f;
    const uint32_t n_blocks = (n_cols + 31u) >> 5;
    for (uint32_t block = 0; block < n_blocks; block++) {
        const char *block_base = row + (uint64_t)block * 34u;
        const float d = __half2float(*(const __half *)block_base);
        const int8_t *qs = (const int8_t *)(block_base + 2u);
        const uint32_t base = block << 5;
        const uint32_t count = min(32u, n_cols - base);
        for (uint32_t qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

/* Dynamic shared memory above the 48KB static limit needs an opt-in. */
static int glm_attn_smem_ok(const void *func, uint64_t smem_bytes,
                            const char *what) {
    if (smem_bytes <= 48u * 1024u) return 1;
    if (smem_bytes > (uint64_t)INT_MAX) return 0;
    return cuda_ok(cudaFuncSetAttribute(func,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem_bytes),
                   what);
}

/* ------------------------------------------------------------------------
 * Selection-buffer fills (kernel_glm_fill_selected_range[_batch]).
 * ------------------------------------------------------------------------ */

__global__ static void glm_attn_fill_selected_range_kernel(
        uint32_t *selected, uint32_t n_selected) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_selected) selected[gid] = gid;
}

__global__ static void glm_attn_fill_selected_range_batch_kernel(
        uint32_t *selected, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_selected, uint32_t pad_row) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n_tokens * n_selected;
    if (gid >= total || n_selected == 0u) return;
    const uint32_t token = gid / n_selected;
    const uint32_t slot = gid - token * n_selected;
    const uint32_t visible = pos0 + token + 1u;
    selected[gid] = slot < visible ? slot : pad_row;
}

extern "C" int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected) {
    if (!selected || n_selected == 0) return 0;
    if (selected->bytes < (uint64_t)n_selected * sizeof(uint32_t)) {
        fprintf(stderr, "ds4: CUDA GLM selected range received undersized buffer\n");
        return 0;
    }
    glm_attn_fill_selected_range_kernel<<<(n_selected + 255u) / 256u, 256>>>(
            (uint32_t *)selected->ptr, n_selected);
    return cuda_ok(cudaGetLastError(), "glm fill selected range launch");
}

extern "C" int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row) {
    if (!selected || n_tokens == 0 || n_selected == 0) return 0;
    const uint64_t total = (uint64_t)n_tokens * n_selected;
    if (total / n_tokens != n_selected) return 0;
    if (total > UINT64_MAX / sizeof(uint32_t)) return 0;
    if (total > UINT32_MAX) return 0;
    if (selected->bytes < total * sizeof(uint32_t)) {
        fprintf(stderr, "ds4: CUDA GLM selected range batch received undersized buffer\n");
        return 0;
    }
    glm_attn_fill_selected_range_batch_kernel<<<
            (uint32_t)((total + 255u) / 256u), 256>>>(
            (uint32_t *)selected->ptr, n_tokens, pos0, n_selected, pad_row);
    return cuda_ok(cudaGetLastError(), "glm fill selected range batch launch");
}

/* ------------------------------------------------------------------------
 * Dense fallback attention (kernel_glm_attention_full, mode 2, which is the
 * only mode the Metal launcher dispatches).
 *
 * Grid (n_tokens, n_head), 256 threads.  Dynamic shared memory holds the
 * reduction scratch (256 floats) followed by one score per visible row.
 * ------------------------------------------------------------------------ */

__global__ static void glm_attn_full_kernel(
        float *heads,
        const float *q,
        const char *key_cache,
        const char *value_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        uint32_t n_head,
        uint32_t qk_dim,
        uint32_t value_dim,
        uint32_t cache_f16,
        float scale) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;

    const uint32_t nth = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t visible = min(cache_len, pos0 + token + 1u);
    extern __shared__ float glm_attn_full_sm[];
    float *red = glm_attn_full_sm;
    float *scores = glm_attn_full_sm + 256u;

    const float *qh = q + ((uint64_t)token * n_head + head) * qk_dim;

    for (uint32_t s = tid; s < visible; s += nth) {
        const uint64_t kbase = ((uint64_t)s * n_head + head) * qk_dim;
        float dotv = 0.0f;
        for (uint32_t i = 0; i < qk_dim; i++) {
            dotv += qh[i] * glm_attn_cache_load(key_cache, kbase + i, cache_f16);
        }
        scores[s] = dotv * scale;
    }
    __syncthreads();

    /* Metal's mode-2 kernel runs the softmax normalization serially on one
     * thread; keep that for exact-order parity. */
    if (tid == 0u) {
        float max_score = -INFINITY;
        for (uint32_t s = 0; s < visible; s++) {
            max_score = fmaxf(max_score, scores[s]);
        }
        float sum = 0.0f;
        for (uint32_t s = 0; s < visible; s++) {
            const float w = expf(scores[s] - max_score);
            scores[s] = w;
            sum += w;
        }
        red[0] = fmaxf(sum, 1.0e-20f);
    }
    __syncthreads();

    const float denom = red[0];
    float *out = heads + ((uint64_t)token * n_head + head) * value_dim;
    for (uint32_t d = tid; d < value_dim; d += nth) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < visible; s++) {
            const uint64_t vbase = ((uint64_t)s * n_head + head) * value_dim;
            acc += scores[s] *
                   glm_attn_cache_load(value_cache, vbase + d, cache_f16);
        }
        out[d] = acc / denom;
    }
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
    if (!heads || !q || !key_cache || !value_cache ||
        n_tokens == 0 || cache_len == 0 || cache_cap == 0 ||
        n_head == 0 || qk_dim == 0 || value_dim == 0 ||
        (qk_dim & 3u) != 0 ||
        cache_len > cache_cap ||
        pos0 > cache_len || n_tokens > cache_len - pos0) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    if (heads->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        key_cache->bytes < (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes ||
        value_cache->bytes < (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes) {
        fprintf(stderr, "ds4: CUDA GLM attention received undersized buffers\n");
        return 0;
    }
    const uint64_t smem = (256u + (uint64_t)cache_len) * sizeof(float);
    if (!glm_attn_smem_ok((const void *)glm_attn_full_kernel, smem,
                          "glm full attention smem opt-in")) {
        return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    glm_attn_full_kernel<<<grid, 256, (size_t)smem>>>(
            (float *)heads->ptr,
            (const float *)q->ptr,
            (const char *)key_cache->ptr,
            (const char *)value_cache->ptr,
            pos0, n_tokens, cache_len, n_head, qk_dim, value_dim,
            cache_f16 ? 1u : 0u,
            1.0f / sqrtf((float)qk_dim));
    return cuda_ok(cudaGetLastError(), "glm full attention launch");
}

/* ------------------------------------------------------------------------
 * FlashAttention fallback.
 *
 * The Metal backend stages an f16 copy of the dense KV caches and runs the
 * llama.cpp-style kernel_flash_attn_ext_f16_dk256_dv256 with a causal
 * prefill mask.  Mathematically that is dense causal attention over
 * cache_len rows with scale 1/sqrt(qk_dim).  Here it is computed with a
 * chunked online softmax so no cache_len-sized score buffer is needed
 * (hence no Metal-style 8192-row cap).  KV is read straight from the dense
 * caches instead of an f16 staging copy, which only changes rounding when
 * the cache is f32.
 *
 * Grid (n_tokens, n_head), 256 threads; thread d owns output dim d, which
 * requires value_dim == 256 (the launcher enforces qk_dim == value_dim ==
 * 256, same as Metal).
 * ------------------------------------------------------------------------ */

__global__ static void glm_attn_flash_kernel(
        float *heads,
        const float *q,
        const char *key_cache,
        const char *value_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        uint32_t n_head,
        uint32_t qk_dim,
        uint32_t value_dim,
        uint32_t cache_f16,
        float scale) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t visible = min(cache_len, pos0 + token + 1u);
    const float *qh = q + ((uint64_t)token * n_head + head) * qk_dim;

    __shared__ float w[256];
    __shared__ float red[256];

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float acc = 0.0f; /* output accumulator for dim d == tid */

    for (uint32_t base = 0; base < visible; base += 256u) {
        const uint32_t cn = min(256u, visible - base);

        float score = -FLT_MAX / 2.0f;
        if (tid < cn) {
            const uint32_t s = base + tid;
            const uint64_t kbase = ((uint64_t)s * n_head + head) * qk_dim;
            float dotv = 0.0f;
            for (uint32_t i = 0; i < qk_dim; i++) {
                dotv += qh[i] *
                        glm_attn_cache_load(key_cache, kbase + i, cache_f16);
            }
            score = dotv * scale;
        }

        red[tid] = score;
        __syncthreads();
        for (uint32_t step = 128u; step > 0; step >>= 1) {
            if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
            __syncthreads();
        }
        const float chunk_max = red[0];
        __syncthreads();

        const float new_m = fmaxf(M, chunk_max);
        const float wv = (tid < cn) ? expf(score - new_m) : 0.0f;
        w[tid] = wv;
        red[tid] = wv;
        __syncthreads();
        for (uint32_t step = 128u; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            __syncthreads();
        }
        const float chunk_sum = red[0];

        const float old_scale = expf(M - new_m);
        float a = 0.0f;
        for (uint32_t r = 0; r < cn; r++) {
            const uint64_t vbase =
                ((uint64_t)(base + r) * n_head + head) * value_dim;
            a += w[r] * glm_attn_cache_load(value_cache, vbase + tid, cache_f16);
        }
        acc = acc * old_scale + a;
        S = S * old_scale + chunk_sum;
        M = new_m;
        __syncthreads();
    }

    heads[((uint64_t)token * n_head + head) * value_dim + tid] =
        acc / fmaxf(S, 1.0e-20f);
}

static int glm_attn_flash_launch(
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
    if (!heads || !q || !key_cache || !value_cache ||
        n_tokens == 0 || cache_len == 0 || cache_cap == 0 ||
        n_head == 0 || qk_dim != 256u || value_dim != 256u ||
        cache_len > cache_cap ||
        pos0 > cache_len || n_tokens > cache_len - pos0) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    if (heads->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        key_cache->bytes < (uint64_t)cache_cap * n_head * qk_dim * cache_elem_bytes ||
        value_cache->bytes < (uint64_t)cache_cap * n_head * value_dim * cache_elem_bytes) {
        fprintf(stderr, "ds4: CUDA GLM FlashAttention received undersized buffers\n");
        return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    glm_attn_flash_kernel<<<grid, 256>>>(
            (float *)heads->ptr,
            (const float *)q->ptr,
            (const char *)key_cache->ptr,
            (const char *)value_cache->ptr,
            pos0, n_tokens, cache_len, n_head, qk_dim, value_dim,
            cache_f16 ? 1u : 0u,
            1.0f / sqrtf((float)qk_dim));
    return cuda_ok(cudaGetLastError(), "glm flash attention launch");
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
    return glm_attn_flash_launch(heads, q, key_cache, value_cache,
                                 pos0, n_tokens, cache_len, cache_cap,
                                 n_head, qk_dim, value_dim, cache_f16);
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
    /* Metal's staged variant skips the KV->f16 staging copy because
     * kernel_glm_build_kv_cache_flash already produced it.  That builder
     * also writes the dense key/value caches (see dsv4_misc.metal:1111), so
     * on CUDA the staged entry can read the caches directly: same data, no
     * staging buffer required. */
    if (pos0 != 0 || n_tokens != cache_len) return 0;
    return glm_attn_flash_launch(heads, q, key_cache, value_cache,
                                 pos0, n_tokens, cache_len, cache_cap,
                                 n_head, qk_dim, value_dim, cache_f16);
}

/* ------------------------------------------------------------------------
 * Indexed (DSA) attention over selected compact-KV rows, single fused
 * kernel: scores + softmax + lora accumulation + q8_0 value projection.
 *
 * Port of kernel_glm_attention_indexed_batch (which is the decode kernel
 * kernel_glm_attention_indexed_decode plus a token grid dimension; the
 * decode entry launches it with n_tokens == 1).  Handles f16 and f32
 * caches; the Metal f16 fast path is just a vectorized version of the same
 * math.
 *
 * Grid (n_head, n_tokens), 256 threads.  Dynamic shared memory:
 * red[256] + scores[n_selected] + lora_sum[kv_lora_dim].
 * ------------------------------------------------------------------------ */

__global__ static void glm_attn_indexed_kernel(
        float *heads,
        const float *q,
        const float *qk_low,
        const char *kv_lora_cache,
        const char *k_rope_cache,
        const char *value_weight,
        const uint32_t *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        uint32_t value_row_bytes,
        float scale,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens || n_selected == 0u) return;
    const uint32_t nth = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t qk_dim = qk_nope + qk_rope;

    extern __shared__ float glm_attn_indexed_sm[];
    float *red = glm_attn_indexed_sm;
    float *scores = glm_attn_indexed_sm + 256u;
    float *lora_sum = scores + n_selected;

    const float *qh =
        q + ((uint64_t)token * n_head + head) * qk_dim;
    const float *low =
        qk_low + ((uint64_t)token * n_head + head) * kv_lora_dim;
    const uint32_t *token_selected =
        selected + (uint64_t)token * n_selected;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        glm_attn_rope_corr_dims((int)qk_rope, (int)n_ctx_orig, freq_base,
                                beta_fast, beta_slow, &corr0, &corr1);
    }

    float local_max = -INFINITY;
    for (uint32_t s = tid; s < n_selected; s += nth) {
        const uint32_t row = token_selected[s];
        float score = -INFINITY;
        if (row < cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            for (uint32_t j = 0; j < kv_lora_dim; j++) {
                dotv += low[j] *
                        glm_attn_cache_load(kv_lora_cache, lora_base + j,
                                            cache_f16);
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float2 y = glm_attn_rope_pair(k_rope_cache, rope_base, r,
                                                    row, qk_rope, cache_f16,
                                                    freq_base, freq_scale,
                                                    ext_factor, attn_factor,
                                                    corr0, corr1);
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

    float local_sum = 0.0f;
    for (uint32_t s = tid; s < n_selected; s += nth) {
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

    for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_selected; s++) {
            const uint32_t row = token_selected[s];
            if (row < cache_cap) {
                acc += scores[s] *
                       glm_attn_cache_load(kv_lora_cache,
                                           (uint64_t)row * kv_lora_dim + j,
                                           cache_f16);
            }
        }
        lora_sum[j] = acc / denom;
    }
    __syncthreads();

    float *out =
        heads + ((uint64_t)token * n_head + head) * value_dim;
    for (uint32_t d = tid; d < value_dim; d += nth) {
        const char *row =
            value_weight + ((uint64_t)head * value_dim + d) * value_row_bytes;
        out[d] = glm_attn_q8_dot_row(row, lora_sum, kv_lora_dim);
    }
}

static int glm_attn_indexed_launch(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
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
        float                 beta_slow,
        const char           *label) {
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!heads || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !model_map || !selected ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 ||
        n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0 ||
        value_dim == 0 || qk_dim < qk_nope ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    const uint64_t value_row_bytes = (uint64_t)((kv_lora_dim + 31u) / 32u) * 34u;
    const uint64_t value_weight_bytes =
        (uint64_t)n_head * value_dim * value_row_bytes;
    if (heads->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        qk_low->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * cache_elem_bytes ||
        selected->bytes < (uint64_t)n_tokens * n_selected * sizeof(uint32_t)) {
        fprintf(stderr, "ds4: CUDA GLM indexed attention received undersized buffers\n");
        return 0;
    }
    if (value_weight_offset > model_size ||
        value_weight_bytes > model_size - value_weight_offset) {
        fprintf(stderr, "ds4: CUDA GLM indexed attention value range is outside the mapped model\n");
        return 0;
    }
    const char *value_weight = cuda_model_range_ptr(
            model_map, value_weight_offset, value_weight_bytes,
            "glm attn v_b weights");
    if (!value_weight) return 0;

    const uint64_t smem =
        (256u + (uint64_t)n_selected + kv_lora_dim) * sizeof(float);
    if (!glm_attn_smem_ok((const void *)glm_attn_indexed_kernel, smem,
                          "glm indexed attention smem opt-in")) {
        return 0;
    }
    dim3 grid(n_head, n_tokens, 1);
    glm_attn_indexed_kernel<<<grid, 256, (size_t)smem>>>(
            (float *)heads->ptr,
            (const float *)q->ptr,
            (const float *)qk_low->ptr,
            (const char *)kv_lora_cache->ptr,
            (const char *)k_rope_cache->ptr,
            value_weight,
            (const uint32_t *)selected->ptr,
            n_tokens, n_selected, cache_cap,
            cache_f16 ? 1u : 0u,
            n_head, kv_lora_dim, qk_nope, qk_rope, value_dim, n_ctx_orig,
            (uint32_t)value_row_bytes,
            1.0f / sqrtf((float)qk_dim),
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), label);
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
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
    return glm_attn_indexed_launch(heads, q, qk_low, kv_lora_cache,
                                   k_rope_cache, model_map, model_size,
                                   value_weight_offset, selected,
                                   1u /* n_tokens */, n_selected, cache_cap,
                                   cache_f16, n_head, kv_lora_dim, qk_nope,
                                   qk_rope, value_dim, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor,
                                   attn_factor, beta_fast, beta_slow,
                                   "glm indexed attention decode launch");
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
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
    return glm_attn_indexed_launch(heads, q, qk_low, kv_lora_cache,
                                   k_rope_cache, model_map, model_size,
                                   value_weight_offset, selected,
                                   n_tokens, n_selected, cache_cap,
                                   cache_f16, n_head, kv_lora_dim, qk_nope,
                                   qk_rope, value_dim, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor,
                                   attn_factor, beta_fast, beta_slow,
                                   "glm indexed attention batch launch");
}

/* ------------------------------------------------------------------------
 * Split (two-phase) grouped indexed decode.
 *
 * Phase 1 (partial): one warp per (head, selected-row block) computes an
 * online-softmax partial over its rows and stores the unnormalized lora
 * accumulator plus its (max, sum) pair.  Port of
 * kernel_glm_attention_indexed_decode_split_group8_partial_impl with the
 * same fixed shape (kv_lora_dim == 512, qk_rope == 64, f16 cache):
 * lane l owns lora elements {c*128 + l*4 + k : c in 0..3, k in 0..3}.
 * ponytail: Metal groups 8 heads per threadgroup to share a staged KV tile;
 * here each warp reads global memory directly - same math, add staging only
 * if profiling demands it.
 *
 * Phase 2 (reduce): port of
 * kernel_glm_attention_indexed_decode_split_group8_reduce_impl - merges the
 * per-block partials with softmax rescaling and applies the q8_0 value
 * projection.
 * ------------------------------------------------------------------------ */

__global__ static void glm_attn_split_g8_partial_kernel(
        float *partial_lora,
        float *partial_ms,
        const float *q,
        const float *qk_low,
        const char *kv_lora_cache, /* f16 */
        const char *k_rope_cache,  /* f16 */
        const uint32_t *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t qk_nope,
        uint32_t n_ctx_orig,
        uint32_t block_rows,
        float scale,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t block = blockIdx.y;
    const uint32_t lane = threadIdx.x; /* 32 threads per block */
    if (head >= n_head || n_selected == 0u || block_rows == 0u) return;

    const uint32_t kv_lora_dim = 512u;
    const uint32_t qk_rope = 64u;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint32_t block_start = block * block_rows;
    const uint32_t block_end = min(n_selected, block_start + block_rows);

    const float *qh = q + (uint64_t)head * qk_dim;
    const float *low = qk_low + (uint64_t)head * kv_lora_dim;

    float lowv[16];
#pragma unroll
    for (uint32_t c = 0; c < 4u; c++) {
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) {
            lowv[c * 4u + k] = low[c * 128u + lane * 4u + k];
        }
    }
    float qrope[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < 16u) {
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) {
            qrope[k] = qh[qk_nope + lane * 4u + k];
        }
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        glm_attn_rope_corr_dims((int)qk_rope, (int)n_ctx_orig, freq_base,
                                beta_fast, beta_slow, &corr0, &corr1);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float o[16];
#pragma unroll
    for (uint32_t i = 0; i < 16u; i++) o[i] = 0.0f;

    for (uint32_t idx = block_start; idx < block_end; idx++) {
        const uint32_t row = selected[idx];
        const bool valid_row = row < cache_cap;
        float kvv[16];
        float part = 0.0f;
        if (valid_row) {
            const __half *kv_row =
                (const __half *)kv_lora_cache + (uint64_t)row * kv_lora_dim;
#pragma unroll
            for (uint32_t c = 0; c < 4u; c++) {
#pragma unroll
                for (uint32_t k = 0; k < 4u; k++) {
                    const float v =
                        __half2float(kv_row[c * 128u + lane * 4u + k]);
                    kvv[c * 4u + k] = v;
                    part += lowv[c * 4u + k] * v;
                }
            }
            if (lane < 16u) {
                const uint64_t rope_base = (uint64_t)row * qk_rope;
                const uint32_t r = lane * 4u;
                const float2 y0 = glm_attn_rope_pair(k_rope_cache, rope_base,
                                                     r, row, qk_rope, 1u,
                                                     freq_base, freq_scale,
                                                     ext_factor, attn_factor,
                                                     corr0, corr1);
                const float2 y1 = glm_attn_rope_pair(k_rope_cache, rope_base,
                                                     r + 2u, row, qk_rope, 1u,
                                                     freq_base, freq_scale,
                                                     ext_factor, attn_factor,
                                                     corr0, corr1);
                part += qrope[0] * y0.x + qrope[1] * y0.y +
                        qrope[2] * y1.x + qrope[3] * y1.y;
            }
        } else {
#pragma unroll
            for (uint32_t i = 0; i < 16u; i++) kvv[i] = 0.0f;
        }
        /* warp all-reduce so every lane sees the full dot product */
        for (uint32_t off = 16u; off > 0; off >>= 1) {
            part += __shfl_xor_sync(0xffffffffu, part, off);
        }
        if (valid_row) {
            const float score = part * scale;
            const float new_m = fmaxf(M, score);
            const float old_scale = expf(M - new_m);
            const float row_scale = expf(score - new_m);
#pragma unroll
            for (uint32_t i = 0; i < 16u; i++) {
                o[i] = o[i] * old_scale + kvv[i] * row_scale;
            }
            S = S * old_scale + row_scale;
            M = new_m;
        }
    }

    float *out =
        partial_lora + ((uint64_t)block * n_head + head) * kv_lora_dim;
#pragma unroll
    for (uint32_t c = 0; c < 4u; c++) {
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) {
            out[c * 128u + lane * 4u + k] = o[c * 4u + k];
        }
    }
    if (lane == 0u) {
        float *ms = partial_ms + ((uint64_t)block * n_head + head) * 2u;
        ms[0] = M;
        ms[1] = S;
    }
}

__global__ static void glm_attn_split_g8_reduce_kernel(
        float *heads,
        const float *partial_lora,
        const float *partial_ms,
        const char *value_weight,
        uint32_t n_head,
        uint32_t n_blocks,
        uint32_t value_dim,
        uint32_t value_row_bytes) {
    const uint32_t head = blockIdx.x;
    if (head >= n_head || n_blocks == 0u || n_blocks > 64u) return;

    const uint32_t kv_lora_dim = 512u;
    const uint32_t nth = blockDim.x;
    const uint32_t tid = threadIdx.x;

    __shared__ float red[256];
    __shared__ float block_scale[64];
    __shared__ float lora_sum[512];

    float local_m = -FLT_MAX / 2.0f;
    if (tid < n_blocks) {
        const float *ms = partial_ms + ((uint64_t)tid * n_head + head) * 2u;
        local_m = ms[1] > 0.0f ? ms[0] : -FLT_MAX / 2.0f;
    }
    red[tid] = local_m;
    __syncthreads();

    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }
    const float max_m = red[0];
    __syncthreads();

    float local_denom = 0.0f;
    if (tid < n_blocks) {
        const float *ms = partial_ms + ((uint64_t)tid * n_head + head) * 2u;
        const float s = ms[1];
        const float e = s > 0.0f ? expf(ms[0] - max_m) : 0.0f;
        block_scale[tid] = e;
        local_denom = s * e;
    }
    red[tid] = local_denom;
    __syncthreads();

    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    const float denom = fmaxf(red[0], 1.0e-20f);

    for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint32_t b = 0; b < n_blocks; b++) {
            const float *src =
                partial_lora + ((uint64_t)b * n_head + head) * kv_lora_dim;
            acc += src[j] * block_scale[b];
        }
        lora_sum[j] = acc / denom;
    }
    __syncthreads();

    float *out = heads + (uint64_t)head * value_dim;
    for (uint32_t d = tid; d < value_dim; d += nth) {
        const char *row =
            value_weight + ((uint64_t)head * value_dim + d) * value_row_bytes;
        out[d] = glm_attn_q8_dot_row(row, lora_sum, kv_lora_dim);
    }
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
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
    /* selected_rows_valid only permits skipping the row bounds check; the
     * kernel always checks, which is correct in both cases. */
    (void)selected_rows_valid;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint32_t needed_blocks =
        block_rows != 0u ? (n_selected + block_rows - 1u) / block_rows : 0u;
    if (!heads || !partial_lora || !partial_ms || !q || !qk_low ||
        !kv_lora_cache || !k_rope_cache || !model_map || !selected ||
        n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || (n_head % 8u) != 0 ||
        kv_lora_dim != 512u ||
        qk_nope == 0 || qk_rope != 64u ||
        value_dim == 0 || qk_dim < qk_nope ||
        block_rows == 0u || needed_blocks == 0u ||
        n_blocks < needed_blocks || n_blocks > 64u ||
        !cache_f16 ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }
    const uint64_t value_row_bytes = (uint64_t)(kv_lora_dim / 32u) * 34u;
    const uint64_t value_weight_bytes =
        (uint64_t)n_head * value_dim * value_row_bytes;
    if (heads->bytes < (uint64_t)n_head * value_dim * sizeof(float) ||
        partial_lora->bytes <
            (uint64_t)n_blocks * n_head * kv_lora_dim * sizeof(float) ||
        partial_ms->bytes < (uint64_t)n_blocks * n_head * 2u * sizeof(float) ||
        q->bytes < (uint64_t)n_head * qk_dim * sizeof(float) ||
        qk_low->bytes < (uint64_t)n_head * kv_lora_dim * sizeof(float) ||
        kv_lora_cache->bytes <
            (uint64_t)cache_cap * kv_lora_dim * sizeof(uint16_t) ||
        k_rope_cache->bytes <
            (uint64_t)cache_cap * qk_rope * sizeof(uint16_t) ||
        selected->bytes < (uint64_t)n_selected * sizeof(uint32_t)) {
        fprintf(stderr, "ds4: CUDA GLM split grouped indexed attention received undersized buffers\n");
        return 0;
    }
    if (value_weight_offset > model_size ||
        value_weight_bytes > model_size - value_weight_offset) {
        fprintf(stderr, "ds4: CUDA GLM split grouped indexed attention value range is outside the mapped model\n");
        return 0;
    }
    const char *value_weight = cuda_model_range_ptr(
            model_map, value_weight_offset, value_weight_bytes,
            "glm attn v_b weights");
    if (!value_weight) return 0;

    dim3 partial_grid(n_head, n_blocks, 1);
    glm_attn_split_g8_partial_kernel<<<partial_grid, 32>>>(
            (float *)partial_lora->ptr,
            (float *)partial_ms->ptr,
            (const float *)q->ptr,
            (const float *)qk_low->ptr,
            (const char *)kv_lora_cache->ptr,
            (const char *)k_rope_cache->ptr,
            (const uint32_t *)selected->ptr,
            n_selected, cache_cap, n_head, qk_nope, n_ctx_orig, block_rows,
            1.0f / sqrtf((float)qk_dim),
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
    if (!cuda_ok(cudaGetLastError(),
                 "glm split grouped indexed attention partial launch")) {
        return 0;
    }

    glm_attn_split_g8_reduce_kernel<<<n_head, 256>>>(
            (float *)heads->ptr,
            (const float *)partial_lora->ptr,
            (const float *)partial_ms->ptr,
            value_weight,
            n_head, n_blocks, value_dim, (uint32_t)value_row_bytes);
    return cuda_ok(cudaGetLastError(),
                   "glm split grouped indexed attention reduce launch");
}

/* ------------------------------------------------------------------------
 * Indexed batch attention with lora output: same scores as the indexed
 * kernels but the softmax-weighted sum of kv_lora rows is written directly
 * (no q8_0 value projection).  One kernel covers all three entry points:
 *   - selected list (lora / lora_valid): row = selected[token][s]
 *   - causal (lora_causal, selected == NULL): row = s, s < pos0 + token + 1
 * Port of kernel_glm_attention_indexed_batch_lora_group8_vec_impl and
 * ..._vec_causal_impl (also covering the group8 pad0==1 fallback layout).
 * The "valid" variant only relaxes the row bounds check, which the kernel
 * always performs.
 *
 * Grid (n_head, n_tokens), 256 threads.  Dynamic shared memory:
 * red[256] + scores[n_selected].
 * ------------------------------------------------------------------------ */

__global__ static void glm_attn_batch_lora_kernel(
        float *lora_out,
        const float *q,
        const float *qk_low,
        const char *kv_lora_cache,
        const char *k_rope_cache,
        const uint32_t *selected, /* NULL => causal row = s */
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t n_ctx_orig,
        float scale,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens || n_selected == 0u) return;
    const uint32_t nth = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint32_t count =
        selected ? n_selected : min(n_selected, pos0 + token + 1u);
    if (count == 0u) return;

    extern __shared__ float glm_attn_lora_sm[];
    float *red = glm_attn_lora_sm;
    float *scores = glm_attn_lora_sm + 256u;

    const float *qh =
        q + ((uint64_t)token * n_head + head) * qk_dim;
    const float *low =
        qk_low + ((uint64_t)token * n_head + head) * kv_lora_dim;
    const uint32_t *token_selected =
        selected ? selected + (uint64_t)token * n_selected : NULL;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        glm_attn_rope_corr_dims((int)qk_rope, (int)n_ctx_orig, freq_base,
                                beta_fast, beta_slow, &corr0, &corr1);
    }

    float local_max = -FLT_MAX / 2.0f;
    for (uint32_t s = tid; s < count; s += nth) {
        const uint32_t row = token_selected ? token_selected[s] : s;
        float score = -FLT_MAX / 2.0f;
        if (row < cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            for (uint32_t j = 0; j < kv_lora_dim; j++) {
                dotv += low[j] *
                        glm_attn_cache_load(kv_lora_cache, lora_base + j,
                                            cache_f16);
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float2 y = glm_attn_rope_pair(k_rope_cache, rope_base, r,
                                                    row, qk_rope, cache_f16,
                                                    freq_base, freq_scale,
                                                    ext_factor, attn_factor,
                                                    corr0, corr1);
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

    float local_sum = 0.0f;
    for (uint32_t s = tid; s < count; s += nth) {
        const uint32_t row = token_selected ? token_selected[s] : s;
        /* Invalid rows contribute nothing to the softmax sum, matching the
         * Metal vec kernels' online accumulation. */
        const float w =
            row < cache_cap ? expf(scores[s] - max_score) : 0.0f;
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    __syncthreads();

    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    const float sum = red[0];
    const float inv_s = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float *out =
        lora_out + ((uint64_t)token * n_head + head) * kv_lora_dim;
    for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < count; s++) {
            const uint32_t row = token_selected ? token_selected[s] : s;
            if (row < cache_cap) {
                acc += scores[s] *
                       glm_attn_cache_load(kv_lora_cache,
                                           (uint64_t)row * kv_lora_dim + j,
                                           cache_f16);
            }
        }
        out[j] = acc * inv_s;
    }
}

static int glm_attn_batch_lora_launch(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected, /* NULL => causal */
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
        float                 beta_slow,
        const char           *label) {
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 ||
        n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0 ||
        qk_dim < qk_nope ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow)) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    if (lora_out->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        qk_low->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * cache_elem_bytes ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * cache_elem_bytes ||
        (selected &&
         selected->bytes < (uint64_t)n_tokens * n_selected * sizeof(uint32_t))) {
        fprintf(stderr, "ds4: CUDA GLM indexed batch attention-lora received undersized buffers\n");
        return 0;
    }
    const uint64_t smem = (256u + (uint64_t)n_selected) * sizeof(float);
    if (!glm_attn_smem_ok((const void *)glm_attn_batch_lora_kernel, smem,
                          "glm batch attention-lora smem opt-in")) {
        return 0;
    }
    dim3 grid(n_head, n_tokens, 1);
    glm_attn_batch_lora_kernel<<<grid, 256, (size_t)smem>>>(
            (float *)lora_out->ptr,
            (const float *)q->ptr,
            (const float *)qk_low->ptr,
            (const char *)kv_lora_cache->ptr,
            (const char *)k_rope_cache->ptr,
            selected ? (const uint32_t *)selected->ptr : NULL,
            n_tokens, pos0, n_selected, cache_cap,
            cache_f16 ? 1u : 0u,
            n_head, kv_lora_dim, qk_nope, qk_rope, n_ctx_orig,
            1.0f / sqrtf((float)qk_dim),
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), label);
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_tensor(
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
    if (!selected) return 0;
    return glm_attn_batch_lora_launch(lora_out, q, qk_low, kv_lora_cache,
                                      k_rope_cache, selected,
                                      n_tokens, 0u /* pos0 */, n_selected,
                                      cache_cap, cache_f16, n_head,
                                      kv_lora_dim, qk_nope, qk_rope,
                                      n_ctx_orig, freq_base, freq_scale,
                                      ext_factor, attn_factor,
                                      beta_fast, beta_slow,
                                      "glm indexed batch attention-lora launch");
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
    /* "valid" asserts every selected row is in-bounds; the shared kernel
     * bounds-checks anyway, which is correct for that case too. */
    if (!selected) return 0;
    return glm_attn_batch_lora_launch(lora_out, q, qk_low, kv_lora_cache,
                                      k_rope_cache, selected,
                                      n_tokens, 0u /* pos0 */, n_selected,
                                      cache_cap, cache_f16, n_head,
                                      kv_lora_dim, qk_nope, qk_rope,
                                      n_ctx_orig, freq_base, freq_scale,
                                      ext_factor, attn_factor,
                                      beta_fast, beta_slow,
                                      "glm indexed batch attention-lora valid launch");
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
    /* The Metal causal launcher additionally pins the fast-path shape. */
    if (kv_lora_dim != 512u || qk_rope != 64u || !cache_f16 ||
        pos0 > n_selected || n_tokens > n_selected - pos0) {
        return 0;
    }
    return glm_attn_batch_lora_launch(lora_out, q, qk_low, kv_lora_cache,
                                      k_rope_cache, NULL /* causal */,
                                      n_tokens, pos0, n_selected,
                                      cache_cap, cache_f16, n_head,
                                      kv_lora_dim, qk_nope, qk_rope,
                                      n_ctx_orig, freq_base, freq_scale,
                                      ext_factor, attn_factor,
                                      beta_fast, beta_slow,
                                      "glm causal indexed batch attention-lora launch");
}
