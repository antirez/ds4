/* ds4_cuda_glm_indexer.inc: CUDA port of the GLM-5.2 indexer / rope-tail /
 * low-rank QK Metal kernels (launchers in ds4_metal.m, shaders in
 * metal/dsv4_misc.metal). This file is #included near the end of ds4_cuda.cu
 * and relies on its internals: cuda_ok(), cuda_model_range_ptr(),
 * warp_sum_f32(), struct ds4_gpu_tensor {ptr,bytes}.
 *
 * Entry points implemented here:
 *   ds4_gpu_glm_rope_tail_tensor
 *   ds4_gpu_glm_indexer_rope_tail_tensor
 *   ds4_gpu_glm_indexer_score_one_tensor
 *   ds4_gpu_glm_indexer_scores_batch_tensor
 *   ds4_gpu_glm_qk_lowrank_q8_0_tensor
 *   ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor
 *   ds4_gpu_glm_value_project_q8_0_batch_heads_tensor
 *
 * ponytail: only the scalar/generic Metal kernels are ported (plus the
 * decode-hot score-one direct kernel). The Metal tiled-simdgroup batch
 * scores, glm52-specialized qk-lowrank and the value-project MMA fast
 * paths are perf variants with identical math; add WMMA versions here if
 * profiling shows these launches on the critical path. */

/* ---------------------------------------------------------------------------
 * Device helpers.
 * ------------------------------------------------------------------------- */

/* Port of glm_cache_load_f32_or_f16 (metal/dsv4_misc.metal). */
__device__ static float glm_idx_cache_load(
        const char *base,
        uint64_t index,
        uint32_t cache_f16) {
    if (cache_f16 != 0u) {
        return __half2float(((const __half *)base)[index]);
    }
    return ((const float *)base)[index];
}

/* Port of glm_q8_0_dot_row_dev_f32 (metal/dsv4_misc.metal). One q8_0 block
 * is 34 bytes: __half scale followed by 32 int8 quants. */
__device__ static float glm_idx_q8_0_dot_row(
        const char *row,
        const float *x,
        uint32_t n_cols) {
    float acc = 0.0f;
    const uint32_t n_blocks = (n_cols + 31u) >> 5;
    for (uint32_t block = 0; block < n_blocks; block++) {
        const char *block_base = row + (uint64_t)block * 34u;
        const float d = __half2float(*(const __half *)block_base);
        const int8_t *qs = (const int8_t *)(block_base + 2u);
        const uint32_t base = block << 5;
        const uint32_t count = n_cols - base < 32u ? n_cols - base : 32u;
        for (uint32_t qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static uint64_t glm_idx_q8_0_row_bytes(uint32_t n_cols) {
    return (((uint64_t)n_cols + 31u) / 32u) * 34u;
}

/* ---------------------------------------------------------------------------
 * GLM RoPE tail (kernel_glm_indexer_rope_tail_f32).
 *
 * Rotates a rot_dim slice starting at rot_offset inside each head vector,
 * with contiguous (i, i+1) pairs and YaRN corrections. Both
 * ds4_gpu_glm_rope_tail_tensor (rot_offset = head_dim - rot_dim) and
 * ds4_gpu_glm_indexer_rope_tail_tensor (rot_offset = 0) route here, exactly
 * like the shared Metal launcher ds4_gpu_glm_rope_tail_offset_tensor.
 * ------------------------------------------------------------------------- */

__global__ static void glm_rope_tail_offset_kernel(
        float *x,
        uint32_t n_tokens,
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
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half_rot = rot_dim / 2u;
    const uint32_t pairs = n_tokens * n_head * half_rot;
    if (gid >= pairs) return;
    const uint32_t pair = gid % half_rot;
    const uint32_t tmp = gid / half_rot;
    const uint32_t head = tmp % n_head;
    const uint32_t token = tmp / n_head;
    const uint32_t i = pair * 2u;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)rot_dim *
                       logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)rot_dim *
                      logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(rot_dim - 1u), corr1);
    }

    const float theta_extrap = (float)(pos0 + token) *
        powf(freq_base, -((float)i) / (float)rot_dim);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float c = cosf(theta) * mscale;
    const float s = sinf(theta) * mscale;

    float *row = x + ((uint64_t)token * n_head + head) * head_dim + rot_offset;
    const float x0 = row[i];
    const float x1 = row[i + 1u];
    row[i] = x0 * c - x1 * s;
    row[i + 1u] = x0 * s + x1 * c;
}

static int glm_rope_tail_offset_launch(
        ds4_gpu_tensor *x,
        uint32_t n_tokens,
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
        float beta_slow,
        const char *label) {
    if (!x || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        rot_dim == 0 || rot_offset > head_dim || rot_dim > head_dim - rot_offset ||
        (rot_dim & 1u) != 0 ||
        pos0 > UINT32_MAX - n_tokens ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        x->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    const uint32_t pairs = n_tokens * n_head * (rot_dim / 2u);
    glm_rope_tail_offset_kernel<<<(pairs + 255u) / 256u, 256>>>(
        (float *)x->ptr,
        n_tokens, n_head, head_dim, rot_dim, rot_offset,
        pos0, n_ctx_orig,
        freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), label);
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
    return glm_rope_tail_offset_launch(x, n_tokens, n_head, head_dim, rot_dim,
                                       head_dim - rot_dim, pos0, n_ctx_orig,
                                       freq_base, freq_scale, ext_factor,
                                       attn_factor, beta_fast, beta_slow,
                                       "glm rope tail launch");
}

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
    return glm_rope_tail_offset_launch(x, n_tokens, n_head, head_dim, rot_dim,
                                       0, pos0, n_ctx_orig,
                                       freq_base, freq_scale, ext_factor,
                                       attn_factor, beta_fast, beta_slow,
                                       "glm indexer rope tail launch");
}

/* ---------------------------------------------------------------------------
 * GLM indexer scores, single token (kernel_glm_indexer_score_one +
 * kernel_glm_indexer_score_one_direct).
 *
 * scores[row] = sum_h relu(dot(q[h], key_cache[row]) * scale) * weights[h]
 * Note: unlike the DeepSeek indexer, relu is applied AFTER scaling and the
 * final sum is NOT rescaled.
 * ------------------------------------------------------------------------- */

__global__ static void glm_indexer_score_one_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const char *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t cache_f16,
        float scale) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    __shared__ float partial[256];
    float score = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float k = glm_idx_cache_load(indexer_key_cache,
                                               (uint64_t)row * head_dim + d,
                                               cache_f16);
            dot += qh[d] * k;
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            score += fmaxf(partial[0] * scale, 0.0f) * weights[h];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[row] = score;
}

/* Fast path for the GLM indexer decode shape (n_head=32, head_dim=128).
 * One block of 128 threads per cache row, one warp per head, four heads
 * per pass. Mirrors kernel_glm_indexer_score_one_direct. */
__global__ static void glm_indexer_score_one_direct32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const char *indexer_key_cache,
        uint32_t n_rows,
        uint32_t cache_f16,
        float scale) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (row >= n_rows || tid >= 128u) return;

    __shared__ float krow[128];
    __shared__ float psum[4];
    krow[tid] = glm_idx_cache_load(indexer_key_cache,
                                   (uint64_t)row * 128u + tid,
                                   cache_f16);
    __syncthreads();

    float acc = 0.0f;
    for (uint32_t head0 = 0; head0 < 32u; head0 += 4u) {
        const uint32_t head = head0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)head * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float s = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        s = warp_sum_f32(s);
        if (lane == 0) psum[warp] = fmaxf(s * scale, 0.0f) * weights[head];
        __syncthreads();
        if (tid == 0) acc += psum[0] + psum[1] + psum[2] + psum[3];
        __syncthreads();
    }
    if (tid == 0) scores[row] = acc;
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
    if (!scores || !q || !weights || !indexer_key_cache ||
        n_rows == 0 || n_head == 0 || head_dim == 0 ||
        !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    if (scores->bytes < (uint64_t)n_rows * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_head * sizeof(float) ||
        indexer_key_cache->bytes < (uint64_t)n_rows * head_dim * cache_elem_bytes) {
        return 0;
    }
    if (n_head == 32u && head_dim == 128u) {
        glm_indexer_score_one_direct32_kernel<<<n_rows, 128>>>(
            (float *)scores->ptr,
            (const float *)q->ptr,
            (const float *)weights->ptr,
            (const char *)indexer_key_cache->ptr,
            n_rows, cache_f16 ? 1u : 0u, scale);
        return cuda_ok(cudaGetLastError(), "glm indexer score one direct launch");
    }
    glm_indexer_score_one_kernel<<<n_rows, 256>>>(
        (float *)scores->ptr,
        (const float *)q->ptr,
        (const float *)weights->ptr,
        (const char *)indexer_key_cache->ptr,
        n_rows, n_head, head_dim, cache_f16 ? 1u : 0u, scale);
    return cuda_ok(cudaGetLastError(), "glm indexer score one launch");
}

/* ---------------------------------------------------------------------------
 * GLM indexer scores, token batch (kernel_glm_indexer_scores_batch).
 *
 * scores[token][row] as above, with causal masking: rows at or beyond
 * pos0 + token + 1 get -INF.
 * ------------------------------------------------------------------------- */

__global__ static void glm_indexer_scores_batch_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const char *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t cache_f16,
        float scale) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (row >= n_rows || token >= n_tokens) return;

    float *dst = scores + (uint64_t)token * n_rows + row;
    const uint32_t visible = min(pos0 + token + 1u, n_rows);
    if (row >= visible) {
        if (threadIdx.x == 0) *dst = -INFINITY;
        return;
    }

    __shared__ float partial[256];
    float score = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)token * n_head + h) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float k = glm_idx_cache_load(indexer_key_cache,
                                               (uint64_t)row * head_dim + d,
                                               cache_f16);
            dot += qh[d] * k;
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float *w = weights + (uint64_t)token * n_head;
            score += fmaxf(partial[0] * scale, 0.0f) * w[h];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *dst = score;
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
    /* head_dim must be 128, matching the Metal launcher's contract. */
    if (!scores || !q || !weights || !indexer_key_cache ||
        n_rows == 0 || n_tokens == 0 || n_head == 0 || head_dim != 128u ||
        pos0 >= n_rows || n_tokens > n_rows - pos0 ||
        !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }
    const uint64_t cache_elem_bytes = cache_f16 ? sizeof(uint16_t) : sizeof(float);
    if (scores->bytes < (uint64_t)n_rows * n_tokens * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        indexer_key_cache->bytes < (uint64_t)n_rows * head_dim * cache_elem_bytes) {
        return 0;
    }
    dim3 grid(n_rows, n_tokens, 1);
    glm_indexer_scores_batch_kernel<<<grid, 128>>>(
        (float *)scores->ptr,
        (const float *)q->ptr,
        (const float *)weights->ptr,
        (const char *)indexer_key_cache->ptr,
        n_rows, n_tokens, pos0, n_head, head_dim,
        cache_f16 ? 1u : 0u, scale);
    return cuda_ok(cudaGetLastError(), "glm indexer batch scores launch");
}

/* ---------------------------------------------------------------------------
 * GLM low-rank QK projection over q8_0 weights
 * (kernel_glm_qk_lowrank_q8_0 / _batch).
 *
 * Per head h and token t:
 *   qk_low[t][h][j] = dot(weight_row(h * kv_lora_dim + j)[0..qk_nope),
 *                         q[t][h][0..qk_nope))
 * The single-token entry point is the batch kernel with n_tokens = 1.
 * ------------------------------------------------------------------------- */

__global__ static void glm_qk_lowrank_q8_0_kernel(
        float *qk_low,
        const float *q,
        const char *weight,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_dim,
        uint32_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens) return;
    const float *qh = q + ((uint64_t)token * n_head + head) * qk_dim;
    float *out = qk_low + ((uint64_t)token * n_head + head) * kv_lora_dim;
    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        const char *row = weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        out[j] = glm_idx_q8_0_dot_row(row, qh, qk_nope);
    }
}

static int glm_qk_lowrank_q8_0_launch(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim,
        const char           *label) {
    if (!qk_low || !q || !model_map ||
        n_tokens == 0 || n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_nope > qk_dim) {
        return 0;
    }
    const uint64_t row_bytes = glm_idx_q8_0_row_bytes(qk_nope);
    const uint64_t weight_bytes = (uint64_t)n_head * kv_lora_dim * row_bytes;
    if (qk_low->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float)) {
        return 0;
    }
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        fprintf(stderr, "ds4: CUDA GLM qk lowrank range is outside the mapped model\n");
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, "glm qk lowrank");
    if (!wptr) return 0;
    dim3 grid(n_head, n_tokens, 1);
    glm_qk_lowrank_q8_0_kernel<<<grid, 256>>>(
        (float *)qk_low->ptr,
        (const float *)q->ptr,
        wptr,
        n_tokens, n_head, kv_lora_dim, qk_nope, qk_dim,
        (uint32_t)row_bytes);
    return cuda_ok(cudaGetLastError(), label);
}

extern "C" int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    return glm_qk_lowrank_q8_0_launch(qk_low, q, model_map, model_size,
                                      weight_offset, 1, n_head, kv_lora_dim,
                                      qk_nope, qk_dim,
                                      "glm qk lowrank launch");
}

extern "C" int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    return glm_qk_lowrank_q8_0_launch(qk_low, q, model_map, model_size,
                                      weight_offset, n_tokens, n_head,
                                      kv_lora_dim, qk_nope, qk_dim,
                                      "glm batch qk lowrank launch");
}

/* ---------------------------------------------------------------------------
 * GLM value projection over q8_0 weights, per (token, head)
 * (kernel_glm_value_project_q8_0_batch_heads).
 *
 * heads[t][h][d] = dot(weight_row(h * value_dim + d),
 *                      lora[t][h][0..kv_lora_dim))
 * The lora vector is staged in dynamic shared memory like the Metal
 * threadgroup buffer.
 * ------------------------------------------------------------------------- */

__global__ static void glm_value_project_q8_0_batch_heads_kernel(
        float *heads,
        const float *lora,
        const char *weight,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint32_t row_bytes) {
    extern __shared__ float glm_idx_vp_x[];
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens) return;
    const float *src = lora + ((uint64_t)token * n_head + head) * kv_lora_dim;
    float *out = heads + ((uint64_t)token * n_head + head) * value_dim;

    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        glm_idx_vp_x[j] = src[j];
    }
    __syncthreads();

    for (uint32_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
        const char *row = weight + ((uint64_t)head * value_dim + d) * row_bytes;
        out[d] = glm_idx_q8_0_dot_row(row, glm_idx_vp_x, kv_lora_dim);
    }
}

extern "C" int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim) {
    if (!heads || !lora || !model_map ||
        n_tokens == 0 || n_head == 0 ||
        kv_lora_dim == 0 || value_dim == 0) {
        return 0;
    }
    const uint64_t row_bytes = glm_idx_q8_0_row_bytes(kv_lora_dim);
    const uint64_t weight_bytes = (uint64_t)n_head * value_dim * row_bytes;
    if (heads->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float) ||
        lora->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float)) {
        return 0;
    }
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) {
        fprintf(stderr, "ds4: CUDA GLM batch value project range is outside the mapped model\n");
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            weight_bytes, "glm value project");
    if (!wptr) return 0;
    dim3 grid(n_head, n_tokens, 1);
    glm_value_project_q8_0_batch_heads_kernel
        <<<grid, 256, (size_t)kv_lora_dim * sizeof(float)>>>(
        (float *)heads->ptr,
        (const float *)lora->ptr,
        wptr,
        n_tokens, n_head, kv_lora_dim, value_dim,
        (uint32_t)row_bytes);
    return cuda_ok(cudaGetLastError(), "glm batch value project launch");
}
