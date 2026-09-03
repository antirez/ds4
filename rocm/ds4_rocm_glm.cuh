// GLM-5.x ROCm kernels.  This mirrors the Metal GLM graph entry points with
// direct kernels first; faster Strix-specific kernels can replace these one by
// one without changing the graph contract.

__device__ __forceinline__ static float glm_rocm_cache_load(
        const char *base,
        uint64_t index,
        bool cache_f16) {
    return cache_f16 ? __half2float(((const __half *)base)[index])
                     : ((const float *)base)[index];
}

__device__ __forceinline__ static void glm_rocm_cache_store(
        char *base,
        uint64_t index,
        bool cache_f16,
        float x) {
    if (cache_f16) ((__half *)base)[index] = __float2half(x);
    else ((float *)base)[index] = x;
}

__device__ __forceinline__ static void glm_rocm_rope_pair(
        uint32_t row,
        uint32_t r,
        uint32_t n_rot,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float *c_out,
        float *s_out) {
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1u), corr1);
    }
    const float inv_ndims = -1.0f / (float)n_rot;
    const float theta_extrap = (float)row * powf(freq_base, inv_ndims * (float)r);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)r) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *c_out = cosf(theta) * mscale;
    *s_out = sinf(theta) * mscale;
}

__device__ __forceinline__ static float2 glm_rocm_rotated_cache_rope_pair(
        const char *base,
        uint64_t rope_base,
        uint32_t r,
        uint32_t row,
        uint32_t qk_rope,
        bool cache_f16,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    float c, s;
    glm_rocm_rope_pair(row, r, qk_rope, n_ctx_orig, freq_base, freq_scale,
                       ext_factor, attn_factor, beta_fast, beta_slow, &c, &s);
    const float x0 = glm_rocm_cache_load(base, rope_base + r, cache_f16);
    const float x1 = glm_rocm_cache_load(base, rope_base + r + 1u, cache_f16);
    return make_float2(x0 * c - x1 * s, x0 * s + x1 * c);
}

__device__ __forceinline__ static float4 glm_rocm_load4_f32(const float *p) {
    return make_float4(p[0], p[1], p[2], p[3]);
}

__device__ __forceinline__ static void glm_rocm_store4_f32(float *p, float4 v) {
    p[0] = v.x;
    p[1] = v.y;
    p[2] = v.z;
    p[3] = v.w;
}

__device__ __forceinline__ static float glm_rocm_dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static float glm_rocm_q8_0_dot_row(
        const unsigned char *row,
        const float *x,
        uint32_t n_cols) {
    const uint32_t blocks = (n_cols + 31u) >> 5u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const unsigned char *blk = row + (uint64_t)b * 34u;
        const float d = q8_0_scale_scalar(blk);
        const int8_t *qs = (const int8_t *)(blk + 2u);
        const uint32_t base = b << 5u;
        const uint32_t count = min(32u, n_cols - base);
        for (uint32_t i = 0; i < count; i++) {
            acc += d * (float)qs[i] * x[base + i];
        }
    }
    return acc;
}

static int glm_rocm_u32_add_checked(uint32_t a, uint32_t b, uint32_t *out) {
    const uint64_t v = (uint64_t)a + b;
    if (!out || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int glm_rocm_launch_blocks(uint64_t n, uint32_t block, uint32_t *blocks) {
    if (!blocks || block == 0u) return 0;
    const uint64_t b = (n + block - 1u) / block;
    if (b > UINT32_MAX) return 0;
    *blocks = (uint32_t)b;
    return 1;
}

static int glm_rocm_tensor_has_cache2(
        const ds4_gpu_tensor *t,
        uint64_t a,
        uint64_t b,
        uint64_t elem_size) {
    return cuda_tensor_has_elems2(t, a, b, elem_size);
}

static int glm_rocm_tensor_has_cache3(
        const ds4_gpu_tensor *t,
        uint64_t a,
        uint64_t b,
        uint64_t c,
        uint64_t elem_size) {
    return cuda_tensor_has_elems3(t, a, b, c, elem_size);
}

static int glm_rocm_model_f32_range(
        uint64_t model_size,
        uint64_t offset,
        uint64_t elems,
        uint64_t *bytes_out) {
    uint64_t bytes = 0;
    if (!cuda_u64_mul_checked(elems, sizeof(float), &bytes) ||
        !cuda_model_range_fits(model_size, offset, bytes)) {
        return 0;
    }
    if (bytes_out) *bytes_out = bytes;
    return 1;
}

static int glm_rocm_check_token_span(
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t *end_out) {
    if (n_tokens == 0u || pos0 > UINT32_MAX - n_tokens) return 0;
    if (end_out) *end_out = pos0 + n_tokens;
    return 1;
}

static int glm_rocm_check_pos_span(
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap) {
    uint32_t end = 0;
    return glm_rocm_check_token_span(pos0, n_tokens, &end) && end <= cache_cap;
}

static int glm53_rocm_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > UINT64_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static int glm53_rocm_tensor_has(
        const ds4_gpu_tensor *tensor,
        uint64_t elements,
        uint64_t elem_size) {
    return tensor && tensor->ptr &&
        (elements == 0u || elem_size <= UINT64_MAX / elements) &&
        tensor->bytes >= elements * elem_size;
}

static const float *glm53_rocm_weight_f32(
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t elements,
        const char *label) {
    uint64_t bytes = 0;
    if (!model_map ||
        !glm53_rocm_mul_u64(elements, sizeof(float), &bytes) ||
        offset > model_size || bytes > model_size - offset) {
        fprintf(stderr, "ds4: GLM-5.3 %s range is outside the mapped model\n",
                label ? label : "weight");
        return NULL;
    }
    return (const float *)cuda_model_range_ptr(
        model_map, offset, bytes, label);
}

__global__ static void glm53_rocm_embedding_bf16_kernel(
        float *out,
        const uint16_t *weights,
        const int32_t *tokens,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_vocab) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (i >= n) return;
    const uint32_t token_row = (uint32_t)(i / n_embd);
    const uint32_t col = (uint32_t)(i - (uint64_t)token_row * n_embd);
    const int32_t token = tokens[token_row];
    if (token < 0 || (uint32_t)token >= n_vocab) {
        out[i] = 0.0f;
        return;
    }
    const uint16_t bits = weights[(uint64_t)(uint32_t)token * n_embd + col];
    out[i] = __uint_as_float((uint32_t)bits << 16);
}

__global__ static void glm53_rocm_f32_to_bf16_kernel(
        hip_bfloat16 *out,
        const float *x,
        uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = hip_bfloat16(x[i]);
}

__global__ static void glm53_rocm_matvec_bf16_f32_kernel(
        float *out,
        const uint16_t *weights,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = blockIdx.x * 8u + warp;
    const uint32_t row = blockIdx.y;
    float sum = 0.0f;
    if (col < out_dim) {
        const uint16_t *wrow = weights + (uint64_t)col * in_dim;
        const float *xrow = x + (uint64_t)row * in_dim;
        for (uint32_t i = lane; i < in_dim; i += 32u) {
            const float w = __uint_as_float((uint32_t)wrow[i] << 16);
            sum = fmaf(w, xrow[i], sum);
        }
    }
    sum = warp_sum_f32(sum);
    if (lane == 0u && col < out_dim) {
        out[(uint64_t)row * out_dim + col] = sum;
    }
}

__global__ static void glm53_rocm_matvec_bf16_row_f32_kernel(
        float *out,
        const uint16_t *weights,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t col = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (col >= out_dim) return;
    const uint16_t *wrow = weights + (uint64_t)col * in_dim;
    const float *xrow = x + (uint64_t)row * in_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const float w = __uint_as_float((uint32_t)wrow[i] << 16);
        sum = fmaf(w, xrow[i], sum);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        out[(uint64_t)row * out_dim + col] = partial[0];
    }
}

extern "C" int ds4_gpu_glm53_embedding_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const ds4_gpu_tensor *token_ids,
        uint32_t              n_tokens,
        uint32_t              n_embd,
        uint32_t              n_vocab) {
    if (!out || !token_ids || !model_map || n_tokens == 0u ||
        n_embd == 0u || n_vocab == 0u ||
        weight_offset > model_size) {
        return 0;
    }
    uint64_t weight_elements = 0, weight_bytes = 0;
    uint64_t output_elements = 0, output_bytes = 0, token_bytes = 0;
    if (!glm53_rocm_mul_u64(n_vocab, n_embd, &weight_elements) ||
        !glm53_rocm_mul_u64(weight_elements, sizeof(uint16_t), &weight_bytes) ||
        !glm53_rocm_mul_u64(n_tokens, n_embd, &output_elements) ||
        !glm53_rocm_mul_u64(output_elements, sizeof(float), &output_bytes) ||
        !glm53_rocm_mul_u64(n_tokens, sizeof(int32_t), &token_bytes)) {
        return 0;
    }
    if (weight_bytes > model_size - weight_offset ||
        token_ids->bytes < token_bytes || out->bytes < output_bytes) {
        return 0;
    }
    const char *weights = cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes,
        "GLM-5.3 BF16 embedding");
    if (!weights) return 0;
    glm53_rocm_embedding_bf16_kernel<<<
        (unsigned)((output_elements + 255u) / 256u), 256u>>>(
            (float *)out->ptr, (const uint16_t *)weights,
            (const int32_t *)token_ids->ptr, n_tokens, n_embd, n_vocab);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 BF16 embedding launch");
}

extern "C" int ds4_gpu_glm53_matmul_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows) {
    if (!out || !x || !model_map || !g_cublas_ready || in_dim == 0u ||
        out_dim == 0u || n_rows == 0u ||
        weight_offset > model_size) {
        return 0;
    }
    uint64_t weight_elements = 0, weight_bytes = 0;
    uint64_t input_elements = 0, input_bytes = 0;
    uint64_t output_elements = 0, output_bytes = 0;
    if (!glm53_rocm_mul_u64(out_dim, in_dim, &weight_elements) ||
        !glm53_rocm_mul_u64(weight_elements, sizeof(uint16_t), &weight_bytes) ||
        !glm53_rocm_mul_u64(n_rows, in_dim, &input_elements) ||
        !glm53_rocm_mul_u64(input_elements, sizeof(float), &input_bytes) ||
        !glm53_rocm_mul_u64(n_rows, out_dim, &output_elements) ||
        !glm53_rocm_mul_u64(output_elements, sizeof(float), &output_bytes)) {
        return 0;
    }
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < input_bytes || out->bytes < output_bytes) {
        return 0;
    }
    const char *weights = cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes,
        "GLM-5.3 BF16 matrix");
    if (!weights) return 0;
    if (n_rows <= 8u) {
        if (out_dim <= 128u && in_dim >= 4096u &&
            getenv("DS4_ROCM_GLM_DISABLE_BF16_ROW_MATVEC") == NULL) {
            const dim3 row_grid(out_dim, n_rows, 1u);
            glm53_rocm_matvec_bf16_row_f32_kernel<<<row_grid, 256u>>>(
                (float *)out->ptr, (const uint16_t *)weights,
                (const float *)x->ptr, in_dim, out_dim);
            return cuda_ok(cudaGetLastError(),
                           "GLM-5.3 BF16/F32 row matvec launch");
        }
        const dim3 grid((out_dim + 7u) / 8u, n_rows, 1u);
        glm53_rocm_matvec_bf16_f32_kernel<<<grid, 256u>>>(
            (float *)out->ptr, (const uint16_t *)weights,
            (const float *)x->ptr, in_dim, out_dim);
        return cuda_ok(cudaGetLastError(),
                       "GLM-5.3 BF16/F32 matvec launch");
    }
    hip_bfloat16 *xb = (hip_bfloat16 *)cuda_tmp_alloc(
        input_elements * sizeof(hip_bfloat16),
        "GLM-5.3 BF16 matmul activations");
    if (!xb) return 0;
    glm53_rocm_f32_to_bf16_kernel<<<
        (unsigned)((input_elements + 255u) / 256u), 256u>>>(
            xb, (const float *)x->ptr, input_elements);
    if (!cuda_ok(cudaGetLastError(),
                 "GLM-5.3 BF16 activation conversion launch")) {
        return 0;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)out_dim, (int)n_rows, (int)in_dim,
        &alpha,
        weights, HIPBLAS_R_16B, (int)in_dim,
        xb, HIPBLAS_R_16B, (int)in_dim,
        &beta,
        out->ptr, CUDA_R_32F, (int)out_dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(status, "GLM-5.3 BF16 matmul");
}

__global__ static void glm53_rocm_matmul_q4_K_q8_K_kernel(
        float *out,
        const cuda_block_q4_K *weights,
        const cuda_block_q8_K *xq,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_rows) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t col = blockIdx.x * 32u + row_lane;
    const uint32_t row = blockIdx.y;
    if (col >= out_dim || row >= n_rows) return;
    const cuda_block_q4_K *wrow = weights + (uint64_t)col * n_blocks;
    const cuda_block_q8_K *xrow = xq + (uint64_t)row * n_blocks;
    float sum = 0.0f;
    for (uint32_t b = lane; b < n_blocks; b += 8u) {
        sum += dev_dot_q4_K_q8_K_block(wrow + b, xrow + b);
    }
    sum = quarter_warp_sum_f32(sum, lane);
    if (lane == 0u) out[(uint64_t)row * out_dim + col] = sum;
}

extern "C" int ds4_gpu_matmul_q4_K_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint64_t              n_rows) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_rows == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_rows > UINT32_MAX || (in_dim % CUDA_QK_K) != 0u) {
        return 0;
    }
    const uint64_t n_blocks = in_dim / CUDA_QK_K;
    uint64_t weight_blocks = 0, weight_bytes = 0;
    uint64_t input_elems = 0, output_elems = 0, xq_blocks = 0, xq_bytes = 0;
    if (!glm53_rocm_mul_u64(out_dim, n_blocks, &weight_blocks) ||
        !glm53_rocm_mul_u64(weight_blocks,
                            sizeof(cuda_block_q4_K), &weight_bytes) ||
        !glm53_rocm_mul_u64(n_rows, in_dim, &input_elems) ||
        !glm53_rocm_mul_u64(n_rows, out_dim, &output_elems) ||
        !glm53_rocm_mul_u64(n_rows, n_blocks, &xq_blocks) ||
        !glm53_rocm_mul_u64(xq_blocks, sizeof(cuda_block_q8_K), &xq_bytes) ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        !glm53_rocm_tensor_has(x, input_elems, sizeof(float)) ||
        !glm53_rocm_tensor_has(out, output_elems, sizeof(float))) {
        return 0;
    }
    const char *weight = cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "GLM-5.3 Q4_K matrix");
    if (!weight) return 0;
    cuda_block_q8_K *xq = (cuda_block_q8_K *)cuda_tmp_alloc(
        xq_bytes,
        "GLM-5.3 Q4_K activations");
    if (!xq) return 0;
    const dim3 quant_grid((uint32_t)n_blocks, (uint32_t)n_rows, 1u);
    q8_K_quantize_kernel<<<quant_grid, 256u>>>(
        xq, (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)n_rows);
    if (!cuda_ok(cudaGetLastError(),
                 "GLM-5.3 Q4_K activation quantization launch")) {
        return 0;
    }
    const dim3 grid(((uint32_t)out_dim + 31u) / 32u,
                    (uint32_t)n_rows, 1u);
    glm53_rocm_matmul_q4_K_q8_K_kernel<<<grid, 256u>>>(
        (float *)out->ptr, (const cuda_block_q4_K *)weight, xq,
        (uint32_t)n_blocks, (uint32_t)out_dim, (uint32_t)n_rows);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 Q4_K matmul launch");
}

enum {
    GLM53_ROCM_KDA_DIM = 128,
    GLM53_ROCM_KDA_HISTORY = 3,
};

__device__ __forceinline__ static float glm53_rocm_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ static float glm53_rocm_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ static float glm53_rocm_lane0(float x) {
    return __shfl(x, 0, 32);
}

__global__ static void glm53_rocm_kda_decode_kernel(
        float *out,
        float *conv_state,
        float *state,
        const float *q_in,
        const float *k_in,
        const float *v_in,
        const float *raw_gate,
        const float *raw_beta,
        const float *output_gate,
        const float *q_conv,
        const float *k_conv,
        const float *v_conv,
        const float *a_log,
        const float *dt_bias,
        const float *output_norm,
        uint32_t n_heads,
        uint32_t n_rows,
        float lower_bound,
        float norm_eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (row >= n_rows || head >= n_heads || tid >= GLM53_ROCM_KDA_DIM) {
        return;
    }

    __shared__ float scratch[656];
    float *sq = scratch;
    float *sk = sq + GLM53_ROCM_KDA_DIM;
    float *sd = sk + GLM53_ROCM_KDA_DIM;
    float *sv = sd + GLM53_ROCM_KDA_DIM;
    float *so = sv + GLM53_ROCM_KDA_DIM;
    float *reduce_q = so + GLM53_ROCM_KDA_DIM;
    float *reduce_k = reduce_q + 4;
    float *reduce_o = reduce_k + 4;
    float *beta_shared = reduce_o + 4;

    const uint32_t projection = n_heads * GLM53_ROCM_KDA_DIM;
    const uint32_t channel = head * GLM53_ROCM_KDA_DIM + tid;
    const uint64_t input_base =
        (uint64_t)row * projection + head * GLM53_ROCM_KDA_DIM;
    const uint64_t conv_row_stride =
        (uint64_t)3 * GLM53_ROCM_KDA_HISTORY * projection;
    float *q_state = conv_state + (uint64_t)row * conv_row_stride;
    float *k_state = q_state + GLM53_ROCM_KDA_HISTORY * projection;
    float *v_state = k_state + GLM53_ROCM_KDA_HISTORY * projection;

    float q_acc = 0.0f;
    float k_acc = 0.0f;
    float v_acc = 0.0f;
    for (uint32_t w = 0; w < GLM53_ROCM_KDA_HISTORY; w++) {
        q_acc = fmaf(q_state[(uint64_t)w * projection + channel],
                     q_conv[(uint64_t)channel * 4u + w], q_acc);
        k_acc = fmaf(k_state[(uint64_t)w * projection + channel],
                     k_conv[(uint64_t)channel * 4u + w], k_acc);
        v_acc = fmaf(v_state[(uint64_t)w * projection + channel],
                     v_conv[(uint64_t)channel * 4u + w], v_acc);
    }
    const float q_new = q_in[input_base + tid];
    const float k_new = k_in[input_base + tid];
    const float v_new = v_in[input_base + tid];
    q_acc = fmaf(q_new, q_conv[(uint64_t)channel * 4u + 3u], q_acc);
    k_acc = fmaf(k_new, k_conv[(uint64_t)channel * 4u + 3u], k_acc);
    v_acc = fmaf(v_new, v_conv[(uint64_t)channel * 4u + 3u], v_acc);

    q_state[channel] = q_state[projection + channel];
    q_state[projection + channel] = q_state[2ull * projection + channel];
    q_state[2ull * projection + channel] = q_new;
    k_state[channel] = k_state[projection + channel];
    k_state[projection + channel] = k_state[2ull * projection + channel];
    k_state[2ull * projection + channel] = k_new;
    v_state[channel] = v_state[projection + channel];
    v_state[projection + channel] = v_state[2ull * projection + channel];
    v_state[2ull * projection + channel] = v_new;

    sq[tid] = glm53_rocm_silu(q_acc);
    sk[tid] = glm53_rocm_silu(k_acc);
    sv[tid] = glm53_rocm_silu(v_acc);
    const float gate = raw_gate[input_base + tid] + dt_bias[channel];
    sd[tid] = expf(lower_bound *
        glm53_rocm_sigmoid(expf(a_log[head]) * gate));
    if (tid == 0u) {
        beta_shared[0] = glm53_rocm_sigmoid(
            raw_beta[(uint64_t)row * n_heads + head]);
    }
    __syncthreads();

    float q_sumsq = warp_sum_f32(sq[tid] * sq[tid]);
    float k_sumsq = warp_sum_f32(sk[tid] * sk[tid]);
    if (lane == 0u) {
        reduce_q[warp] = q_sumsq;
        reduce_k[warp] = k_sumsq;
    }
    __syncthreads();
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = glm53_rocm_lane0(warp_sum_f32(q_total));
    k_total = glm53_rocm_lane0(warp_sum_f32(k_total));
    sq[tid] *= rsqrtf(q_total + 1.0e-6f) * 0.08838834764831845f;
    sk[tid] *= rsqrtf(k_total + 1.0e-6f);
    __syncthreads();

    const uint32_t k0 = lane * 4u;
    const float4 q4 = *(const float4 *)(sq + k0);
    const float4 k4 = *(const float4 *)(sk + k0);
    const float4 decay4 = *(const float4 *)(sd + k0);
    const uint64_t state_head =
        ((uint64_t)row * n_heads + head) *
        GLM53_ROCM_KDA_DIM * GLM53_ROCM_KDA_DIM;

    for (uint32_t value = warp; value < GLM53_ROCM_KDA_DIM; value += 4u) {
        float4 *hptr = (float4 *)(state + state_head +
            (uint64_t)value * GLM53_ROCM_KDA_DIM + k0);
        float4 h = *hptr;
        h.x *= decay4.x;
        h.y *= decay4.y;
        h.z *= decay4.z;
        h.w *= decay4.w;
        const float hk = glm53_rocm_lane0(warp_sum_f32(dot4_f32(h, k4)));
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h.x = fmaf(k4.x, delta_v, h.x);
        h.y = fmaf(k4.y, delta_v, h.y);
        h.z = fmaf(k4.z, delta_v, h.z);
        h.w = fmaf(k4.w, delta_v, h.w);
        *hptr = h;
        const float hq = glm53_rocm_lane0(warp_sum_f32(dot4_f32(h, q4)));
        if (lane == 0u) so[value] = hq;
    }
    __syncthreads();

    float o_sumsq = warp_sum_f32(so[tid] * so[tid]);
    if (lane == 0u) reduce_o[warp] = o_sumsq;
    __syncthreads();
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = glm53_rocm_lane0(warp_sum_f32(o_total));
    const float o_scale =
        rsqrtf(o_total / (float)GLM53_ROCM_KDA_DIM + norm_eps);
    const float out_gate = glm53_rocm_sigmoid(output_gate[input_base + tid]);
    out[input_base + tid] =
        so[tid] * o_scale * output_norm[tid] * out_gate;
}

__global__ static void glm53_rocm_kda_prefill_prepare_kernel(
        float *q,
        float *k,
        float *v,
        float *raw_gate,
        float *conv_state,
        const float *q_conv,
        const float *k_conv,
        const float *v_conv,
        const float *a_log,
        const float *dt_bias,
        uint32_t n_heads,
        uint32_t n_tokens,
        float lower_bound) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (head >= n_heads || tid >= GLM53_ROCM_KDA_DIM) return;

    __shared__ float sq[GLM53_ROCM_KDA_DIM];
    __shared__ float sk[GLM53_ROCM_KDA_DIM];
    __shared__ float reduce_q[4];
    __shared__ float reduce_k[4];
    const uint32_t projection = n_heads * GLM53_ROCM_KDA_DIM;
    const uint32_t channel = head * GLM53_ROCM_KDA_DIM + tid;
    float *q_state = conv_state;
    float *k_state = q_state + GLM53_ROCM_KDA_HISTORY * projection;
    float *v_state = k_state + GLM53_ROCM_KDA_HISTORY * projection;

    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint64_t index = (uint64_t)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (uint32_t w = 0; w < GLM53_ROCM_KDA_HISTORY; w++) {
            q_acc = fmaf(q_state[(uint64_t)w * projection + channel],
                         q_conv[(uint64_t)channel * 4u + w], q_acc);
            k_acc = fmaf(k_state[(uint64_t)w * projection + channel],
                         k_conv[(uint64_t)channel * 4u + w], k_acc);
            v_acc = fmaf(v_state[(uint64_t)w * projection + channel],
                         v_conv[(uint64_t)channel * 4u + w], v_acc);
        }
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fmaf(q_new, q_conv[(uint64_t)channel * 4u + 3u], q_acc);
        k_acc = fmaf(k_new, k_conv[(uint64_t)channel * 4u + 3u], k_acc);
        v_acc = fmaf(v_new, v_conv[(uint64_t)channel * 4u + 3u], v_acc);
        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ull * projection + channel];
        q_state[2ull * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ull * projection + channel];
        k_state[2ull * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ull * projection + channel];
        v_state[2ull * projection + channel] = v_new;

        sq[tid] = glm53_rocm_silu(q_acc);
        sk[tid] = glm53_rocm_silu(k_acc);
        v[index] = glm53_rocm_silu(v_acc);
        raw_gate[index] = expf(lower_bound * glm53_rocm_sigmoid(
            expf(a_log[head]) * (raw_gate[index] + dt_bias[channel])));
        __syncthreads();

        float q_total = warp_sum_f32(sq[tid] * sq[tid]);
        float k_total = warp_sum_f32(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[warp] = q_total;
            reduce_k[warp] = k_total;
        }
        __syncthreads();
        q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = glm53_rocm_lane0(warp_sum_f32(q_total));
        k_total = glm53_rocm_lane0(warp_sum_f32(k_total));
        q[index] = sq[tid] * rsqrtf(q_total + 1.0e-6f) *
                   0.08838834764831845f;
        k[index] = sk[tid] * rsqrtf(k_total + 1.0e-6f);
        __syncthreads();
    }
}

__global__ static void glm53_rocm_kda_prefill_recurrence_kernel(
        float *out,
        float *state,
        const float *q,
        const float *k,
        const float *v,
        const float *decay,
        const float *raw_beta,
        uint32_t n_heads,
        uint32_t n_tokens) {
    const uint32_t head = blockIdx.x;
    const uint32_t value = blockIdx.y * 4u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (head >= n_heads || value >= GLM53_ROCM_KDA_DIM) return;
    const uint32_t projection = n_heads * GLM53_ROCM_KDA_DIM;
    const uint32_t k0 = lane * 4u;
    float4 *state_ptr = (float4 *)(state +
        ((uint64_t)head * GLM53_ROCM_KDA_DIM + value) *
        GLM53_ROCM_KDA_DIM + k0);
    float4 h = *state_ptr;

    for (uint32_t token = 0; token < n_tokens; token++) {
        const uint64_t base =
            (uint64_t)token * projection + head * GLM53_ROCM_KDA_DIM;
        const float4 q4 = *(const float4 *)(q + base + k0);
        const float4 k4 = *(const float4 *)(k + base + k0);
        const float4 d4 = *(const float4 *)(decay + base + k0);
        h.x *= d4.x;
        h.y *= d4.y;
        h.z *= d4.z;
        h.w *= d4.w;
        const float hk = glm53_rocm_lane0(warp_sum_f32(dot4_f32(h, k4)));
        const float beta = glm53_rocm_sigmoid(
            raw_beta[(uint64_t)token * n_heads + head]);
        const float delta_v = (v[base + value] - hk) * beta;
        h.x = fmaf(k4.x, delta_v, h.x);
        h.y = fmaf(k4.y, delta_v, h.y);
        h.z = fmaf(k4.z, delta_v, h.z);
        h.w = fmaf(k4.w, delta_v, h.w);
        const float result =
            glm53_rocm_lane0(warp_sum_f32(dot4_f32(h, q4)));
        if (lane == 0u) out[base + value] = result;
    }
    *state_ptr = h;
}

__global__ static void glm53_rocm_kda_prefill_output_kernel(
        float *out,
        const float *output_gate,
        const float *output_norm,
        uint32_t n_heads,
        uint32_t n_tokens,
        float norm_eps) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (token >= n_tokens || head >= n_heads || tid >= GLM53_ROCM_KDA_DIM) {
        return;
    }
    __shared__ float partial[4];
    const uint32_t projection = n_heads * GLM53_ROCM_KDA_DIM;
    const uint64_t index = (uint64_t)token * projection +
        head * GLM53_ROCM_KDA_DIM + tid;
    const float raw = out[index];
    float total = warp_sum_f32(raw * raw);
    if (lane == 0u) partial[warp] = total;
    __syncthreads();
    total = lane < 4u ? partial[lane] : 0.0f;
    total = glm53_rocm_lane0(warp_sum_f32(total));
    const float scale =
        rsqrtf(total / (float)GLM53_ROCM_KDA_DIM + norm_eps);
    out[index] = raw * scale * output_norm[tid] *
        glm53_rocm_sigmoid(output_gate[index]);
}

extern "C" int ds4_gpu_glm53_kda_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_rows,
        float                 gate_lower_bound,
        float                 norm_eps) {
    uint64_t projection = 0, activations = 0, conv_elements = 0;
    uint64_t state_elements = 0;
    if (n_heads == 0u || n_rows == 0u || gate_lower_bound >= 0.0f ||
        !glm53_rocm_mul_u64(n_heads, GLM53_ROCM_KDA_DIM, &projection) ||
        !glm53_rocm_mul_u64(projection, n_rows, &activations) ||
        !glm53_rocm_mul_u64(activations,
            3u * GLM53_ROCM_KDA_HISTORY, &conv_elements) ||
        !glm53_rocm_mul_u64(activations,
            GLM53_ROCM_KDA_DIM, &state_elements) ||
        !glm53_rocm_tensor_has(q, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(k, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(v, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(raw_gate, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(raw_beta,
            (uint64_t)n_rows * n_heads, sizeof(float)) ||
        !glm53_rocm_tensor_has(output_gate, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(out, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(conv_state, conv_elements, sizeof(float)) ||
        !glm53_rocm_tensor_has(recurrent_state, state_elements, sizeof(float))) {
        fprintf(stderr, "ds4: GLM-5.3 KDA decode received invalid buffers\n");
        return 0;
    }
    const float *qw = glm53_rocm_weight_f32(model_map, model_size,
        q_conv_offset, projection * 4u, "KDA Q convolution");
    const float *kw = glm53_rocm_weight_f32(model_map, model_size,
        k_conv_offset, projection * 4u, "KDA K convolution");
    const float *vw = glm53_rocm_weight_f32(model_map, model_size,
        v_conv_offset, projection * 4u, "KDA V convolution");
    const float *a_log = glm53_rocm_weight_f32(model_map, model_size,
        a_log_offset, n_heads, "KDA A_log");
    const float *dt_bias = glm53_rocm_weight_f32(model_map, model_size,
        dt_bias_offset, projection, "KDA dt bias");
    const float *output_norm = glm53_rocm_weight_f32(model_map, model_size,
        output_norm_offset, GLM53_ROCM_KDA_DIM, "KDA output norm");
    if (!qw || !kw || !vw || !a_log || !dt_bias || !output_norm) return 0;
    const dim3 grid(n_rows, n_heads, 1u);
    glm53_rocm_kda_decode_kernel<<<grid, GLM53_ROCM_KDA_DIM>>>(
        (float *)out->ptr, (float *)conv_state->ptr,
        (float *)recurrent_state->ptr, (const float *)q->ptr,
        (const float *)k->ptr, (const float *)v->ptr,
        (const float *)raw_gate->ptr, (const float *)raw_beta->ptr,
        (const float *)output_gate->ptr, qw, kw, vw, a_log, dt_bias,
        output_norm, n_heads, n_rows, gate_lower_bound, norm_eps);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 KDA decode launch");
}

extern "C" int ds4_gpu_glm53_kda_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        ds4_gpu_tensor       *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_tokens,
        float                 gate_lower_bound,
        float                 norm_eps) {
    uint64_t projection = 0, activations = 0, conv_elements = 0;
    uint64_t state_elements = 0;
    if (n_heads == 0u || n_tokens == 0u || gate_lower_bound >= 0.0f ||
        !glm53_rocm_mul_u64(n_heads, GLM53_ROCM_KDA_DIM, &projection) ||
        !glm53_rocm_mul_u64(projection, n_tokens, &activations) ||
        !glm53_rocm_mul_u64(projection,
            3u * GLM53_ROCM_KDA_HISTORY, &conv_elements) ||
        !glm53_rocm_mul_u64(projection,
            GLM53_ROCM_KDA_DIM, &state_elements) ||
        !glm53_rocm_tensor_has(q, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(k, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(v, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(raw_gate, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(raw_beta,
            (uint64_t)n_tokens * n_heads, sizeof(float)) ||
        !glm53_rocm_tensor_has(output_gate, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(out, activations, sizeof(float)) ||
        !glm53_rocm_tensor_has(conv_state, conv_elements, sizeof(float)) ||
        !glm53_rocm_tensor_has(recurrent_state, state_elements, sizeof(float))) {
        fprintf(stderr, "ds4: GLM-5.3 KDA prefill received invalid buffers\n");
        return 0;
    }
    const float *qw = glm53_rocm_weight_f32(model_map, model_size,
        q_conv_offset, projection * 4u, "KDA Q convolution");
    const float *kw = glm53_rocm_weight_f32(model_map, model_size,
        k_conv_offset, projection * 4u, "KDA K convolution");
    const float *vw = glm53_rocm_weight_f32(model_map, model_size,
        v_conv_offset, projection * 4u, "KDA V convolution");
    const float *a_log = glm53_rocm_weight_f32(model_map, model_size,
        a_log_offset, n_heads, "KDA A_log");
    const float *dt_bias = glm53_rocm_weight_f32(model_map, model_size,
        dt_bias_offset, projection, "KDA dt bias");
    const float *output_norm = glm53_rocm_weight_f32(model_map, model_size,
        output_norm_offset, GLM53_ROCM_KDA_DIM, "KDA output norm");
    if (!qw || !kw || !vw || !a_log || !dt_bias || !output_norm) return 0;

    glm53_rocm_kda_prefill_prepare_kernel<<<
        n_heads, GLM53_ROCM_KDA_DIM>>>(
            (float *)q->ptr, (float *)k->ptr, (float *)v->ptr,
            (float *)raw_gate->ptr, (float *)conv_state->ptr,
            qw, kw, vw, a_log, dt_bias, n_heads, n_tokens,
            gate_lower_bound);
    if (!cuda_ok(cudaGetLastError(),
                 "GLM-5.3 KDA prefill prepare launch")) {
        return 0;
    }
    const dim3 recurrence_grid(n_heads, 32u, 1u);
    glm53_rocm_kda_prefill_recurrence_kernel<<<recurrence_grid, 128u>>>(
        (float *)out->ptr, (float *)recurrent_state->ptr,
        (const float *)q->ptr, (const float *)k->ptr,
        (const float *)v->ptr, (const float *)raw_gate->ptr,
        (const float *)raw_beta->ptr, n_heads, n_tokens);
    if (!cuda_ok(cudaGetLastError(),
                 "GLM-5.3 KDA prefill recurrence launch")) {
        return 0;
    }
    const dim3 output_grid(n_tokens, n_heads, 1u);
    glm53_rocm_kda_prefill_output_kernel<<<
        output_grid, GLM53_ROCM_KDA_DIM>>>(
            (float *)out->ptr, (const float *)output_gate->ptr,
            output_norm, n_heads, n_tokens, norm_eps);
    return cuda_ok(cudaGetLastError(), "GLM-5.3 KDA prefill output launch");
}

__global__ static void glm_kv_lora_rms_norm_kernel(
        float *out,
        const float *kv_raw,
        const float *weight,
        uint32_t n_tokens,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        float eps) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const float *src = kv_raw + (uint64_t)token * kv_raw_dim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        const float v = src[i];
        ss += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)kv_lora_dim + eps);
    float *dst = out + (uint64_t)token * kv_lora_dim;
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        dst[i] = src[i] * scale * weight[i];
    }
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
        bool cache_f16) {
    const uint32_t token = blockIdx.x;
    const uint32_t part = blockIdx.y;
    if (token >= n_tokens || part > 1u) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    if (part == 0u) {
        const float *src = kv_norm + (uint64_t)token * kv_lora_dim;
        for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
            glm_rocm_cache_store(kv_lora_cache, (uint64_t)pos * kv_lora_dim + i, cache_f16, src[i]);
        }
    } else {
        const float *src = kv_raw + (uint64_t)token * kv_raw_dim + kv_lora_dim;
        for (uint32_t i = threadIdx.x; i < qk_rope; i += blockDim.x) {
            glm_rocm_cache_store(k_rope_cache, (uint64_t)pos * qk_rope + i, cache_f16, src[i]);
        }
    }
}

__global__ static void glm_qkv_norm_store_compact_kv_kernel(
        float *q_out,
        const float *q,
        const float *q_weight,
        uint32_t q_n,
        char *kv_lora_cache,
        char *k_rope_cache,
        const float *kv_raw,
        const float *kv_weight,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope,
        bool cache_f16,
        float eps) {
    const uint32_t token = blockIdx.x;
    const uint32_t part = blockIdx.y;
    if (token >= n_tokens || part > 2u) return;

    if (part == 2u) {
        const uint32_t pos = pos0 + token;
        if (pos >= cache_cap) return;
        const float *src = kv_raw + (uint64_t)token * kv_raw_dim + kv_lora_dim;
        for (uint32_t i = threadIdx.x; i < qk_rope; i += blockDim.x) {
            glm_rocm_cache_store(k_rope_cache, (uint64_t)pos * qk_rope + i, cache_f16, src[i]);
        }
        return;
    }

    const bool kv_task = part != 0u;
    const uint32_t n = kv_task ? kv_lora_dim : q_n;
    const float *src = kv_task ? kv_raw + (uint64_t)token * kv_raw_dim
                               : q + (uint64_t)token * q_n;
    const float *w = kv_task ? kv_weight : q_weight;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = src[i];
        ss += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    if (!kv_task) {
        float *dst = q_out + (uint64_t)token * q_n;
        for (uint32_t i = threadIdx.x; i < q_n; i += blockDim.x) dst[i] = src[i] * scale * w[i];
    } else {
        const uint32_t pos = pos0 + token;
        if (pos >= cache_cap) return;
        for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
            glm_rocm_cache_store(kv_lora_cache, (uint64_t)pos * kv_lora_dim + i,
                                 cache_f16, src[i] * scale * w[i]);
        }
    }
}

__global__ static void glm_store_indexer_k_kernel(
        char *indexer_key_cache,
        const float *raw_k,
        const float *weight,
        const float *bias,
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
        bool cache_f16) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const float *src = raw_k + (uint64_t)token * head_dim;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) sum += src[i];
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = partial[0] / (float)head_dim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)head_dim + eps);

    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        if (i < rot_dim) {
            if (i & 1u) continue;
            float c, s;
            glm_rocm_rope_pair(pos, i, rot_dim, n_ctx_orig, freq_base, freq_scale,
                               ext_factor, attn_factor, beta_fast, beta_slow, &c, &s);
            const float x0 = (src[i] - mean) * inv * weight[i] + bias[i];
            const float x1 = (src[i + 1u] - mean) * inv * weight[i + 1u] + bias[i + 1u];
            glm_rocm_cache_store(indexer_key_cache, (uint64_t)pos * head_dim + i,
                                 cache_f16, x0 * c - x1 * s);
            glm_rocm_cache_store(indexer_key_cache, (uint64_t)pos * head_dim + i + 1u,
                                 cache_f16, x0 * s + x1 * c);
        } else {
            const float x = (src[i] - mean) * inv * weight[i] + bias[i];
            glm_rocm_cache_store(indexer_key_cache, (uint64_t)pos * head_dim + i, cache_f16, x);
        }
    }
}

__global__ static void glm_k_b_project_q8_0_kernel(
        float *out,
        const unsigned char *weight,
        const float *kv_norm,
        uint32_t n_tokens,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t n_head,
        uint32_t row_bytes) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t q = threadIdx.x + blockIdx.z * blockDim.x;
    if (token >= n_tokens || head >= n_head || q >= qk_nope) return;
    const float *kv = kv_norm + (uint64_t)token * kv_lora_dim;
    float acc = 0.0f;
    const uint32_t b = q >> 5u;
    const uint32_t j_in_block = q & 31u;
    for (uint32_t j = 0; j < kv_lora_dim; j++) {
        const unsigned char *row = weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        const unsigned char *blk = row + (uint64_t)b * 34u;
        acc += q8_0_scale_scalar(blk) * (float)((const int8_t *)(blk + 2u))[j_in_block] * kv[j];
    }
    out[((uint64_t)token * n_head + head) * qk_nope + q] = acc;
}

__global__ static void glm_k_b_project_q8_0_head_kernel(
        float *out,
        const unsigned char *weight,
        const float *kv_norm,
        uint32_t n_tokens,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t n_head,
        uint32_t row_bytes) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;
    extern __shared__ float x[];
    const float *kv = kv_norm + (uint64_t)token * kv_lora_dim;
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) x[i] = kv[i];
    __syncthreads();

    float *dst = out + ((uint64_t)token * n_head + head) * qk_nope;
    for (uint32_t q = threadIdx.x; q < qk_nope; q += blockDim.x) {
        const uint32_t b = q >> 5u;
        const uint32_t j_in_block = q & 31u;
        float acc = 0.0f;
        for (uint32_t j = 0; j < kv_lora_dim; j++) {
            const unsigned char *row =
                weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
            const unsigned char *blk = row + (uint64_t)b * 34u;
            acc += q8_0_scale_scalar(blk) *
                   (float)((const int8_t *)(blk + 2u))[j_in_block] *
                   x[j];
        }
        dst[q] = acc;
    }
}

__global__ static void glm_q8_project_rows_kernel(
        float *out,
        const unsigned char *weight,
        const float *x,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t x_stride,
        uint32_t x_head_stride,
        uint32_t row_bytes) {
    const uint32_t out_row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (token >= n_tokens || out_row >= n_head * out_dim) return;
    const uint32_t head = out_row / out_dim;
    const uint32_t d = out_row - head * out_dim;
    const float *xr = x + (uint64_t)token * x_stride + (uint64_t)head * x_head_stride;
    const unsigned char *row = weight + ((uint64_t)head * out_dim + d) * row_bytes;
    float acc = 0.0f;
    const uint32_t blocks = (in_dim + 31u) >> 5u;
    for (uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint32_t base = b << 5u;
        const uint32_t count = min(32u, in_dim - base);
        const unsigned char *blk = row + (uint64_t)b * 34u;
        const float scale = q8_0_scale_scalar(blk);
        const int8_t *qs = (const int8_t *)(blk + 2u);
        for (uint32_t i = 0; i < count; i++) acc += scale * (float)qs[i] * xr[base + i];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        out[((uint64_t)token * n_head + head) * out_dim + d] = partial[0];
    }
}

__global__ static void glm_q8_project_head_kernel(
        float *out,
        const unsigned char *weight,
        const float *x,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t x_stride,
        uint32_t x_head_stride,
        uint32_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;
    extern __shared__ float shx[];
    const float *xr = x + (uint64_t)token * x_stride + (uint64_t)head * x_head_stride;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) shx[i] = xr[i];
    __syncthreads();

    float *dst = out + ((uint64_t)token * n_head + head) * out_dim;
    for (uint32_t d = threadIdx.x; d < out_dim; d += blockDim.x) {
        const unsigned char *row = weight + ((uint64_t)head * out_dim + d) * row_bytes;
        dst[d] = glm_rocm_q8_0_dot_row(row, shx, in_dim);
    }
}

/* Diagnostic port of the heterogeneous GLM branch's one-token Q8-head
 * projection.  One wave cooperates on each output row instead of assigning a
 * complete serial dot product to one thread.  The reduction changes FP32
 * summation order, so keep it opt-in until layer and frontier-logit comparisons
 * establish the acceptable numerical envelope. */
__global__ static void glm_q8_project_head_wave_kernel(
        float *out,
        const unsigned char *weight,
        const float *x,
        uint32_t n_head,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t x_stride,
        uint32_t x_head_stride,
        uint32_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t waves_per_block = blockDim.x >> 5u;
    const uint32_t d = blockIdx.x * waves_per_block + wave;
    const uint32_t head = blockIdx.y;
    const uint32_t token = blockIdx.z;
    if (d >= out_dim || head >= n_head) return;

    const float *xr =
        x + (uint64_t)token * x_stride + (uint64_t)head * x_head_stride;
    const unsigned char *row =
        weight + ((uint64_t)head * out_dim + d) * row_bytes;
    float acc = 0.0f;
    for (uint32_t i = lane; i < in_dim; i += 32u) {
        const unsigned char *blk = row + (uint64_t)(i >> 5u) * 34u;
        const float scale = q8_0_scale_scalar(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[i & 31u];
        acc += scale * (float)q * xr[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) {
        out[((uint64_t)token * n_head + head) * out_dim + d] = acc;
    }
}

__global__ static void glm_build_kv_cache_kernel(
        char *key_cache,
        char *value_cache,
        const float *kv_raw,
        const float *k_nope,
        const float *value,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        bool cache_f16) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const uint64_t kbase = ((uint64_t)pos * n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * n_head + head) * value_dim;
    const float *kn = k_nope + ((uint64_t)token * n_head + head) * qk_nope;
    const float *val = value + ((uint64_t)token * n_head + head) * value_dim;
    const float *raw = kv_raw + (uint64_t)token * kv_raw_dim;
    for (uint32_t i = threadIdx.x; i < qk_nope; i += blockDim.x) {
        glm_rocm_cache_store(key_cache, kbase + i, cache_f16, kn[i]);
    }
    for (uint32_t r = threadIdx.x * 2u; r < qk_rope; r += blockDim.x * 2u) {
        float c, s;
        glm_rocm_rope_pair(pos, r, qk_rope, n_ctx_orig, freq_base, freq_scale,
                           ext_factor, attn_factor, beta_fast, beta_slow, &c, &s);
        const float x0 = raw[kv_lora_dim + r];
        const float x1 = raw[kv_lora_dim + r + 1u];
        const uint32_t dst = qk_nope + r;
        glm_rocm_cache_store(key_cache, kbase + dst, cache_f16, x0 * c - x1 * s);
        glm_rocm_cache_store(key_cache, kbase + dst + 1u, cache_f16, x0 * s + x1 * c);
    }
    for (uint32_t i = threadIdx.x; i < value_dim; i += blockDim.x) {
        glm_rocm_cache_store(value_cache, vbase + i, cache_f16, val[i]);
    }
}

__global__ static void glm_attention_full_kernel(
        float *heads,
        const float *q,
        const char *key_cache,
        const char *value_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t qk_dim,
        uint32_t value_dim,
        bool cache_f16) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;
    const uint32_t visible = min(cache_len, pos0 + token + 1u);
    if (visible > cache_cap) return;
    extern __shared__ float sh[];
    float *red = sh;
    float *scores = sh + 256u;
    const float *qh = q + ((uint64_t)token * n_head + head) * qk_dim;
    const float scale = rsqrtf((float)qk_dim);

    float local_max = -INFINITY;
    for (uint32_t row = threadIdx.x; row < visible; row += blockDim.x) {
        const uint64_t kbase = ((uint64_t)row * n_head + head) * qk_dim;
        float dotv = 0.0f;
        for (uint32_t i = 0; i < qk_dim; i++) {
            dotv += qh[i] * glm_rocm_cache_load(key_cache, kbase + i, cache_f16);
        }
        const float score = dotv * scale;
        scores[row] = score;
        local_max = fmaxf(local_max, score);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_score = red[0];
    float local_sum = 0.0f;
    for (uint32_t row = threadIdx.x; row < visible; row += blockDim.x) {
        const float w = expf(scores[row] - max_score);
        scores[row] = w;
        local_sum += w;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float denom = fmaxf(red[0], 1.0e-20f);
    float *out = heads + ((uint64_t)token * n_head + head) * value_dim;
    for (uint32_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t row = 0; row < visible; row++) {
            const uint64_t vbase = ((uint64_t)row * n_head + head) * value_dim;
            acc += scores[row] * glm_rocm_cache_load(value_cache, vbase + d, cache_f16);
        }
        out[d] = acc / denom;
    }
}

__global__ static void glm_fill_selected_range_kernel(int32_t *selected, uint32_t n_selected) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_selected) selected[i] = (int32_t)i;
}

__global__ static void glm_fill_selected_range_batch_kernel(
        int32_t *selected,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t pad_row) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * n_selected;
    if (gid >= total || n_selected == 0u) return;
    const uint32_t token = (uint32_t)(gid / n_selected);
    const uint32_t slot = (uint32_t)(gid - (uint64_t)token * n_selected);
    const uint32_t visible = pos0 + token + 1u;
    selected[gid] = (int32_t)(slot < visible ? slot : pad_row);
}

__global__ static void glm_indexer_rope_tail_kernel(
        float *x,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t pairs = (uint64_t)n_tokens * n_head * (rot_dim >> 1u);
    if (gid >= pairs) return;
    const uint32_t pair = (uint32_t)(gid % (rot_dim >> 1u));
    const uint64_t tmp = gid / (rot_dim >> 1u);
    const uint32_t head = (uint32_t)(tmp % n_head);
    const uint32_t token = (uint32_t)(tmp / n_head);
    const uint32_t r = pair << 1u;
    float *row = x + ((uint64_t)token * n_head + head) * head_dim;
    float c, s;
    glm_rocm_rope_pair(pos0 + token, r, rot_dim, n_ctx_orig, freq_base, freq_scale,
                       ext_factor, attn_factor, beta_fast, beta_slow, &c, &s);
    const float x0 = row[r];
    const float x1 = row[r + 1u];
    row[r] = x0 * c - x1 * s;
    row[r + 1u] = x0 * s + x1 * c;
}

__global__ static void glm_indexer_score_one_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const char *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim,
        float scale,
        bool cache_f16) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    __shared__ float partial[256];
    float score = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        float acc = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            acc += qh[d] * glm_rocm_cache_load(indexer_key_cache, (uint64_t)row * head_dim + d, cache_f16);
        }
        partial[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0u) score += fmaxf(partial[0] * scale, 0.0f) * weights[h];
        __syncthreads();
    }
    if (threadIdx.x == 0u) scores[row] = score;
}

__global__ static void glm_indexer_scores_batch_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const char *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t row_group_size,
        uint32_t n_head,
        uint32_t head_dim,
        float scale,
        bool cache_f16) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (row >= n_rows || token >= n_tokens) return;
    float *dst = scores + (uint64_t)token * n_rows + row;
    if (row >= (pos0 + token + 1u) / row_group_size) {
        if (threadIdx.x == 0u) *dst = -INFINITY;
        return;
    }
    __shared__ float partial[256];
    float score = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)token * n_head + h) * head_dim;
        float acc = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            acc += qh[d] * glm_rocm_cache_load(indexer_key_cache, (uint64_t)row * head_dim + d, cache_f16);
        }
        partial[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0u) score += fmaxf(partial[0] * scale, 0.0f) *
                                        weights[(uint64_t)token * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0u) *dst = score;
}

__global__ static void glm_attention_indexed_lora_kernel(
        float *lora_out,
        const float *q,
        const float *qk_low,
        const char *kv_lora_cache,
        const char *k_rope_cache,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t n_ctx_orig,
        bool causal_range,
        bool has_selected,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens || n_selected == 0u) return;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const float scale = rsqrtf((float)qk_dim);
    extern __shared__ float sh[];
    float *red = sh;
    float *scores = sh + 256u;
    const float *qh = q + ((uint64_t)token * n_head + head) * qk_dim;
    const float *low = qk_low + ((uint64_t)token * n_head + head) * kv_lora_dim;

    float local_max = -INFINITY;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        int32_t row_i = causal_range ? (int32_t)s
                       : (has_selected ? selected[(uint64_t)token * n_selected + s] : -1);
        const uint32_t visible = pos0 + token + 1u;
        bool valid = row_i >= 0 && (uint32_t)row_i < cache_cap;
        if (causal_range) valid = valid && (uint32_t)row_i < visible;
        const uint32_t row = valid ? (uint32_t)row_i : 0u;
        float score = -INFINITY;
        if (valid) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            for (uint32_t j = 0; j < kv_lora_dim; j++) {
                dotv += low[j] * glm_rocm_cache_load(kv_lora_cache, lora_base + j, cache_f16);
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float2 y = glm_rocm_rotated_cache_rope_pair(k_rope_cache,
                                                                  rope_base,
                                                                  r,
                                                                  row,
                                                                  qk_rope,
                                                                  cache_f16,
                                                                  n_ctx_orig,
                                                                  freq_base,
                                                                  freq_scale,
                                                                  ext_factor,
                                                                  attn_factor,
                                                                  beta_fast,
                                                                  beta_slow);
                dotv += qh[qk_nope + r] * y.x + qh[qk_nope + r + 1u] * y.y;
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_score = red[0];
    __syncthreads();
    if (!isfinite(max_score)) {
        float *out = lora_out + ((uint64_t)token * n_head + head) * kv_lora_dim;
        for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
            out[j] = 0.0f;
        }
        return;
    }
    float local_sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        const float w = expf(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float denom = fmaxf(red[0], 1.0e-20f);
    float *out = lora_out + ((uint64_t)token * n_head + head) * kv_lora_dim;
    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_selected; s++) {
            int32_t row_i = causal_range ? (int32_t)s
                           : (has_selected ? selected[(uint64_t)token * n_selected + s] : -1);
            const uint32_t visible = pos0 + token + 1u;
            bool valid = row_i >= 0 && (uint32_t)row_i < cache_cap;
            if (causal_range) valid = valid && (uint32_t)row_i < visible;
            if (valid) {
                acc += scores[s] * glm_rocm_cache_load(kv_lora_cache,
                                                       (uint64_t)(uint32_t)row_i * kv_lora_dim + j,
                                                       cache_f16);
            }
        }
        out[j] = acc / denom;
    }
}

__global__ static void glm_attention_indexed_decode_split_partial_kernel(
        float *partial_lora,
        float *partial_ms,
        const float *q,
        const float *qk_low,
        const char *kv_lora_cache,
        const char *k_rope_cache,
        const int32_t *selected,
        uint32_t n_selected,
        bool selected_rows_valid,
        uint32_t cache_cap,
        bool cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t n_ctx_orig,
        uint32_t block_rows,
        uint32_t n_blocks,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t block = blockIdx.y;
    if (head >= n_head || block >= n_blocks || block_rows == 0u) return;

    const uint32_t qk_dim = qk_nope + qk_rope;
    const float scale = rsqrtf((float)qk_dim);
    const uint32_t block_start = block * block_rows;
    const uint32_t block_end = block_start < n_selected ?
        min(n_selected, block_start + block_rows) : block_start;
    const uint32_t rows = block_end - block_start;

    extern __shared__ float sh[];
    float *red = sh;
    float *scores = sh + 256u;
    const float *qh = q + (uint64_t)head * qk_dim;
    const float *low = qk_low + (uint64_t)head * kv_lora_dim;

    float local_max = -INFINITY;
    for (uint32_t s = threadIdx.x; s < rows; s += blockDim.x) {
        const int32_t row_i = selected[block_start + s];
        const bool valid =
            selected_rows_valid ||
            (row_i >= 0 && (uint32_t)row_i < cache_cap);
        const uint32_t row =
            selected_rows_valid ? (uint32_t)row_i :
            (valid ? (uint32_t)row_i : 0u);
        float score = -INFINITY;
        if (valid) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            for (uint32_t j = 0; j < kv_lora_dim; j++) {
                dotv += low[j] * glm_rocm_cache_load(kv_lora_cache, lora_base + j, cache_f16);
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float2 y = glm_rocm_rotated_cache_rope_pair(k_rope_cache,
                                                                  rope_base,
                                                                  r,
                                                                  row,
                                                                  qk_rope,
                                                                  cache_f16,
                                                                  n_ctx_orig,
                                                                  freq_base,
                                                                  freq_scale,
                                                                  ext_factor,
                                                                  attn_factor,
                                                                  beta_fast,
                                                                  beta_slow);
                dotv += qh[qk_nope + r] * y.x + qh[qk_nope + r + 1u] * y.y;
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_score = red[0];
    float *out = partial_lora + ((uint64_t)block * n_head + head) * kv_lora_dim;
    float *ms = partial_ms + ((uint64_t)block * n_head + head) * 2u;
    if (!isfinite(max_score)) {
        for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) out[j] = 0.0f;
        if (threadIdx.x == 0u) {
            ms[0] = -INFINITY;
            ms[1] = 0.0f;
        }
        return;
    }

    float local_sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < rows; s += blockDim.x) {
        const float w = expf(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float sum = red[0];
    if (threadIdx.x == 0u) {
        ms[0] = max_score;
        ms[1] = sum;
    }

    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < rows; s++) {
            const int32_t row_i = selected[block_start + s];
            const bool valid =
                selected_rows_valid ||
                (row_i >= 0 && (uint32_t)row_i < cache_cap);
            if (valid) {
                acc += scores[s] * glm_rocm_cache_load(kv_lora_cache,
                                                       (uint64_t)(uint32_t)row_i * kv_lora_dim + j,
                                                       cache_f16);
            }
        }
        out[j] = acc;
    }
}

__global__ static void glm_attention_indexed_decode_split_group8_partial_valid_kernel(
        float *partial_lora,
        float *partial_ms,
        const float *q,
        const float *qk_low,
        const char *kv_lora_cache,
        const char *k_rope_cache,
        const int32_t *selected,
        uint32_t n_selected,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t n_ctx_orig,
        uint32_t block_rows,
        uint32_t n_blocks,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t lane = threadIdx.x;
    const uint32_t head_in_group = threadIdx.y;
    const uint32_t group_heads = 8u;
    const uint32_t stage_rows = 16u;
    const uint32_t head = blockIdx.x * group_heads + head_in_group;
    const uint32_t block = blockIdx.y;
    if (lane >= 32u ||
        head_in_group >= group_heads ||
        head >= n_head ||
        block >= n_blocks ||
        n_selected == 0u ||
        block_rows == 0u ||
        kv_lora_dim != 512u ||
        qk_rope != 64u) {
        return;
    }

    const uint32_t qk_dim = qk_nope + qk_rope;
    const float scale = rsqrtf((float)qk_dim);
    const uint32_t block_start = block * block_rows;
    const uint32_t block_end = block_start < n_selected ?
        min(n_selected, block_start + block_rows) : block_start;
    const uint32_t rope_pairs = qk_rope >> 1u;
    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float sh[];
    float *kv_shared = sh;
    float *rope_shared = kv_shared + stage_rows * kv_lora_dim;

    const float *qh = q + (uint64_t)head * qk_dim;
    const float *low = qk_low + (uint64_t)head * kv_lora_dim;

    const float4 low0 = glm_rocm_load4_f32(low + (lane + 0u) * 4u);
    const float4 low1 = glm_rocm_load4_f32(low + (lane + 32u) * 4u);
    const float4 low2 = glm_rocm_load4_f32(low + (lane + 64u) * 4u);
    const float4 low3 = glm_rocm_load4_f32(low + (lane + 96u) * 4u);
    const float4 qrope = lane < 16u ?
        glm_rocm_load4_f32(qh + qk_nope + lane * 4u) :
        make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float M = -INFINITY;
    float S = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t base = block_start; base < block_end; base += stage_rows) {
        const uint32_t rows = min(stage_rows, block_end - base);
        for (uint32_t off = tid; off < rows * kv_lora_dim; off += 256u) {
            const uint32_t rr = off / kv_lora_dim;
            const uint32_t d = off - rr * kv_lora_dim;
            const uint32_t row = (uint32_t)selected[base + rr];
            kv_shared[off] =
                __half2float(((const __half *)kv_lora_cache)
                             [(uint64_t)row * kv_lora_dim + d]);
        }
        for (uint32_t off = tid; off < rows * rope_pairs; off += 256u) {
            const uint32_t rr = off / rope_pairs;
            const uint32_t pair = off - rr * rope_pairs;
            const uint32_t r = pair << 1u;
            const uint32_t row = (uint32_t)selected[base + rr];
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            const float2 y =
                glm_rocm_rotated_cache_rope_pair(k_rope_cache,
                                                 rope_base,
                                                 r,
                                                 row,
                                                 qk_rope,
                                                 true,
                                                 n_ctx_orig,
                                                 freq_base,
                                                 freq_scale,
                                                 ext_factor,
                                                 attn_factor,
                                                 beta_fast,
                                                 beta_slow);
            rope_shared[(uint64_t)rr * qk_rope + r] = y.x;
            rope_shared[(uint64_t)rr * qk_rope + r + 1u] = y.y;
        }
        __syncthreads();

        for (uint32_t rr = 0; rr < rows; rr++) {
            const float *kv_row = kv_shared + (uint64_t)rr * kv_lora_dim;
            const float *rope_row = rope_shared + (uint64_t)rr * qk_rope;
            float partial = 0.0f;
            partial += glm_rocm_dot4(low0, glm_rocm_load4_f32(kv_row + (lane + 0u) * 4u));
            partial += glm_rocm_dot4(low1, glm_rocm_load4_f32(kv_row + (lane + 32u) * 4u));
            partial += glm_rocm_dot4(low2, glm_rocm_load4_f32(kv_row + (lane + 64u) * 4u));
            partial += glm_rocm_dot4(low3, glm_rocm_load4_f32(kv_row + (lane + 96u) * 4u));
            if (lane < 16u) {
                partial += glm_rocm_dot4(qrope,
                                         glm_rocm_load4_f32(rope_row + lane * 4u));
            }

            float score = warp_sum_f32(partial) * scale;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            score = __shfl(score, 0, 32);
#else
            score = __shfl_sync(FULL_WARP_MASK, score, 0, 32);
#endif
            const float new_m = fmaxf(M, score);
            const float old_scale = expf(M - new_m);
            const float row_scale = expf(score - new_m);
            const float4 kv0 = glm_rocm_load4_f32(kv_row + (lane + 0u) * 4u);
            const float4 kv1 = glm_rocm_load4_f32(kv_row + (lane + 32u) * 4u);
            const float4 kv2 = glm_rocm_load4_f32(kv_row + (lane + 64u) * 4u);
            const float4 kv3 = glm_rocm_load4_f32(kv_row + (lane + 96u) * 4u);
            o0 = make_float4(o0.x * old_scale + kv0.x * row_scale,
                             o0.y * old_scale + kv0.y * row_scale,
                             o0.z * old_scale + kv0.z * row_scale,
                             o0.w * old_scale + kv0.w * row_scale);
            o1 = make_float4(o1.x * old_scale + kv1.x * row_scale,
                             o1.y * old_scale + kv1.y * row_scale,
                             o1.z * old_scale + kv1.z * row_scale,
                             o1.w * old_scale + kv1.w * row_scale);
            o2 = make_float4(o2.x * old_scale + kv2.x * row_scale,
                             o2.y * old_scale + kv2.y * row_scale,
                             o2.z * old_scale + kv2.z * row_scale,
                             o2.w * old_scale + kv2.w * row_scale);
            o3 = make_float4(o3.x * old_scale + kv3.x * row_scale,
                             o3.y * old_scale + kv3.y * row_scale,
                             o3.z * old_scale + kv3.z * row_scale,
                             o3.w * old_scale + kv3.w * row_scale);
            S = S * old_scale + row_scale;
            M = new_m;
        }
        __syncthreads();
    }

    float *out =
        partial_lora + ((uint64_t)block * n_head + head) * kv_lora_dim;
    glm_rocm_store4_f32(out + (lane + 0u) * 4u, o0);
    glm_rocm_store4_f32(out + (lane + 32u) * 4u, o1);
    glm_rocm_store4_f32(out + (lane + 64u) * 4u, o2);
    glm_rocm_store4_f32(out + (lane + 96u) * 4u, o3);
    if (lane == 0u) {
        float *ms = partial_ms + ((uint64_t)block * n_head + head) * 2u;
        ms[0] = M;
        ms[1] = S;
    }
}

__global__ static void glm_attention_indexed_decode_split_reduce_kernel(
        float *heads,
        const float *partial_lora,
        const float *partial_ms,
        const unsigned char *value_weight,
        uint32_t n_selected,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint32_t row_bytes,
        uint32_t n_blocks) {
    const uint32_t head = blockIdx.x;
    if (head >= n_head || n_selected == 0u || n_blocks == 0u || n_blocks > 64u) return;

    extern __shared__ float sh[];
    float *red = sh;
    float *block_scale = sh + 256u;
    float *lora_sum = sh + 320u;

    float local_m = -INFINITY;
    if (threadIdx.x < n_blocks) {
        const float *ms = partial_ms + ((uint64_t)threadIdx.x * n_head + head) * 2u;
        local_m = ms[1] > 0.0f ? ms[0] : -INFINITY;
    }
    red[threadIdx.x] = local_m;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_m = red[0];
    if (!isfinite(max_m)) {
        for (uint32_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
            heads[(uint64_t)head * value_dim + d] = 0.0f;
        }
        return;
    }

    float local_denom = 0.0f;
    if (threadIdx.x < n_blocks) {
        const float *ms = partial_ms + ((uint64_t)threadIdx.x * n_head + head) * 2u;
        const float s = ms[1];
        const float e = s > 0.0f ? expf(ms[0] - max_m) : 0.0f;
        block_scale[threadIdx.x] = e;
        local_denom = s * e;
    }
    red[threadIdx.x] = local_denom;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float denom = fmaxf(red[0], 1.0e-20f);

    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t b = 0; b < n_blocks; b++) {
            const float *src = partial_lora + ((uint64_t)b * n_head + head) * kv_lora_dim;
            acc += src[j] * block_scale[b];
        }
        lora_sum[j] = acc / denom;
    }
    __syncthreads();

    float *dst = heads + (uint64_t)head * value_dim;
    for (uint32_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
        const unsigned char *row = value_weight + ((uint64_t)head * value_dim + d) * row_bytes;
        dst[d] = glm_rocm_q8_0_dot_row(row, lora_sum, kv_lora_dim);
    }
}

static int glm_rocm_check_q8_rows(
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t rows,
        uint32_t cols,
        const char *label,
        const unsigned char **ptr_out,
        uint32_t *row_bytes_out) {
    const uint64_t blocks = (cols + 31u) >> 5u;
    uint64_t row_bytes = 0, bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(rows, row_bytes, &bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, bytes)) {
        return 0;
    }
    const char *ptr = cuda_model_range_ptr(model_map, weight_offset, bytes, label);
    if (!ptr) return 0;
    *ptr_out = (const unsigned char *)ptr;
    *row_bytes_out = (uint32_t)row_bytes;
    return 1;
}

extern "C" int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor *selected,
        uint32_t n_selected) {
    if (!table || n_selected > DS4_ROCM_N_EXPERT_USED ||
        (n_selected != 0u && !cuda_tensor_has_i32(selected, n_selected))) {
        return 0;
    }
    int32_t ids[DS4_ROCM_N_EXPERT_USED] = {0};
    if (n_selected != 0u &&
        !ds4_gpu_tensor_read((ds4_gpu_tensor *)selected, 0, ids, n_selected * sizeof(ids[0]))) {
        return 0;
    }
    if (!ds4_gpu_stream_expert_cache_begin_selected_load(table, ids, n_selected)) {
        return 0;
    }
    return ds4_gpu_routed_moe_set_selected_override(ids, n_selected);
}

extern "C" int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    return ds4_gpu_rope_tail_tensor(x, n_tokens, n_head, head_dim, rot_dim, pos0,
                                    n_ctx_orig, false, freq_base, freq_scale,
                                    ext_factor, attn_factor, beta_fast, beta_slow);
}

extern "C" int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *kv_raw,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_tokens,
        uint32_t kv_raw_dim,
    uint32_t kv_lora_dim,
    float eps) {
    uint64_t weight_bytes = 0;
    if (!out || !kv_raw || !model_map || n_tokens == 0u ||
        kv_raw_dim == 0u || kv_lora_dim == 0u ||
        !cuda_tensor_has_elems2(kv_raw, n_tokens, kv_raw_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, kv_lora_dim, sizeof(float)) ||
        !glm_rocm_model_f32_range(model_size, weight_offset, kv_lora_dim, &weight_bytes)) {
        return 0;
    }
    const float *w = (const float *)cuda_model_range_ptr(model_map,
                                                         weight_offset,
                                                         weight_bytes,
                                                         "glm_kv_lora_norm");
    if (!w) return 0;
    glm_kv_lora_rms_norm_kernel<<<n_tokens, 256>>>((float *)out->ptr,
                                                   (const float *)kv_raw->ptr,
                                                   w,
                                                   n_tokens,
                                                   kv_raw_dim,
                                                   kv_lora_dim,
                                                   eps);
    return cuda_ok(cudaGetLastError(), "glm kv_lora rms norm launch");
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
    const unsigned char *w = NULL;
    uint32_t row_bytes = 0;
    if (!out || !kv_norm || !model_map || n_tokens == 0u ||
        n_head == 0u || kv_lora_dim == 0u || qk_nope == 0u ||
        kv_lora_dim > DS4_ROCM_ATTENTION_SCORE_CAP ||
        !cuda_tensor_has_elems2(kv_norm, n_tokens, kv_lora_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(out, n_tokens, n_head, qk_nope, sizeof(float)) ||
        !glm_rocm_check_q8_rows(model_map, model_size, weight_offset,
                                (uint64_t)n_head * kv_lora_dim, qk_nope,
                                "glm_k_b", &w, &row_bytes)) {
        return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    glm_k_b_project_q8_0_head_kernel<<<grid, 256, (size_t)kv_lora_dim * sizeof(float)>>>(
            (float *)out->ptr,
            w,
            (const float *)kv_norm->ptr,
            n_tokens,
            kv_lora_dim,
            qk_nope,
            n_head,
            row_bytes);
    return cuda_ok(cudaGetLastError(), "glm k_b project launch");
}

extern "C" int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor *kv_lora_cache,
        ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope,
        bool cache_f16) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!kv_lora_cache || (qk_rope != 0u && !k_rope_cache) ||
        !kv_norm || !kv_raw ||
        !glm_rocm_check_pos_span(pos0, n_tokens, cache_cap) ||
        kv_raw_dim == 0u || kv_lora_dim == 0u ||
        kv_lora_dim > kv_raw_dim || qk_rope > kv_raw_dim - kv_lora_dim ||
        !cuda_tensor_has_elems2(kv_norm, n_tokens, kv_lora_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(kv_raw, n_tokens, kv_raw_dim, sizeof(float)) ||
        !glm_rocm_tensor_has_cache2(kv_lora_cache, cache_cap, kv_lora_dim, elem) ||
        (qk_rope != 0u &&
         !glm_rocm_tensor_has_cache2(k_rope_cache, cache_cap, qk_rope, elem))) {
        fprintf(stderr,
                "ds4: GLM compact KV store received invalid buffers "
                "(pos=%u tokens=%u cap=%u raw_dim=%u lora=%u rope=%u "
                "kv_cache=%llu rope_cache=%llu norm=%llu raw=%llu)\n",
                pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope,
                (unsigned long long)(kv_lora_cache ? kv_lora_cache->bytes : 0u),
                (unsigned long long)(k_rope_cache ? k_rope_cache->bytes : 0u),
                (unsigned long long)(kv_norm ? kv_norm->bytes : 0u),
                (unsigned long long)(kv_raw ? kv_raw->bytes : 0u));
        return 0;
    }
    dim3 grid(n_tokens, qk_rope != 0u ? 2u : 1u, 1);
    glm_store_compact_kv_kernel<<<grid, 256>>>((char *)kv_lora_cache->ptr,
                                               k_rope_cache ?
                                                   (char *)k_rope_cache->ptr : NULL,
                                               (const float *)kv_norm->ptr,
                                               (const float *)kv_raw->ptr,
                                               pos0,
                                               n_tokens,
                                               cache_cap,
                                               kv_raw_dim,
                                               kv_lora_dim,
                                               qk_rope,
                                               cache_f16);
    return cuda_ok(cudaGetLastError(), "glm store compact kv launch");
}

extern "C" int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor *q_out,
        const ds4_gpu_tensor *q,
        const void *model_map,
        uint64_t model_size,
        uint64_t q_weight_offset,
        uint32_t q_n,
        ds4_gpu_tensor *kv_lora_cache,
        ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t kv_weight_offset,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope,
        bool cache_f16,
        float eps) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    uint64_t q_weight_bytes = 0;
    uint64_t kv_weight_bytes = 0;
    if (!q_out || !q || !model_map || !kv_lora_cache ||
        (qk_rope != 0u && !k_rope_cache) || !kv_raw ||
        !glm_rocm_check_pos_span(pos0, n_tokens, cache_cap) ||
        q_n == 0u || kv_raw_dim == 0u || kv_lora_dim == 0u ||
        kv_lora_dim > kv_raw_dim || qk_rope > kv_raw_dim - kv_lora_dim ||
        !cuda_tensor_has_elems2(q, n_tokens, q_n, sizeof(float)) ||
        !cuda_tensor_has_elems2(q_out, n_tokens, q_n, sizeof(float)) ||
        !cuda_tensor_has_elems2(kv_raw, n_tokens, kv_raw_dim, sizeof(float)) ||
        !glm_rocm_tensor_has_cache2(kv_lora_cache, cache_cap, kv_lora_dim, elem) ||
        (qk_rope != 0u &&
         !glm_rocm_tensor_has_cache2(k_rope_cache, cache_cap, qk_rope, elem)) ||
        !glm_rocm_model_f32_range(model_size, q_weight_offset, q_n, &q_weight_bytes) ||
        !glm_rocm_model_f32_range(model_size, kv_weight_offset, kv_lora_dim, &kv_weight_bytes)) {
        return 0;
    }
    const float *qw = (const float *)cuda_model_range_ptr(model_map, q_weight_offset,
                                                          q_weight_bytes,
                                                          "glm_q_norm");
    const float *kvw = (const float *)cuda_model_range_ptr(model_map, kv_weight_offset,
                                                           kv_weight_bytes,
                                                           "glm_kv_norm");
    if (!qw || !kvw) return 0;
    dim3 grid(n_tokens, qk_rope != 0u ? 3u : 2u, 1);
    glm_qkv_norm_store_compact_kv_kernel<<<grid, 256>>>((float *)q_out->ptr,
                                                        (const float *)q->ptr,
                                                        qw,
                                                        q_n,
                                                        (char *)kv_lora_cache->ptr,
                                                        k_rope_cache ?
                                                            (char *)k_rope_cache->ptr : NULL,
                                                        (const float *)kv_raw->ptr,
                                                        kvw,
                                                        pos0,
                                                        n_tokens,
                                                        cache_cap,
                                                        kv_raw_dim,
                                                        kv_lora_dim,
                                                        qk_rope,
                                                        cache_f16,
                                                        eps);
    return cuda_ok(cudaGetLastError(), "glm qkv norm compact kv launch");
}

extern "C" int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t bias_offset,
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
        bool cache_f16) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    uint64_t bytes = 0;
    if (!indexer_key_cache || !raw_k || !model_map || n_tokens == 0u ||
        !glm_rocm_check_pos_span(pos0, n_tokens, cache_cap) ||
        head_dim == 0u ||
        rot_dim > head_dim || (rot_dim & 1u) ||
        !cuda_tensor_has_elems2(raw_k, n_tokens, head_dim, sizeof(float)) ||
        !glm_rocm_tensor_has_cache2(indexer_key_cache, cache_cap, head_dim, elem) ||
        !glm_rocm_model_f32_range(model_size, weight_offset, head_dim, &bytes) ||
        !cuda_model_range_fits(model_size, bias_offset, bytes)) {
        return 0;
    }
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset, bytes, "glm_indexer_weight");
    const float *b = (const float *)cuda_model_range_ptr(model_map, bias_offset, bytes, "glm_indexer_bias");
    if (!w || !b) return 0;
    glm_store_indexer_k_kernel<<<n_tokens, 256>>>((char *)indexer_key_cache->ptr,
                                                  (const float *)raw_k->ptr,
                                                  w,
                                                  b,
                                                  pos0,
                                                  n_tokens,
                                                  cache_cap,
                                                  head_dim,
                                                  rot_dim,
                                                  n_ctx_orig,
                                                  eps,
                                                  freq_base,
                                                  freq_scale,
                                                  ext_factor,
                                                  attn_factor,
                                                  beta_fast,
                                                  beta_slow,
                                                  cache_f16);
    return cuda_ok(cudaGetLastError(), "glm store indexer k launch");
}

extern "C" int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor *key_cache,
        ds4_gpu_tensor *value_cache,
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
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        bool cache_f16) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    uint32_t qk_dim = 0;
    if (!glm_rocm_u32_add_checked(qk_nope, qk_rope, &qk_dim)) return 0;
    if (!key_cache || !value_cache || !kv_raw || !k_nope || !value ||
        !glm_rocm_check_pos_span(pos0, n_tokens, cache_cap) ||
        n_head == 0u || kv_raw_dim == 0u || kv_lora_dim == 0u ||
        qk_nope == 0u || value_dim == 0u ||
        qk_rope == 0u || (qk_rope & 1u) ||
        !cuda_tensor_has_elems2(kv_raw, n_tokens, kv_raw_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(k_nope, n_tokens, n_head, qk_nope, sizeof(float)) ||
        !cuda_tensor_has_elems3(value, n_tokens, n_head, value_dim, sizeof(float)) ||
        !glm_rocm_tensor_has_cache3(key_cache, cache_cap, n_head, qk_dim, elem) ||
        !glm_rocm_tensor_has_cache3(value_cache, cache_cap, n_head, value_dim, elem)) {
        return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    glm_build_kv_cache_kernel<<<grid, 256>>>((char *)key_cache->ptr,
                                             (char *)value_cache->ptr,
                                             (const float *)kv_raw->ptr,
                                             (const float *)k_nope->ptr,
                                             (const float *)value->ptr,
                                             pos0,
                                             n_tokens,
                                             cache_cap,
                                             n_head,
                                             kv_raw_dim,
                                             kv_lora_dim,
                                             qk_nope,
                                             qk_rope,
                                             value_dim,
                                             n_ctx_orig,
                                             freq_base,
                                             freq_scale,
                                             ext_factor,
                                             attn_factor,
                                             beta_fast,
                                             beta_slow,
                                             cache_f16);
    return cuda_ok(cudaGetLastError(), "glm build kv cache launch");
}

extern "C" int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor *key_cache,
        ds4_gpu_tensor *value_cache,
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
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        bool cache_f16) {
    return ds4_gpu_glm_build_kv_cache_tensor(key_cache, value_cache, kv_raw, k_nope, value,
                                             pos0, n_tokens, cache_cap, n_head,
                                             kv_raw_dim, kv_lora_dim, qk_nope, qk_rope,
                                             value_dim, n_ctx_orig, freq_base, freq_scale,
                                             ext_factor, attn_factor, beta_fast, beta_slow,
                                             cache_f16);
}

extern "C" int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t qk_dim,
        uint32_t value_dim,
        bool cache_f16) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    uint32_t end_pos = 0;
    if (!heads || !q || !key_cache || !value_cache || n_tokens == 0u ||
        cache_len == 0u || cache_len > cache_cap ||
        !glm_rocm_check_token_span(pos0, n_tokens, &end_pos) ||
        end_pos > cache_len ||
        n_head == 0u || qk_dim == 0u || value_dim == 0u ||
        !cuda_attention_score_buffer_fits(cache_len) ||
        !cuda_tensor_has_elems3(q, n_tokens, n_head, qk_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(heads, n_tokens, n_head, value_dim, sizeof(float)) ||
        !glm_rocm_tensor_has_cache3(key_cache, cache_cap, n_head, qk_dim, elem) ||
        !glm_rocm_tensor_has_cache3(value_cache, cache_cap, n_head, value_dim, elem)) {
        return 0;
    }
    const size_t shmem = ((size_t)256u + cache_len) * sizeof(float);
    dim3 grid(n_tokens, n_head, 1);
    glm_attention_full_kernel<<<grid, 256, shmem>>>((float *)heads->ptr,
                                                    (const float *)q->ptr,
                                                    (const char *)key_cache->ptr,
                                                    (const char *)value_cache->ptr,
                                                    pos0,
                                                    n_tokens,
                                                    cache_len,
                                                    cache_cap,
                                                    n_head,
                                                    qk_dim,
                                                    value_dim,
                                                    cache_f16);
    return cuda_ok(cudaGetLastError(), "glm attention full launch");
}

extern "C" int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t qk_dim,
        uint32_t value_dim,
        bool cache_f16) {
    return ds4_gpu_glm_attention_full_tensor(heads, q, key_cache, value_cache, pos0,
                                             n_tokens, cache_len, cache_cap, n_head,
                                             qk_dim, value_dim, cache_f16);
}

extern "C" int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t qk_dim,
        uint32_t value_dim,
        bool cache_f16) {
    return ds4_gpu_glm_attention_full_tensor(heads, q, key_cache, value_cache, pos0,
                                             n_tokens, cache_len, cache_cap, n_head,
                                             qk_dim, value_dim, cache_f16);
}

extern "C" int ds4_gpu_glm_fill_selected_range_tensor(ds4_gpu_tensor *selected, uint32_t n_selected) {
    if (n_selected == 0u) return selected != NULL;
    if (!cuda_tensor_has_i32(selected, n_selected)) return 0;
    glm_fill_selected_range_kernel<<<(n_selected + 255u) / 256u, 256>>>((int32_t *)selected->ptr, n_selected);
    return cuda_ok(cudaGetLastError(), "glm fill selected range launch");
}

extern "C" int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t pad_row) {
    uint32_t end_pos = 0;
    if (!selected || n_tokens == 0u || n_selected == 0u ||
        !glm_rocm_check_token_span(pos0, n_tokens, &end_pos) ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t))) {
        return 0;
    }
    uint64_t total = 0;
    uint32_t blocks = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_selected, &total) ||
        !glm_rocm_launch_blocks(total, 256u, &blocks)) {
        return 0;
    }
    glm_fill_selected_range_batch_kernel<<<blocks, 256>>>((int32_t *)selected->ptr,
                                                                         n_tokens,
                                                                         pos0,
                                                                         n_selected,
                                                                         pad_row);
    return cuda_ok(cudaGetLastError(), "glm fill selected range batch launch");
}

extern "C" int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    if (!x || rot_dim > head_dim || (rot_dim & 1u) ||
        (n_tokens != 0u && !glm_rocm_check_token_span(pos0, n_tokens, NULL)) ||
        !cuda_tensor_has_elems3(x, n_tokens, n_head, head_dim, sizeof(float))) {
        return 0;
    }
    uint64_t token_heads = 0;
    uint64_t pairs = 0;
    uint32_t blocks = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_head, &token_heads) ||
        !cuda_u64_mul_checked(token_heads, rot_dim >> 1u, &pairs)) {
        return 0;
    }
    if (pairs == 0u) return 1;
    if (!glm_rocm_launch_blocks(pairs, 256u, &blocks)) return 0;
    glm_indexer_rope_tail_kernel<<<blocks, 256>>>((float *)x->ptr,
                                                                 n_tokens,
                                                                 n_head,
                                                                 head_dim,
                                                                 rot_dim,
                                                                 pos0,
                                                                 n_ctx_orig,
                                                                 freq_base,
                                                                 freq_scale,
                                                                 ext_factor,
                                                                 attn_factor,
                                                                 beta_fast,
                                                                 beta_slow);
    return cuda_ok(cudaGetLastError(), "glm indexer rope tail launch");
}

__global__ static void glm53_rocm_indexer_pool_update_kernel(
        char *pool_cache,
        float *tail,
        const float *raw_k,
        const float *gate,
        const float *norm_weight,
        const float *norm_bias,
        const uint16_t *ape,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t head_dim,
        uint32_t pool_size,
        float eps,
        bool cache_f16) {
    const uint32_t tid = threadIdx.x;
    if (head_dim == 0u || pool_size == 0u || tid >= head_dim ||
        n_tokens == 0u) {
        return;
    }
    float *const tail_k = tail;
    float *const tail_gate = tail + (uint64_t)pool_size * head_dim;
    extern __shared__ float shared[];
    float *rows = shared;
    float *mean = rows + (uint64_t)pool_size * head_dim;
    float *inv = mean + pool_size;
    const uint32_t pool = pos0 / pool_size + blockIdx.x;
    const uint32_t pool_start = pool * pool_size;
    const uint32_t input_end = pos0 + n_tokens;
    if (pool_start >= input_end || pool_start + pool_size <= pos0) return;
    const bool complete = pool_start + pool_size <= input_end;

    for (uint32_t r = 0; r < pool_size; r++) {
        const uint32_t pos = pool_start + r;
        float k_value = 0.0f;
        float gate_value = 0.0f;
        if (pos >= pos0 && pos < input_end) {
            const uint32_t src_row = pos - pos0;
            k_value = raw_k[(uint64_t)src_row * head_dim + tid];
            gate_value = gate[(uint64_t)src_row * head_dim + tid];
            if (!complete) {
                tail_k[(uint64_t)r * head_dim + tid] = k_value;
                tail_gate[(uint64_t)r * head_dim + tid] = gate_value;
            }
        } else {
            k_value = tail_k[(uint64_t)r * head_dim + tid];
            gate_value = tail_gate[(uint64_t)r * head_dim + tid];
        }
        rows[(uint64_t)r * head_dim + tid] = k_value;
    }
    __syncthreads();
    if (!complete || pool >= (cache_cap + pool_size - 1u) / pool_size) {
        return;
    }

    if (tid < pool_size) {
        const uint32_t r = tid;
        float sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            sum += rows[(uint64_t)r * head_dim + d];
        }
        const float m = sum / (float)head_dim;
        float ss = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            const float delta = rows[(uint64_t)r * head_dim + d] - m;
            ss = fmaf(delta, delta, ss);
        }
        mean[r] = m;
        inv[r] = rsqrtf(ss / (float)head_dim + eps);
    }
    __syncthreads();

    float max_logit = -INFINITY;
    float logits[4];
    for (uint32_t r = 0; r < pool_size; r++) {
        const uint32_t pos = pool_start + r;
        const float gate_value = pos >= pos0
            ? gate[(uint64_t)(pos - pos0) * head_dim + tid]
            : tail_gate[(uint64_t)r * head_dim + tid];
        const float ape_value =
            __uint_as_float((uint32_t)ape[(uint64_t)r * head_dim + tid] << 16);
        logits[r] = gate_value + ape_value;
        max_logit = fmaxf(max_logit, logits[r]);
    }
    float denom = 0.0f;
    for (uint32_t r = 0; r < pool_size; r++) {
        logits[r] = expf(logits[r] - max_logit);
        denom += logits[r];
    }
    float pooled = 0.0f;
    for (uint32_t r = 0; r < pool_size; r++) {
        const float normalized =
            (rows[(uint64_t)r * head_dim + tid] - mean[r]) * inv[r] *
            norm_weight[tid] + norm_bias[tid];
        pooled = fmaf(logits[r] / denom, normalized, pooled);
    }
    glm_rocm_cache_store(pool_cache,
        (uint64_t)pool * head_dim + tid, cache_f16, pooled);
}

extern "C" int ds4_gpu_glm53_indexer_pool_update_tensor(
        ds4_gpu_tensor       *pool_cache,
        ds4_gpu_tensor       *tail_k,
        ds4_gpu_tensor       *tail_gate,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              norm_weight_offset,
        uint64_t              norm_bias_offset,
        uint64_t              ape_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              pool_size,
        float                 eps,
        bool                  cache_f16) {
    if (!pool_cache || !tail_k || !tail_gate || !raw_k || !gate ||
        !model_map || n_tokens == 0u || cache_cap == 0u || head_dim == 0u ||
        head_dim > 1024u || pool_size != 4u || !isfinite(eps) || eps <= 0.0f ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        fprintf(stderr,
                "ds4: GLM-5.3 indexer pool received invalid arguments "
                "(pos=%u tokens=%u cap=%u dim=%u pool=%u)\n",
                pos0, n_tokens, cache_cap, head_dim, pool_size);
        return 0;
    }
    const uint64_t pool_cap =
        ((uint64_t)cache_cap + pool_size - 1u) / pool_size;
    const uint64_t cache_elem_size = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!glm53_rocm_tensor_has(pool_cache, pool_cap * head_dim,
                               cache_elem_size) ||
        !glm53_rocm_tensor_has(tail_k,
            2u * (uint64_t)pool_size * head_dim, sizeof(float)) ||
        !glm53_rocm_tensor_has(tail_gate,
            (uint64_t)pool_size * head_dim, sizeof(float)) ||
        (char *)tail_gate->ptr !=
            (char *)tail_k->ptr +
                (uint64_t)pool_size * head_dim * sizeof(float) ||
        !glm53_rocm_tensor_has(raw_k,
            (uint64_t)n_tokens * head_dim, sizeof(float)) ||
        !glm53_rocm_tensor_has(gate,
            (uint64_t)n_tokens * head_dim, sizeof(float))) {
        fprintf(stderr,
                "ds4: GLM-5.3 indexer pool received invalid buffers "
                "(cache=%llu tail_k=%llu tail_gate=%llu raw=%llu gate=%llu)\n",
                (unsigned long long)pool_cache->bytes,
                (unsigned long long)tail_k->bytes,
                (unsigned long long)tail_gate->bytes,
                (unsigned long long)raw_k->bytes,
                (unsigned long long)gate->bytes);
        return 0;
    }
    const float *norm_weight = glm53_rocm_weight_f32(
        model_map, model_size, norm_weight_offset, head_dim,
        "indexer pool norm weight");
    const float *norm_bias = glm53_rocm_weight_f32(
        model_map, model_size, norm_bias_offset, head_dim,
        "indexer pool norm bias");
    const uint64_t ape_bytes =
        (uint64_t)pool_size * head_dim * sizeof(uint16_t);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
        fprintf(stderr, "ds4: GLM-5.3 indexer pool APE range is invalid\n");
        return 0;
    }
    const uint16_t *ape = (const uint16_t *)cuda_model_range_ptr(
        model_map, ape_offset, ape_bytes, "indexer pool APE");
    if (!norm_weight || !norm_bias || !ape) {
        fprintf(stderr,
                "ds4: GLM-5.3 indexer pool could not resolve model weights "
                "(norm=%d bias=%d ape=%d)\n",
                norm_weight != NULL, norm_bias != NULL, ape != NULL);
        return 0;
    }

    const uint64_t shared_bytes =
        ((uint64_t)pool_size * head_dim + 2u * pool_size) * sizeof(float);
    uint32_t done = 0u;
    const uint32_t leading = pos0 % pool_size;
    if (leading != 0u) {
        uint32_t chunk = pool_size - leading;
        if (chunk > n_tokens) chunk = n_tokens;
        glm53_rocm_indexer_pool_update_kernel<<<
            1u, head_dim, shared_bytes>>>(
                (char *)pool_cache->ptr, (float *)tail_k->ptr,
                (const float *)raw_k->ptr, (const float *)gate->ptr,
                norm_weight, norm_bias, ape, pos0, chunk, cache_cap,
                head_dim, pool_size, eps, cache_f16);
        if (!cuda_ok(cudaGetLastError(),
                     "GLM-5.3 leading indexer pool update launch")) {
            return 0;
        }
        done = chunk;
    }
    const uint32_t remaining = n_tokens - done;
    const uint32_t full_tokens = remaining - remaining % pool_size;
    if (full_tokens != 0u) {
        const uint32_t groups = full_tokens / pool_size;
        glm53_rocm_indexer_pool_update_kernel<<<
            groups, head_dim, shared_bytes>>>(
                (char *)pool_cache->ptr, (float *)tail_k->ptr,
                (const float *)raw_k->ptr + (uint64_t)done * head_dim,
                (const float *)gate->ptr + (uint64_t)done * head_dim,
                norm_weight, norm_bias, ape, pos0 + done, full_tokens,
                cache_cap, head_dim, pool_size, eps, cache_f16);
        if (!cuda_ok(cudaGetLastError(),
                     "GLM-5.3 full indexer pool update launch")) {
            return 0;
        }
        done += full_tokens;
    }
    if (done < n_tokens) {
        const uint32_t tail_tokens = n_tokens - done;
        glm53_rocm_indexer_pool_update_kernel<<<
            1u, head_dim, shared_bytes>>>(
                (char *)pool_cache->ptr, (float *)tail_k->ptr,
                (const float *)raw_k->ptr + (uint64_t)done * head_dim,
                (const float *)gate->ptr + (uint64_t)done * head_dim,
                norm_weight, norm_bias, ape, pos0 + done, tail_tokens,
                cache_cap, head_dim, pool_size, eps, cache_f16);
        if (!cuda_ok(cudaGetLastError(),
                     "GLM-5.3 trailing indexer pool update launch")) {
            return 0;
        }
    }
    return 1;
}

__global__ static void glm53_rocm_expand_pool_selection_kernel(
        uint32_t *raw_selected,
        const uint32_t *pool_selected,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t selected_pools,
        uint32_t index_topk,
        uint32_t pool_size,
        uint32_t output_width) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * output_width;
    if (gid >= total || output_width == 0u || pool_size == 0u) return;
    const uint32_t token = (uint32_t)(gid / output_width);
    const uint32_t slot = (uint32_t)(gid - (uint64_t)token * output_width);
    uint32_t value = UINT32_MAX;
    if (slot < index_topk) {
        const uint32_t pool_slot = slot / pool_size;
        if (pool_slot < selected_pools) {
            const uint32_t pool = pool_selected[
                (uint64_t)token * selected_pools + pool_slot];
            value = pool * pool_size + slot % pool_size;
        }
    } else {
        const uint32_t tail_slot = slot - index_topk;
        const uint32_t visible = pos0 + token + 1u;
        const uint32_t tail_count = visible % pool_size;
        if (tail_slot < tail_count) {
            value = visible - tail_count + tail_slot;
        }
    }
    raw_selected[gid] = value;
}

extern "C" int ds4_gpu_glm53_expand_pool_selection_tensor(
        ds4_gpu_tensor       *raw_selected,
        const ds4_gpu_tensor *pool_selected,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              selected_pools,
        uint32_t              index_topk,
        uint32_t              pool_size,
        uint32_t              output_width) {
    if (!raw_selected || !pool_selected || n_tokens == 0u ||
        selected_pools == 0u || index_topk == 0u || pool_size == 0u ||
        selected_pools > index_topk / pool_size ||
        output_width < index_topk + pool_size - 1u ||
        !glm53_rocm_tensor_has(raw_selected,
            (uint64_t)n_tokens * output_width, sizeof(uint32_t)) ||
        !glm53_rocm_tensor_has(pool_selected,
            (uint64_t)n_tokens * selected_pools, sizeof(uint32_t))) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_tokens * output_width;
    glm53_rocm_expand_pool_selection_kernel<<<
        (unsigned)((total + 255u) / 256u), 256u>>>(
            (uint32_t *)raw_selected->ptr,
            (const uint32_t *)pool_selected->ptr, n_tokens, pos0,
            selected_pools, index_topk, pool_size, output_width);
    return cuda_ok(cudaGetLastError(),
                   "GLM-5.3 pool selection expansion launch");
}

extern "C" int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim,
        float scale,
        bool cache_f16) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!scores || !q || !weights || !indexer_key_cache ||
        n_rows == 0u || n_head == 0u || head_dim == 0u ||
        !cuda_tensor_has_f32(scores, n_rows) ||
        !cuda_tensor_has_elems2(q, n_head, head_dim, sizeof(float)) ||
        !cuda_tensor_has_f32(weights, n_head) ||
        !glm_rocm_tensor_has_cache2(indexer_key_cache, n_rows, head_dim, elem)) {
        return 0;
    }
    glm_indexer_score_one_kernel<<<n_rows, 256>>>((float *)scores->ptr,
                                                  (const float *)q->ptr,
                                                  (const float *)weights->ptr,
                                                  (const char *)indexer_key_cache->ptr,
                                                  n_rows,
                                                  n_head,
                                                  head_dim,
                                                  scale,
                                                  cache_f16);
    return cuda_ok(cudaGetLastError(), "glm indexer score one launch");
}

extern "C" int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        float scale,
        bool cache_f16) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    uint32_t end_pos = 0;
    if (!scores || !q || !weights || !indexer_key_cache || n_tokens == 0u ||
        n_rows == 0u || n_head == 0u || head_dim == 0u ||
        !glm_rocm_check_token_span(pos0, n_tokens, &end_pos) ||
        end_pos > n_rows ||
        !cuda_tensor_has_elems2(scores, n_tokens, n_rows, sizeof(float)) ||
        !cuda_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, n_head, sizeof(float)) ||
        !glm_rocm_tensor_has_cache2(indexer_key_cache, n_rows, head_dim, elem)) {
        return 0;
    }
    dim3 grid(n_rows, n_tokens, 1);
    glm_indexer_scores_batch_kernel<<<grid, 256>>>((float *)scores->ptr,
                                                   (const float *)q->ptr,
                                                   (const float *)weights->ptr,
                                                   (const char *)indexer_key_cache->ptr,
                                                   n_rows,
                                                   n_tokens,
                                                   pos0,
                                                   1u,
                                                   n_head,
                                                   head_dim,
                                                   scale,
                                                   cache_f16);
    return cuda_ok(cudaGetLastError(), "glm indexer scores batch launch");
}

extern "C" int ds4_gpu_glm53_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              pool_size,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    uint32_t end_pos = 0;
    if (!scores || !q || !weights || !indexer_key_cache || n_tokens == 0u ||
        n_rows == 0u || pool_size != 4u || n_head == 0u ||
        head_dim != 128u || !isfinite(scale) || scale <= 0.0f ||
        !glm_rocm_check_token_span(pos0, n_tokens, &end_pos) ||
        end_pos / pool_size > n_rows ||
        !cuda_tensor_has_elems2(scores, n_tokens, n_rows, sizeof(float)) ||
        !cuda_tensor_has_elems3(q, n_tokens, n_head, head_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, n_head, sizeof(float)) ||
        !glm_rocm_tensor_has_cache2(indexer_key_cache, n_rows, head_dim, elem)) {
        return 0;
    }
    const dim3 grid(n_rows, n_tokens, 1u);
    glm_indexer_scores_batch_kernel<<<grid, 256u>>>(
        (float *)scores->ptr, (const float *)q->ptr,
        (const float *)weights->ptr,
        (const char *)indexer_key_cache->ptr,
        n_rows, n_tokens, pos0, pool_size, n_head, head_dim,
        scale, cache_f16);
    return cuda_ok(cudaGetLastError(),
                   "GLM-5.3 grouped indexer scores launch");
}

extern "C" int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *q,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_dim) {
    const unsigned char *w = NULL;
    uint32_t row_bytes = 0;
    uint64_t x_stride64 = 0;
    if (!qk_low || !q || !model_map ||
        n_head == 0u || kv_lora_dim == 0u || qk_nope == 0u || qk_dim == 0u ||
        qk_nope > qk_dim ||
        qk_nope > DS4_ROCM_ATTENTION_SCORE_CAP ||
        !cuda_u64_mul_checked(n_head, qk_dim, &x_stride64) ||
        x_stride64 > UINT32_MAX ||
        !cuda_tensor_has_elems2(q, n_head, qk_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(qk_low, n_head, kv_lora_dim, sizeof(float)) ||
        !glm_rocm_check_q8_rows(model_map, model_size, weight_offset,
                                (uint64_t)n_head * kv_lora_dim, qk_nope,
                                "glm_qk_lowrank", &w, &row_bytes)) {
        return 0;
    }
    const char *wave_decode_env =
        getenv("DS4_ROCM_GLM_QK_LOW_WAVE_DECODE");
    if (wave_decode_env == NULL || cuda_env_present(wave_decode_env)) {
        const uint32_t threads = 256u;
        const uint32_t waves_per_block = threads / 32u;
        const dim3 grid((kv_lora_dim + waves_per_block - 1u) /
                            waves_per_block,
                        n_head,
                        1u);
        glm_q8_project_head_wave_kernel<<<grid, threads>>>(
                (float *)qk_low->ptr,
                w,
                (const float *)q->ptr,
                n_head,
                qk_nope,
                kv_lora_dim,
                (uint32_t)x_stride64,
                qk_dim,
                row_bytes);
        return cuda_ok(cudaGetLastError(),
                       "glm wave-parallel qk lowrank launch");
    }
    glm_q8_project_head_kernel<<<dim3(n_head, 1, 1), 256, (size_t)qk_nope * sizeof(float)>>>(
            (float *)qk_low->ptr,
            w,
            (const float *)q->ptr,
            1,
            n_head,
            qk_nope,
            kv_lora_dim,
            (uint32_t)x_stride64,
            qk_dim,
            row_bytes);
    return cuda_ok(cudaGetLastError(), "glm qk lowrank launch");
}

extern "C" int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *q,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_dim) {
    const unsigned char *w = NULL;
    uint32_t row_bytes = 0;
    uint64_t x_stride64 = 0;
    if (!qk_low || !q || !model_map || n_tokens == 0u ||
        n_head == 0u || kv_lora_dim == 0u || qk_nope == 0u || qk_dim == 0u ||
        qk_nope > qk_dim ||
        qk_nope > DS4_ROCM_ATTENTION_SCORE_CAP ||
        !cuda_u64_mul_checked(n_head, qk_dim, &x_stride64) ||
        x_stride64 > UINT32_MAX ||
        !cuda_tensor_has_elems3(q, n_tokens, n_head, qk_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(qk_low, n_tokens, n_head, kv_lora_dim, sizeof(float)) ||
        !glm_rocm_check_q8_rows(model_map, model_size, weight_offset,
                                (uint64_t)n_head * kv_lora_dim, qk_nope,
                                "glm_qk_lowrank_batch", &w, &row_bytes)) {
        return 0;
    }
    if (n_tokens > 1u &&
        (qk_nope & 31u) == 0u &&
        cuda_runtime_config()->glm_grouped_qk_low) {
        const uint32_t rows_per_block = 32u;
        const uint32_t token_tile = 32u;
        const uint32_t block_tile = 8u;
        cuda_launch_grouped_q8_a_sharedx_strided(
                (float *)qk_low->ptr,
                w,
                (const float *)q->ptr,
                n_tokens,
                n_head,
                qk_nope >> 5u,
                kv_lora_dim,
                (uint32_t)x_stride64,
                qk_dim,
                row_bytes,
                rows_per_block,
                token_tile,
                block_tile);
        return cuda_ok(cudaGetLastError(),
                       "glm grouped qk lowrank batch launch");
    }
    dim3 grid(n_head, n_tokens, 1);
    glm_q8_project_head_kernel<<<grid, 256, (size_t)qk_nope * sizeof(float)>>>(
            (float *)qk_low->ptr,
            w,
            (const float *)q->ptr,
            n_tokens,
            n_head,
            qk_nope,
            kv_lora_dim,
            (uint32_t)x_stride64,
            qk_dim,
            row_bytes);
    return cuda_ok(cudaGetLastError(), "glm qk lowrank batch launch");
}

extern "C" int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor *heads,
        const ds4_gpu_tensor *lora,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim) {
    const unsigned char *w = NULL;
    uint32_t row_bytes = 0;
    uint64_t x_stride64 = 0;
    if (!heads || !lora || !model_map || n_tokens == 0u ||
        n_head == 0u || kv_lora_dim == 0u || value_dim == 0u ||
        kv_lora_dim > DS4_ROCM_ATTENTION_SCORE_CAP ||
        !cuda_u64_mul_checked(n_head, kv_lora_dim, &x_stride64) ||
        x_stride64 > UINT32_MAX ||
        !cuda_tensor_has_elems3(lora, n_tokens, n_head, kv_lora_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(heads, n_tokens, n_head, value_dim, sizeof(float)) ||
        !glm_rocm_check_q8_rows(model_map, model_size, weight_offset,
                                (uint64_t)n_head * value_dim, kv_lora_dim,
                                "glm_value_project", &w, &row_bytes)) {
        return 0;
    }
    if (n_tokens > 1u &&
        (kv_lora_dim & 31u) == 0u &&
        cuda_runtime_config()->glm_grouped_value_project) {
        const uint32_t rows_per_block = 32u;
        const uint32_t token_tile = 32u;
        const uint32_t block_tile = 16u;
        cuda_launch_grouped_q8_a_sharedx(
                (float *)heads->ptr,
                w,
                (const float *)lora->ptr,
                n_tokens,
                n_head,
                kv_lora_dim >> 5u,
                value_dim,
                row_bytes,
                rows_per_block,
                token_tile,
                block_tile);
        return cuda_ok(cudaGetLastError(),
                       "glm grouped value project batch heads launch");
    }
    const char *wave_decode_env =
        getenv("DS4_ROCM_GLM_VALUE_PROJECT_WAVE_DECODE");
    /*
     * The wave-per-output-row kernel is the validated production decode path.
     * Keep an explicit =0 fallback for correctness diagnostics.
     */
    if (n_tokens == 1u &&
        (wave_decode_env == NULL || cuda_env_present(wave_decode_env))) {
        const uint32_t threads = 256u;
        const uint32_t waves_per_block = threads / 32u;
        dim3 grid((value_dim + waves_per_block - 1u) / waves_per_block,
                  n_head,
                  n_tokens);
        glm_q8_project_head_wave_kernel<<<grid, threads>>>(
                (float *)heads->ptr,
                w,
                (const float *)lora->ptr,
                n_head,
                kv_lora_dim,
                value_dim,
                (uint32_t)x_stride64,
                kv_lora_dim,
                row_bytes);
        const cudaError_t launch_err = cudaGetLastError();
        if (launch_err == cudaSuccess) {
            static int notice_printed = 0;
            if (!notice_printed) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX
                        "GLM one-token value projection using "
                        "wave-parallel Q8 rows\n");
                notice_printed = 1;
            }
            return 1;
        }
        return cuda_ok(launch_err,
                       "glm wave-parallel value project launch");
    }
    dim3 grid(n_head, n_tokens, 1);
    glm_q8_project_head_kernel<<<grid, 256, (size_t)kv_lora_dim * sizeof(float)>>>(
            (float *)heads->ptr,
            w,
            (const float *)lora->ptr,
            n_tokens,
            n_head,
            kv_lora_dim,
            value_dim,
            (uint32_t)x_stride64,
            kv_lora_dim,
            row_bytes);
    return cuda_ok(cudaGetLastError(), "glm value project batch heads launch");
}

__global__ static void glm_causal_gemm_cache_to_f16_kernel(
        __half *dst,
        const char *src,
        uint64_t n,
        bool src_f16) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(glm_rocm_cache_load(src, i, src_f16));
}

__global__ static void glm_causal_gemm_rope_to_f16_kernel(
        __half *dst,
        const char *src,
        uint32_t n_rows,
        uint32_t qk_rope,
        bool src_f16,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint64_t pair = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t pairs = (uint64_t)n_rows * (qk_rope >> 1u);
    if (pair >= pairs) return;
    const uint32_t row = (uint32_t)(pair / (qk_rope >> 1u));
    const uint32_t r =
        (uint32_t)(pair - (uint64_t)row * (qk_rope >> 1u)) << 1u;
    const float2 y = glm_rocm_rotated_cache_rope_pair(src,
                                                       (uint64_t)row * qk_rope,
                                                       r,
                                                       row,
                                                       qk_rope,
                                                       src_f16,
                                                       n_ctx_orig,
                                                       freq_base,
                                                       freq_scale,
                                                       ext_factor,
                                                       attn_factor,
                                                       beta_fast,
                                                       beta_slow);
    dst[(uint64_t)row * qk_rope + r] = __float2half(y.x);
    dst[(uint64_t)row * qk_rope + r + 1u] = __float2half(y.y);
}

__global__ static void glm_causal_gemm_gather_head_f16_kernel(
        __half *low_h,
        __half *qrope_h,
        const float *low,
        const float *q,
        uint32_t n_tokens,
        uint32_t head,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope) {
    const uint64_t total = (uint64_t)n_tokens * kv_lora_dim;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t token = (uint32_t)(i / kv_lora_dim);
        const uint32_t j =
            (uint32_t)(i - (uint64_t)token * kv_lora_dim);
        low_h[i] = __float2half(
            low[((uint64_t)token * n_head + head) * kv_lora_dim + j]);
    }
    const uint64_t rope_total = (uint64_t)n_tokens * qk_rope;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < rope_total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t token = (uint32_t)(i / qk_rope);
        const uint32_t r = (uint32_t)(i - (uint64_t)token * qk_rope);
        const uint32_t qk_dim = qk_nope + qk_rope;
        qrope_h[i] = __float2half(
            q[((uint64_t)token * n_head + head) * qk_dim + qk_nope + r]);
    }
}

__global__ static void glm_causal_gemm_softmax_f16_kernel(
        __half *probs,
        float *scores,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t pos0,
        float scale) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    float *row = scores + (uint64_t)token * n_selected;
    __shared__ float reduce[256];
    const uint32_t visible = min(n_selected, pos0 + token + 1u);
    float vmax = -INFINITY;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        const float v = s < visible ? row[s] * scale : -INFINITY;
        row[s] = v;
        vmax = fmaxf(vmax, v);
    }
    reduce[threadIdx.x] = vmax;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] =
                fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    float sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        const float v = s < visible ? expf(row[s] - max_score) : 0.0f;
        row[s] = v;
        sum += v;
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = 1.0f / fmaxf(reduce[0], 1.0e-20f);
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        probs[(uint64_t)token * n_selected + s] =
            __float2half(row[s] * inv);
    }
}

__global__ static void glm_causal_gemm_scatter_head_kernel(
        float *out,
        const float *head_out,
        uint32_t n_tokens,
        uint32_t head,
        uint32_t n_head,
        uint32_t kv_lora_dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * kv_lora_dim;
    if (i >= n) return;
    const uint32_t token = (uint32_t)(i / kv_lora_dim);
    const uint32_t j = (uint32_t)(i - (uint64_t)token * kv_lora_dim);
    out[((uint64_t)token * n_head + head) * kv_lora_dim + j] =
        head_out[i];
}

__global__ static void glm_selected_gemm_kv_to_f16_kernel(
        __half *dst,
        const char *src,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t kv_lora_dim,
        uint32_t cache_cap,
        bool src_f16) {
    const uint64_t total =
        (uint64_t)n_tokens * n_selected * kv_lora_dim;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t selected_row = i / kv_lora_dim;
        const uint32_t j =
            (uint32_t)(i - selected_row * kv_lora_dim);
        const int32_t cache_row = selected[selected_row];
        dst[i] =
            cache_row >= 0 && (uint32_t)cache_row < cache_cap ?
                __float2half(glm_rocm_cache_load(
                    src,
                    (uint64_t)(uint32_t)cache_row * kv_lora_dim + j,
                    src_f16)) :
                __float2half(0.0f);
    }
}

__global__ static void glm_selected_gemm_rope_to_f16_kernel(
        __half *dst,
        const char *src,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t qk_rope,
        uint32_t cache_cap,
        bool src_f16,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t rope_pairs = qk_rope >> 1u;
    const uint64_t total =
        (uint64_t)n_tokens * n_selected * rope_pairs;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t selected_row = i / rope_pairs;
        const uint32_t r =
            (uint32_t)(i - selected_row * rope_pairs) << 1u;
        const int32_t cache_row = selected[selected_row];
        float2 y = make_float2(0.0f, 0.0f);
        if (cache_row >= 0 && (uint32_t)cache_row < cache_cap) {
            y = glm_rocm_rotated_cache_rope_pair(
                    src,
                    (uint64_t)(uint32_t)cache_row * qk_rope,
                    r,
                    (uint32_t)cache_row,
                    qk_rope,
                    src_f16,
                    n_ctx_orig,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    beta_fast,
                    beta_slow);
        }
        const uint64_t out = selected_row * qk_rope + r;
        dst[out] = __float2half(y.x);
        dst[out + 1u] = __float2half(y.y);
    }
}

__global__ static void glm_selected_gemm_softmax_f16_kernel(
        __half *probs,
        float *scores,
        uint32_t n_tokens,
        uint32_t n_selected,
        float scale) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    float *row = scores + (uint64_t)token * n_selected;
    __shared__ float reduce[256];
    float vmax = -INFINITY;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        const float v = row[s] * scale;
        row[s] = v;
        vmax = fmaxf(vmax, v);
    }
    reduce[threadIdx.x] = vmax;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] =
                fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    float sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        const float v = expf(row[s] - max_score);
        row[s] = v;
        sum += v;
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = 1.0f / fmaxf(reduce[0], 1.0e-20f);
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        probs[(uint64_t)token * n_selected + s] =
            __float2half(row[s] * inv);
    }
}

__global__ static void glm_selected_gemm_gather_heads_f16_kernel(
        __half *low_h,
        __half *qrope_h,
        const float *low,
        const float *q,
        uint32_t n_tokens,
        uint32_t head0,
        uint32_t head_count,
        uint32_t head_capacity,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope) {
    const uint64_t low_per_token =
        (uint64_t)head_capacity * kv_lora_dim;
    const uint64_t low_total = (uint64_t)n_tokens * low_per_token;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < low_total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t token = (uint32_t)(i / low_per_token);
        const uint64_t token_i = i - (uint64_t)token * low_per_token;
        const uint32_t head_local = (uint32_t)(token_i / kv_lora_dim);
        const uint32_t j =
            (uint32_t)(token_i - (uint64_t)head_local * kv_lora_dim);
        if (head_local < head_count) {
            low_h[i] = __float2half(
                low[((uint64_t)token * n_head + head0 + head_local) *
                    kv_lora_dim + j]);
        }
    }

    const uint64_t rope_per_token =
        (uint64_t)head_capacity * qk_rope;
    const uint64_t rope_total = (uint64_t)n_tokens * rope_per_token;
    const uint32_t qk_dim = qk_nope + qk_rope;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < rope_total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t token = (uint32_t)(i / rope_per_token);
        const uint64_t token_i = i - (uint64_t)token * rope_per_token;
        const uint32_t head_local = (uint32_t)(token_i / qk_rope);
        const uint32_t r =
            (uint32_t)(token_i - (uint64_t)head_local * qk_rope);
        if (head_local < head_count) {
            qrope_h[i] = __float2half(
                q[((uint64_t)token * n_head + head0 + head_local) *
                  qk_dim + qk_nope + r]);
        }
    }
}

__global__ static void glm_selected_gemm_softmax_heads_f16_kernel(
        __half *probs,
        float *scores,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t head_count,
        uint32_t head_capacity,
        float scale) {
    const uint32_t token = blockIdx.x;
    const uint32_t head_local = blockIdx.y;
    if (token >= n_tokens || head_local >= head_count) return;
    const uint64_t row_index =
        (uint64_t)token * head_capacity + head_local;
    float *row = scores + row_index * n_selected;
    __shared__ float reduce[256];
    float vmax = -INFINITY;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        const float v = row[s] * scale;
        row[s] = v;
        vmax = fmaxf(vmax, v);
    }
    reduce[threadIdx.x] = vmax;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] =
                fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    float sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        const float v = expf(row[s] - max_score);
        row[s] = v;
        sum += v;
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = 1.0f / fmaxf(reduce[0], 1.0e-20f);
    __half *prob_row = probs + row_index * n_selected;
    for (uint32_t s = threadIdx.x; s < n_selected; s += blockDim.x) {
        prob_row[s] = __float2half(row[s] * inv);
    }
}

__global__ static void glm_selected_gemm_scatter_heads_kernel(
        float *out,
        const float *head_out,
        uint32_t n_tokens,
        uint32_t head0,
        uint32_t head_count,
        uint32_t head_capacity,
        uint32_t n_head,
        uint32_t kv_lora_dim) {
    const uint64_t n =
        (uint64_t)n_tokens * head_count * kv_lora_dim;
    const uint64_t i =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t per_token = (uint64_t)head_count * kv_lora_dim;
    const uint32_t token = (uint32_t)(i / per_token);
    const uint64_t token_i = i - (uint64_t)token * per_token;
    const uint32_t head_local = (uint32_t)(token_i / kv_lora_dim);
    const uint32_t j =
        (uint32_t)(token_i - (uint64_t)head_local * kv_lora_dim);
    out[((uint64_t)token * n_head + head0 + head_local) *
        kv_lora_dim + j] =
        head_out[((uint64_t)token * head_capacity + head_local) *
                 kv_lora_dim + j];
}

static int glm_causal_gemm_scratch_part(
        uint64_t *offset,
        uint64_t bytes,
        uint64_t *part_offset) {
    uint64_t aligned = 0;
    if (!offset || !part_offset ||
        !cuda_u64_add_checked(*offset, 255u, &aligned)) {
        return 0;
    }
    aligned &= ~255ull;
    if (!cuda_u64_add_checked(aligned, bytes, offset)) return 0;
    *part_offset = aligned;
    return 1;
}

enum glm_selected_gemm_profile_phase {
    GLM_SELECTED_PROFILE_KV_GATHER = 0,
    GLM_SELECTED_PROFILE_ROPE_GATHER,
    GLM_SELECTED_PROFILE_QUERY_GATHER,
    GLM_SELECTED_PROFILE_LORA_SCORE,
    GLM_SELECTED_PROFILE_ROPE_SCORE,
    GLM_SELECTED_PROFILE_SOFTMAX,
    GLM_SELECTED_PROFILE_VALUE,
    GLM_SELECTED_PROFILE_SCATTER,
    GLM_SELECTED_PROFILE_PHASE_COUNT
};

struct glm_selected_gemm_profile {
    int enabled;
    float ms[GLM_SELECTED_PROFILE_PHASE_COUNT];
};

static cudaEvent_t g_glm_selected_profile_begin;
static cudaEvent_t g_glm_selected_profile_end;
static int g_glm_selected_profile_events_ready = 0;
static int g_glm_selected_profile_claimed = 0;

static uint32_t glm_selected_gemm_head_tile(void) {
    static int initialized = 0;
    static uint32_t tile = 16u;
    if (initialized) return tile;
    initialized = 1;
    const char *env = getenv("DS4_ROCM_GLM_SELECTED_ATTN_HEAD_TILE");
    if (!env || !env[0]) return tile;
    char *end = NULL;
    errno = 0;
    const unsigned long value = strtoul(env, &end, 10);
    if (errno == 0 && end && *end == '\0' &&
        (value == 1ul || value == 2ul ||
         value == 4ul || value == 8ul ||
         value == 16ul || value == 32ul ||
         value == 64ul)) {
        tile = (uint32_t)value;
        return tile;
    }
    fprintf(stderr,
            "ds4: invalid DS4_ROCM_GLM_SELECTED_ATTN_HEAD_TILE='%s'; "
            "using 16 (valid values: 1,2,4,8,16,32,64)\n",
            env);
    return tile;
}

static void glm_selected_gemm_profile_init(
        glm_selected_gemm_profile *profile) {
    if (!profile) return;
    memset(profile, 0, sizeof(*profile));
    if (!cuda_env_present(
            getenv("DS4_ROCM_GLM_SELECTED_ATTN_PROFILE")) ||
        g_glm_selected_profile_claimed) {
        return;
    }
    g_glm_selected_profile_claimed = 1;
    if (!g_glm_selected_profile_events_ready) {
        if (cudaEventCreate(&g_glm_selected_profile_begin) != cudaSuccess) {
            return;
        }
        if (cudaEventCreate(&g_glm_selected_profile_end) != cudaSuccess) {
            (void)cudaEventDestroy(g_glm_selected_profile_begin);
            return;
        }
        g_glm_selected_profile_events_ready = 1;
    }
    profile->enabled = 1;
}

static void glm_selected_gemm_profile_begin(
        glm_selected_gemm_profile *profile) {
    if (!profile || !profile->enabled) return;
    if (cudaEventRecord(g_glm_selected_profile_begin, 0) != cudaSuccess) {
        profile->enabled = 0;
    }
}

static void glm_selected_gemm_profile_end(
        glm_selected_gemm_profile *profile,
        glm_selected_gemm_profile_phase phase) {
    if (!profile || !profile->enabled ||
        phase >= GLM_SELECTED_PROFILE_PHASE_COUNT) {
        return;
    }
    float elapsed = 0.0f;
    if (cudaEventRecord(g_glm_selected_profile_end, 0) != cudaSuccess ||
        cudaEventSynchronize(g_glm_selected_profile_end) != cudaSuccess ||
        cudaEventElapsedTime(&elapsed,
                             g_glm_selected_profile_begin,
                             g_glm_selected_profile_end) != cudaSuccess) {
        profile->enabled = 0;
        return;
    }
    profile->ms[phase] += elapsed;
}

static void glm_selected_gemm_profile_report(
        const glm_selected_gemm_profile *profile,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t n_head,
        uint32_t head_tile) {
    if (!profile || !profile->enabled) return;
    float total = 0.0f;
    for (uint32_t i = 0; i < GLM_SELECTED_PROFILE_PHASE_COUNT; i++) {
        total += profile->ms[i];
    }
    fprintf(stderr,
            "ds4: ROCm GLM selected attention profile "
            "tokens=%u selected=%u heads=%u tile=%u "
            "kv_gather=%.3fms rope_gather=%.3fms "
            "query_gather=%.3fms lora_score=%.3fms "
            "rope_score=%.3fms softmax=%.3fms "
            "value=%.3fms scatter=%.3fms total=%.3fms\n",
            n_tokens,
            n_selected,
            n_head,
            head_tile,
            profile->ms[GLM_SELECTED_PROFILE_KV_GATHER],
            profile->ms[GLM_SELECTED_PROFILE_ROPE_GATHER],
            profile->ms[GLM_SELECTED_PROFILE_QUERY_GATHER],
            profile->ms[GLM_SELECTED_PROFILE_LORA_SCORE],
            profile->ms[GLM_SELECTED_PROFILE_ROPE_SCORE],
            profile->ms[GLM_SELECTED_PROFILE_SOFTMAX],
            profile->ms[GLM_SELECTED_PROFILE_VALUE],
            profile->ms[GLM_SELECTED_PROFILE_SCATTER],
            total);
}

static int glm_attention_indexed_lora_causal_gemm(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        bool cache_f16,
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
        float beta_slow) {
    if (!g_cublas_ready || !lora_out || !q || !qk_low ||
        !kv_lora_cache || (qk_rope != 0u && !k_rope_cache) ||
        n_tokens < 256u || n_selected == 0u ||
        n_head == 0u || kv_lora_dim == 0u ||
        (qk_rope & 1u) != 0u ||
        n_tokens > INT_MAX || n_selected > INT_MAX ||
        kv_lora_dim > INT_MAX || qk_rope > INT_MAX) {
        return 0;
    }

    uint64_t kv_count = 0, rope_count = 0;
    uint64_t low_count = 0, qrope_count = 0;
    uint64_t score_count = 0, head_count = 0;
    if (!cuda_u64_mul_checked(n_selected, kv_lora_dim, &kv_count) ||
        !cuda_u64_mul_checked(n_selected, qk_rope, &rope_count) ||
        !cuda_u64_mul_checked(n_tokens, kv_lora_dim, &low_count) ||
        !cuda_u64_mul_checked(n_tokens, qk_rope, &qrope_count) ||
        !cuda_u64_mul_checked(n_tokens, n_selected, &score_count) ||
        !cuda_u64_mul_checked(n_tokens, kv_lora_dim, &head_count)) {
        return 0;
    }

    uint64_t kv_bytes = 0, rope_bytes = 0;
    uint64_t low_bytes = 0, qrope_bytes = 0;
    uint64_t score_bytes = 0, prob_bytes = 0, head_bytes = 0;
    if (!cuda_u64_mul_checked(kv_count, sizeof(__half), &kv_bytes) ||
        !cuda_u64_mul_checked(rope_count, sizeof(__half), &rope_bytes) ||
        !cuda_u64_mul_checked(low_count, sizeof(__half), &low_bytes) ||
        !cuda_u64_mul_checked(qrope_count, sizeof(__half), &qrope_bytes) ||
        !cuda_u64_mul_checked(score_count, sizeof(float), &score_bytes) ||
        !cuda_u64_mul_checked(score_count, sizeof(__half), &prob_bytes) ||
        !cuda_u64_mul_checked(head_count, sizeof(float), &head_bytes)) {
        return 0;
    }

    uint64_t scratch_bytes = 0;
    uint64_t kv_offset = 0, rope_offset = 0;
    uint64_t low_offset = 0, qrope_offset = 0;
    uint64_t score_offset = 0, prob_offset = 0, head_offset = 0;
    if (!glm_causal_gemm_scratch_part(&scratch_bytes, kv_bytes, &kv_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, rope_bytes, &rope_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, low_bytes, &low_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, qrope_bytes, &qrope_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, score_bytes, &score_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, prob_bytes, &prob_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, head_bytes, &head_offset)) {
        return 0;
    }

    char *scratch =
        (char *)cuda_tmp_alloc(scratch_bytes, "glm causal attention gemm");
    if (!scratch) return 0;
    __half *kv_h = (__half *)(scratch + kv_offset);
    __half *rope_h = (__half *)(scratch + rope_offset);
    __half *low_h = (__half *)(scratch + low_offset);
    __half *qrope_h = (__half *)(scratch + qrope_offset);
    float *scores = (float *)(scratch + score_offset);
    __half *probs = (__half *)(scratch + prob_offset);
    float *head_out = (float *)(scratch + head_offset);

    glm_causal_gemm_cache_to_f16_kernel<<<
        (kv_count + 255u) / 256u, 256>>>(
            kv_h,
            (const char *)kv_lora_cache->ptr,
            kv_count,
            cache_f16);
    if (!cuda_ok(cudaGetLastError(), "glm causal attention kv f16 launch")) {
        return 0;
    }
    if (qk_rope != 0u) {
        const uint64_t rope_pairs =
            (uint64_t)n_selected * (qk_rope >> 1u);
        glm_causal_gemm_rope_to_f16_kernel<<<
            (rope_pairs + 255u) / 256u, 256>>>(
                rope_h,
                (const char *)k_rope_cache->ptr,
                n_selected,
                qk_rope,
                cache_f16,
                n_ctx_orig,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow);
        if (!cuda_ok(cudaGetLastError(),
                     "glm causal attention rope f16 launch")) {
            return 0;
        }
    }

    const float one = 1.0f;
    const float zero = 0.0f;
    const float scale = 1.0f / sqrtf((float)(qk_nope + qk_rope));
    const uint32_t gather_blocks = (uint32_t)((low_count + 255u) / 256u);
    const uint32_t scatter_blocks =
        (uint32_t)((head_count + 255u) / 256u);
    for (uint32_t head = 0; head < n_head; head++) {
        glm_causal_gemm_gather_head_f16_kernel<<<gather_blocks, 256>>>(
            low_h,
            qrope_h,
            (const float *)qk_low->ptr,
            (const float *)q->ptr,
            n_tokens,
            head,
            n_head,
            kv_lora_dim,
            qk_nope,
            qk_rope);
        if (!cuda_ok(cudaGetLastError(),
                     "glm causal attention query gather launch")) {
            return 0;
        }
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)n_selected,
                                         (int)n_tokens,
                                         (int)kv_lora_dim,
                                         &one,
                                         kv_h,
                                         CUDA_R_16F,
                                         (int)kv_lora_dim,
                                         low_h,
                                         CUDA_R_16F,
                                         (int)kv_lora_dim,
                                         &zero,
                                         scores,
                                         CUDA_R_32F,
                                         (int)n_selected,
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "glm causal attention lora score gemm")) {
            return 0;
        }
        if (qk_rope != 0u) {
            st = cublasGemmEx(g_cublas,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              (int)n_selected,
                              (int)n_tokens,
                              (int)qk_rope,
                              &one,
                              rope_h,
                              CUDA_R_16F,
                              (int)qk_rope,
                              qrope_h,
                              CUDA_R_16F,
                              (int)qk_rope,
                              &one,
                              scores,
                              CUDA_R_32F,
                              (int)n_selected,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT);
            if (!cublas_ok(st, "glm causal attention rope score gemm")) {
                return 0;
            }
        }
        glm_causal_gemm_softmax_f16_kernel<<<n_tokens, 256>>>(
            probs, scores, n_tokens, n_selected, pos0, scale);
        if (!cuda_ok(cudaGetLastError(),
                     "glm causal attention softmax launch")) {
            return 0;
        }
        st = cublasGemmEx(g_cublas,
                          CUBLAS_OP_N,
                          CUBLAS_OP_N,
                          (int)kv_lora_dim,
                          (int)n_tokens,
                          (int)n_selected,
                          &one,
                          kv_h,
                          CUDA_R_16F,
                          (int)kv_lora_dim,
                          probs,
                          CUDA_R_16F,
                          (int)n_selected,
                          &zero,
                          head_out,
                          CUDA_R_32F,
                          (int)kv_lora_dim,
                          CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "glm causal attention value gemm")) return 0;
        glm_causal_gemm_scatter_head_kernel<<<scatter_blocks, 256>>>(
            (float *)lora_out->ptr,
            head_out,
            n_tokens,
            head,
            n_head,
            kv_lora_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "glm causal attention head scatter launch")) {
            return 0;
        }
    }
    return 1;
}

static int glm_attention_indexed_lora_selected_gemm(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
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
        float beta_slow) {
    if (!g_cublas_ready || !lora_out || !q || !qk_low ||
        !kv_lora_cache || !k_rope_cache || !selected ||
        n_tokens < 64u || n_tokens > 256u ||
        n_selected == 0u || n_selected > cache_cap ||
        n_head == 0u || kv_lora_dim == 0u ||
        qk_rope == 0u || (qk_rope & 1u) != 0u ||
        n_tokens > INT_MAX || n_selected > INT_MAX ||
        kv_lora_dim > INT_MAX || qk_rope > INT_MAX) {
        return 0;
    }

    const uint32_t requested_head_tile = glm_selected_gemm_head_tile();
    const uint32_t head_tile = min(requested_head_tile, n_head);
    if (head_tile > 1u) {
        static int notice_printed = 0;
        if (!notice_printed) {
            fprintf(stderr,
                    "ds4: ROCm GLM selected attention using head tile %u\n",
                    head_tile);
            notice_printed = 1;
        }
    }

    uint64_t selected_rows = 0;
    uint64_t kv_count = 0, rope_count = 0;
    uint64_t low_count = 0, qrope_count = 0;
    uint64_t score_count = 0, head_count = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_selected, &selected_rows) ||
        !cuda_u64_mul_checked(selected_rows, kv_lora_dim, &kv_count) ||
        !cuda_u64_mul_checked(selected_rows, qk_rope, &rope_count) ||
        !cuda_u64_mul_checked(n_tokens, kv_lora_dim, &low_count) ||
        !cuda_u64_mul_checked(n_tokens, qk_rope, &qrope_count) ||
        !cuda_u64_mul_checked(n_tokens, n_selected, &score_count) ||
        !cuda_u64_mul_checked(n_tokens, kv_lora_dim, &head_count)) {
        return 0;
    }
    if (!cuda_u64_mul_checked(low_count, head_tile, &low_count) ||
        !cuda_u64_mul_checked(qrope_count, head_tile, &qrope_count) ||
        !cuda_u64_mul_checked(score_count, head_tile, &score_count) ||
        !cuda_u64_mul_checked(head_count, head_tile, &head_count)) {
        return 0;
    }

    uint64_t kv_bytes = 0, rope_bytes = 0;
    uint64_t low_bytes = 0, qrope_bytes = 0;
    uint64_t score_bytes = 0, prob_bytes = 0, head_bytes = 0;
    if (!cuda_u64_mul_checked(kv_count, sizeof(__half), &kv_bytes) ||
        !cuda_u64_mul_checked(rope_count, sizeof(__half), &rope_bytes) ||
        !cuda_u64_mul_checked(low_count, sizeof(__half), &low_bytes) ||
        !cuda_u64_mul_checked(qrope_count, sizeof(__half), &qrope_bytes) ||
        !cuda_u64_mul_checked(score_count, sizeof(float), &score_bytes) ||
        !cuda_u64_mul_checked(score_count, sizeof(__half), &prob_bytes) ||
        !cuda_u64_mul_checked(head_count, sizeof(float), &head_bytes)) {
        return 0;
    }

    uint64_t scratch_bytes = 0;
    uint64_t kv_offset = 0, rope_offset = 0;
    uint64_t low_offset = 0, qrope_offset = 0;
    uint64_t score_offset = 0, prob_offset = 0, head_offset = 0;
    if (!glm_causal_gemm_scratch_part(&scratch_bytes, kv_bytes, &kv_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, rope_bytes, &rope_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, low_bytes, &low_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, qrope_bytes, &qrope_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, score_bytes, &score_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, prob_bytes, &prob_offset) ||
        !glm_causal_gemm_scratch_part(&scratch_bytes, head_bytes, &head_offset)) {
        return 0;
    }

    char *scratch =
        (char *)cuda_tmp_alloc(scratch_bytes, "glm selected attention gemm");
    if (!scratch) return 0;
    __half *kv_h = (__half *)(scratch + kv_offset);
    __half *rope_h = (__half *)(scratch + rope_offset);
    __half *low_h = (__half *)(scratch + low_offset);
    __half *qrope_h = (__half *)(scratch + qrope_offset);
    float *scores = (float *)(scratch + score_offset);
    __half *probs = (__half *)(scratch + prob_offset);
    float *head_out = (float *)(scratch + head_offset);

    glm_selected_gemm_profile profile;
    glm_selected_gemm_profile_init(&profile);

    const uint32_t kv_blocks =
        (uint32_t)min((kv_count + 255u) / 256u, 4096ull);
    glm_selected_gemm_profile_begin(&profile);
    glm_selected_gemm_kv_to_f16_kernel<<<kv_blocks, 256>>>(
        kv_h,
        (const char *)kv_lora_cache->ptr,
        (const int32_t *)selected->ptr,
        n_tokens,
        n_selected,
        kv_lora_dim,
        cache_cap,
        cache_f16);
    if (!cuda_ok(cudaGetLastError(),
                 "glm selected attention kv gather launch")) {
        return 0;
    }
    glm_selected_gemm_profile_end(
        &profile, GLM_SELECTED_PROFILE_KV_GATHER);
    const uint64_t rope_pairs = selected_rows * (qk_rope >> 1u);
    const uint32_t rope_blocks =
        (uint32_t)min((rope_pairs + 255u) / 256u, 4096ull);
    glm_selected_gemm_profile_begin(&profile);
    glm_selected_gemm_rope_to_f16_kernel<<<rope_blocks, 256>>>(
        rope_h,
        (const char *)k_rope_cache->ptr,
        (const int32_t *)selected->ptr,
        n_tokens,
        n_selected,
        qk_rope,
        cache_cap,
        cache_f16,
        n_ctx_orig,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow);
    if (!cuda_ok(cudaGetLastError(),
                 "glm selected attention rope gather launch")) {
        return 0;
    }
    glm_selected_gemm_profile_end(
        &profile, GLM_SELECTED_PROFILE_ROPE_GATHER);

    const float one = 1.0f;
    const float zero = 0.0f;
    const float scale = 1.0f / sqrtf((float)(qk_nope + qk_rope));
    const uint32_t gather_blocks =
        (uint32_t)min((low_count + 255u) / 256u, 4096ull);
    const long long kv_stride =
        (long long)n_selected * (long long)kv_lora_dim;
    const long long rope_stride =
        (long long)n_selected * (long long)qk_rope;
    const long long low_stride =
        (long long)head_tile * (long long)kv_lora_dim;
    const long long qrope_stride =
        (long long)head_tile * (long long)qk_rope;
    const long long score_stride =
        (long long)head_tile * (long long)n_selected;
    for (uint32_t head0 = 0; head0 < n_head; head0 += head_tile) {
        const uint32_t tile_heads = min(head_tile, n_head - head0);
        glm_selected_gemm_profile_begin(&profile);
        if (head_tile == 1u) {
            glm_causal_gemm_gather_head_f16_kernel<<<gather_blocks, 256>>>(
                low_h,
                qrope_h,
                (const float *)qk_low->ptr,
                (const float *)q->ptr,
                n_tokens,
                head0,
                n_head,
                kv_lora_dim,
                qk_nope,
                qk_rope);
        } else {
            glm_selected_gemm_gather_heads_f16_kernel<<<gather_blocks, 256>>>(
                low_h,
                qrope_h,
                (const float *)qk_low->ptr,
                (const float *)q->ptr,
                n_tokens,
                head0,
                tile_heads,
                head_tile,
                n_head,
                kv_lora_dim,
                qk_nope,
                qk_rope);
        }
        if (!cuda_ok(cudaGetLastError(),
                     "glm selected attention query gather launch")) {
            return 0;
        }
        glm_selected_gemm_profile_end(
            &profile, GLM_SELECTED_PROFILE_QUERY_GATHER);
        glm_selected_gemm_profile_begin(&profile);
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            g_cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            (int)n_selected,
            (int)tile_heads,
            (int)kv_lora_dim,
            &one,
            kv_h,
            CUDA_R_16F,
            (int)kv_lora_dim,
            kv_stride,
            low_h,
            CUDA_R_16F,
            (int)kv_lora_dim,
            low_stride,
            &zero,
            scores,
            CUDA_R_32F,
            (int)n_selected,
            score_stride,
            (int)n_tokens,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "glm selected attention lora score gemm")) {
            return 0;
        }
        glm_selected_gemm_profile_end(
            &profile, GLM_SELECTED_PROFILE_LORA_SCORE);
        glm_selected_gemm_profile_begin(&profile);
        st = cublasGemmStridedBatchedEx(
            g_cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            (int)n_selected,
            (int)tile_heads,
            (int)qk_rope,
            &one,
            rope_h,
            CUDA_R_16F,
            (int)qk_rope,
            rope_stride,
            qrope_h,
            CUDA_R_16F,
            (int)qk_rope,
            qrope_stride,
            &one,
            scores,
            CUDA_R_32F,
            (int)n_selected,
            score_stride,
            (int)n_tokens,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "glm selected attention rope score gemm")) {
            return 0;
        }
        glm_selected_gemm_profile_end(
            &profile, GLM_SELECTED_PROFILE_ROPE_SCORE);
        glm_selected_gemm_profile_begin(&profile);
        if (head_tile == 1u) {
            glm_selected_gemm_softmax_f16_kernel<<<n_tokens, 256>>>(
                probs, scores, n_tokens, n_selected, scale);
        } else {
            const dim3 softmax_grid(n_tokens, tile_heads, 1u);
            glm_selected_gemm_softmax_heads_f16_kernel<<<
                softmax_grid, 256>>>(
                    probs,
                    scores,
                    n_tokens,
                    n_selected,
                    tile_heads,
                    head_tile,
                    scale);
        }
        if (!cuda_ok(cudaGetLastError(),
                     "glm selected attention softmax launch")) {
            return 0;
        }
        glm_selected_gemm_profile_end(
            &profile, GLM_SELECTED_PROFILE_SOFTMAX);
        glm_selected_gemm_profile_begin(&profile);
        st = cublasGemmStridedBatchedEx(
            g_cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)kv_lora_dim,
            (int)tile_heads,
            (int)n_selected,
            &one,
            kv_h,
            CUDA_R_16F,
            (int)kv_lora_dim,
            kv_stride,
            probs,
            CUDA_R_16F,
            (int)n_selected,
            score_stride,
            &zero,
            head_out,
            CUDA_R_32F,
            (int)kv_lora_dim,
            low_stride,
            (int)n_tokens,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "glm selected attention value gemm")) {
            return 0;
        }
        glm_selected_gemm_profile_end(
            &profile, GLM_SELECTED_PROFILE_VALUE);
        const uint64_t scatter_count =
            (uint64_t)n_tokens * tile_heads * kv_lora_dim;
        const uint32_t scatter_blocks =
            (uint32_t)((scatter_count + 255u) / 256u);
        glm_selected_gemm_profile_begin(&profile);
        if (head_tile == 1u) {
            glm_causal_gemm_scatter_head_kernel<<<scatter_blocks, 256>>>(
                (float *)lora_out->ptr,
                head_out,
                n_tokens,
                head0,
                n_head,
                kv_lora_dim);
        } else {
            glm_selected_gemm_scatter_heads_kernel<<<scatter_blocks, 256>>>(
                (float *)lora_out->ptr,
                head_out,
                n_tokens,
                head0,
                tile_heads,
                head_tile,
                n_head,
                kv_lora_dim);
        }
        if (!cuda_ok(cudaGetLastError(),
                     "glm selected attention head scatter launch")) {
            return 0;
        }
        glm_selected_gemm_profile_end(
            &profile, GLM_SELECTED_PROFILE_SCATTER);
    }
    glm_selected_gemm_profile_report(
        &profile, n_tokens, n_selected, n_head, head_tile);
    return 1;
}

static int glm_attention_indexed_lora_launch(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t n_ctx_orig,
        bool causal_range,
        bool has_selected,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    uint32_t qk_dim = 0;
    if (!glm_rocm_u32_add_checked(qk_nope, qk_rope, &qk_dim)) return 0;
    if (causal_range) {
        uint32_t end_pos = 0;
        if (!glm_rocm_check_token_span(pos0, n_tokens, &end_pos) ||
            end_pos > cache_cap ||
            n_selected < end_pos) {
            return 0;
        }
    }
    if (!lora_out || !q || !qk_low || !kv_lora_cache ||
        (qk_rope != 0u && !k_rope_cache) ||
        n_tokens == 0u || n_selected == 0u ||
        cache_cap == 0u || n_selected > cache_cap ||
        n_head == 0u || kv_lora_dim == 0u ||
        qk_nope == 0u || (qk_rope & 1u) != 0u ||
        !cuda_attention_score_buffer_fits(n_selected) ||
        (qk_rope != 0u && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0u && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        !cuda_tensor_has_elems3(lora_out, n_tokens, n_head, kv_lora_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(q, n_tokens, n_head, qk_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(qk_low, n_tokens, n_head, kv_lora_dim, sizeof(float)) ||
        (has_selected && !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t))) ||
        !glm_rocm_tensor_has_cache2(kv_lora_cache, cache_cap, kv_lora_dim, elem) ||
        (qk_rope != 0u &&
         !glm_rocm_tensor_has_cache2(k_rope_cache, cache_cap, qk_rope, elem))) {
        return 0;
    }
    /* Once the visible context exceeds the indexer's top-k, each token carries
     * its own selected row list.  Gather those rows into per-token FP16
     * matrices, then batch the score and value products through BLAS.  Keep
     * the path bounded: at 256 x 2048 selected rows the temporary workspace is
     * already roughly 580 MiB.  An explicit zero retains the scalar-kernel
     * fallback for diagnosis and numerical comparisons. */
    const char *selected_attn_gemm_env =
        getenv("DS4_ROCM_GLM_SELECTED_ATTN_GEMM");
    if (!causal_range && has_selected &&
        (selected_attn_gemm_env == NULL ||
         cuda_env_present(selected_attn_gemm_env)) &&
        glm_attention_indexed_lora_selected_gemm(lora_out,
                                                  q,
                                                  qk_low,
                                                  kv_lora_cache,
                                                  k_rope_cache,
                                                  selected,
                                                  n_tokens,
                                                  n_selected,
                                                  cache_cap,
                                                  cache_f16,
                                                  n_head,
                                                  kv_lora_dim,
                                                  qk_nope,
                                                  qk_rope,
                                                  n_ctx_orig,
                                                  freq_base,
                                                  freq_scale,
                                                  ext_factor,
                                                  attn_factor,
                                                  beta_fast,
                                                  beta_slow)) {
        static int notice_printed = 0;
        if (!notice_printed) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "GLM selected indexed prefill using fp16 "
                    DS4_GPU_BLAS_NAME
                    " strided-batched attention GEMMs "
                    "(tokens=%u selected=%u cache=%s)\n",
                    n_tokens,
                    n_selected,
                    cache_f16 ? "f16" : "f32");
            notice_printed = 1;
        }
        return 1;
    }
    /* Dense causal prefill can use contiguous cache rows directly.  Keep an
     * explicit zero fallback for diagnosis and numerical comparisons with the
     * scalar attention kernel. */
    const char *causal_attn_gemm_env =
        getenv("DS4_ROCM_GLM_CAUSAL_ATTN_GEMM");
    if (causal_range && !has_selected &&
        (causal_attn_gemm_env == NULL ||
         cuda_env_present(causal_attn_gemm_env)) &&
        glm_attention_indexed_lora_causal_gemm(lora_out,
                                                q,
                                                qk_low,
                                                kv_lora_cache,
                                                k_rope_cache,
                                                n_tokens,
                                                pos0,
                                                n_selected,
                                                cache_f16,
                                                n_head,
                                                kv_lora_dim,
                                                qk_nope,
                                                qk_rope,
                                                n_ctx_orig,
                                                freq_base,
                                                freq_scale,
                                                ext_factor,
                                                attn_factor,
                                                beta_fast,
                                                beta_slow)) {
        static int notice_printed = 0;
        if (!notice_printed) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "GLM causal indexed prefill using fp16 "
                    DS4_GPU_BLAS_NAME
                    " attention GEMMs (tokens=%u rows=%u cache=%s)\n",
                    n_tokens,
                    n_selected,
                    cache_f16 ? "f16" : "f32");
            notice_printed = 1;
        }
        return 1;
    }
    dim3 grid(n_head, n_tokens, 1);
    const size_t shmem = ((size_t)256u + n_selected) * sizeof(float);
    glm_attention_indexed_lora_kernel<<<grid, 256, shmem>>>((float *)lora_out->ptr,
                                                            (const float *)q->ptr,
                                                            (const float *)qk_low->ptr,
                                                            (const char *)kv_lora_cache->ptr,
                                                            k_rope_cache ?
                                                                (const char *)k_rope_cache->ptr : NULL,
                                                            has_selected ? (const int32_t *)selected->ptr : NULL,
                                                            n_tokens,
                                                            pos0,
                                                            n_selected,
                                                            cache_cap,
                                                            cache_f16,
                                                            n_head,
                                                            kv_lora_dim,
                                                            qk_nope,
                                                            qk_rope,
                                                            n_ctx_orig,
                                                            causal_range,
                                                            has_selected,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            beta_fast,
                                                            beta_slow);
    return cuda_ok(cudaGetLastError(), "glm indexed lora launch");
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_tensor(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
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
        float beta_slow) {
    return glm_attention_indexed_lora_launch(lora_out, q, qk_low, kv_lora_cache,
                                             k_rope_cache, selected, n_tokens, 0,
                                             n_selected, cache_cap, cache_f16,
                                             n_head, kv_lora_dim, qk_nope, qk_rope,
                                             n_ctx_orig, false, true, freq_base,
                                             freq_scale, ext_factor, attn_factor,
                                             beta_fast, beta_slow);
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
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
        float beta_slow) {
    return glm_attention_indexed_lora_launch(lora_out, q, qk_low, kv_lora_cache,
                                             k_rope_cache, NULL, n_tokens, pos0,
                                             n_selected, cache_cap, cache_f16,
                                             n_head, kv_lora_dim, qk_nope, qk_rope,
                                             n_ctx_orig, true, false, freq_base,
                                             freq_scale, ext_factor, attn_factor,
                                             beta_fast, beta_slow);
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
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
        float beta_slow) {
    return glm_attention_indexed_lora_launch(lora_out, q, qk_low, kv_lora_cache,
                                             k_rope_cache, selected, n_tokens, 0,
                                             n_selected, cache_cap, cache_f16,
                                             n_head, kv_lora_dim, qk_nope, qk_rope,
                                             n_ctx_orig, false, true, freq_base,
                                             freq_scale, ext_factor, attn_factor,
                                             beta_fast, beta_slow);
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void *model_map,
        uint64_t model_size,
        uint64_t value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t qk_dim = 0;
    ds4_gpu_tensor lora_tmp;
    uint64_t tmp_elems = 0;
    uint64_t q_elems = 0;
    uint64_t low_elems = 0;
    uint64_t heads_elems = 0;
    uint64_t q_bytes = 0;
    uint64_t low_bytes = 0;
    uint64_t heads_bytes = 0;
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!glm_rocm_u32_add_checked(qk_nope, qk_rope, &qk_dim) ||
        !cuda_u64_mul3_checked(n_tokens, n_head, kv_lora_dim, &tmp_elems) ||
        !cuda_u64_mul3_checked(n_tokens, n_head, qk_dim, &q_elems) ||
        !cuda_u64_mul3_checked(n_tokens, n_head, kv_lora_dim, &low_elems) ||
        !cuda_u64_mul3_checked(n_tokens, n_head, value_dim, &heads_elems) ||
        !cuda_u64_mul_checked(tmp_elems, sizeof(float), &lora_tmp.bytes) ||
        !cuda_u64_mul_checked(q_elems, sizeof(float), &q_bytes) ||
        !cuda_u64_mul_checked(low_elems, sizeof(float), &low_bytes) ||
        !cuda_u64_mul_checked(heads_elems, sizeof(float), &heads_bytes)) {
        return 0;
    }
    if (!heads || !q || !qk_low || !kv_lora_cache ||
        (qk_rope != 0u && !k_rope_cache) ||
        !model_map || !selected ||
        n_tokens == 0u || n_selected == 0u ||
        cache_cap == 0u || n_selected > cache_cap ||
        n_head == 0u || kv_lora_dim == 0u || value_dim == 0u ||
        qk_nope == 0u || (qk_rope & 1u) != 0u ||
        !cuda_attention_score_buffer_fits(n_selected) ||
        (qk_rope != 0u && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0u && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        !cuda_tensor_has_bytes(q, q_bytes) ||
        !cuda_tensor_has_bytes(qk_low, low_bytes) ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t)) ||
        !cuda_tensor_has_bytes(heads, heads_bytes) ||
        !glm_rocm_tensor_has_cache2(kv_lora_cache, cache_cap, kv_lora_dim, elem) ||
        (qk_rope != 0u &&
         !glm_rocm_tensor_has_cache2(k_rope_cache, cache_cap, qk_rope, elem))) {
        return 0;
    }
    lora_tmp.ptr = cuda_tmp_alloc(lora_tmp.bytes, "glm indexed batch lora");
    lora_tmp.owner = 0;
    if (!lora_tmp.ptr) return 0;
    if (!ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(&lora_tmp, q, qk_low,
                                                               kv_lora_cache, k_rope_cache,
                                                               selected, n_tokens, n_selected,
                                                               cache_cap, cache_f16, n_head,
                                                               kv_lora_dim, qk_nope, qk_rope,
                                                               n_ctx_orig, freq_base, freq_scale,
                                                               ext_factor, attn_factor, beta_fast,
                                                               beta_slow)) {
        return 0;
    }
    return ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(heads, &lora_tmp, model_map,
                                                             model_size, value_weight_offset,
                                                             n_tokens, n_head, kv_lora_dim,
                                                             value_dim);
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void *model_map,
        uint64_t model_size,
        uint64_t value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        bool cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t qk_dim = 0;
    uint64_t q_bytes = 0;
    uint64_t low_bytes = 0;
    uint64_t heads_bytes = 0;
    const uint64_t elem = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!glm_rocm_u32_add_checked(qk_nope, qk_rope, &qk_dim) ||
        !cuda_u64_mul3_checked(n_head, qk_dim, sizeof(float), &q_bytes) ||
        !cuda_u64_mul3_checked(n_head, kv_lora_dim, sizeof(float), &low_bytes) ||
        !cuda_u64_mul3_checked(n_head, value_dim, sizeof(float), &heads_bytes)) {
        return 0;
    }
    if (!heads ||
        !q ||
        !qk_low ||
        !kv_lora_cache ||
        (qk_rope != 0u && !k_rope_cache) ||
        !selected ||
        !model_map ||
        n_selected == 0u ||
        cache_cap == 0u ||
        n_selected > cache_cap ||
        n_head == 0u ||
        kv_lora_dim == 0u ||
        value_dim == 0u ||
        qk_nope == 0u ||
        (qk_rope & 1u) != 0u ||
        !cuda_attention_score_buffer_fits(n_selected) ||
        (qk_rope != 0u && (!isfinite(freq_base) || freq_base <= 0.0f)) ||
        (qk_rope != 0u && (!isfinite(freq_scale) || freq_scale <= 0.0f)) ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        !cuda_tensor_has_bytes(q, q_bytes) ||
        !cuda_tensor_has_elems2(qk_low, n_head, kv_lora_dim, sizeof(float)) ||
        !cuda_tensor_has_i32(selected, n_selected) ||
        !cuda_tensor_has_bytes(heads, heads_bytes) ||
        !glm_rocm_tensor_has_cache2(kv_lora_cache, cache_cap, kv_lora_dim, elem) ||
        (qk_rope != 0u &&
         !glm_rocm_tensor_has_cache2(k_rope_cache, cache_cap, qk_rope, elem))) {
        return 0;
    }
    ds4_gpu_tensor q_view = *q;
    ds4_gpu_tensor low_view = *qk_low;
    ds4_gpu_tensor sel_view = *selected;
    ds4_gpu_tensor heads_view = *heads;
    ds4_gpu_tensor lora_tmp;
    lora_tmp.bytes = low_bytes;
    lora_tmp.ptr = cuda_tmp_alloc(lora_tmp.bytes, "glm indexed decode lora");
    lora_tmp.owner = 0;
    if (!lora_tmp.ptr) return 0;
    q_view.bytes = q_bytes;
    low_view.bytes = low_bytes;
    sel_view.bytes = (uint64_t)n_selected * sizeof(int32_t);
    heads_view.bytes = heads_bytes;
    if (!glm_attention_indexed_lora_launch(&lora_tmp, &q_view, &low_view, kv_lora_cache,
                                           k_rope_cache, &sel_view, 1, 0, n_selected,
                                           cache_cap, cache_f16, n_head, kv_lora_dim,
                                           qk_nope, qk_rope, n_ctx_orig, false, true,
                                           freq_base, freq_scale, ext_factor, attn_factor,
                                           beta_fast, beta_slow)) {
        return 0;
    }
    return ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(&heads_view, &lora_tmp,
                                                             model_map, model_size,
                                                             value_weight_offset, 1,
                                                             n_head, kv_lora_dim,
                                                             value_dim);
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
        ds4_gpu_tensor *heads,
        ds4_gpu_tensor *partial_lora,
        ds4_gpu_tensor *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void *model_map,
        uint64_t model_size,
        uint64_t value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t n_selected,
        bool selected_rows_valid,
        uint32_t cache_cap,
        bool cache_f16,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        uint32_t block_rows,
        uint32_t n_blocks,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const unsigned char *value_weight = NULL;
    uint32_t row_bytes = 0;
    uint32_t qk_dim = 0;
    const uint32_t needed_blocks =
        block_rows != 0u ? (n_selected + block_rows - 1u) / block_rows : 0u;
    if (!glm_rocm_u32_add_checked(qk_nope, qk_rope, &qk_dim) ||
        !heads || !partial_lora || !partial_ms || !q || !qk_low ||
        !kv_lora_cache || !k_rope_cache || !model_map || !selected ||
        n_selected == 0u || cache_cap == 0u || n_selected > cache_cap ||
        n_head == 0u || (n_head % 8u) != 0u ||
        kv_lora_dim != 512u ||
        qk_nope == 0u || qk_rope != 64u ||
        value_dim == 0u || qk_dim < qk_nope ||
        block_rows == 0u || needed_blocks == 0u ||
        n_blocks < needed_blocks || n_blocks > 64u ||
        !cache_f16 ||
        !isfinite(freq_base) || freq_base <= 0.0f ||
        !isfinite(freq_scale) || freq_scale <= 0.0f ||
        !isfinite(ext_factor) || !isfinite(attn_factor) ||
        !isfinite(beta_fast) || !isfinite(beta_slow) ||
        !cuda_tensor_has_elems2(q, n_head, qk_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(qk_low, n_head, kv_lora_dim, sizeof(float)) ||
        !cuda_tensor_has_i32(selected, n_selected) ||
        !cuda_tensor_has_elems2(heads, n_head, value_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(partial_lora, n_blocks, n_head, kv_lora_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(partial_ms, n_blocks, n_head, 2u, sizeof(float)) ||
        !glm_rocm_tensor_has_cache2(kv_lora_cache, cache_cap, kv_lora_dim, sizeof(__half)) ||
        !glm_rocm_tensor_has_cache2(k_rope_cache, cache_cap, qk_rope, sizeof(__half)) ||
        !glm_rocm_check_q8_rows(model_map, model_size, value_weight_offset,
                                (uint64_t)n_head * value_dim, kv_lora_dim,
                                "glm_value_project_split", &value_weight, &row_bytes)) {
        return 0;
    }

    if (selected_rows_valid) {
        const uint32_t group8_stage_rows = 16u;
        const size_t partial_shmem =
            ((size_t)group8_stage_rows * kv_lora_dim +
             (size_t)group8_stage_rows * qk_rope) * sizeof(float);
        glm_attention_indexed_decode_split_group8_partial_valid_kernel<<<
                dim3(n_head / 8u, n_blocks, 1),
                dim3(32u, 8u, 1),
                partial_shmem>>>(
                (float *)partial_lora->ptr,
                (float *)partial_ms->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const char *)kv_lora_cache->ptr,
                (const char *)k_rope_cache->ptr,
                (const int32_t *)selected->ptr,
                n_selected,
                n_head,
                kv_lora_dim,
                qk_nope,
                qk_rope,
                n_ctx_orig,
                block_rows,
                n_blocks,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow);
    } else {
        const size_t partial_shmem = ((size_t)256u + block_rows) * sizeof(float);
        glm_attention_indexed_decode_split_partial_kernel<<<dim3(n_head, n_blocks, 1),
                                                            256,
                                                            partial_shmem>>>(
                (float *)partial_lora->ptr,
                (float *)partial_ms->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const char *)kv_lora_cache->ptr,
                (const char *)k_rope_cache->ptr,
                (const int32_t *)selected->ptr,
                n_selected,
                selected_rows_valid,
                cache_cap,
                cache_f16,
                n_head,
                kv_lora_dim,
                qk_nope,
                qk_rope,
                n_ctx_orig,
                block_rows,
                n_blocks,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow);
    }
    if (!cuda_ok(cudaGetLastError(), "glm indexed split partial launch")) return 0;

    const size_t reduce_shmem = ((size_t)256u + 64u + kv_lora_dim) * sizeof(float);
    glm_attention_indexed_decode_split_reduce_kernel<<<dim3(n_head, 1, 1),
                                                       256,
                                                       reduce_shmem>>>(
            (float *)heads->ptr,
            (const float *)partial_lora->ptr,
            (const float *)partial_ms->ptr,
            value_weight,
            n_selected,
            n_head,
            kv_lora_dim,
            value_dim,
            row_bytes,
            n_blocks);
    return cuda_ok(cudaGetLastError(), "glm indexed split reduce launch");
}

__device__ __forceinline__ static float glm_router_sigmoid_dev(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ static bool glm_router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

template <uint32_t N_EXPERT>
__global__ static void glm_router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const float *logits,
        uint32_t n_tokens,
        uint32_t n_expert_used,
        float expert_weight_scale) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * N_EXPERT;
    float *prob = probs + (uint64_t)t * N_EXPERT;
    int32_t *sel = selected + (uint64_t)t * n_expert_used;
    float *w = weights + (uint64_t)t * n_expert_used;
    float local_prob[N_EXPERT / 32u];
    float local_score[N_EXPERT / 32u];

    #pragma unroll
    for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = glm_router_sigmoid_dev(log[e]);
        local_prob[j] = p;
        local_score[j] = p + bias[e];
        prob[e] = p;
    }
    __syncwarp();

    float out_prob[DS4_ROCM_N_EXPERT_USED] = {0.0f};
    uint32_t out_idx[DS4_ROCM_N_EXPERT_USED] = {0};
    for (uint32_t k = 0; k < n_expert_used; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;

        #pragma unroll
        for (uint32_t j = 0; j < N_EXPERT / 32u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (glm_router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }

        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(FULL_WARP_MASK, best_score, mask);
            const float other_prob = __shfl_xor_sync(FULL_WARP_MASK, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(FULL_WARP_MASK, best_idx, mask);
            if (glm_router_score_better(other_score, other_idx, best_score, best_idx)) {
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
        for (uint32_t j = 0; j < n_expert_used; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t j = 0; j < n_expert_used; j++) {
            w[j] = w[j] / sum * expert_weight_scale;
        }
    }
}

static int glm_router_select_launch(
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs,
        const void *model_map,
        uint64_t model_size,
        uint64_t bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    const uint32_t active_n_expert = n_expert != 0u ? n_expert : DS4_ROCM_N_EXPERT;
    const uint32_t active_n_expert_used = n_expert_used != 0u ? n_expert_used : DS4_ROCM_N_EXPERT_USED;
    const float active_scale = expert_weight_scale != 0.0f ? expert_weight_scale : DS4_ROCM_EXPERT_WEIGHT_SCALE;
    if (!selected || !weights || !probs || !logits || !model_map || n_tokens == 0 ||
        (active_n_expert != DS4_ROCM_N_EXPERT &&
         active_n_expert != DS4_ROCM_GLM53_N_EXPERT &&
         active_n_expert != DS4_ROCM_MAX_N_EXPERT) ||
        active_n_expert_used == 0u ||
        active_n_expert_used > active_n_expert ||
        active_n_expert_used > DS4_ROCM_N_EXPERT_USED ||
        !(active_scale > 0.0f) ||
        !cuda_tensor_has_elems2(logits, n_tokens, active_n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems2(probs, n_tokens, active_n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, active_n_expert_used, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, active_n_expert_used, sizeof(float)) ||
        !cuda_model_range_fits(model_size, bias_offset, (uint64_t)active_n_expert * sizeof(float))) {
        return 0;
    }

    const float *bias = (const float *)cuda_model_range_ptr(
            model_map,
            bias_offset,
            (uint64_t)active_n_expert * sizeof(float),
            "glm_router_bias");
    if (!bias) return 0;

    dim3 block(32, 4, 1);
    if (active_n_expert == DS4_ROCM_MAX_N_EXPERT) {
        glm_router_select_warp_topk_kernel<DS4_ROCM_MAX_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                (const float *)logits->ptr,
                n_tokens,
                active_n_expert_used,
                active_scale);
    } else if (active_n_expert == DS4_ROCM_GLM53_N_EXPERT) {
        glm_router_select_warp_topk_kernel<DS4_ROCM_GLM53_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                (const float *)logits->ptr,
                n_tokens,
                active_n_expert_used,
                active_scale);
    } else {
        glm_router_select_warp_topk_kernel<DS4_ROCM_N_EXPERT><<<(n_tokens + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                (const float *)logits->ptr,
                n_tokens,
                active_n_expert_used,
                active_scale);
    }
    return cuda_ok(cudaGetLastError(), "glm_router_select launch");
}

extern "C" int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs,
        const void *model_map,
        uint64_t model_size,
        uint64_t bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale) {
    return glm_router_select_launch(selected, weights, probs, model_map, model_size,
                                    bias_offset, logits, n_expert, n_expert_used,
                                    expert_weight_scale, 1);
}

extern "C" int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs,
        const void *model_map,
        uint64_t model_size,
        uint64_t bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    return glm_router_select_launch(selected, weights, probs, model_map, model_size,
                                    bias_offset, logits, n_expert, n_expert_used,
                                    expert_weight_scale, n_tokens);
}

static int glm_rocm_routed_moe_wrap(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float swiglu_clamp,
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride,
        bool force_resident) {
    (void)mid_token_stride;
    if (gate_type != up_type ||
        gate_expert_bytes != up_expert_bytes ||
        gate_row_bytes != up_row_bytes ||
        n_tokens == 0u ||
        n_expert == 0u ||
        n_expert > DS4_ROCM_N_EXPERT_USED) {
        return 0;
    }
    uint64_t pair_elems = 0;
    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    uint64_t gate_pair_bytes = 0;
    uint64_t tmp_bytes = 0;
    if (!cuda_u64_mul3_checked(n_tokens, n_expert, expert_mid_dim, &pair_elems) ||
        !cuda_u64_mul_checked(pair_elems, sizeof(float), &gate_bytes) ||
        !cuda_u64_mul3_checked(n_tokens, n_expert, out_dim, &pair_elems) ||
        !cuda_u64_mul_checked(pair_elems, sizeof(float), &down_bytes) ||
        !cuda_u64_mul_checked(2u, gate_bytes, &gate_pair_bytes) ||
        gate_pair_bytes > UINT64_MAX - down_bytes) {
        return 0;
    }
    tmp_bytes = gate_pair_bytes + down_bytes;
    char *tmp = (char *)cuda_tmp_alloc(tmp_bytes, "glm moe scratch");
    if (!tmp) return 0;
    ds4_gpu_tensor gate_tmp = {tmp, gate_bytes, 0};
    ds4_gpu_tensor up_tmp = {tmp + gate_bytes, gate_bytes, 0};
    ds4_gpu_tensor down_tmp = {tmp + gate_pair_bytes, down_bytes, 0};
    if (n_tokens == 1u) {
        return ds4_gpu_routed_moe_one_tensor(out, &gate_tmp, &up_tmp, mid, &down_tmp,
                                             model_map, model_size, gate_offset,
                                             up_offset, down_offset, gate_type,
                                             down_type, gate_expert_bytes,
                                             gate_row_bytes, down_expert_bytes,
                                             down_row_bytes, expert_in_dim,
                                             expert_mid_dim, out_dim, selected,
                                             weights, n_total_expert, n_expert,
                                             swiglu_clamp, x, NULL, layer_index,
                                             force_resident);
    }
    return ds4_gpu_routed_moe_batch_tensor(out, &gate_tmp, &up_tmp, mid, &down_tmp,
                                           model_map, model_size, gate_offset,
                                           up_offset, down_offset, gate_type,
                                           down_type, gate_expert_bytes,
                                           gate_row_bytes, down_expert_bytes,
                                           down_row_bytes, expert_in_dim,
                                           expert_mid_dim, out_dim, selected,
                                           weights, n_total_expert, n_expert,
                                           swiglu_clamp, x, layer_index, n_tokens, NULL,
                                           force_resident);
}

extern "C" int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float swiglu_clamp,
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        bool force_resident) {
    return glm_rocm_routed_moe_wrap(out, mid, model_map, model_size, gate_offset,
                                    up_offset, down_offset, gate_type, up_type,
                                    down_type, gate_expert_bytes, gate_row_bytes,
                                    up_expert_bytes, up_row_bytes, down_expert_bytes,
                                    down_row_bytes, expert_in_dim, expert_mid_dim,
                                    out_dim, selected, weights, n_total_expert,
                                    n_expert, swiglu_clamp, layer_index, x, 1,
                                    n_expert * expert_mid_dim,
                                    force_resident);
}

extern "C" int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float swiglu_clamp,
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride,
        bool force_resident) {
    return glm_rocm_routed_moe_wrap(out, mid, model_map, model_size, gate_offset,
                                    up_offset, down_offset, gate_type, up_type,
                                    down_type, gate_expert_bytes, gate_row_bytes,
                                    up_expert_bytes, up_row_bytes, down_expert_bytes,
                                    down_row_bytes, expert_in_dim, expert_mid_dim,
                                    out_dim, selected, weights, n_total_expert,
                                    n_expert, swiglu_clamp, layer_index, x, n_tokens,
                                    mid_token_stride, force_resident);
}

extern "C" int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float swiglu_clamp,
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride) {
    return glm_rocm_routed_moe_wrap(out, mid, model_map, model_size, gate_offset,
                                    up_offset, down_offset, gate_type, up_type,
                                    down_type, gate_expert_bytes, gate_row_bytes,
                                    up_expert_bytes, up_row_bytes, down_expert_bytes,
                                    down_row_bytes, expert_in_dim, expert_mid_dim,
                                    out_dim, selected, weights, n_total_expert,
                                    n_expert, swiglu_clamp, layer_index, x, n_tokens,
                                    mid_token_stride, false);
}
