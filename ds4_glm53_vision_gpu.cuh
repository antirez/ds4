/* GLM-5.3 vision operations shared by CUDA and ROCm. */

#ifndef DS4_GLM53_VISION_TYPES_DEFINED
#define DS4_GLM53_VISION_TYPES_DEFINED
#define DS4_GLM53_VISION_LAYERS 24u
typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t q_norm;
    uint64_t k_norm;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t gate_weight;
    uint64_t gate_bias;
    uint64_t up_weight;
    uint64_t up_bias;
    uint64_t down_weight;
    uint64_t down_bias;
} ds4_glm53_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t downsample_weight;
    uint64_t downsample_bias;
    uint64_t merger_proj;
    uint64_t merger_norm;
    uint64_t merger_norm_bias;
    uint64_t merger_gate;
    uint64_t merger_up;
    uint64_t merger_down;
    ds4_glm53_vision_layer_weights layer[DS4_GLM53_VISION_LAYERS];
} ds4_glm53_vision_weights;
#endif

#ifndef DS4_GLM53_VISION_STREAM
#define DS4_GLM53_VISION_STREAM 0
#endif

__device__ __forceinline__ static float glm53_vision_bf16(
        const uint16_t *p) {
    return __uint_as_float((uint32_t)(*p) << 16);
}

__device__ __forceinline__ static float glm53_vision_erf_approx(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * expf(-a * a));
}

__global__ static void glm53_vision_bias_kernel(
        float          *x,
        const uint16_t *bias,
        const float    *residual,
        uint64_t        count,
        uint32_t        width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = x[i] + glm53_vision_bf16(bias + i % width);
    if (residual) v += residual[i];
    x[i] = v;
}

__global__ static void glm53_vision_rms_kernel(
        float          *out,
        const float    *x,
        const uint16_t *weight,
        uint32_t        width,
        float           eps) {
    __shared__ float partial[256];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * width;
    float *yr = out + (uint64_t)row * width;
    float sum = 0.0f;
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        sum = fmaf(xr[d], xr[d], sum);
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)width + eps);
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        yr[d] = xr[d] * inv * glm53_vision_bf16(weight + d);
    }
}

__global__ static void glm53_vision_qkv_rope_kernel(
        float          *q,
        float          *k,
        float          *v,
        const float    *qkv,
        const uint16_t *bias,
        const uint16_t *q_weight,
        const uint16_t *k_weight,
        uint32_t        rows,
        uint32_t        grid_w,
        float           eps) {
    __shared__ float qsum[32];
    __shared__ float ksum[32];
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= rows || head >= 16u || lane >= 32u) return;
    const uint64_t qkv_base = (uint64_t)row * 3072u +
                              (uint64_t)head * 64u;
    const uint64_t out_base = (uint64_t)row * 1024u +
                              (uint64_t)head * 64u;
    float q0 = qkv[qkv_base + lane] +
               glm53_vision_bf16(bias + (uint64_t)head * 64u + lane);
    float q1 = qkv[qkv_base + lane + 32u] +
               glm53_vision_bf16(bias + (uint64_t)head * 64u + lane + 32u);
    float k0 = qkv[qkv_base + 1024u + lane] +
               glm53_vision_bf16(bias + 1024u +
                                 (uint64_t)head * 64u + lane);
    float k1 = qkv[qkv_base + 1024u + lane + 32u] +
               glm53_vision_bf16(bias + 1024u +
                                 (uint64_t)head * 64u + lane + 32u);
    qsum[lane] = fmaf(q0, q0, q1 * q1);
    ksum[lane] = fmaf(k0, k0, k1 * k1);
    __syncthreads();
    for (uint32_t stride = 16u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            qsum[lane] += qsum[lane + stride];
            ksum[lane] += ksum[lane + stride];
        }
        __syncthreads();
    }
    const float qinv = rsqrtf(qsum[0] / 64.0f + eps);
    const float kinv = rsqrtf(ksum[0] / 64.0f + eps);
    q0 *= qinv * glm53_vision_bf16(q_weight + lane);
    q1 *= qinv * glm53_vision_bf16(q_weight + lane + 32u);
    k0 *= kinv * glm53_vision_bf16(k_weight + lane);
    k1 *= kinv * glm53_vision_bf16(k_weight + lane + 32u);

    const uint32_t merge_w = grid_w / 2u;
    const uint32_t group_index = row / 4u;
    const uint32_t within = row & 3u;
    const uint32_t py = (group_index / merge_w) * 2u + within / 2u;
    const uint32_t px = (group_index % merge_w) * 2u + within % 2u;
    const uint32_t freq_index = lane & 15u;
    const uint32_t pos = lane < 16u ? py : px;
    const float inv_freq = powf(10000.0f, -(float)freq_index / 16.0f);
    const float angle = (float)pos * inv_freq;
    const float cs = cosf(angle);
    const float sn = sinf(angle);
    q[out_base + lane] = q0 * cs - q1 * sn;
    q[out_base + lane + 32u] = q1 * cs + q0 * sn;
    k[out_base + lane] = k0 * cs - k1 * sn;
    k[out_base + lane + 32u] = k1 * cs + k0 * sn;
    v[out_base + lane] = qkv[qkv_base + 2048u + lane] +
                         glm53_vision_bf16(bias + 2048u +
                                           (uint64_t)head * 64u + lane);
    v[out_base + lane + 32u] = qkv[qkv_base + 2048u + lane + 32u] +
                               glm53_vision_bf16(
                                   bias + 2048u +
                                   (uint64_t)head * 64u + lane + 32u);
}

__global__ static void glm53_vision_attention_kernel(
        float       *out,
        const float *q,
        const float *k,
        const float *v,
        uint32_t     rows) {
    __shared__ float dot[32];
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= rows || head >= 16u || lane >= 32u) return;
    const uint64_t base = (uint64_t)row * 1024u +
                          (uint64_t)head * 64u;
    const float q0 = q[base + lane];
    const float q1 = q[base + lane + 32u];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float max_score = -INFINITY;
    float denom = 0.0f;
    for (uint32_t key_row = 0; key_row < rows; key_row++) {
        const uint64_t kb = (uint64_t)key_row * 1024u +
                            (uint64_t)head * 64u;
        dot[lane] = q0 * k[kb + lane] + q1 * k[kb + lane + 32u];
        __syncthreads();
        for (uint32_t stride = 16u; stride != 0u; stride >>= 1u) {
            if (lane < stride) dot[lane] += dot[lane + stride];
            __syncthreads();
        }
        const float score = dot[0] * 0.125f;
        const float next_max = fmaxf(max_score, score);
        const float old_scale = key_row == 0u ? 0.0f :
                                expf(max_score - next_max);
        const float new_scale = expf(score - next_max);
        denom = denom * old_scale + new_scale;
        acc0 = acc0 * old_scale + new_scale * v[kb + lane];
        acc1 = acc1 * old_scale + new_scale * v[kb + lane + 32u];
        max_score = next_max;
        __syncthreads();
    }
    out[base + lane] = acc0 / denom;
    out[base + lane + 32u] = acc1 / denom;
}

__global__ static void glm53_vision_swiglu_bias_kernel(
        float          *out,
        const float    *gate,
        const uint16_t *gate_bias,
        const float    *up,
        const uint16_t *up_bias,
        uint64_t        count,
        uint32_t        width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t d = (uint32_t)(i % width);
    const float g = fminf(gate[i] + glm53_vision_bf16(gate_bias + d), 10.0f);
    const float u = fminf(fmaxf(up[i] + glm53_vision_bf16(up_bias + d),
                                -10.0f), 10.0f);
    out[i] = (g / (1.0f + expf(-g))) * u;
}

__global__ static void glm53_vision_reorder_kernel(
        float       *out,
        const float *x,
        uint64_t     count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t d = (uint32_t)(i % 4096u);
    const uint64_t row = i / 4096u;
    const uint32_t channel = d / 4u;
    const uint32_t within = d & 3u;
    out[i] = x[(row * 4u + within) * 1024u + channel];
}

__global__ static void glm53_vision_layernorm_gelu_kernel(
        float          *out,
        const float    *x,
        const uint16_t *weight,
        const uint16_t *bias,
        uint32_t        width,
        float           eps) {
    __shared__ float partial[256];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * width;
    float *yr = out + (uint64_t)row * width;
    float sum = 0.0f;
    for (uint32_t d = tid; d < width; d += blockDim.x) sum += xr[d];
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float mean = partial[0] / (float)width;
    float var = 0.0f;
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        const float centered = xr[d] - mean;
        var = fmaf(centered, centered, var);
    }
    partial[tid] = var;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)width + eps);
    const float inv_sqrt2 = 0.7071067811865475f;
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        float value = (xr[d] - mean) * inv * glm53_vision_bf16(weight + d) +
                      glm53_vision_bf16(bias + d);
        yr[d] = 0.5f * value *
                (1.0f + glm53_vision_erf_approx(value * inv_sqrt2));
    }
}

__global__ static void glm53_vision_scatter_hc_kernel(
        float       *hc,
        const float *image,
        uint32_t     dst_row,
        uint32_t     image_row,
        uint32_t     rows,
        uint32_t     total_rows,
        uint32_t     width,
        uint32_t     n_hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)rows * n_hc * width;
    if (i >= count) return;
    const uint32_t d = (uint32_t)(i % width);
    const uint64_t linear_row = i / width;
    const uint32_t image_delta = (uint32_t)(linear_row / n_hc);
    const uint32_t hc_index = (uint32_t)(linear_row % n_hc);
    if (dst_row + image_delta >= total_rows) return;
    const uint64_t dst = ((uint64_t)(dst_row + image_delta) * n_hc +
                          hc_index) * width + d;
    const uint64_t src = (uint64_t)(image_row + image_delta) * width + d;
    hc[dst] = image[src];
}

static const uint16_t *glm53_vision_weight(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    elements,
        const char *label) {
    if (!model_map || elements > UINT64_MAX / sizeof(uint16_t) ||
        offset > model_size) return NULL;
    const uint64_t bytes = elements * sizeof(uint16_t);
    if (bytes > model_size - offset) return NULL;
#if defined(__HIP_PLATFORM_AMD__)
    return (const uint16_t *)cuda_model_range_ptr(
            model_map, offset, bytes, label);
#else
    return (const uint16_t *)cuda_resolve_weight_ptr(
            model_map, offset, bytes, 0, label);
#endif
}

static int glm53_vision_launch_ok(const char *label) {
    return cuda_ok(cudaGetLastError(), label);
}

extern "C" int ds4_gpu_glm53_vision_encode(
        float                          *out,
        const float                    *patches,
        uint32_t                        grid_h,
        uint32_t                        grid_w,
        const void                     *model_map,
        uint64_t                        model_size,
        const ds4_glm53_vision_weights *weights) {
    if (!out || !patches || !model_map || !weights || grid_h == 0u ||
        grid_w == 0u || (grid_h & 1u) != 0u || (grid_w & 1u) != 0u ||
        grid_h > UINT32_MAX / grid_w) return 0;
    const uint32_t rows = grid_h * grid_w;
    const uint32_t merged_rows = rows / 4u;
    const uint64_t row1024 = (uint64_t)rows * 1024u;
    const uint64_t row3072 = (uint64_t)rows * 3072u;
    const uint64_t row4096 = (uint64_t)rows * 4096u;
    const uint64_t merged10240 = (uint64_t)merged_rows * 10240u;
    if (row4096 > SIZE_MAX / sizeof(float) ||
        merged10240 > SIZE_MAX / sizeof(float) ||
        merged10240 > UINT32_MAX) return 0;

    ds4_gpu_tensor *patch = NULL, *a = NULL, *b = NULL, *qkv = NULL;
    ds4_gpu_tensor *q = NULL, *k = NULL, *v = NULL, *attn = NULL;
    ds4_gpu_tensor *gate = NULL, *up = NULL, *mid = NULL;
    ds4_gpu_tensor *cur = NULL, *tmp = NULL;
    const uint16_t *bias = NULL;
    const uint16_t *merger_norm = NULL, *merger_bias = NULL;
    const uint64_t merged4096 = (uint64_t)merged_rows * 4096u;
    int ok = 0;
#define VISION_ALLOC(name_, count_) do { \
        name_ = ds4_gpu_tensor_alloc((count_) * sizeof(float)); \
        if (!(name_)) goto cleanup; \
    } while (0)
    VISION_ALLOC(patch, (uint64_t)rows * 1176u);
    VISION_ALLOC(a, row1024);
    VISION_ALLOC(b, row1024);
    VISION_ALLOC(qkv, row3072);
    VISION_ALLOC(q, row1024);
    VISION_ALLOC(k, row1024);
    VISION_ALLOC(v, row1024);
    VISION_ALLOC(attn, row1024);
    VISION_ALLOC(gate, row4096);
    VISION_ALLOC(up, row4096);
    VISION_ALLOC(mid, row4096);
#undef VISION_ALLOC
    if (!ds4_gpu_tensor_write(patch, 0, patches,
            (uint64_t)rows * 1176u * sizeof(float)) ||
        !ds4_gpu_begin_commands()) goto cleanup;
    ok = ds4_gpu_glm53_matmul_bf16(
            a, model_map, model_size, weights->patch_weight,
            1176u, 1024u, patch, rows);
    if (ok) {
        bias = glm53_vision_weight(model_map, model_size,
                weights->patch_bias, 1024u, "vision patch bias");
        if (!bias) ok = 0;
    }
    if (ok) {
        glm53_vision_bias_kernel<<<
            (unsigned)((row1024 + 255u) / 256u), 256u, 0,
            DS4_GLM53_VISION_STREAM>>>(
                (float *)a->ptr, bias, NULL, row1024, 1024u);
        ok = glm53_vision_launch_ok("GLM-5.3 vision patch bias");
    }

    cur = a;
    tmp = b;
    for (uint32_t il = 0; ok && il < DS4_GLM53_VISION_LAYERS; il++) {
        const ds4_glm53_vision_layer_weights *w = &weights->layer[il];
        const uint16_t *norm = glm53_vision_weight(
                model_map, model_size, w->norm1, 1024u, "vision norm1");
        if (!norm) { ok = 0; break; }
        glm53_vision_rms_kernel<<<rows, 256u, 0,
            DS4_GLM53_VISION_STREAM>>>(
                (float *)tmp->ptr, (const float *)cur->ptr,
                norm, 1024u, 1.0e-5f);
        ok = glm53_vision_launch_ok("GLM-5.3 vision norm1");
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                qkv, model_map, model_size, w->qkv_weight,
                1024u, 3072u, tmp, rows);
        const uint16_t *qkv_bias = NULL, *q_norm = NULL, *k_norm = NULL;
        if (ok) {
            qkv_bias = glm53_vision_weight(model_map, model_size,
                    w->qkv_bias, 3072u, "vision QKV bias");
            q_norm = glm53_vision_weight(model_map, model_size,
                    w->q_norm, 64u, "vision Q norm");
            k_norm = glm53_vision_weight(model_map, model_size,
                    w->k_norm, 64u, "vision K norm");
            if (!qkv_bias || !q_norm || !k_norm) ok = 0;
        }
        if (ok) {
            glm53_vision_qkv_rope_kernel<<<dim3(rows, 16u, 1u), 32u, 0,
                DS4_GLM53_VISION_STREAM>>>(
                    (float *)q->ptr, (float *)k->ptr, (float *)v->ptr,
                    (const float *)qkv->ptr, qkv_bias, q_norm, k_norm,
                    rows, grid_w, 1.0e-5f);
            ok = glm53_vision_launch_ok("GLM-5.3 vision QKV");
        }
        if (ok) {
            glm53_vision_attention_kernel<<<dim3(rows, 16u, 1u), 32u, 0,
                DS4_GLM53_VISION_STREAM>>>(
                    (float *)attn->ptr, (const float *)q->ptr,
                    (const float *)k->ptr, (const float *)v->ptr, rows);
            ok = glm53_vision_launch_ok("GLM-5.3 vision attention");
        }
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                tmp, model_map, model_size, w->attn_proj_weight,
                1024u, 1024u, attn, rows);
        if (ok) {
            bias = glm53_vision_weight(model_map, model_size,
                    w->attn_proj_bias, 1024u, "vision attention bias");
            if (!bias) ok = 0;
        }
        if (ok) {
            glm53_vision_bias_kernel<<<
                (unsigned)((row1024 + 255u) / 256u), 256u, 0,
                DS4_GLM53_VISION_STREAM>>>(
                    (float *)tmp->ptr, bias, (const float *)cur->ptr,
                    row1024, 1024u);
            ok = glm53_vision_launch_ok("GLM-5.3 vision attention residual");
        }
        ds4_gpu_tensor *swap = cur; cur = tmp; tmp = swap;

        if (ok) {
            norm = glm53_vision_weight(model_map, model_size,
                    w->norm2, 1024u, "vision norm2");
            if (!norm) ok = 0;
        }
        if (ok) {
            glm53_vision_rms_kernel<<<rows, 256u, 0,
                DS4_GLM53_VISION_STREAM>>>(
                    (float *)tmp->ptr, (const float *)cur->ptr,
                    norm, 1024u, 1.0e-5f);
            ok = glm53_vision_launch_ok("GLM-5.3 vision norm2");
        }
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                gate, model_map, model_size, w->gate_weight,
                1024u, 4096u, tmp, rows);
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                up, model_map, model_size, w->up_weight,
                1024u, 4096u, tmp, rows);
        const uint16_t *gate_bias = NULL, *up_bias = NULL;
        if (ok) {
            gate_bias = glm53_vision_weight(model_map, model_size,
                    w->gate_bias, 4096u, "vision gate bias");
            up_bias = glm53_vision_weight(model_map, model_size,
                    w->up_bias, 4096u, "vision up bias");
            if (!gate_bias || !up_bias) ok = 0;
        }
        if (ok) {
            glm53_vision_swiglu_bias_kernel<<<
                (unsigned)((row4096 + 255u) / 256u), 256u, 0,
                DS4_GLM53_VISION_STREAM>>>(
                    (float *)mid->ptr, (const float *)gate->ptr, gate_bias,
                    (const float *)up->ptr, up_bias, row4096, 4096u);
            ok = glm53_vision_launch_ok("GLM-5.3 vision SwiGLU");
        }
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                tmp, model_map, model_size, w->down_weight,
                4096u, 1024u, mid, rows);
        if (ok) {
            bias = glm53_vision_weight(model_map, model_size,
                    w->down_bias, 1024u, "vision down bias");
            if (!bias) ok = 0;
        }
        if (ok) {
            glm53_vision_bias_kernel<<<
                (unsigned)((row1024 + 255u) / 256u), 256u, 0,
                DS4_GLM53_VISION_STREAM>>>(
                    (float *)tmp->ptr, bias, (const float *)cur->ptr,
                    row1024, 1024u);
            ok = glm53_vision_launch_ok("GLM-5.3 vision down residual");
        }
        swap = cur; cur = tmp; tmp = swap;
    }
    if (ok) {
        const uint16_t *norm = glm53_vision_weight(model_map, model_size,
                weights->post_norm, 1024u, "vision post norm");
        if (!norm) ok = 0;
        else {
            glm53_vision_rms_kernel<<<rows, 256u, 0,
                DS4_GLM53_VISION_STREAM>>>(
                    (float *)tmp->ptr, (const float *)cur->ptr,
                    norm, 1024u, 1.0e-5f);
            ok = glm53_vision_launch_ok("GLM-5.3 vision post norm");
        }
    }
    if (ok) {
        glm53_vision_reorder_kernel<<<
            (unsigned)((merged4096 + 255u) / 256u), 256u, 0,
            DS4_GLM53_VISION_STREAM>>>(
                (float *)cur->ptr, (const float *)tmp->ptr, merged4096);
        ok = glm53_vision_launch_ok("GLM-5.3 vision downsample reorder");
    }
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            tmp, model_map, model_size, weights->downsample_weight,
            4096u, 4096u, cur, merged_rows);
    if (ok) {
        bias = glm53_vision_weight(model_map, model_size,
                weights->downsample_bias, 4096u, "vision downsample bias");
        if (!bias) ok = 0;
    }
    if (ok) {
        glm53_vision_bias_kernel<<<
            (unsigned)((merged4096 + 255u) / 256u), 256u, 0,
            DS4_GLM53_VISION_STREAM>>>(
                (float *)tmp->ptr, bias, NULL, merged4096, 4096u);
        ok = glm53_vision_launch_ok("GLM-5.3 vision downsample bias");
    }
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            cur, model_map, model_size, weights->merger_proj,
            4096u, 4096u, tmp, merged_rows);
    if (ok) {
        merger_norm = glm53_vision_weight(model_map, model_size,
                weights->merger_norm, 4096u, "vision merger norm");
        merger_bias = glm53_vision_weight(model_map, model_size,
                weights->merger_norm_bias, 4096u, "vision merger norm bias");
        if (!merger_norm || !merger_bias) ok = 0;
    }
    if (ok) {
        glm53_vision_layernorm_gelu_kernel<<<merged_rows, 256u, 0,
            DS4_GLM53_VISION_STREAM>>>(
                (float *)tmp->ptr, (const float *)cur->ptr,
                merger_norm, merger_bias, 4096u, 1.0e-5f);
        ok = glm53_vision_launch_ok("GLM-5.3 vision merger norm");
    }
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            gate, model_map, model_size, weights->merger_gate,
            4096u, 10240u, tmp, merged_rows);
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            up, model_map, model_size, weights->merger_up,
            4096u, 10240u, tmp, merged_rows);
    if (ok) ok = ds4_gpu_swiglu_tensor(
            mid, gate, up, (uint32_t)merged10240, 10.0f, 1.0f);
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            cur, model_map, model_size, weights->merger_down,
            10240u, 4096u, mid, merged_rows);
    if (ds4_gpu_end_commands() == 0) ok = 0;
    if (ok) ok = ds4_gpu_tensor_read(
            cur, 0, out, merged4096 * sizeof(float));

cleanup:
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(attn);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(qkv);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(patch);
    return ok;
}

extern "C" int ds4_gpu_glm53_scatter_image_hc(
        ds4_gpu_tensor       *hc,
        const ds4_gpu_tensor *image,
        uint32_t              dst_row,
        uint32_t              image_row,
        uint32_t              rows,
        uint32_t              total_rows,
        uint32_t              n_embd,
        uint32_t              n_hc) {
    if (!hc || !image || rows == 0u || n_embd == 0u || n_hc == 0u ||
        dst_row > total_rows || rows > total_rows - dst_row) return 0;
    const uint64_t count = (uint64_t)rows * n_hc * n_embd;
    glm53_vision_scatter_hc_kernel<<<
        (unsigned)((count + 255u) / 256u), 256u, 0,
        DS4_GLM53_VISION_STREAM>>>(
            (float *)hc->ptr, (const float *)image->ptr,
            dst_row, image_row, rows, total_rows, n_embd, n_hc);
    return glm53_vision_launch_ok("GLM-5.3 vision scatter");
}

#undef DS4_GLM53_VISION_STREAM
