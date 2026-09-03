/* DeepSeek V4 Flash Vision-Exp operations shared by CUDA and ROCm. */

#ifndef DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_LAYERS 32u
#define DS4_DEEPSEEK4_LANGUAGE_LAYERS 43u
#define DS4_DEEPSEEK4_MTP_LAYERS 3u

typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t mlp_w1;
    uint64_t mlp_w2;
} ds4_deepseek4_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t aligner_w1;
    uint64_t aligner_w1_bias;
    uint64_t aligner_w2;
    uint64_t aligner_w2_bias;
    uint64_t image_start;
    uint64_t image_pad;
    uint64_t image_newline;
    uint64_t image_end;
    uint64_t visual_router_bias[DS4_DEEPSEEK4_LANGUAGE_LAYERS];
    uint64_t mtp_visual_router_bias[DS4_DEEPSEEK4_MTP_LAYERS];
    uint64_t hash_router_bias[3];
    ds4_deepseek4_vision_layer_weights layer[DS4_DEEPSEEK4_VISION_LAYERS];
} ds4_deepseek4_vision_weights;
#endif

#ifndef DS4_DEEPSEEK4_VISION_STREAM
#define DS4_DEEPSEEK4_VISION_STREAM 0
#endif

__device__ __forceinline__ static float deepseek4_vision_round_bf16_dev(
        float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x00007fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__global__ static void deepseek4_vision_round_bf16_kernel(
        float    *x,
        uint64_t  count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = deepseek4_vision_round_bf16_dev(x[i]);
}

__global__ static void deepseek4_vision_qkv_rope_kernel(
        float          *q,
        float          *k,
        float          *v,
        const float    *qkv,
        const uint16_t *bias,
        uint32_t        rows,
        uint32_t        grid_w) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= rows || head >= 16u || lane >= 32u) return;
    const uint64_t qkv_base = (uint64_t)row * 3072u +
                              (uint64_t)head * 64u;
    const uint64_t out_base = (uint64_t)row * 1024u +
                              (uint64_t)head * 64u;
    const float q0 = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + lane] +
            glm53_vision_bf16(bias + (uint64_t)head * 64u + lane));
    const float q1 = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + lane + 32u] +
            glm53_vision_bf16(bias + (uint64_t)head * 64u + lane + 32u));
    const float k0 = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + 1024u + lane] +
            glm53_vision_bf16(bias + 1024u +
                              (uint64_t)head * 64u + lane));
    const float k1 = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + 1024u + lane + 32u] +
            glm53_vision_bf16(bias + 1024u +
                              (uint64_t)head * 64u + lane + 32u));
    const uint32_t y = row / grid_w;
    const uint32_t x = row - y * grid_w;
    const uint32_t pos = lane < 16u ? y : x;
    const uint32_t freq = lane & 15u;
    const float inv_freq = powf(10000.0f, -(float)freq / 16.0f);
    const float angle = (float)pos * inv_freq;
    const float cs = cosf(angle);
    const float sn = sinf(angle);
    q[out_base + lane] = deepseek4_vision_round_bf16_dev(q0 * cs - q1 * sn);
    q[out_base + lane + 32u] =
        deepseek4_vision_round_bf16_dev(q1 * cs + q0 * sn);
    k[out_base + lane] = deepseek4_vision_round_bf16_dev(k0 * cs - k1 * sn);
    k[out_base + lane + 32u] =
        deepseek4_vision_round_bf16_dev(k1 * cs + k0 * sn);
    v[out_base + lane] = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + 2048u + lane] +
            glm53_vision_bf16(bias + 2048u +
                              (uint64_t)head * 64u + lane));
    v[out_base + lane + 32u] = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + 2048u + lane + 32u] +
            glm53_vision_bf16(bias + 2048u +
                              (uint64_t)head * 64u + lane + 32u));
}

__global__ static void deepseek4_vision_swiglu_split_kernel(
        float       *out,
        const float *gate_up,
        uint64_t     count,
        uint32_t     width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t d = i % width;
    const uint64_t row = i / width;
    const uint64_t source = row * width * 2u + d;
    const float gate = gate_up[source];
    const float up = gate_up[source + width];
    out[i] = deepseek4_vision_round_bf16_dev(
            (gate / (1.0f + expf(-gate))) * up);
}

__global__ static void deepseek4_vision_add_residual_kernel(
        float       *x,
        const float *residual,
        uint64_t     count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        x[i] = deepseek4_vision_round_bf16_dev(x[i] + residual[i]);
}

__global__ static void deepseek4_vision_aligner_reorder_kernel(
        float       *out,
        const float *x,
        uint32_t     grid_h,
        uint32_t     grid_w,
        uint32_t     output_rows) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (d >= 9216u || row >= output_rows) return;
    const uint32_t output_w = (grid_w + 2u) / 3u;
    const uint32_t block_y = row / output_w;
    const uint32_t block_x = row - block_y * output_w;
    const uint32_t channel = d / 9u;
    const uint32_t within = d - channel * 9u;
    const uint32_t source_y = block_y * 3u + within / 3u;
    const uint32_t source_x = block_x * 3u + within % 3u;
    float value = 0.0f;
    if (source_y < grid_h && source_x < grid_w) {
        const uint64_t source_row = (uint64_t)source_y * grid_w + source_x;
        value = x[source_row * 1024u + channel];
    }
    out[(uint64_t)row * 9216u + d] = value;
}

__global__ static void deepseek4_vision_gelu_bias_kernel(
        float          *out,
        const float    *x,
        const uint16_t *bias,
        uint64_t        count,
        uint32_t        width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t d = i % width;
    const float value = deepseek4_vision_round_bf16_dev(
            x[i] + glm53_vision_bf16(bias + d));
    out[i] = deepseek4_vision_round_bf16_dev(
            0.5f * value *
            (1.0f + glm53_vision_erf_approx(
                value * 0.7071067811865475f)));
}

static int deepseek4_vision_round_tensor(
        ds4_gpu_tensor *x,
        uint64_t        count,
        const char     *label) {
    deepseek4_vision_round_bf16_kernel<<<
        (unsigned)((count + 255u) / 256u), 256u, 0,
        DS4_DEEPSEEK4_VISION_STREAM>>>((float *)x->ptr, count);
    return glm53_vision_launch_ok(label);
}

extern "C" int ds4_gpu_deepseek4_vision_encode(
        float                              *out,
        const float                        *patches,
        uint32_t                            grid_h,
        uint32_t                            grid_w,
        const void                         *model_map,
        uint64_t                            model_size,
        const ds4_deepseek4_vision_weights *weights) {
    if (!out || !patches || !model_map || !weights || grid_h == 0u ||
        grid_w == 0u || grid_h > UINT32_MAX / grid_w) return 0;
    const uint32_t rows = grid_h * grid_w;
    const uint32_t aligned_rows =
        ((grid_h + 2u) / 3u) * ((grid_w + 2u) / 3u);
    const uint64_t row1024 = (uint64_t)rows * 1024u;
    const uint64_t row2816 = (uint64_t)rows * 2816u;
    const uint64_t row3072 = (uint64_t)rows * 3072u;
    const uint64_t row5632 = (uint64_t)rows * 5632u;
    const uint64_t aligned4096 = (uint64_t)aligned_rows * 4096u;
    const uint64_t aligned9216 = (uint64_t)aligned_rows * 9216u;
    if (row5632 > SIZE_MAX / sizeof(float) ||
        aligned9216 > SIZE_MAX / sizeof(float)) return 0;

    ds4_gpu_tensor *patch = NULL, *a = NULL, *b = NULL, *qkv = NULL;
    ds4_gpu_tensor *q = NULL, *k = NULL, *v = NULL, *attn = NULL;
    ds4_gpu_tensor *mlp_w1 = NULL, *mlp_mid = NULL;
    ds4_gpu_tensor *align_in = NULL, *align_a = NULL, *align_b = NULL;
    ds4_gpu_tensor *cur = NULL, *tmp = NULL;
    const uint16_t *bias = NULL;
    int ok = 0;
#define DSV4_VISION_ALLOC(name_, count_) do { \
        name_ = ds4_gpu_tensor_alloc((count_) * sizeof(float)); \
        if (!(name_)) goto cleanup; \
    } while (0)
    DSV4_VISION_ALLOC(patch, (uint64_t)rows * 588u);
    DSV4_VISION_ALLOC(a, row1024);
    DSV4_VISION_ALLOC(b, row1024);
    DSV4_VISION_ALLOC(qkv, row3072);
    DSV4_VISION_ALLOC(q, row1024);
    DSV4_VISION_ALLOC(k, row1024);
    DSV4_VISION_ALLOC(v, row1024);
    DSV4_VISION_ALLOC(attn, row1024);
    DSV4_VISION_ALLOC(mlp_w1, row5632);
    DSV4_VISION_ALLOC(mlp_mid, row2816);
    DSV4_VISION_ALLOC(align_in, aligned9216);
    DSV4_VISION_ALLOC(align_a, aligned4096);
    DSV4_VISION_ALLOC(align_b, aligned4096);
#undef DSV4_VISION_ALLOC
    if (!ds4_gpu_tensor_write(
            patch, 0, patches,
            (uint64_t)rows * 588u * sizeof(float)) ||
        !ds4_gpu_begin_commands()) goto cleanup;
    ok = ds4_gpu_glm53_matmul_bf16(
            a, model_map, model_size, weights->patch_weight,
            588u, 1024u, patch, rows);
    if (ok) {
        bias = glm53_vision_weight(model_map, model_size,
                weights->patch_bias, 1024u, "DeepSeek vision patch bias");
        if (!bias) ok = 0;
    }
    if (ok) {
        glm53_vision_bias_kernel<<<
            (unsigned)((row1024 + 255u) / 256u), 256u, 0,
            DS4_DEEPSEEK4_VISION_STREAM>>>(
                (float *)a->ptr, bias, NULL, row1024, 1024u);
        ok = glm53_vision_launch_ok("DeepSeek vision patch bias");
    }
    if (ok) ok = deepseek4_vision_round_tensor(
            a, row1024, "DeepSeek vision patch round");

    cur = a;
    tmp = b;
    for (uint32_t il = 0; ok && il < DS4_DEEPSEEK4_VISION_LAYERS; il++) {
        const ds4_deepseek4_vision_layer_weights *w = &weights->layer[il];
        const uint16_t *norm = glm53_vision_weight(
                model_map, model_size, w->norm1, 1024u,
                "DeepSeek vision norm1");
        if (!norm) { ok = 0; break; }
        glm53_vision_rms_kernel<<<rows, 256u, 0,
            DS4_DEEPSEEK4_VISION_STREAM>>>(
                (float *)tmp->ptr, (const float *)cur->ptr,
                norm, 1024u, 1.0e-6f);
        ok = glm53_vision_launch_ok("DeepSeek vision norm1");
        if (ok) ok = deepseek4_vision_round_tensor(
                tmp, row1024, "DeepSeek vision norm1 round");
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                qkv, model_map, model_size, w->qkv_weight,
                1024u, 3072u, tmp, rows);
        const uint16_t *qkv_bias = NULL;
        if (ok) {
            qkv_bias = glm53_vision_weight(
                    model_map, model_size, w->qkv_bias, 3072u,
                    "DeepSeek vision QKV bias");
            if (!qkv_bias) ok = 0;
        }
        if (ok) {
            deepseek4_vision_qkv_rope_kernel<<<
                dim3(rows, 16u, 1u), 32u, 0,
                DS4_DEEPSEEK4_VISION_STREAM>>>(
                    (float *)q->ptr, (float *)k->ptr, (float *)v->ptr,
                    (const float *)qkv->ptr, qkv_bias, rows, grid_w);
            ok = glm53_vision_launch_ok("DeepSeek vision QKV");
        }
        if (ok) {
            glm53_vision_attention_kernel<<<
                dim3(rows, 16u, 1u), 32u, 0,
                DS4_DEEPSEEK4_VISION_STREAM>>>(
                    (float *)attn->ptr, (const float *)q->ptr,
                    (const float *)k->ptr, (const float *)v->ptr, rows);
            ok = glm53_vision_launch_ok("DeepSeek vision attention");
        }
        if (ok) ok = deepseek4_vision_round_tensor(
                attn, row1024, "DeepSeek vision attention round");
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                tmp, model_map, model_size, w->attn_proj_weight,
                1024u, 1024u, attn, rows);
        if (ok) {
            bias = glm53_vision_weight(
                    model_map, model_size, w->attn_proj_bias, 1024u,
                    "DeepSeek vision attention bias");
            if (!bias) ok = 0;
        }
        if (ok) {
            glm53_vision_bias_kernel<<<
                (unsigned)((row1024 + 255u) / 256u), 256u, 0,
                DS4_DEEPSEEK4_VISION_STREAM>>>(
                    (float *)tmp->ptr, bias, (const float *)cur->ptr,
                    row1024, 1024u);
            ok = glm53_vision_launch_ok(
                    "DeepSeek vision attention residual");
        }
        if (ok) ok = deepseek4_vision_round_tensor(
                tmp, row1024, "DeepSeek vision attention residual round");
        ds4_gpu_tensor *swap = cur; cur = tmp; tmp = swap;

        if (ok) {
            norm = glm53_vision_weight(
                    model_map, model_size, w->norm2, 1024u,
                    "DeepSeek vision norm2");
            if (!norm) ok = 0;
        }
        if (ok) {
            glm53_vision_rms_kernel<<<rows, 256u, 0,
                DS4_DEEPSEEK4_VISION_STREAM>>>(
                    (float *)tmp->ptr, (const float *)cur->ptr,
                    norm, 1024u, 1.0e-6f);
            ok = glm53_vision_launch_ok("DeepSeek vision norm2");
        }
        if (ok) ok = deepseek4_vision_round_tensor(
                tmp, row1024, "DeepSeek vision norm2 round");
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                mlp_w1, model_map, model_size, w->mlp_w1,
                1024u, 5632u, tmp, rows);
        if (ok) ok = deepseek4_vision_round_tensor(
                mlp_w1, row5632, "DeepSeek vision MLP input round");
        if (ok) {
            deepseek4_vision_swiglu_split_kernel<<<
                (unsigned)((row2816 + 255u) / 256u), 256u, 0,
                DS4_DEEPSEEK4_VISION_STREAM>>>(
                    (float *)mlp_mid->ptr, (const float *)mlp_w1->ptr,
                    row2816, 2816u);
            ok = glm53_vision_launch_ok("DeepSeek vision SwiGLU");
        }
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                tmp, model_map, model_size, w->mlp_w2,
                2816u, 1024u, mlp_mid, rows);
        if (ok) ok = deepseek4_vision_round_tensor(
                tmp, row1024, "DeepSeek vision MLP output round");
        if (ok) {
            deepseek4_vision_add_residual_kernel<<<
                (unsigned)((row1024 + 255u) / 256u), 256u, 0,
                DS4_DEEPSEEK4_VISION_STREAM>>>(
                    (float *)tmp->ptr, (const float *)cur->ptr, row1024);
            ok = glm53_vision_launch_ok("DeepSeek vision MLP residual");
        }
        swap = cur; cur = tmp; tmp = swap;
    }
    if (ok) {
        const uint16_t *norm = glm53_vision_weight(
                model_map, model_size, weights->post_norm, 1024u,
                "DeepSeek vision final norm");
        if (!norm) ok = 0;
        else {
            glm53_vision_rms_kernel<<<rows, 256u, 0,
                DS4_DEEPSEEK4_VISION_STREAM>>>(
                    (float *)tmp->ptr, (const float *)cur->ptr,
                    norm, 1024u, 1.0e-6f);
            ok = glm53_vision_launch_ok("DeepSeek vision final norm");
        }
    }
    if (ok) ok = deepseek4_vision_round_tensor(
            tmp, row1024, "DeepSeek vision final norm round");
    if (ok) {
        dim3 grid((9216u + 255u) / 256u, aligned_rows, 1u);
        deepseek4_vision_aligner_reorder_kernel<<<
            grid, 256u, 0, DS4_DEEPSEEK4_VISION_STREAM>>>(
                (float *)align_in->ptr, (const float *)tmp->ptr,
                grid_h, grid_w, aligned_rows);
        ok = glm53_vision_launch_ok("DeepSeek vision aligner reorder");
    }
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            align_a, model_map, model_size, weights->aligner_w1,
            9216u, 4096u, align_in, aligned_rows);
    if (ok) {
        bias = glm53_vision_weight(
                model_map, model_size, weights->aligner_w1_bias, 4096u,
                "DeepSeek vision aligner hidden bias");
        if (!bias) ok = 0;
    }
    if (ok) {
        deepseek4_vision_gelu_bias_kernel<<<
            (unsigned)((aligned4096 + 255u) / 256u), 256u, 0,
            DS4_DEEPSEEK4_VISION_STREAM>>>(
                (float *)align_b->ptr, (const float *)align_a->ptr,
                bias, aligned4096, 4096u);
        ok = glm53_vision_launch_ok("DeepSeek vision aligner GELU");
    }
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            align_a, model_map, model_size, weights->aligner_w2,
            4096u, 4096u, align_b, aligned_rows);
    if (ok) {
        bias = glm53_vision_weight(
                model_map, model_size, weights->aligner_w2_bias, 4096u,
                "DeepSeek vision aligner output bias");
        if (!bias) ok = 0;
    }
    if (ok) {
        glm53_vision_bias_kernel<<<
            (unsigned)((aligned4096 + 255u) / 256u), 256u, 0,
            DS4_DEEPSEEK4_VISION_STREAM>>>(
                (float *)align_a->ptr, bias, NULL, aligned4096, 4096u);
        ok = glm53_vision_launch_ok(
                "DeepSeek vision aligner output bias");
    }
    if (ok) ok = deepseek4_vision_round_tensor(
            align_a, aligned4096, "DeepSeek vision aligner output round");
    if (ds4_gpu_end_commands() == 0) ok = 0;
    if (ok) ok = ds4_gpu_tensor_read(
            align_a, 0, out, aligned4096 * sizeof(float));

cleanup:
    ds4_gpu_tensor_free(align_b);
    ds4_gpu_tensor_free(align_a);
    ds4_gpu_tensor_free(align_in);
    ds4_gpu_tensor_free(mlp_mid);
    ds4_gpu_tensor_free(mlp_w1);
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

#undef DS4_DEEPSEEK4_VISION_STREAM
