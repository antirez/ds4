/* CUDA port of the Metal GLM-5.2 MoE and general utility backend functions.
 *
 * This file is included at the end of ds4_cuda.cu and uses its internals
 * (cuda_ok, cuda_model_range_ptr, cuda_model_copy_to_device_streamed,
 * g_ssd_streaming_mode, g_stream_selected_cache, g_stream_expert_cache, ...).
 *
 * Ports:
 *   - kernel_glm_router_select_one            (metal/dsv4_misc.metal)
 *   - kernel_glm_qX_K_pair_swiglu / down      (metal/moe.metal, scalar variants)
 *   - kernel_add3_f32, kernel_add_rms_norm_mul_f32_4, kernel_dsv4_sort_i32_rows_asc
 *   - streaming full-layer expert cache        (rocm/ds4_rocm_runtime.cuh design,
 *     rebuilt on ds4_cuda.cu's own streamed-copy + expert-cache machinery)
 */

/* =========================================================================
 * K-quant block layouts missing from ds4_cuda.cu (q2/q4 already exist).
 * ========================================================================= */

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[CUDA_QK_K / 8];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q5_K;

typedef struct {
    uint8_t ql[CUDA_QK_K / 2];
    uint8_t qh[CUDA_QK_K / 4];
    int8_t scales[CUDA_QK_K / 16];
    uint16_t d;
} cuda_block_q6_K;

/* =========================================================================
 * Per-element dequantization (faithful port of metal/moe.metal
 * ds4_glm_qX_K_value helpers).
 * ========================================================================= */

__device__ __forceinline__ static float glm_half_bits_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ static void glm_get_scale_min_k4_just2(
        uint32_t j, const uint8_t *q, uint32_t *sc, uint32_t *mn) {
    if (j < 4u) {
        *sc = q[j] & 63u;
        *mn = q[j + 4u] & 63u;
    } else {
        *sc = (q[j + 4u] & 0x0Fu) | ((q[j - 4u] & 0xC0u) >> 2u);
        *mn = (q[j + 4u] >> 4u) | ((q[j] & 0xC0u) >> 2u);
    }
}

struct glm_deq_q2_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const cuda_block_q2_K *xb =
            (const cuda_block_q2_K *)blocks + (k / CUDA_QK_K);
        const uint32_t idx = k & (CUDA_QK_K - 1u);
        const uint32_t group = idx / 16u;
        const uint32_t l = idx & 15u;
        const uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
        const uint32_t shift = ((group / 2u) & 3u) * 2u;
        const uint32_t q = ((uint32_t)xb->qs[q_base + l] >> shift) & 0x03u;
        const uint32_t sc = (uint32_t)xb->scales[group];
        return glm_half_bits_to_f32(xb->d) * (float)(sc & 0x0Fu) * (float)q -
               glm_half_bits_to_f32(xb->dmin) * (float)(sc >> 4u);
    }
};

/* IQ2_XXS scalar dequant. Layout: {half d; uint16 qs[QK_K/8];}; each 32-element
 * sub-block packs 4 grid indices (aux0) + a 4-bit scale and four 7-bit sign
 * indices (aux1). value = 0.125 * d * ls * grid[j] * sign, matching the dp4a
 * path dev_dot_iq2_xxs_q8_K_block_lut. Uses ds4_cuda.cu's cuda_iq2xxs_grid /
 * cuda_ksigns_iq2xs tables (in scope: this file is included at end of ds4_cuda.cu). */
struct glm_deq_iq2_xxs {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const cuda_block_iq2_xxs *xb =
            (const cuda_block_iq2_xxs *)blocks + (k >> 8);
        const uint32_t within = k & 255u;
        const uint32_t ib32 = within >> 5;
        const uint32_t kk = within & 31u;
        const uint32_t l = kk >> 3;
        const uint32_t j = kk & 7u;
        const uint16_t *q2 = xb->qs + 4u * ib32;
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint32_t grid_idx = (aux0 >> (8u * l)) & 0xffu;
        const uint32_t sign_idx = (aux1 >> (7u * l)) & 127u;
        const uint64_t g = cuda_iq2xxs_grid[grid_idx];
        const int32_t gj = (int32_t)(uint32_t)(uint8_t)(g >> (8u * j));
        const uint8_t signs = cuda_ksigns_iq2xs[sign_idx];
        const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
        return 0.125f * dev_f16_to_f32(xb->d) * (float)ls * (float)gj * sign;
    }
};

struct glm_deq_q4_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const cuda_block_q4_K *xb =
            (const cuda_block_q4_K *)blocks + (k / CUDA_QK_K);
        const uint32_t idx = k & (CUDA_QK_K - 1u);
        const uint32_t group = idx / 32u;
        const uint32_t l = idx & 31u;
        uint32_t sc, mn;
        glm_get_scale_min_k4_just2(group, xb->scales, &sc, &mn);
        const uint32_t byte_off = (group >> 1u) * 32u + l;
        const uint32_t shift = (group & 1u) * 4u;
        const uint32_t q = ((uint32_t)xb->qs[byte_off] >> shift) & 0x0Fu;
        return glm_half_bits_to_f32(xb->d) * (float)sc * (float)q -
               glm_half_bits_to_f32(xb->dmin) * (float)mn;
    }
};

struct glm_deq_q5_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const cuda_block_q5_K *xb =
            (const cuda_block_q5_K *)blocks + (k / CUDA_QK_K);
        const uint32_t idx = k & (CUDA_QK_K - 1u);
        const uint32_t group = idx / 32u;
        const uint32_t l = idx & 31u;
        uint32_t sc, mn;
        glm_get_scale_min_k4_just2(group, xb->scales, &sc, &mn);
        const uint32_t ql_base = (group >> 1u) * 32u + l;
        const uint32_t shift = (group & 1u) * 4u;
        uint32_t q = ((uint32_t)xb->qs[ql_base] >> shift) & 0x0Fu;
        q += (xb->qh[l] & (uint8_t)(1u << group)) ? 16u : 0u;
        return glm_half_bits_to_f32(xb->d) * (float)sc * (float)q -
               glm_half_bits_to_f32(xb->dmin) * (float)mn;
    }
};

struct glm_deq_q6_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const cuda_block_q6_K *xb =
            (const cuda_block_q6_K *)blocks + (k / CUDA_QK_K);
        const uint32_t idx = k & (CUDA_QK_K - 1u);
        const uint32_t n128 = idx >> 7u;
        const uint32_t r = idx & 127u;
        const uint32_t l = r & 31u;
        const uint32_t quarter = r >> 5u;
        const uint32_t ql_base = n128 * 64u;
        const uint32_t qh_base = n128 * 32u;
        const uint32_t sc_base = n128 * 8u;
        uint32_t q;
        int sc;
        if (quarter == 0u) {
            q = (xb->ql[ql_base + l] & 0x0Fu) |
                (((xb->qh[qh_base + l] >> 0u) & 3u) << 4u);
            sc = (int)xb->scales[sc_base + l / 16u + 0u];
        } else if (quarter == 1u) {
            q = (xb->ql[ql_base + 32u + l] & 0x0Fu) |
                (((xb->qh[qh_base + l] >> 2u) & 3u) << 4u);
            sc = (int)xb->scales[sc_base + l / 16u + 2u];
        } else if (quarter == 2u) {
            q = (xb->ql[ql_base + l] >> 4u) |
                (((xb->qh[qh_base + l] >> 4u) & 3u) << 4u);
            sc = (int)xb->scales[sc_base + l / 16u + 4u];
        } else {
            q = (xb->ql[ql_base + 32u + l] >> 4u) |
                (((xb->qh[qh_base + l] >> 6u) & 3u) << 4u);
            sc = (int)xb->scales[sc_base + l / 16u + 6u];
        }
        return glm_half_bits_to_f32(xb->d) * (float)sc * (float)((int)q - 32);
    }
};

/* Tensor type ids (GGUF), mirrors DS4_METAL_TENSOR_* in ds4_metal.m. */
enum {
    GLM_CUDA_TENSOR_Q2_K = 10,
    GLM_CUDA_TENSOR_Q4_K = 12,
    GLM_CUDA_TENSOR_Q5_K = 13,
    GLM_CUDA_TENSOR_Q6_K = 14,
    GLM_CUDA_TENSOR_IQ2_XXS = 16
};

/* =========================================================================
 * Small utility kernels.
 * ========================================================================= */

__global__ static void glm_add3_kernel(
        float *out, const float *a, const float *b, const float *c, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i] + c[i];
}

/* sum_out = a + b; norm_out = rms_norm(sum_out) * w.  Single row, mirrors
 * kernel_add_rms_norm_mul_f32_4 in metal/norm.metal (rows == 1 dispatch). */
__global__ static void glm_add_rms_norm_weight_kernel(
        float *norm_out,
        float *sum_out,
        const float *a,
        const float *b,
        const float *w,
        uint32_t n,
        float eps) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = a[i] + b[i];
        sum_out[i] = v;
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        norm_out[i] = sum_out[i] * scale * w[i];
    }
}

/* Per-row ascending bitonic sort of int32 ids, port of
 * kernel_dsv4_sort_i32_rows_asc (metal/dsv4_misc.metal).  row_width must be a
 * power of two (enforced by the launcher, as on Metal). */
__global__ static void glm_sort_i32_rows_asc_kernel(
        int32_t *dst, const int32_t *src, uint32_t row_width, uint32_t n_rows) {
    extern __shared__ int32_t glm_sort_row_tmp[];
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint32_t n_threads = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint64_t base = (uint64_t)row * row_width;

    for (uint32_t i = tid; i < row_width; i += n_threads) {
        glm_sort_row_tmp[i] = src[base + i];
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= row_width; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < row_width; i += n_threads) {
                const uint32_t other = i ^ j;
                if (other > i && other < row_width) {
                    const int32_t va = glm_sort_row_tmp[i];
                    const int32_t vb = glm_sort_row_tmp[other];
                    const bool up = (i & k) == 0u;
                    if ((up && va > vb) || (!up && va < vb)) {
                        glm_sort_row_tmp[i] = vb;
                        glm_sort_row_tmp[other] = va;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < row_width; i += n_threads) {
        dst[base + i] = glm_sort_row_tmp[i];
    }
}

/* =========================================================================
 * GLM router: sigmoid probabilities, selection score = prob + bias, top-k,
 * route weights = selected prob / sum(selected probs) * scale.
 * Port of kernel_glm_router_select_one (metal/dsv4_misc.metal).
 * One warp per token; n_expert <= 256, k_used <= n_expert.
 * ========================================================================= */

__device__ __forceinline__ static float glm_router_sigmoid_dev(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ static bool glm_router_better_dev(
        float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void glm_router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const float *logits,
        uint32_t n_expert,
        uint32_t k_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    const uint32_t lane = threadIdx.x;
    const uint32_t token = blockIdx.x * blockDim.y + threadIdx.y;
    if (token >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)token * n_expert;
    float *prob = probs + (uint64_t)token * n_expert;
    int32_t *sel = selected + (uint64_t)token * k_used;
    float *w = weights + (uint64_t)token * k_used;

    float local_prob[8];
    float local_score[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        if (e < n_expert) {
            const float p = glm_router_sigmoid_dev(log[e]);
            local_prob[j] = p;
            local_score[j] = p + bias[e];
            prob[e] = p;
        } else {
            local_prob[j] = 0.0f;
            local_score[j] = -INFINITY;
        }
    }
    __syncwarp();

    float sum = 0.0f;
    for (uint32_t k = 0; k < k_used; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (glm_router_better_dev(local_score[j], e, best_score, best_idx)) {
                best_score = local_score[j];
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (glm_router_better_dev(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            if (lane + j * 32u == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            sel[k] = (int32_t)best_idx;
            w[k] = best_prob;
        }
        sum += best_prob;
    }

    if (lane == 0) {
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t k = 0; k < k_used; k++) {
            w[k] = w[k] / sum * expert_weight_scale;
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
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_tokens == 0 ||
        n_expert == 0 || n_expert > 256u ||
        n_expert_used == 0 || n_expert_used > n_expert) {
        return 0;
    }
    if (logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float)) {
        fprintf(stderr, "ds4: CUDA GLM router received undersized buffers\n");
        return 0;
    }
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    if (bias_offset > model_size || bias_bytes > model_size - bias_offset) {
        fprintf(stderr, "ds4: CUDA GLM router bias range is outside the mapped model\n");
        return 0;
    }
    const float *bias = (const float *)cuda_model_range_ptr(model_map,
            bias_offset, bias_bytes, "glm_router_bias");
    if (!bias) return 0;

    dim3 block(32, 4, 1);
    glm_router_select_kernel<<<(n_tokens + 3u) / 4u, block>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            (float *)probs->ptr,
            bias,
            (const float *)logits->ptr,
            n_expert,
            n_expert_used,
            expert_weight_scale,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "glm router select launch");
}

/* =========================================================================
 * GLM routed MoE kernels.
 *
 * pair swiglu: mid[token][slot][row] =
 *     silu(gate_row . x) * (up_row . x) * route_weight[token][slot]
 * down: out[token][row] = sum_slot(down_row . mid[token][slot])
 * Route weights are folded into mid, exactly as in metal/moe.metal.
 * One warp per output row.
 * ========================================================================= */

template <typename DEQ>
__global__ static void glm_moe_pair_swiglu_kernel(
        float *mid,
        const char *gate,
        const char *up,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint32_t in_dim,
        uint32_t mid_dim,
        uint32_t n_total_expert,
        uint32_t n_expert_used,
        uint32_t n_tokens,
        uint32_t mid_token_stride,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t slot = blockIdx.y;
    const uint32_t token = blockIdx.z;
    if (row >= mid_dim || slot >= n_expert_used || token >= n_tokens) return;

    const uint64_t selected_off = (uint64_t)token * n_expert_used + slot;
    const uint64_t mid_off = (uint64_t)token * mid_token_stride +
                             (uint64_t)slot * mid_dim + row;
    const int expert = selected[selected_off];
    if (expert < 0 || (uint32_t)expert >= n_total_expert) {
        if (lane == 0) mid[mid_off] = 0.0f;
        return;
    }

    const char *gate_row = gate +
        (uint64_t)(uint32_t)expert * gate_expert_bytes +
        (uint64_t)row * gate_row_bytes;
    const char *up_row = up +
        (uint64_t)(uint32_t)expert * up_expert_bytes +
        (uint64_t)row * up_row_bytes;
    const float *token_x = x + (uint64_t)token * in_dim;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint32_t k = lane; k < in_dim; k += 32u) {
        const float xv = token_x[k];
        acc_gate += DEQ::value(gate_row, k) * xv;
        acc_up += DEQ::value(up_row, k) * xv;
    }
    #pragma unroll
    for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
        acc_gate += __shfl_xor_sync(0xffffffffu, acc_gate, mask);
        acc_up += __shfl_xor_sync(0xffffffffu, acc_up, mask);
    }
    if (lane == 0) {
        const float g = acc_gate;
        const float sw = g / (1.0f + expf(-g));
        mid[mid_off] = sw * acc_up * weights[selected_off];
    }
}

template <typename DEQ>
__global__ static void glm_moe_down_kernel(
        float *out,
        const char *down,
        const int32_t *selected,
        const float *mid,
        uint32_t mid_dim,
        uint32_t out_dim,
        uint32_t n_total_expert,
        uint32_t n_expert_used,
        uint32_t n_tokens,
        uint32_t mid_token_stride,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t token = blockIdx.y;
    if (row >= out_dim || token >= n_tokens) return;

    const uint64_t selected_base = (uint64_t)token * n_expert_used;
    const uint64_t mid_base = (uint64_t)token * mid_token_stride;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) continue;
        const char *down_row = down +
            (uint64_t)(uint32_t)expert * down_expert_bytes +
            (uint64_t)row * down_row_bytes;
        const float *slot_mid = mid + mid_base + (uint64_t)slot * mid_dim;
        for (uint32_t k = lane; k < mid_dim; k += 32u) {
            acc += DEQ::value(down_row, k) * slot_mid[k];
        }
    }
    #pragma unroll
    for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
        acc += __shfl_xor_sync(0xffffffffu, acc, mask);
    }
    if (lane == 0) {
        out[(uint64_t)token * out_dim + row] = acc;
    }
}

static bool glm_cuda_gate_pair_type_supported(uint32_t gate_type, uint32_t up_type) {
    return gate_type == up_type &&
           (gate_type == GLM_CUDA_TENSOR_Q2_K ||
            gate_type == GLM_CUDA_TENSOR_Q4_K ||
            gate_type == GLM_CUDA_TENSOR_Q5_K ||
            gate_type == GLM_CUDA_TENSOR_IQ2_XXS);
}

static bool glm_cuda_down_type_supported(uint32_t down_type) {
    return down_type == GLM_CUDA_TENSOR_IQ2_XXS ||
           down_type == GLM_CUDA_TENSOR_Q2_K ||
           down_type == GLM_CUDA_TENSOR_Q4_K ||
           down_type == GLM_CUDA_TENSOR_Q5_K ||
           down_type == GLM_CUDA_TENSOR_Q6_K;
}

/* =========================================================================
 * Streaming full-layer expert cache (GLM prefill).  Two ping-pong slots so
 * the next layer can load while the current one computes; port of the ROCm
 * design in rocm/ds4_rocm_runtime.cuh rebuilt on ds4_cuda.cu's streamed copy.
 * ========================================================================= */

struct glm_stream_layer_expert_cache {
    int active;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t capacity;
    char *base;
    char *gate;
    char *up;
    char *down;
};

static glm_stream_layer_expert_cache g_glm_stream_layer_expert_cache[2];

static void glm_stream_layer_expert_cache_release(void) {
    for (uint32_t i = 0; i < 2u; i++) {
        glm_stream_layer_expert_cache &c = g_glm_stream_layer_expert_cache[i];
        if (c.base) (void)cudaFree(c.base);
        memset(&c, 0, sizeof(c));
    }
}

static int glm_stream_layer_expert_cache_apply(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    if (!g_ssd_streaming_mode || !gate_w || !up_w || !down_w) return 0;
    for (uint32_t i = 0; i < 2u; i++) {
        const glm_stream_layer_expert_cache &c = g_glm_stream_layer_expert_cache[i];
        if (c.active &&
            c.model_map == model_map &&
            c.layer == layer &&
            c.n_total_expert == n_total_expert &&
            c.gate_offset == gate_offset &&
            c.up_offset == up_offset &&
            c.down_offset == down_offset &&
            c.gate_expert_bytes == gate_expert_bytes &&
            c.down_expert_bytes == down_expert_bytes &&
            c.gate && c.up && c.down) {
            *gate_w = c.gate;
            *up_w = c.up;
            *down_w = c.down;
            return 1;
        }
    }
    return 0;
}

extern "C" int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table) {
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    if (!g_ssd_streaming_mode || !model_map || model_size == 0 ||
        n_total_expert == 0 ||
        gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > UINT64_MAX / 2u ||
        2u * gate_bytes > UINT64_MAX - down_bytes) {
        return 0;
    }
    const uint64_t total_bytes = 2u * gate_bytes + down_bytes;
    if (gate_offset > model_size || up_offset > model_size ||
        down_offset > model_size ||
        gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }

    glm_stream_layer_expert_cache &slot =
        g_glm_stream_layer_expert_cache[layer & 1u];
    slot.active = 0;
    if (slot.capacity < total_bytes) {
        if (slot.base) {
            (void)cudaFree(slot.base);
            memset(&slot, 0, sizeof(slot));
        }
        void *base = NULL;
        cudaError_t err = cudaMalloc(&base, (size_t)total_bytes);
        if (err != cudaSuccess) {
            /* Try again after releasing the per-expert resident cache. */
            (void)cudaGetLastError();
            cuda_stream_expert_cache_release_all();
            err = cudaMalloc(&base, (size_t)total_bytes);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming full-layer expert cache allocation "
                    "failed for layer=%u (%.2f GiB): %s\n",
                    layer,
                    (double)total_bytes / 1073741824.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        slot.base = (char *)base;
        slot.capacity = total_bytes;
    }

    slot.gate = slot.base;
    slot.up = slot.base + gate_bytes;
    slot.down = slot.base + 2u * gate_bytes;
    if (!cuda_model_copy_to_device_streamed(slot.gate, model_map, model_size,
                                            gate_offset, gate_bytes,
                                            "layer moe_gate") ||
        !cuda_model_copy_to_device_streamed(slot.up, model_map, model_size,
                                            up_offset, gate_bytes,
                                            "layer moe_up") ||
        !cuda_model_copy_to_device_streamed(slot.down, model_map, model_size,
                                            down_offset, down_bytes,
                                            "layer moe_down")) {
        return 0;
    }

    slot.active = 1;
    slot.model_map = model_map;
    slot.layer = layer;
    slot.n_total_expert = n_total_expert;
    slot.gate_offset = gate_offset;
    slot.up_offset = up_offset;
    slot.down_offset = down_offset;
    slot.gate_expert_bytes = gate_expert_bytes;
    slot.down_expert_bytes = down_expert_bytes;
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_release_layer_cache(void) {
    glm_stream_layer_expert_cache_release();
    return 1;
}

/* Seeds the per-expert resident streaming cache from a resident full-layer
 * cache using the LAST n_seed_tokens tokens of the batch-selected ids
 * (device tensor, n_tokens x n_selected int32), so decode starts warm. */
extern "C" int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_seed_tokens,
        uint32_t n_selected) {
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    if (!g_ssd_streaming_mode || !model_map || !selected || !selected->ptr ||
        n_tokens == 0 || n_seed_tokens == 0 ||
        n_total_expert == 0 || n_selected == 0 ||
        gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        (uint64_t)n_tokens > UINT64_MAX / n_selected ||
        selected->bytes < (uint64_t)n_tokens * n_selected * sizeof(int32_t)) {
        return 0;
    }

    const char *layer_gate = NULL;
    const char *layer_up = NULL;
    const char *layer_down = NULL;
    if (!glm_stream_layer_expert_cache_apply(model_map, layer, n_total_expert,
                                             gate_offset, up_offset, down_offset,
                                             gate_expert_bytes, down_expert_bytes,
                                             &layer_gate, &layer_up, &layer_down)) {
        return 0;
    }

    if (n_seed_tokens > n_tokens) n_seed_tokens = n_tokens;
    const uint64_t n_ids = (uint64_t)n_seed_tokens * n_selected;
    std::vector<int32_t> ids((size_t)n_ids);
    const uint64_t src_off =
        (uint64_t)(n_tokens - n_seed_tokens) * n_selected * sizeof(int32_t);
    if (!cuda_ok(cudaMemcpy(ids.data(),
                            (const char *)selected->ptr + src_off,
                            (size_t)n_ids * sizeof(int32_t),
                            cudaMemcpyDeviceToHost),
                 "glm layer seed selected ids copy")) {
        return 0;
    }

    std::vector<uint8_t> seen(n_total_expert, 0);
    std::vector<int32_t> unique;
    unique.reserve((size_t)n_ids);
    for (uint64_t i = 0; i < n_ids; i++) {
        const int32_t expert = ids[(size_t)i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming full-layer seed expert id %d "
                    "outside 0..%u (layer=%u)\n",
                    expert, n_total_expert, layer);
            return 0;
        }
        if (seen[(uint32_t)expert]) continue;
        seen[(uint32_t)expert] = 1;
        unique.push_back(expert);
    }
    if (unique.empty()) return 1;

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         0);
    if (!cache || !cache->valid) return 0;

    for (size_t u = 0; u < unique.size(); u++) {
        const uint32_t expert = (uint32_t)unique[u];
        int cache_slot = cuda_stream_expert_cache_find(cache,
                                                       model_map,
                                                       model_size,
                                                       layer,
                                                       n_total_expert,
                                                       expert,
                                                       gate_offset,
                                                       up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes);
        if (cache_slot >= 0) {
            cache->slots[(uint32_t)cache_slot].age = ++cache->tick;
            continue;
        }
        const uint32_t load_slot = cuda_stream_expert_cache_lru_slot(cache);
        const int append = !cache->slots[load_slot].valid;
        const uint64_t gate_src = (uint64_t)expert * gate_expert_bytes;
        const uint64_t down_src = (uint64_t)expert * down_expert_bytes;
        const uint64_t gate_dst = (uint64_t)load_slot * gate_expert_bytes;
        const uint64_t down_dst = (uint64_t)load_slot * down_expert_bytes;
        if (!cuda_ok(cudaMemcpy(cache->gate_ptr + gate_dst,
                                layer_gate + gate_src,
                                (size_t)gate_expert_bytes,
                                cudaMemcpyDeviceToDevice),
                     "glm layer seed gate copy") ||
            !cuda_ok(cudaMemcpy(cache->up_ptr + gate_dst,
                                layer_up + gate_src,
                                (size_t)gate_expert_bytes,
                                cudaMemcpyDeviceToDevice),
                     "glm layer seed up copy") ||
            !cuda_ok(cudaMemcpy(cache->down_ptr + down_dst,
                                layer_down + down_src,
                                (size_t)down_expert_bytes,
                                cudaMemcpyDeviceToDevice),
                     "glm layer seed down copy")) {
            return 0;
        }
        cuda_stream_expert_cache_slot &entry = cache->slots[load_slot];
        entry.valid = 1;
        entry.model_map = model_map;
        entry.model_size = model_size;
        entry.layer = layer;
        entry.n_total_expert = n_total_expert;
        entry.expert = expert;
        entry.gate_offset = gate_offset;
        entry.up_offset = up_offset;
        entry.down_offset = down_offset;
        entry.gate_expert_bytes = gate_expert_bytes;
        entry.down_expert_bytes = down_expert_bytes;
        entry.age = ++cache->tick;
        if (append && cache->count < cache->capacity) cache->count++;
    }
    return 1;
}

/* Device-tensor variant of the early selected-expert load: reads the router's
 * selected ids back to the host and delegates to the existing host-id loader
 * that fills g_stream_selected_cache.  Port of the Metal launcher at
 * ds4_metal.m:12595 (no command-batch juggling needed on CUDA: the stream is
 * in-order and cudaMemcpy synchronizes). */
extern "C" int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor *selected,
        uint32_t n_selected) {
    if (!g_ssd_streaming_mode ||
        getenv("DS4_CUDA_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL ||
        getenv("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL) {
        return 1;
    }
    if (!table || !selected || n_selected == 0 || n_selected > 8u) {
        return 1;
    }
    /* Full layer already resident: nothing to load. */
    const char *g = NULL, *u = NULL, *d = NULL;
    if (glm_stream_layer_expert_cache_apply(table->model_map,
                                            table->layer,
                                            table->n_total_expert,
                                            table->gate_offset,
                                            table->up_offset,
                                            table->down_offset,
                                            table->gate_expert_bytes,
                                            table->down_expert_bytes,
                                            &g, &u, &d)) {
        return 1;
    }

    int32_t selected_ids[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if (!ds4_gpu_tensor_read(selected, 0, selected_ids,
                             (uint64_t)n_selected * sizeof(selected_ids[0]))) {
        return 0;
    }
    return ds4_gpu_stream_expert_cache_begin_selected_load(table,
                                                           selected_ids,
                                                           n_selected);
}

/* =========================================================================
 * GLM routed MoE launcher (decode + batch).
 *
 * Note vs the Metal launcher (ds4_metal.m:28085): the Metal-side streaming
 * bookkeeping (note_decode_token at layer 3, selected-hotness notes,
 * per-layer/global pruning, split resident/missing dispatch) belongs to the
 * Metal per-expert buffer cache and has no CUDA counterpart; on CUDA the
 * equivalent residency management lives inside
 * ds4_gpu_stream_expert_cache_begin_selected_load / g_stream_expert_cache LRU.
 * ========================================================================= */

static int glm_routed_moe_launch(
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
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride,
        bool force_resident) {
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_tokens > 65535u ||
        n_total_expert == 0 || n_expert == 0 || n_expert > 256u ||
        n_expert > n_total_expert ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        up_expert_bytes == 0 || up_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (expert_in_dim % CUDA_QK_K) != 0 ||
        (expert_mid_dim % CUDA_QK_K) != 0 ||
        !glm_cuda_gate_pair_type_supported(gate_type, up_type) ||
        !glm_cuda_down_type_supported(down_type)) {
        return 0;
    }
    const uint64_t per_token_mid = (uint64_t)n_expert * expert_mid_dim;
    if ((uint64_t)mid_token_stride < per_token_mid ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / up_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        fprintf(stderr, "ds4: CUDA GLM routed MoE received inconsistent expert strides\n");
        return 0;
    }
    const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t up_tensor_bytes = (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || up_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: CUDA GLM routed MoE tensor range is outside the mapped model\n");
        return 0;
    }
    const uint64_t mid_values =
        (uint64_t)(n_tokens - 1u) * mid_token_stride + per_token_mid;
    const uint64_t slot_values = (uint64_t)n_tokens * n_expert;
    if (x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        mid->bytes < mid_values * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float) ||
        selected->bytes < slot_values * sizeof(int32_t) ||
        weights->bytes < slot_values * sizeof(float)) {
        fprintf(stderr, "ds4: CUDA GLM routed MoE received undersized activation buffers\n");
        return 0;
    }

    /* Weight source resolution:
     *   1. compact selected-expert streaming cache (this layer's routed
     *      experts already on device; selected ids remapped to compact slots)
     *   2. resident full-layer streaming cache (prefill)
     *   3. resident/mapped full expert tables. */
    const ds4_gpu_tensor *selected_exec = selected;
    const char *gate_w = NULL;
    const char *up_w = NULL;
    const char *down_w = NULL;
    const int streaming = g_ssd_streaming_mode && !force_resident;

    /* The shared streaming caches store one gate/up stride; GLM gate and up
     * always share type and dims so the strides match by construction. */
    const int uniform_gate_up = gate_expert_bytes == up_expert_bytes;

    int use_selected_cache = 0;
    for (int attempt = 0; attempt < 2 && !use_selected_cache; attempt++) {
        use_selected_cache =
            streaming && uniform_gate_up &&
            g_stream_selected_cache.valid &&
            g_stream_selected_cache.model_map == model_map &&
            g_stream_selected_cache.layer == layer_index &&
            g_stream_selected_cache.n_total_expert == n_total_expert &&
            g_stream_selected_cache.slot_count >= slot_values &&
            g_stream_selected_cache.gate_offset == gate_offset &&
            g_stream_selected_cache.up_offset == up_offset &&
            g_stream_selected_cache.down_offset == down_offset &&
            g_stream_selected_cache.gate_expert_bytes == gate_expert_bytes &&
            g_stream_selected_cache.down_expert_bytes == down_expert_bytes &&
            g_stream_selected_cache.gate_ptr &&
            g_stream_selected_cache.up_ptr &&
            g_stream_selected_cache.down_ptr &&
            g_stream_selected_cache.slot_selected_tensor.ptr &&
            g_stream_selected_cache.slot_selected_tensor.bytes >=
                slot_values * sizeof(int32_t);
        if (use_selected_cache || attempt == 1 || !streaming || !uniform_gate_up) break;
        /* Safety net: decode reached the MoE without an early selected load
         * (e.g. the cache was invalidated).  Load the selected experts now
         * instead of pulling the full expert tables through the map. */
        const char *lg = NULL, *lu = NULL, *ld = NULL;
        if (n_tokens != 1u || n_expert > 8u ||
            glm_stream_layer_expert_cache_apply(model_map, layer_index,
                                                n_total_expert,
                                                gate_offset, up_offset,
                                                down_offset,
                                                gate_expert_bytes,
                                                down_expert_bytes,
                                                &lg, &lu, &ld)) {
            break;
        }
        int32_t ids[8];
        if (!ds4_gpu_tensor_read(selected, 0, ids,
                                 (uint64_t)n_expert * sizeof(ids[0]))) {
            return 0;
        }
        const ds4_gpu_stream_expert_table table = {
            model_map, model_size, layer_index, n_total_expert,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
        };
        if (!ds4_gpu_stream_expert_cache_begin_selected_load(&table, ids, n_expert)) {
            return 0;
        }
    }

    if (use_selected_cache) {
        selected_exec = &g_stream_selected_cache.slot_selected_tensor;
        gate_w = g_stream_selected_cache.gate_ptr;
        up_w = g_stream_selected_cache.up_ptr;
        down_w = g_stream_selected_cache.down_ptr;
    } else if (streaming && uniform_gate_up &&
               glm_stream_layer_expert_cache_apply(model_map, layer_index,
                                                   n_total_expert,
                                                   gate_offset, up_offset,
                                                   down_offset,
                                                   gate_expert_bytes,
                                                   down_expert_bytes,
                                                   &gate_w, &up_w, &down_w)) {
        /* full-layer cache hit; original selected ids remain valid */
    } else {
        gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_tensor_bytes,
                                      "glm_moe_gate");
        up_w = cuda_model_range_ptr(model_map, up_offset, up_tensor_bytes,
                                    "glm_moe_up");
        down_w = cuda_model_range_ptr(model_map, down_offset, down_tensor_bytes,
                                      "glm_moe_down");
    }
    if (!gate_w || !up_w || !down_w) return 0;

    const uint32_t rows_per_block = 4u;
    dim3 block(32, rows_per_block, 1);
    dim3 pair_grid((expert_mid_dim + rows_per_block - 1u) / rows_per_block,
                   n_expert,
                   n_tokens);
    dim3 down_grid((out_dim + rows_per_block - 1u) / rows_per_block,
                   n_tokens,
                   1);
    const int32_t *sel_ptr = (const int32_t *)selected_exec->ptr;
    const float *w_ptr = (const float *)weights->ptr;
    const float *x_ptr = (const float *)x->ptr;
    float *mid_ptr = (float *)mid->ptr;
    float *out_ptr = (float *)out->ptr;

    switch (gate_type) {
    case GLM_CUDA_TENSOR_IQ2_XXS:
        glm_moe_pair_swiglu_kernel<glm_deq_iq2_xxs><<<pair_grid, block>>>(
                mid_ptr, gate_w, up_w, x_ptr, sel_ptr, w_ptr,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q2_K:
        glm_moe_pair_swiglu_kernel<glm_deq_q2_K><<<pair_grid, block>>>(
                mid_ptr, gate_w, up_w, x_ptr, sel_ptr, w_ptr,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q4_K:
        glm_moe_pair_swiglu_kernel<glm_deq_q4_K><<<pair_grid, block>>>(
                mid_ptr, gate_w, up_w, x_ptr, sel_ptr, w_ptr,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes);
        break;
    default: /* Q5_K, validated above */
        glm_moe_pair_swiglu_kernel<glm_deq_q5_K><<<pair_grid, block>>>(
                mid_ptr, gate_w, up_w, x_ptr, sel_ptr, w_ptr,
                expert_in_dim, expert_mid_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes);
        break;
    }
    if (!cuda_ok(cudaGetLastError(), "glm routed moe pair launch")) return 0;

    switch (down_type) {
    case GLM_CUDA_TENSOR_IQ2_XXS:
        glm_moe_down_kernel<glm_deq_iq2_xxs><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q2_K:
        glm_moe_down_kernel<glm_deq_q2_K><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q4_K:
        glm_moe_down_kernel<glm_deq_q4_K><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    case GLM_CUDA_TENSOR_Q5_K:
        glm_moe_down_kernel<glm_deq_q5_K><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    default: /* Q6_K, validated above */
        glm_moe_down_kernel<glm_deq_q6_K><<<down_grid, block>>>(
                out_ptr, down_w, sel_ptr, mid_ptr,
                expert_mid_dim, out_dim, n_total_expert, n_expert,
                n_tokens, mid_token_stride, down_expert_bytes, down_row_bytes);
        break;
    }
    return cuda_ok(cudaGetLastError(), "glm routed moe down launch");
}

/* =========================================================================
 * extern "C" backend entry points.
 * ========================================================================= */

extern "C" int ds4_gpu_tensor_copy_f32_to_f16(
        ds4_gpu_tensor *dst,
        uint64_t dst_offset,
        const ds4_gpu_tensor *src,
        uint64_t src_offset,
        uint64_t count) {
    if (!dst || !src || !dst->ptr || !src->ptr) return 0;
    if (count == 0) return 1;
    if (count > UINT64_MAX / sizeof(float) ||
        count > UINT64_MAX / sizeof(uint16_t)) {
        return 0;
    }
    const uint64_t src_bytes = count * sizeof(float);
    const uint64_t dst_bytes = count * sizeof(uint16_t);
    if (src_offset > src->bytes || src_bytes > src->bytes - src_offset ||
        dst_offset > dst->bytes || dst_bytes > dst->bytes - dst_offset) {
        return 0;
    }
    f32_to_f16_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>(
            (__half *)((char *)dst->ptr + dst_offset),
            (const float *)((const char *)src->ptr + src_offset),
            count);
    return cuda_ok(cudaGetLastError(), "tensor copy f32 to f16 launch");
}

/* On Metal this closes the current compute encoder so later blit work can be
 * encoded.  The CUDA stream is in-order with no encoder concept: no-op. */
extern "C" int ds4_gpu_flush_encoder(void) {
    return 1;
}

extern "C" void ds4_gpu_release_q8_f16_cache(void) {
    cuda_q8_f16_cache_release_all();
}

extern "C" int ds4_gpu_add3_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t n) {
    if (!out || !a || !b || !c || n == 0 ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float) ||
        c->bytes < (uint64_t)n * sizeof(float)) {
        return 0;
    }
    glm_add3_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out->ptr, (const float *)a->ptr,
            (const float *)b->ptr, (const float *)c->ptr, n);
    return cuda_ok(cudaGetLastError(), "add3 launch");
}

extern "C" int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor *norm_out,
        ds4_gpu_tensor *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n,
        float eps) {
    if (!norm_out || !sum_out || !a || !b || !model_map || n == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n * sizeof(float);
    if (norm_out->bytes < row_bytes || sum_out->bytes < row_bytes ||
        a->bytes < row_bytes || b->bytes < row_bytes ||
        weight_offset > model_size || row_bytes > model_size - weight_offset) {
        return 0;
    }
    const float *w = (const float *)cuda_model_range_ptr(model_map,
            weight_offset, row_bytes, "add_rms_weight");
    if (!w) return 0;
    glm_add_rms_norm_weight_kernel<<<1, 256>>>(
            (float *)norm_out->ptr,
            (float *)sum_out->ptr,
            (const float *)a->ptr,
            (const float *)b->ptr,
            w, n, eps);
    return cuda_ok(cudaGetLastError(), "add rms_norm_weight launch");
}

extern "C" int ds4_gpu_sort_i32_rows_asc_tensor(
        ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src,
        uint32_t row_width,
        uint32_t n_rows) {
    if (!dst || !src || row_width == 0 || n_rows == 0 ||
        (row_width & (row_width - 1u)) != 0) {
        return 0;
    }
    const uint64_t bytes = (uint64_t)row_width * n_rows * sizeof(int32_t);
    if (src->bytes < bytes || dst->bytes < bytes) {
        fprintf(stderr, "ds4: CUDA row sort received undersized buffers\n");
        return 0;
    }
    const uint32_t shared_bytes = row_width * (uint32_t)sizeof(int32_t);
    if (shared_bytes > 48u * 1024u) {
        fprintf(stderr, "ds4: CUDA row sort scratch exceeds shared memory limit\n");
        return 0;
    }
    uint32_t threads = row_width < 256u ? row_width : 256u;
    glm_sort_i32_rows_asc_kernel<<<n_rows, threads, shared_bytes>>>(
            (int32_t *)dst->ptr, (const int32_t *)src->ptr, row_width, n_rows);
    return cuda_ok(cudaGetLastError(), "sort i32 rows asc launch");
}

/* The Metal "decode mpp" and "model view" variants only change dispatch
 * strategy / buffer wrapping; the math is the plain Q8_0 matmul. */
extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset,
                                      in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset,
                                      in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        float clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate, up, mid,
                                                     model_map, model_size,
                                                     gate_offset, up_offset,
                                                     in_dim, out_dim, x, clamp);
}

/* Mid-only fused shared expert: the Metal impl runs a fused kernel with no
 * gate/up outputs.  Here we keep two small persistent scratch buffers and
 * delegate to the existing pair-matmul + swiglu path. */
extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        float clamp) {
    static ds4_gpu_tensor scratch_gate = {NULL, 0, 0};
    static ds4_gpu_tensor scratch_up = {NULL, 0, 0};
    if (!mid || out_dim == 0 || out_dim > UINT64_MAX / sizeof(float)) return 0;
    const uint64_t bytes = out_dim * sizeof(float);
    if (scratch_gate.bytes < bytes) {
        if (scratch_gate.ptr) (void)cudaFree(scratch_gate.ptr);
        if (scratch_up.ptr) (void)cudaFree(scratch_up.ptr);
        scratch_gate.ptr = NULL;
        scratch_up.ptr = NULL;
        scratch_gate.bytes = 0;
        scratch_up.bytes = 0;
        void *g = NULL;
        void *u = NULL;
        if (!cuda_ok(cudaMalloc(&g, (size_t)bytes), "shared mid gate scratch") ||
            !cuda_ok(cudaMalloc(&u, (size_t)bytes), "shared mid up scratch")) {
            if (g) (void)cudaFree(g);
            return 0;
        }
        scratch_gate.ptr = g;
        scratch_gate.bytes = bytes;
        scratch_up.ptr = u;
        scratch_up.bytes = bytes;
    }
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&scratch_gate, &scratch_up,
                                                     mid,
                                                     model_map, model_size,
                                                     gate_offset, up_offset,
                                                     in_dim, out_dim, x, clamp);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        float clamp) {
    if (!gate || !up || !mid || !x || !model_map || n_tok == 0 ||
        out_dim == 0 || n_tok > UINT32_MAX ||
        out_dim > UINT32_MAX || n_tok * out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (gate->bytes < out_bytes || up->bytes < out_bytes || mid->bytes < out_bytes) {
        return 0;
    }
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up, model_map, model_size,
                                           gate_offset, up_offset,
                                           in_dim, out_dim, out_dim,
                                           x, n_tok) &&
           ds4_gpu_swiglu_tensor(mid, gate, up,
                                 (uint32_t)(n_tok * out_dim), clamp, 1.0f);
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
    return glm_router_select_launch(selected, weights, probs,
                                    model_map, model_size, bias_offset,
                                    logits, n_expert, n_expert_used,
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
    return glm_router_select_launch(selected, weights, probs,
                                    model_map, model_size, bias_offset,
                                    logits, n_expert, n_expert_used,
                                    expert_weight_scale, n_tokens);
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
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        bool force_resident) {
    if (n_expert == 0 || (uint64_t)n_expert * expert_mid_dim > UINT32_MAX) return 0;
    return glm_routed_moe_launch(out, mid, model_map, model_size,
                                 gate_offset, up_offset, down_offset,
                                 gate_type, up_type, down_type,
                                 gate_expert_bytes, gate_row_bytes,
                                 up_expert_bytes, up_row_bytes,
                                 down_expert_bytes, down_row_bytes,
                                 expert_in_dim, expert_mid_dim, out_dim,
                                 selected, weights,
                                 n_total_expert, n_expert, layer_index, x,
                                 1u,
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
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride) {
    return glm_routed_moe_launch(out, mid, model_map, model_size,
                                 gate_offset, up_offset, down_offset,
                                 gate_type, up_type, down_type,
                                 gate_expert_bytes, gate_row_bytes,
                                 up_expert_bytes, up_row_bytes,
                                 down_expert_bytes, down_row_bytes,
                                 expert_in_dim, expert_mid_dim, out_dim,
                                 selected, weights,
                                 n_total_expert, n_expert, layer_index, x,
                                 n_tokens, mid_token_stride, false);
}

/* On Metal "direct scalar q4" only picks different pipelines for the same
 * math (ds4_metal.m routes both through the shared batch impl); on CUDA the
 * batch launcher is already the scalar path. */
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
        uint32_t layer_index,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t mid_token_stride) {
    return glm_routed_moe_launch(out, mid, model_map, model_size,
                                 gate_offset, up_offset, down_offset,
                                 gate_type, up_type, down_type,
                                 gate_expert_bytes, gate_row_bytes,
                                 up_expert_bytes, up_row_bytes,
                                 down_expert_bytes, down_row_bytes,
                                 expert_in_dim, expert_mid_dim, out_dim,
                                 selected, weights,
                                 n_total_expert, n_expert, layer_index, x,
                                 n_tokens, mid_token_stride, false);
}
