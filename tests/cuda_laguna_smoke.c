#include "ds4_gpu.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "cuda-laguna: FAIL: %s (line %d)\n", msg, __LINE__); \
        return 1; \
    } \
} while (0)

enum {
    QK_K = 256,
    HEAD_DIM = 128,
};

typedef struct {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    uint16_t d;
} block_q3_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
} block_q6_K;

typedef struct {
    uint8_t *data;
    uint64_t size;
    uint64_t used;
} model_blob;

static uint64_t blob_alloc(model_blob *blob, uint64_t bytes) {
    const uint64_t offset = (blob->used + 255u) & ~255ull;
    if (offset > blob->size || bytes > blob->size - offset) return UINT64_MAX;
    blob->used = offset + bytes;
    return offset;
}

static void fill_q2_ones(void *ptr, uint64_t blocks) {
    block_q2_K *out = (block_q2_K *)ptr;
    for (uint64_t i = 0; i < blocks; i++) {
        memset(&out[i], 0, sizeof(out[i]));
        memset(out[i].scales, 0x01, sizeof(out[i].scales));
        memset(out[i].qs, 0x55, sizeof(out[i].qs));
        out[i].d = 0x3c00u;
    }
}

static void q3_pack_scales(uint8_t packed[12], const uint8_t values[16]) {
    memset(packed, 0, 12u);
    for (uint32_t i = 0u; i < 16u; i++) {
        const uint8_t value = values[i] & 63u;
        if (i < 8u) {
            packed[i] |= value & 15u;
        } else {
            packed[i - 8u] |= (value & 15u) << 4u;
        }
        if (i < 4u) {
            packed[i + 8u] |= (value >> 4u) & 3u;
        } else if (i < 8u) {
            packed[i + 4u] |= ((value >> 4u) & 3u) << 2u;
        } else if (i < 12u) {
            packed[i] |= ((value >> 4u) & 3u) << 4u;
        } else {
            packed[i - 4u] |= ((value >> 4u) & 3u) << 6u;
        }
    }
}

static void fill_q3_pattern(void *ptr, uint64_t blocks) {
    block_q3_K *out = (block_q3_K *)ptr;
    uint8_t scale_values[16];
    for (uint32_t i = 0u; i < 16u; i++) {
        scale_values[i] = (uint8_t)(25u + (7u * i) % 17u);
    }
    for (uint64_t i = 0; i < blocks; i++) {
        memset(&out[i], 0, sizeof(out[i]));
        for (uint32_t j = 0u; j < sizeof(out[i].hmask); j++) {
            out[i].hmask[j] = (uint8_t)(0x5au ^ (j * 13u));
        }
        for (uint32_t j = 0u; j < sizeof(out[i].qs); j++) {
            out[i].qs[j] = (uint8_t)(0xe4u ^ (j * 29u));
        }
        q3_pack_scales(out[i].scales, scale_values);
        out[i].d = 0x2c00u;
    }
}

static void fill_q4_ones(void *ptr, uint64_t blocks) {
    block_q4_K *out = (block_q4_K *)ptr;
    for (uint64_t i = 0; i < blocks; i++) {
        memset(&out[i], 0, sizeof(out[i]));
        out[i].d = 0x3c00u;
        for (uint32_t j = 0; j < 4u; j++) out[i].scales[j] = 1u;
        for (uint32_t j = 8u; j < 12u; j++) out[i].scales[j] = 1u;
        memset(out[i].qs, 0x11, sizeof(out[i].qs));
    }
}

static void fill_q6_ones(void *ptr, uint64_t blocks) {
    block_q6_K *out = (block_q6_K *)ptr;
    for (uint64_t i = 0; i < blocks; i++) {
        memset(&out[i], 0, sizeof(out[i]));
        memset(out[i].ql, 0x11, sizeof(out[i].ql));
        memset(out[i].qh, 0xaa, sizeof(out[i].qh));
        memset(out[i].scales, 1, sizeof(out[i].scales));
        out[i].d = 0x3c00u;
    }
}

static void fill_q4_pattern(void *ptr, uint64_t blocks) {
    block_q4_K *out = (block_q4_K *)ptr;
    for (uint64_t i = 0; i < blocks; i++) {
        out[i].d = 0x3800u;
        out[i].dmin = 0x3400u;
        for (uint32_t j = 0; j < sizeof(out[i].scales); j++) {
            out[i].scales[j] = (uint8_t)(3u + 17u * j + 11u * i);
        }
        for (uint32_t j = 0; j < sizeof(out[i].qs); j++) {
            out[i].qs[j] = (uint8_t)(13u * j + 29u * i + 7u);
        }
    }
}

static void fill_q6_pattern(void *ptr, uint64_t blocks) {
    block_q6_K *out = (block_q6_K *)ptr;
    for (uint64_t i = 0; i < blocks; i++) {
        out[i].d = 0x3000u;
        for (uint32_t j = 0; j < sizeof(out[i].ql); j++) {
            out[i].ql[j] = (uint8_t)(19u * j + 23u * i + 5u);
        }
        for (uint32_t j = 0; j < sizeof(out[i].qh); j++) {
            out[i].qh[j] = (uint8_t)(11u * j + 31u * i + 3u);
        }
        for (uint32_t j = 0; j < sizeof(out[i].scales); j++) {
            out[i].scales[j] = (int8_t)((int32_t)((7u * j + 5u * i) % 15u) -
                                        7);
        }
    }
}

static float f16_ref(uint16_t value) {
    const int sign = (value & 0x8000u) ? -1 : 1;
    const uint32_t exponent = (value >> 10u) & 31u;
    const uint32_t mantissa = value & 1023u;
    if (exponent == 0u) {
        return sign * ldexpf((float)mantissa, -24);
    }
    if (exponent == 31u) {
        return mantissa ? NAN : sign * INFINITY;
    }
    return sign * ldexpf((float)(1024u + mantissa),
                         (int)exponent - 25);
}

static float q4_value_ref(const block_q4_K *block, uint32_t k) {
    const uint32_t group = k >> 5u;
    const uint32_t lane = k & 31u;
    uint8_t scale;
    uint8_t min_scale;
    if (group < 4u) {
        scale = block->scales[group] & 63u;
        min_scale = block->scales[group + 4u] & 63u;
    } else {
        scale = (block->scales[group + 4u] & 15u) |
                ((block->scales[group - 4u] >> 6u) << 4u);
        min_scale = (block->scales[group + 4u] >> 4u) |
                    ((block->scales[group] >> 6u) << 4u);
    }
    const uint32_t byte_off = (group >> 1u) * 32u + lane;
    const uint32_t q = (block->qs[byte_off] >> ((group & 1u) * 4u)) & 15u;
    return f16_ref(block->d) * (float)scale * (float)q -
           f16_ref(block->dmin) * (float)min_scale;
}

static float q6_value_ref(const block_q6_K *block, uint32_t k) {
    const uint32_t half = k >> 7u;
    const uint32_t within = k & 127u;
    const uint32_t lane = within & 31u;
    const uint32_t quarter = within >> 5u;
    const uint32_t ql_base = half * 64u;
    const uint32_t qh_base = half * 32u;
    const uint32_t scale_base = half * 8u;
    uint32_t q;
    int32_t scale;
    if (quarter == 0u) {
        q = (block->ql[ql_base + lane] & 15u) |
            (((block->qh[qh_base + lane] >> 0u) & 3u) << 4u);
        scale = block->scales[scale_base + lane / 16u];
    } else if (quarter == 1u) {
        q = (block->ql[ql_base + 32u + lane] & 15u) |
            (((block->qh[qh_base + lane] >> 2u) & 3u) << 4u);
        scale = block->scales[scale_base + lane / 16u + 2u];
    } else if (quarter == 2u) {
        q = (block->ql[ql_base + lane] >> 4u) |
            (((block->qh[qh_base + lane] >> 4u) & 3u) << 4u);
        scale = block->scales[scale_base + lane / 16u + 4u];
    } else {
        q = (block->ql[ql_base + 32u + lane] >> 4u) |
            (((block->qh[qh_base + lane] >> 6u) & 3u) << 4u);
        scale = block->scales[scale_base + lane / 16u + 6u];
    }
    return f16_ref(block->d) * (float)scale * (float)((int32_t)q - 32);
}

static float q3_value_ref(const block_q3_K *block, uint32_t k) {
    const uint32_t half = k >> 7u;
    const uint32_t within = k & 127u;
    const uint32_t plane = within >> 5u;
    const uint32_t lane = within & 31u;
    const uint32_t group =
        half * 8u + plane * 2u + (lane >= 16u ? 1u : 0u);
    uint32_t scale;
    if (group < 4u) {
        scale = (block->scales[group] & 15u) |
                (((block->scales[group + 8u] >> 0u) & 3u) << 4u);
    } else if (group < 8u) {
        scale = (block->scales[group] & 15u) |
                (((block->scales[group + 4u] >> 2u) & 3u) << 4u);
    } else if (group < 12u) {
        scale = (block->scales[group - 8u] >> 4u) |
                (((block->scales[group] >> 4u) & 3u) << 4u);
    } else {
        scale = (block->scales[group - 8u] >> 4u) |
                (((block->scales[group - 4u] >> 6u) & 3u) << 4u);
    }
    const uint32_t q_index = half * 32u + lane;
    const uint32_t low =
        (block->qs[q_index] >> (2u * plane)) & 3u;
    const uint8_t high_mask = (uint8_t)(1u << (4u * half + plane));
    const int32_t q = (int32_t)low -
        ((block->hmask[lane] & high_mask) ? 0 : 4);
    return f16_ref(block->d) * ((float)scale - 32.0f) * (float)q;
}

static int close_enough(float got, float expected, float atol, float rtol) {
    return isfinite(got) &&
           fabsf(got - expected) <= atol + rtol * fabsf(expected);
}

static int check_quant_ops(model_blob *blob,
                           uint64_t q4_offset,
                           uint64_t q6_offset,
                           uint64_t embed_offset) {
    const uint32_t in_dim = QK_K;
    const uint32_t out_dim = 2;
    const uint32_t n_tokens = 4;
    float x_host[n_tokens * in_dim];
    float out_host[n_tokens * out_dim];
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            x_host[(uint64_t)t * in_dim + i] = 0.25f * (float)(t + 1u);
        }
    }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_host));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(out_host));
    ds4_gpu_tensor *embed = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    CHECK(x && out && embed, "quant tensor allocation");
    CHECK(ds4_gpu_tensor_write(x, 0, x_host, sizeof(x_host)), "write quant x");
    CHECK(ds4_gpu_matmul_quant_tensor(out, blob->data, blob->size,
                                      q4_offset, 12u, in_dim, out_dim,
                                      x, n_tokens),
          "Q4_K batch matmul");
    CHECK(ds4_gpu_tensor_read(out, 0, out_host, sizeof(out_host)),
          "read Q4_K matmul");
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            const block_q4_K *row =
                (const block_q4_K *)(blob->data + q4_offset) + r;
            float expected = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                expected += q4_value_ref(row, i) *
                            x_host[(uint64_t)t * in_dim + i];
            }
            CHECK(close_enough(out_host[t * out_dim + r], expected,
                               0.02f, 2e-4f),
                  "Q4_K matmul numeric");
        }
    }
    CHECK(ds4_gpu_matmul_q6_K_tensor(out, blob->data, blob->size,
                                     q6_offset, in_dim, out_dim,
                                     x, n_tokens),
          "Q6_K batch matmul");
    CHECK(ds4_gpu_tensor_read(out, 0, out_host, sizeof(out_host)),
          "read Q6_K matmul");
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            const block_q6_K *row =
                (const block_q6_K *)(blob->data + q6_offset) + r;
            float expected = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                expected += q6_value_ref(row, i) *
                            x_host[(uint64_t)t * in_dim + i];
            }
            CHECK(close_enough(out_host[t * out_dim + r], expected,
                               0.02f, 2e-4f),
                  "Q6_K matmul numeric");
        }
    }
    CHECK(ds4_gpu_matmul_quant_tensor(out, blob->data, blob->size,
                                      q4_offset, 12u, in_dim, out_dim,
                                      x, 1u),
          "Q4_K decode matmul");
    CHECK(ds4_gpu_tensor_read(out, 0, out_host,
                              out_dim * sizeof(float)),
          "read Q4_K decode matmul");
    for (uint32_t r = 0; r < out_dim; r++) {
        const block_q4_K *row =
            (const block_q4_K *)(blob->data + q4_offset) + r;
        float expected = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            expected += q4_value_ref(row, i) * x_host[i];
        }
        CHECK(close_enough(out_host[r], expected, 0.02f, 2e-4f),
              "Q4_K decode matmul numeric");
    }
    CHECK(ds4_gpu_matmul_q6_K_tensor(out, blob->data, blob->size,
                                     q6_offset, in_dim, out_dim, x, 1u),
          "Q6_K decode matmul");
    CHECK(ds4_gpu_tensor_read(out, 0, out_host,
                              out_dim * sizeof(float)),
          "read Q6_K decode matmul");
    for (uint32_t r = 0; r < out_dim; r++) {
        const block_q6_K *row =
            (const block_q6_K *)(blob->data + q6_offset) + r;
        float expected = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            expected += q6_value_ref(row, i) * x_host[i];
        }
        CHECK(close_enough(out_host[r], expected, 0.02f, 2e-4f),
              "Q6_K decode matmul numeric");
    }
    CHECK(ds4_gpu_embed_token_quant_tensor(embed, blob->data, blob->size,
                                            embed_offset, 12u, 2u, 1u,
                                            in_dim),
          "Q4_K token embedding");
    float embed_host[in_dim];
    CHECK(ds4_gpu_tensor_read(embed, 0, embed_host, sizeof(embed_host)),
          "read Q4_K embedding");
    for (uint32_t i = 0; i < in_dim; i++) {
        CHECK(embed_host[i] == 1.0f, "Q4_K embedding numeric");
    }
    ds4_gpu_tensor_free(embed);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return 0;
}

static void norm_rope_ref(float *x,
                          const float *weight,
                          uint32_t n_tokens,
                          uint32_t n_head,
                          uint32_t n_rot,
                          uint32_t pos0,
                          uint32_t n_ctx_orig,
                          float freq_base,
                          float freq_scale,
                          float ext_factor,
                          float attn_factor,
                          float beta_fast,
                          float beta_slow,
                          float eps) {
    const uint32_t half_rot = n_rot / 2u;
    float corr0 = 0.0f;
    float corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = fmaxf(
            0.0f,
            floorf((float)n_rot *
                   logf((float)n_ctx_orig /
                        (beta_fast * 2.0f * (float)M_PI)) / denom));
        corr1 = fminf(
            (float)(n_rot - 1u),
            ceilf((float)n_rot *
                  logf((float)n_ctx_orig /
                       (beta_slow * 2.0f * (float)M_PI)) / denom));
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            float *row = x + ((uint64_t)t * n_head + h) * HEAD_DIM;
            float ss = 0.0f;
            for (uint32_t i = 0; i < HEAD_DIM; i++) ss += row[i] * row[i];
            const float inv = 1.0f / sqrtf(ss / HEAD_DIM + eps);
            for (uint32_t i = 0; i < HEAD_DIM; i++) {
                row[i] *= inv * weight[i];
            }
            for (uint32_t i = 0; i < half_rot; i++) {
                const uint32_t rel_i0 = 2u * i;
                const float theta_extrap = (float)(pos0 + t) *
                    powf(freq_base, -(float)(2u * i) / (float)n_rot);
                const float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float magnitude = attn_factor;
                if (ext_factor != 0.0f) {
                    const float ramp =
                        ((float)(rel_i0 / 2u) - corr0) /
                        fmaxf(0.001f, corr1 - corr0);
                    const float mix =
                        (1.0f - fminf(1.0f, fmaxf(0.0f, ramp))) *
                        ext_factor;
                    theta = theta_interp * (1.0f - mix) +
                            theta_extrap * mix;
                    magnitude *= 1.0f + 0.1f * logf(1.0f / freq_scale);
                }
                const float c = cosf(theta) * magnitude;
                const float s = sinf(theta) * magnitude;
                const float x0 = row[i];
                const float x1 = row[i + half_rot];
                row[i] = x0 * c - x1 * s;
                row[i + half_rot] = x0 * s + x1 * c;
            }
        }
    }
}

static int check_norm_rope(model_blob *blob, uint64_t norm_offset) {
    const uint32_t n_tokens = 2;
    const uint32_t n_head = 3;
    const uint32_t n_rot = 64;
    float input[n_tokens * n_head * HEAD_DIM];
    float expected[n_tokens * n_head * HEAD_DIM];
    float got[n_tokens * n_head * HEAD_DIM];
    float weight[HEAD_DIM];
    memcpy(weight, blob->data + norm_offset, sizeof(weight));
    for (uint32_t i = 0; i < n_tokens * n_head * HEAD_DIM; i++) {
        input[i] = ((int32_t)(i % 23u) - 11) * 0.03125f;
    }
    memcpy(expected, input, sizeof(input));
    const float freq_scale = 0.25f;
    const float ext_factor = 1.0f;
    const float attn_factor = 1.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;
    norm_rope_ref(expected, weight, n_tokens, n_head, n_rot,
                  17u, 8192u, 500000.0f, freq_scale, ext_factor,
                  attn_factor, beta_fast, beta_slow, 1e-6f);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    CHECK(x, "norm tensor allocation");
    CHECK(ds4_gpu_tensor_write(x, 0, input, sizeof(input)), "write norm input");
    CHECK(ds4_gpu_laguna_head_rms_norm_rope_tensor(
                  x, blob->data, blob->size, norm_offset,
                  n_tokens, n_head, HEAD_DIM, n_rot, 17u, 8192u,
                  500000.0f, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, 1e-6f),
          "Laguna head norm/RoPE");
    CHECK(ds4_gpu_tensor_read(x, 0, got, sizeof(got)), "read norm/RoPE");
    for (uint32_t i = 0; i < n_tokens * n_head * HEAD_DIM; i++) {
        CHECK(close_enough(got[i], expected[i], 4e-4f, 4e-4f),
              "Laguna norm/RoPE numeric");
    }
    memcpy(expected, input, sizeof(input));
    norm_rope_ref(expected, weight, n_tokens, n_head, n_rot,
                  17u, 8192u, 500000.0f, 1.0f, 0.0f,
                  1.0f, 0.0f, 0.0f, 1e-6f);
    CHECK(ds4_gpu_tensor_write(x, 0, input, sizeof(input)),
          "rewrite SWA norm input");
    CHECK(ds4_gpu_laguna_head_rms_norm_rope_tensor(
                  x, blob->data, blob->size, norm_offset,
                  n_tokens, n_head, HEAD_DIM, n_rot, 17u, 8192u,
                  500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1e-6f),
          "Laguna SWA head norm/RoPE");
    CHECK(ds4_gpu_tensor_read(x, 0, got, sizeof(got)),
          "read SWA norm/RoPE");
    for (uint32_t i = 0; i < n_tokens * n_head * HEAD_DIM; i++) {
        CHECK(close_enough(got[i], expected[i], 4e-4f, 4e-4f),
              "Laguna SWA norm/RoPE numeric");
    }
    ds4_gpu_tensor_free(x);
    return 0;
}

static float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static void attention_ref(float *out,
                          const float *q,
                          const float *keys,
                          const float *values,
                          const float *gate,
                          uint32_t pos0,
                          uint32_t n_tokens,
                          uint32_t cache_cap,
                          uint32_t n_head,
                          uint32_t n_head_kv) {
    const uint32_t heads_per_kv = n_head / n_head_kv;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            const uint32_t kh = h / heads_per_kv;
            const uint32_t query_pos = pos0 + t;
            const uint32_t key_count =
                query_pos + 1u < cache_cap ? query_pos + 1u : cache_cap;
            const uint32_t key_start = query_pos + 1u - key_count;
            float scores[8];
            float max_score = -INFINITY;
            for (uint32_t k = key_start; k <= query_pos; k++) {
                float score = 0.0f;
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    score += q[((uint64_t)t * n_head + h) * HEAD_DIM + d] *
                             keys[((uint64_t)k * n_head_kv + kh) * HEAD_DIM + d];
                }
                scores[k] = score / sqrtf((float)HEAD_DIM);
                if (scores[k] > max_score) max_score = scores[k];
            }
            float denom = 0.0f;
            for (uint32_t k = key_start; k <= query_pos; k++) {
                scores[k] = expf(scores[k] - max_score);
                denom += scores[k];
            }
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                float value = 0.0f;
                for (uint32_t k = key_start; k <= query_pos; k++) {
                    value += scores[k] *
                        values[((uint64_t)k * n_head_kv + kh) * HEAD_DIM + d];
                }
                out[((uint64_t)t * n_head + h) * HEAD_DIM + d] =
                    value / denom * softplus(gate[(uint64_t)t * n_head + h]);
            }
        }
    }
}

static int check_attention(void) {
    const uint32_t n_tokens = 3;
    const uint32_t n_head = 6;
    const uint32_t n_head_kv = 1;
    const uint32_t cache_cap = 4;
    const uint64_t q_values = (uint64_t)n_tokens * n_head * HEAD_DIM;
    const uint64_t kv_values = (uint64_t)n_tokens * n_head_kv * HEAD_DIM;
    float q[q_values], k[kv_values], v[kv_values], gate[n_tokens * n_head];
    float expected[q_values], got[q_values];
    for (uint64_t i = 0; i < q_values; i++) q[i] = 0.125f;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            k[(uint64_t)t * HEAD_DIM + d] = 0.125f * (float)(t + 1u);
            v[(uint64_t)t * HEAD_DIM + d] = (float)(t + 1u);
        }
        for (uint32_t h = 0; h < n_head; h++) {
            gate[t * n_head + h] = 0.1f * (float)h;
        }
    }
    attention_ref(expected, q, k, v, gate, 0u, n_tokens, cache_cap,
                  n_head, n_head_kv);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(sizeof(got));
    ds4_gpu_tensor *key_cache =
        ds4_gpu_tensor_alloc((uint64_t)cache_cap * HEAD_DIM * sizeof(uint16_t));
    ds4_gpu_tensor *value_cache =
        ds4_gpu_tensor_alloc((uint64_t)cache_cap * HEAD_DIM * sizeof(uint16_t));
    ds4_gpu_tensor *staged_key =
        ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
    ds4_gpu_tensor *staged_value =
        ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *k_t = ds4_gpu_tensor_alloc(sizeof(k));
    ds4_gpu_tensor *v_t = ds4_gpu_tensor_alloc(sizeof(v));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc(sizeof(gate));
    CHECK(heads && key_cache && value_cache && staged_key && staged_value &&
          q_t && k_t && v_t && gate_t, "attention tensor allocation");
    CHECK(ds4_gpu_tensor_write(q_t, 0, q, sizeof(q)) &&
          ds4_gpu_tensor_write(k_t, 0, k, sizeof(k)) &&
          ds4_gpu_tensor_write(v_t, 0, v, sizeof(v)) &&
          ds4_gpu_tensor_write(gate_t, 0, gate, sizeof(gate)),
          "write attention tensors");
    CHECK(ds4_gpu_laguna_attention_prefill_tensor(
                  heads, key_cache, value_cache, staged_key, staged_value,
                  q_t, k_t, v_t, gate_t, 0u, n_tokens, cache_cap,
                  n_head, n_head_kv, HEAD_DIM,
                  1.0f / sqrtf((float)HEAD_DIM)),
          "Laguna prefill attention");
    CHECK(ds4_gpu_tensor_read(heads, 0, got, sizeof(got)),
          "read prefill attention");
    for (uint64_t i = 0; i < q_values; i++) {
        CHECK(close_enough(got[i], expected[i], 5e-4f, 5e-4f),
              "Laguna prefill attention numeric");
    }

    float q_decode[n_head * HEAD_DIM];
    float k_decode[HEAD_DIM];
    float v_decode[HEAD_DIM];
    float gate_decode[n_head];
    float decode_got[n_head * HEAD_DIM];
    for (uint32_t i = 0; i < n_head * HEAD_DIM; i++) q_decode[i] = 0.125f;
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        k_decode[i] = 0.5f;
        v_decode[i] = 4.0f;
    }
    for (uint32_t h = 0; h < n_head; h++) {
        gate_decode[h] = 0.1f * (float)h;
    }
    CHECK(ds4_gpu_tensor_write(q_t, 0, q_decode, sizeof(q_decode)) &&
          ds4_gpu_tensor_write(k_t, 0, k_decode, sizeof(k_decode)) &&
          ds4_gpu_tensor_write(v_t, 0, v_decode, sizeof(v_decode)) &&
          ds4_gpu_tensor_write(gate_t, 0, gate_decode, sizeof(gate_decode)),
          "write decode attention tensors");
    CHECK(ds4_gpu_laguna_store_attention_tensor(
                  heads, key_cache, value_cache, q_t, k_t, v_t, gate_t,
                  3u, cache_cap, 0u, 4u,
                  n_head, n_head_kv, HEAD_DIM,
                  1.0f / sqrtf((float)HEAD_DIM)),
          "Laguna decode attention");
    CHECK(ds4_gpu_tensor_read(heads, 0, decode_got, sizeof(decode_got)),
          "read decode attention");
    float all_k[7u * HEAD_DIM], all_v[7u * HEAD_DIM];
    memcpy(all_k, k, sizeof(k));
    memcpy(all_v, v, sizeof(v));
    memcpy(all_k + 3 * HEAD_DIM, k_decode, sizeof(k_decode));
    memcpy(all_v + 3 * HEAD_DIM, v_decode, sizeof(v_decode));
    for (uint32_t h = 0; h < n_head; h++) {
        float scores[4];
        float max_score = -INFINITY;
        for (uint32_t t = 0; t < 4u; t++) {
            float score = 0.0f;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                score += q_decode[(uint64_t)h * HEAD_DIM + d] *
                         all_k[(uint64_t)t * HEAD_DIM + d];
            }
            scores[t] = score / sqrtf((float)HEAD_DIM);
            if (scores[t] > max_score) max_score = scores[t];
        }
        float sum = 0.0f, value = 0.0f;
        for (uint32_t t = 0; t < 4u; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
            value += scores[t] * all_v[(uint64_t)t * HEAD_DIM];
        }
        const float expected_decode = value / sum * softplus(gate_decode[h]);
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            CHECK(close_enough(
                      decode_got[(uint64_t)h * HEAD_DIM + d],
                      expected_decode, 5e-4f, 5e-4f),
                  "Laguna decode attention numeric");
        }
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            k[(uint64_t)t * HEAD_DIM + d] = 0.125f * (float)(t + 5u);
            v[(uint64_t)t * HEAD_DIM + d] = (float)(t + 5u);
        }
        memcpy(all_k + (uint64_t)(t + 4u) * HEAD_DIM,
               k + (uint64_t)t * HEAD_DIM,
               HEAD_DIM * sizeof(float));
        memcpy(all_v + (uint64_t)(t + 4u) * HEAD_DIM,
               v + (uint64_t)t * HEAD_DIM,
               HEAD_DIM * sizeof(float));
    }
    attention_ref(expected, q, all_k, all_v, gate, 4u, n_tokens, cache_cap,
                  n_head, n_head_kv);
    CHECK(ds4_gpu_tensor_write(q_t, 0, q, sizeof(q)) &&
          ds4_gpu_tensor_write(k_t, 0, k, sizeof(k)) &&
          ds4_gpu_tensor_write(v_t, 0, v, sizeof(v)) &&
          ds4_gpu_tensor_write(gate_t, 0, gate, sizeof(gate)),
          "write wrapped prefill tensors");
    CHECK(ds4_gpu_laguna_attention_prefill_tensor(
                  heads, key_cache, value_cache, staged_key, staged_value,
                  q_t, k_t, v_t, gate_t, 4u, n_tokens, cache_cap,
                  n_head, n_head_kv, HEAD_DIM,
                  1.0f / sqrtf((float)HEAD_DIM)),
          "Laguna wrapped prefill attention");
    CHECK(ds4_gpu_tensor_read(heads, 0, got, sizeof(got)),
          "read wrapped prefill attention");
    for (uint64_t i = 0; i < q_values; i++) {
        CHECK(close_enough(got[i], expected[i], 5e-4f, 5e-4f),
              "Laguna wrapped prefill attention numeric");
    }

    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(v_t);
    ds4_gpu_tensor_free(k_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(staged_value);
    ds4_gpu_tensor_free(staged_key);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    return 0;
}

static int check_dflash_blackwell_attention(void) {
    enum {
        n_tokens = 3,
        n_head = 9,
        n_head_kv = 1,
        cache_cap = 4
    };
    const uint64_t q_values = (uint64_t)n_tokens * n_head * HEAD_DIM;
    const uint64_t kv_values =
        (uint64_t)n_tokens * n_head_kv * HEAD_DIM;
    float q[q_values], k[kv_values], v[kv_values];
    float gate[n_tokens * n_head];
    float portable[q_values], blackwell[q_values];
    for (uint64_t i = 0; i < q_values; i++) {
        q[i] = 0.015625f * (float)(1u + i % 13u);
    }
    for (uint64_t i = 0; i < kv_values; i++) {
        k[i] = 0.03125f * (float)(1u + i % 7u);
        v[i] = 0.0625f * (float)(1u + i % 11u);
    }
    for (uint32_t i = 0; i < n_tokens * n_head; i++) {
        gate[i] = 0.05f * (float)(i % n_head);
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(sizeof(portable));
    ds4_gpu_tensor *key_cache =
        ds4_gpu_tensor_alloc((uint64_t)cache_cap * HEAD_DIM *
                             sizeof(uint16_t));
    ds4_gpu_tensor *value_cache =
        ds4_gpu_tensor_alloc((uint64_t)cache_cap * HEAD_DIM *
                             sizeof(uint16_t));
    ds4_gpu_tensor *staged_key =
        ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
    ds4_gpu_tensor *staged_value =
        ds4_gpu_tensor_alloc(kv_values * sizeof(uint16_t));
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *k_t = ds4_gpu_tensor_alloc(sizeof(k));
    ds4_gpu_tensor *v_t = ds4_gpu_tensor_alloc(sizeof(v));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc(sizeof(gate));
    CHECK(heads && key_cache && value_cache && staged_key && staged_value &&
          q_t && k_t && v_t && gate_t,
          "DFlash attention tensor allocation");
    CHECK(ds4_gpu_tensor_write(q_t, 0, q, sizeof(q)) &&
          ds4_gpu_tensor_write(k_t, 0, k, sizeof(k)) &&
          ds4_gpu_tensor_write(v_t, 0, v, sizeof(v)) &&
          ds4_gpu_tensor_write(gate_t, 0, gate, sizeof(gate)),
          "write DFlash attention tensors");
    CHECK(setenv("DS4_CUDA_DFLASH_NO_BLACKWELL", "1", 1) == 0,
          "select portable DFlash attention");
    CHECK(ds4_gpu_laguna_attention_prefill_tensor(
              heads, key_cache, value_cache, staged_key, staged_value,
              q_t, k_t, v_t, gate_t, 0u, n_tokens, cache_cap,
              n_head, n_head_kv, HEAD_DIM,
              1.0f / sqrtf((float)HEAD_DIM)) &&
          ds4_gpu_tensor_read(heads, 0, portable, sizeof(portable)),
          "portable DFlash attention");
    CHECK(unsetenv("DS4_CUDA_DFLASH_NO_BLACKWELL") == 0,
          "select Blackwell DFlash attention");
    CHECK(ds4_gpu_laguna_attention_prefill_tensor(
              heads, key_cache, value_cache, staged_key, staged_value,
              q_t, k_t, v_t, gate_t, 0u, n_tokens, cache_cap,
              n_head, n_head_kv, HEAD_DIM,
              1.0f / sqrtf((float)HEAD_DIM)) &&
          ds4_gpu_tensor_read(heads, 0, blackwell, sizeof(blackwell)),
          "Blackwell DFlash attention");
    for (uint64_t i = 0; i < q_values; i++) {
        CHECK(close_enough(blackwell[i], portable[i], 1e-6f, 1e-6f),
              "Blackwell DFlash attention equivalence");
    }

    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(v_t);
    ds4_gpu_tensor_free(k_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(staged_value);
    ds4_gpu_tensor_free(staged_key);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    return 0;
}

static int check_long_decode_attention(void) {
    enum { n_head = 6, n_head_kv = 1, cache_cap = 4096 };
    const uint64_t cache_values =
        (uint64_t)cache_cap * n_head_kv * HEAD_DIM;
    uint16_t *key_cache_host =
        malloc((size_t)cache_values * sizeof(uint16_t));
    uint16_t *value_cache_host =
        malloc((size_t)cache_values * sizeof(uint16_t));
    CHECK(key_cache_host && value_cache_host,
          "long attention host allocation");
    const uint16_t key_bits[] = {0x3000u, 0x3400u, 0x3800u};
    const uint16_t value_bits[] = {
        0x3800u, 0x3c00u, 0x3e00u, 0x4000u
    };
    for (uint32_t t = 0; t < cache_cap; t++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            const uint64_t i = (uint64_t)t * HEAD_DIM + d;
            key_cache_host[i] = key_bits[(t + d) % 3u];
            value_cache_host[i] = value_bits[(3u * t + d) % 4u];
        }
    }
    float q[n_head * HEAD_DIM];
    float k[HEAD_DIM];
    float v[HEAD_DIM];
    float gate[n_head];
    float scalar[n_head * HEAD_DIM];
    float split8[n_head * HEAD_DIM];
    float split_default[n_head * HEAD_DIM];
    for (uint32_t h = 0; h < n_head; h++) {
        gate[h] = 0.1f * (float)h;
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            q[(uint64_t)h * HEAD_DIM + d] =
                0.015625f * (float)(1u + (d + 3u * h) % 11u);
        }
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        k[d] = f16_ref(key_bits[(cache_cap - 1u + d) % 3u]);
        v[d] = f16_ref(value_bits[
            (3u * (cache_cap - 1u) + d) % 4u]);
    }

    ds4_gpu_tensor *heads =
        ds4_gpu_tensor_alloc(sizeof(scalar));
    ds4_gpu_tensor *key_cache =
        ds4_gpu_tensor_alloc(cache_values * sizeof(uint16_t));
    ds4_gpu_tensor *value_cache =
        ds4_gpu_tensor_alloc(cache_values * sizeof(uint16_t));
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *k_t = ds4_gpu_tensor_alloc(sizeof(k));
    ds4_gpu_tensor *v_t = ds4_gpu_tensor_alloc(sizeof(v));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc(sizeof(gate));
    CHECK(heads && key_cache && value_cache && q_t && k_t && v_t && gate_t,
          "long attention tensor allocation");
    CHECK(ds4_gpu_tensor_write(
              key_cache, 0, key_cache_host,
              cache_values * sizeof(uint16_t)) &&
          ds4_gpu_tensor_write(
              value_cache, 0, value_cache_host,
              cache_values * sizeof(uint16_t)) &&
          ds4_gpu_tensor_write(q_t, 0, q, sizeof(q)) &&
          ds4_gpu_tensor_write(k_t, 0, k, sizeof(k)) &&
          ds4_gpu_tensor_write(v_t, 0, v, sizeof(v)) &&
          ds4_gpu_tensor_write(gate_t, 0, gate, sizeof(gate)),
          "write long attention tensors");

    CHECK(setenv("DS4_CUDA_LAGUNA_NO_SPLIT_DECODE", "1", 1) == 0,
          "disable split attention");
    CHECK(ds4_gpu_laguna_store_attention_tensor(
              heads, key_cache, value_cache, q_t, k_t, v_t, gate_t,
              cache_cap - 1u, cache_cap, 0u, cache_cap,
              n_head, n_head_kv, HEAD_DIM,
              1.0f / sqrtf((float)HEAD_DIM)) &&
          ds4_gpu_tensor_read(heads, 0, scalar, sizeof(scalar)),
          "long scalar attention");
    CHECK(unsetenv("DS4_CUDA_LAGUNA_NO_SPLIT_DECODE") == 0 &&
          setenv("DS4_CUDA_LAGUNA_NO_BLACKWELL_SPLIT16", "1", 1) == 0,
          "select portable split attention");
    CHECK(ds4_gpu_laguna_store_attention_tensor(
              heads, key_cache, value_cache, q_t, k_t, v_t, gate_t,
              cache_cap - 1u, cache_cap, 0u, cache_cap,
              n_head, n_head_kv, HEAD_DIM,
              1.0f / sqrtf((float)HEAD_DIM)) &&
          ds4_gpu_tensor_read(heads, 0, split8, sizeof(split8)),
          "long portable split attention");
    CHECK(unsetenv("DS4_CUDA_LAGUNA_NO_BLACKWELL_SPLIT16") == 0,
          "select default split attention");
    CHECK(ds4_gpu_laguna_store_attention_tensor(
              heads, key_cache, value_cache, q_t, k_t, v_t, gate_t,
              cache_cap - 1u, cache_cap, 0u, cache_cap,
              n_head, n_head_kv, HEAD_DIM,
              1.0f / sqrtf((float)HEAD_DIM)) &&
          ds4_gpu_tensor_read(
              heads, 0, split_default, sizeof(split_default)),
          "long default split attention");
    for (uint32_t i = 0; i < n_head * HEAD_DIM; i++) {
        CHECK(close_enough(split8[i], scalar[i], 2e-4f, 2e-4f),
              "long portable split attention numeric");
        CHECK(close_enough(split_default[i], scalar[i], 2e-4f, 2e-4f),
              "long default split attention numeric");
    }

    /* Exercise the Blackwell split-16 threshold itself, not only the
     * full-cache case above. */
    const uint32_t threshold_count = 2048u;
    CHECK(setenv("DS4_CUDA_LAGUNA_NO_SPLIT_DECODE", "1", 1) == 0,
          "disable threshold split attention");
    CHECK(ds4_gpu_laguna_store_attention_tensor(
              heads, key_cache, value_cache, q_t, k_t, v_t, gate_t,
              threshold_count - 1u, cache_cap, 0u, threshold_count,
              n_head, n_head_kv, HEAD_DIM,
              1.0f / sqrtf((float)HEAD_DIM)) &&
          ds4_gpu_tensor_read(heads, 0, scalar, sizeof(scalar)),
          "threshold scalar attention");
    CHECK(unsetenv("DS4_CUDA_LAGUNA_NO_SPLIT_DECODE") == 0 &&
          setenv("DS4_CUDA_LAGUNA_NO_BLACKWELL_SPLIT16", "1", 1) == 0,
          "select threshold portable split attention");
    CHECK(ds4_gpu_laguna_store_attention_tensor(
              heads, key_cache, value_cache, q_t, k_t, v_t, gate_t,
              threshold_count - 1u, cache_cap, 0u, threshold_count,
              n_head, n_head_kv, HEAD_DIM,
              1.0f / sqrtf((float)HEAD_DIM)) &&
          ds4_gpu_tensor_read(heads, 0, split8, sizeof(split8)),
          "threshold portable split attention");
    CHECK(unsetenv("DS4_CUDA_LAGUNA_NO_BLACKWELL_SPLIT16") == 0,
          "select threshold default split attention");
    CHECK(ds4_gpu_laguna_store_attention_tensor(
              heads, key_cache, value_cache, q_t, k_t, v_t, gate_t,
              threshold_count - 1u, cache_cap, 0u, threshold_count,
              n_head, n_head_kv, HEAD_DIM,
              1.0f / sqrtf((float)HEAD_DIM)) &&
          ds4_gpu_tensor_read(
              heads, 0, split_default, sizeof(split_default)),
          "threshold default split attention");
    for (uint32_t i = 0; i < n_head * HEAD_DIM; i++) {
        CHECK(close_enough(split8[i], scalar[i], 2e-4f, 2e-4f),
              "threshold portable split attention numeric");
        CHECK(close_enough(split_default[i], scalar[i], 2e-4f, 2e-4f),
              "threshold default split attention numeric");
    }

    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(v_t);
    ds4_gpu_tensor_free(k_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(heads);
    free(value_cache_host);
    free(key_cache_host);
    return 0;
}

static int check_moe(model_blob *blob,
                     const ds4_gpu_laguna_moe_desc *routed,
                     const ds4_gpu_laguna_moe_desc *shared,
                     float row_sum) {
    const uint32_t dim = QK_K;
    enum { n_tokens = 2 };
    float x_host[dim];
    for (uint32_t i = 0; i < dim; i++) x_host[i] = 0.01f;
    const int32_t selected_host[2] = {0, 1};
    const float weights_host[2] = {0.25f, 0.75f};
    const int32_t shared_selected_host[1] = {0};
    const float shared_weight_host[1] = {1.0f};
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(x_host));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(sizeof(selected_host));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(sizeof(weights_host));
    ds4_gpu_tensor *shared_selected =
        ds4_gpu_tensor_alloc(sizeof(shared_selected_host));
    ds4_gpu_tensor *shared_weight =
        ds4_gpu_tensor_alloc(sizeof(shared_weight_host));
    ds4_gpu_tensor *routed_out = ds4_gpu_tensor_alloc(dim * sizeof(float));
    ds4_gpu_tensor *routed_mid =
        ds4_gpu_tensor_alloc(2u * dim * sizeof(float));
    ds4_gpu_tensor *shared_out = ds4_gpu_tensor_alloc(dim * sizeof(float));
    ds4_gpu_tensor *shared_mid = ds4_gpu_tensor_alloc(dim * sizeof(float));
    CHECK(x && selected && weights && shared_selected && shared_weight &&
          routed_out && routed_mid && shared_out && shared_mid,
          "MoE tensor allocation");
    CHECK(ds4_gpu_tensor_write(x, 0, x_host, sizeof(x_host)) &&
          ds4_gpu_tensor_write(selected, 0, selected_host,
                               sizeof(selected_host)) &&
          ds4_gpu_tensor_write(weights, 0, weights_host,
                               sizeof(weights_host)) &&
          ds4_gpu_tensor_write(shared_selected, 0, shared_selected_host,
                               sizeof(shared_selected_host)) &&
          ds4_gpu_tensor_write(shared_weight, 0, shared_weight_host,
                               sizeof(shared_weight_host)),
          "write MoE tensors");
    CHECK(ds4_gpu_laguna_routed_shared_moe_one_tensor(
                  routed_out, routed_mid, shared_out, shared_mid,
                  blob->data, blob->size, routed, shared,
                  dim, dim, dim, selected, weights, 2u, 2u,
                  shared_selected, shared_weight, x),
          "Laguna routed/shared MoE");
    float routed_host[dim], shared_host[dim];
    CHECK(ds4_gpu_tensor_read(routed_out, 0, routed_host,
                              sizeof(routed_host)) &&
          ds4_gpu_tensor_read(shared_out, 0, shared_host,
                              sizeof(shared_host)),
          "read MoE outputs");
    const float projection = 0.01f * row_sum;
    const float expected =
        (projection / (1.0f + expf(-projection))) *
        projection * row_sum;
    for (uint32_t i = 0; i < dim; i++) {
        CHECK(close_enough(routed_host[i], expected, 0.08f, 2e-3f),
              "Laguna routed MoE numeric");
        CHECK(close_enough(shared_host[i], expected, 0.08f, 2e-3f),
              "Laguna shared MoE numeric");
    }

    float batch_x_host[n_tokens * dim];
    const int32_t batch_selected_host[n_tokens * 2u] = {0, 1, 1, 0};
    const float batch_weights_host[n_tokens * 2u] = {
        0.25f, 0.75f, 0.60f, 0.40f
    };
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < dim; i++) {
            batch_x_host[(uint64_t)t * dim + i] = 0.01f * (float)(t + 1u);
        }
    }
    ds4_gpu_tensor *batch_x = ds4_gpu_tensor_alloc(sizeof(batch_x_host));
    ds4_gpu_tensor *batch_selected =
        ds4_gpu_tensor_alloc(sizeof(batch_selected_host));
    ds4_gpu_tensor *batch_weights =
        ds4_gpu_tensor_alloc(sizeof(batch_weights_host));
    ds4_gpu_tensor *batch_out =
        ds4_gpu_tensor_alloc(n_tokens * dim * sizeof(float));
    ds4_gpu_tensor *batch_mid =
        ds4_gpu_tensor_alloc(n_tokens * 2u * dim * sizeof(float));
    CHECK(batch_x && batch_selected && batch_weights && batch_out && batch_mid,
          "batch MoE tensor allocation");
    CHECK(ds4_gpu_tensor_write(batch_x, 0, batch_x_host,
                               sizeof(batch_x_host)) &&
          ds4_gpu_tensor_write(batch_selected, 0, batch_selected_host,
                               sizeof(batch_selected_host)) &&
          ds4_gpu_tensor_write(batch_weights, 0, batch_weights_host,
                               sizeof(batch_weights_host)),
          "write batch MoE tensors");
    CHECK(ds4_gpu_glm_routed_moe_batch_tensor(
                  batch_out, batch_mid, blob->data, blob->size,
                  routed->gate_offset, routed->up_offset, routed->down_offset,
                  routed->gate_type, routed->up_type, routed->down_type,
                  routed->gate_expert_bytes, routed->gate_row_bytes,
                  routed->up_expert_bytes, routed->up_row_bytes,
                  routed->down_expert_bytes, routed->down_row_bytes,
                  dim, dim, dim, batch_selected, batch_weights, 2u, 2u, 0u,
                  batch_x, n_tokens, 2u * dim, false),
          "Laguna routed batch MoE");
    float batch_out_host[n_tokens * dim];
    CHECK(ds4_gpu_tensor_read(batch_out, 0, batch_out_host,
                              sizeof(batch_out_host)),
          "read batch MoE output");
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float batch_projection =
            0.01f * (float)(t + 1u) * row_sum;
        const float batch_expected =
            (batch_projection / (1.0f + expf(-batch_projection))) *
            batch_projection * row_sum;
        for (uint32_t i = 0; i < dim; i++) {
            CHECK(close_enough(batch_out_host[(uint64_t)t * dim + i],
                               batch_expected, 0.20f, 2e-3f),
                  "Laguna routed batch MoE numeric");
        }
    }
    ds4_gpu_tensor_free(batch_mid);
    ds4_gpu_tensor_free(batch_out);
    ds4_gpu_tensor_free(batch_weights);
    ds4_gpu_tensor_free(batch_selected);
    ds4_gpu_tensor_free(batch_x);
    ds4_gpu_tensor_free(shared_mid);
    ds4_gpu_tensor_free(shared_out);
    ds4_gpu_tensor_free(routed_mid);
    ds4_gpu_tensor_free(routed_out);
    ds4_gpu_tensor_free(shared_weight);
    ds4_gpu_tensor_free(shared_selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
    return 0;
}

int main(void) {
    CHECK(ds4_gpu_init(), "ds4_gpu_init");
    const uint64_t blob_size = 2u * 1024u * 1024u;
    void *host = NULL;
    CHECK(cudaMallocHost(&host, blob_size) == cudaSuccess,
          "allocate pinned model blob");
    memset(host, 0, blob_size);
    model_blob blob = {(uint8_t *)host, blob_size, 0};

    const uint64_t q4_offset =
        blob_alloc(&blob, 2u * sizeof(block_q4_K));
    const uint64_t q6_offset =
        blob_alloc(&blob, 2u * sizeof(block_q6_K));
    const uint64_t embed_offset =
        blob_alloc(&blob, 2u * sizeof(block_q4_K));
    const uint64_t norm_offset =
        blob_alloc(&blob, HEAD_DIM * sizeof(float));
    CHECK(q4_offset != UINT64_MAX && q6_offset != UINT64_MAX &&
          embed_offset != UINT64_MAX && norm_offset != UINT64_MAX,
          "allocate basic model ranges");
    fill_q4_pattern(blob.data + q4_offset, 2u);
    fill_q6_pattern(blob.data + q6_offset, 2u);
    fill_q4_ones(blob.data + embed_offset, 2u);
    float *norm = (float *)(blob.data + norm_offset);
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        norm[i] = 0.75f + (float)(i % 7u) * 0.03125f;
    }

    const uint32_t dim = QK_K;
    const uint64_t q4_matrix_bytes =
        (uint64_t)dim * sizeof(block_q4_K);
    const uint64_t q6_matrix_bytes =
        (uint64_t)dim * sizeof(block_q6_K);
    const uint64_t q2_matrix_bytes =
        (uint64_t)dim * sizeof(block_q2_K);
    const uint64_t q3_matrix_bytes =
        (uint64_t)dim * sizeof(block_q3_K);
    ds4_gpu_laguna_moe_desc routed = {0};
    routed.gate_offset = blob_alloc(&blob, 2u * q4_matrix_bytes);
    routed.up_offset = blob_alloc(&blob, 2u * q4_matrix_bytes);
    routed.down_offset = blob_alloc(&blob, 2u * q6_matrix_bytes);
    routed.gate_type = 12u;
    routed.up_type = 12u;
    routed.down_type = 14u;
    routed.gate_expert_bytes = q4_matrix_bytes;
    routed.gate_row_bytes = sizeof(block_q4_K);
    routed.up_expert_bytes = q4_matrix_bytes;
    routed.up_row_bytes = sizeof(block_q4_K);
    routed.down_expert_bytes = q6_matrix_bytes;
    routed.down_row_bytes = sizeof(block_q6_K);
    ds4_gpu_laguna_moe_desc shared = routed;
    shared.gate_offset = blob_alloc(&blob, q4_matrix_bytes);
    shared.up_offset = blob_alloc(&blob, q4_matrix_bytes);
    shared.down_offset = blob_alloc(&blob, q6_matrix_bytes);
    ds4_gpu_laguna_moe_desc revised_routed = routed;
    revised_routed.down_offset =
        blob_alloc(&blob, 2u * q4_matrix_bytes);
    revised_routed.down_type = 12u;
    revised_routed.down_expert_bytes = q4_matrix_bytes;
    revised_routed.down_row_bytes = sizeof(block_q4_K);
    ds4_gpu_laguna_moe_desc revised_shared = revised_routed;
    revised_shared.gate_offset = blob_alloc(&blob, q4_matrix_bytes);
    revised_shared.up_offset = blob_alloc(&blob, q4_matrix_bytes);
    revised_shared.down_offset = blob_alloc(&blob, q4_matrix_bytes);
    ds4_gpu_laguna_moe_desc q2_routed = {0};
    q2_routed.gate_offset = blob_alloc(&blob, 2u * q2_matrix_bytes);
    q2_routed.up_offset = blob_alloc(&blob, 2u * q2_matrix_bytes);
    q2_routed.down_offset = blob_alloc(&blob, 2u * q2_matrix_bytes);
    q2_routed.gate_type = 10u;
    q2_routed.up_type = 10u;
    q2_routed.down_type = 10u;
    q2_routed.gate_expert_bytes = q2_matrix_bytes;
    q2_routed.gate_row_bytes = sizeof(block_q2_K);
    q2_routed.up_expert_bytes = q2_matrix_bytes;
    q2_routed.up_row_bytes = sizeof(block_q2_K);
    q2_routed.down_expert_bytes = q2_matrix_bytes;
    q2_routed.down_row_bytes = sizeof(block_q2_K);
    ds4_gpu_laguna_moe_desc q2_shared = q2_routed;
    q2_shared.gate_offset = blob_alloc(&blob, q2_matrix_bytes);
    q2_shared.up_offset = blob_alloc(&blob, q2_matrix_bytes);
    q2_shared.down_offset = blob_alloc(&blob, q2_matrix_bytes);
    ds4_gpu_laguna_moe_desc q3_routed = {0};
    q3_routed.gate_offset = blob_alloc(&blob, 2u * q3_matrix_bytes);
    q3_routed.up_offset = blob_alloc(&blob, 2u * q3_matrix_bytes);
    q3_routed.down_offset = blob_alloc(&blob, 2u * q3_matrix_bytes);
    q3_routed.gate_type = 11u;
    q3_routed.up_type = 11u;
    q3_routed.down_type = 11u;
    q3_routed.gate_expert_bytes = q3_matrix_bytes;
    q3_routed.gate_row_bytes = sizeof(block_q3_K);
    q3_routed.up_expert_bytes = q3_matrix_bytes;
    q3_routed.up_row_bytes = sizeof(block_q3_K);
    q3_routed.down_expert_bytes = q3_matrix_bytes;
    q3_routed.down_row_bytes = sizeof(block_q3_K);
    ds4_gpu_laguna_moe_desc q3_shared = q3_routed;
    q3_shared.gate_offset = blob_alloc(&blob, q3_matrix_bytes);
    q3_shared.up_offset = blob_alloc(&blob, q3_matrix_bytes);
    q3_shared.down_offset = blob_alloc(&blob, q3_matrix_bytes);
    CHECK(routed.gate_offset != UINT64_MAX &&
          routed.up_offset != UINT64_MAX &&
          routed.down_offset != UINT64_MAX &&
          shared.gate_offset != UINT64_MAX &&
          shared.up_offset != UINT64_MAX &&
          shared.down_offset != UINT64_MAX &&
          revised_routed.down_offset != UINT64_MAX &&
          revised_shared.gate_offset != UINT64_MAX &&
          revised_shared.up_offset != UINT64_MAX &&
          revised_shared.down_offset != UINT64_MAX &&
          q2_routed.gate_offset != UINT64_MAX &&
          q2_routed.up_offset != UINT64_MAX &&
          q2_routed.down_offset != UINT64_MAX &&
          q2_shared.gate_offset != UINT64_MAX &&
          q2_shared.up_offset != UINT64_MAX &&
          q2_shared.down_offset != UINT64_MAX &&
          q3_routed.gate_offset != UINT64_MAX &&
          q3_routed.up_offset != UINT64_MAX &&
          q3_routed.down_offset != UINT64_MAX &&
          q3_shared.gate_offset != UINT64_MAX &&
          q3_shared.up_offset != UINT64_MAX &&
          q3_shared.down_offset != UINT64_MAX,
          "allocate MoE model ranges");
    fill_q4_ones(blob.data + routed.gate_offset,
                 2u * dim);
    fill_q4_ones(blob.data + routed.up_offset,
                 2u * dim);
    fill_q6_ones(blob.data + routed.down_offset,
                 2u * dim);
    fill_q4_ones(blob.data + shared.gate_offset, dim);
    fill_q4_ones(blob.data + shared.up_offset, dim);
    fill_q6_ones(blob.data + shared.down_offset, dim);
    fill_q4_ones(blob.data + revised_routed.down_offset,
                 2u * dim);
    fill_q4_ones(blob.data + revised_shared.gate_offset, dim);
    fill_q4_ones(blob.data + revised_shared.up_offset, dim);
    fill_q4_ones(blob.data + revised_shared.down_offset, dim);
    fill_q2_ones(blob.data + q2_routed.gate_offset, 2u * dim);
    fill_q2_ones(blob.data + q2_routed.up_offset, 2u * dim);
    fill_q2_ones(blob.data + q2_routed.down_offset, 2u * dim);
    fill_q2_ones(blob.data + q2_shared.gate_offset, dim);
    fill_q2_ones(blob.data + q2_shared.up_offset, dim);
    fill_q2_ones(blob.data + q2_shared.down_offset, dim);
    fill_q3_pattern(blob.data + q3_routed.gate_offset, 2u * dim);
    fill_q3_pattern(blob.data + q3_routed.up_offset, 2u * dim);
    fill_q3_pattern(blob.data + q3_routed.down_offset, 2u * dim);
    fill_q3_pattern(blob.data + q3_shared.gate_offset, dim);
    fill_q3_pattern(blob.data + q3_shared.up_offset, dim);
    fill_q3_pattern(blob.data + q3_shared.down_offset, dim);
    float q3_row_sum = 0.0f;
    const block_q3_K *q3_row =
        (const block_q3_K *)(blob.data + q3_routed.gate_offset);
    for (uint32_t i = 0u; i < dim; i++) {
        q3_row_sum += q3_value_ref(q3_row, i);
    }

    CHECK(ds4_gpu_set_model_map(blob.data, blob.size), "set synthetic model map");
    int rc = check_quant_ops(&blob, q4_offset, q6_offset, embed_offset);
    if (rc == 0) rc = check_norm_rope(&blob, norm_offset);
    if (rc == 0) rc = check_attention();
    if (rc == 0) rc = check_dflash_blackwell_attention();
    if (rc == 0) rc = check_long_decode_attention();
    if (rc == 0) rc = check_moe(&blob, &routed, &shared, (float)dim);
    if (rc == 0) {
        rc = check_moe(
                &blob, &revised_routed, &revised_shared, (float)dim);
    }
    if (rc == 0) {
        rc = check_moe(&blob, &q2_routed, &q2_shared, (float)dim);
    }
    if (rc == 0) {
        rc = check_moe(&blob, &q3_routed, &q3_shared, q3_row_sum);
    }
    ds4_gpu_cleanup();
    (void)cudaFreeHost(host);
    if (rc == 0) puts("cuda Laguna regression: OK");
    return rc;
}
