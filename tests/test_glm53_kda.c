#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

#ifdef DS4_ROCM_BUILD
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} test_block_q4_K;

typedef struct {
    uint16_t d;
    int8_t qs[32];
} test_block_q8_0;

extern int ds4_gpu_matmul_q4_K_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint64_t n_rows);

extern int ds4_gpu_glm_attention_indexed_decode_tensor(
    ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
    const ds4_gpu_tensor *k_rope_cache, const void *model_map,
    uint64_t model_size, uint64_t value_weight_offset,
    const ds4_gpu_tensor *selected, uint32_t n_selected,
    uint32_t cache_cap, bool cache_f16, uint32_t n_head,
    uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_rope,
    uint32_t value_dim, uint32_t n_ctx_orig, float freq_base,
    float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow);
#endif

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void require_close(const char *what, float actual, float expected, float tolerance) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "%s: got %.9g, expected %.9g (tolerance %.9g)\n",
                what, actual, expected, tolerance);
        exit(1);
    }
}

static uint16_t f32_to_bf16(float value) {
    union { float f; uint32_t u; } bits = { .f = value };
    const uint32_t rounding = 0x7fffu + ((bits.u >> 16) & 1u);
    return (uint16_t)((bits.u + rounding) >> 16);
}

static float bf16_to_f32(uint16_t value) {
    union { uint32_t u; float f; } bits = { .u = (uint32_t)value << 16 };
    return bits.f;
}

int main(void) {
    enum {
        D = 128,
        HEADS = 2,
        PROJECTION = HEADS * D,
        TOKENS = 17,
        Q_CONV_OFFSET = 0,
        K_CONV_OFFSET = 4096,
        V_CONV_OFFSET = 8192,
        A_LOG_OFFSET = 12288,
        DT_BIAS_OFFSET = 16384,
        NORM_OFFSET = 20480,
        POOL_NORM_OFFSET = 22528,
        POOL_BIAS_OFFSET = 24576,
        POOL_APE_OFFSET = 28672,
        BF16_OFFSET = 32768,
        BF16_IN = 64,
        BF16_OUT = 64,
        BF16_ROWS = 16,
        Q4_OFFSET = 49152,
        Q4_IN = 256,
        Q4_OUT = 37,
        Q4_ROWS = 3,
        Q8_OFFSET = 60000,
        MODEL_BYTES = 65536,
    };

    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    float *q_conv = (float *)(model + Q_CONV_OFFSET);
    float *k_conv = (float *)(model + K_CONV_OFFSET);
    float *v_conv = (float *)(model + V_CONV_OFFSET);
    float *a_log = (float *)(model + A_LOG_OFFSET);
    float *dt_bias = (float *)(model + DT_BIAS_OFFSET);
    float *norm = (float *)(model + NORM_OFFSET);
    for (uint32_t channel = 0; channel < PROJECTION; channel++) {
        q_conv[channel * 4u + 3u] = 1.0f;
        k_conv[channel * 4u + 3u] = 1.0f;
        v_conv[channel * 4u + 3u] = 1.0f;
        dt_bias[channel] = 0.0f;
    }
    for (uint32_t head = 0; head < HEADS; head++) a_log[head] = 0.0f;
    for (uint32_t d = 0; d < D; d++) norm[d] = 1.0f;
    float *pool_norm = (float *)(model + POOL_NORM_OFFSET);
    float *pool_bias = (float *)(model + POOL_BIAS_OFFSET);
    uint16_t *pool_ape = (uint16_t *)(model + POOL_APE_OFFSET);
    for (uint32_t d = 0; d < D; d++) {
        pool_norm[d] = 0.75f + 0.002f * (float)d;
        pool_bias[d] = -0.1f + 0.001f * (float)d;
        for (uint32_t r = 0; r < 4u; r++) {
            pool_ape[r * D + d] = f32_to_bf16(
                0.03f * (float)r - 0.0005f * (float)d);
        }
    }

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map registration");

    uint16_t *bf16_weights = (uint16_t *)(model + BF16_OFFSET);
    for (uint32_t o = 0; o < BF16_OUT; o++) {
        for (uint32_t i = 0; i < BF16_IN; i++) {
            const float value = 0.002f * (float)((int)(o % 11u) - 5) +
                                0.001f * (float)((int)(i % 13u) - 6);
            bf16_weights[o * BF16_IN + i] = f32_to_bf16(value);
        }
    }
    float bf16_input[BF16_ROWS * BF16_IN];
    float bf16_expected[BF16_ROWS * BF16_OUT];
    for (uint32_t row = 0; row < BF16_ROWS; row++) {
        for (uint32_t i = 0; i < BF16_IN; i++) {
            bf16_input[row * BF16_IN + i] =
                0.02f * (float)((int)(i % 17u) - 8) + 0.005f * (float)row;
        }
        for (uint32_t o = 0; o < BF16_OUT; o++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < BF16_IN; i++) {
                sum += bf16_to_f32(bf16_weights[o * BF16_IN + i]) *
                       bf16_input[row * BF16_IN + i];
            }
            bf16_expected[row * BF16_OUT + o] = sum;
        }
    }
    ds4_gpu_tensor *bf16_x = ds4_gpu_tensor_alloc(sizeof(bf16_input));
    ds4_gpu_tensor *bf16_out = ds4_gpu_tensor_alloc(sizeof(bf16_expected));
    require_ok(bf16_x && bf16_out, "BF16 tensor allocation");
    require_ok(ds4_gpu_tensor_write(bf16_x, 0, bf16_input, sizeof(bf16_input)),
               "BF16 input write");
    require_ok(ds4_gpu_glm53_matmul_bf16(
        bf16_out, model, MODEL_BYTES, BF16_OFFSET,
        BF16_IN, BF16_OUT, bf16_x, 1), "BF16 decode matmul");
    float bf16_actual[BF16_ROWS * BF16_OUT];
    require_ok(ds4_gpu_tensor_read(bf16_out, 0, bf16_actual,
                                   BF16_OUT * sizeof(float)),
               "BF16 decode output read");
    for (uint32_t i = 0; i < BF16_OUT; i++)
        require_close("BF16 decode matmul", bf16_actual[i], bf16_expected[i], 2e-6f);
    require_ok(ds4_gpu_glm53_matmul_bf16(
        bf16_out, model, MODEL_BYTES, BF16_OFFSET,
        BF16_IN, BF16_OUT, bf16_x, BF16_ROWS), "BF16 prefill matmul");
    require_ok(ds4_gpu_tensor_read(bf16_out, 0, bf16_actual, sizeof(bf16_actual)),
               "BF16 prefill output read");
    for (uint32_t i = 0; i < BF16_ROWS * BF16_OUT; i++)
        require_close("BF16 prefill matmul", bf16_actual[i], bf16_expected[i], 2e-4f);

#ifdef DS4_ROCM_BUILD
    test_block_q4_K *q4_weights = (test_block_q4_K *)(model + Q4_OFFSET);
    for (uint32_t o = 0; o < Q4_OUT; o++) {
        test_block_q4_K *block = q4_weights + o;
        const uint8_t q = (uint8_t)(1u + o % 15u);
        block->d = 0x3c00u;
        block->dmin = 0u;
        for (uint32_t group = 0; group < 4u; group++) {
            block->scales[group] = 1u;
            block->scales[group + 4u] = 0u;
        }
        for (uint32_t group = 4u; group < 8u; group++)
            block->scales[group + 4u] = 1u;
        for (uint32_t i = 0; i < sizeof(block->qs); i++)
            block->qs[i] = (uint8_t)(q | (q << 4u));
    }
    float q4_input[Q4_ROWS * Q4_IN];
    const float q4_row_values[Q4_ROWS] = {1.0f, -0.5f, 0.25f};
    for (uint32_t row = 0; row < Q4_ROWS; row++)
        for (uint32_t i = 0; i < Q4_IN; i++)
            q4_input[row * Q4_IN + i] = q4_row_values[row];
    ds4_gpu_tensor *q4_x = ds4_gpu_tensor_alloc(sizeof(q4_input));
    ds4_gpu_tensor *q4_out =
        ds4_gpu_tensor_alloc((uint64_t)Q4_ROWS * Q4_OUT * sizeof(float));
    require_ok(q4_x && q4_out, "Q4_K tensor allocation");
    require_ok(ds4_gpu_tensor_write(q4_x, 0, q4_input, sizeof(q4_input)),
               "Q4_K input write");
    require_ok(ds4_gpu_matmul_q4_K_tensor(
        q4_out, model, MODEL_BYTES, Q4_OFFSET,
        Q4_IN, Q4_OUT, q4_x, Q4_ROWS), "Q4_K dense matmul");
    float q4_actual[Q4_ROWS * Q4_OUT];
    require_ok(ds4_gpu_tensor_read(q4_out, 0, q4_actual, sizeof(q4_actual)),
               "Q4_K output read");
    for (uint32_t row = 0; row < Q4_ROWS; row++) {
        for (uint32_t o = 0; o < Q4_OUT; o++) {
            const float expected =
                256.0f * (float)(1u + o % 15u) * q4_row_values[row];
            require_close("Q4_K dense matmul",
                          q4_actual[row * Q4_OUT + o], expected, 1e-3f);
        }
    }
    ds4_gpu_tensor_free(q4_out);
    ds4_gpu_tensor_free(q4_x);
#endif

    enum { COMPACT_LORA = 512, COMPACT_TOKENS = 2, COMPACT_CAP = 4 };
    float compact_norm[COMPACT_TOKENS * COMPACT_LORA];
    float compact_raw[COMPACT_TOKENS * COMPACT_LORA];
    for (uint32_t i = 0; i < COMPACT_TOKENS * COMPACT_LORA; i++) {
        compact_norm[i] = 0.001f * (float)((int)(i % 101u) - 50);
        compact_raw[i] = compact_norm[i] + 1.0f;
    }
    ds4_gpu_tensor *compact_norm_gpu =
        ds4_gpu_tensor_alloc(sizeof(compact_norm));
    ds4_gpu_tensor *compact_raw_gpu =
        ds4_gpu_tensor_alloc(sizeof(compact_raw));
    ds4_gpu_tensor *compact_cache_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)COMPACT_CAP * COMPACT_LORA * sizeof(float));
    ds4_gpu_tensor *zero_rope_cache_gpu = NULL;
#ifndef DS4_ROCM_BUILD
    zero_rope_cache_gpu = ds4_gpu_tensor_alloc(1u);
#endif
    require_ok(compact_norm_gpu && compact_raw_gpu && compact_cache_gpu
#ifndef DS4_ROCM_BUILD
               && zero_rope_cache_gpu
#endif
               ,
               "zero-RoPE compact tensor allocation");
    require_ok(ds4_gpu_tensor_write(compact_norm_gpu, 0, compact_norm,
                                    sizeof(compact_norm)),
               "zero-RoPE compact norm write");
    require_ok(ds4_gpu_tensor_write(compact_raw_gpu, 0, compact_raw,
                                    sizeof(compact_raw)),
               "zero-RoPE compact raw write");
    require_ok(ds4_gpu_glm_store_compact_kv_tensor(
        compact_cache_gpu, zero_rope_cache_gpu,
        compact_norm_gpu, compact_raw_gpu,
        1, COMPACT_TOKENS, COMPACT_CAP,
        COMPACT_LORA, COMPACT_LORA, 0, false),
        "GLM-5.3 zero-RoPE compact KV store");
    float compact_actual[COMPACT_CAP * COMPACT_LORA];
    require_ok(ds4_gpu_tensor_read(
        compact_cache_gpu,
        (uint64_t)COMPACT_LORA * sizeof(float),
        compact_actual,
        sizeof(compact_norm)),
        "zero-RoPE compact cache read");
    for (uint32_t i = 0; i < COMPACT_TOKENS * COMPACT_LORA; i++) {
        require_close("zero-RoPE compact KV", compact_actual[i],
                      compact_norm[i], 0.0f);
    }
    ds4_gpu_tensor_free(zero_rope_cache_gpu);
    ds4_gpu_tensor_free(compact_cache_gpu);
    ds4_gpu_tensor_free(compact_raw_gpu);
    ds4_gpu_tensor_free(compact_norm_gpu);

#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
    enum {
        DENSE_ATTN_PREFIX = 128,
        DENSE_ATTN_TOKENS = 256,
        DENSE_ATTN_CAP = DENSE_ATTN_PREFIX + DENSE_ATTN_TOKENS,
        DENSE_ATTN_HEADS = 8,
        DENSE_ATTN_LORA = 512,
    };
    const uint64_t dense_attn_cache_count =
        (uint64_t)DENSE_ATTN_CAP * DENSE_ATTN_LORA;
    const uint64_t dense_attn_q_count =
        (uint64_t)DENSE_ATTN_TOKENS * DENSE_ATTN_HEADS * DENSE_ATTN_LORA;
    float *dense_attn_cache =
        malloc((size_t)dense_attn_cache_count * sizeof(*dense_attn_cache));
    float *dense_attn_q =
        malloc((size_t)dense_attn_q_count * sizeof(*dense_attn_q));
    float *dense_attn_reference =
        malloc((size_t)dense_attn_q_count * sizeof(*dense_attn_reference));
    float *dense_attn_actual =
        malloc((size_t)dense_attn_q_count * sizeof(*dense_attn_actual));
    require_ok(dense_attn_cache && dense_attn_q && dense_attn_reference &&
               dense_attn_actual, "dense compact attention host allocation");
    for (uint64_t i = 0; i < dense_attn_cache_count; i++) {
        dense_attn_cache[i] =
            0.004f * (float)((int)(i % 101u) - 50);
    }
    for (uint64_t i = 0; i < dense_attn_q_count; i++) {
        dense_attn_q[i] =
            0.003f * (float)((int)(i % 127u) - 63);
    }

    ds4_gpu_tensor *dense_attn_cache_input_gpu = ds4_gpu_tensor_alloc(
        dense_attn_cache_count * sizeof(float));
    ds4_gpu_tensor *dense_attn_cache_gpu = ds4_gpu_tensor_alloc(
        dense_attn_cache_count * sizeof(uint16_t));
    ds4_gpu_tensor *dense_attn_rope_gpu = ds4_gpu_tensor_alloc(1u);
    ds4_gpu_tensor *dense_attn_q_gpu = ds4_gpu_tensor_alloc(
        dense_attn_q_count * sizeof(float));
    ds4_gpu_tensor *dense_attn_dummy_q_gpu = ds4_gpu_tensor_alloc(
        dense_attn_q_count * sizeof(float));
    ds4_gpu_tensor *dense_attn_reference_gpu = ds4_gpu_tensor_alloc(
        dense_attn_q_count * sizeof(float));
    ds4_gpu_tensor *dense_attn_actual_gpu = ds4_gpu_tensor_alloc(
        dense_attn_q_count * sizeof(float));
    require_ok(dense_attn_cache_input_gpu && dense_attn_cache_gpu &&
               dense_attn_rope_gpu && dense_attn_q_gpu &&
               dense_attn_dummy_q_gpu && dense_attn_reference_gpu &&
               dense_attn_actual_gpu,
               "dense compact attention GPU allocation");
    require_ok(ds4_gpu_tensor_write(
        dense_attn_cache_input_gpu, 0, dense_attn_cache,
        dense_attn_cache_count * sizeof(float)),
        "dense compact attention cache write");
    require_ok(ds4_gpu_tensor_write(
        dense_attn_q_gpu, 0, dense_attn_q,
        dense_attn_q_count * sizeof(float)),
        "dense compact attention Q write");
    require_ok(ds4_gpu_tensor_fill_f32(
        dense_attn_dummy_q_gpu, 0.0f, dense_attn_q_count),
        "dense compact attention dummy Q clear");
    require_ok(ds4_gpu_glm_store_compact_kv_tensor(
        dense_attn_cache_gpu, dense_attn_rope_gpu,
        dense_attn_cache_input_gpu, dense_attn_cache_input_gpu,
        0, DENSE_ATTN_CAP, DENSE_ATTN_CAP,
        DENSE_ATTN_LORA, DENSE_ATTN_LORA, 0, true),
        "dense compact attention F16 cache store");
    require_ok(ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        dense_attn_reference_gpu, dense_attn_dummy_q_gpu,
        dense_attn_q_gpu, dense_attn_cache_gpu, dense_attn_rope_gpu,
        DENSE_ATTN_TOKENS, DENSE_ATTN_PREFIX, DENSE_ATTN_CAP,
        DENSE_ATTN_CAP, true, DENSE_ATTN_HEADS, DENSE_ATTN_LORA,
        DENSE_ATTN_LORA, 0, 0, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f),
        "dense compact attention scalar reference");
    require_ok(ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
        dense_attn_actual_gpu, dense_attn_q_gpu, dense_attn_cache_gpu,
        DENSE_ATTN_PREFIX, DENSE_ATTN_TOKENS, DENSE_ATTN_CAP,
        DENSE_ATTN_CAP, true, DENSE_ATTN_HEADS, DENSE_ATTN_LORA,
        DENSE_ATTN_LORA),
        "dense compact attention GEMM path");
    require_ok(ds4_gpu_tensor_read(
        dense_attn_reference_gpu, 0, dense_attn_reference,
        dense_attn_q_count * sizeof(float)),
        "dense compact attention reference read");
    require_ok(ds4_gpu_tensor_read(
        dense_attn_actual_gpu, 0, dense_attn_actual,
        dense_attn_q_count * sizeof(float)),
        "dense compact attention output read");
    double dense_attn_sq_error = 0.0;
    float dense_attn_max_error = 0.0f;
    for (uint64_t i = 0; i < dense_attn_q_count; i++) {
        const float error = fabsf(
            dense_attn_actual[i] - dense_attn_reference[i]);
        dense_attn_max_error = fmaxf(dense_attn_max_error, error);
        dense_attn_sq_error += (double)error * (double)error;
    }
    const double dense_attn_rms_error =
        sqrt(dense_attn_sq_error / (double)dense_attn_q_count);
    if (dense_attn_max_error > 5e-4f || dense_attn_rms_error > 1e-4) {
        fprintf(stderr,
                "dense compact attention diverged from scalar reference "
                "(max %.9g, RMS %.9g)\n",
                dense_attn_max_error, dense_attn_rms_error);
        return 1;
    }
    ds4_gpu_tensor_free(dense_attn_actual_gpu);
    ds4_gpu_tensor_free(dense_attn_reference_gpu);
    ds4_gpu_tensor_free(dense_attn_dummy_q_gpu);
    ds4_gpu_tensor_free(dense_attn_q_gpu);
    ds4_gpu_tensor_free(dense_attn_rope_gpu);
    ds4_gpu_tensor_free(dense_attn_cache_gpu);
    ds4_gpu_tensor_free(dense_attn_cache_input_gpu);
    free(dense_attn_actual);
    free(dense_attn_reference);
    free(dense_attn_q);
    free(dense_attn_cache);

#endif

#ifdef __APPLE__
    enum {
        F32_ATTN_TOKENS = 2,
        F32_ATTN_HEADS = 8,
        F32_ATTN_LORA = 512,
        F32_ATTN_NOPE = 64,
    };
    const uint64_t f32_attn_lora_count =
        (uint64_t)F32_ATTN_TOKENS * F32_ATTN_HEADS * F32_ATTN_LORA;
    const uint64_t f32_attn_q_count =
        (uint64_t)F32_ATTN_TOKENS * F32_ATTN_HEADS * F32_ATTN_NOPE;
    float *f32_attn_low = calloc((size_t)f32_attn_lora_count, sizeof(float));
    float *f32_attn_q = calloc((size_t)f32_attn_q_count, sizeof(float));
    float *f32_attn_cache = malloc(
        (size_t)F32_ATTN_TOKENS * F32_ATTN_LORA * sizeof(float));
    float *f32_attn_actual = malloc((size_t)f32_attn_lora_count * sizeof(float));
    require_ok(f32_attn_low && f32_attn_q && f32_attn_cache && f32_attn_actual,
               "FP32 causal attention host allocation");
    for (uint32_t row = 0; row < F32_ATTN_TOKENS; row++) {
        const float value = 1.0f + 2.0f * (float)row;
        for (uint32_t i = 0; i < F32_ATTN_LORA; i++) {
            f32_attn_cache[(uint64_t)row * F32_ATTN_LORA + i] = value;
        }
    }

    ds4_gpu_tensor *f32_attn_out_gpu =
        ds4_gpu_tensor_alloc(f32_attn_lora_count * sizeof(float));
    ds4_gpu_tensor *f32_attn_low_gpu =
        ds4_gpu_tensor_alloc(f32_attn_lora_count * sizeof(float));
    ds4_gpu_tensor *f32_attn_q_gpu =
        ds4_gpu_tensor_alloc(f32_attn_q_count * sizeof(float));
    ds4_gpu_tensor *f32_attn_cache_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)F32_ATTN_TOKENS * F32_ATTN_LORA * sizeof(float));
    ds4_gpu_tensor *f32_attn_rope_gpu = ds4_gpu_tensor_alloc(sizeof(float));
    require_ok(f32_attn_out_gpu && f32_attn_low_gpu && f32_attn_q_gpu &&
               f32_attn_cache_gpu && f32_attn_rope_gpu,
               "FP32 causal attention GPU allocation");
    require_ok(ds4_gpu_tensor_write(f32_attn_low_gpu, 0, f32_attn_low,
                                    f32_attn_lora_count * sizeof(float)),
               "FP32 causal attention low-rank Q write");
    require_ok(ds4_gpu_tensor_write(f32_attn_q_gpu, 0, f32_attn_q,
                                    f32_attn_q_count * sizeof(float)),
               "FP32 causal attention Q write");
    require_ok(ds4_gpu_tensor_write(
                   f32_attn_cache_gpu, 0, f32_attn_cache,
                   (uint64_t)F32_ATTN_TOKENS * F32_ATTN_LORA * sizeof(float)),
               "FP32 causal attention cache write");

    require_ok(ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
                   f32_attn_out_gpu,
                   f32_attn_low_gpu,
                   f32_attn_cache_gpu,
                   0,
                   F32_ATTN_TOKENS,
                   F32_ATTN_TOKENS,
                   F32_ATTN_TOKENS,
                   false,
                   F32_ATTN_HEADS,
                   F32_ATTN_LORA,
                   F32_ATTN_NOPE),
               "FP32 dense compact causal attention");
    require_ok(ds4_gpu_tensor_read(f32_attn_out_gpu, 0, f32_attn_actual,
                                   f32_attn_lora_count * sizeof(float)),
               "FP32 dense compact causal attention read");
    for (uint32_t token = 0; token < F32_ATTN_TOKENS; token++) {
        const float expected = token == 0 ? 1.0f : 2.0f;
        for (uint64_t i = (uint64_t)token * F32_ATTN_HEADS * F32_ATTN_LORA;
             i < (uint64_t)(token + 1u) * F32_ATTN_HEADS * F32_ATTN_LORA;
             i++) {
            require_close("FP32 dense compact causal attention",
                          f32_attn_actual[i], expected, 1e-4f);
        }
    }

    require_ok(ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                   f32_attn_out_gpu,
                   f32_attn_q_gpu,
                   f32_attn_low_gpu,
                   f32_attn_cache_gpu,
                   f32_attn_rope_gpu,
                   F32_ATTN_TOKENS,
                   0,
                   F32_ATTN_TOKENS,
                   F32_ATTN_TOKENS,
                   false,
                   F32_ATTN_HEADS,
                   F32_ATTN_LORA,
                   F32_ATTN_NOPE,
                   0,
                   0,
                   0.0f,
                   0.0f,
                   0.0f,
                   1.0f,
                   0.0f,
                   0.0f),
               "FP32 indexed causal attention");
    require_ok(ds4_gpu_tensor_read(f32_attn_out_gpu, 0, f32_attn_actual,
                                   f32_attn_lora_count * sizeof(float)),
               "FP32 indexed causal attention read");
    for (uint32_t token = 0; token < F32_ATTN_TOKENS; token++) {
        const float expected = token == 0 ? 1.0f : 2.0f;
        for (uint64_t i = (uint64_t)token * F32_ATTN_HEADS * F32_ATTN_LORA;
             i < (uint64_t)(token + 1u) * F32_ATTN_HEADS * F32_ATTN_LORA;
             i++) {
            require_close("FP32 indexed causal attention",
                          f32_attn_actual[i], expected, 1e-4f);
        }
    }

    ds4_gpu_tensor_free(f32_attn_rope_gpu);
    ds4_gpu_tensor_free(f32_attn_cache_gpu);
    ds4_gpu_tensor_free(f32_attn_q_gpu);
    ds4_gpu_tensor_free(f32_attn_low_gpu);
    ds4_gpu_tensor_free(f32_attn_out_gpu);
    free(f32_attn_actual);
    free(f32_attn_cache);
    free(f32_attn_q);
    free(f32_attn_low);
#endif

#ifdef DS4_ROCM_BUILD
    enum {
        ATTN_LORA = 32,
        ATTN_NOPE = 1,
        ATTN_HEADS = 1,
        ATTN_VALUE = 1,
        ATTN_SELECTED = 2,
        ATTN_CAP = 3,
    };
    test_block_q8_0 *value_weight =
        (test_block_q8_0 *)(model + Q8_OFFSET);
    value_weight->d = 0x3c00u;
    for (uint32_t i = 0; i < ATTN_LORA; i++) value_weight->qs[i] = 1;
    float attn_q[ATTN_HEADS * ATTN_NOPE] = {0.0f};
    float attn_low[ATTN_HEADS * ATTN_LORA] = {0.0f};
    float attn_cache[ATTN_CAP * ATTN_LORA];
    for (uint32_t row = 0; row < ATTN_CAP; row++) {
        for (uint32_t i = 0; i < ATTN_LORA; i++) {
            attn_cache[row * ATTN_LORA + i] = 1.0f + 2.0f * (float)row;
        }
    }
    int32_t attn_selected[ATTN_SELECTED] = {0, 1};
    ds4_gpu_tensor *attn_heads_gpu =
        ds4_gpu_tensor_alloc(ATTN_HEADS * ATTN_VALUE * sizeof(float));
    ds4_gpu_tensor *attn_q_gpu = ds4_gpu_tensor_alloc(sizeof(attn_q));
    ds4_gpu_tensor *attn_low_gpu = ds4_gpu_tensor_alloc(sizeof(attn_low));
    ds4_gpu_tensor *attn_cache_gpu = ds4_gpu_tensor_alloc(sizeof(attn_cache));
    ds4_gpu_tensor *attn_selected_gpu =
        ds4_gpu_tensor_alloc(sizeof(attn_selected));
    require_ok(attn_heads_gpu && attn_q_gpu && attn_low_gpu &&
               attn_cache_gpu && attn_selected_gpu,
               "zero-RoPE indexed attention allocation");
    require_ok(ds4_gpu_tensor_write(attn_q_gpu, 0, attn_q, sizeof(attn_q)),
               "zero-RoPE attention Q write");
    require_ok(ds4_gpu_tensor_write(attn_low_gpu, 0, attn_low,
                                    sizeof(attn_low)),
               "zero-RoPE attention low-rank Q write");
    require_ok(ds4_gpu_tensor_write(attn_cache_gpu, 0, attn_cache,
                                    sizeof(attn_cache)),
               "zero-RoPE attention cache write");
    require_ok(ds4_gpu_tensor_write(attn_selected_gpu, 0, attn_selected,
                                    sizeof(attn_selected)),
               "zero-RoPE attention selection write");
    require_ok(ds4_gpu_glm_attention_indexed_decode_tensor(
        attn_heads_gpu, attn_q_gpu, attn_low_gpu, attn_cache_gpu, NULL,
        model, MODEL_BYTES, Q8_OFFSET, attn_selected_gpu,
        ATTN_SELECTED, ATTN_CAP, false, ATTN_HEADS, ATTN_LORA,
        ATTN_NOPE, 0, ATTN_VALUE, 0, 0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f), "GLM-5.3 zero-RoPE indexed decode attention");
    float attn_actual = 0.0f;
    require_ok(ds4_gpu_tensor_read(attn_heads_gpu, 0, &attn_actual,
                                    sizeof(attn_actual)),
               "zero-RoPE attention output read");
    require_close("zero-RoPE indexed attention", attn_actual, 64.0f, 1e-4f);
    ds4_gpu_tensor_free(attn_selected_gpu);
    ds4_gpu_tensor_free(attn_cache_gpu);
    ds4_gpu_tensor_free(attn_low_gpu);
    ds4_gpu_tensor_free(attn_q_gpu);
    ds4_gpu_tensor_free(attn_heads_gpu);

    enum {
        BATCH_ATTN_TOKENS = 128,
        BATCH_ATTN_HEADS = 64,
        BATCH_ATTN_LORA = 32,
        BATCH_ATTN_NOPE = 1,
        BATCH_ATTN_REPEATS = 16,
    };
    const uint64_t batch_q_count =
        (uint64_t)BATCH_ATTN_TOKENS * BATCH_ATTN_HEADS * BATCH_ATTN_NOPE;
    const uint64_t batch_lora_count =
        (uint64_t)BATCH_ATTN_TOKENS * BATCH_ATTN_HEADS * BATCH_ATTN_LORA;
    const uint64_t batch_cache_count =
        (uint64_t)BATCH_ATTN_TOKENS * BATCH_ATTN_LORA;
    float *batch_q = calloc((size_t)batch_q_count, sizeof(*batch_q));
    float *batch_low = malloc((size_t)batch_lora_count * sizeof(*batch_low));
    float *batch_cache = malloc((size_t)batch_cache_count * sizeof(*batch_cache));
    float *batch_expected = malloc((size_t)batch_lora_count * sizeof(*batch_expected));
    float *batch_actual = malloc((size_t)batch_lora_count * sizeof(*batch_actual));
    require_ok(batch_q && batch_low && batch_cache &&
               batch_expected && batch_actual,
               "indexed attention determinism host allocation");
    for (uint64_t i = 0; i < batch_lora_count; i++) {
        batch_low[i] = 0.001f * (float)((int)(i % 127u) - 63);
    }
    for (uint64_t i = 0; i < batch_cache_count; i++) {
        batch_cache[i] = 0.002f * (float)((int)(i % 113u) - 56);
    }
    ds4_gpu_tensor *batch_out_gpu =
        ds4_gpu_tensor_alloc(batch_lora_count * sizeof(float));
    ds4_gpu_tensor *batch_q_gpu =
        ds4_gpu_tensor_alloc(batch_q_count * sizeof(float));
    ds4_gpu_tensor *batch_low_gpu =
        ds4_gpu_tensor_alloc(batch_lora_count * sizeof(float));
    ds4_gpu_tensor *batch_cache_gpu =
        ds4_gpu_tensor_alloc(batch_cache_count * sizeof(float));
    require_ok(batch_out_gpu && batch_q_gpu && batch_low_gpu && batch_cache_gpu,
               "indexed attention determinism GPU allocation");
    require_ok(ds4_gpu_tensor_write(batch_q_gpu, 0, batch_q,
                                    batch_q_count * sizeof(float)),
               "indexed attention determinism Q write");
    require_ok(ds4_gpu_tensor_write(batch_low_gpu, 0, batch_low,
                                    batch_lora_count * sizeof(float)),
               "indexed attention determinism low-rank Q write");
    require_ok(ds4_gpu_tensor_write(batch_cache_gpu, 0, batch_cache,
                                    batch_cache_count * sizeof(float)),
               "indexed attention determinism cache write");
    for (uint32_t repeat = 0; repeat < BATCH_ATTN_REPEATS; repeat++) {
        require_ok(ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
            batch_out_gpu, batch_q_gpu, batch_low_gpu, batch_cache_gpu, NULL,
            BATCH_ATTN_TOKENS, 0, BATCH_ATTN_TOKENS, BATCH_ATTN_TOKENS,
            false, BATCH_ATTN_HEADS, BATCH_ATTN_LORA, BATCH_ATTN_NOPE, 0,
            0, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f),
            "indexed attention determinism launch");
        require_ok(ds4_gpu_tensor_read(batch_out_gpu, 0, batch_actual,
                                       batch_lora_count * sizeof(float)),
                   "indexed attention determinism output read");
        if (repeat == 0) {
            memcpy(batch_expected, batch_actual,
                   (size_t)batch_lora_count * sizeof(float));
        } else if (memcmp(batch_expected, batch_actual,
                          (size_t)batch_lora_count * sizeof(float)) != 0) {
            fprintf(stderr,
                    "indexed attention changed on repeat %u\n", repeat);
            return 1;
        }
    }
    ds4_gpu_tensor_free(batch_cache_gpu);
    ds4_gpu_tensor_free(batch_low_gpu);
    ds4_gpu_tensor_free(batch_q_gpu);
    ds4_gpu_tensor_free(batch_out_gpu);
    free(batch_actual);
    free(batch_expected);
    free(batch_cache);
    free(batch_low);
    free(batch_q);
#endif

    enum { POOL = 4, POOL_TOKENS = 11, POOL_CAP = 16, POOL_COUNT = 4 };
    float pool_raw[POOL_TOKENS * D];
    float pool_gate_values[POOL_TOKENS * D];
    for (uint32_t t = 0; t < POOL_TOKENS; t++) {
        for (uint32_t d = 0; d < D; d++) {
            pool_raw[t * D + d] =
                0.01f * (float)t + 0.002f * (float)((int)(d % 19u) - 9);
            pool_gate_values[t * D + d] =
                -0.2f + 0.07f * (float)t - 0.001f * (float)(d % 23u);
        }
    }
    ds4_gpu_tensor *pool_cache =
        ds4_gpu_tensor_alloc((uint64_t)POOL_COUNT * D * sizeof(float));
    const uint64_t pool_tail_bytes =
        (uint64_t)POOL * D * sizeof(float);
    ds4_gpu_tensor *pool_tail_k =
        ds4_gpu_tensor_alloc(2u * pool_tail_bytes);
    ds4_gpu_tensor *pool_tail_gate =
        ds4_gpu_tensor_view(pool_tail_k, pool_tail_bytes, pool_tail_bytes);
    ds4_gpu_tensor *pool_raw_gpu =
        ds4_gpu_tensor_alloc(8u * D * sizeof(float));
    ds4_gpu_tensor *pool_gate_gpu =
        ds4_gpu_tensor_alloc(8u * D * sizeof(float));
    require_ok(pool_cache && pool_tail_k && pool_tail_gate &&
               pool_raw_gpu && pool_gate_gpu, "pool tensor allocation");
    require_ok(ds4_gpu_tensor_fill_f32(pool_cache, 0.0f,
                                       (uint64_t)POOL_COUNT * D),
               "pool cache clear");
    require_ok(ds4_gpu_tensor_fill_f32(pool_tail_k, 0.0f,
                                       (uint64_t)POOL * D),
               "pool K tail clear");
    require_ok(ds4_gpu_tensor_fill_f32(pool_tail_gate, 0.0f,
                                       (uint64_t)POOL * D),
               "pool gate tail clear");
    const uint32_t pool_chunks[] = {3, 8};
    uint32_t pool_pos = 0;
    for (uint32_t c = 0;
         c < sizeof(pool_chunks) / sizeof(pool_chunks[0]); c++) {
        const uint32_t rows = pool_chunks[c];
        require_ok(ds4_gpu_tensor_write(pool_raw_gpu, 0,
                                        pool_raw + (uint64_t)pool_pos * D,
                                        (uint64_t)rows * D * sizeof(float)),
                   "pool raw write");
        require_ok(ds4_gpu_tensor_write(pool_gate_gpu, 0,
                                        pool_gate_values + (uint64_t)pool_pos * D,
                                        (uint64_t)rows * D * sizeof(float)),
                   "pool gate write");
        require_ok(ds4_gpu_glm53_indexer_pool_update_tensor(
            pool_cache, pool_tail_k, pool_tail_gate,
            pool_raw_gpu, pool_gate_gpu,
            model, MODEL_BYTES, POOL_NORM_OFFSET, POOL_BIAS_OFFSET, POOL_APE_OFFSET,
            pool_pos, rows, POOL_CAP, D, POOL, 1e-6f, false),
            "GLM-5.3 indexer pool update");
        pool_pos += rows;
    }
    float pool_actual[POOL_COUNT * D];
    require_ok(ds4_gpu_tensor_read(pool_cache, 0, pool_actual,
                                   sizeof(pool_actual)), "pool cache read");
    for (uint32_t p = 0; p < 2u; p++) {
        float means[POOL], invs[POOL];
        for (uint32_t r = 0; r < POOL; r++) {
            const float *row = pool_raw + (uint64_t)(p * POOL + r) * D;
            float sum = 0.0f;
            for (uint32_t d = 0; d < D; d++) sum += row[d];
            means[r] = sum / (float)D;
            float ss = 0.0f;
            for (uint32_t d = 0; d < D; d++) {
                const float delta = row[d] - means[r];
                ss += delta * delta;
            }
            invs[r] = 1.0f / sqrtf(ss / (float)D + 1e-6f);
        }
        for (uint32_t d = 0; d < D; d++) {
            float logits[POOL], max_logit = -FLT_MAX, denom = 0.0f;
            for (uint32_t r = 0; r < POOL; r++) {
                logits[r] = pool_gate_values[(uint64_t)(p * POOL + r) * D + d] +
                    bf16_to_f32(pool_ape[r * D + d]);
                if (logits[r] > max_logit) max_logit = logits[r];
            }
            for (uint32_t r = 0; r < POOL; r++) {
                logits[r] = expf(logits[r] - max_logit);
                denom += logits[r];
            }
            float expected_pool = 0.0f;
            for (uint32_t r = 0; r < POOL; r++) {
                const float value =
                    (pool_raw[(uint64_t)(p * POOL + r) * D + d] - means[r]) *
                    invs[r] * pool_norm[d] + pool_bias[d];
                expected_pool += logits[r] / denom * value;
            }
            require_close("GLM-5.3 pool", pool_actual[p * D + d],
                          expected_pool, 2e-5f);
        }
    }

    enum { SCORE_ROWS = 3, SCORE_TOKENS = 5, SCORE_POS0 = 4 };
    float score_q[SCORE_TOKENS * HEADS * D];
    float score_weights[SCORE_TOKENS * HEADS];
    float score_cache[SCORE_ROWS * D];
    for (uint32_t t = 0; t < SCORE_TOKENS; t++) {
        for (uint32_t h = 0; h < HEADS; h++) {
            score_weights[t * HEADS + h] =
                0.25f + 0.1f * (float)t - 0.05f * (float)h;
            for (uint32_t d = 0; d < D; d++) {
                score_q[((uint64_t)t * HEADS + h) * D + d] =
                    0.01f * (float)(t + 1u) +
                    0.02f * (float)h +
                    0.0001f * (float)d;
            }
        }
    }
    for (uint32_t row = 0; row < SCORE_ROWS; row++) {
        for (uint32_t d = 0; d < D; d++) {
            score_cache[row * D + d] =
                0.03f * (float)(row + 1u) - 0.0002f * (float)d;
        }
    }
    ds4_gpu_tensor *score_q_gpu = ds4_gpu_tensor_alloc(sizeof(score_q));
    ds4_gpu_tensor *score_weights_gpu =
        ds4_gpu_tensor_alloc(sizeof(score_weights));
    ds4_gpu_tensor *score_cache_gpu = ds4_gpu_tensor_alloc(sizeof(score_cache));
    ds4_gpu_tensor *scores_gpu = ds4_gpu_tensor_alloc(
        (uint64_t)SCORE_TOKENS * SCORE_ROWS * sizeof(float));
    require_ok(score_q_gpu && score_weights_gpu && score_cache_gpu && scores_gpu,
               "grouped scorer tensor allocation");
    require_ok(ds4_gpu_tensor_write(score_q_gpu, 0, score_q, sizeof(score_q)),
               "grouped scorer Q write");
    require_ok(ds4_gpu_tensor_write(score_weights_gpu, 0, score_weights,
                                    sizeof(score_weights)),
               "grouped scorer weights write");
    require_ok(ds4_gpu_tensor_write(score_cache_gpu, 0, score_cache,
                                    sizeof(score_cache)),
               "grouped scorer cache write");
    const float score_scale = 0.125f;
    require_ok(ds4_gpu_glm53_indexer_scores_batch_tensor(
        scores_gpu, score_q_gpu, score_weights_gpu, score_cache_gpu,
        SCORE_ROWS, SCORE_TOKENS, SCORE_POS0, POOL,
        HEADS, D, score_scale, false), "GLM-5.3 grouped indexer scores");
    float scores_actual[SCORE_TOKENS * SCORE_ROWS];
    require_ok(ds4_gpu_tensor_read(scores_gpu, 0, scores_actual,
                                   sizeof(scores_actual)),
               "grouped scorer output read");
    for (uint32_t t = 0; t < SCORE_TOKENS; t++) {
        const uint32_t visible = (SCORE_POS0 + t + 1u) / POOL;
        for (uint32_t row = 0; row < SCORE_ROWS; row++) {
            const float actual_score = scores_actual[t * SCORE_ROWS + row];
            if (row >= visible) {
                if (!isinf(actual_score) || actual_score >= 0.0f) {
                    fprintf(stderr,
                            "grouped scorer row %u token %u should be hidden\n",
                            row, t);
                    return 1;
                }
                continue;
            }
            float expected_score = 0.0f;
            for (uint32_t h = 0; h < HEADS; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; d++) {
                    dot += score_q[((uint64_t)t * HEADS + h) * D + d] *
                           score_cache[row * D + d];
                }
                expected_score += score_weights[t * HEADS + h] * dot;
            }
            require_close("GLM-5.3 grouped indexer score", actual_score,
                          expected_score * score_scale, 2e-5f);
        }
    }
    ds4_gpu_tensor_free(scores_gpu);
    ds4_gpu_tensor_free(score_cache_gpu);
    ds4_gpu_tensor_free(score_weights_gpu);
    ds4_gpu_tensor_free(score_q_gpu);

    enum { SELECTED_POOLS = 512, INDEX_TOPK = 2048, SELECT_ROWS = 5,
           SELECT_WIDTH = 2051 };
    uint32_t pool_ids[SELECT_ROWS * SELECTED_POOLS];
    for (uint32_t t = 0; t < SELECT_ROWS; t++) {
        for (uint32_t i = 0; i < SELECTED_POOLS; i++) {
            pool_ids[t * SELECTED_POOLS + i] = SELECTED_POOLS - 1u - i;
        }
    }
    ds4_gpu_tensor *pool_ids_gpu = ds4_gpu_tensor_alloc(sizeof(pool_ids));
    ds4_gpu_tensor *raw_ids_gpu =
        ds4_gpu_tensor_alloc((uint64_t)SELECT_ROWS * SELECT_WIDTH * sizeof(uint32_t));
    require_ok(pool_ids_gpu && raw_ids_gpu, "pool selection tensor allocation");
    require_ok(ds4_gpu_tensor_write(pool_ids_gpu, 0, pool_ids, sizeof(pool_ids)),
               "pool selection write");
    require_ok(ds4_gpu_glm53_expand_pool_selection_tensor(
        raw_ids_gpu, pool_ids_gpu, SELECT_ROWS, INDEX_TOPK,
        SELECTED_POOLS, INDEX_TOPK, POOL, SELECT_WIDTH),
        "pool selection expansion");
    uint32_t raw_ids[SELECT_ROWS * SELECT_WIDTH];
    require_ok(ds4_gpu_tensor_read(raw_ids_gpu, 0, raw_ids, sizeof(raw_ids)),
               "pool selection read");
    for (uint32_t t = 0; t < SELECT_ROWS; t++) {
        for (uint32_t i = 0; i < INDEX_TOPK; i++) {
            const uint32_t p = pool_ids[t * SELECTED_POOLS + i / POOL];
            if (raw_ids[t * SELECT_WIDTH + i] != p * POOL + i % POOL) {
                fprintf(stderr, "pool expansion mismatch at row %u slot %u\n", t, i);
                return 1;
            }
        }
        const uint32_t visible = INDEX_TOPK + t + 1u;
        const uint32_t tail_count = visible % POOL;
        for (uint32_t i = 0; i < POOL - 1u; i++) {
            const uint32_t expected_id =
                i < tail_count ? visible - tail_count + i : UINT32_MAX;
            if (raw_ids[t * SELECT_WIDTH + INDEX_TOPK + i] != expected_id) {
                fprintf(stderr, "pool tail mismatch at row %u slot %u\n", t, i);
                return 1;
            }
        }
    }
    ds4_gpu_tensor_free(raw_ids_gpu);
    ds4_gpu_tensor_free(pool_ids_gpu);
    ds4_gpu_tensor_free(pool_gate_gpu);
    ds4_gpu_tensor_free(pool_raw_gpu);
    ds4_gpu_tensor_free(pool_tail_gate);
    ds4_gpu_tensor_free(pool_tail_k);
    ds4_gpu_tensor_free(pool_cache);

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *beta = ds4_gpu_tensor_alloc(HEADS * sizeof(float));
    ds4_gpu_tensor *output_gate = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(PROJECTION * sizeof(float));
    ds4_gpu_tensor *conv = ds4_gpu_tensor_alloc(9u * PROJECTION * sizeof(float));
    ds4_gpu_tensor *state = ds4_gpu_tensor_alloc((uint64_t)HEADS * D * D * sizeof(float));
    require_ok(q && k && v && gate && beta && output_gate && out && conv && state,
               "decode tensor allocation");

    float ones[PROJECTION], zeros[PROJECTION], beta_zero[HEADS];
    for (uint32_t i = 0; i < PROJECTION; i++) {
        ones[i] = 1.0f;
        zeros[i] = 0.0f;
    }
    for (uint32_t i = 0; i < HEADS; i++) beta_zero[i] = 0.0f;
    require_ok(ds4_gpu_tensor_write(q, 0, ones, sizeof(ones)), "Q write");
    require_ok(ds4_gpu_tensor_write(k, 0, ones, sizeof(ones)), "K write");
    require_ok(ds4_gpu_tensor_write(v, 0, ones, sizeof(ones)), "V write");
    require_ok(ds4_gpu_tensor_write(gate, 0, zeros, sizeof(zeros)), "gate write");
    require_ok(ds4_gpu_tensor_write(output_gate, 0, zeros, sizeof(zeros)), "output gate write");
    require_ok(ds4_gpu_tensor_write(beta, 0, beta_zero, sizeof(beta_zero)), "beta write");
    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, 9u * PROJECTION), "conv clear");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, (uint64_t)HEADS * D * D), "state clear");
    require_ok(ds4_gpu_glm53_kda_decode(
        out, conv, state, q, k, v, gate, beta, output_gate,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        HEADS, 1, -5.0f, 1e-5f), "KDA decode");
    float actual[PROJECTION];
    require_ok(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)), "output read");
    const float silu_one = 1.0f / (1.0f + expf(-1.0f));
    const float raw = 0.5f * silu_one / sqrtf((float)D);
    const float expected = 0.5f * raw / sqrtf(raw * raw + 1e-5f);
    for (uint32_t i = 0; i < PROJECTION; i++)
        require_close("KDA decode", actual[i], expected, 2e-5f);

    float qs[TOKENS * PROJECTION], ks[TOKENS * PROJECTION];
    float vs[TOKENS * PROJECTION], gates[TOKENS * PROJECTION];
    float output_gates[TOKENS * PROJECTION], betas[TOKENS * HEADS];
    for (uint32_t t = 0; t < TOKENS; t++) {
        for (uint32_t h = 0; h < HEADS; h++)
            betas[t * HEADS + h] = -0.25f + 0.2f * (float)t + 0.1f * (float)h;
        for (uint32_t d = 0; d < PROJECTION; d++) {
            const uint32_t i = t * PROJECTION + d;
            qs[i] = 0.1f + 0.002f * (float)(d % 17u) + 0.03f * (float)t;
            ks[i] = -0.08f + 0.001f * (float)(d % 23u) + 0.02f * (float)t;
            vs[i] = 0.05f - 0.0015f * (float)(d % 13u) + 0.04f * (float)t;
            gates[i] = -0.2f + 0.003f * (float)(d % 11u);
            output_gates[i] = 0.15f - 0.002f * (float)(d % 7u);
        }
    }

    require_ok(ds4_gpu_tensor_fill_f32(conv, 0.0f, 9u * PROJECTION), "decode conv reset");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, (uint64_t)HEADS * D * D), "decode state reset");
    float decode_outputs[TOKENS * PROJECTION];
    for (uint32_t t = 0; t < TOKENS; t++) {
        const uint32_t off = t * PROJECTION;
        require_ok(ds4_gpu_tensor_write(q, 0, qs + off, PROJECTION * sizeof(float)), "decode Q write");
        require_ok(ds4_gpu_tensor_write(k, 0, ks + off, PROJECTION * sizeof(float)), "decode K write");
        require_ok(ds4_gpu_tensor_write(v, 0, vs + off, PROJECTION * sizeof(float)), "decode V write");
        require_ok(ds4_gpu_tensor_write(gate, 0, gates + off, PROJECTION * sizeof(float)), "decode gate write");
        require_ok(ds4_gpu_tensor_write(output_gate, 0, output_gates + off, PROJECTION * sizeof(float)), "decode output gate write");
        require_ok(ds4_gpu_tensor_write(beta, 0, betas + t * HEADS, HEADS * sizeof(float)), "decode beta write");
        require_ok(ds4_gpu_glm53_kda_decode(
            out, conv, state, q, k, v, gate, beta, output_gate,
            model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
            A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
            HEADS, 1, -5.0f, 1e-5f), "consistency decode");
        require_ok(ds4_gpu_tensor_read(out, 0, decode_outputs + off,
                                       PROJECTION * sizeof(float)), "decode output read");
    }

    ds4_gpu_tensor *pq = ds4_gpu_tensor_alloc(sizeof(qs));
    ds4_gpu_tensor *pk = ds4_gpu_tensor_alloc(sizeof(ks));
    ds4_gpu_tensor *pv = ds4_gpu_tensor_alloc(sizeof(vs));
    ds4_gpu_tensor *pg = ds4_gpu_tensor_alloc(sizeof(gates));
    ds4_gpu_tensor *poutput_gate = ds4_gpu_tensor_alloc(sizeof(output_gates));
    ds4_gpu_tensor *pbeta = ds4_gpu_tensor_alloc(sizeof(betas));
    ds4_gpu_tensor *pout = ds4_gpu_tensor_alloc(sizeof(qs));
    ds4_gpu_tensor *pconv = ds4_gpu_tensor_alloc(9u * PROJECTION * sizeof(float));
    ds4_gpu_tensor *pstate = ds4_gpu_tensor_alloc((uint64_t)HEADS * D * D * sizeof(float));
    require_ok(pq && pk && pv && pg && poutput_gate && pbeta && pout && pconv && pstate,
               "prefill tensor allocation");
    require_ok(ds4_gpu_tensor_write(pq, 0, qs, sizeof(qs)), "prefill Q write");
    require_ok(ds4_gpu_tensor_write(pk, 0, ks, sizeof(ks)), "prefill K write");
    require_ok(ds4_gpu_tensor_write(pv, 0, vs, sizeof(vs)), "prefill V write");
    require_ok(ds4_gpu_tensor_write(pg, 0, gates, sizeof(gates)), "prefill gate write");
    require_ok(ds4_gpu_tensor_write(poutput_gate, 0, output_gates, sizeof(output_gates)), "prefill output gate write");
    require_ok(ds4_gpu_tensor_write(pbeta, 0, betas, sizeof(betas)), "prefill beta write");
    require_ok(ds4_gpu_tensor_fill_f32(pconv, 0.0f, 9u * PROJECTION), "prefill conv clear");
    require_ok(ds4_gpu_tensor_fill_f32(pstate, 0.0f, (uint64_t)HEADS * D * D), "prefill state clear");
    require_ok(ds4_gpu_glm53_kda_prefill(
        pout, pconv, pstate, pq, pk, pv, pg, pbeta, poutput_gate,
        model, MODEL_BYTES, Q_CONV_OFFSET, K_CONV_OFFSET, V_CONV_OFFSET,
        A_LOG_OFFSET, DT_BIAS_OFFSET, NORM_OFFSET,
        HEADS, TOKENS, -5.0f, 1e-5f), "KDA prefill");
    float prefill_outputs[TOKENS * PROJECTION];
    require_ok(ds4_gpu_tensor_read(pout, 0, prefill_outputs, sizeof(prefill_outputs)),
               "prefill output read");
    for (uint32_t i = 0; i < TOKENS * PROJECTION; i++)
        require_close("KDA prefill/decode", prefill_outputs[i], decode_outputs[i], 5e-5f);

    ds4_gpu_tensor_free(pstate);
    ds4_gpu_tensor_free(pconv);
    ds4_gpu_tensor_free(pout);
    ds4_gpu_tensor_free(pbeta);
    ds4_gpu_tensor_free(poutput_gate);
    ds4_gpu_tensor_free(pg);
    ds4_gpu_tensor_free(pv);
    ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(state);
    ds4_gpu_tensor_free(conv);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(output_gate);
    ds4_gpu_tensor_free(beta);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(bf16_out);
    ds4_gpu_tensor_free(bf16_x);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("GLM-5.3 KDA GPU tests: PASS");
    return 0;
}
