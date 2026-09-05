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

#ifdef __APPLE__
/* Normal-range only, and truncating rather than rounding.  Both are fine here:
 * the compound-producer fixture uses values with at most seven explicit
 * mantissa bits, well inside the half normal range, so truncation to ten bits
 * is exact and the encoding round-trips. */
static uint16_t f32_to_f16(float value) {
    union { float f; uint32_t u; } b = { .f = value };
    const uint32_t sign = (b.u >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((b.u >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = (b.u >> 13) & 0x3ffu;
    if (exp <= 0 || exp >= 31) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float f16_to_f32(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t)(value >> 10) & 0x1fu;
    const uint32_t mant = (uint32_t)value & 0x3ffu;
    union { uint32_t u; float f; } b;
    b.u = exp == 0 ? sign : (sign | ((exp - 15u + 127u) << 23) | (mant << 13));
    return b.f;
}
#endif

/* Exercises ds4_gpu_glm53_matmul_bf16 at one width.  The reference is
 * accumulated in double and compared with a relative tolerance; a stride or
 * indexing error moves a result far more than that, which is what this is
 * here to catch. */
static void check_bf16_matmul(const uint8_t *model, size_t model_bytes,
                              uint64_t offset, uint32_t in_dim,
                              uint32_t out_dim, uint32_t rows,
                              const char *what) {
    uint16_t *w = (uint16_t *)(void *)((uint8_t *)(uintptr_t)model + offset);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            w[(size_t)o * in_dim + i] = f32_to_bf16(
                0.002f * (float)((int)(o % 11u) - 5) +
                0.001f * (float)((int)(i % 13u) - 6));
        }
    }
    const size_t x_bytes   = (size_t)rows * in_dim * sizeof(float);
    const size_t out_bytes = (size_t)rows * out_dim * sizeof(float);
    float *x        = malloc(x_bytes);
    float *expected = malloc(out_bytes);
    float *actual   = malloc(out_bytes);
    require_ok(x && expected && actual, "wide BF16 host allocation");
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            x[(size_t)r * in_dim + i] =
                0.02f * (float)((int)(i % 17u) - 8) + 0.005f * (float)r;
        }
        for (uint32_t o = 0; o < out_dim; o++) {
            double sum = 0.0;
            for (uint32_t i = 0; i < in_dim; i++) {
                sum += (double)bf16_to_f32(w[(size_t)o * in_dim + i]) *
                       (double)x[(size_t)r * in_dim + i];
            }
            expected[(size_t)r * out_dim + o] = (float)sum;
        }
    }
    ds4_gpu_tensor *gx   = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *gout = ds4_gpu_tensor_alloc(out_bytes);
    require_ok(gx && gout, "wide BF16 tensor allocation");
    require_ok(ds4_gpu_tensor_write(gx, 0, x, x_bytes), "wide BF16 input write");

    require_ok(ds4_gpu_glm53_matmul_bf16(gout, model, model_bytes, offset,
                                         in_dim, out_dim, gx, 1), what);
    require_ok(ds4_gpu_tensor_read(gout, 0, actual, out_dim * sizeof(float)),
               "wide BF16 decode output read");
    for (uint32_t o = 0; o < out_dim; o++) {
        require_close(what, actual[o], expected[o],
                      2e-5f * (fabsf(expected[o]) + 1.0f));
    }

    require_ok(ds4_gpu_glm53_matmul_bf16(gout, model, model_bytes, offset,
                                         in_dim, out_dim, gx, rows), what);
    require_ok(ds4_gpu_tensor_read(gout, 0, actual, out_bytes),
               "wide BF16 prefill output read");
    for (uint32_t i = 0; i < rows * out_dim; i++) {
        require_close(what, actual[i], expected[i],
                      2e-5f * (fabsf(expected[i]) + 1.0f));
    }
    ds4_gpu_tensor_free(gx);
    ds4_gpu_tensor_free(gout);
    free(x);
    free(expected);
    free(actual);
}

#ifdef __APPLE__
static void require_prefill_dispatch(uint32_t feature, bool expected,
                                     const char *what) {
    const uint32_t dispatched = ds4_gpu_test_glm53_prefill_take_dispatches();
    require_ok(dispatched == (expected ? feature : 0u), what);
}

/* Exactness oracle for the GLM 5.3 Flash prefill qk-low token tile.
 *
 * kernel_glm_qk_lowrank_q8_0_batch_t<TT> changes only which threadgroup
 * computes which outputs and how many tokens one thread carries; every output
 * keeps the reference kernel's expression, block order and column order.  So
 * each tile must reproduce kernel_glm_qk_lowrank_q8_0_batch bit for bit at the
 * model's shape, including the partial tail tile that a 1596-token chunk and a
 * 33- or 1-token prompt produce.  The dispatch runs through the same selection
 * the graph uses; DS4_METAL_DISABLE_GLM53_PREFILL_QK_LOW picks the reference
 * and DS4_METAL_GLM53_PREFILL_QK_LOW_TILE picks the tile. */
static void check_glm53_kda_inputs(uint8_t *model, uint64_t model_bytes,
                                  uint64_t offset) {
    const uint32_t widths[6] = {8192, 8192, 8192, 128, 128, 64};
    uint64_t offsets[6], end = offset;
    ds4_gpu_tensor *out[6];
    float *reference[6];
    for (unsigned slot = 0; slot < 6; slot++) {
        offsets[slot] = end;
        const uint64_t count = (uint64_t)4096u * widths[slot];
        end += count * sizeof(uint16_t);
        require_ok(end <= model_bytes, "KDA input fixture range");
        uint16_t *w = (uint16_t *)(void *)(model + offsets[slot]);
        for (uint64_t i = 0; i < count; i++) {
            const uint32_t h = (uint32_t)(i + slot * 8191u) * 1664525u + 1013904223u;
            w[i] = (uint16_t)((0x3b00u + h % 2048u) | ((h & 1u) << 15));
            if (i % 127u == 0) w[i] = (uint16_t)((h & 0x8000u) | (h % 128u));
        }
        out[slot] = ds4_gpu_tensor_alloc(widths[slot] * sizeof(float));
        reference[slot] = malloc(widths[slot] * sizeof(float));
        require_ok(out[slot] && reference[slot], "KDA input output buffers");
    }
    float input[4096], actual[8192];
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    require_ok(x != NULL, "KDA input activation buffer");
    for (unsigned fixture = 0; fixture < 4; fixture++) {
        for (uint32_t i = 0; i < 4096; i++) {
            input[i] = (float)((int)((i * 1664525u + 1013904223u) % 4097u) - 2048) * 0.00123f;
            if (fixture == 1) input[i] = (i & 1u) ? -0.0f : 0.0f;
            if (fixture == 2) input[i] *= 100000.0f;
            if (fixture == 3) input[i] = (i & 1u) ? -0.00003f : 0.00003f;
        }
        require_ok(ds4_gpu_tensor_write(x, 0, input, sizeof(input)), "KDA input activation write");
        for (unsigned arm = 0; arm < 4; arm++) {
            if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_KDA_INPUTS", "1", 1);
            if (arm == 3) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
            for (unsigned i = 0; i < 6; i++)
                require_ok(ds4_gpu_tensor_fill_f32(out[i], -17.0f, widths[i]), "KDA input poison");
            const int fused = arm != 0 && ds4_gpu_glm53_kda_inputs_bf16(
                out, offsets, model, model_bytes, x);
            require_ok(fused == (arm == 1), "KDA input policy selection");
            if (!fused) {
                for (unsigned i = 0; i < 6; i++)
                    require_ok(ds4_gpu_glm53_matmul_bf16(out[i], model, model_bytes,
                        offsets[i], 4096, widths[i], x, 1), "KDA input separate reference");
            }
            require_prefill_dispatch(DS4_GPU_GLM53_KDA_INPUTS, arm == 1, "KDA input coverage");
            for (unsigned i = 0; i < 6; i++) {
                require_ok(ds4_gpu_tensor_read(out[i], 0, actual, widths[i] * sizeof(float)),
                           "KDA input output read");
                if (arm == 0) memcpy(reference[i], actual, widths[i] * sizeof(float));
                else require_ok(memcmp(reference[i], actual, widths[i] * sizeof(float)) == 0,
                                "KDA input bitwise six projections");
            }
            unsetenv("DS4_METAL_DISABLE_GLM53_KDA_INPUTS");
            unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
        }
    }
    const uint64_t valid = offsets[5];
    offsets[5] = model_bytes - 1u;
    require_ok(!ds4_gpu_glm53_kda_inputs_bf16(out, offsets, model, model_bytes, x),
               "KDA input invalid last weight range");
    offsets[5] = valid;
    ds4_gpu_tensor *last = out[5]; out[5] = NULL;
    require_ok(!ds4_gpu_glm53_kda_inputs_bf16(out, offsets, model, model_bytes, x),
               "KDA input missing last output");
    out[5] = last;
    require_prefill_dispatch(DS4_GPU_GLM53_KDA_INPUTS, false, "KDA input invalid range coverage");
    for (unsigned i = 0; i < 6; i++) {
        ds4_gpu_tensor_free(out[i]); free(reference[i]);
    }
    ds4_gpu_tensor_free(x);
}

/* Independent persistent histories catch a scheduling error even when the
 * current normalized output hides it. Also exercise the untuned shapes. */
static void check_glm53_kda_decode_values4(uint8_t *model, uint64_t model_bytes,
                                          uint64_t offset) {
    enum { H = 64, D = 128, P = H * D, Q = 0, K = P * 16,
           V = K + P * 16, A = V + P * 16, DT = A + H * 4,
           NORM = DT + P * 4 };
    require_ok(offset + NORM + D * sizeof(float) <= model_bytes,
               "KDA decode values fixture range");
    float *w = (float *)(void *)(model + offset);
    for (uint32_t i = 0; i < A / 4; i++)
        w[i] = (float)((int)((i * 1664525u + 1013904223u) % 1021u) - 510) * 0.001f;
    for (uint32_t i = 0; i < H; i++) w[A / 4 + i] = -1.5f + (float)(i % 7u) * 0.1f;
    for (uint32_t i = 0; i < P; i++) w[DT / 4 + i] = (float)((int)(i % 17u) - 8) * 0.1f;
    for (uint32_t i = 0; i < D; i++) w[NORM / 4 + i] = 0.8f + (float)(i % 23u) * 0.01f;
    const uint32_t shapes[][2] = {{64, 1}, {64, 2}, {2, 1}};
    for (unsigned shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        const uint32_t heads = shapes[shape][0], rows = shapes[shape][1];
        const uint32_t n = heads * D * rows;
        const uint64_t sizes[] = {n * sizeof(float), (uint64_t)n * 9 * sizeof(float),
                                  (uint64_t)n * D * sizeof(float)};
        float *actual = malloc(sizes[2]), *reference = malloc(sizes[2]);
        ds4_gpu_tensor *input[6], *out[3][3];
        require_ok(actual && reference, "KDA decode values host buffers");
        for (unsigned i = 0; i < 6; i++) {
            input[i] = ds4_gpu_tensor_alloc(sizes[0]);
            require_ok(input[i] != NULL, "KDA decode values input buffer");
        }
        for (unsigned b = 0; b < 3; b++) {
            for (uint64_t j = 0; j < sizes[b] / sizeof(float); j++)
                actual[j] = (float)((int)((j * 1664525u + 1013904223u) % 2047u) - 1023) * 0.0005f;
            for (unsigned arm = 0; arm < 3; arm++) {
                out[arm][b] = ds4_gpu_tensor_alloc(sizes[b]);
                require_ok(out[arm][b] != NULL, "KDA decode values output buffer");
                require_ok(ds4_gpu_tensor_write(out[arm][b], 0, actual, sizes[b]),
                           "KDA decode values initial state");
            }
        }
        for (unsigned step = 0; step < 16; step++) {
            for (unsigned i = 0; i < 6; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    const uint32_t h = (j + step * n + i * n * 7u) * 1664525u + 1013904223u;
                    actual[j] = step == 7 ? 0.0f : (float)((int)(h % 8191u) - 4095) * 0.0007f;
                }
                require_ok(ds4_gpu_tensor_write(input[i], 0, actual, sizes[0]),
                           "KDA decode values input write");
            }
            for (unsigned arm = 0; arm < 3; arm++) {
                if (arm == 0) setenv("DS4_METAL_DISABLE_GLM53_DECODE_KDA_VALUES4", "1", 1);
                else unsetenv("DS4_METAL_DISABLE_GLM53_DECODE_KDA_VALUES4");
                if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
                require_ok(ds4_gpu_tensor_fill_f32(out[arm][0], -17.0f, n),
                           "KDA decode values poison output");
                require_ok(ds4_gpu_glm53_kda_decode(
                    out[arm][0], out[arm][1], out[arm][2], input[0], input[1], input[2],
                    input[3], input[4], input[5], model, model_bytes, offset + Q, offset + K,
                    offset + V, offset + A, offset + DT, offset + NORM,
                    heads, rows, -5.0f, 1e-6f), "KDA decode values dispatch");
                require_prefill_dispatch(DS4_GPU_GLM53_DECODE_KDA_VALUES4,
                                         arm == 1 && shape == 0, "KDA decode values coverage");
                unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
            }
            for (unsigned b = 0; b < 3; b++) {
                require_ok(ds4_gpu_tensor_read(out[0][b], 0, reference, sizes[b]),
                           "KDA decode values reference read");
                for (unsigned arm = 1; arm < 3; arm++) {
                    require_ok(ds4_gpu_tensor_read(out[arm][b], 0, actual, sizes[b]),
                               "KDA decode values candidate read");
                    require_ok(memcmp(reference, actual, sizes[b]) == 0,
                               "KDA decode values bitwise output/history/state");
                }
            }
        }
        for (unsigned i = 0; i < 6; i++) ds4_gpu_tensor_free(input[i]);
        for (unsigned arm = 0; arm < 3; arm++)
            for (unsigned b = 0; b < 3; b++) ds4_gpu_tensor_free(out[arm][b]);
        free(actual); free(reference);
    }
}

static void check_glm53_bf16_short_rows(uint8_t *model, uint64_t model_bytes,
                                        uint64_t weight_offset) {
    enum { IN = 128, OUT = 8192 };
    require_ok(weight_offset + (uint64_t)IN * OUT * 2 <= model_bytes,
               "short BF16 fixture range");
    uint16_t *w = (uint16_t *)(void *)(model + weight_offset);
    for (uint32_t i = 0; i < IN * OUT; i++) {
        const uint32_t h = i * 1664525u + 1013904223u;
        w[i] = (uint16_t)((0x3b00u + h % 2048u) | ((h & 1u) << 15u));
    }
    float input[IN], reference[OUT], actual[OUT];
    for (uint32_t i = 0; i < IN; i++) input[i] = (float)((int)(i % 17u) - 8) * 0.012345f;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(actual));
    require_ok(x && out, "short BF16 buffers");
    require_ok(ds4_gpu_tensor_write(x, 0, input, sizeof(input)), "short BF16 input");
    require_ok(unsetenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_BF16_NSG4") == 0,
               "clear inherited BF16 grouping rollback");
    for (unsigned arm = 0; arm < 3; arm++) {
        if (arm == 0) setenv("DS4_METAL_DISABLE_GLM53_BF16_SHORT_ROWS", "1", 1);
        else unsetenv("DS4_METAL_DISABLE_GLM53_BF16_SHORT_ROWS");
        if (arm == 2) setenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING", "1", 1);
        require_ok(ds4_gpu_tensor_fill_f32(out, -17.0f, OUT), "short BF16 poison");
        require_ok(ds4_gpu_glm53_matmul_bf16(out, model, model_bytes,
                       weight_offset, IN, OUT, x, 1), "short BF16 dispatch");
        require_prefill_dispatch(DS4_GPU_GLM53_BF16_SHORT_ROWS, arm == 1,
                                 "short BF16 grouping coverage");
        require_ok(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)), "short BF16 read");
        if (arm == 0) memcpy(reference, actual, sizeof(actual));
        else require_ok(memcmp(reference, actual, sizeof(actual)) == 0,
                        "short BF16 bitwise output");
    }
    unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
}

/* Exercise cross-SIMD probability reads and selected-ID publication in the
 * shared GLM router, including the 512-thread GLM-5.3 sort. */
#endif /* __APPLE__: these oracles exercise Metal-specific dispatch switches. */

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
        /* Real GLM 5.3 widths: 4096 is kda_{q,k,v}, and 512/1024 the
         * low-rank gate projections. */
        WIDE512_OFFSET = 65536,   WIDE512_IN = 512,   WIDE512_OUT = 4,
        WIDE1024_OFFSET = 73728,  WIDE1024_IN = 1024, WIDE1024_OUT = 4,
        WIDE4096_OFFSET = 90112,  WIDE4096_IN = 4096, WIDE4096_OUT = 2,
        WIDE_ROWS = 3,
        /* Compare each compound HC producer against its unfused projection. */
        HC_N = 16384, HC_MIX = 24, HC_EMBD = 4096, HC_HC = 4,
        HC_F16W_OFFSET  = 131072,   /* HC_N * HC_MIX * 2 = 786432 */
        HC_BF16W_OFFSET = 917504,
        HC_SCALE_OFFSET = 1703936,  /* 3 floats  */
        HC_BASE_OFFSET  = 1703968,  /* 24 floats */
        HC_NORM_OFFSET  = 1704064,  /* 4096 floats */
        /* BF16 matvec + HC-expand epilogue fixture */
        FUSED_W_OFFSET = 1720448,   /* FUSED_IN * FUSED_OUT * 2 = 131072 */
        FUSED_IN = 1024, FUSED_OUT = 64, FUSED_HC = 4,
        /* Q8_0 value rows for the split-vs-generic attention check:
         * 16 heads x 8 values x 544 bytes = 69632 */
        SPLIT_V_OFFSET = 1851520,
        /* GLM 5.3 attn_k_b for the prefill qk-low oracle:
         * 64 heads x 512 rows x 272 bytes = 8912896 */
        QK_LOW_KB_OFFSET = 2097152,
        /* Synthetic Q4_K expert matrices for all tail sizes and an empty expert. */
        MOE_GATE_OFFSET = 11010048,
        MOE_UP_OFFSET   = MOE_GATE_OFFSET + 36 * 256 * 144,
        MOE_DOWN_OFFSET = MOE_UP_OFFSET + 36 * 256 * 144,
        KP_Q_OFFSET = MOE_DOWN_OFFSET + 36 * 256 * 144,
        KP_K_OFFSET = KP_Q_OFFSET + 64 * 128 * 4 * 4,
        KP_V_OFFSET = KP_K_OFFSET + 64 * 128 * 4 * 4,
        KP_A_OFFSET = KP_V_OFFSET + 64 * 128 * 4 * 4,
        KP_DT_OFFSET = KP_A_OFFSET + 64 * 4,
        KP_NORM_OFFSET = KP_DT_OFFSET + 64 * 128 * 4,
#ifdef __APPLE__
        KI_WEIGHT_OFFSET = KP_NORM_OFFSET + 128 * 4,
        MODEL_BYTES = KI_WEIGHT_OFFSET + 4096 * 24896 * 2,
#else
        MODEL_BYTES = KP_NORM_OFFSET + 128 * 4,
#endif
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

    /* BF16_IN above is 64; these cover the widths the model actually runs. */
    check_bf16_matmul(model, MODEL_BYTES, WIDE512_OFFSET, WIDE512_IN,
                      WIDE512_OUT, WIDE_ROWS, "BF16 matmul in_dim=512");
    check_bf16_matmul(model, MODEL_BYTES, WIDE1024_OFFSET, WIDE1024_IN,
                      WIDE1024_OUT, WIDE_ROWS, "BF16 matmul in_dim=1024");
    check_bf16_matmul(model, MODEL_BYTES, WIDE4096_OFFSET, WIDE4096_IN,
                      WIDE4096_OUT, WIDE_ROWS, "BF16 matmul in_dim=4096");

#ifdef __APPLE__
    /*
     * F16 and BF16 have different reference matvec reductions. Each fused
     * producer must match its own four-dispatch chain, including every split
     * weight. Comparing the fused types to each other misses a shared error.
     */
    for (unsigned fixture = 0; fixture < 3; fixture++) {
        static const float exact_both[8] = {
            0.5f, -0.5f, 1.0f, -1.0f, 1.5f, -1.5f, 0.25f, -0.75f
        };
        uint16_t *hc_f16 = (uint16_t *)(model + HC_F16W_OFFSET);
        uint16_t *hc_bf16 = (uint16_t *)(model + HC_BF16W_OFFSET);
        for (uint32_t i = 0; i < (uint32_t)(HC_N * HC_MIX); i++) {
            const uint32_t hash = (i * 1664525u + 1013904223u) ^ (i >> 5u);
            const float w = exact_both[(hash >> 16u) % 8u] *
                (fixture == 0 ? 0.03125f : (i % 3u == 0 ? 0.25f : 0.00390625f));
            union { float f; uint32_t u; } b = { .f = w };
            hc_f16[i] = f32_to_f16(w);
            hc_bf16[i] = (uint16_t)(b.u >> 16);
            /* the encodings must round-trip to the same float, or the
             * comparison below would be measuring the fixture, not the kernel */
            require_close("HC fixture encoding", f16_to_f32(hc_f16[i]),
                          bf16_to_f32(hc_bf16[i]), 0.0f);
        }
        float *hc_scale = (float *)(model + HC_SCALE_OFFSET);
        for (int i = 0; i < 3; i++) hc_scale[i] = 0.5f + 0.25f * (float)i;
        float *hc_base = (float *)(model + HC_BASE_OFFSET);
        for (int i = 0; i < HC_MIX; i++) hc_base[i] = 0.125f * (float)((i % 5) - 2);
        float *hc_norm = (float *)(model + HC_NORM_OFFSET);
        for (int i = 0; i < HC_EMBD; i++) hc_norm[i] = 1.0f + 0.001f * (float)(i % 7);

        float *hc_x = malloc((size_t)HC_N * sizeof(float));
        require_ok(hc_x != NULL, "HC residual allocation");
        for (int i = 0; i < HC_N; i++) {
            const float x = 0.01f * (float)((i % 23) - 11) + 0.002f * (float)(i % 5);
            hc_x[i] = fixture == 2 ? 0.0f :
                x * (fixture == 1 && i % 7 == 0 ? 32.0f : 1.0f);
        }

        ds4_gpu_tensor *hc_res = ds4_gpu_tensor_alloc((size_t)HC_N * sizeof(float));
        require_ok(hc_res != NULL, "HC residual tensor");
        require_ok(ds4_gpu_tensor_write(hc_res, 0, hc_x,
                                        (size_t)HC_N * sizeof(float)),
                   "HC residual write");

        for (int pass = 0; pass < 2; pass++) {
            const uint32_t iters = fixture == 0 ? 1u : 20u;
            ds4_gpu_tensor *flat = ds4_gpu_tensor_alloc(HC_N * sizeof(float));
            ds4_gpu_tensor *ref_mix = ds4_gpu_tensor_alloc(HC_MIX * sizeof(float));
            ds4_gpu_tensor *ref_spl = ds4_gpu_tensor_alloc(HC_MIX * sizeof(float));
            ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(HC_EMBD * sizeof(float));
            ds4_gpu_tensor *ref_nrm = ds4_gpu_tensor_alloc(HC_EMBD * sizeof(float));
            ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(HC_MIX * sizeof(float));
            ds4_gpu_tensor *spl = ds4_gpu_tensor_alloc(HC_MIX * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(HC_EMBD * sizeof(float));
            ds4_gpu_tensor *nrm = ds4_gpu_tensor_alloc(HC_EMBD * sizeof(float));
            require_ok(flat && ref_mix && ref_spl && ref_out && ref_nrm &&
                       mix && spl && out && nrm, "HC output tensors");
            require_ok(ds4_gpu_begin_commands(), "HC reference begin");
            require_ok(ds4_gpu_rms_norm_plain_tensor(flat, hc_res, HC_N, 1.0e-6f), "HC reference RMS");
            require_ok(pass == 0 ?
                ds4_gpu_matmul_f16_tensor(ref_mix, model, MODEL_BYTES, HC_F16W_OFFSET, HC_N, HC_MIX, flat, 1) :
                ds4_gpu_glm53_matmul_bf16(ref_mix, model, MODEL_BYTES, HC_BF16W_OFFSET, HC_N, HC_MIX, flat, 1),
                "HC reference projection");
            require_ok(ds4_gpu_hc_split_weighted_sum_tensor(ref_out, ref_spl, ref_mix, hc_res,
                model, MODEL_BYTES, HC_SCALE_OFFSET, HC_BASE_OFFSET, HC_EMBD, HC_HC, iters, 1.0e-3f),
                "HC reference split/collapse");
            require_ok(ds4_gpu_rms_norm_weight_tensor(ref_nrm, ref_out, model, MODEL_BYTES,
                HC_NORM_OFFSET, HC_EMBD, 1.0e-6f), "HC reference weighted RMS");
            /* Reuse the same completion counter in one command stream. */
            for (unsigned repeat = 0; repeat < 3; repeat++) {
                require_ok(ds4_gpu_tensor_fill_f32(spl, -17.0f, HC_MIX),
                           "HC poison split before counter reuse");
                const int rc = pass == 0
                    ? ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
                          mix, out, nrm, spl, hc_res, model, MODEL_BYTES,
                          HC_F16W_OFFSET, HC_SCALE_OFFSET, HC_BASE_OFFSET,
                          HC_NORM_OFFSET, HC_N, HC_MIX, HC_EMBD, HC_HC,
                          iters, 1.0e-6f, 1.0e-3f, 1.0e-6f)
                    : ds4_gpu_hc_rms_norm_mix_split_norm_bf16_tensor(
                          mix, out, nrm, spl, hc_res, model, MODEL_BYTES,
                          HC_BF16W_OFFSET, HC_SCALE_OFFSET, HC_BASE_OFFSET,
                          HC_NORM_OFFSET, HC_N, HC_MIX, HC_EMBD, HC_HC,
                          iters, 1.0e-6f, 1.0e-3f, 1.0e-6f);
                require_ok(rc > 0, pass == 0 ? "HC producer f16" : "HC producer bf16");
            }
            require_ok(ds4_gpu_end_commands(), "HC producer end");
            const ds4_gpu_tensor *actual[] = {mix, spl, out, nrm};
            const ds4_gpu_tensor *reference[] = {ref_mix, ref_spl, ref_out, ref_nrm};
            const uint32_t counts[] = {HC_MIX, HC_MIX, HC_EMBD, HC_EMBD};
            const char *names[] = {"mix", "split", "collapse", "pre-norm"};
            for (unsigned stage = 0; stage < 4; stage++) {
                float a[HC_EMBD], b[HC_EMBD];
                const size_t bytes = counts[stage] * sizeof(float);
                require_ok(ds4_gpu_tensor_read(actual[stage], 0, a, bytes) &&
                           ds4_gpu_tensor_read(reference[stage], 0, b, bytes), "HC readback");
                if (memcmp(a, b, bytes) != 0) {
                    for (uint32_t i = 0; i < counts[stage]; i++) {
                        if (memcmp(a + i, b + i, sizeof(float)) != 0) {
                            fprintf(stderr, "HC %s fixture %u %s[%u]: got %.9g, reference %.9g\n",
                                pass == 0 ? "F16" : "BF16", fixture, names[stage], i, a[i], b[i]);
                            break;
                        }
                    }
                    exit(1);
                }
            }
            ds4_gpu_tensor_free(mix);
            ds4_gpu_tensor_free(spl);
            ds4_gpu_tensor_free(out);
            ds4_gpu_tensor_free(nrm);
            ds4_gpu_tensor_free(flat);
            ds4_gpu_tensor_free(ref_mix);
            ds4_gpu_tensor_free(ref_spl);
            ds4_gpu_tensor_free(ref_out);
            ds4_gpu_tensor_free(ref_nrm);
        }
        ds4_gpu_tensor_free(hc_res);
        free(hc_x);
    }

    /*
     * BF16 matvec with the HC expansion folded into its epilogue must equal
     * the separate matvec followed by ds4_gpu_hc_expand_tensor, exactly.  The
     * fused kernel reuses the same row accumulation and repeats the expand
     * arithmetic in the same operand order, so anything but bit-identical
     * output is a bug -- most likely a stride or an index.
     */
    {
        uint16_t *fw = (uint16_t *)(model + FUSED_W_OFFSET);
        for (uint32_t o = 0; o < FUSED_OUT; o++) {
            for (uint32_t i = 0; i < FUSED_IN; i++) {
                fw[(size_t)o * FUSED_IN + i] = f32_to_bf16(
                    0.003f * (float)((int)((o * 7u + i) % 17u) - 8));
            }
        }
        float fx[FUSED_IN], fres[FUSED_HC * FUSED_OUT];
        float fpost[FUSED_HC], fcomb[FUSED_HC * FUSED_HC];
        for (int i = 0; i < FUSED_IN; i++)
            fx[i] = 0.01f * (float)((i % 19) - 9);
        for (int i = 0; i < FUSED_HC * FUSED_OUT; i++)
            fres[i] = 0.05f * (float)((i % 13) - 6);
        for (int i = 0; i < FUSED_HC; i++) fpost[i] = 0.25f + 0.125f * (float)i;
        for (int i = 0; i < FUSED_HC * FUSED_HC; i++)
            fcomb[i] = 0.1f * (float)((i % 7) - 3);

        ds4_gpu_tensor *tx   = ds4_gpu_tensor_alloc(sizeof(fx));
        ds4_gpu_tensor *tres = ds4_gpu_tensor_alloc(sizeof(fres));
        ds4_gpu_tensor *tpost = ds4_gpu_tensor_alloc(sizeof(fpost));
        ds4_gpu_tensor *tcomb = ds4_gpu_tensor_alloc(sizeof(fcomb));
        ds4_gpu_tensor *out_ref = ds4_gpu_tensor_alloc(FUSED_OUT * sizeof(float));
        ds4_gpu_tensor *hc_ref  = ds4_gpu_tensor_alloc(sizeof(fres));
        ds4_gpu_tensor *out_fus = ds4_gpu_tensor_alloc(FUSED_OUT * sizeof(float));
        ds4_gpu_tensor *hc_fus  = ds4_gpu_tensor_alloc(sizeof(fres));
        require_ok(tx && tres && tpost && tcomb && out_ref && hc_ref &&
                   out_fus && hc_fus, "fused epilogue tensors");
        require_ok(ds4_gpu_tensor_write(tx, 0, fx, sizeof(fx)) &&
                   ds4_gpu_tensor_write(tres, 0, fres, sizeof(fres)) &&
                   ds4_gpu_tensor_write(tpost, 0, fpost, sizeof(fpost)) &&
                   ds4_gpu_tensor_write(tcomb, 0, fcomb, sizeof(fcomb)),
                   "fused epilogue inputs");

        require_ok(ds4_gpu_glm53_matmul_bf16(
                       out_ref, model, MODEL_BYTES, FUSED_W_OFFSET,
                       FUSED_IN, FUSED_OUT, tx, 1),
                   "reference BF16 matvec");
        require_ok(ds4_gpu_hc_expand_tensor(hc_ref, out_ref, tres, tpost, tcomb,
                                            FUSED_OUT, FUSED_HC),
                   "reference HC expand");

        const int fused = ds4_gpu_glm53_matmul_bf16_hc_expand4(
            out_fus, hc_fus, model, MODEL_BYTES, FUSED_W_OFFSET,
            FUSED_IN, FUSED_OUT, tx, tres, tpost, tcomb, FUSED_HC);
        if (fused == 0) {
            fprintf(stderr,
                    "BF16 matvec + HC expand: not available on this device, skipped\n");
        } else {
            float a[FUSED_OUT], b[FUSED_OUT];
            float ha[FUSED_HC * FUSED_OUT], hb[FUSED_HC * FUSED_OUT];
            require_ok(ds4_gpu_tensor_read(out_ref, 0, a, sizeof(a)) &&
                       ds4_gpu_tensor_read(out_fus, 0, b, sizeof(b)) &&
                       ds4_gpu_tensor_read(hc_ref, 0, ha, sizeof(ha)) &&
                       ds4_gpu_tensor_read(hc_fus, 0, hb, sizeof(hb)),
                       "fused epilogue readback");
            for (int i = 0; i < FUSED_OUT; i++)
                require_close("fused epilogue block_out", b[i], a[i], 0.0f);
            for (int i = 0; i < FUSED_HC * FUSED_OUT; i++)
                require_close("fused epilogue hc stream", hb[i], ha[i], 0.0f);
        }
        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(tres);
        ds4_gpu_tensor_free(tpost);
        ds4_gpu_tensor_free(tcomb);
        ds4_gpu_tensor_free(out_ref);
        ds4_gpu_tensor_free(hc_ref);
        ds4_gpu_tensor_free(out_fus);
        ds4_gpu_tensor_free(hc_fus);
    }
#endif /* __APPLE__: fused HC producers and epilogues are Metal-only. */

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
#ifdef __APPLE__
    /* Never silently compare the reference with itself because the parent
     * process has disabled tuning. Each oracle asserts dispatch coverage. */
    require_ok(unsetenv("DS4_METAL_DISABLE_GLM53_FLASH_TUNING") == 0,
               "clear inherited aggregate tuning switch for kernel oracles");
    require_ok(unsetenv("DS4_METAL_DISABLE_GLM53_PREFILL_INDEXED_ATTN") == 0,
               "clear inherited attention tuning switch for kernel oracles");
    ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM53_PREFILL);
    ds4_gpu_test_glm53_prefill_take_dispatches();
    check_glm53_bf16_short_rows(model, MODEL_BYTES, QK_LOW_KB_OFFSET);
    check_glm53_kda_decode_values4(model, MODEL_BYTES, QK_LOW_KB_OFFSET);
    check_glm53_kda_inputs(model, MODEL_BYTES, KI_WEIGHT_OFFSET);
    ds4_gpu_test_set_flags(0);
#endif
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("GLM-5.3 KDA GPU tests: PASS");
    return 0;
}
