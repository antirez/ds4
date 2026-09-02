#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void *model_fixture_allocation;
static size_t model_fixture_size;
static void *moe_rows8_fixture_allocation;
static void *moe_q2_rows8_fixture_allocation;
static void *vision_fc2_fixture_allocation;
static void *dense_matvec_bench_allocation;

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
}

static void test_fast_path_capabilities(void) {
    ds4_qwen4_fast_path_caps caps;
    require_ok(ds4_gpu_qwen4_query_fast_path_caps(&caps),
               "Qwen fast-path capability query");
    require_ok(caps.version == DS4_QWEN4_FAST_PATH_CAPS_VERSION,
               "Qwen fast-path capability version");
    require_ok(caps.struct_size == sizeof(caps),
               "Qwen fast-path capability size");
    require_ok(caps.required_mask == DS4_QWEN4_FAST_METAL_REQUIRED_MASK,
               "Qwen mandatory Metal capability mask");
    require_ok(caps.missing_mask == 0u,
               caps.missing_reason[0] ? caps.missing_reason
                                      : "Qwen mandatory Metal capabilities");
    require_ok((caps.available_mask & DS4_QWEN4_FAST_EXACT_Q2_MOE) != 0u,
               "Qwen exact IQ2_XXS/Q2_K prefill capability");
    require_ok((caps.available_mask & DS4_QWEN4_FAST_DECODE_Q2) != 0u,
               "Qwen IQ2_XXS/Q2_K decode capability");

    const char *forced = "kernel_qwen4_qsa_score_m1_f32_bf16";
    require_ok(setenv("DS4_QWEN4_TEST_MISSING_PIPELINE", forced, 1) == 0,
               "set Qwen missing-pipeline override");
    require_ok(ds4_gpu_qwen4_query_fast_path_caps(&caps),
               "Qwen overridden fast-path capability query");
    require_ok((caps.missing_mask & DS4_QWEN4_FAST_QSA_M1) != 0u,
               "Qwen missing M=1 capability is reported");
    require_ok((caps.available_mask & DS4_QWEN4_FAST_QSA_STREAM_TOPK) != 0u,
               "Qwen unrelated streaming capability remains available");
    require_ok(strstr(caps.missing_reason, forced) != NULL,
               "Qwen missing pipeline reason");
    require_ok(unsetenv("DS4_QWEN4_TEST_MISSING_PIPELINE") == 0,
               "clear Qwen missing-pipeline override");

    const char *forced_q2 =
        "kernel_qwen4_moe_iq2_xxs_gate_up_rows8";
    require_ok(setenv("DS4_QWEN4_TEST_MISSING_PIPELINE", forced_q2, 1) == 0,
               "set Qwen Q2 missing-pipeline override");
    require_ok(ds4_gpu_qwen4_query_fast_path_caps(&caps),
               "Qwen Q2 overridden fast-path capability query");
    require_ok((caps.available_mask & DS4_QWEN4_FAST_EXACT_Q2_MOE) == 0u,
               "Qwen missing exact Q2 capability is reported");
    require_ok((caps.available_mask & DS4_QWEN4_FAST_DECODE_Q2) != 0u,
               "Qwen unrelated Q2 decode capability remains available");
    require_ok(strstr(caps.missing_reason, forced_q2) != NULL,
               "Qwen missing Q2 pipeline reason");
    require_ok(unsetenv("DS4_QWEN4_TEST_MISSING_PIPELINE") == 0,
               "clear Qwen Q2 missing-pipeline override");
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

static uint16_t f32_to_f16(float value) {
    _Float16 half = (_Float16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float f16_to_f32(uint16_t value) {
    _Float16 half;
    memcpy(&half, &value, sizeof(half));
    return (float)half;
}

static ds4_gpu_tensor *tensor_from(const void *data, size_t bytes) {
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    require_ok(tensor != NULL, "tensor allocation");
    require_ok(ds4_gpu_tensor_write(tensor, 0, data, bytes), "tensor upload");
    return tensor;
}

static void require_array_close(const char *what, const float *a,
                                const float *b, size_t count,
                                float abs_tol, float rel_tol) {
    float max_error = 0.0f;
    size_t max_at = 0;
    for (size_t i = 0; i < count; i++) {
        const float error = fabsf(a[i] - b[i]);
        const float tolerance = abs_tol + rel_tol * fabsf(b[i]);
        if (!isfinite(a[i]) || error > tolerance) {
            fprintf(stderr,
                    "FAIL: %s at %zu: got %.9g expected %.9g error %.9g tolerance %.9g\n",
                    what, i, a[i], b[i], error, tolerance);
            exit(1);
        }
        if (error > max_error) {
            max_error = error;
            max_at = i;
        }
    }
    printf("%s PASS count=%zu max_error=%.3g at=%zu\n",
           what, count, max_error, max_at);
}

static void require_array_close_stats(const char *what, const float *a,
                                      const float *b, size_t count,
                                      float abs_tol, float rel_tol) {
    float max_error = 0.0f;
    double error_sum = 0.0;
    size_t max_at = 0;
    for (size_t i = 0; i < count; i++) {
        const float error = fabsf(a[i] - b[i]);
        const float tolerance = abs_tol + rel_tol * fabsf(b[i]);
        if (!isfinite(a[i]) || error > tolerance) {
            fprintf(stderr,
                    "FAIL: %s at %zu: got %.9g expected %.9g error %.9g tolerance %.9g\n",
                    what, i, a[i], b[i], error, tolerance);
            exit(1);
        }
        error_sum += error;
        if (error > max_error) {
            max_error = error;
            max_at = i;
        }
    }
    printf("%s PASS count=%zu max_error=%.3g at=%zu mean_error=%.3g\n",
           what, count, max_error, max_at,
           count ? error_sum / (double)count : 0.0);
}

static void test_vision_gelu_tail(void) {
    static const float input[] = {
        -20.0f, -16.0f, -12.93887f, -10.0f, -8.0f, -4.0f,
        0.0f, 4.0f, 8.0f, 10.55229f, 16.0f, 20.0f,
    };
    enum { COUNT = sizeof(input) / sizeof(input[0]) };
    float expected[COUNT];
    float actual[COUNT];
    uint16_t bias[COUNT] = {0};
    for (size_t i = 0; i < COUNT; i++) {
        const float value = input[i];
        const float inner = 0.7978845608028654f *
            fmaf(0.044715f * value * value, value, value);
        expected[i] = 0.5f * value * (1.0f + tanhf(inner));
    }
    ds4_gpu_tensor *x = tensor_from(input, sizeof(input));
    ds4_gpu_tensor *b = tensor_from(bias, sizeof(bias));
    require_ok(ds4_gpu_qwen4_vision_gelu_tanh(x, b, COUNT),
               "Qwen vision GELU tail dispatch");
    require_ok(ds4_gpu_tensor_read(x, 0, actual, sizeof(actual)),
               "Qwen vision GELU tail readback");
    require_array_close("Qwen vision GELU finite tail", actual, expected,
                        COUNT, 2e-6f, 2e-6f);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(x);
}

typedef struct {
    uint16_t d;
    int8_t qs[32];
} test_block_q8_0;

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} test_block_q4_0;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} test_block_q4_k;

typedef struct {
    uint16_t d;
    uint16_t qs[32];
} test_block_iq2_xxs;

typedef struct {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
} test_block_q2_k;

_Static_assert(sizeof(test_block_q8_0) == 34u, "Q8_0 block size");
_Static_assert(sizeof(test_block_q4_0) == 18u, "Q4_0 block size");
_Static_assert(sizeof(test_block_q4_k) == 144u, "Q4_K block size");
_Static_assert(sizeof(test_block_iq2_xxs) == 66u, "IQ2_XXS block size");
_Static_assert(sizeof(test_block_q2_k) == 84u, "Q2_K block size");

enum {
    QWEN4_VISION_FC2_LOGICAL_DIM = 4304,
    QWEN4_VISION_FC2_PHYSICAL_DIM = 4320,
    QWEN4_VISION_FC2_OUT_DIM = 1152,
};
_Static_assert(QWEN4_VISION_FC2_PHYSICAL_DIM / 32u == 135u,
               "Qwen vision FC2 must use 135 Q8_0 blocks per row");
_Static_assert(QWEN4_VISION_FC2_PHYSICAL_DIM -
                   QWEN4_VISION_FC2_LOGICAL_DIM == 16u,
               "Qwen vision FC2 physical tail");

static void fill_q8_0_matrix(test_block_q8_0 *blocks,
                             uint32_t out_dim,
                             uint32_t in_dim,
                             uint32_t seed) {
    const uint32_t row_blocks = in_dim / 32u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < row_blocks; block++) {
            test_block_q8_0 *qb = blocks + (size_t)row * row_blocks + block;
            const float delta = 0.00625f * (float)(1u + ((row + block + seed) % 7u));
            qb->d = f32_to_f16(delta);
            for (uint32_t i = 0; i < 32u; i++)
                qb->qs[i] = (int8_t)((int)((row * 19u + block * 7u +
                    i * 11u + seed) % 251u) - 125);
        }
    }
}

static float q8_0_value(const test_block_q8_0 *matrix,
                        uint32_t in_dim,
                        uint32_t row,
                        uint32_t col) {
    const uint32_t row_blocks = in_dim / 32u;
    const test_block_q8_0 *qb = matrix +
        (size_t)row * row_blocks + col / 32u;
    return f16_to_f32(qb->d) * (float)qb->qs[col % 32u];
}

static void q8_0_matmul_reference(float *out,
                                  const test_block_q8_0 *matrix,
                                  const float *x,
                                  uint32_t in_dim,
                                  uint32_t out_dim,
                                  uint32_t rows) {
    for (uint32_t token = 0; token < rows; token++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < in_dim; k++)
                sum = fmaf(q8_0_value(matrix, in_dim, row, k),
                           x[(size_t)token * in_dim + k], sum);
            out[(size_t)token * out_dim + row] = sum;
        }
    }
}

static void fill_q4_0_matrix(test_block_q4_0 *blocks,
                             uint32_t out_dim,
                             uint32_t in_dim,
                             uint32_t seed) {
    const uint32_t row_blocks = in_dim / 32u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < row_blocks; block++) {
            test_block_q4_0 *qb = blocks +
                (size_t)row * row_blocks + block;
            qb->d = f32_to_f16(0.0125f *
                (float)(1u + ((row + block + seed) % 7u)));
            for (uint32_t i = 0; i < 16u; i++) {
                const uint8_t q0 = (uint8_t)((row * 11u + block * 5u +
                    i * 3u + seed) & 15u);
                const uint8_t q1 = (uint8_t)((row * 7u + block * 13u +
                    i * 9u + seed + 1u) & 15u);
                qb->qs[i] = (uint8_t)(q0 | (q1 << 4));
            }
        }
    }
}

static float q4_0_value(const test_block_q4_0 *matrix,
                        uint32_t in_dim,
                        uint32_t row,
                        uint32_t col) {
    const uint32_t row_blocks = in_dim / 32u;
    const test_block_q4_0 *qb = matrix +
        (size_t)row * row_blocks + col / 32u;
    const uint32_t within = col & 31u;
    const uint8_t packed = qb->qs[within & 15u];
    const uint8_t q = within < 16u ? packed & 15u : packed >> 4u;
    return f16_to_f32(qb->d) * ((float)q - 8.0f);
}

static void test_q8_0_matmul(void) {
    enum { IN = 256, OUT = 11, ROWS = 17 };
    test_block_q8_0 weights[OUT * (IN / 32)];
    float input[ROWS * IN];
    float expected[ROWS * OUT];
    float actual[ROWS * OUT];
    fill_q8_0_matrix(weights, OUT, IN, 3u);
    for (uint32_t i = 0; i < ROWS * IN; i++)
        input[i] = sinf((float)(i + 1u) * 0.013f) * 0.7f;
    q8_0_matmul_reference(expected, weights, input, IN, OUT, ROWS);

    ds4_gpu_tensor *wt = tensor_from(weights, sizeof(weights));
    ds4_gpu_tensor *xt = tensor_from(input, sizeof(input));
    ds4_gpu_tensor *yt = ds4_gpu_tensor_alloc(sizeof(actual));
    require_ok(yt != NULL, "Q8_0 output allocation");
    require_ok(setenv("DS4_QWEN4_Q8_0_EXACT_MIN_ROWS", "1", 1) == 0,
               "enable Q8_0 rows8 fixture");
    require_ok(ds4_gpu_qwen4_q8_0_matmul(
                   yt, wt, xt, IN, OUT, ROWS),
               "Qwen Q8_0 rows8 dispatch");
    require_ok(unsetenv("DS4_QWEN4_Q8_0_EXACT_MIN_ROWS") == 0,
               "clear Q8_0 rows8 fixture");
    require_ok(ds4_gpu_tensor_read(yt, 0, actual, sizeof(actual)),
               "Q8_0 rows8 readback");
    require_array_close("Qwen standard Q8_0 rows8", actual, expected,
                        ROWS * OUT, 3e-4f, 3e-4f);

    ds4_gpu_tensor *m1 = ds4_gpu_tensor_alloc(OUT * sizeof(float));
    require_ok(m1 != NULL, "Q8_0 M=1 output allocation");
    require_ok(ds4_gpu_qwen4_q8_0_matmul(m1, wt, xt, IN, OUT, 1u),
               "Qwen Q8_0 M=1 dispatch");
    require_ok(ds4_gpu_tensor_read(m1, 0, actual, OUT * sizeof(float)),
               "Q8_0 M=1 readback");
    require_array_close("Qwen standard Q8_0 M=1", actual, expected,
                        OUT, 3e-4f, 3e-4f);
    ds4_gpu_tensor_free(m1);
    ds4_gpu_tensor_free(yt);
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(wt);
}

static void moe_topk_reference(const float *scores, uint32_t experts,
                               uint32_t top_k, int32_t *selected,
                               float *weights) {
    float best[16];
    for (uint32_t slot = 0; slot < top_k; slot++) {
        best[slot] = -INFINITY;
        selected[slot] = -1;
    }
    float max_score = -INFINITY;
    for (uint32_t expert = 0; expert < experts; expert++) {
        const float score = scores[expert];
        if (score > max_score) max_score = score;
        uint32_t insert = top_k;
        for (uint32_t slot = 0; slot < top_k; slot++) {
            if (score > best[slot] ||
                (score == best[slot] && (int32_t)expert < selected[slot])) {
                insert = slot;
                break;
            }
        }
        if (insert < top_k) {
            for (uint32_t slot = top_k - 1u; slot > insert; slot--) {
                best[slot] = best[slot - 1u];
                selected[slot] = selected[slot - 1u];
            }
            best[insert] = score;
            selected[insert] = (int32_t)expert;
        }
    }
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < top_k; slot++) {
        weights[slot] = expf(best[slot] - max_score);
        sum += weights[slot];
    }
    for (uint32_t slot = 0; slot < top_k; slot++) weights[slot] /= sum;
}

static size_t align_size(size_t value, size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static void fill_q4_k_matrix(test_block_q4_k *blocks,
                             uint32_t out_dim,
                             uint32_t physical_in_dim,
                             uint32_t seed) {
    const uint32_t row_blocks = physical_in_dim / 256u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < row_blocks; block++) {
            test_block_q4_k *qb = blocks + (size_t)row * row_blocks + block;
            memset(qb, 0, sizeof(*qb));
            const float delta = 0.004f *
                (float)(1u + ((row + block + seed) % 5u));
            qb->d = f32_to_f16(delta);
            qb->dmin = f32_to_f16(0.0f);
            for (uint32_t i = 0; i < 4u; i++) qb->scales[i] = 1u;
            for (uint32_t i = 8u; i < 12u; i++) qb->scales[i] = 1u;
            for (uint32_t group = 0; group < 8u; group++) {
                for (uint32_t i = 0; i < 32u; i++) {
                    const uint8_t q = (uint8_t)(
                        (row * 3u + block * 5u + group * 7u +
                         i * 11u + seed) & 15u);
                    uint8_t *packed = &qb->qs[(group >> 1u) * 32u + i];
                    if (group & 1u) *packed |= (uint8_t)(q << 4u);
                    else *packed |= q;
                }
            }
        }
    }
}

static double monotonic_seconds(void) {
    struct timespec ts;
    require_ok(clock_gettime(CLOCK_MONOTONIC, &ts) == 0,
               "monotonic clock");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_double(const void *a, const void *b) {
    const double lhs = *(const double *)a;
    const double rhs = *(const double *)b;
    return (lhs > rhs) - (lhs < rhs);
}

static double bench_dense_matvec_batch(bool q4,
                                       void *model,
                                       size_t model_size,
                                       size_t weight_offset,
                                       ds4_gpu_tensor *x,
                                       ds4_gpu_tensor *out,
                                       uint32_t in_dim,
                                       uint32_t out_dim,
                                       uint32_t iterations) {
    const double start = monotonic_seconds();
    require_ok(ds4_gpu_begin_commands(), "dense matvec benchmark begin");
    for (uint32_t i = 0; i < iterations; i++) {
        if (q4) {
            require_ok(ds4_gpu_matmul_quant_tensor(
                           out, model, model_size, weight_offset, 12u,
                           in_dim, out_dim, x, 1u),
                       "Q4_K dense matvec benchmark dispatch");
        } else {
            require_ok(ds4_gpu_qwen4_q8_0_matmul_model(
                           out, model, model_size, weight_offset, x,
                           in_dim, out_dim, 1u),
                       "Q8_0 dense matvec benchmark dispatch");
        }
    }
    require_ok(ds4_gpu_end_commands(), "dense matvec benchmark end");
    return (monotonic_seconds() - start) * 1000.0 / (double)iterations;
}

static void benchmark_dense_q8_vs_q4(void) {
    static const struct {
        const char *name;
        uint32_t in_dim;
        uint32_t out_dim;
    } shapes[] = {
        {"GDN qkv", 2560u, 10240u},
        {"QSA q", 2560u, 12288u},
        {"GDN out", 6144u, 2560u},
    };
    enum { SAMPLES = 9, ITERATIONS = 64 };

    size_t allocation_size = 0u;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        const size_t q8_bytes = (size_t)shapes[i].out_dim *
            (shapes[i].in_dim / 32u) * sizeof(test_block_q8_0);
        const size_t q4_bytes = (size_t)shapes[i].out_dim *
            (shapes[i].in_dim / 256u) * sizeof(test_block_q4_k);
        const size_t needed = align_size(q8_bytes, 4096u) + q4_bytes;
        if (needed > allocation_size) allocation_size = needed;
    }
    allocation_size = align_size(allocation_size, 4096u);
    require_ok(posix_memalign(&dense_matvec_bench_allocation,
                              4096u, allocation_size) == 0,
               "dense matvec benchmark model allocation");
    memset(dense_matvec_bench_allocation, 0, allocation_size);
    require_ok(ds4_gpu_set_model_map(dense_matvec_bench_allocation,
                                     allocation_size),
               "dense matvec benchmark model registration");

    puts("Qwen dense decode matvec benchmark (batched command buffer):");
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        const uint32_t in_dim = shapes[shape].in_dim;
        const uint32_t out_dim = shapes[shape].out_dim;
        const size_t q8_bytes = (size_t)out_dim * (in_dim / 32u) *
            sizeof(test_block_q8_0);
        const size_t q4_offset = align_size(q8_bytes, 4096u);
        const size_t q4_bytes = (size_t)out_dim * (in_dim / 256u) *
            sizeof(test_block_q4_k);
        require_ok(q4_offset + q4_bytes <= allocation_size,
                   "dense matvec benchmark model bounds");
        test_block_q8_0 *q8 = dense_matvec_bench_allocation;
        test_block_q4_k *q4 = (test_block_q4_k *)(
            (uint8_t *)dense_matvec_bench_allocation + q4_offset);
        fill_q8_0_matrix(q8, out_dim, in_dim, 101u + (uint32_t)shape);
        fill_q4_k_matrix(q4, out_dim, in_dim, 151u + (uint32_t)shape);

        float *input = malloc((size_t)in_dim * sizeof(float));
        require_ok(input != NULL, "dense matvec benchmark input allocation");
        for (uint32_t i = 0; i < in_dim; i++)
            input[i] = sinf((float)(i + 1u) * 0.0031f);
        ds4_gpu_tensor *x = tensor_from(input, (size_t)in_dim * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
            (size_t)out_dim * sizeof(float));
        require_ok(out != NULL, "dense matvec benchmark output allocation");

        (void)bench_dense_matvec_batch(false,
                                       dense_matvec_bench_allocation,
                                       allocation_size, 0u, x, out,
                                       in_dim, out_dim, 2u);
        (void)bench_dense_matvec_batch(true,
                                       dense_matvec_bench_allocation,
                                       allocation_size, q4_offset, x, out,
                                       in_dim, out_dim, 2u);
        double q8_ms[SAMPLES], q4_ms[SAMPLES];
        for (uint32_t sample = 0; sample < SAMPLES; sample++) {
            if ((sample & 1u) == 0u) {
                q8_ms[sample] = bench_dense_matvec_batch(
                    false, dense_matvec_bench_allocation, allocation_size,
                    0u, x, out, in_dim, out_dim, ITERATIONS);
                q4_ms[sample] = bench_dense_matvec_batch(
                    true, dense_matvec_bench_allocation, allocation_size,
                    q4_offset, x, out, in_dim, out_dim, ITERATIONS);
            } else {
                q4_ms[sample] = bench_dense_matvec_batch(
                    true, dense_matvec_bench_allocation, allocation_size,
                    q4_offset, x, out, in_dim, out_dim, ITERATIONS);
                q8_ms[sample] = bench_dense_matvec_batch(
                    false, dense_matvec_bench_allocation, allocation_size,
                    0u, x, out, in_dim, out_dim, ITERATIONS);
            }
        }
        qsort(q8_ms, SAMPLES, sizeof(q8_ms[0]), compare_double);
        qsort(q4_ms, SAMPLES, sizeof(q4_ms[0]), compare_double);
        const double q8_median = q8_ms[SAMPLES / 2u];
        const double q4_median = q4_ms[SAMPLES / 2u];
        printf("  %-8s %u->%u: Q8_0 %.4f ms, Q4_K %.4f ms, %.2fx\n",
               shapes[shape].name, in_dim, out_dim,
               q8_median, q4_median, q8_median / q4_median);

        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(x);
        free(input);
    }
}

static double bench_moe_q4_batch(bool q4_0,
                                 void *model,
                                 size_t model_size,
                                 size_t gate_offset,
                                 size_t up_offset,
                                 size_t down_offset,
                                 ds4_gpu_tensor *x,
                                 ds4_gpu_tensor *selected,
                                 ds4_gpu_tensor *route,
                                 ds4_gpu_tensor *mid,
                                 ds4_gpu_tensor *out,
                                 uint32_t iterations) {
    enum {
        IN = 2560, FF = 640, OUT = 2560,
        EXPERTS = 10, TOP_K = 10,
    };
    const double start = monotonic_seconds();
    require_ok(ds4_gpu_begin_commands(), "MoE Q4 benchmark begin");
    for (uint32_t i = 0; i < iterations; i++) {
        const int ok = q4_0
            ? ds4_gpu_qwen4_moe_q4_0_model(
                  out, mid, x, selected, route, model, model_size,
                  gate_offset, up_offset, down_offset,
                  IN, FF, OUT, EXPERTS, TOP_K, 1u)
            : ds4_gpu_qwen4_moe_q4_k_model(
                  out, mid, x, selected, route, model, model_size,
                  gate_offset, up_offset, down_offset,
                  IN, FF, OUT, EXPERTS, TOP_K, 1u);
        require_ok(ok, q4_0 ? "Q4_0 MoE benchmark dispatch"
                            : "Q4_K MoE benchmark dispatch");
    }
    require_ok(ds4_gpu_end_commands(), "MoE Q4 benchmark end");
    return (monotonic_seconds() - start) * 1000.0 / (double)iterations;
}

static void benchmark_moe_q4_k_vs_q4_0(void) {
    enum {
        IN = 2560, FF = 640, DOWN = 768, OUT = 2560,
        EXPERTS = 10, TOP_K = 10, SAMPLES = 7, ITERATIONS = 48,
    };
    const size_t gate_bytes = (size_t)EXPERTS * FF * (IN / 256u) *
        sizeof(test_block_q4_k);
    const size_t down_bytes = (size_t)EXPERTS * OUT * (DOWN / 256u) *
        sizeof(test_block_q4_k);
    size_t cursor = 0u;
    const size_t q4k_gate = cursor;
    cursor = align_size(cursor + gate_bytes, 4096u);
    const size_t q4k_up = cursor;
    cursor = align_size(cursor + gate_bytes, 4096u);
    const size_t q4k_down = cursor;
    cursor = align_size(cursor + down_bytes, 4096u);
    const size_t q40_gate = cursor;
    cursor = align_size(cursor + gate_bytes, 4096u);
    const size_t q40_up = cursor;
    cursor = align_size(cursor + gate_bytes, 4096u);
    const size_t q40_down = cursor;
    cursor = align_size(cursor + down_bytes, 4096u);
    require_ok(posix_memalign(&dense_matvec_bench_allocation,
                              4096u, cursor) == 0,
               "Q4 expert benchmark model allocation");
    memset(dense_matvec_bench_allocation, 0, cursor);
    uint8_t *model = dense_matvec_bench_allocation;
    fill_q4_k_matrix((test_block_q4_k *)(model + q4k_gate),
                     EXPERTS * FF, IN, 101u);
    fill_q4_k_matrix((test_block_q4_k *)(model + q4k_up),
                     EXPERTS * FF, IN, 131u);
    fill_q4_k_matrix((test_block_q4_k *)(model + q4k_down),
                     EXPERTS * OUT, DOWN, 151u);
    fill_q4_0_matrix((test_block_q4_0 *)(model + q40_gate),
                     EXPERTS * FF, IN, 101u);
    fill_q4_0_matrix((test_block_q4_0 *)(model + q40_up),
                     EXPERTS * FF, IN, 131u);
    fill_q4_0_matrix((test_block_q4_0 *)(model + q40_down),
                     EXPERTS * OUT, DOWN, 151u);
    require_ok(ds4_gpu_set_model_map(model, cursor),
               "Q4 expert benchmark model registration");

    float input[IN];
    int32_t ids[TOP_K];
    float weights[TOP_K];
    for (uint32_t i = 0; i < IN; i++)
        input[i] = sinf((float)(i + 1u) * 0.0031f);
    for (uint32_t i = 0; i < TOP_K; i++) {
        ids[i] = (int32_t)i;
        weights[i] = 1.0f / (float)TOP_K;
    }
    ds4_gpu_tensor *x = tensor_from(input, sizeof(input));
    ds4_gpu_tensor *selected = tensor_from(ids, sizeof(ids));
    ds4_gpu_tensor *route = tensor_from(weights, sizeof(weights));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(
        (size_t)TOP_K * FF * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((size_t)OUT * sizeof(float));
    require_ok(mid && out, "Q4 expert benchmark tensor allocation");

    (void)bench_moe_q4_batch(false, model, cursor,
                             q4k_gate, q4k_up, q4k_down,
                             x, selected, route, mid, out, 2u);
    (void)bench_moe_q4_batch(true, model, cursor,
                             q40_gate, q40_up, q40_down,
                             x, selected, route, mid, out, 2u);
    double q4k_ms[SAMPLES], q40_ms[SAMPLES];
    for (uint32_t sample = 0; sample < SAMPLES; sample++) {
        if ((sample & 1u) == 0u) {
            q4k_ms[sample] = bench_moe_q4_batch(
                false, model, cursor, q4k_gate, q4k_up, q4k_down,
                x, selected, route, mid, out, ITERATIONS);
            q40_ms[sample] = bench_moe_q4_batch(
                true, model, cursor, q40_gate, q40_up, q40_down,
                x, selected, route, mid, out, ITERATIONS);
        } else {
            q40_ms[sample] = bench_moe_q4_batch(
                true, model, cursor, q40_gate, q40_up, q40_down,
                x, selected, route, mid, out, ITERATIONS);
            q4k_ms[sample] = bench_moe_q4_batch(
                false, model, cursor, q4k_gate, q4k_up, q4k_down,
                x, selected, route, mid, out, ITERATIONS);
        }
    }
    qsort(q4k_ms, SAMPLES, sizeof(q4k_ms[0]), compare_double);
    qsort(q40_ms, SAMPLES, sizeof(q40_ms[0]), compare_double);
    printf("Qwen routed MoE M=1 real-active geometry: "
           "Q4_K %.4f ms/layer, Q4_0 %.4f ms/layer, %.2fx\n",
           q4k_ms[SAMPLES / 2u], q40_ms[SAMPLES / 2u],
           q4k_ms[SAMPLES / 2u] / q40_ms[SAMPLES / 2u]);

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(route);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
}

static double bench_gdn_batch(ds4_gpu_tensor *out,
                              ds4_gpu_tensor *state,
                              const ds4_gpu_tensor *q,
                              const ds4_gpu_tensor *k,
                              const ds4_gpu_tensor *v,
                              const ds4_gpu_tensor *decay,
                              const ds4_gpu_tensor *beta,
                              uint32_t iterations) {
    enum { KEY_HEADS = 16, VALUE_HEADS = 48, HEAD_DIM = 128 };
    const double start = monotonic_seconds();
    require_ok(ds4_gpu_begin_commands(), "GDN benchmark begin");
    for (uint32_t i = 0; i < iterations; i++) {
        require_ok(ds4_gpu_qwen4_gdn_prefill(
                       out, state, q, k, v, decay, beta, NULL,
                       1u, KEY_HEADS, VALUE_HEADS, HEAD_DIM, 4u),
                   "GDN benchmark dispatch");
    }
    require_ok(ds4_gpu_end_commands(), "GDN benchmark end");
    return (monotonic_seconds() - start) * 1000.0 / (double)iterations;
}

static void benchmark_gdn_decode(void) {
    enum {
        KEY_HEADS = 16,
        VALUE_HEADS = 48,
        HEAD_DIM = 128,
        SAMPLES = 9,
        ITERATIONS = 128,
    };
    const size_t qk_elements = KEY_HEADS * HEAD_DIM;
    const size_t value_elements = VALUE_HEADS * HEAD_DIM;
    const size_t state_elements = VALUE_HEADS * HEAD_DIM * HEAD_DIM;
    float *q_data = malloc(qk_elements * sizeof(float));
    float *k_data = malloc(qk_elements * sizeof(float));
    float *v_data = malloc(value_elements * sizeof(float));
    float decay_data[VALUE_HEADS];
    float beta_data[VALUE_HEADS];
    require_ok(q_data && k_data && v_data, "GDN benchmark host allocation");
    for (size_t i = 0; i < qk_elements; i++) {
        q_data[i] = sinf((float)(i + 1u) * 0.003f) * 0.08f;
        k_data[i] = cosf((float)(i + 1u) * 0.005f) * 0.08f;
    }
    for (size_t i = 0; i < value_elements; i++)
        v_data[i] = sinf((float)(i + 1u) * 0.007f) * 0.1f;
    for (uint32_t i = 0; i < VALUE_HEADS; i++) {
        decay_data[i] = 0.97f;
        beta_data[i] = 0.11f;
    }
    ds4_gpu_tensor *q = tensor_from(q_data, qk_elements * sizeof(float));
    ds4_gpu_tensor *k = tensor_from(k_data, qk_elements * sizeof(float));
    ds4_gpu_tensor *v = tensor_from(v_data, value_elements * sizeof(float));
    ds4_gpu_tensor *decay = tensor_from(decay_data, sizeof(decay_data));
    ds4_gpu_tensor *beta = tensor_from(beta_data, sizeof(beta_data));
    ds4_gpu_tensor *state = ds4_gpu_tensor_alloc(
        state_elements * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        value_elements * sizeof(float));
    require_ok(state && out, "GDN benchmark device allocation");
    require_ok(ds4_gpu_tensor_fill_f32(state, 0.0f, state_elements),
               "GDN benchmark state clear");
    (void)bench_gdn_batch(out, state, q, k, v, decay, beta, 4u);
    double samples[SAMPLES];
    for (uint32_t sample = 0; sample < SAMPLES; sample++) {
        samples[sample] = bench_gdn_batch(
            out, state, q, k, v, decay, beta, ITERATIONS);
    }
    qsort(samples, SAMPLES, sizeof(samples[0]), compare_double);
    printf("Qwen GDN decode benchmark: %.4f ms/dispatch\n",
           samples[SAMPLES / 2u]);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(state);
    ds4_gpu_tensor_free(beta);
    ds4_gpu_tensor_free(decay);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    free(v_data);
    free(k_data);
    free(q_data);
}

static void q4_k_scale_min(const uint8_t scales[12], uint32_t group,
                           uint32_t *scale, uint32_t *minimum) {
    if (group < 4u) {
        *scale = scales[group] & 63u;
        *minimum = scales[group + 4u] & 63u;
    } else {
        *scale = (scales[group + 4u] & 15u) |
            ((scales[group - 4u] & 0xc0u) >> 2u);
        *minimum = (scales[group + 4u] >> 4u) |
            ((scales[group] & 0xc0u) >> 2u);
    }
}

static float q4_k_value(const test_block_q4_k *matrix,
                        uint32_t physical_in_dim,
                        uint32_t row,
                        uint32_t col) {
    const uint32_t row_blocks = physical_in_dim / 256u;
    const test_block_q4_k *qb = matrix +
        (size_t)row * row_blocks + col / 256u;
    const uint32_t within = col % 256u;
    const uint32_t group = within / 32u;
    const uint32_t i = within % 32u;
    const uint8_t packed = qb->qs[(group >> 1u) * 32u + i];
    const uint32_t q = (packed >> ((group & 1u) * 4u)) & 15u;
    uint32_t scale, minimum;
    q4_k_scale_min(qb->scales, group, &scale, &minimum);
    return f16_to_f32(qb->d) * (float)scale * (float)q -
           f16_to_f32(qb->dmin) * (float)minimum;
}

static uint8_t iq2_sign_mask(uint32_t code) {
    uint32_t parity = 0u;
    for (uint32_t bits = code; bits != 0u; bits >>= 1u)
        parity ^= bits & 1u;
    return (uint8_t)(code | (parity << 7u));
}

static void fill_iq2_xxs_matrix(test_block_iq2_xxs *blocks,
                                uint32_t out_dim,
                                uint32_t physical_in_dim,
                                uint32_t seed) {
    const uint32_t row_blocks = physical_in_dim / 256u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < row_blocks; block++) {
            test_block_iq2_xxs *qb =
                blocks + (size_t)row * row_blocks + block;
            memset(qb, 0, sizeof(*qb));
            qb->d = f32_to_f16(0.0005f *
                (float)(1u + ((row + 3u * block + seed) % 7u)));
            for (uint32_t group = 0; group < 8u; group++) {
                uint16_t *q = qb->qs + 4u * group;
                uint8_t *selectors = (uint8_t *)q;
                uint32_t aux =
                    ((row + block + group + seed) & 3u) << 28u;
                for (uint32_t sub = 0; sub < 4u; sub++) {
                    selectors[sub] = (uint8_t)(
                        (row + block + group + sub + seed) & 1u);
                    const uint32_t sign_code =
                        (row * 11u + block * 7u + group * 5u +
                         sub * 13u + seed) & 127u;
                    aux |= sign_code << (7u * sub);
                }
                q[2] = (uint16_t)aux;
                q[3] = (uint16_t)(aux >> 16u);
            }
        }
    }
}

static float iq2_xxs_value(const test_block_iq2_xxs *matrix,
                           uint32_t physical_in_dim,
                           uint32_t row,
                           uint32_t col) {
    const uint32_t row_blocks = physical_in_dim / 256u;
    const test_block_iq2_xxs *qb = matrix +
        (size_t)row * row_blocks + col / 256u;
    const uint32_t within = col % 256u;
    const uint32_t group = within / 32u;
    const uint32_t in_group = within % 32u;
    const uint32_t sub = in_group / 8u;
    const uint32_t j = in_group % 8u;
    const uint16_t *q = qb->qs + 4u * group;
    const uint8_t *selectors = (const uint8_t *)q;
    const uint32_t aux = (uint32_t)q[2] | ((uint32_t)q[3] << 16u);
    const uint32_t sign_code = (aux >> (7u * sub)) & 127u;
    const uint8_t signs = iq2_sign_mask(sign_code);
    /* Fixtures use the first two official IQ2_XXS grid rows. Grid 0 is
     * eight 8s; grid 1 is 43 followed by seven 8s on little-endian Metal. */
    const float grid = selectors[sub] == 1u && j == 0u ? 43.0f : 8.0f;
    const float sign = signs & (uint8_t)(1u << j) ? -1.0f : 1.0f;
    return f16_to_f32(qb->d) *
           (0.5f + (float)(aux >> 28u)) * 0.25f * grid * sign;
}

static void fill_q2_k_matrix(test_block_q2_k *blocks,
                             uint32_t out_dim,
                             uint32_t physical_in_dim,
                             uint32_t seed) {
    const uint32_t row_blocks = physical_in_dim / 256u;
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < row_blocks; block++) {
            test_block_q2_k *qb =
                blocks + (size_t)row * row_blocks + block;
            memset(qb, 0, sizeof(*qb));
            qb->d = f32_to_f16(0.001f *
                (float)(1u + ((row + block + seed) % 5u)));
            qb->dmin = f32_to_f16(0.0003f *
                (float)(1u + ((2u * row + block + seed) % 3u)));
            for (uint32_t group = 0; group < 16u; group++) {
                const bool poison_alignment_tail =
                    physical_in_dim == 768u && block == 2u && group >= 8u;
                const uint32_t scale = poison_alignment_tail ? 15u :
                    1u + ((row + block + group + seed) % 7u);
                const uint32_t minimum = poison_alignment_tail ? 0u :
                    (row + 2u * block + group + seed) % 4u;
                qb->scales[group] =
                    (uint8_t)(scale | (minimum << 4u));
                const uint32_t q_base =
                    32u * (group / 8u) + 16u * (group & 1u);
                const uint32_t shift = ((group / 2u) & 3u) * 2u;
                for (uint32_t j = 0; j < 16u; j++) {
                    const uint32_t q = poison_alignment_tail ? 3u :
                        (row * 3u + block * 5u + group * 7u +
                         j * 11u + seed) & 3u;
                    qb->qs[q_base + j] |= (uint8_t)(q << shift);
                }
            }
        }
    }
}

static float q2_k_value(const test_block_q2_k *matrix,
                        uint32_t physical_in_dim,
                        uint32_t row,
                        uint32_t col) {
    const uint32_t row_blocks = physical_in_dim / 256u;
    const test_block_q2_k *qb = matrix +
        (size_t)row * row_blocks + col / 256u;
    const uint32_t within = col % 256u;
    const uint32_t group = within / 16u;
    const uint32_t j = within % 16u;
    const uint32_t q_base =
        32u * (group / 8u) + 16u * (group & 1u);
    const uint32_t shift = ((group / 2u) & 3u) * 2u;
    const uint32_t q = (qb->qs[q_base + j] >> shift) & 3u;
    const uint32_t scale = qb->scales[group] & 15u;
    const uint32_t minimum = qb->scales[group] >> 4u;
    return f16_to_f32(qb->d) * (float)scale * (float)q -
           f16_to_f32(qb->dmin) * (float)minimum;
}

static void test_model_q8_0_paths(void) {
    enum {
        IN = 256, OUT = 13, ROWS = 3, VOCAB = 5,
        HC_HIDDEN = 9, HC_STREAMS = 4,
    };
    const size_t matrix_bytes = (size_t)OUT * (IN / 32u) *
        sizeof(test_block_q8_0);
    const size_t embedding_bytes = (size_t)VOCAB * (IN / 32u) *
        sizeof(test_block_q8_0);
    const size_t hc_bytes = (size_t)(HC_HIDDEN * HC_STREAMS) *
        (IN / 32u) * sizeof(test_block_q8_0);
    size_t cursor = 0u;
    const size_t a_offset = cursor;
    cursor = align_size(cursor + matrix_bytes, 64u);
    const size_t b_offset = cursor;
    cursor = align_size(cursor + matrix_bytes, 64u);
    const size_t embedding_offset = cursor;
    cursor = align_size(cursor + embedding_bytes, 64u);
    const size_t hc_offset = cursor;
    cursor = align_size(cursor + hc_bytes, 4096u);
    if (cursor < 131072u) cursor = 131072u;
    require_ok(posix_memalign(&model_fixture_allocation, 4096u, cursor) == 0,
               "Q8_0 model fixture allocation");
    model_fixture_size = cursor;
    memset(model_fixture_allocation, 0, cursor);
    uint8_t *model = model_fixture_allocation;
    test_block_q8_0 *a = (test_block_q8_0 *)(model + a_offset);
    test_block_q8_0 *b = (test_block_q8_0 *)(model + b_offset);
    test_block_q8_0 *embedding =
        (test_block_q8_0 *)(model + embedding_offset);
    test_block_q8_0 *hc = (test_block_q8_0 *)(model + hc_offset);
    fill_q8_0_matrix(a, OUT, IN, 17u);
    fill_q8_0_matrix(b, OUT, IN, 29u);
    fill_q8_0_matrix(embedding, VOCAB, IN, 41u);
    fill_q8_0_matrix(hc, HC_HIDDEN * HC_STREAMS, IN, 53u);
    require_ok(ds4_gpu_set_model_map(model, model_fixture_size),
               "Q8_0 model fixture registration");

    float input[ROWS * IN];
    uint16_t input_bf16[ROWS * IN];
    float rounded[ROWS * IN];
    for (uint32_t i = 0; i < ROWS * IN; i++) {
        input[i] = cosf((float)(i + 3u) * 0.017f) * 0.4f;
        input_bf16[i] = f32_to_bf16(input[i]);
        rounded[i] = bf16_to_f32(input_bf16[i]);
    }
    float expected[ROWS * OUT], actual[ROWS * OUT];
    q8_0_matmul_reference(expected, a, input, IN, OUT, ROWS);
    ds4_gpu_tensor *x = tensor_from(input, sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(actual));
    require_ok(out != NULL, "model Q8_0 output allocation");
    require_ok(ds4_gpu_qwen4_q8_0_matmul_model(
        out, model, model_fixture_size, a_offset, x, IN, OUT, ROWS),
        "model Q8_0 projection");
    require_ok(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)),
               "model Q8_0 readback");
    require_array_close("Qwen model standard Q8_0", actual, expected,
                        ROWS * OUT, 4e-4f, 4e-4f);

    float expected_bf16[ROWS * OUT];
    q8_0_matmul_reference(expected_bf16, a, rounded, IN, OUT, ROWS);
    ds4_gpu_tensor *xb = tensor_from(input_bf16, sizeof(input_bf16));
    require_ok(ds4_gpu_qwen4_q8_0_matmul_model_bf16(
        out, model, model_fixture_size, a_offset, xb, IN, OUT, ROWS),
        "model Q8_0 BF16-input projection");
    require_ok(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)),
               "model Q8_0 BF16-input readback");
    require_array_close("Qwen Q8_0 BF16-input", actual, expected_bf16,
                        ROWS * OUT, 4e-4f, 4e-4f);

    float a_m1[OUT], b_m1[OUT];
    q8_0_matmul_reference(a_m1, a, input, IN, OUT, 1u);
    q8_0_matmul_reference(b_m1, b, input, IN, OUT, 1u);
    ds4_gpu_tensor *oa = ds4_gpu_tensor_alloc(sizeof(a_m1));
    ds4_gpu_tensor *ob = ds4_gpu_tensor_alloc(sizeof(b_m1));
    require_ok(oa && ob, "Q8_0 paired output allocation");
    require_ok(ds4_gpu_qwen4_q8_0_pair_model(
        oa, ob, model, model_fixture_size, a_offset,
        model, model_fixture_size, b_offset, x, IN, OUT),
        "Q8_0 paired dispatch");
    require_ok(ds4_gpu_tensor_read(oa, 0, actual, sizeof(a_m1)) &&
               ds4_gpu_tensor_read(ob, 0, actual + OUT, sizeof(b_m1)),
               "Q8_0 paired readback");
    require_array_close("Qwen Q8_0 pair A", actual, a_m1,
                        OUT, 4e-4f, 4e-4f);
    require_array_close("Qwen Q8_0 pair B", actual + OUT, b_m1,
                        OUT, 4e-4f, 4e-4f);

    require_ok(ds4_gpu_qwen4_q8_0_concat_model(
        oa, ob, model, model_fixture_size, a_offset, OUT,
        model, model_fixture_size, b_offset, OUT, x, IN),
        "Q8_0 concatenated dispatch");
    require_ok(ds4_gpu_tensor_read(oa, 0, actual, sizeof(a_m1)) &&
               ds4_gpu_tensor_read(ob, 0, actual + OUT, sizeof(b_m1)),
               "Q8_0 concatenated readback");
    require_array_close("Qwen Q8_0 concat A", actual, a_m1,
                        OUT, 4e-4f, 4e-4f);
    require_array_close("Qwen Q8_0 concat B", actual + OUT, b_m1,
                        OUT, 4e-4f, 4e-4f);

    float activated[OUT];
    for (uint32_t i = 0; i < OUT; i++)
        activated[i] = a_m1[i] / (1.0f + expf(-a_m1[i]));
    require_ok(ds4_gpu_qwen4_q8_0_silu_model(
        oa, model, model_fixture_size, a_offset, x, IN, OUT, 1u),
        "Q8_0 SiLU dispatch");
    require_ok(ds4_gpu_tensor_read(oa, 0, actual, sizeof(activated)),
               "Q8_0 SiLU readback");
    require_array_close("Qwen Q8_0 fused SiLU", actual, activated,
                        OUT, 4e-4f, 4e-4f);
    for (uint32_t i = 0; i < OUT; i++) activated[i] *= b_m1[i];
    require_ok(ds4_gpu_qwen4_q8_0_swiglu_model(
        oa, model, model_fixture_size, a_offset,
        model, model_fixture_size, b_offset, x, IN, OUT, 1u),
        "Q8_0 SwiGLU dispatch");
    require_ok(ds4_gpu_tensor_read(oa, 0, actual, sizeof(activated)),
               "Q8_0 SwiGLU readback");
    require_array_close("Qwen Q8_0 fused SwiGLU", actual, activated,
                        OUT, 5e-4f, 5e-4f);

    const int32_t token_ids[] = {4, 1, -1};
    float embedding_expected[3 * IN];
    float embedding_actual[3 * IN];
    for (uint32_t token = 0; token < 3u; token++) {
        for (uint32_t dim = 0; dim < IN; dim++) {
            embedding_expected[token * IN + dim] = token_ids[token] < 0
                ? 0.0f : q8_0_value(embedding, IN,
                                    (uint32_t)token_ids[token], dim);
        }
    }
    ds4_gpu_tensor *ids = tensor_from(token_ids, sizeof(token_ids));
    ds4_gpu_tensor *emb = ds4_gpu_tensor_alloc(sizeof(embedding_actual));
    require_ok(emb != NULL, "Q8_0 embedding allocation");
    require_ok(ds4_gpu_qwen4_q8_0_embedding_model(
        emb, model, model_fixture_size, embedding_offset,
        ids, IN, VOCAB, 3u), "Q8_0 embedding dispatch");
    require_ok(ds4_gpu_tensor_read(emb, 0, embedding_actual,
                                   sizeof(embedding_actual)),
               "Q8_0 embedding readback");
    require_array_close("Qwen standard Q8_0 embedding",
                        embedding_actual, embedding_expected,
                        3u * IN, 1e-6f, 1e-6f);

    float normalized[HC_STREAMS * HC_HIDDEN];
    float hc_expected[HC_HIDDEN], hc_actual[HC_HIDDEN];
    for (uint32_t i = 0; i < HC_STREAMS * HC_HIDDEN; i++)
        normalized[i] = sinf((float)(i + 1u) * 0.09f);
    for (uint32_t dim = 0; dim < HC_HIDDEN; dim++) {
        float value = 0.0f;
        for (uint32_t stream = 0; stream < HC_STREAMS; stream++) {
            float raw = 0.0f;
            const uint32_t row = stream * HC_HIDDEN + dim;
            for (uint32_t k = 0; k < IN; k++)
                raw = fmaf(q8_0_value(hc, IN, row, k), input[k], raw);
            value = fmaf(normalized[row], 1.0f / (1.0f + expf(-raw)), value);
        }
        hc_expected[dim] = value / (float)HC_STREAMS;
    }
    ds4_gpu_tensor *norm = tensor_from(normalized, sizeof(normalized));
    ds4_gpu_tensor *mixed = ds4_gpu_tensor_alloc(sizeof(hc_actual));
    require_ok(mixed != NULL, "Q8_0 HC mix allocation");
    require_ok(ds4_gpu_qwen4_q8_0_hc_up_mix_model(
        mixed, norm, x, model, model_fixture_size, hc_offset,
        IN, HC_HIDDEN, 1u, HC_STREAMS), "Q8_0 HC up/mix dispatch");
    require_ok(ds4_gpu_tensor_read(mixed, 0, hc_actual, sizeof(hc_actual)),
               "Q8_0 HC up/mix readback");
    require_array_close("Qwen Q8_0 HC up/mix", hc_actual, hc_expected,
                        HC_HIDDEN, 5e-4f, 5e-4f);

    float inject_partials[HC_STREAMS * HC_STREAMS];
    float streams[HC_STREAMS * OUT];
    float streams_actual[HC_STREAMS * OUT];
    float streams_expected[HC_STREAMS * OUT];
    for (uint32_t i = 0; i < HC_STREAMS * HC_STREAMS; i++)
        inject_partials[i] = ((int)(i % 7u) - 3) * 0.08f;
    for (uint32_t i = 0; i < HC_STREAMS * OUT; i++)
        streams[i] = streams_expected[i] = ((int)(i % 11u) - 5) * 0.03f;
    for (uint32_t stream = 0; stream < HC_STREAMS; stream++) {
        float raw = 0.0f;
        for (uint32_t source = 0; source < HC_STREAMS; source++)
            raw += inject_partials[source * HC_STREAMS + stream];
        const float inject = 2.0f / (1.0f + expf(-raw));
        for (uint32_t dim = 0; dim < OUT; dim++)
            streams_expected[stream * OUT + dim] = fmaf(
                a_m1[dim], inject, streams_expected[stream * OUT + dim]);
    }
    ds4_gpu_tensor *partials = tensor_from(inject_partials,
                                            sizeof(inject_partials));
    ds4_gpu_tensor *stream_t = tensor_from(streams, sizeof(streams));
    require_ok(ds4_gpu_qwen4_q8_0_hc_write_model(
        stream_t, partials, model, model_fixture_size, a_offset,
        x, IN, OUT, 1u, HC_STREAMS), "Q8_0 HC write dispatch");
    require_ok(ds4_gpu_tensor_read(stream_t, 0, streams_actual,
                                   sizeof(streams_actual)),
               "Q8_0 HC write readback");
    require_array_close("Qwen Q8_0 HC write", streams_actual, streams_expected,
                        HC_STREAMS * OUT, 5e-4f, 5e-4f);

    /* Tiny-verifier fusion parity: the four row-1 fusions above must match
     * their standalone arithmetic for multi-row MTP batches as well.  The
     * references stay per-row CPU dot products over the same Q8_0 blocks. */
    {
        float a_rows[ROWS * OUT], b_rows[ROWS * OUT];
        q8_0_matmul_reference(a_rows, a, input, IN, OUT, ROWS);
        q8_0_matmul_reference(b_rows, b, input, IN, OUT, ROWS);
        ds4_gpu_tensor *o_rows =
            ds4_gpu_tensor_alloc((uint64_t)ROWS * OUT * sizeof(float));
        require_ok(o_rows != NULL, "Q8_0 multirow output allocation");

        float silu_rows[ROWS * OUT];
        for (uint32_t i = 0; i < ROWS * OUT; i++)
            silu_rows[i] = a_rows[i] / (1.0f + expf(-a_rows[i]));
        require_ok(ds4_gpu_qwen4_q8_0_silu_model(
            o_rows, model, model_fixture_size, a_offset,
            x, IN, OUT, ROWS), "Q8_0 SiLU multirow dispatch");
        require_ok(ds4_gpu_tensor_read(o_rows, 0, actual,
                                       (uint64_t)ROWS * OUT * sizeof(float)),
                   "Q8_0 SiLU multirow readback");
        require_array_close("Qwen Q8_0 fused SiLU rows>1", actual, silu_rows,
                            ROWS * OUT, 4e-4f, 4e-4f);

        float swiglu_rows[ROWS * OUT];
        for (uint32_t i = 0; i < ROWS * OUT; i++)
            swiglu_rows[i] = silu_rows[i] * b_rows[i];
        require_ok(ds4_gpu_qwen4_q8_0_swiglu_model(
            o_rows, model, model_fixture_size, a_offset,
            model, model_fixture_size, b_offset, x, IN, OUT, ROWS),
            "Q8_0 SwiGLU multirow dispatch");
        require_ok(ds4_gpu_tensor_read(o_rows, 0, actual,
                                       (uint64_t)ROWS * OUT * sizeof(float)),
                   "Q8_0 SwiGLU multirow readback");
        require_array_close("Qwen Q8_0 fused SwiGLU rows>1", actual,
                            swiglu_rows, ROWS * OUT, 5e-4f, 5e-4f);

        float normalized_rows[ROWS * HC_STREAMS * HC_HIDDEN];
        for (uint32_t i = 0; i < ROWS * HC_STREAMS * HC_HIDDEN; i++)
            normalized_rows[i] = sinf((float)(i + 5u) * 0.023f);
        float hc_rows_expected[ROWS * HC_HIDDEN];
        for (uint32_t row = 0; row < ROWS; row++) {
            for (uint32_t dim = 0; dim < HC_HIDDEN; dim++) {
                float value = 0.0f;
                for (uint32_t stream = 0; stream < HC_STREAMS; stream++) {
                    float raw = 0.0f;
                    const uint32_t hc_row = stream * HC_HIDDEN + dim;
                    for (uint32_t k = 0; k < IN; k++)
                        raw = fmaf(q8_0_value(hc, IN, hc_row, k),
                                   input[row * IN + k], raw);
                    value = fmaf(
                        normalized_rows[
                            row * HC_STREAMS * HC_HIDDEN +
                            stream * HC_HIDDEN + dim],
                        1.0f / (1.0f + expf(-raw)), value);
                }
                hc_rows_expected[row * HC_HIDDEN + dim] =
                    value / (float)HC_STREAMS;
            }
        }
        ds4_gpu_tensor *norm_rows = tensor_from(normalized_rows,
                                                sizeof(normalized_rows));
        ds4_gpu_tensor *mixed_rows = ds4_gpu_tensor_alloc(
            (uint64_t)ROWS * HC_HIDDEN * sizeof(float));
        require_ok(norm_rows && mixed_rows,
                   "Q8_0 HC up/mix multirow allocation");
        require_ok(ds4_gpu_qwen4_q8_0_hc_up_mix_model(
            mixed_rows, norm_rows, x, model, model_fixture_size, hc_offset,
            IN, HC_HIDDEN, ROWS, HC_STREAMS),
            "Q8_0 HC up/mix multirow dispatch");
        float hc_rows_actual[ROWS * HC_HIDDEN];
        require_ok(ds4_gpu_tensor_read(mixed_rows, 0, hc_rows_actual,
                                       sizeof(hc_rows_actual)),
                   "Q8_0 HC up/mix multirow readback");
        require_array_close("Qwen Q8_0 HC up/mix rows>1", hc_rows_actual,
                            hc_rows_expected, ROWS * HC_HIDDEN,
                            5e-4f, 5e-4f);
        ds4_gpu_tensor_free(mixed_rows);
        ds4_gpu_tensor_free(norm_rows);

        float partials_rows[ROWS * HC_STREAMS * HC_STREAMS];
        float streams_rows[ROWS * HC_STREAMS * OUT];
        float streams_rows_actual[ROWS * HC_STREAMS * OUT];
        for (uint32_t i = 0; i < ROWS * HC_STREAMS * HC_STREAMS; i++)
            partials_rows[i] = ((int)(i % 5u) - 2) * 0.11f;
        for (uint32_t i = 0; i < ROWS * HC_STREAMS * OUT; i++)
            streams_rows[i] = ((int)(i % 9u) - 4) * 0.021f;
        for (uint32_t row = 0; row < ROWS; row++) {
            for (uint32_t stream = 0; stream < HC_STREAMS; stream++) {
                float raw = 0.0f;
                for (uint32_t source = 0; source < HC_STREAMS; source++)
                    raw += partials_rows[
                        row * HC_STREAMS * HC_STREAMS +
                        source * HC_STREAMS + stream];
                const float inject = 2.0f / (1.0f + expf(-raw));
                for (uint32_t dim = 0; dim < OUT; dim++)
                    streams_rows[
                        row * HC_STREAMS * OUT + stream * OUT + dim] = fmaf(
                        a_rows[row * OUT + dim], inject,
                        streams_rows[
                            row * HC_STREAMS * OUT + stream * OUT + dim]);
            }
        }
        float streams_rows_init[ROWS * HC_STREAMS * OUT];
        for (uint32_t i = 0; i < ROWS * HC_STREAMS * OUT; i++)
            streams_rows_init[i] = ((int)(i % 9u) - 4) * 0.021f;
        memcpy(streams_rows, streams_rows_init, sizeof(streams_rows));
        for (uint32_t row = 0; row < ROWS; row++) {
            for (uint32_t stream = 0; stream < HC_STREAMS; stream++) {
                float raw = 0.0f;
                for (uint32_t source = 0; source < HC_STREAMS; source++)
                    raw += partials_rows[
                        row * HC_STREAMS * HC_STREAMS +
                        source * HC_STREAMS + stream];
                const float inject = 2.0f / (1.0f + expf(-raw));
                for (uint32_t dim = 0; dim < OUT; dim++)
                    streams_rows[
                        row * HC_STREAMS * OUT + stream * OUT + dim] = fmaf(
                        a_rows[row * OUT + dim], inject,
                        streams_rows_init[
                            row * HC_STREAMS * OUT + stream * OUT + dim]);
            }
        }
        ds4_gpu_tensor *partials_t = tensor_from(partials_rows,
                                                  sizeof(partials_rows));
        ds4_gpu_tensor *streams_t = tensor_from(streams_rows_init,
                                                 sizeof(streams_rows_init));
        require_ok(partials_t && streams_t,
                   "Q8_0 HC write multirow allocation");
        require_ok(ds4_gpu_qwen4_q8_0_hc_write_model(
            streams_t, partials_t, model, model_fixture_size, a_offset,
            x, IN, OUT, ROWS, HC_STREAMS),
            "Q8_0 HC write multirow dispatch");
        require_ok(ds4_gpu_tensor_read(streams_t, 0, streams_rows_actual,
                                       sizeof(streams_rows_actual)),
                   "Q8_0 HC write multirow readback");
        require_array_close("Qwen Q8_0 HC write rows>1", streams_rows_actual,
                            streams_rows, ROWS * HC_STREAMS * OUT,
                            5e-4f, 5e-4f);
        ds4_gpu_tensor_free(streams_t);
        ds4_gpu_tensor_free(partials_t);
        ds4_gpu_tensor_free(o_rows);
    }

    ds4_gpu_tensor_free(stream_t);
    ds4_gpu_tensor_free(partials);
    ds4_gpu_tensor_free(mixed);
    ds4_gpu_tensor_free(norm);
    ds4_gpu_tensor_free(emb);
    ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(ob);
    ds4_gpu_tensor_free(oa);
    ds4_gpu_tensor_free(xb);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
}

static void test_model_q8_0_mul_mm_rows(void) {
    /* Prefill-sized batches route through the tiled tensor-core kernel_mul_mm
     * shared with the GLM/DeepSeek dense path.  out_dim 67 and the 33-row
     * case exercise the bounds-checked variant.  The tiled kernel rounds
     * dequantized weights and activations to F16 in its threadgroup tiles
     * and accumulates F32, so the reference tolerance covers that rounding
     * rather than demanding bit parity with the scalar dot kernels. */
    enum { IN = 512, OUT = 67, ALIGNED_ROWS = 64, TAIL_ROWS = 33 };
    const size_t weight_offset = 4096u;
    const size_t weight_bytes =
        (size_t)OUT * (IN / 32u) * sizeof(test_block_q8_0);
    const size_t model_size = weight_offset + weight_bytes + 256u;
    void *allocation = NULL;
    require_ok(posix_memalign(&allocation, 4096u,
                              align_size(model_size, 4096u)) == 0,
               "Q8_0 mul_mm model allocation");
    memset(allocation, 0, align_size(model_size, 4096u));
    uint8_t *model = allocation;
    test_block_q8_0 *weights =
        (test_block_q8_0 *)(model + weight_offset);
    fill_q8_0_matrix(weights, OUT, IN, 91u);
    require_ok(ds4_gpu_set_model_map(model, model_size),
               "Q8_0 mul_mm model registration");

    const uint32_t row_cases[] = { ALIGNED_ROWS, TAIL_ROWS };
    for (uint32_t c = 0; c < 2u; c++) {
        const uint32_t rows = row_cases[c];
        float *input = malloc((size_t)rows * IN * sizeof(float));
        float *expected = malloc((size_t)rows * OUT * sizeof(float));
        float *actual = malloc((size_t)rows * OUT * sizeof(float));
        require_ok(input && expected && actual,
                   "Q8_0 mul_mm host allocations");
        for (uint32_t i = 0; i < rows * IN; i++)
            input[i] = cosf((float)(i + 7u) * 0.011f) * 0.4f;
        q8_0_matmul_reference(expected, weights, input, IN, OUT, rows);
        ds4_gpu_tensor *x = tensor_from(input,
                                        (size_t)rows * IN * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
            (size_t)rows * OUT * sizeof(float));
        require_ok(x && out, "Q8_0 mul_mm tensor allocation");
        require_ok(ds4_gpu_qwen4_q8_0_matmul_model(
                       out, model, model_size, weight_offset,
                       x, IN, OUT, rows),
                   "Q8_0 mul_mm dispatch");
        require_ok(ds4_gpu_tensor_read(out, 0, actual,
                                       (size_t)rows * OUT * sizeof(float)),
                   "Q8_0 mul_mm readback");
        char label[64];
        snprintf(label, sizeof(label), "Qwen Q8_0 mul_mm rows=%u", rows);
        require_array_close(label, actual, expected,
                            (size_t)rows * OUT, 2e-1f, 4e-3f);
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(x);
        free(actual);
        free(expected);
        free(input);
    }
    free(allocation);
}

static void test_model_q8_0_mpp_rows(void) {
    /* Prefill-sized batches take the Metal4 TensorOps direct-RHS NAX route
     * when the session compiled it (M5-class automatic enable, or the
     * DS4_METAL_ENABLE_TENSOR opt-in this suite sets in main, which covers
     * pre-M5 machines through the portable fallback).  The 128/64/96-row
     * cases pin the n128/n64/n32 token tiles, 200 rows exercises the
     * split-prefix dispatch (192-row NAX prefix plus an 8-row boundary-checked
     * tail), and 33 rows stays under the split threshold so the whole
     * projection keeps the tiled kernel.  out_dim 67 is not 64-aligned and
     * must route entirely through the tiled kernel even at aligned row
     * counts.  The NAX kernel dequantizes Q8_0 weights to F16 tiles and
     * multiplies against the direct F32 activation tile, so the reference
     * tolerance covers the same F16 weight rounding as the tiled fixture. */
    if (!ds4_gpu_metal4_tensor_route_enabled()) {
        puts("Qwen Q8_0 TensorOps route not compiled; MPP fixture skipped");
        return;
    }
    enum { IN = 512 };
    const uint32_t row_cases[] = {128u, 64u, 96u, 200u, 33u};
    const uint32_t out_cases[] = {128u, 67u};
    const size_t weight_offset = 4096u;
    for (uint32_t oc = 0u; oc < 2u; oc++) {
        const uint32_t out_dim = out_cases[oc];
        const size_t weight_bytes =
            (size_t)out_dim * (IN / 32u) * sizeof(test_block_q8_0);
        const size_t model_size = weight_offset + weight_bytes + 256u;
        void *allocation = NULL;
        require_ok(posix_memalign(&allocation, 4096u,
                                  align_size(model_size, 4096u)) == 0,
                   "Q8_0 MPP model allocation");
        memset(allocation, 0, align_size(model_size, 4096u));
        uint8_t *model = allocation;
        test_block_q8_0 *weights =
            (test_block_q8_0 *)(model + weight_offset);
        fill_q8_0_matrix(weights, out_dim, IN, 173u);
        require_ok(ds4_gpu_set_model_map(model, model_size),
                   "Q8_0 MPP model registration");

        for (uint32_t c = 0u; c < sizeof(row_cases) / sizeof(row_cases[0]);
             c++) {
            const uint32_t rows = row_cases[c];
            float *input = malloc((size_t)rows * IN * sizeof(float));
            float *expected = malloc((size_t)rows * out_dim * sizeof(float));
            float *actual = malloc((size_t)rows * out_dim * sizeof(float));
            require_ok(input && expected && actual,
                       "Q8_0 MPP host allocations");
            for (uint32_t i = 0; i < rows * IN; i++)
                input[i] = cosf((float)(i + 13u) * 0.007f) * 0.35f;
            q8_0_matmul_reference(expected, weights, input, IN, out_dim, rows);
            ds4_gpu_tensor *x = tensor_from(
                input, (size_t)rows * IN * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                (size_t)rows * out_dim * sizeof(float));
            require_ok(x && out, "Q8_0 MPP tensor allocation");
            require_ok(ds4_gpu_qwen4_q8_0_matmul_model(
                           out, model, model_size, weight_offset,
                           x, IN, out_dim, rows),
                       "Q8_0 MPP dispatch");
            require_ok(ds4_gpu_tensor_read(out, 0, actual,
                                           (size_t)rows * out_dim * sizeof(float)),
                       "Q8_0 MPP readback");
            char label[64];
            snprintf(label, sizeof(label),
                     "Qwen Q8_0 MPP rows=%u out=%u", rows, out_dim);
            require_array_close(label, actual, expected,
                                (size_t)rows * out_dim, 2e-1f, 4e-3f);
            ds4_gpu_tensor_free(out);
            ds4_gpu_tensor_free(x);
            free(actual);
            free(expected);
            free(input);
        }
        free(allocation);
    }
}

static void test_model_q8_0_mpp_f32stage_rows(void) {
    /* The F32-staged NAX quality route (DS4_QWEN4_Q8_0_MPP_F32STAGE=1)
     * stages the Q8_0 weight tile as fp32, so the only remaining deviation
     * from the exact CPU dequant reference is the TensorOps accumulate
     * order.  On the portable fallback that measures bit-exact; the bound
     * below keeps headroom for the M5 hardware accumulate (the
     * exp/m5-tensor-precision GT sweep measured ~1.7e-4 rms there) while
     * staying far tighter than the binary16-staged fixture. */
    if (!ds4_gpu_metal4_tensor_route_enabled()) {
        puts("Qwen Q8_0 TensorOps route not compiled; MPP f32stage fixture skipped");
        return;
    }
    enum { IN = 512, OUT = 128 };
    const uint32_t row_cases[] = {128u, 96u};
    const size_t weight_offset = 4096u;
    const size_t weight_bytes =
        (size_t)OUT * (IN / 32u) * sizeof(test_block_q8_0);
    const size_t model_size = weight_offset + weight_bytes + 256u;
    void *allocation = NULL;
    require_ok(posix_memalign(&allocation, 4096u,
                              align_size(model_size, 4096u)) == 0,
               "Q8_0 MPP f32stage model allocation");
    memset(allocation, 0, align_size(model_size, 4096u));
    uint8_t *model = allocation;
    test_block_q8_0 *weights =
        (test_block_q8_0 *)(model + weight_offset);
    fill_q8_0_matrix(weights, OUT, IN, 211u);
    require_ok(ds4_gpu_set_model_map(model, model_size),
               "Q8_0 MPP f32stage model registration");

    setenv("DS4_QWEN4_Q8_0_MPP_F32STAGE", "1", 1);
    for (uint32_t c = 0u; c < sizeof(row_cases) / sizeof(row_cases[0]);
         c++) {
        const uint32_t rows = row_cases[c];
        float *input = malloc((size_t)rows * IN * sizeof(float));
        float *expected = malloc((size_t)rows * OUT * sizeof(float));
        float *actual = malloc((size_t)rows * OUT * sizeof(float));
        require_ok(input && expected && actual,
                   "Q8_0 MPP f32stage host allocations");
        for (uint32_t i = 0; i < rows * IN; i++)
            input[i] = cosf((float)(i + 17u) * 0.009f) * 0.3f;
        q8_0_matmul_reference(expected, weights, input, IN, OUT, rows);
        ds4_gpu_tensor *x = tensor_from(
            input, (size_t)rows * IN * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
            (size_t)rows * OUT * sizeof(float));
        require_ok(x && out, "Q8_0 MPP f32stage tensor allocation");
        require_ok(ds4_gpu_qwen4_q8_0_matmul_model(
                       out, model, model_size, weight_offset,
                       x, IN, OUT, rows),
                   "Q8_0 MPP f32stage dispatch");
        require_ok(ds4_gpu_tensor_read(out, 0, actual,
                                       (size_t)rows * OUT * sizeof(float)),
                   "Q8_0 MPP f32stage readback");
        char label[64];
        snprintf(label, sizeof(label),
                 "Qwen Q8_0 MPP f32stage rows=%u", rows);
        require_array_close(label, actual, expected,
                            (size_t)rows * OUT, 4e-3f, 4e-3f);
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(x);
        free(actual);
        free(expected);
        free(input);
    }
    unsetenv("DS4_QWEN4_Q8_0_MPP_F32STAGE");
    free(allocation);
}

static void test_model_bf16_matmul_rows(void) {
    /* Prefill-sized BF16 control projections (MoE router, GDN decay/beta)
     * route through the tiled tensor-core kernel_mul_mm_bf16_f32.  The
     * tiled kernel stages BF16 weights and F32 activations into F16
     * threadgroup tiles and accumulates F32, so the rows>=32 reference
     * models that operand rounding; the below-threshold case pins the
     * scalar path's exact BF16 arithmetic.  out_dim 67 and rows 33
     * exercise the bounds-checked variant. */
    enum { IN = 512 };
    const size_t weight_offset = 4096u;
    const uint32_t out_cases[] = { 67u, 128u };
    const uint32_t row_cases[] = { 64u, 33u, 8u };
    for (uint32_t oc = 0; oc < 2u; oc++) {
        const uint32_t out_dim = out_cases[oc];
        const size_t weight_bytes = (size_t)out_dim * IN * sizeof(uint16_t);
        const size_t model_size = weight_offset + weight_bytes + 256u;
        void *allocation = NULL;
        require_ok(posix_memalign(&allocation, 4096u,
                                  align_size(model_size, 4096u)) == 0,
                   "BF16 mul_mm model allocation");
        memset(allocation, 0, align_size(model_size, 4096u));
        uint8_t *model = allocation;
        uint16_t *weights = (uint16_t *)(model + weight_offset);
        for (uint32_t o = 0; o < out_dim; o++)
            for (uint32_t k = 0; k < IN; k++)
                weights[(size_t)o * IN + k] = f32_to_bf16(
                    cosf((float)((o * IN + k) % 521u) * 0.017f) * 0.05f +
                    sinf((float)o * 0.013f) * 0.03f);
        require_ok(ds4_gpu_set_model_map(model, model_size),
                   "BF16 mul_mm model registration");
        for (uint32_t rc = 0; rc < 2u; rc++) {
            const uint32_t rows = row_cases[rc];
            float *input = malloc((size_t)rows * IN * sizeof(float));
            float *expected = malloc((size_t)rows * out_dim * sizeof(float));
            float *actual = malloc((size_t)rows * out_dim * sizeof(float));
            require_ok(input && expected && actual,
                       "BF16 mul_mm host allocations");
            for (uint32_t i = 0; i < rows * IN; i++)
                input[i] = cosf((float)(i + 7u) * 0.011f) * 0.4f;
            for (uint32_t r = 0; r < rows; r++) {
                for (uint32_t o = 0; o < out_dim; o++) {
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < IN; k++) {
                        const float weight =
                            bf16_to_f32(weights[(size_t)o * IN + k]);
                        if (rows >= 32u) {
                            const _Float16 w16 = (_Float16)weight;
                            const _Float16 x16 = (_Float16)input[r * IN + k];
                            sum += (float)w16 * (float)x16;
                        } else {
                            sum += weight * input[r * IN + k];
                        }
                    }
                    expected[(size_t)r * out_dim + o] = sum;
                }
            }
            ds4_gpu_tensor *x = tensor_from(
                input, (size_t)rows * IN * sizeof(float));
            ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
                (size_t)rows * out_dim * sizeof(float));
            require_ok(x && out, "BF16 mul_mm tensor allocation");
            require_ok(ds4_gpu_qwen4_bf16_matmul_model(
                           out, model, model_size, weight_offset,
                           x, IN, out_dim, rows),
                       "BF16 mul_mm dispatch");
            require_ok(ds4_gpu_tensor_read(
                           out, 0, actual,
                           (size_t)rows * out_dim * sizeof(float)),
                       "BF16 mul_mm readback");
            char label[64];
            snprintf(label, sizeof(label),
                     "Qwen BF16 matmul out=%u rows=%u", out_dim, rows);
            require_array_close(label, actual, expected,
                                (size_t)rows * out_dim,
                                rows >= 32u ? 1e-2f : 2e-5f, 4e-3f);
            ds4_gpu_tensor_free(out);
            ds4_gpu_tensor_free(x);
            free(actual);
            free(expected);
            free(input);
        }
        free(allocation);
    }
}


static void test_vision_fc2_q8_0_stride(void) {
    enum { ROWS = 2, MODEL_GUARD = 256, OUTPUT_GUARD = 64 };
    const size_t row_blocks = QWEN4_VISION_FC2_PHYSICAL_DIM / 32u;
    const size_t weight_offset = 4096u;
    const size_t weight_bytes =
        (size_t)QWEN4_VISION_FC2_OUT_DIM * row_blocks *
        sizeof(test_block_q8_0);
    const size_t model_size = weight_offset + weight_bytes + MODEL_GUARD;
    const size_t allocation_size = align_size(model_size, 4096u);
    const size_t input_count =
        (size_t)ROWS * QWEN4_VISION_FC2_PHYSICAL_DIM;
    const size_t output_count =
        (size_t)ROWS * QWEN4_VISION_FC2_OUT_DIM;
    const size_t guarded_output_count = output_count + OUTPUT_GUARD;

    require_ok(posix_memalign(&vision_fc2_fixture_allocation,
                              4096u, allocation_size) == 0,
               "Qwen vision FC2 model allocation");
    void *allocation = vision_fc2_fixture_allocation;
    memset(allocation, 0xa7, allocation_size);
    uint8_t *model = allocation;
    test_block_q8_0 *weights =
        (test_block_q8_0 *)(model + weight_offset);
    fill_q8_0_matrix(weights, QWEN4_VISION_FC2_OUT_DIM,
                     QWEN4_VISION_FC2_PHYSICAL_DIM, 71u);
    uint8_t model_guard[MODEL_GUARD];
    memcpy(model_guard, model + weight_offset + weight_bytes,
           sizeof(model_guard));

    float *input = malloc(input_count * sizeof(float));
    float *expected = malloc(output_count * sizeof(float));
    float *initial = malloc(guarded_output_count * sizeof(float));
    float *actual = malloc(guarded_output_count * sizeof(float));
    require_ok(input && expected && initial && actual,
               "Qwen vision FC2 host allocations");
    for (uint32_t row = 0u; row < ROWS; row++) {
        for (uint32_t dim = 0u;
             dim < QWEN4_VISION_FC2_PHYSICAL_DIM; dim++) {
            input[(size_t)row * QWEN4_VISION_FC2_PHYSICAL_DIM + dim] =
                dim < QWEN4_VISION_FC2_LOGICAL_DIM
                    ? 0.0125f * cosf((float)(row * 17u + dim + 1u) * 0.013f)
                    : 0.0f;
        }
    }
    q8_0_matmul_reference(
        expected, weights, input, QWEN4_VISION_FC2_PHYSICAL_DIM,
        QWEN4_VISION_FC2_OUT_DIM, ROWS);
    for (size_t i = 0; i < guarded_output_count; i++)
        initial[i] = 12345.25f;

    require_ok(ds4_gpu_set_model_map(model, model_size),
               "Qwen vision FC2 model registration");
    ds4_gpu_tensor *x = tensor_from(input, input_count * sizeof(float));
    ds4_gpu_tensor *out = tensor_from(
        initial, guarded_output_count * sizeof(float));
    require_ok(ds4_gpu_qwen4_vision_fc2_q8_0_model(
                   out, model, model_size, weight_offset, x, ROWS),
               "Qwen vision FC2 135-block dispatch");
    require_ok(ds4_gpu_tensor_read(
                   out, 0, actual, guarded_output_count * sizeof(float)),
               "Qwen vision FC2 readback");
    require_array_close("Qwen vision FC2 Q8_0 4320-row stride",
                        actual, expected, output_count, 2e-3f, 5e-4f);
    require_ok(memcmp(actual + output_count, initial + output_count,
                      OUTPUT_GUARD * sizeof(float)) == 0,
               "Qwen vision FC2 output guard");
    require_ok(memcmp(model + weight_offset + weight_bytes, model_guard,
                      sizeof(model_guard)) == 0,
               "Qwen vision FC2 model guard");

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    require_ok(ds4_gpu_set_model_map(model_fixture_allocation,
                                     model_fixture_size),
               "restore Q8_0 model fixture after vision FC2");
    free(actual);
    free(initial);
    free(expected);
    free(input);
}

static void test_moe_topk(void) {
    enum { ROWS = 2, EXPERTS = 512, TOP_K = 10 };
    float logits[ROWS * EXPERTS];
    int32_t expected_ids[ROWS * TOP_K], actual_ids[ROWS * TOP_K];
    float expected_weights[ROWS * TOP_K], actual_weights[ROWS * TOP_K];
    for (uint32_t i = 0; i < ROWS * EXPERTS; i++)
        logits[i] = sinf((float)(i * 17u + 3u) * 0.01f);
    for (uint32_t row = 0; row < ROWS; row++)
        moe_topk_reference(logits + row * EXPERTS, EXPERTS, TOP_K,
                           expected_ids + row * TOP_K,
                           expected_weights + row * TOP_K);
    ds4_gpu_tensor *lt = tensor_from(logits, sizeof(logits));
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(sizeof(actual_ids));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(sizeof(actual_weights));
    require_ok(it && wt, "Qwen top-k output allocations");
    require_ok(ds4_gpu_qwen4_moe_topk(it, wt, lt, ROWS, EXPERTS, TOP_K),
               "Qwen 512-way top-10 routing");
    require_ok(ds4_gpu_tensor_read(it, 0, actual_ids, sizeof(actual_ids)) &&
               ds4_gpu_tensor_read(wt, 0, actual_weights,
                                   sizeof(actual_weights)),
               "Qwen top-k readback");
    require_ok(memcmp(actual_ids, expected_ids, sizeof(actual_ids)) == 0,
               "Qwen ordered top-k ids");
    require_array_close("Qwen ordered top-k weights", actual_weights,
                        expected_weights, ROWS * TOP_K, 2e-6f, 2e-6f);
    ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(it);
    ds4_gpu_tensor_free(lt);
}

static void test_moe_q4_k(void) {
    enum {
        IN = 256, FF = 640, DOWN = 768, OUT = 33,
        EXPERTS = 12, TOP_K = 10, ROWS = 9,
    };
    const size_t gate_bytes = (size_t)EXPERTS * FF * (IN / 256u) *
        sizeof(test_block_q4_k);
    const size_t down_bytes = (size_t)EXPERTS * OUT * (DOWN / 256u) *
        sizeof(test_block_q4_k);
    size_t cursor = 0u;
    const size_t gate_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t up_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t down_offset = cursor;
    cursor = align_size(cursor + down_bytes, 4096u);
    require_ok(posix_memalign(&moe_rows8_fixture_allocation,
                              4096u, cursor) == 0,
               "Q4_K MoE model allocation");
    uint8_t *model = moe_rows8_fixture_allocation;
    memset(model, 0, cursor);
    test_block_q4_k *gate = (test_block_q4_k *)(model + gate_offset);
    test_block_q4_k *up = (test_block_q4_k *)(model + up_offset);
    test_block_q4_k *down = (test_block_q4_k *)(model + down_offset);
    fill_q4_k_matrix(gate, EXPERTS * FF, IN, 5u);
    fill_q4_k_matrix(up, EXPERTS * FF, IN, 19u);
    fill_q4_k_matrix(down, EXPERTS * OUT, DOWN, 37u);
    require_ok(ds4_gpu_set_model_map(model, cursor),
               "Q4_K MoE model fixture registration");

    float input[ROWS * IN];
    int32_t selected[ROWS * TOP_K];
    float route[ROWS * TOP_K];
    float expected_mid[ROWS * TOP_K * FF];
    float expected_out[ROWS * OUT];
    for (uint32_t i = 0; i < ROWS * IN; i++)
        input[i] = cosf((float)(i + 1u) * 0.021f) * 0.2f;
    for (uint32_t row = 0; row < ROWS; row++) {
        float route_sum = 0.0f;
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            selected[row * TOP_K + slot] =
                (int32_t)((row * 7u + slot * 5u) % EXPERTS);
            route[row * TOP_K + slot] = (float)(slot + 1u);
            route_sum += route[row * TOP_K + slot];
        }
        for (uint32_t slot = 0; slot < TOP_K; slot++)
            route[row * TOP_K + slot] /= route_sum;
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            const uint32_t expert = (uint32_t)selected[row * TOP_K + slot];
            for (uint32_t output = 0; output < FF; output++) {
                const uint32_t weight_row = expert * FF + output;
                float g = 0.0f, u = 0.0f;
                for (uint32_t k = 0; k < IN; k++) {
                    const float xv = input[row * IN + k];
                    g = fmaf(q4_k_value(gate, IN, weight_row, k), xv, g);
                    u = fmaf(q4_k_value(up, IN, weight_row, k), xv, u);
                }
                expected_mid[((size_t)row * TOP_K + slot) * FF + output] =
                    (g / (1.0f + expf(-g))) * u;
            }
        }
        for (uint32_t output = 0; output < OUT; output++) {
            float total = 0.0f;
            for (uint32_t slot = 0; slot < TOP_K; slot++) {
                const uint32_t expert =
                    (uint32_t)selected[row * TOP_K + slot];
                const uint32_t weight_row = expert * OUT + output;
                float sum = 0.0f;
                for (uint32_t k = 0; k < FF; k++)
                    sum = fmaf(q4_k_value(down, DOWN, weight_row, k),
                        expected_mid[((size_t)row * TOP_K + slot) * FF + k],
                        sum);
                total = fmaf(sum, route[row * TOP_K + slot], total);
            }
            expected_out[row * OUT + output] = total;
        }
    }

    ds4_gpu_tensor *xt = tensor_from(input, sizeof(input));
    ds4_gpu_tensor *st = tensor_from(selected, sizeof(selected));
    ds4_gpu_tensor *rt = tensor_from(route, sizeof(route));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(sizeof(expected_mid));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(expected_out));
    float *actual_mid = malloc(sizeof(expected_mid));
    float *actual_out = malloc(sizeof(expected_out));
    require_ok(mid && out && actual_mid && actual_out,
               "Q4_K MoE output allocations");
    require_ok(setenv("DS4_QWEN4_MOE_EXACT_MIN_ROWS", "1", 1) == 0,
               "enable Q4_K rows8 fixture");
    require_ok(ds4_gpu_qwen4_moe_q4_k_model(
        out, mid, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, ROWS),
        "Qwen Q4_K exact rows8 MoE");
    require_ok(unsetenv("DS4_QWEN4_MOE_EXACT_MIN_ROWS") == 0,
               "clear Q4_K rows8 fixture");
    require_ok(ds4_gpu_tensor_read(mid, 0, actual_mid, sizeof(expected_mid)) &&
               ds4_gpu_tensor_read(out, 0, actual_out, sizeof(expected_out)),
               "Q4_K MoE readback");
    require_array_close_stats("Qwen standard Q4_K MoE mid", actual_mid,
                              expected_mid, ROWS * TOP_K * FF,
                              3e-3f, 3e-3f);
    require_array_close_stats("Qwen standard Q4_K MoE output", actual_out,
                              expected_out, ROWS * OUT, 2e-2f, 3e-3f);

    ds4_gpu_tensor *mid_m1 = ds4_gpu_tensor_alloc(TOP_K * FF * sizeof(float));
    ds4_gpu_tensor *out_m1 = ds4_gpu_tensor_alloc(OUT * sizeof(float));
    require_ok(mid_m1 && out_m1, "Q4_K M=1 allocations");
    require_ok(ds4_gpu_qwen4_moe_q4_k_model(
        out_m1, mid_m1, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, 1u),
        "Qwen Q4_K M=1 MoE");
    require_ok(ds4_gpu_tensor_read(out_m1, 0, actual_out,
                                   OUT * sizeof(float)),
               "Q4_K M=1 readback");
    require_array_close("Qwen standard Q4_K M=1", actual_out, expected_out,
                        OUT, 2e-2f, 3e-3f);
    ds4_gpu_tensor_free(out_m1);
    ds4_gpu_tensor_free(mid_m1);
    free(actual_out);
    free(actual_mid);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(rt);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(xt);
    if (model_fixture_allocation != NULL)
        require_ok(ds4_gpu_set_model_map(model_fixture_allocation,
                                         model_fixture_size),
                   "restore Q8_0 model fixture registration");
}

static void test_moe_q4_0(void) {
    enum {
        IN = 256, FF = 64, DOWN = 256, OUT = 33,
        EXPERTS = 12, TOP_K = 10,
    };
    const size_t gate_bytes = (size_t)EXPERTS * FF * (IN / 32u) *
        sizeof(test_block_q4_0);
    const size_t down_bytes = (size_t)EXPERTS * OUT * (DOWN / 32u) *
        sizeof(test_block_q4_0);
    size_t cursor = 0u;
    const size_t gate_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t up_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t down_offset = cursor;
    cursor = align_size(cursor + down_bytes, 4096u);
    void *allocation = NULL;
    require_ok(posix_memalign(&allocation, 4096u, cursor) == 0,
               "Q4_0 MoE model allocation");
    uint8_t *model = allocation;
    memset(model, 0, cursor);
    test_block_q4_0 *gate = (test_block_q4_0 *)(model + gate_offset);
    test_block_q4_0 *up = (test_block_q4_0 *)(model + up_offset);
    test_block_q4_0 *down = (test_block_q4_0 *)(model + down_offset);
    fill_q4_0_matrix(gate, EXPERTS * FF, IN, 7u);
    fill_q4_0_matrix(up, EXPERTS * FF, IN, 23u);
    fill_q4_0_matrix(down, EXPERTS * OUT, DOWN, 41u);
    require_ok(ds4_gpu_set_model_map(model, cursor),
               "Q4_0 MoE model fixture registration");

    float input[IN];
    int32_t selected[TOP_K];
    float route[TOP_K];
    float expected_mid[TOP_K * FF];
    float expected_out[OUT];
    for (uint32_t i = 0; i < IN; i++)
        input[i] = cosf((float)(i + 1u) * 0.021f) * 0.2f;
    for (uint32_t slot = 0; slot < TOP_K; slot++) {
        selected[slot] = (int32_t)((slot * 5u) % EXPERTS);
        route[slot] = (float)(slot + 1u) / 55.0f;
        const uint32_t expert = (uint32_t)selected[slot];
        for (uint32_t output = 0; output < FF; output++) {
            const uint32_t weight_row = expert * FF + output;
            float g = 0.0f, u = 0.0f;
            for (uint32_t k = 0; k < IN; k++) {
                g = fmaf(q4_0_value(gate, IN, weight_row, k), input[k], g);
                u = fmaf(q4_0_value(up, IN, weight_row, k), input[k], u);
            }
            expected_mid[slot * FF + output] =
                (g / (1.0f + expf(-g))) * u;
        }
    }
    for (uint32_t output = 0; output < OUT; output++) {
        float total = 0.0f;
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            const uint32_t expert = (uint32_t)selected[slot];
            const uint32_t weight_row = expert * OUT + output;
            float sum = 0.0f;
            for (uint32_t k = 0; k < FF; k++)
                sum = fmaf(q4_0_value(down, DOWN, weight_row, k),
                           expected_mid[slot * FF + k], sum);
            total = fmaf(sum, route[slot], total);
        }
        expected_out[output] = total;
    }

    ds4_gpu_tensor *xt = tensor_from(input, sizeof(input));
    ds4_gpu_tensor *st = tensor_from(selected, sizeof(selected));
    ds4_gpu_tensor *rt = tensor_from(route, sizeof(route));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(sizeof(expected_mid));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(expected_out));
    float actual_mid[TOP_K * FF], actual_out[OUT];
    require_ok(mid && out, "Q4_0 MoE output allocations");
    require_ok(ds4_gpu_qwen4_moe_q4_0_model(
        out, mid, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, 1u),
        "Qwen Q4_0 M=1 MoE");
    require_ok(ds4_gpu_tensor_read(mid, 0, actual_mid, sizeof(actual_mid)) &&
               ds4_gpu_tensor_read(out, 0, actual_out, sizeof(actual_out)),
               "Q4_0 MoE readback");
    require_array_close("Qwen Q4_0 M=1 MoE mid", actual_mid, expected_mid,
                        TOP_K * FF, 4e-4f, 4e-4f);
    require_array_close("Qwen Q4_0 M=1 MoE output", actual_out, expected_out,
                        OUT, 5e-4f, 5e-4f);

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(rt);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(xt);
    if (model_fixture_allocation != NULL)
        require_ok(ds4_gpu_set_model_map(model_fixture_allocation,
                                         model_fixture_size),
                   "restore Q8_0 model fixture after Q4_0 MoE");
    free(allocation);
}

static void test_moe_q4_0_mul_mm_id(void) {
    /* The grouped tensor-core MoE path engages at DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS
     * rows and tiles in 512-row chunks; 600 rows covers a full tile plus a
     * ragged tail.  The reference stays the exact scalar CPU dot product; the
     * tiled kernels round dequantized weights and activations to F16 tiles, so
     * the tolerance covers that rounding. */
    enum {
        IN = 256, FF = 64, DOWN = 256, OUT = 33,
        EXPERTS = 12, TOP_K = 10, ROWS = 600,
    };
    const size_t gate_bytes = (size_t)EXPERTS * FF * (IN / 32u) *
        sizeof(test_block_q4_0);
    const size_t down_bytes = (size_t)EXPERTS * OUT * (DOWN / 32u) *
        sizeof(test_block_q4_0);
    size_t cursor = 0u;
    const size_t gate_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t up_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t down_offset = cursor;
    cursor = align_size(cursor + down_bytes, 4096u);
    void *allocation = NULL;
    require_ok(posix_memalign(&allocation, 4096u, cursor) == 0,
               "Q4_0 MoE mul_mm_id model allocation");
    uint8_t *model = allocation;
    memset(model, 0, cursor);
    test_block_q4_0 *gate = (test_block_q4_0 *)(model + gate_offset);
    test_block_q4_0 *up = (test_block_q4_0 *)(model + up_offset);
    test_block_q4_0 *down = (test_block_q4_0 *)(model + down_offset);
    fill_q4_0_matrix(gate, EXPERTS * FF, IN, 7u);
    fill_q4_0_matrix(up, EXPERTS * FF, IN, 23u);
    fill_q4_0_matrix(down, EXPERTS * OUT, DOWN, 41u);
    require_ok(ds4_gpu_set_model_map(model, cursor),
               "Q4_0 MoE mul_mm_id model fixture registration");

    float *input = malloc((size_t)ROWS * IN * sizeof(float));
    int32_t *selected = malloc((size_t)ROWS * TOP_K * sizeof(int32_t));
    float *route = malloc((size_t)ROWS * TOP_K * sizeof(float));
    float *expected_mid = malloc((size_t)ROWS * TOP_K * FF * sizeof(float));
    float *expected_out = malloc((size_t)ROWS * OUT * sizeof(float));
    require_ok(input && selected && route && expected_mid && expected_out,
               "Q4_0 MoE mul_mm_id host allocations");
    /* (slot * 5 + row) mod 12 is injective in slot because gcd(5, 12) is 1,
     * so every row keeps ten distinct experts as the route map requires. */
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t i = 0; i < IN; i++)
            input[(size_t)row * IN + i] =
                cosf((float)(row * 31u + i + 1u) * 0.017f) * 0.2f;
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            selected[(size_t)row * TOP_K + slot] =
                (int32_t)((slot * 5u + row) % EXPERTS);
            route[(size_t)row * TOP_K + slot] =
                (float)(slot + 1u) / 55.0f;
        }
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            const uint32_t expert =
                (uint32_t)selected[(size_t)row * TOP_K + slot];
            for (uint32_t output = 0; output < FF; output++) {
                const uint32_t weight_row = expert * FF + output;
                float g = 0.0f, u = 0.0f;
                for (uint32_t k = 0; k < IN; k++) {
                    g = fmaf(q4_0_value(gate, IN, weight_row, k),
                             input[(size_t)row * IN + k], g);
                    u = fmaf(q4_0_value(up, IN, weight_row, k),
                             input[(size_t)row * IN + k], u);
                }
                expected_mid[((size_t)row * TOP_K + slot) * FF + output] =
                    (g / (1.0f + expf(-g))) * u;
            }
        }
        for (uint32_t output = 0; output < OUT; output++) {
            float total = 0.0f;
            for (uint32_t slot = 0; slot < TOP_K; slot++) {
                const uint32_t expert =
                    (uint32_t)selected[(size_t)row * TOP_K + slot];
                const uint32_t weight_row = expert * OUT + output;
                float sum = 0.0f;
                for (uint32_t k = 0; k < FF; k++)
                    sum = fmaf(q4_0_value(down, DOWN, weight_row, k),
                               expected_mid[
                                   ((size_t)row * TOP_K + slot) * FF + k],
                               sum);
                total = fmaf(sum, route[(size_t)row * TOP_K + slot], total);
            }
            expected_out[(size_t)row * OUT + output] = total;
        }
    }

    ds4_gpu_tensor *xt = tensor_from(input, (size_t)ROWS * IN * sizeof(float));
    ds4_gpu_tensor *st = tensor_from(selected,
                                     (size_t)ROWS * TOP_K * sizeof(int32_t));
    ds4_gpu_tensor *rt = tensor_from(route,
                                     (size_t)ROWS * TOP_K * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(
        (size_t)ROWS * TOP_K * FF * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        (size_t)ROWS * OUT * sizeof(float));
    float *actual_mid = malloc((size_t)ROWS * TOP_K * FF * sizeof(float));
    float *actual_out = malloc((size_t)ROWS * OUT * sizeof(float));
    require_ok(mid && out && actual_mid && actual_out,
               "Q4_0 MoE mul_mm_id tensor allocations");
    setenv("DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS", "512", 1);
    setenv("DS4_QWEN4_MOE_MUL_MM_TILE_ROWS", "512", 1);
    require_ok(ds4_gpu_qwen4_moe_q4_0_model(
        out, mid, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, ROWS),
        "Qwen Q4_0 grouped MoE dispatch");
    require_ok(ds4_gpu_tensor_read(mid, 0, actual_mid,
                                   (size_t)ROWS * TOP_K * FF * sizeof(float)) &&
               ds4_gpu_tensor_read(out, 0, actual_out,
                                   (size_t)ROWS * OUT * sizeof(float)),
               "Q4_0 MoE mul_mm_id readback");
    require_array_close("Qwen Q4_0 grouped MoE mid", actual_mid,
                        expected_mid, (size_t)ROWS * TOP_K * FF,
                        2e-2f, 4e-3f);
    require_array_close("Qwen Q4_0 grouped MoE output", actual_out,
                        expected_out, (size_t)ROWS * OUT, 5e-2f, 8e-3f);

    /* The work-tile row granularity only regroups which threadgroup computes
     * which (token, slot) assignment row: every output element's dot
     * products, accumulation order, and destination stay fixed, so a single
     * 600-row tile (one full tile plus a ragged tail folded together) must
     * be BYTE-IDENTICAL to the 512-row tiling above. */
    float *retiled_mid = malloc((size_t)ROWS * TOP_K * FF * sizeof(float));
    float *retiled_out = malloc((size_t)ROWS * OUT * sizeof(float));
    require_ok(retiled_mid && retiled_out,
               "Q4_0 MoE retile host allocations");
    setenv("DS4_QWEN4_MOE_MUL_MM_TILE_ROWS", "600", 1);
    require_ok(ds4_gpu_qwen4_moe_q4_0_model(
        out, mid, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, ROWS),
        "Qwen Q4_0 grouped MoE single-tile dispatch");
    require_ok(ds4_gpu_tensor_read(mid, 0, retiled_mid,
                                   (size_t)ROWS * TOP_K * FF * sizeof(float)) &&
               ds4_gpu_tensor_read(out, 0, retiled_out,
                                   (size_t)ROWS * OUT * sizeof(float)),
               "Q4_0 MoE single-tile readback");
    require_ok(memcmp(retiled_mid, actual_mid,
                      (size_t)ROWS * TOP_K * FF * sizeof(float)) == 0 &&
               memcmp(retiled_out, actual_out,
                      (size_t)ROWS * OUT * sizeof(float)) == 0,
               "Q4_0 MoE tile-size byte identity (512 vs 600 rows)");
    unsetenv("DS4_QWEN4_MOE_MUL_MM_TILE_ROWS");
    unsetenv("DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS");
    free(retiled_out);
    free(retiled_mid);

    /* F32-staged quality route for the STANDARD Q4_0-routed pack
     * (DS4_QWEN4_MOE_MUL_MM_ID_F32STAGE=1 or the shared
     * DS4_METAL_MPP_MOE_F32STAGE), the same pass the Q4_K fixture pins:
     * fp32 operand staging removes the binary16 tile rounding, leaving
     * only the TensorOps accumulate order.  Measured on the portable
     * fallback at 6.0e-8 mid/output (vs 3.4e-4 / 2.9e-4 binary16-staged);
     * bounds keep headroom for the M5 hardware accumulate.  Skipped when
     * the TensorOps route is not compiled. */
    if (ds4_gpu_metal4_tensor_route_enabled()) {
        setenv("DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS", "512", 1);
        setenv("DS4_QWEN4_MOE_MUL_MM_ID_F32STAGE", "1", 1);
        require_ok(ds4_gpu_qwen4_moe_q4_0_model(
            out, mid, xt, st, rt, model, cursor,
            gate_offset, up_offset, down_offset,
            IN, FF, OUT, EXPERTS, TOP_K, ROWS),
            "Qwen Q4_0 grouped MoE f32stage dispatch");
        require_ok(ds4_gpu_tensor_read(mid, 0, actual_mid,
                                       (size_t)ROWS * TOP_K * FF * sizeof(float)) &&
                   ds4_gpu_tensor_read(out, 0, actual_out,
                                       (size_t)ROWS * OUT * sizeof(float)),
                   "Q4_0 MoE f32stage readback");
        require_array_close("Qwen Q4_0 grouped MoE f32stage mid",
                            actual_mid, expected_mid,
                            (size_t)ROWS * TOP_K * FF, 4e-3f, 4e-3f);
        require_array_close("Qwen Q4_0 grouped MoE f32stage output",
                            actual_out, expected_out,
                            (size_t)ROWS * OUT, 2e-2f, 4e-3f);
        unsetenv("DS4_QWEN4_MOE_MUL_MM_ID_F32STAGE");
        unsetenv("DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS");
    }

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(rt);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(xt);
    free(actual_out);
    free(actual_mid);
    free(expected_out);
    free(expected_mid);
    free(route);
    free(selected);
    free(input);
    if (model_fixture_allocation != NULL)
        require_ok(ds4_gpu_set_model_map(model_fixture_allocation,
                                         model_fixture_size),
                   "restore Q8_0 model fixture after Q4_0 MoE mul_mm_id");
    free(allocation);
}

static void test_moe_q4_k_mul_mm_id(void) {
    /* The v3 standard pack's grouped Q4_K routed path at prefill rows: same
     * dispatch contract as the Q4_0 fixture, at the production geometry
     * (640-wide expert FF, zero-padded 768-wide down rows).  With the
     * TensorOps route compiled this exercises kernel_mul_mm_id_q4_K_f32_mpp;
     * otherwise the simdgroup id kernel.  The reference is the exact scalar
     * CPU dot product, with the F16 tile-staging tolerance of the grouped
     * family. */
    enum {
        IN = 256, FF = 640, DOWN = 768, OUT = 33,
        EXPERTS = 12, TOP_K = 10, ROWS = 600,
    };
    const size_t gate_bytes = (size_t)EXPERTS * FF * (IN / 256u) *
        sizeof(test_block_q4_k);
    const size_t down_bytes = (size_t)EXPERTS * OUT * (DOWN / 256u) *
        sizeof(test_block_q4_k);
    size_t cursor = 0u;
    const size_t gate_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t up_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t down_offset = cursor;
    cursor = align_size(cursor + down_bytes, 4096u);
    void *allocation = NULL;
    require_ok(posix_memalign(&allocation, 4096u, cursor) == 0,
               "Q4_K MoE mul_mm_id model allocation");
    uint8_t *model = allocation;
    memset(model, 0, cursor);
    test_block_q4_k *gate = (test_block_q4_k *)(model + gate_offset);
    test_block_q4_k *up = (test_block_q4_k *)(model + up_offset);
    test_block_q4_k *down = (test_block_q4_k *)(model + down_offset);
    fill_q4_k_matrix(gate, EXPERTS * FF, IN, 11u);
    fill_q4_k_matrix(up, EXPERTS * FF, IN, 29u);
    fill_q4_k_matrix(down, EXPERTS * OUT, DOWN, 43u);
    require_ok(ds4_gpu_set_model_map(model, cursor),
               "Q4_K MoE mul_mm_id model fixture registration");

    float *input = malloc((size_t)ROWS * IN * sizeof(float));
    int32_t *selected = malloc((size_t)ROWS * TOP_K * sizeof(int32_t));
    float *route = malloc((size_t)ROWS * TOP_K * sizeof(float));
    float *expected_mid = malloc((size_t)ROWS * TOP_K * FF * sizeof(float));
    float *expected_out = malloc((size_t)ROWS * OUT * sizeof(float));
    require_ok(input && selected && route && expected_mid && expected_out,
               "Q4_K MoE mul_mm_id host allocations");
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t i = 0; i < IN; i++)
            input[(size_t)row * IN + i] =
                cosf((float)(row * 29u + i + 3u) * 0.013f) * 0.2f;
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            selected[(size_t)row * TOP_K + slot] =
                (int32_t)((slot * 5u + row) % EXPERTS);
            route[(size_t)row * TOP_K + slot] =
                (float)(slot + 1u) / 55.0f;
        }
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            const uint32_t expert =
                (uint32_t)selected[(size_t)row * TOP_K + slot];
            for (uint32_t output = 0; output < FF; output++) {
                const uint32_t weight_row = expert * FF + output;
                float g = 0.0f, u = 0.0f;
                for (uint32_t k = 0; k < IN; k++) {
                    g = fmaf(q4_k_value(gate, IN, weight_row, k),
                             input[(size_t)row * IN + k], g);
                    u = fmaf(q4_k_value(up, IN, weight_row, k),
                             input[(size_t)row * IN + k], u);
                }
                expected_mid[((size_t)row * TOP_K + slot) * FF + output] =
                    (g / (1.0f + expf(-g))) * u;
            }
        }
        for (uint32_t output = 0; output < OUT; output++) {
            float total = 0.0f;
            for (uint32_t slot = 0; slot < TOP_K; slot++) {
                const uint32_t expert =
                    (uint32_t)selected[(size_t)row * TOP_K + slot];
                const uint32_t weight_row = expert * OUT + output;
                float sum = 0.0f;
                for (uint32_t k = 0; k < FF; k++)
                    sum = fmaf(q4_k_value(down, DOWN, weight_row, k),
                               expected_mid[
                                   ((size_t)row * TOP_K + slot) * FF + k],
                               sum);
                total = fmaf(sum, route[(size_t)row * TOP_K + slot], total);
            }
            expected_out[(size_t)row * OUT + output] = total;
        }
    }

    ds4_gpu_tensor *xt = tensor_from(input, (size_t)ROWS * IN * sizeof(float));
    ds4_gpu_tensor *st = tensor_from(selected,
                                     (size_t)ROWS * TOP_K * sizeof(int32_t));
    ds4_gpu_tensor *rt = tensor_from(route,
                                     (size_t)ROWS * TOP_K * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(
        (size_t)ROWS * TOP_K * FF * sizeof(float));

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        (size_t)ROWS * OUT * sizeof(float));
    float *actual_mid = malloc((size_t)ROWS * TOP_K * FF * sizeof(float));
    float *actual_out = malloc((size_t)ROWS * OUT * sizeof(float));
    require_ok(mid && out && actual_mid && actual_out,
               "Q4_K MoE mul_mm_id tensor allocations");
    setenv("DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS", "512", 1);
    setenv("DS4_QWEN4_MOE_MUL_MM_TILE_ROWS", "512", 1);
    require_ok(ds4_gpu_qwen4_moe_q4_k_model(
        out, mid, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, ROWS),
        "Qwen Q4_K grouped MoE dispatch");
    require_ok(ds4_gpu_tensor_read(mid, 0, actual_mid,
                                   (size_t)ROWS * TOP_K * FF * sizeof(float)) &&
               ds4_gpu_tensor_read(out, 0, actual_out,
                                   (size_t)ROWS * OUT * sizeof(float)),
               "Q4_K MoE mul_mm_id readback");
    require_array_close("Qwen Q4_K grouped MoE mid", actual_mid,
                        expected_mid, (size_t)ROWS * TOP_K * FF,
                        2e-2f, 4e-3f);
    require_array_close("Qwen Q4_K grouped MoE output", actual_out,
                        expected_out, (size_t)ROWS * OUT, 5e-2f, 8e-3f);

    /* Tile regrouping must stay byte-identical exactly as on the Q4_0
     * route: work-tile granularity only regroups (token, slot) assignments. */
    float *retiled_mid = malloc((size_t)ROWS * TOP_K * FF * sizeof(float));
    float *retiled_out = malloc((size_t)ROWS * OUT * sizeof(float));
    require_ok(retiled_mid && retiled_out,
               "Q4_K MoE retile host allocations");
    setenv("DS4_QWEN4_MOE_MUL_MM_TILE_ROWS", "600", 1);
    require_ok(ds4_gpu_qwen4_moe_q4_k_model(
        out, mid, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, ROWS),
        "Qwen Q4_K grouped MoE single-tile dispatch");
    require_ok(ds4_gpu_tensor_read(mid, 0, retiled_mid,
                                   (size_t)ROWS * TOP_K * FF * sizeof(float)) &&
               ds4_gpu_tensor_read(out, 0, retiled_out,
                                   (size_t)ROWS * OUT * sizeof(float)),
               "Q4_K MoE single-tile readback");
    require_ok(memcmp(retiled_mid, actual_mid,
                      (size_t)ROWS * TOP_K * FF * sizeof(float)) == 0 &&
               memcmp(retiled_out, actual_out,
                      (size_t)ROWS * OUT * sizeof(float)) == 0,
               "Q4_K MoE tile-size byte identity (512 vs 600 rows)");
    unsetenv("DS4_QWEN4_MOE_MUL_MM_TILE_ROWS");
    unsetenv("DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS");
    free(retiled_out);

    /* F32-staged quality route (DS4_QWEN4_MOE_MUL_MM_ID_F32STAGE=1 or the
     * shared DS4_METAL_MPP_MOE_F32STAGE): fp32 operand staging removes the
     * binary16 tile rounding, leaving only the TensorOps accumulate order.
     * Measured on the portable fallback at ~1.9e-6 mid / 6.1e-5 output
     * (vs 2.0e-3 / 3.9e-2 binary16-staged); the bounds keep headroom for
     * the M5 hardware accumulate (exp/m5-tensor-precision measured ~1.7e-4
     * rms there on iq2_xxs) while staying far tighter than the binary16
     * fixture above.  Skipped when the TensorOps route is not compiled —
     * the env would otherwise just relabel the simdgroup tiles. */
    if (ds4_gpu_metal4_tensor_route_enabled()) {
    setenv("DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS", "512", 1);
    setenv("DS4_QWEN4_MOE_MUL_MM_ID_F32STAGE", "1", 1);
    require_ok(ds4_gpu_qwen4_moe_q4_k_model(
        out, mid, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, ROWS),
        "Qwen Q4_K grouped MoE f32stage dispatch");
    require_ok(ds4_gpu_tensor_read(mid, 0, actual_mid,
                                   (size_t)ROWS * TOP_K * FF * sizeof(float)) &&
               ds4_gpu_tensor_read(out, 0, actual_out,
                                   (size_t)ROWS * OUT * sizeof(float)),
               "Q4_K MoE f32stage readback");
    require_array_close("Qwen Q4_K grouped MoE f32stage mid", actual_mid,
                        expected_mid, (size_t)ROWS * TOP_K * FF,
                        4e-3f, 4e-3f);
    require_array_close("Qwen Q4_K grouped MoE f32stage output", actual_out,
                        expected_out, (size_t)ROWS * OUT, 2e-2f, 4e-3f);
    unsetenv("DS4_QWEN4_MOE_MUL_MM_ID_F32STAGE");
    unsetenv("DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS");
    }
    free(retiled_mid);

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(rt);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(xt);
    free(actual_out);
    free(actual_mid);
    free(expected_out);
    free(expected_mid);
    free(route);
    free(selected);
    free(input);
    if (model_fixture_allocation != NULL)
        require_ok(ds4_gpu_set_model_map(model_fixture_allocation,
                                         model_fixture_size),
                   "restore Q8_0 model fixture after Q4_K MoE mul_mm_id");
    free(allocation);
}

static void test_moe_iq2_xxs_q2_k(void) {
    enum {
        IN = 256, FF = 640, DOWN = 768, OUT = 33,
        EXPERTS = 12, TOP_K = 10, ROWS = 9,
    };
    const size_t gate_bytes = (size_t)EXPERTS * FF * (IN / 256u) *
        sizeof(test_block_iq2_xxs);
    const size_t down_bytes = (size_t)EXPERTS * OUT * (DOWN / 256u) *
        sizeof(test_block_q2_k);
    size_t cursor = 0u;
    const size_t gate_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t up_offset = cursor;
    cursor = align_size(cursor + gate_bytes, 64u);
    const size_t down_offset = cursor;
    cursor = align_size(cursor + down_bytes, 4096u);
    require_ok(posix_memalign(&moe_q2_rows8_fixture_allocation,
                              4096u, cursor) == 0,
               "IQ2_XXS/Q2_K MoE model allocation");
    uint8_t *model = moe_q2_rows8_fixture_allocation;
    memset(model, 0, cursor);
    test_block_iq2_xxs *gate =
        (test_block_iq2_xxs *)(model + gate_offset);
    test_block_iq2_xxs *up =
        (test_block_iq2_xxs *)(model + up_offset);
    test_block_q2_k *down =
        (test_block_q2_k *)(model + down_offset);
    fill_iq2_xxs_matrix(gate, EXPERTS * FF, IN, 5u);
    fill_iq2_xxs_matrix(up, EXPERTS * FF, IN, 19u);
    fill_q2_k_matrix(down, EXPERTS * OUT, DOWN, 37u);
    require_ok(ds4_gpu_set_model_map(model, cursor),
               "IQ2_XXS/Q2_K MoE model fixture registration");

    float input[ROWS * IN];
    int32_t selected[ROWS * TOP_K];
    float route[ROWS * TOP_K];
    float expected_mid[ROWS * TOP_K * FF];
    float expected_out[ROWS * OUT];
    for (uint32_t i = 0; i < ROWS * IN; i++)
        input[i] = cosf((float)(i + 1u) * 0.021f) * 0.2f;
    for (uint32_t row = 0; row < ROWS; row++) {
        float route_sum = 0.0f;
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            selected[row * TOP_K + slot] =
                (int32_t)((row * 7u + slot * 5u) % EXPERTS);
            route[row * TOP_K + slot] = (float)(slot + 1u);
            route_sum += route[row * TOP_K + slot];
        }
        for (uint32_t slot = 0; slot < TOP_K; slot++)
            route[row * TOP_K + slot] /= route_sum;
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t slot = 0; slot < TOP_K; slot++) {
            const uint32_t expert = (uint32_t)selected[row * TOP_K + slot];
            for (uint32_t output = 0; output < FF; output++) {
                const uint32_t weight_row = expert * FF + output;
                float g = 0.0f, u = 0.0f;
                for (uint32_t k = 0; k < IN; k++) {
                    const float xv = input[row * IN + k];
                    g = fmaf(iq2_xxs_value(gate, IN, weight_row, k), xv, g);
                    u = fmaf(iq2_xxs_value(up, IN, weight_row, k), xv, u);
                }
                expected_mid[((size_t)row * TOP_K + slot) * FF + output] =
                    (g / (1.0f + expf(-g))) * u;
            }
        }
        for (uint32_t output = 0; output < OUT; output++) {
            float total = 0.0f;
            for (uint32_t slot = 0; slot < TOP_K; slot++) {
                const uint32_t expert =
                    (uint32_t)selected[row * TOP_K + slot];
                const uint32_t weight_row = expert * OUT + output;
                float sum = 0.0f;
                for (uint32_t k = 0; k < FF; k++)
                    sum = fmaf(q2_k_value(down, DOWN, weight_row, k),
                        expected_mid[((size_t)row * TOP_K + slot) * FF + k],
                        sum);
                total = fmaf(sum, route[row * TOP_K + slot], total);
            }
            expected_out[row * OUT + output] = total;
        }
    }

    ds4_gpu_tensor *xt = tensor_from(input, sizeof(input));
    ds4_gpu_tensor *st = tensor_from(selected, sizeof(selected));
    ds4_gpu_tensor *rt = tensor_from(route, sizeof(route));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(sizeof(expected_mid));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(expected_out));
    float *actual_mid = malloc(sizeof(expected_mid));
    float *actual_out = malloc(sizeof(expected_out));
    require_ok(mid && out && actual_mid && actual_out,
               "IQ2_XXS/Q2_K MoE output allocations");
    require_ok(setenv("DS4_QWEN4_MOE_EXACT_MIN_ROWS", "1", 1) == 0,
               "enable IQ2_XXS/Q2_K rows8 fixture");
    require_ok(ds4_gpu_qwen4_moe_iq2_xxs_q2_k_model(
        out, mid, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, ROWS),
        "Qwen IQ2_XXS/Q2_K exact rows8 MoE");
    require_ok(unsetenv("DS4_QWEN4_MOE_EXACT_MIN_ROWS") == 0,
               "clear IQ2_XXS/Q2_K rows8 fixture");
    require_ok(ds4_gpu_tensor_read(mid, 0, actual_mid, sizeof(expected_mid)) &&
               ds4_gpu_tensor_read(out, 0, actual_out, sizeof(expected_out)),
               "IQ2_XXS/Q2_K MoE readback");
    require_array_close_stats("Qwen IQ2_XXS gate/up rows8 mid", actual_mid,
                              expected_mid, ROWS * TOP_K * FF,
                              5e-3f, 2e-3f);
    require_array_close_stats("Qwen Q2_K down rows8 output", actual_out,
                              expected_out, ROWS * OUT, 3e-2f, 3e-3f);

    ds4_gpu_tensor *mid_m1 = ds4_gpu_tensor_alloc(TOP_K * FF * sizeof(float));
    ds4_gpu_tensor *out_m1 = ds4_gpu_tensor_alloc(OUT * sizeof(float));
    require_ok(mid_m1 && out_m1, "IQ2_XXS/Q2_K M=1 allocations");
    require_ok(unsetenv("DS4_QWEN4_MOE_Q2_DOWN_ROWS2") == 0,
               "enable default IQ2_XXS/Q2_K rows2 M=1 down kernel");
    require_ok(ds4_gpu_qwen4_moe_iq2_xxs_q2_k_model(
        out_m1, mid_m1, xt, st, rt, model, cursor,
        gate_offset, up_offset, down_offset,
        IN, FF, OUT, EXPERTS, TOP_K, 1u),
        "Qwen IQ2_XXS/Q2_K M=1 MoE");
    require_ok(ds4_gpu_tensor_read(out_m1, 0, actual_out,
                                   OUT * sizeof(float)),
               "IQ2_XXS/Q2_K M=1 readback");
    require_array_close("Qwen IQ2_XXS/Q2_K M=1", actual_out, expected_out,
                        OUT, 3e-2f, 3e-3f);

    ds4_gpu_tensor_free(out_m1);
    ds4_gpu_tensor_free(mid_m1);
    free(actual_out);
    free(actual_mid);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(rt);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(xt);
    if (model_fixture_allocation != NULL)
        require_ok(ds4_gpu_set_model_map(model_fixture_allocation,
                                         model_fixture_size),
                   "restore Q8_0 model fixture after Q2 MoE");
}

enum {
    /* One final live row plus every supported MTP partial-commit slot. */
    PLE_FIX_TOKENS = DS4_QWEN4_MTP_MAX_DRAFTS + 1,
    PLE_FIX_STREAMS = 4,
    PLE_FIX_DIM = 128,
    PLE_FIX_CHANNELS = PLE_FIX_STREAMS * PLE_FIX_DIM,
    PLE_FIX_STATE_LEN = 9,
    PLE_FIX_KEY_NORM_OFFSET = 86000,
    PLE_FIX_QUERY_NORM_OFFSET = 87100,
    PLE_FIX_CONV_NORM_OFFSET = 88200,
    PLE_FIX_CONV_WEIGHT_OFFSET = 89300,
};

static float sigmoid_reference(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static void ple_reference(
        float *out, float *gated_norm, float *state,
        const float *hidden, const float *key_raw, const float *value_raw,
        const uint8_t *mask, const uint16_t *key_weight,
        const uint16_t *query_weight, const uint16_t *conv_norm_weight,
        const uint16_t *conv_weight, uint32_t tokens) {
    const float eps = 1.0e-6f;
    for (uint32_t token = 0; token < tokens; token++) {
        for (uint32_t stream = 0; stream < PLE_FIX_STREAMS; stream++) {
            const size_t base =
                ((size_t)token * PLE_FIX_STREAMS + stream) * PLE_FIX_DIM;
            const size_t weight_base = (size_t)stream * PLE_FIX_DIM;
            float key_sum = 0.0f;
            float query_sum = 0.0f;
            float value_sum = 0.0f;
            for (uint32_t dim = 0; dim < PLE_FIX_DIM; dim++) {
                const float key = key_raw[base + dim];
                const float query = hidden[base + dim];
                const float value = value_raw[(size_t)token * PLE_FIX_DIM + dim];
                key_sum = fmaf(key, key, key_sum);
                query_sum = fmaf(query, query, query_sum);
                value_sum = fmaf(value, value, value_sum);
            }
            const float key_scale =
                1.0f / sqrtf(key_sum / (float)PLE_FIX_DIM + eps);
            const float query_scale =
                1.0f / sqrtf(query_sum / (float)PLE_FIX_DIM + eps);
            float dot = 0.0f;
            for (uint32_t dim = 0; dim < PLE_FIX_DIM; dim++) {
                const float key = key_raw[base + dim] * key_scale *
                    bf16_to_f32(key_weight[weight_base + dim]);
                const float query = hidden[base + dim] * query_scale *
                    bf16_to_f32(query_weight[weight_base + dim]);
                dot = fmaf(key, query, dot);
            }
            dot /= sqrtf((float)PLE_FIX_DIM);
            const float sign = dot > 0.0f ? 1.0f :
                               (dot < 0.0f ? -1.0f : 0.0f);
            const float gate = sigmoid_reference(
                sign * sqrtf(fmaxf(fabsf(dot), 1.0e-6f)));
            const float norm_scale = 1.0f / sqrtf(
                gate * gate * value_sum / (float)PLE_FIX_DIM + eps);
            const bool active = !mask || mask[token] != 0u;
            for (uint32_t dim = 0; dim < PLE_FIX_DIM; dim++) {
                const float value = active
                    ? gate * value_raw[(size_t)token * PLE_FIX_DIM + dim]
                    : 0.0f;
                out[base + dim] = value;
                gated_norm[base + dim] = active
                    ? value * norm_scale *
                      bf16_to_f32(conv_norm_weight[weight_base + dim])
                    : 0.0f;
            }
        }
    }

    for (uint32_t channel = 0; channel < PLE_FIX_CHANNELS; channel++) {
        float history[PLE_FIX_STATE_LEN];
        for (uint32_t i = 0; i < PLE_FIX_STATE_LEN; i++)
            history[i] = state[(size_t)channel * PLE_FIX_STATE_LEN + i];
        for (uint32_t token = 0; token < tokens; token++) {
            const size_t at = (size_t)token * PLE_FIX_CHANNELS + channel;
            const float current = gated_norm[at];
            const uint16_t *weight = conv_weight + (size_t)channel * 4u;
            float sum = current * bf16_to_f32(weight[3]);
            sum = fmaf(history[0], bf16_to_f32(weight[0]), sum);
            sum = fmaf(history[3], bf16_to_f32(weight[1]), sum);
            sum = fmaf(history[6], bf16_to_f32(weight[2]), sum);
            for (uint32_t i = 0; i + 1u < PLE_FIX_STATE_LEN; i++)
                history[i] = history[i + 1u];
            history[PLE_FIX_STATE_LEN - 1u] = current;
            out[at] += sum * sigmoid_reference(sum);
        }
        for (uint32_t i = 0; i < PLE_FIX_STATE_LEN; i++)
            state[(size_t)channel * PLE_FIX_STATE_LEN + i] = history[i];
    }
}

static void test_ple_gate_conv(void) {
    require_ok(model_fixture_allocation != NULL,
               "PLE fixture model map availability");
    uint8_t *model = model_fixture_allocation;
    uint16_t *key_weight = (uint16_t *)(model + PLE_FIX_KEY_NORM_OFFSET);
    uint16_t *query_weight = (uint16_t *)(model + PLE_FIX_QUERY_NORM_OFFSET);
    uint16_t *conv_norm_weight =
        (uint16_t *)(model + PLE_FIX_CONV_NORM_OFFSET);
    uint16_t *conv_weight =
        (uint16_t *)(model + PLE_FIX_CONV_WEIGHT_OFFSET);
    for (uint32_t channel = 0; channel < PLE_FIX_CHANNELS; channel++) {
        key_weight[channel] = f32_to_bf16(
            0.75f + 0.015625f * (float)(channel % 13u));
        query_weight[channel] = f32_to_bf16(
            0.8125f + 0.0078125f * (float)(channel % 17u));
        conv_norm_weight[channel] = f32_to_bf16(
            0.6875f + 0.01171875f * (float)(channel % 11u));
        for (uint32_t tap = 0; tap < 4u; tap++)
            conv_weight[(size_t)channel * 4u + tap] = f32_to_bf16(
                0.015625f * (float)((int)((channel + tap * 3u) % 9u) - 4));
    }

    const size_t stream_count =
        (size_t)PLE_FIX_TOKENS * PLE_FIX_CHANNELS;
    const size_t value_count = (size_t)PLE_FIX_TOKENS * PLE_FIX_DIM;
    const size_t state_count =
        (size_t)PLE_FIX_CHANNELS * PLE_FIX_STATE_LEN;
    float *hidden = malloc(stream_count * sizeof(float));
    float *key_raw = malloc(stream_count * sizeof(float));
    float *value_raw = malloc(value_count * sizeof(float));
    float *initial_state = malloc(state_count * sizeof(float));
    float *expected = malloc(stream_count * sizeof(float));
    float *expected_norm = malloc(stream_count * sizeof(float));
    float *expected_state = malloc(state_count * sizeof(float));
    float *actual = malloc(stream_count * sizeof(float));
    float *actual_norm = malloc(stream_count * sizeof(float));
    float *actual_state = malloc(state_count * sizeof(float));
    require_ok(hidden && key_raw && value_raw && initial_state && expected &&
                   expected_norm && expected_state && actual && actual_norm &&
                   actual_state,
               "PLE host allocations");
    for (size_t i = 0; i < stream_count; i++) {
        hidden[i] = 0.00625f * (float)((int)((i * 7u + 5u) % 67u) - 33);
        key_raw[i] = 0.005f * (float)((int)((i * 11u + 3u) % 71u) - 35);
    }
    for (size_t i = 0; i < value_count; i++)
        value_raw[i] = 0.0078125f * (float)((int)((i * 13u + 1u) % 61u) - 30);
    for (size_t i = 0; i < state_count; i++)
        initial_state[i] = 0.001953125f *
            (float)((int)((i * 5u + 2u) % 23u) - 11);
    uint8_t mask[PLE_FIX_TOKENS];
    memset(mask, 1, sizeof(mask));
    mask[1] = 0u;
    memcpy(expected_state, initial_state, state_count * sizeof(float));
    ple_reference(expected, expected_norm, expected_state,
                  hidden, key_raw, value_raw, mask, key_weight, query_weight,
                  conv_norm_weight, conv_weight, PLE_FIX_TOKENS);

    ds4_gpu_tensor *hidden_t = tensor_from(
        hidden, stream_count * sizeof(float));
    ds4_gpu_tensor *key_t = tensor_from(
        key_raw, stream_count * sizeof(float));
    ds4_gpu_tensor *value_t = tensor_from(
        value_raw, value_count * sizeof(float));
    ds4_gpu_tensor *mask_t = tensor_from(mask, sizeof(mask));
    ds4_gpu_tensor *state_t = tensor_from(
        initial_state, state_count * sizeof(float));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(stream_count * sizeof(float));
    ds4_gpu_tensor *norm_t = ds4_gpu_tensor_alloc(stream_count * sizeof(float));
    require_ok(out_t && norm_t, "PLE device allocations");
    require_ok(ds4_gpu_qwen4_ple_gate_conv_model(
                   out_t, norm_t, state_t, hidden_t, key_t, value_t, mask_t,
                   model, model_fixture_size,
                   PLE_FIX_KEY_NORM_OFFSET, PLE_FIX_QUERY_NORM_OFFSET,
                   PLE_FIX_CONV_NORM_OFFSET, PLE_FIX_CONV_WEIGHT_OFFSET,
                   PLE_FIX_TOKENS, PLE_FIX_STREAMS, PLE_FIX_DIM, 4u, 3u,
                   1.0e-6f),
               "PLE fused gate and convolution dispatch");
    require_ok(ds4_gpu_tensor_read(out_t, 0, actual,
                                   stream_count * sizeof(float)),
               "PLE output readback");
    require_ok(ds4_gpu_tensor_read(norm_t, 0, actual_norm,
                                   stream_count * sizeof(float)),
               "PLE normalized value readback");
    require_ok(ds4_gpu_tensor_read(state_t, 0, actual_state,
                                   state_count * sizeof(float)),
               "PLE state readback");
    require_array_close("Qwen PLE gated dilated convolution", actual,
                        expected, stream_count, 2e-5f, 3e-5f);
    require_array_close("Qwen PLE normalized gated values", actual_norm,
                        expected_norm, stream_count, 2e-5f, 3e-5f);
    require_array_close("Qwen PLE convolution state", actual_state,
                        expected_state, state_count, 2e-5f, 3e-5f);

    const uint32_t first = 2u;
    const uint32_t second = PLE_FIX_TOKENS - first;
    ds4_gpu_tensor *split_state_t = tensor_from(
        initial_state, state_count * sizeof(float));
    ds4_gpu_tensor *split_out_a = ds4_gpu_tensor_alloc(
        (size_t)first * PLE_FIX_CHANNELS * sizeof(float));
    ds4_gpu_tensor *split_norm_a = ds4_gpu_tensor_alloc(
        (size_t)first * PLE_FIX_CHANNELS * sizeof(float));
    ds4_gpu_tensor *split_out_b = ds4_gpu_tensor_alloc(
        (size_t)second * PLE_FIX_CHANNELS * sizeof(float));
    ds4_gpu_tensor *split_norm_b = ds4_gpu_tensor_alloc(
        (size_t)second * PLE_FIX_CHANNELS * sizeof(float));
    ds4_gpu_tensor *hidden_b = tensor_from(
        hidden + (size_t)first * PLE_FIX_CHANNELS,
        (size_t)second * PLE_FIX_CHANNELS * sizeof(float));
    ds4_gpu_tensor *key_b = tensor_from(
        key_raw + (size_t)first * PLE_FIX_CHANNELS,
        (size_t)second * PLE_FIX_CHANNELS * sizeof(float));
    ds4_gpu_tensor *value_b = tensor_from(
        value_raw + (size_t)first * PLE_FIX_DIM,
        (size_t)second * PLE_FIX_DIM * sizeof(float));
    ds4_gpu_tensor *mask_b = tensor_from(mask + first, second * sizeof(uint8_t));
    require_ok(split_out_a && split_norm_a && split_out_b && split_norm_b,
               "PLE split device allocations");
    require_ok(ds4_gpu_qwen4_ple_gate_conv_model(
                   split_out_a, split_norm_a, split_state_t,
                   hidden_t, key_t, value_t, mask_t,
                   model, model_fixture_size,
                   PLE_FIX_KEY_NORM_OFFSET, PLE_FIX_QUERY_NORM_OFFSET,
                   PLE_FIX_CONV_NORM_OFFSET, PLE_FIX_CONV_WEIGHT_OFFSET,
                   first, PLE_FIX_STREAMS, PLE_FIX_DIM, 4u, 3u, 1.0e-6f),
               "PLE split first dispatch");
    require_ok(ds4_gpu_qwen4_ple_gate_conv_model(
                   split_out_b, split_norm_b, split_state_t,
                   hidden_b, key_b, value_b, mask_b,
                   model, model_fixture_size,
                   PLE_FIX_KEY_NORM_OFFSET, PLE_FIX_QUERY_NORM_OFFSET,
                   PLE_FIX_CONV_NORM_OFFSET, PLE_FIX_CONV_WEIGHT_OFFSET,
                   second, PLE_FIX_STREAMS, PLE_FIX_DIM, 4u, 3u, 1.0e-6f),
               "PLE split second dispatch");
    require_ok(ds4_gpu_tensor_read(
                   split_out_a, 0, actual,
                   (size_t)first * PLE_FIX_CHANNELS * sizeof(float)),
               "PLE split first output readback");
    require_ok(ds4_gpu_tensor_read(
                   split_out_b, 0,
                   actual + (size_t)first * PLE_FIX_CHANNELS,
                   (size_t)second * PLE_FIX_CHANNELS * sizeof(float)),
               "PLE split second output readback");
    require_ok(ds4_gpu_tensor_read(
                   split_norm_a, 0, actual_norm,
                   (size_t)first * PLE_FIX_CHANNELS * sizeof(float)),
               "PLE split first normalized readback");
    require_ok(ds4_gpu_tensor_read(
                   split_norm_b, 0,
                   actual_norm + (size_t)first * PLE_FIX_CHANNELS,
                   (size_t)second * PLE_FIX_CHANNELS * sizeof(float)),
               "PLE split second normalized readback");
    require_array_close("Qwen PLE split output continuity", actual,
                        expected, stream_count, 2e-5f, 3e-5f);
    require_array_close("Qwen PLE split normalized continuity", actual_norm,
                        expected_norm, stream_count, 2e-5f, 3e-5f);
    require_ok(ds4_gpu_tensor_read(split_state_t, 0, actual_state,
                                   state_count * sizeof(float)),
               "PLE split state readback");
    require_array_close("Qwen PLE split state continuity", actual_state,
                        expected_state, state_count, 2e-5f, 3e-5f);

    const uint32_t capture_slots = PLE_FIX_TOKENS - 1u;
    float *capture_seq = malloc(
        (size_t)capture_slots * state_count * sizeof(float));
    float *prefix_state = malloc(state_count * sizeof(float));
    ds4_gpu_tensor *capture_state_t = tensor_from(
        initial_state, state_count * sizeof(float));
    ds4_gpu_tensor *capture_out_t = ds4_gpu_tensor_alloc(
        stream_count * sizeof(float));
    ds4_gpu_tensor *capture_norm_t = ds4_gpu_tensor_alloc(
        stream_count * sizeof(float));
    ds4_gpu_tensor *capture_seq_t = ds4_gpu_tensor_alloc(
        (size_t)capture_slots * state_count * sizeof(float));
    require_ok(capture_seq && prefix_state && capture_state_t &&
                   capture_out_t && capture_norm_t && capture_seq_t,
               "PLE capture allocation");
    require_ok(ds4_gpu_qwen4_ple_gate_conv_capture_model(
                   capture_out_t, capture_norm_t, capture_state_t,
                   capture_seq_t, hidden_t, key_t, value_t, mask_t,
                   model, model_fixture_size,
                   PLE_FIX_KEY_NORM_OFFSET, PLE_FIX_QUERY_NORM_OFFSET,
                   PLE_FIX_CONV_NORM_OFFSET, PLE_FIX_CONV_WEIGHT_OFFSET,
                   PLE_FIX_TOKENS, PLE_FIX_STREAMS, PLE_FIX_DIM,
                   4u, 3u, capture_slots, 1.0e-6f),
               "PLE verifier state capture dispatch");
    require_ok(ds4_gpu_tensor_read(
                   capture_seq_t, 0, capture_seq,
                   (size_t)capture_slots * state_count * sizeof(float)),
               "PLE verifier state capture readback");
    for (uint32_t accepted = 1u; accepted <= capture_slots; accepted++) {
        ds4_gpu_tensor *prefix_state_t = tensor_from(
            initial_state, state_count * sizeof(float));
        ds4_gpu_tensor *prefix_out_t = ds4_gpu_tensor_alloc(
            (size_t)accepted * PLE_FIX_CHANNELS * sizeof(float));
        ds4_gpu_tensor *prefix_norm_t = ds4_gpu_tensor_alloc(
            (size_t)accepted * PLE_FIX_CHANNELS * sizeof(float));
        require_ok(prefix_state_t && prefix_out_t && prefix_norm_t &&
                       ds4_gpu_qwen4_ple_gate_conv_model(
                           prefix_out_t, prefix_norm_t, prefix_state_t,
                           hidden_t, key_t, value_t, mask_t,
                           model, model_fixture_size,
                           PLE_FIX_KEY_NORM_OFFSET,
                           PLE_FIX_QUERY_NORM_OFFSET,
                           PLE_FIX_CONV_NORM_OFFSET,
                           PLE_FIX_CONV_WEIGHT_OFFSET,
                           accepted, PLE_FIX_STREAMS, PLE_FIX_DIM,
                           4u, 3u, 1.0e-6f),
                   "PLE accepted-prefix replay dispatch");
        require_ok(ds4_gpu_tensor_read(
                       prefix_state_t, 0, prefix_state,
                       state_count * sizeof(float)),
                   "PLE accepted-prefix state readback");
        require_ok(memcmp(
                       prefix_state,
                       capture_seq + (size_t)(accepted - 1u) * state_count,
                       state_count * sizeof(float)) == 0,
                   "PLE captured state byte equality");
        ds4_gpu_tensor_free(prefix_norm_t);
        ds4_gpu_tensor_free(prefix_out_t);
        ds4_gpu_tensor_free(prefix_state_t);
    }
    puts("Qwen PLE verifier capture/replay byte equality PASS");
    ds4_gpu_tensor_free(capture_seq_t);
    ds4_gpu_tensor_free(capture_norm_t);
    ds4_gpu_tensor_free(capture_out_t);
    ds4_gpu_tensor_free(capture_state_t);
    free(prefix_state);
    free(capture_seq);

    ds4_gpu_tensor_free(mask_b);
    ds4_gpu_tensor_free(value_b);
    ds4_gpu_tensor_free(key_b);
    ds4_gpu_tensor_free(hidden_b);
    ds4_gpu_tensor_free(split_norm_b);
    ds4_gpu_tensor_free(split_out_b);
    ds4_gpu_tensor_free(split_norm_a);
    ds4_gpu_tensor_free(split_out_a);
    ds4_gpu_tensor_free(split_state_t);
    ds4_gpu_tensor_free(norm_t);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(state_t);
    ds4_gpu_tensor_free(mask_t);
    ds4_gpu_tensor_free(value_t);
    ds4_gpu_tensor_free(key_t);
    ds4_gpu_tensor_free(hidden_t);
    free(actual_state);
    free(actual_norm);
    free(actual);
    free(expected_state);
    free(expected_norm);
    free(expected);
    free(initial_state);
    free(value_raw);
    free(key_raw);
    free(hidden);
}

static void test_ple_long_tiled_schedule(void) {
    enum { TOKENS = 2064, TILE = 512 };
    const size_t stream_count = (size_t)TOKENS * PLE_FIX_CHANNELS;
    const size_t value_count = (size_t)TOKENS * PLE_FIX_DIM;
    const size_t state_count =
        (size_t)PLE_FIX_CHANNELS * PLE_FIX_STATE_LEN;
    float *hidden = malloc(stream_count * sizeof(float));
    float *key = malloc(stream_count * sizeof(float));
    float *value = malloc(value_count * sizeof(float));
    float *initial_state = malloc(state_count * sizeof(float));
    float *full = malloc(stream_count * sizeof(float));
    float *tiled = malloc(stream_count * sizeof(float));
    float *full_state = malloc(state_count * sizeof(float));
    float *tiled_state = malloc(state_count * sizeof(float));
    require_ok(hidden && key && value && initial_state && full && tiled &&
                   full_state && tiled_state,
               "long tiled PLE host allocation");
    for (size_t i = 0; i < stream_count; i++) {
        hidden[i] = 0.003125f *
            (float)((int)((i * 17u + 5u) % 97u) - 48);
        key[i] = 0.0025f *
            (float)((int)((i * 19u + 11u) % 89u) - 44);
    }
    for (size_t i = 0; i < value_count; i++)
        value[i] = 0.00390625f *
            (float)((int)((i * 23u + 3u) % 83u) - 41);
    for (size_t i = 0; i < state_count; i++)
        initial_state[i] = 0.001f * (float)((int)(i % 31u) - 15);

    ds4_gpu_tensor *hidden_t = tensor_from(
        hidden, stream_count * sizeof(float));
    ds4_gpu_tensor *key_t = tensor_from(key, stream_count * sizeof(float));
    ds4_gpu_tensor *value_t = tensor_from(
        value, value_count * sizeof(float));
    ds4_gpu_tensor *full_out_t = ds4_gpu_tensor_alloc(
        stream_count * sizeof(float));
    ds4_gpu_tensor *full_norm_t = ds4_gpu_tensor_alloc(
        stream_count * sizeof(float));
    ds4_gpu_tensor *full_state_t = tensor_from(
        initial_state, state_count * sizeof(float));
    require_ok(full_out_t && full_norm_t,
               "long tiled PLE reference allocation");
    require_ok(ds4_gpu_qwen4_ple_gate_conv_model(
                   full_out_t, full_norm_t, full_state_t,
                   hidden_t, key_t, value_t, NULL,
                   model_fixture_allocation, model_fixture_size,
                   PLE_FIX_KEY_NORM_OFFSET, PLE_FIX_QUERY_NORM_OFFSET,
                   PLE_FIX_CONV_NORM_OFFSET, PLE_FIX_CONV_WEIGHT_OFFSET,
                   TOKENS, PLE_FIX_STREAMS, PLE_FIX_DIM, 4u, 3u, 1.0e-6f),
               "long tiled PLE reference dispatch");
    require_ok(ds4_gpu_tensor_read(full_out_t, 0, full,
                                   stream_count * sizeof(float)) &&
                   ds4_gpu_tensor_read(full_state_t, 0, full_state,
                                       state_count * sizeof(float)),
               "long tiled PLE reference readback");

    ds4_gpu_tensor *tiled_out_t = ds4_gpu_tensor_alloc(
        stream_count * sizeof(float));
    ds4_gpu_tensor *tiled_norm_t = ds4_gpu_tensor_alloc(
        stream_count * sizeof(float));
    ds4_gpu_tensor *tiled_state_t = tensor_from(
        initial_state, state_count * sizeof(float));
    require_ok(tiled_out_t && tiled_norm_t && ds4_gpu_begin_commands(),
               "long tiled PLE command batch");
    for (uint32_t start = 0; start < TOKENS; start += TILE) {
        const uint32_t count = TOKENS - start < TILE ? TOKENS - start : TILE;
        const uint64_t stream_offset =
            (uint64_t)start * PLE_FIX_CHANNELS * sizeof(float);
        const uint64_t stream_bytes =
            (uint64_t)count * PLE_FIX_CHANNELS * sizeof(float);
        const uint64_t value_offset =
            (uint64_t)start * PLE_FIX_DIM * sizeof(float);
        const uint64_t value_bytes =
            (uint64_t)count * PLE_FIX_DIM * sizeof(float);
        ds4_gpu_tensor *hidden_view = ds4_gpu_tensor_view(
            hidden_t, stream_offset, stream_bytes);
        ds4_gpu_tensor *key_view = ds4_gpu_tensor_view(
            key_t, stream_offset, stream_bytes);
        ds4_gpu_tensor *value_view = ds4_gpu_tensor_view(
            value_t, value_offset, value_bytes);
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
            tiled_out_t, stream_offset, stream_bytes);
        ds4_gpu_tensor *norm_view = ds4_gpu_tensor_view(
            tiled_norm_t, stream_offset, stream_bytes);
        require_ok(hidden_view && key_view && value_view && out_view &&
                       norm_view &&
                       ds4_gpu_qwen4_ple_gate_conv_model(
                           out_view, norm_view, tiled_state_t,
                           hidden_view, key_view, value_view, NULL,
                           model_fixture_allocation, model_fixture_size,
                           PLE_FIX_KEY_NORM_OFFSET,
                           PLE_FIX_QUERY_NORM_OFFSET,
                           PLE_FIX_CONV_NORM_OFFSET,
                           PLE_FIX_CONV_WEIGHT_OFFSET,
                           count, PLE_FIX_STREAMS, PLE_FIX_DIM,
                           4u, 3u, 1.0e-6f),
                   "long tiled PLE dispatch");
        ds4_gpu_tensor_free(norm_view);
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(value_view);
        ds4_gpu_tensor_free(key_view);
        ds4_gpu_tensor_free(hidden_view);
        if (start + count < TOKENS)
            require_ok(ds4_gpu_flush_commands(),
                       "long tiled PLE command handoff");
    }
    require_ok(ds4_gpu_end_commands(), "long tiled PLE batch completion");
    require_ok(ds4_gpu_tensor_read(tiled_out_t, 0, tiled,
                                   stream_count * sizeof(float)) &&
                   ds4_gpu_tensor_read(tiled_state_t, 0, tiled_state,
                                       state_count * sizeof(float)),
               "long tiled PLE readback");
    require_ok(memcmp(full, tiled, stream_count * sizeof(float)) == 0,
               "long tiled PLE output exactness");
    require_ok(memcmp(full_state, tiled_state,
                      state_count * sizeof(float)) == 0,
               "long tiled PLE state exactness");
    printf("Qwen PLE 2064 vs 512-token command-buffer tiles exact PASS\n");

    ds4_gpu_tensor_free(tiled_state_t);
    ds4_gpu_tensor_free(tiled_norm_t);
    ds4_gpu_tensor_free(tiled_out_t);
    ds4_gpu_tensor_free(full_state_t);
    ds4_gpu_tensor_free(full_norm_t);
    ds4_gpu_tensor_free(full_out_t);
    ds4_gpu_tensor_free(value_t);
    ds4_gpu_tensor_free(key_t);
    ds4_gpu_tensor_free(hidden_t);
    free(tiled_state);
    free(full_state);
    free(tiled);
    free(full);
    free(initial_state);
    free(value);
    free(key);
    free(hidden);
}

enum {
    /* One final live row plus every supported MTP partial-commit slot. */
    GDN_FIX_TOKENS = DS4_QWEN4_MTP_MAX_DRAFTS + 1,
    GDN_FIX_DIM = 128,
    GDN_FIX_KH = 1,
    GDN_FIX_VH = 3,
    GDN_FIX_CONV_WIDTH = 4,
    GDN_FIX_KEY_DIM = GDN_FIX_KH * GDN_FIX_DIM,
    GDN_FIX_VALUE_DIM = GDN_FIX_VH * GDN_FIX_DIM,
    GDN_FIX_CONV_DIM = 2 * GDN_FIX_KEY_DIM + GDN_FIX_VALUE_DIM,
    GDN_FIX_CONV_OFFSET = 70000,
    GDN_FIX_A_LOG_OFFSET = 76000,
    GDN_FIX_DT_BIAS_OFFSET = 76256,
    GDN_FIX_NORM_OFFSET = 76512,
};

static float silu_reference(float value) {
    return value / (1.0f + expf(-value));
}

static void gdn_prepare_reference(
        float *q, float *k, float *v, float *decay, float *beta,
        float *conv_state, const float *mixed_qkv, const float *raw_decay,
        const float *raw_beta, const uint8_t *mask,
        const uint16_t *conv_weight, const uint16_t *a_log,
        const uint16_t *dt_bias, uint32_t tokens) {
    for (uint32_t channel = 0; channel < GDN_FIX_CONV_DIM; channel++) {
        float history[GDN_FIX_CONV_WIDTH];
        for (uint32_t tap = 0; tap < GDN_FIX_CONV_WIDTH; tap++)
            history[tap] = conv_state[channel * GDN_FIX_CONV_WIDTH + tap];
        for (uint32_t token = 0; token < tokens; token++) {
            for (uint32_t tap = 0; tap + 1u < GDN_FIX_CONV_WIDTH; tap++)
                history[tap] = history[tap + 1u];
            history[GDN_FIX_CONV_WIDTH - 1u] = mask && !mask[token]
                ? 0.0f
                : mixed_qkv[(size_t)token * GDN_FIX_CONV_DIM + channel];
            float sum = 0.0f;
            for (uint32_t tap = 0; tap < GDN_FIX_CONV_WIDTH; tap++)
                sum = fmaf(history[tap],
                           bf16_to_f32(conv_weight[
                               channel * GDN_FIX_CONV_WIDTH + tap]), sum);
            const float value = silu_reference(sum);
            if (channel < GDN_FIX_KEY_DIM) {
                q[(size_t)token * GDN_FIX_KEY_DIM + channel] = value;
            } else if (channel < 2u * GDN_FIX_KEY_DIM) {
                k[(size_t)token * GDN_FIX_KEY_DIM +
                  channel - GDN_FIX_KEY_DIM] = value;
            } else {
                v[(size_t)token * GDN_FIX_VALUE_DIM +
                  channel - 2u * GDN_FIX_KEY_DIM] = value;
            }
        }
        for (uint32_t tap = 0; tap < GDN_FIX_CONV_WIDTH; tap++)
            conv_state[channel * GDN_FIX_CONV_WIDTH + tap] = history[tap];
    }

    for (uint32_t token = 0; token < tokens; token++) {
        for (uint32_t head = 0; head < GDN_FIX_KH; head++) {
            const size_t base =
                ((size_t)token * GDN_FIX_KH + head) * GDN_FIX_DIM;
            float qsum = 0.0f, ksum = 0.0f;
            for (uint32_t dim = 0; dim < GDN_FIX_DIM; dim++) {
                qsum = fmaf(q[base + dim], q[base + dim], qsum);
                ksum = fmaf(k[base + dim], k[base + dim], ksum);
            }
            const float qscale =
                1.0f / sqrtf((qsum + 1.0e-6f) * GDN_FIX_DIM);
            const float kscale = 1.0f / sqrtf(ksum + 1.0e-6f);
            for (uint32_t dim = 0; dim < GDN_FIX_DIM; dim++) {
                q[base + dim] *= qscale;
                k[base + dim] *= kscale;
            }
        }
        for (uint32_t head = 0; head < GDN_FIX_VH; head++) {
            const size_t at = (size_t)token * GDN_FIX_VH + head;
            const bool active = !mask || mask[token] != 0u;
            const float a = active ? raw_decay[at] : 0.0f;
            const float b = active ? raw_beta[at] : 0.0f;
            const float shifted = a + bf16_to_f32(dt_bias[head]);
            const float softplus = shifted > 20.0f
                ? shifted : log1pf(expf(shifted));
            decay[at] = expf(-expf(bf16_to_f32(a_log[head])) * softplus);
            beta[at] = 1.0f / (1.0f + expf(-b));
        }
    }
}

static void run_gdn_prepare_fixture(
        const void *model, size_t model_size, uint32_t tokens,
        const float *mixed_qkv, const float *raw_decay,
        const float *raw_beta, const uint8_t *mask,
        ds4_gpu_tensor *conv_state, float *q, float *k, float *v,
        float *decay, float *beta) {
    const size_t qk_count = (size_t)tokens * GDN_FIX_KEY_DIM;
    const size_t value_count = (size_t)tokens * GDN_FIX_VALUE_DIM;
    const size_t gate_count = (size_t)tokens * GDN_FIX_VH;
    ds4_gpu_tensor *mixed_t = tensor_from(
        mixed_qkv, (size_t)tokens * GDN_FIX_CONV_DIM * sizeof(float));
    ds4_gpu_tensor *raw_decay_t = tensor_from(
        raw_decay, gate_count * sizeof(float));
    ds4_gpu_tensor *raw_beta_t = tensor_from(
        raw_beta, gate_count * sizeof(float));
    ds4_gpu_tensor *mask_t = mask ? tensor_from(mask, tokens) : NULL;
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(qk_count * sizeof(float));
    ds4_gpu_tensor *k_t = ds4_gpu_tensor_alloc(qk_count * sizeof(float));
    ds4_gpu_tensor *v_t = ds4_gpu_tensor_alloc(value_count * sizeof(float));
    ds4_gpu_tensor *decay_t = ds4_gpu_tensor_alloc(gate_count * sizeof(float));
    ds4_gpu_tensor *beta_t = ds4_gpu_tensor_alloc(gate_count * sizeof(float));
    require_ok(q_t && k_t && v_t && decay_t && beta_t,
               "Gated DeltaNet preparation output allocation");
    require_ok(ds4_gpu_qwen4_gdn_prepare_model(
                   q_t, k_t, v_t, decay_t, beta_t, conv_state,
                   mixed_t, raw_decay_t, raw_beta_t, mask_t,
                   model, model_size, GDN_FIX_CONV_OFFSET,
                   GDN_FIX_A_LOG_OFFSET, GDN_FIX_DT_BIAS_OFFSET,
                   tokens, GDN_FIX_KH, GDN_FIX_VH, GDN_FIX_DIM,
                   GDN_FIX_CONV_WIDTH),
               "Gated DeltaNet preparation dispatch");
    require_ok(ds4_gpu_tensor_read(q_t, 0, q, qk_count * sizeof(float)),
               "Gated DeltaNet prepared Q readback");
    require_ok(ds4_gpu_tensor_read(k_t, 0, k, qk_count * sizeof(float)),
               "Gated DeltaNet prepared K readback");
    require_ok(ds4_gpu_tensor_read(v_t, 0, v, value_count * sizeof(float)),
               "Gated DeltaNet prepared V readback");
    require_ok(ds4_gpu_tensor_read(decay_t, 0, decay,
                                   gate_count * sizeof(float)),
               "Gated DeltaNet decay readback");
    require_ok(ds4_gpu_tensor_read(beta_t, 0, beta,
                                   gate_count * sizeof(float)),
               "Gated DeltaNet beta readback");
    ds4_gpu_tensor_free(beta_t);
    ds4_gpu_tensor_free(decay_t);
    ds4_gpu_tensor_free(v_t);
    ds4_gpu_tensor_free(k_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(mask_t);
    ds4_gpu_tensor_free(raw_beta_t);
    ds4_gpu_tensor_free(raw_decay_t);
    ds4_gpu_tensor_free(mixed_t);
}

static void test_gdn_prepare_and_output_norm(void) {
    void *model = model_fixture_allocation;
    const size_t model_size = model_fixture_size;
    require_ok(model != NULL && model_size > GDN_FIX_NORM_OFFSET +
                   GDN_FIX_DIM * sizeof(uint16_t),
               "Gated DeltaNet model fixture range");
    uint16_t *conv_weight = (uint16_t *)((uint8_t *)model +
                                         GDN_FIX_CONV_OFFSET);
    uint16_t *a_log = (uint16_t *)((uint8_t *)model +
                                   GDN_FIX_A_LOG_OFFSET);
    uint16_t *dt_bias = (uint16_t *)((uint8_t *)model +
                                     GDN_FIX_DT_BIAS_OFFSET);
    uint16_t *norm_weight = (uint16_t *)((uint8_t *)model +
                                         GDN_FIX_NORM_OFFSET);
    for (uint32_t channel = 0; channel < GDN_FIX_CONV_DIM; channel++)
        for (uint32_t tap = 0; tap < GDN_FIX_CONV_WIDTH; tap++)
            conv_weight[channel * GDN_FIX_CONV_WIDTH + tap] =
                f32_to_bf16(0.03f * (float)(tap + 1u) +
                            0.002f * (float)((int)(channel % 9u) - 4));
    for (uint32_t head = 0; head < GDN_FIX_VH; head++) {
        a_log[head] = f32_to_bf16(logf(0.25f + 0.15f * (float)head));
        dt_bias[head] = f32_to_bf16(-0.35f + 0.2f * (float)head);
    }
    for (uint32_t dim = 0; dim < GDN_FIX_DIM; dim++)
        norm_weight[dim] =
            f32_to_bf16(0.8f + 0.002f * (float)(dim % 29u));

    float mixed[GDN_FIX_TOKENS * GDN_FIX_CONV_DIM];
    float raw_decay[GDN_FIX_TOKENS * GDN_FIX_VH];
    float raw_beta[GDN_FIX_TOKENS * GDN_FIX_VH];
    uint8_t mask[GDN_FIX_TOKENS];
    memset(mask, 1, sizeof(mask));
    mask[1] = 0u;
    float initial_state[GDN_FIX_CONV_DIM * GDN_FIX_CONV_WIDTH];
    for (size_t i = 0; i < sizeof(mixed) / sizeof(mixed[0]); i++)
        mixed[i] = 0.0125f * (float)((int)((i * 7u + 3u) % 31u) - 15);
    for (size_t i = 0; i < sizeof(raw_decay) / sizeof(raw_decay[0]); i++) {
        raw_decay[i] = 0.04f * (float)((int)(i % 7u) - 3);
        raw_beta[i] = 0.075f * (float)((int)(i % 5u) - 2);
    }
    for (size_t i = 0; i < sizeof(initial_state) / sizeof(initial_state[0]); i++)
        initial_state[i] = 0.005f * (float)((int)(i % 13u) - 6);

    float expected_q[GDN_FIX_TOKENS * GDN_FIX_KEY_DIM];
    float expected_k[GDN_FIX_TOKENS * GDN_FIX_KEY_DIM];
    float expected_v[GDN_FIX_TOKENS * GDN_FIX_VALUE_DIM];
    float expected_decay[GDN_FIX_TOKENS * GDN_FIX_VH];
    float expected_beta[GDN_FIX_TOKENS * GDN_FIX_VH];
    float expected_state[GDN_FIX_CONV_DIM * GDN_FIX_CONV_WIDTH];
    memcpy(expected_state, initial_state, sizeof(expected_state));
    gdn_prepare_reference(expected_q, expected_k, expected_v,
                          expected_decay, expected_beta, expected_state,
                          mixed, raw_decay, raw_beta, mask, conv_weight,
                          a_log, dt_bias, GDN_FIX_TOKENS);

    float actual_q[GDN_FIX_TOKENS * GDN_FIX_KEY_DIM];
    float actual_k[GDN_FIX_TOKENS * GDN_FIX_KEY_DIM];
    float actual_v[GDN_FIX_TOKENS * GDN_FIX_VALUE_DIM];
    float actual_decay[GDN_FIX_TOKENS * GDN_FIX_VH];
    float actual_beta[GDN_FIX_TOKENS * GDN_FIX_VH];
    float actual_state[GDN_FIX_CONV_DIM * GDN_FIX_CONV_WIDTH];
    ds4_gpu_tensor *state_t = tensor_from(initial_state, sizeof(initial_state));
    run_gdn_prepare_fixture(model, model_size, GDN_FIX_TOKENS,
                            mixed, raw_decay, raw_beta, mask, state_t,
                            actual_q, actual_k, actual_v,
                            actual_decay, actual_beta);
    require_ok(ds4_gpu_tensor_read(state_t, 0, actual_state,
                                   sizeof(actual_state)),
               "Gated DeltaNet convolution state readback");
    require_array_close("Qwen GDN prepared Q", actual_q, expected_q,
                        sizeof(actual_q) / sizeof(actual_q[0]), 2e-6f, 2e-5f);
    require_array_close("Qwen GDN prepared K", actual_k, expected_k,
                        sizeof(actual_k) / sizeof(actual_k[0]), 2e-6f, 2e-5f);
    require_array_close("Qwen GDN prepared V", actual_v, expected_v,
                        sizeof(actual_v) / sizeof(actual_v[0]), 2e-6f, 2e-5f);
    require_array_close("Qwen GDN decay transform", actual_decay,
                        expected_decay,
                        sizeof(actual_decay) / sizeof(actual_decay[0]),
                        2e-6f, 2e-5f);
    require_array_close("Qwen GDN beta transform", actual_beta, expected_beta,
                        sizeof(actual_beta) / sizeof(actual_beta[0]),
                        2e-6f, 2e-5f);
    require_array_close("Qwen GDN convolution state", actual_state,
                        expected_state,
                        sizeof(actual_state) / sizeof(actual_state[0]),
                        2e-6f, 2e-5f);
    ds4_gpu_tensor_free(state_t);

    float split_q[GDN_FIX_TOKENS * GDN_FIX_KEY_DIM];
    float split_k[GDN_FIX_TOKENS * GDN_FIX_KEY_DIM];
    float split_v[GDN_FIX_TOKENS * GDN_FIX_VALUE_DIM];
    float split_decay[GDN_FIX_TOKENS * GDN_FIX_VH];
    float split_beta[GDN_FIX_TOKENS * GDN_FIX_VH];
    state_t = tensor_from(initial_state, sizeof(initial_state));
    run_gdn_prepare_fixture(model, model_size, 2u,
                            mixed, raw_decay, raw_beta, mask, state_t,
                            split_q, split_k, split_v, split_decay, split_beta);
    run_gdn_prepare_fixture(
        model, model_size, GDN_FIX_TOKENS - 2u,
        mixed + 2u * GDN_FIX_CONV_DIM, raw_decay + 2u * GDN_FIX_VH,
        raw_beta + 2u * GDN_FIX_VH, mask + 2u, state_t,
        split_q + 2u * GDN_FIX_KEY_DIM, split_k + 2u * GDN_FIX_KEY_DIM,
        split_v + 2u * GDN_FIX_VALUE_DIM, split_decay + 2u * GDN_FIX_VH,
        split_beta + 2u * GDN_FIX_VH);
    require_array_close("Qwen GDN chunked preparation Q", split_q, expected_q,
                        sizeof(split_q) / sizeof(split_q[0]), 2e-6f, 2e-5f);
    require_array_close("Qwen GDN chunked preparation K", split_k, expected_k,
                        sizeof(split_k) / sizeof(split_k[0]), 2e-6f, 2e-5f);
    require_array_close("Qwen GDN chunked preparation V", split_v, expected_v,
                        sizeof(split_v) / sizeof(split_v[0]), 2e-6f, 2e-5f);
    require_ok(ds4_gpu_tensor_read(state_t, 0, actual_state,
                                   sizeof(actual_state)),
               "chunked Gated DeltaNet convolution state readback");
    require_array_close("Qwen GDN chunked convolution state", actual_state,
                        expected_state,
                        sizeof(actual_state) / sizeof(actual_state[0]),
                        2e-6f, 2e-5f);
    ds4_gpu_tensor_free(state_t);

    const uint32_t capture_slots = GDN_FIX_TOKENS - 1u;
    const size_t conv_state_count =
        (size_t)GDN_FIX_CONV_DIM * GDN_FIX_CONV_WIDTH;
    float *capture_seq = malloc(
        (size_t)capture_slots * conv_state_count * sizeof(float));
    float prefix_state[GDN_FIX_CONV_DIM * GDN_FIX_CONV_WIDTH];
    ds4_gpu_tensor *mixed_t = tensor_from(mixed, sizeof(mixed));
    ds4_gpu_tensor *raw_decay_t = tensor_from(raw_decay, sizeof(raw_decay));
    ds4_gpu_tensor *raw_beta_t = tensor_from(raw_beta, sizeof(raw_beta));
    ds4_gpu_tensor *mask_t = tensor_from(mask, sizeof(mask));
    ds4_gpu_tensor *capture_state_t = tensor_from(
        initial_state, sizeof(initial_state));
    ds4_gpu_tensor *capture_seq_t = ds4_gpu_tensor_alloc(
        (size_t)capture_slots * sizeof(initial_state));
    ds4_gpu_tensor *capture_q_t = ds4_gpu_tensor_alloc(sizeof(actual_q));
    ds4_gpu_tensor *capture_k_t = ds4_gpu_tensor_alloc(sizeof(actual_k));
    ds4_gpu_tensor *capture_v_t = ds4_gpu_tensor_alloc(sizeof(actual_v));
    ds4_gpu_tensor *capture_decay_t =
        ds4_gpu_tensor_alloc(sizeof(actual_decay));
    ds4_gpu_tensor *capture_beta_t =
        ds4_gpu_tensor_alloc(sizeof(actual_beta));
    require_ok(capture_seq && mixed_t && raw_decay_t && raw_beta_t && mask_t &&
                   capture_state_t && capture_seq_t && capture_q_t &&
                   capture_k_t && capture_v_t && capture_decay_t &&
                   capture_beta_t,
               "GDN convolution capture allocation");
    require_ok(ds4_gpu_qwen4_gdn_prepare_capture_model(
                   capture_q_t, capture_k_t, capture_v_t,
                   capture_decay_t, capture_beta_t, capture_state_t,
                   capture_seq_t, mixed_t, raw_decay_t, raw_beta_t, mask_t,
                   model, model_size, GDN_FIX_CONV_OFFSET,
                   GDN_FIX_A_LOG_OFFSET, GDN_FIX_DT_BIAS_OFFSET,
                   GDN_FIX_TOKENS, GDN_FIX_KH, GDN_FIX_VH, GDN_FIX_DIM,
                   GDN_FIX_CONV_WIDTH, capture_slots),
               "GDN convolution verifier capture dispatch");
    require_ok(ds4_gpu_tensor_read(
                   capture_seq_t, 0, capture_seq,
                   (size_t)capture_slots * sizeof(initial_state)),
               "GDN convolution verifier capture readback");
    for (uint32_t accepted = 1u; accepted <= capture_slots; accepted++) {
        ds4_gpu_tensor *prefix_state_t = tensor_from(
            initial_state, sizeof(initial_state));
        run_gdn_prepare_fixture(
            model, model_size, accepted, mixed, raw_decay, raw_beta, mask,
            prefix_state_t, actual_q, actual_k, actual_v,
            actual_decay, actual_beta);
        require_ok(ds4_gpu_tensor_read(
                       prefix_state_t, 0, prefix_state,
                       sizeof(prefix_state)),
                   "GDN convolution accepted-prefix state readback");
        require_ok(memcmp(
                       prefix_state,
                       capture_seq +
                           (size_t)(accepted - 1u) * conv_state_count,
                       sizeof(prefix_state)) == 0,
                   "GDN convolution captured state byte equality");
        ds4_gpu_tensor_free(prefix_state_t);
    }
    puts("Qwen GDN convolution capture/replay byte equality PASS");
    ds4_gpu_tensor_free(capture_beta_t);
    ds4_gpu_tensor_free(capture_decay_t);
    ds4_gpu_tensor_free(capture_v_t);
    ds4_gpu_tensor_free(capture_k_t);
    ds4_gpu_tensor_free(capture_q_t);
    ds4_gpu_tensor_free(capture_seq_t);
    ds4_gpu_tensor_free(capture_state_t);
    ds4_gpu_tensor_free(mask_t);
    ds4_gpu_tensor_free(raw_beta_t);
    ds4_gpu_tensor_free(raw_decay_t);
    ds4_gpu_tensor_free(mixed_t);
    free(capture_seq);

    float core[GDN_FIX_TOKENS * GDN_FIX_VALUE_DIM];
    float z[GDN_FIX_TOKENS * GDN_FIX_VALUE_DIM];
    float norm_expected[GDN_FIX_TOKENS * GDN_FIX_VALUE_DIM];
    float norm_actual[GDN_FIX_TOKENS * GDN_FIX_VALUE_DIM];
    for (size_t i = 0; i < sizeof(core) / sizeof(core[0]); i++) {
        core[i] = 0.0175f * (float)((int)(i % 37u) - 18);
        z[i] = 0.035f * (float)((int)((i * 5u) % 23u) - 11);
    }
    const uint32_t rows = GDN_FIX_TOKENS * GDN_FIX_VH;
    for (uint32_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (uint32_t dim = 0; dim < GDN_FIX_DIM; dim++) {
            const float value = core[(size_t)row * GDN_FIX_DIM + dim];
            sum = fmaf(value, value, sum);
        }
        const float scale = 1.0f /
            sqrtf(sum / (float)GDN_FIX_DIM + 1.0e-6f);
        for (uint32_t dim = 0; dim < GDN_FIX_DIM; dim++) {
            const size_t at = (size_t)row * GDN_FIX_DIM + dim;
            const float gate = 1.0f / (1.0f + expf(-z[at]));
            norm_expected[at] = core[at] * scale *
                bf16_to_f32(norm_weight[dim]) * gate;
        }
    }
    ds4_gpu_tensor *core_t = tensor_from(core, sizeof(core));
    ds4_gpu_tensor *z_t = tensor_from(z, sizeof(z));
    ds4_gpu_tensor *norm_t = ds4_gpu_tensor_alloc(sizeof(norm_actual));
    require_ok(norm_t != NULL, "Gated DeltaNet output norm allocation");
    require_ok(ds4_gpu_qwen4_gdn_output_norm_model(
                   norm_t, core_t, z_t, model, model_size,
                   GDN_FIX_NORM_OFFSET, GDN_FIX_TOKENS, GDN_FIX_VH,
                   GDN_FIX_DIM, 1.0e-6f),
               "Gated DeltaNet output norm dispatch");
    require_ok(ds4_gpu_tensor_read(norm_t, 0, norm_actual,
                                   sizeof(norm_actual)),
               "Gated DeltaNet output norm readback");
    require_array_close("Qwen GDN sigmoid-gated RMS output", norm_actual,
                        norm_expected,
                        sizeof(norm_actual) / sizeof(norm_actual[0]),
                        2e-6f, 2e-5f);
    ds4_gpu_tensor_free(norm_t);
    ds4_gpu_tensor_free(z_t);
    ds4_gpu_tensor_free(core_t);
}

static void test_gdn_prepare_long_split(void) {
    enum { TOKENS = 4096, SPLIT = 2048 };
    void *model = model_fixture_allocation;
    const size_t model_size = model_fixture_size;
    require_ok(model != NULL, "long Gated DeltaNet model fixture");
    uint16_t *conv_weight = (uint16_t *)((uint8_t *)model +
                                         GDN_FIX_CONV_OFFSET);
    uint16_t *a_log = (uint16_t *)((uint8_t *)model +
                                   GDN_FIX_A_LOG_OFFSET);
    uint16_t *dt_bias = (uint16_t *)((uint8_t *)model +
                                     GDN_FIX_DT_BIAS_OFFSET);
    for (uint32_t channel = 0; channel < GDN_FIX_CONV_DIM; channel++)
        for (uint32_t tap = 0; tap < GDN_FIX_CONV_WIDTH; tap++)
            conv_weight[channel * GDN_FIX_CONV_WIDTH + tap] =
                f32_to_bf16(0.02f * (float)(tap + 1u) +
                            0.001f * (float)((int)(channel % 11u) - 5));
    for (uint32_t head = 0; head < GDN_FIX_VH; head++) {
        a_log[head] = f32_to_bf16(logf(0.3f + 0.1f * (float)head));
        dt_bias[head] = f32_to_bf16(-0.25f + 0.15f * (float)head);
    }

    const size_t mixed_count = (size_t)TOKENS * GDN_FIX_CONV_DIM;
    const size_t qk_count = (size_t)TOKENS * GDN_FIX_KEY_DIM;
    const size_t value_count = (size_t)TOKENS * GDN_FIX_VALUE_DIM;
    const size_t gate_count = (size_t)TOKENS * GDN_FIX_VH;
    const size_t state_count =
        (size_t)GDN_FIX_CONV_DIM * GDN_FIX_CONV_WIDTH;
    float *mixed = malloc(mixed_count * sizeof(float));
    float *raw_decay = malloc(gate_count * sizeof(float));
    float *raw_beta = malloc(gate_count * sizeof(float));
    float *full_q = malloc(qk_count * sizeof(float));
    float *full_k = malloc(qk_count * sizeof(float));
    float *full_v = malloc(value_count * sizeof(float));
    float *full_decay = malloc(gate_count * sizeof(float));
    float *full_beta = malloc(gate_count * sizeof(float));
    float *split_q = malloc(qk_count * sizeof(float));
    float *split_k = malloc(qk_count * sizeof(float));
    float *split_v = malloc(value_count * sizeof(float));
    float *split_decay = malloc(gate_count * sizeof(float));
    float *split_beta = malloc(gate_count * sizeof(float));
    float *initial_state = malloc(state_count * sizeof(float));
    float *full_state = malloc(state_count * sizeof(float));
    float *split_state = malloc(state_count * sizeof(float));
    require_ok(mixed && raw_decay && raw_beta && full_q && full_k && full_v &&
                   full_decay && full_beta && split_q && split_k && split_v &&
                   split_decay && split_beta && initial_state && full_state &&
                   split_state,
               "long Gated DeltaNet host allocation");
    for (size_t i = 0; i < mixed_count; i++)
        mixed[i] = 0.0025f * (float)((int)((i * 13u + 7u) % 101u) - 50);
    for (size_t i = 0; i < gate_count; i++) {
        raw_decay[i] = 0.01f * (float)((int)(i % 29u) - 14);
        raw_beta[i] = 0.015f * (float)((int)((i * 3u) % 31u) - 15);
    }
    for (size_t i = 0; i < state_count; i++)
        initial_state[i] = 0.001f * (float)((int)(i % 37u) - 18);

    ds4_gpu_tensor *state_t = tensor_from(
        initial_state, state_count * sizeof(float));
    run_gdn_prepare_fixture(model, model_size, TOKENS, mixed, raw_decay,
                            raw_beta, NULL, state_t, full_q, full_k, full_v,
                            full_decay, full_beta);
    require_ok(ds4_gpu_tensor_read(state_t, 0, full_state,
                                   state_count * sizeof(float)),
               "long Gated DeltaNet full state readback");
    ds4_gpu_tensor_free(state_t);

    state_t = tensor_from(initial_state, state_count * sizeof(float));
    for (uint32_t start = 0; start < TOKENS; start += SPLIT) {
        run_gdn_prepare_fixture(
            model, model_size, SPLIT,
            mixed + (size_t)start * GDN_FIX_CONV_DIM,
            raw_decay + (size_t)start * GDN_FIX_VH,
            raw_beta + (size_t)start * GDN_FIX_VH, NULL, state_t,
            split_q + (size_t)start * GDN_FIX_KEY_DIM,
            split_k + (size_t)start * GDN_FIX_KEY_DIM,
            split_v + (size_t)start * GDN_FIX_VALUE_DIM,
            split_decay + (size_t)start * GDN_FIX_VH,
            split_beta + (size_t)start * GDN_FIX_VH);
    }
    require_ok(ds4_gpu_tensor_read(state_t, 0, split_state,
                                   state_count * sizeof(float)),
               "long Gated DeltaNet split state readback");
    require_ok(memcmp(full_q, split_q, qk_count * sizeof(float)) == 0,
               "long Gated DeltaNet split Q exactness");
    require_ok(memcmp(full_k, split_k, qk_count * sizeof(float)) == 0,
               "long Gated DeltaNet split K exactness");
    require_ok(memcmp(full_v, split_v, value_count * sizeof(float)) == 0,
               "long Gated DeltaNet split V exactness");
    require_ok(memcmp(full_decay, split_decay,
                      gate_count * sizeof(float)) == 0,
               "long Gated DeltaNet split decay exactness");
    require_ok(memcmp(full_beta, split_beta,
                      gate_count * sizeof(float)) == 0,
               "long Gated DeltaNet split beta exactness");
    require_ok(memcmp(full_state, split_state,
                      state_count * sizeof(float)) == 0,
               "long Gated DeltaNet split convolution state exactness");
    printf("Qwen GDN preparation 4096 vs 2x2048 exact PASS\n");
    ds4_gpu_tensor_free(state_t);

    free(split_state);
    free(full_state);
    free(initial_state);
    free(split_beta);
    free(split_decay);
    free(split_v);
    free(split_k);
    free(split_q);
    free(full_beta);
    free(full_decay);
    free(full_v);
    free(full_k);
    free(full_q);
    free(raw_beta);
    free(raw_decay);
    free(mixed);
}

static void gdn_reference(float *out, float *state, const float *q,
                          const float *k, const float *v,
                          const float *decay, const float *beta,
                          const uint8_t *mask, uint32_t tokens,
                          uint32_t key_heads, uint32_t value_heads,
                          uint32_t dim) {
    const uint32_t repeat = value_heads / key_heads;
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t hv = 0; hv < value_heads; hv++) {
            const uint32_t hk = hv / repeat;
            for (uint32_t dv = 0; dv < dim; dv++) {
                float *row = state + ((size_t)hv * dim + dv) * dim;
                float *yo = out + ((size_t)t * value_heads + hv) * dim + dv;
                if (mask && mask[t] == 0u) {
                    *yo = 0.0f;
                    continue;
                }
                float kv = 0.0f;
                for (uint32_t dk = 0; dk < dim; dk++) {
                    row[dk] *= decay[(size_t)t * value_heads + hv];
                    kv = fmaf(row[dk],
                              k[((size_t)t * key_heads + hk) * dim + dk], kv);
                }
                const float delta =
                    (v[((size_t)t * value_heads + hv) * dim + dv] - kv) *
                    beta[(size_t)t * value_heads + hv];
                float y = 0.0f;
                for (uint32_t dk = 0; dk < dim; dk++) {
                    row[dk] = fmaf(
                        k[((size_t)t * key_heads + hk) * dim + dk], delta,
                        row[dk]);
                    y = fmaf(row[dk],
                             q[((size_t)t * key_heads + hk) * dim + dk], y);
                }
                *yo = y;
            }
        }
    }
}

static void run_gdn(uint32_t rows, uint32_t tokens,
                    const float *q, const float *k, const float *v,
                    const float *decay, const float *beta,
                    const uint8_t *mask, const float *initial_state,
                    float *out, float *state_out) {
    enum { DIM = 128, KH = 1, VH = 3 };
    const size_t qk_bytes = (size_t)tokens * KH * DIM * sizeof(float);
    const size_t value_bytes = (size_t)tokens * VH * DIM * sizeof(float);
    const size_t gate_bytes = (size_t)tokens * VH * sizeof(float);
    const size_t state_bytes = (size_t)VH * DIM * DIM * sizeof(float);
    ds4_gpu_tensor *qt = tensor_from(q, qk_bytes);
    ds4_gpu_tensor *kt = tensor_from(k, qk_bytes);
    ds4_gpu_tensor *vt = tensor_from(v, value_bytes);
    ds4_gpu_tensor *dt = tensor_from(decay, gate_bytes);
    ds4_gpu_tensor *bt = tensor_from(beta, gate_bytes);
    ds4_gpu_tensor *mt = tensor_from(mask, tokens);
    ds4_gpu_tensor *st = tensor_from(initial_state, state_bytes);
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc(value_bytes);
    require_ok(ot != NULL, "Gated DeltaNet output allocation");
    require_ok(ds4_gpu_qwen4_gdn_prefill(
                   ot, st, qt, kt, vt, dt, bt, mt,
                   tokens, KH, VH, DIM, rows),
               "Gated DeltaNet dispatch");
    require_ok(ds4_gpu_tensor_read(ot, 0, out, value_bytes),
               "Gated DeltaNet output readback");
    require_ok(ds4_gpu_tensor_read(st, 0, state_out, state_bytes),
               "Gated DeltaNet state readback");
    ds4_gpu_tensor_free(ot);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(mt);
    ds4_gpu_tensor_free(bt);
    ds4_gpu_tensor_free(dt);
    ds4_gpu_tensor_free(vt);
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(qt);
}

static void test_gdn_bf16_state_decode(void) {
    enum { TOKENS = 1, DIM = 128, KH = 1, VH = 3 };
    const size_t qk_count = (size_t)TOKENS * KH * DIM;
    const size_t value_count = (size_t)TOKENS * VH * DIM;
    const size_t gate_count = (size_t)TOKENS * VH;
    const size_t state_count = (size_t)VH * DIM * DIM;
    float *q = malloc(qk_count * sizeof(float));
    float *k = malloc(qk_count * sizeof(float));
    float *v = malloc(value_count * sizeof(float));
    float *decay = malloc(gate_count * sizeof(float));
    float *beta = malloc(gate_count * sizeof(float));
    uint16_t *initial_bf16 = malloc(state_count * sizeof(uint16_t));
    float *initial_f32 = malloc(state_count * sizeof(float));
    float *float_out = malloc(value_count * sizeof(float));
    float *float_state = malloc(state_count * sizeof(float));
    float *bf16_out = malloc(value_count * sizeof(float));
    uint16_t *bf16_state = malloc(state_count * sizeof(uint16_t));
    require_ok(q && k && v && decay && beta && initial_bf16 && initial_f32 &&
                   float_out && float_state && bf16_out && bf16_state,
               "GDN BF16-state fixture allocation");
    for (size_t i = 0; i < qk_count; i++) {
        q[i] = 0.001f * (float)((int)(i % 31u) - 15);
        k[i] = 0.0015f * (float)((int)(i % 29u) - 14);
    }
    for (size_t i = 0; i < value_count; i++)
        v[i] = 0.01f * (float)((int)(i % 17u) - 8);
    for (size_t i = 0; i < gate_count; i++) {
        decay[i] = 0.995f - 0.0002f * (float)i;
        beta[i] = 0.04f + 0.003f * (float)i;
    }
    for (size_t i = 0; i < state_count; i++) {
        initial_bf16[i] = f32_to_bf16(
            0.0002f * (float)((int)(i % 23u) - 11));
        initial_f32[i] = bf16_to_f32(initial_bf16[i]);
    }
    const uint8_t mask[TOKENS] = {1u};
    run_gdn(4u, TOKENS, q, k, v, decay, beta, mask, initial_f32,
            float_out, float_state);

    ds4_gpu_tensor *qt = tensor_from(q, qk_count * sizeof(float));
    ds4_gpu_tensor *kt = tensor_from(k, qk_count * sizeof(float));
    ds4_gpu_tensor *vt = tensor_from(v, value_count * sizeof(float));
    ds4_gpu_tensor *dt = tensor_from(decay, gate_count * sizeof(float));
    ds4_gpu_tensor *bt = tensor_from(beta, gate_count * sizeof(float));
    ds4_gpu_tensor *mt = tensor_from(mask, sizeof(mask));
    ds4_gpu_tensor *st = tensor_from(
        initial_bf16, state_count * sizeof(uint16_t));
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc(
        value_count * sizeof(float));
    require_ok(qt && kt && vt && dt && bt && mt && st && ot,
               "GDN BF16-state device allocation");
    require_ok(ds4_gpu_qwen4_gdn_prefill_bf16_state(
                   ot, st, qt, kt, vt, dt, bt, mt, TOKENS, KH, VH, DIM, 4u),
               "GDN BF16-state dispatch");
    require_ok(ds4_gpu_tensor_read(
                   ot, 0, bf16_out, value_count * sizeof(float)) &&
                   ds4_gpu_tensor_read(
                       st, 0, bf16_state,
                       state_count * sizeof(uint16_t)),
               "GDN BF16-state readback");
    require_ok(memcmp(float_out, bf16_out,
                      value_count * sizeof(float)) == 0,
               "GDN BF16-state FP32 output exactness");
    for (size_t i = 0; i < state_count; i++) {
        require_ok(bf16_state[i] == f32_to_bf16(float_state[i]),
                   "GDN BF16-state boundary rounding");
    }
    puts("Qwen GDN BF16-state decode output/rounding exact PASS");

    ds4_gpu_tensor_free(ot);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(mt);
    ds4_gpu_tensor_free(bt);
    ds4_gpu_tensor_free(dt);
    ds4_gpu_tensor_free(vt);
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(qt);
    free(bf16_state);
    free(bf16_out);
    free(float_state);
    free(float_out);
    free(initial_f32);
    free(initial_bf16);
    free(beta);
    free(decay);
    free(v);
    free(k);
    free(q);
}

static void test_gdn_bf16_capture_exact(void) {
    enum { TOKENS = 5, DIM = 128, KH = 1, VH = 3, SLOTS = TOKENS - 1 };
    const size_t qk_count = (size_t)TOKENS * KH * DIM;
    const size_t value_count = (size_t)TOKENS * VH * DIM;
    const size_t gate_count = (size_t)TOKENS * VH;
    const size_t state_count = (size_t)VH * DIM * DIM;
    float *q = malloc(qk_count * sizeof(float));
    float *k = malloc(qk_count * sizeof(float));
    float *v = malloc(value_count * sizeof(float));
    float *decay = malloc(gate_count * sizeof(float));
    float *beta = malloc(gate_count * sizeof(float));
    uint16_t *initial = malloc(state_count * sizeof(uint16_t));
    uint16_t *capture = malloc(
        (size_t)SLOTS * state_count * sizeof(uint16_t));
    uint16_t *prefix_state = malloc(state_count * sizeof(uint16_t));
    require_ok(q && k && v && decay && beta && initial && capture &&
                   prefix_state,
               "GDN BF16 capture host allocation");
    for (size_t i = 0; i < qk_count; i++) {
        q[i] = 0.0017f * (float)((int)(i % 31u) - 15);
        k[i] = 0.0013f * (float)((int)(i % 29u) - 14);
    }
    for (size_t i = 0; i < value_count; i++)
        v[i] = 0.009f * (float)((int)(i % 19u) - 9);
    for (size_t i = 0; i < gate_count; i++) {
        decay[i] = 0.994f - 0.00015f * (float)(i % VH);
        beta[i] = 0.035f + 0.004f * (float)(i % 5u);
    }
    for (size_t i = 0; i < state_count; i++)
        initial[i] = f32_to_bf16(
            0.00022f * (float)((int)(i % 23u) - 11));

    ds4_gpu_tensor *qt = tensor_from(q, qk_count * sizeof(float));
    ds4_gpu_tensor *kt = tensor_from(k, qk_count * sizeof(float));
    ds4_gpu_tensor *vt = tensor_from(v, value_count * sizeof(float));
    ds4_gpu_tensor *dt = tensor_from(decay, gate_count * sizeof(float));
    ds4_gpu_tensor *bt = tensor_from(beta, gate_count * sizeof(float));
    ds4_gpu_tensor *st = tensor_from(
        initial, state_count * sizeof(uint16_t));
    ds4_gpu_tensor *seq = ds4_gpu_tensor_alloc(
        (size_t)SLOTS * state_count * sizeof(uint16_t));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(value_count * sizeof(float));
    require_ok(qt && kt && vt && dt && bt && st && seq && out,
               "GDN BF16 capture device allocation");
    require_ok(ds4_gpu_qwen4_gdn_prefill_capture_bf16_state(
                   out, st, seq, qt, kt, vt, dt, bt, NULL,
                   TOKENS, KH, VH, DIM, 4u, SLOTS),
               "GDN BF16 capture dispatch");
    require_ok(ds4_gpu_tensor_read(
                   seq, 0, capture,
                   (size_t)SLOTS * state_count * sizeof(uint16_t)),
               "GDN BF16 capture readback");
    for (uint32_t accepted = 1u; accepted <= SLOTS; accepted++) {
        ds4_gpu_tensor *prefix_st = tensor_from(
            initial, state_count * sizeof(uint16_t));
        ds4_gpu_tensor *prefix_out = ds4_gpu_tensor_alloc(
            (size_t)accepted * VH * DIM * sizeof(float));
        require_ok(prefix_st && prefix_out,
                   "GDN BF16 captured-prefix allocation");
        require_ok(ds4_gpu_qwen4_gdn_prefill_bf16_state(
                       prefix_out, prefix_st, qt, kt, vt, dt, bt, NULL,
                       accepted, KH, VH, DIM, 4u),
                   "GDN BF16 captured-prefix dispatch");
        require_ok(ds4_gpu_tensor_read(
                       prefix_st, 0, prefix_state,
                       state_count * sizeof(uint16_t)),
                   "GDN BF16 captured-prefix readback");
        require_ok(memcmp(
                       prefix_state,
                       capture + (size_t)(accepted - 1u) * state_count,
                       state_count * sizeof(uint16_t)) == 0,
                   "GDN BF16 captured state byte equality");
        ds4_gpu_tensor_free(prefix_out);
        ds4_gpu_tensor_free(prefix_st);
    }
    puts("Qwen GDN BF16 recurrent capture/replay byte equality PASS");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(seq);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(bt);
    ds4_gpu_tensor_free(dt);
    ds4_gpu_tensor_free(vt);
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(qt);
    free(prefix_state);
    free(capture);
    free(initial);
    free(beta);
    free(decay);
    free(v);
    free(k);
    free(q);
}

static void test_gdn_fused_bf16_state_decode(void) {
    enum { TOKENS = 1, DIM = 128, KH = 1, VH = 3, CONV_WIDTH = 4 };
    enum {
        KEY_DIM = KH * DIM,
        VALUE_DIM = VH * DIM,
        CONV_DIM = 2 * KEY_DIM + VALUE_DIM,
    };
    const size_t state_count = (size_t)VH * DIM * DIM;
    float mixed[CONV_DIM];
    float raw_decay[VH];
    float raw_beta[VH];
    float raw_gate[VALUE_DIM];
    float initial_conv[CONV_DIM * CONV_WIDTH];
    uint16_t *initial_state = malloc(state_count * sizeof(uint16_t));
    uint16_t *expected_state = malloc(state_count * sizeof(uint16_t));
    uint16_t *fused_state = malloc(state_count * sizeof(uint16_t));
    float expected_out[VALUE_DIM];
    float fused_out[VALUE_DIM];
    float expected_normalized[VALUE_DIM];
    float full_fused_out[VALUE_DIM];
    float expected_conv[CONV_DIM * CONV_WIDTH];
    float fused_conv[CONV_DIM * CONV_WIDTH];
    require_ok(initial_state && expected_state && fused_state,
               "fused GDN BF16-state host allocation");
    for (size_t i = 0; i < CONV_DIM; i++)
        mixed[i] = 0.0125f * (float)((int)((i * 11u + 5u) % 37u) - 18);
    for (size_t i = 0; i < VH; i++) {
        raw_decay[i] = -0.12f + 0.09f * (float)i;
        raw_beta[i] = 0.17f - 0.08f * (float)i;
    }
    for (size_t i = 0; i < VALUE_DIM; i++)
        raw_gate[i] = 0.025f * (float)((int)(i % 23u) - 11);
    for (size_t i = 0; i < CONV_DIM * CONV_WIDTH; i++)
        initial_conv[i] =
            0.004f * (float)((int)((i * 3u + 1u) % 19u) - 9);
    for (size_t i = 0; i < state_count; i++)
        initial_state[i] = f32_to_bf16(
            0.0003f * (float)((int)((i * 5u + 2u) % 29u) - 14));

    ds4_gpu_tensor *mixed_t = tensor_from(mixed, sizeof(mixed));
    ds4_gpu_tensor *raw_decay_t = tensor_from(raw_decay, sizeof(raw_decay));
    ds4_gpu_tensor *raw_beta_t = tensor_from(raw_beta, sizeof(raw_beta));
    ds4_gpu_tensor *raw_gate_t = tensor_from(raw_gate, sizeof(raw_gate));
    ds4_gpu_tensor *expected_conv_t = tensor_from(
        initial_conv, sizeof(initial_conv));
    ds4_gpu_tensor *expected_state_t = tensor_from(
        initial_state, state_count * sizeof(uint16_t));
    ds4_gpu_tensor *expected_q_t = ds4_gpu_tensor_alloc(
        KEY_DIM * sizeof(float));
    ds4_gpu_tensor *expected_k_t = ds4_gpu_tensor_alloc(
        KEY_DIM * sizeof(float));
    ds4_gpu_tensor *expected_v_t = ds4_gpu_tensor_alloc(
        VALUE_DIM * sizeof(float));
    ds4_gpu_tensor *expected_decay_t = ds4_gpu_tensor_alloc(
        VH * sizeof(float));
    ds4_gpu_tensor *expected_beta_t = ds4_gpu_tensor_alloc(
        VH * sizeof(float));
    ds4_gpu_tensor *expected_out_t = ds4_gpu_tensor_alloc(
        VALUE_DIM * sizeof(float));
    require_ok(mixed_t && raw_decay_t && raw_beta_t && raw_gate_t &&
                   expected_conv_t &&
                   expected_state_t && expected_q_t && expected_k_t &&
                   expected_v_t && expected_decay_t && expected_beta_t &&
                   expected_out_t,
               "reference fused GDN BF16-state device allocation");
    require_ok(ds4_gpu_qwen4_gdn_prepare_model(
                   expected_q_t, expected_k_t, expected_v_t,
                   expected_decay_t, expected_beta_t, expected_conv_t,
                   mixed_t, raw_decay_t, raw_beta_t, NULL,
                   model_fixture_allocation, model_fixture_size,
                   GDN_FIX_CONV_OFFSET, GDN_FIX_A_LOG_OFFSET,
                   GDN_FIX_DT_BIAS_OFFSET, TOKENS, KH, VH, DIM, CONV_WIDTH),
               "reference fused GDN preparation dispatch");
    require_ok(ds4_gpu_qwen4_gdn_prefill_bf16_state(
                   expected_out_t, expected_state_t,
                   expected_q_t, expected_k_t, expected_v_t,
                   expected_decay_t, expected_beta_t, NULL,
                   TOKENS, KH, VH, DIM, 4u),
               "reference fused GDN recurrence dispatch");
    require_ok(ds4_gpu_tensor_read(
                   expected_out_t, 0, expected_out, sizeof(expected_out)) &&
                   ds4_gpu_tensor_read(
                       expected_state_t, 0, expected_state,
                       state_count * sizeof(uint16_t)) &&
                   ds4_gpu_tensor_read(
                       expected_conv_t, 0, expected_conv,
                       sizeof(expected_conv)),
               "reference fused GDN readback");

    ds4_gpu_tensor *fused_conv_t = tensor_from(
        initial_conv, sizeof(initial_conv));
    ds4_gpu_tensor *fused_state_t = tensor_from(
        initial_state, state_count * sizeof(uint16_t));
    ds4_gpu_tensor *fused_q_t = ds4_gpu_tensor_alloc(KEY_DIM * sizeof(float));
    ds4_gpu_tensor *fused_k_t = ds4_gpu_tensor_alloc(KEY_DIM * sizeof(float));
    ds4_gpu_tensor *fused_v_t = ds4_gpu_tensor_alloc(VALUE_DIM * sizeof(float));
    ds4_gpu_tensor *fused_out_t = ds4_gpu_tensor_alloc(
        VALUE_DIM * sizeof(float));
    require_ok(fused_conv_t && fused_state_t && fused_q_t && fused_k_t &&
                   fused_v_t && fused_out_t,
               "fused GDN BF16-state device allocation");
    require_ok(ds4_gpu_qwen4_gdn_decode_bf16_state_model(
                   fused_out_t, fused_state_t,
                   fused_q_t, fused_k_t, fused_v_t, fused_conv_t,
                   mixed_t, raw_decay_t, raw_beta_t,
                   model_fixture_allocation, model_fixture_size,
                   GDN_FIX_CONV_OFFSET, GDN_FIX_A_LOG_OFFSET,
                   GDN_FIX_DT_BIAS_OFFSET, KH, VH, DIM, CONV_WIDTH),
               "fused GDN BF16-state dispatch");
    require_ok(ds4_gpu_tensor_read(
                   fused_out_t, 0, fused_out, sizeof(fused_out)) &&
                   ds4_gpu_tensor_read(
                       fused_state_t, 0, fused_state,
                       state_count * sizeof(uint16_t)) &&
                   ds4_gpu_tensor_read(
                       fused_conv_t, 0, fused_conv, sizeof(fused_conv)),
               "fused GDN BF16-state readback");
    require_array_close("Qwen fused GDN BF16-state output", fused_out,
                        expected_out, VALUE_DIM, 2e-6f, 2e-5f);
    require_ok(memcmp(fused_state, expected_state,
                      state_count * sizeof(uint16_t)) == 0,
               "fused GDN BF16-state boundary equality");
    require_ok(memcmp(fused_conv, expected_conv, sizeof(fused_conv)) == 0,
               "fused GDN convolution state equality");
    puts("Qwen fused GDN BF16-state decode parity PASS");

    ds4_gpu_tensor *expected_normalized_t = ds4_gpu_tensor_alloc(
        VALUE_DIM * sizeof(float));
    require_ok(expected_normalized_t &&
                   ds4_gpu_qwen4_gdn_output_norm_model(
                       expected_normalized_t, expected_out_t, raw_gate_t,
                       model_fixture_allocation, model_fixture_size,
                       GDN_FIX_NORM_OFFSET, TOKENS, VH, DIM, 1.0e-6f),
               "reference full-fusion GDN output norm");
    require_ok(ds4_gpu_tensor_read(
                   expected_normalized_t, 0, expected_normalized,
                   sizeof(expected_normalized)),
               "reference full-fusion GDN output readback");

    ds4_gpu_tensor *full_conv_t = tensor_from(
        initial_conv, sizeof(initial_conv));
    ds4_gpu_tensor *full_state_t = tensor_from(
        initial_state, state_count * sizeof(uint16_t));
    ds4_gpu_tensor *full_q_t = ds4_gpu_tensor_alloc(KEY_DIM * sizeof(float));
    ds4_gpu_tensor *full_k_t = ds4_gpu_tensor_alloc(KEY_DIM * sizeof(float));
    ds4_gpu_tensor *full_v_t = ds4_gpu_tensor_alloc(VALUE_DIM * sizeof(float));
    ds4_gpu_tensor *full_out_t = ds4_gpu_tensor_alloc(
        VALUE_DIM * sizeof(float));
    require_ok(full_conv_t && full_state_t && full_q_t && full_k_t &&
                   full_v_t && full_out_t,
               "fully fused GDN BF16-state device allocation");
    require_ok(ds4_gpu_qwen4_gdn_decode_output_bf16_state_model(
                   full_out_t, full_state_t,
                   full_q_t, full_k_t, full_v_t, full_conv_t,
                   mixed_t, raw_decay_t, raw_beta_t, raw_gate_t,
                   model_fixture_allocation, model_fixture_size,
                   GDN_FIX_CONV_OFFSET, GDN_FIX_A_LOG_OFFSET,
                   GDN_FIX_DT_BIAS_OFFSET, GDN_FIX_NORM_OFFSET,
                   KH, VH, DIM, CONV_WIDTH),
               "fully fused GDN BF16-state dispatch");
    require_ok(ds4_gpu_tensor_read(
                   full_out_t, 0, full_fused_out, sizeof(full_fused_out)) &&
                   ds4_gpu_tensor_read(
                       full_state_t, 0, fused_state,
                       state_count * sizeof(uint16_t)) &&
                   ds4_gpu_tensor_read(
                       full_conv_t, 0, fused_conv, sizeof(fused_conv)),
               "fully fused GDN BF16-state readback");
    require_array_close("Qwen fully fused GDN output", full_fused_out,
                        expected_normalized, VALUE_DIM, 2e-6f, 2e-5f);
    require_ok(memcmp(fused_state, expected_state,
                      state_count * sizeof(uint16_t)) == 0,
               "fully fused GDN BF16-state boundary equality");
    require_ok(memcmp(fused_conv, expected_conv, sizeof(fused_conv)) == 0,
               "fully fused GDN convolution state equality");
    puts("Qwen fully fused GDN BF16-state decode parity PASS");

    ds4_gpu_tensor_free(full_out_t);
    ds4_gpu_tensor_free(full_v_t);
    ds4_gpu_tensor_free(full_k_t);
    ds4_gpu_tensor_free(full_q_t);
    ds4_gpu_tensor_free(full_state_t);
    ds4_gpu_tensor_free(full_conv_t);
    ds4_gpu_tensor_free(expected_normalized_t);

    ds4_gpu_tensor_free(fused_out_t);
    ds4_gpu_tensor_free(fused_v_t);
    ds4_gpu_tensor_free(fused_k_t);
    ds4_gpu_tensor_free(fused_q_t);
    ds4_gpu_tensor_free(fused_state_t);
    ds4_gpu_tensor_free(fused_conv_t);
    ds4_gpu_tensor_free(expected_out_t);
    ds4_gpu_tensor_free(expected_beta_t);
    ds4_gpu_tensor_free(expected_decay_t);
    ds4_gpu_tensor_free(expected_v_t);
    ds4_gpu_tensor_free(expected_k_t);
    ds4_gpu_tensor_free(expected_q_t);
    ds4_gpu_tensor_free(expected_state_t);
    ds4_gpu_tensor_free(expected_conv_t);
    ds4_gpu_tensor_free(raw_beta_t);
    ds4_gpu_tensor_free(raw_decay_t);
    ds4_gpu_tensor_free(mixed_t);
    ds4_gpu_tensor_free(raw_gate_t);
    free(fused_state);
    free(expected_state);
    free(initial_state);
}

static void test_gdn_capture_exact(void) {
    enum { TOKENS = 5, DIM = 128, KH = 1, VH = 3, SLOTS = TOKENS - 1 };
    const size_t qk_count = (size_t)TOKENS * KH * DIM;
    const size_t value_count = (size_t)TOKENS * VH * DIM;
    const size_t gate_count = (size_t)TOKENS * VH;
    const size_t state_count = (size_t)VH * DIM * DIM;
    float *q = malloc(qk_count * sizeof(float));
    float *k = malloc(qk_count * sizeof(float));
    float *v = malloc(value_count * sizeof(float));
    float *decay = malloc(gate_count * sizeof(float));
    float *beta = malloc(gate_count * sizeof(float));
    uint8_t mask[TOKENS] = {1u, 0u, 1u, 1u, 1u};
    float *initial = malloc(state_count * sizeof(float));
    float *capture = malloc((size_t)SLOTS * state_count * sizeof(float));
    float *prefix_state = malloc(state_count * sizeof(float));
    float *prefix_out = malloc(value_count * sizeof(float));
    require_ok(q && k && v && decay && beta && initial && capture &&
                   prefix_state && prefix_out,
               "GDN recurrence capture host allocation");
    for (size_t i = 0; i < qk_count; i++) {
        q[i] = 0.002f * (float)((int)(i % 29u) - 14);
        k[i] = 0.0015f * (float)((int)(i % 31u) - 15);
    }
    for (size_t i = 0; i < value_count; i++)
        v[i] = 0.01f * (float)((int)(i % 19u) - 9);
    for (size_t i = 0; i < gate_count; i++) {
        decay[i] = 0.996f - 0.0002f * (float)(i % VH);
        beta[i] = 0.04f + 0.005f * (float)(i % 5u);
    }
    for (size_t i = 0; i < state_count; i++)
        initial[i] = 0.00025f * (float)((int)(i % 17u) - 8);

    ds4_gpu_tensor *qt = tensor_from(q, qk_count * sizeof(float));
    ds4_gpu_tensor *kt = tensor_from(k, qk_count * sizeof(float));
    ds4_gpu_tensor *vt = tensor_from(v, value_count * sizeof(float));
    ds4_gpu_tensor *dt = tensor_from(decay, gate_count * sizeof(float));
    ds4_gpu_tensor *bt = tensor_from(beta, gate_count * sizeof(float));
    ds4_gpu_tensor *mt = tensor_from(mask, sizeof(mask));
    ds4_gpu_tensor *st = tensor_from(initial, state_count * sizeof(float));
    ds4_gpu_tensor *seq = ds4_gpu_tensor_alloc(
        (size_t)SLOTS * state_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(value_count * sizeof(float));
    require_ok(qt && kt && vt && dt && bt && mt && st && seq && out,
               "GDN recurrence capture device allocation");
    require_ok(ds4_gpu_qwen4_gdn_prefill_capture(
                   out, st, seq, qt, kt, vt, dt, bt, mt,
                   TOKENS, KH, VH, DIM, 4u, SLOTS),
               "GDN recurrence verifier capture dispatch");
    require_ok(ds4_gpu_tensor_read(
                   seq, 0, capture,
                   (size_t)SLOTS * state_count * sizeof(float)),
               "GDN recurrence verifier capture readback");
    for (uint32_t accepted = 1u; accepted <= SLOTS; accepted++) {
        run_gdn(4u, accepted, q, k, v, decay, beta, mask, initial,
                prefix_out, prefix_state);
        require_ok(memcmp(
                       prefix_state,
                       capture + (size_t)(accepted - 1u) * state_count,
                       state_count * sizeof(float)) == 0,
                   "GDN recurrent captured state byte equality");
    }
    puts("Qwen GDN recurrent capture/replay byte equality PASS");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(seq);
    ds4_gpu_tensor_free(st);
    ds4_gpu_tensor_free(mt);
    ds4_gpu_tensor_free(bt);
    ds4_gpu_tensor_free(dt);
    ds4_gpu_tensor_free(vt);
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(qt);
    free(prefix_out);
    free(prefix_state);
    free(capture);
    free(initial);
    free(beta);
    free(decay);
    free(v);
    free(k);
    free(q);
}

static void test_gdn_length(uint32_t tokens) {
    enum { DIM = 128, KH = 1, VH = 3 };
    const size_t qk_count = (size_t)tokens * KH * DIM;
    const size_t value_count = (size_t)tokens * VH * DIM;
    const size_t gate_count = (size_t)tokens * VH;
    const size_t state_count = (size_t)VH * DIM * DIM;
    float *q = malloc(qk_count * sizeof(float));
    float *k = malloc(qk_count * sizeof(float));
    float *v = malloc(value_count * sizeof(float));
    float *decay = malloc(gate_count * sizeof(float));
    float *beta = malloc(gate_count * sizeof(float));
    uint8_t *mask = malloc(tokens);
    float *initial = calloc(state_count, sizeof(float));
    float *out_r1 = malloc(value_count * sizeof(float));
    float *out_r2 = malloc(value_count * sizeof(float));
    float *out_r4 = malloc(value_count * sizeof(float));
    float *state_r1 = malloc(state_count * sizeof(float));
    float *state_r2 = malloc(state_count * sizeof(float));
    float *state_r4 = malloc(state_count * sizeof(float));
    require_ok(q && k && v && decay && beta && mask && initial && out_r1 &&
                   out_r2 && out_r4 && state_r1 && state_r2 && state_r4,
               "Gated DeltaNet host allocation");
    for (size_t i = 0; i < qk_count; i++) {
        q[i] = 0.002f * (float)((int)(i % 29u) - 14);
        k[i] = 0.0015f * (float)((int)(i % 31u) - 15);
    }
    for (size_t i = 0; i < value_count; i++)
        v[i] = 0.01f * (float)((int)(i % 19u) - 9);
    for (size_t i = 0; i < gate_count; i++) {
        decay[i] = 0.996f - 0.0002f * (float)(i % VH);
        beta[i] = 0.04f + 0.005f * (float)(i % 5u);
    }
    for (uint32_t t = 0; t < tokens; t++) mask[t] = (t % 13u) != 5u;

    run_gdn(1, tokens, q, k, v, decay, beta, mask, initial,
            out_r1, state_r1);
    run_gdn(2, tokens, q, k, v, decay, beta, mask, initial,
            out_r2, state_r2);
    run_gdn(4, tokens, q, k, v, decay, beta, mask, initial,
            out_r4, state_r4);
    char label[96];
    snprintf(label, sizeof(label), "Qwen GDN R4/R1 output length=%u", tokens);
    require_array_close(label, out_r4, out_r1, value_count, 1e-6f, 1e-5f);
    snprintf(label, sizeof(label), "Qwen GDN R4/R2 output length=%u", tokens);
    require_array_close(label, out_r4, out_r2, value_count, 1e-6f, 1e-5f);
    snprintf(label, sizeof(label), "Qwen GDN R4/R1 state length=%u", tokens);
    require_array_close(label, state_r4, state_r1, state_count, 1e-6f, 1e-5f);
    snprintf(label, sizeof(label), "Qwen GDN R4/R2 state length=%u", tokens);
    require_array_close(label, state_r4, state_r2, state_count, 1e-6f, 1e-5f);

    if (tokens >= 2048u) {
        const uint32_t split_tokens = 2048u;
        float *out_split = malloc(value_count * sizeof(float));
        float *state_split = malloc(state_count * sizeof(float));
        require_ok(out_split && state_split,
                   "Gated DeltaNet split host allocation");
        memcpy(state_split, initial, state_count * sizeof(float));
        uint32_t start = 0u;
        while (start < tokens) {
            uint32_t count = tokens - start;
            if (count > split_tokens) count = split_tokens;
            run_gdn(4, count,
                    q + (size_t)start * KH * DIM,
                    k + (size_t)start * KH * DIM,
                    v + (size_t)start * VH * DIM,
                    decay + (size_t)start * VH,
                    beta + (size_t)start * VH,
                    mask + start, state_split,
                    out_split + (size_t)start * VH * DIM, state_split);
            start += count;
        }
        snprintf(label, sizeof(label),
                 "Qwen GDN R4 split output length=%u", tokens);
        require_array_close(label, out_split, out_r4, value_count,
                            1e-6f, 1e-5f);
        snprintf(label, sizeof(label),
                 "Qwen GDN R4 split state length=%u", tokens);
        require_array_close(label, state_split, state_r4, state_count,
                            1e-6f, 1e-5f);
        free(state_split);
        free(out_split);
    }

    if (tokens <= 17u) {
        float *ref_out = calloc(value_count, sizeof(float));
        float *ref_state = calloc(state_count, sizeof(float));
        require_ok(ref_out && ref_state, "Gated DeltaNet reference allocation");
        gdn_reference(ref_out, ref_state, q, k, v, decay, beta, mask,
                      tokens, KH, VH, DIM);
        snprintf(label, sizeof(label), "Qwen GDN R4/reference output length=%u",
                 tokens);
        require_array_close(label, out_r4, ref_out, value_count, 2e-6f, 2e-4f);
        snprintf(label, sizeof(label), "Qwen GDN R4/reference state length=%u",
                 tokens);
        require_array_close(label, state_r4, ref_state, state_count,
                            2e-6f, 2e-4f);
        free(ref_state);
        free(ref_out);
    }
    free(state_r4); free(state_r2); free(state_r1);
    free(out_r4); free(out_r2); free(out_r1);
    free(initial); free(mask); free(beta); free(decay); free(v); free(k); free(q);
}

typedef struct { float score; uint32_t index; } ranked_score;

static int ranked_desc(const void *lhs, const void *rhs) {
    const ranked_score *a = lhs;
    const ranked_score *b = rhs;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return a->index < b->index ? -1 : a->index > b->index;
}

typedef enum {
    QSA_INPUT_F32,
    QSA_INPUT_F16,
    QSA_INPUT_BF16,
    QSA_INPUT_F32_BF16,
} qsa_input_type;

static const char *qsa_input_name(qsa_input_type type) {
    switch (type) {
        case QSA_INPUT_F16: return "FP16";
        case QSA_INPUT_BF16: return "BF16";
        case QSA_INPUT_F32_BF16: return "FP32/BF16";
        default: return "FP32";
    }
}

static void test_qsa(uint32_t blocks, qsa_input_type input_type) {
    enum { HEADS = 4, DIM = 128, TOP_K = 512 };
    const uint32_t valid = blocks - 7u;
    float q[HEADS * DIM];
    float *pooled = malloc((size_t)blocks * DIM * sizeof(float));
    float *expected = malloc((size_t)blocks * sizeof(float));
    float *actual = malloc((size_t)blocks * sizeof(float));
    uint16_t *q_lowp =
        (input_type == QSA_INPUT_F16 || input_type == QSA_INPUT_BF16) ?
        malloc((size_t)HEADS * DIM * sizeof(uint16_t)) : NULL;
    uint16_t *pooled_lowp = input_type == QSA_INPUT_F32 ? NULL :
        malloc((size_t)blocks * DIM * sizeof(uint16_t));
    ranked_score *a_rank = malloc((size_t)valid * sizeof(*a_rank));
    ranked_score *e_rank = malloc((size_t)valid * sizeof(*e_rank));
    float *gpu_top_scores = malloc((size_t)TOP_K * sizeof(float));
    uint32_t *gpu_top_indices = malloc((size_t)TOP_K * sizeof(uint32_t));
    require_ok(pooled && expected && actual && a_rank && e_rank &&
                   gpu_top_scores && gpu_top_indices &&
                   (input_type == QSA_INPUT_F32 || pooled_lowp) &&
                   ((input_type != QSA_INPUT_F16 &&
                     input_type != QSA_INPUT_BF16) || q_lowp),
               "QSA host allocation");
    for (uint32_t i = 0; i < HEADS * DIM; i++) {
        q[i] = 0.003f * (float)((int)(i % 41u) - 20);
        if (input_type == QSA_INPUT_F16) {
            q_lowp[i] = f32_to_f16(q[i]);
            q[i] = f16_to_f32(q_lowp[i]);
        } else if (input_type == QSA_INPUT_BF16) {
            q_lowp[i] = f32_to_bf16(q[i]);
            q[i] = bf16_to_f32(q_lowp[i]);
        }
    }
    for (uint32_t b = 0; b < blocks; b++) {
        for (uint32_t d = 0; d < DIM; d++) {
            const size_t at = (size_t)b * DIM + d;
            pooled[at] =
                0.002f * (float)((int)((b * 13u + d * 17u) % 97u) - 48) +
                1e-7f * (float)b;
            if (input_type == QSA_INPUT_F16) {
                pooled_lowp[at] = f32_to_f16(pooled[at]);
                pooled[at] = f16_to_f32(pooled_lowp[at]);
            } else if (input_type == QSA_INPUT_BF16 ||
                       input_type == QSA_INPUT_F32_BF16) {
                pooled_lowp[at] = f32_to_bf16(pooled[at]);
                pooled[at] = bf16_to_f32(pooled_lowp[at]);
            }
        }
        float score = 0.0f;
        for (uint32_t h = 0; h < HEADS; h++) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < DIM; d++)
                dot = fmaf(q[h * DIM + d], pooled[(size_t)b * DIM + d], dot);
            score += fmaxf(dot, 0.0f);
        }
        expected[b] = b < valid ? score / sqrtf((float)DIM) : -INFINITY;
    }
    ds4_gpu_tensor *qt =
        (input_type == QSA_INPUT_F32 || input_type == QSA_INPUT_F32_BF16)
        ? tensor_from(q, sizeof(q))
        : tensor_from(q_lowp, (size_t)HEADS * DIM * sizeof(uint16_t));
    ds4_gpu_tensor *kt = input_type == QSA_INPUT_F32
        ? tensor_from(pooled, (size_t)blocks * DIM * sizeof(float))
        : tensor_from(pooled_lowp,
                      (size_t)blocks * DIM * sizeof(uint16_t));
    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc((size_t)blocks * sizeof(float));
    require_ok(st != NULL, "QSA score allocation");
    const int dispatch_ok = input_type == QSA_INPUT_F16
        ? ds4_gpu_qwen4_qsa_score_m1_f16(
              st, qt, kt, blocks, valid, HEADS, DIM)
        : (input_type == QSA_INPUT_BF16
           ? ds4_gpu_qwen4_qsa_score_m1_bf16(
                 st, qt, kt, blocks, valid, HEADS, DIM)
           : (input_type == QSA_INPUT_F32_BF16
              ? ds4_gpu_qwen4_qsa_score_m1_f32_bf16(
                    st, qt, kt, blocks, valid, HEADS, DIM)
           : ds4_gpu_qwen4_qsa_score_m1(
                 st, qt, kt, blocks, valid, HEADS, DIM)));
    require_ok(dispatch_ok,
               "QSA M=1 dispatch");
    require_ok(ds4_gpu_tensor_read(
                   st, 0, actual, (size_t)blocks * sizeof(float)),
               "QSA score readback");
    for (uint32_t b = 0; b < valid; b++) {
        const float tolerance = 2e-6f + 2e-4f * fabsf(expected[b]);
        if (fabsf(actual[b] - expected[b]) > tolerance) {
            fprintf(stderr, "FAIL: QSA score blocks=%u at=%u got=%g expected=%g\n",
                    blocks, b, actual[b], expected[b]);
            exit(1);
        }
        a_rank[b] = (ranked_score){ .score = actual[b], .index = b };
        e_rank[b] = (ranked_score){ .score = expected[b], .index = b };
    }
    for (uint32_t b = valid; b < blocks; b++)
        require_ok(isinf(actual[b]) && actual[b] < 0.0f,
                   "QSA invalid block mask");
    qsort(a_rank, valid, sizeof(*a_rank), ranked_desc);
    qsort(e_rank, valid, sizeof(*e_rank), ranked_desc);
    const uint32_t top = valid < TOP_K ? valid : TOP_K;
    for (uint32_t i = 0; i < top; i++) {
        if (a_rank[i].index != e_rank[i].index) {
            const float got_ref = expected[a_rank[i].index];
            const float want_ref = e_rank[i].score;
            const float tolerance =
                4e-6f + 4e-4f * fmaxf(fabsf(got_ref), fabsf(want_ref));
            if (fabsf(got_ref - want_ref) > tolerance) {
                fprintf(stderr,
                        "FAIL: ordered QSA top-k blocks=%u rank=%u "
                        "got=%u expected=%u score_delta=%g\n",
                        blocks, i, a_rank[i].index, e_rank[i].index,
                        fabsf(got_ref - want_ref));
                exit(1);
            }
        }
    }
    ds4_gpu_tensor *top_score_t =
        ds4_gpu_tensor_alloc((size_t)TOP_K * sizeof(float));
    ds4_gpu_tensor *top_index_t =
        ds4_gpu_tensor_alloc((size_t)TOP_K * sizeof(uint32_t));
    ds4_gpu_tensor *top_count_t = ds4_gpu_tensor_alloc(sizeof(uint32_t));
    require_ok(top_score_t && top_index_t && top_count_t,
               "QSA ordered top-k allocation");
    require_ok(ds4_gpu_qwen4_qsa_topk_scores(
                   top_score_t, top_index_t, top_count_t, st, blocks, TOP_K),
               "QSA ordered top-k dispatch");
    uint32_t gpu_top_count = 0u;
    require_ok(ds4_gpu_tensor_read(
                   top_count_t, 0, &gpu_top_count, sizeof(gpu_top_count)) &&
               ds4_gpu_tensor_read(
                   top_score_t, 0, gpu_top_scores,
                   (size_t)TOP_K * sizeof(float)) &&
               ds4_gpu_tensor_read(
                   top_index_t, 0, gpu_top_indices,
                   (size_t)TOP_K * sizeof(uint32_t)),
               "QSA ordered top-k readback");
    require_ok(gpu_top_count == top, "QSA ordered top-k count");
    for (uint32_t i = 0; i < top; i++) {
        if (gpu_top_indices[i] != a_rank[i].index ||
            gpu_top_scores[i] != a_rank[i].score) {
            fprintf(stderr,
                    "FAIL: GPU ordered QSA top-k %s blocks=%u rank=%u "
                    "got=(%u,%g) expected=(%u,%g)\n",
                    qsa_input_name(input_type), blocks, i,
                    gpu_top_indices[i], gpu_top_scores[i],
                    a_rank[i].index, a_rank[i].score);
            exit(1);
        }
    }
    printf("Qwen exact M=1 QSA %s scores/top-k PASS blocks=%u valid=%u\n",
           qsa_input_name(input_type), blocks, valid);
    ds4_gpu_tensor_free(top_count_t);
    ds4_gpu_tensor_free(top_index_t);
    ds4_gpu_tensor_free(top_score_t);
    ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qt);
    free(gpu_top_indices); free(gpu_top_scores);
    free(e_rank); free(a_rank); free(pooled_lowp); free(q_lowp);
    free(actual); free(expected); free(pooled);
}

static float qsa_rope_reference(float current, float paired, uint32_t dim,
                                uint32_t rope_dim, uint32_t position,
                                float theta) {
    if (dim >= rope_dim) return current;
    const uint32_t half_dim = rope_dim / 2u;
    const uint32_t freq = dim < half_dim ? dim : dim - half_dim;
    const float angle = (float)position *
        powf(theta, -2.0f * (float)freq / (float)rope_dim);
    return dim < half_dim
        ? current * cosf(angle) - paired * sinf(angle)
        : current * cosf(angle) + paired * sinf(angle);
}

static void test_qsa_prepare(void) {
    enum {
        TOKENS = 5,
        CACHE_POS = 3,
        CACHE_CAP = 8,
        Q_HEADS = 2,
        KV_HEADS = 1,
        HEAD_DIM = 256,
        INDEX_HEADS = 4,
        INDEX_DIM = 128,
        RATIO = 4,
        ROPE_DIM = 64,
        Q_NORM_OFFSET = 78000,
        K_NORM_OFFSET = 78600,
        IQ_NORM_OFFSET = 79200,
        IK_NORM_OFFSET = 79600,
    };
    const float theta = 10000000.0f;
    const float eps = 1.0e-6f;
    int32_t mrope_positions[3 * CACHE_CAP];
    for (uint32_t axis = 0; axis < 3u; axis++)
        for (uint32_t position = 0; position < CACHE_CAP; position++)
            mrope_positions[axis * CACHE_CAP + position] =
                (int32_t)((axis + 1u) * position + axis);
    void *model = model_fixture_allocation;
    const size_t model_size = model_fixture_size;
    require_ok(model && model_size > IK_NORM_OFFSET +
                   INDEX_DIM * sizeof(uint16_t),
               "QSA preparation model fixture range");
    uint16_t *q_weight = (uint16_t *)((uint8_t *)model + Q_NORM_OFFSET);
    uint16_t *k_weight = (uint16_t *)((uint8_t *)model + K_NORM_OFFSET);
    uint16_t *iq_weight = (uint16_t *)((uint8_t *)model + IQ_NORM_OFFSET);
    uint16_t *ik_weight = (uint16_t *)((uint8_t *)model + IK_NORM_OFFSET);
    for (uint32_t dim = 0; dim < HEAD_DIM; dim++) {
        q_weight[dim] = f32_to_bf16(0.75f + 0.001f * (float)(dim % 43u));
        k_weight[dim] = f32_to_bf16(0.85f + 0.001f * (float)(dim % 37u));
    }
    for (uint32_t dim = 0; dim < INDEX_DIM; dim++) {
        iq_weight[dim] = f32_to_bf16(0.9f + 0.001f * (float)(dim % 31u));
        ik_weight[dim] = f32_to_bf16(0.8f + 0.001f * (float)(dim % 29u));
    }

    const size_t q_count = (size_t)TOKENS * Q_HEADS * HEAD_DIM;
    const size_t kv_input_count = (size_t)TOKENS * KV_HEADS * HEAD_DIM;
    const size_t iq_count = (size_t)TOKENS * INDEX_HEADS * INDEX_DIM;
    const size_t q_raw_count = 2u * q_count;
    const size_t index_raw_count =
        (size_t)TOKENS * (INDEX_HEADS + 1u) * INDEX_DIM;
    const size_t kv_cache_count = (size_t)CACHE_CAP * KV_HEADS * HEAD_DIM;
    const size_t raw_cache_count = (size_t)CACHE_CAP * INDEX_DIM;
    const size_t pooled_count = (size_t)(CACHE_CAP / RATIO) * INDEX_DIM;
    float *q_raw = malloc(q_raw_count * sizeof(float));
    float *key_raw = malloc(kv_input_count * sizeof(float));
    float *value_raw = malloc(kv_input_count * sizeof(float));
    float *index_raw = malloc(index_raw_count * sizeof(float));
    float *q_expected = malloc(q_count * sizeof(float));
    float *q_actual = malloc(q_count * sizeof(float));
    float *gate_expected = malloc(q_count * sizeof(float));
    float *gate_actual = malloc(q_count * sizeof(float));
    float *iq_expected = malloc(iq_count * sizeof(float));
    float *iq_actual = malloc(iq_count * sizeof(float));
    uint16_t *key_expected = malloc(kv_cache_count * sizeof(uint16_t));
    uint16_t *key_initial = malloc(kv_cache_count * sizeof(uint16_t));
    uint16_t *key_actual = malloc(kv_cache_count * sizeof(uint16_t));
    uint16_t *value_expected = malloc(kv_cache_count * sizeof(uint16_t));
    uint16_t *value_initial = malloc(kv_cache_count * sizeof(uint16_t));
    uint16_t *value_actual = malloc(kv_cache_count * sizeof(uint16_t));
    uint16_t *raw_expected = malloc(raw_cache_count * sizeof(uint16_t));
    uint16_t *raw_initial = malloc(raw_cache_count * sizeof(uint16_t));
    uint16_t *raw_actual = malloc(raw_cache_count * sizeof(uint16_t));
    uint16_t *pooled_expected = calloc(pooled_count, sizeof(uint16_t));
    uint16_t *pooled_actual = malloc(pooled_count * sizeof(uint16_t));
    require_ok(q_raw && key_raw && value_raw && index_raw && q_expected &&
                   q_actual && gate_expected && gate_actual && iq_expected &&
                   iq_actual && key_expected && key_initial && key_actual &&
                   value_expected && value_initial && value_actual &&
                   raw_expected && raw_initial && raw_actual &&
                   pooled_expected && pooled_actual,
               "QSA preparation host allocation");
    for (size_t i = 0; i < q_raw_count; i++)
        q_raw[i] = 0.009f * (float)((int)((i * 5u + 7u) % 47u) - 23);
    for (size_t i = 0; i < kv_input_count; i++) {
        key_raw[i] = 0.011f * (float)((int)((i * 7u + 2u) % 41u) - 20);
        value_raw[i] = 0.013f * (float)((int)((i * 11u + 3u) % 43u) - 21);
    }
    for (size_t i = 0; i < index_raw_count; i++)
        index_raw[i] =
            0.008f * (float)((int)((i * 13u + 5u) % 53u) - 26);
    for (size_t i = 0; i < kv_cache_count; i++) {
        key_initial[i] = key_expected[i] = f32_to_bf16(
            0.001f * (float)((int)(i % 17u) - 8));
        value_initial[i] = value_expected[i] = f32_to_bf16(
            0.0015f * (float)((int)(i % 19u) - 9));
    }
    for (size_t i = 0; i < raw_cache_count; i++)
        raw_initial[i] = raw_expected[i] = f32_to_bf16(
            0.002f * (float)((int)((i * 3u) % 23u) - 11));

    for (uint32_t token = 0; token < TOKENS; token++) {
        const uint32_t position = CACHE_POS + token;
        for (uint32_t head = 0; head < Q_HEADS; head++) {
            const size_t raw_base =
                (size_t)token * Q_HEADS * HEAD_DIM * 2u +
                (size_t)head * HEAD_DIM * 2u;
            const size_t out_base =
                ((size_t)token * Q_HEADS + head) * HEAD_DIM;
            float sum = 0.0f;
            for (uint32_t dim = 0; dim < HEAD_DIM; dim++)
                sum = fmaf(q_raw[raw_base + dim], q_raw[raw_base + dim], sum);
            const float scale = 1.0f /
                sqrtf(sum / (float)HEAD_DIM + eps);
            for (uint32_t dim = 0; dim < HEAD_DIM; dim++) {
                const uint32_t half_dim = ROPE_DIM / 2u;
                const uint32_t pair = dim < ROPE_DIM
                    ? (dim < half_dim ? dim + half_dim : dim - half_dim)
                    : dim;
                const float current = q_raw[raw_base + dim] * scale *
                    bf16_to_f32(q_weight[dim]);
                const float paired = dim < ROPE_DIM
                    ? q_raw[raw_base + pair] * scale *
                      bf16_to_f32(q_weight[pair])
                    : 0.0f;
                const uint32_t freq = dim < half_dim
                    ? dim : (dim < ROPE_DIM ? dim - half_dim : 0u);
                const uint32_t rope_position = dim < ROPE_DIM
                    ? (uint32_t)mrope_positions[
                        (freq % 3u) * CACHE_CAP + position]
                    : position;
                q_expected[out_base + dim] = qsa_rope_reference(
                    current, paired, dim, ROPE_DIM, rope_position, theta);
                gate_expected[out_base + dim] =
                    q_raw[raw_base + HEAD_DIM + dim];
            }
        }
        for (uint32_t head = 0; head < KV_HEADS; head++) {
            const size_t raw_base =
                ((size_t)token * KV_HEADS + head) * HEAD_DIM;
            const size_t cache_base =
                ((size_t)position * KV_HEADS + head) * HEAD_DIM;
            float sum = 0.0f;
            for (uint32_t dim = 0; dim < HEAD_DIM; dim++)
                sum = fmaf(key_raw[raw_base + dim],
                           key_raw[raw_base + dim], sum);
            const float scale = 1.0f /
                sqrtf(sum / (float)HEAD_DIM + eps);
            for (uint32_t dim = 0; dim < HEAD_DIM; dim++) {
                const uint32_t half_dim = ROPE_DIM / 2u;
                const uint32_t pair = dim < ROPE_DIM
                    ? (dim < half_dim ? dim + half_dim : dim - half_dim)
                    : dim;
                const float current = key_raw[raw_base + dim] * scale *
                    bf16_to_f32(k_weight[dim]);
                const float paired = dim < ROPE_DIM
                    ? key_raw[raw_base + pair] * scale *
                      bf16_to_f32(k_weight[pair])
                    : 0.0f;
                const uint32_t freq = dim < half_dim
                    ? dim : (dim < ROPE_DIM ? dim - half_dim : 0u);
                const uint32_t rope_position = dim < ROPE_DIM
                    ? (uint32_t)mrope_positions[
                        (freq % 3u) * CACHE_CAP + position]
                    : position;
                key_expected[cache_base + dim] = f32_to_bf16(
                    qsa_rope_reference(current, paired, dim, ROPE_DIM,
                                       rope_position, theta));
                value_expected[cache_base + dim] =
                    f32_to_bf16(value_raw[raw_base + dim]);
            }
        }
        for (uint32_t head = 0; head < INDEX_HEADS; head++) {
            const size_t raw_base =
                (size_t)token * (INDEX_HEADS + 1u) * INDEX_DIM +
                (size_t)head * INDEX_DIM;
            const size_t out_base =
                ((size_t)token * INDEX_HEADS + head) * INDEX_DIM;
            float sum = 0.0f;
            for (uint32_t dim = 0; dim < INDEX_DIM; dim++)
                sum = fmaf(index_raw[raw_base + dim],
                           index_raw[raw_base + dim], sum);
            const float scale = 1.0f /
                sqrtf(sum / (float)INDEX_DIM + eps);
            for (uint32_t dim = 0; dim < INDEX_DIM; dim++) {
                const uint32_t half_dim = ROPE_DIM / 2u;
                const uint32_t pair = dim < ROPE_DIM
                    ? (dim < half_dim ? dim + half_dim : dim - half_dim)
                    : dim;
                const float current = index_raw[raw_base + dim] * scale *
                    bf16_to_f32(iq_weight[dim]);
                const float paired = dim < ROPE_DIM
                    ? index_raw[raw_base + pair] * scale *
                      bf16_to_f32(iq_weight[pair])
                    : 0.0f;
                iq_expected[out_base + dim] = qsa_rope_reference(
                    current, paired, dim, ROPE_DIM, position, theta);
            }
        }
        const size_t raw_key_base =
            (size_t)token * (INDEX_HEADS + 1u) * INDEX_DIM +
            (size_t)INDEX_HEADS * INDEX_DIM;
        for (uint32_t dim = 0; dim < INDEX_DIM; dim++)
            raw_expected[(size_t)position * INDEX_DIM + dim] =
                f32_to_bf16(index_raw[raw_key_base + dim]);
    }

    for (uint32_t block = 0; block < CACHE_CAP / RATIO; block++) {
        float pooled[INDEX_DIM];
        float sum = 0.0f;
        for (uint32_t dim = 0; dim < INDEX_DIM; dim++) {
            float value = 0.0f;
            for (uint32_t item = 0; item < RATIO; item++)
                value += bf16_to_f32(raw_expected[
                    ((size_t)block * RATIO + item) * INDEX_DIM + dim]);
            pooled[dim] = bf16_to_f32(f32_to_bf16(value / (float)RATIO));
            sum = fmaf(pooled[dim], pooled[dim], sum);
        }
        const float scale = 1.0f /
            sqrtf(sum / (float)INDEX_DIM + eps);
        for (uint32_t dim = 0; dim < INDEX_DIM; dim++) {
            const uint32_t half_dim = ROPE_DIM / 2u;
            const uint32_t pair = dim < ROPE_DIM
                ? (dim < half_dim ? dim + half_dim : dim - half_dim)
                : dim;
            const float current = pooled[dim] * scale *
                bf16_to_f32(ik_weight[dim]);
            const float paired = dim < ROPE_DIM
                ? pooled[pair] * scale * bf16_to_f32(ik_weight[pair])
                : 0.0f;
            pooled_expected[(size_t)block * INDEX_DIM + dim] =
                f32_to_bf16(qsa_rope_reference(
                    current, paired, dim, ROPE_DIM, block * RATIO, theta));
        }
    }

    ds4_gpu_tensor *q_raw_t = tensor_from(q_raw, q_raw_count * sizeof(float));
    ds4_gpu_tensor *key_raw_t = tensor_from(
        key_raw, kv_input_count * sizeof(float));
    ds4_gpu_tensor *value_raw_t = tensor_from(
        value_raw, kv_input_count * sizeof(float));
    ds4_gpu_tensor *index_raw_t = tensor_from(
        index_raw, index_raw_count * sizeof(float));
    ds4_gpu_tensor *mrope_t = tensor_from(
        mrope_positions, sizeof(mrope_positions));
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *iq_t = ds4_gpu_tensor_alloc(iq_count * sizeof(float));
    ds4_gpu_tensor *key_cache_t = tensor_from(
        key_initial, kv_cache_count * sizeof(uint16_t));
    ds4_gpu_tensor *value_cache_t = tensor_from(
        value_initial, kv_cache_count * sizeof(uint16_t));
    ds4_gpu_tensor *raw_cache_t = tensor_from(
        raw_initial, raw_cache_count * sizeof(uint16_t));
    ds4_gpu_tensor *pooled_t = ds4_gpu_tensor_alloc(
        pooled_count * sizeof(uint16_t));
    require_ok(q_t && gate_t && iq_t && pooled_t,
               "QSA preparation device allocation");
    require_ok(ds4_gpu_qwen4_qsa_prepare_model(
                   q_t, gate_t, iq_t, key_cache_t, value_cache_t,
                   raw_cache_t, pooled_t, q_raw_t, key_raw_t, value_raw_t,
                   index_raw_t, model, model_size, Q_NORM_OFFSET,
                   K_NORM_OFFSET, IQ_NORM_OFFSET, IK_NORM_OFFSET,
                   mrope_t, CACHE_CAP,
                   CACHE_POS, TOKENS, CACHE_CAP, Q_HEADS, KV_HEADS,
                   HEAD_DIM, INDEX_HEADS, INDEX_DIM, RATIO, ROPE_DIM,
                   theta, eps),
               "QSA preparation dispatch");
    require_ok(ds4_gpu_tensor_read(q_t, 0, q_actual,
                                   q_count * sizeof(float)),
               "QSA prepared Q readback");
    require_ok(ds4_gpu_tensor_read(gate_t, 0, gate_actual,
                                   q_count * sizeof(float)),
               "QSA gate readback");
    require_ok(ds4_gpu_tensor_read(iq_t, 0, iq_actual,
                                   iq_count * sizeof(float)),
               "QSA index Q readback");
    require_ok(ds4_gpu_tensor_read(key_cache_t, 0, key_actual,
                                   kv_cache_count * sizeof(uint16_t)),
               "QSA key cache readback");
    require_ok(ds4_gpu_tensor_read(value_cache_t, 0, value_actual,
                                   kv_cache_count * sizeof(uint16_t)),
               "QSA value cache readback");
    require_ok(ds4_gpu_tensor_read(raw_cache_t, 0, raw_actual,
                                   raw_cache_count * sizeof(uint16_t)),
               "QSA raw index cache readback");
    require_ok(ds4_gpu_tensor_read(pooled_t, 0, pooled_actual,
                                   pooled_count * sizeof(uint16_t)),
               "QSA pooled index cache readback");
    require_array_close("Qwen QSA prepared multimodal partial-RoPE Q", q_actual,
                        q_expected, q_count, 3e-6f, 3e-5f);
    require_array_close("Qwen QSA query gate split", gate_actual,
                        gate_expected, q_count, 0.0f, 0.0f);
    require_array_close("Qwen QSA prepared index Q", iq_actual,
                        iq_expected, iq_count, 3e-6f, 3e-5f);
    for (size_t i = 0; i < kv_cache_count; i++) {
        key_actual[i] = f32_to_bf16(bf16_to_f32(key_actual[i]));
        value_actual[i] = f32_to_bf16(bf16_to_f32(value_actual[i]));
        require_ok(key_actual[i] == key_expected[i],
                   "QSA BF16 key cache parity");
        require_ok(value_actual[i] == value_expected[i],
                   "QSA BF16 value cache parity");
    }
    for (size_t i = 0; i < raw_cache_count; i++)
        require_ok(raw_actual[i] == raw_expected[i],
                   "QSA BF16 raw index cache parity");
    for (size_t i = 0; i < pooled_count; i++) {
        const float got = bf16_to_f32(pooled_actual[i]);
        const float want = bf16_to_f32(pooled_expected[i]);
        require_ok(fabsf(got - want) <= 0.004f + 0.002f * fabsf(want),
                   "QSA BF16 pooled index cache parity");
    }
    puts("Qwen QSA unaligned BF16 cache/pooling preparation PASS");

    ds4_gpu_tensor_free(pooled_t);
    ds4_gpu_tensor_free(raw_cache_t);
    ds4_gpu_tensor_free(value_cache_t);
    ds4_gpu_tensor_free(key_cache_t);
    ds4_gpu_tensor_free(iq_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(mrope_t);
    ds4_gpu_tensor_free(index_raw_t);
    ds4_gpu_tensor_free(value_raw_t);
    ds4_gpu_tensor_free(key_raw_t);
    ds4_gpu_tensor_free(q_raw_t);
    free(pooled_actual); free(pooled_expected);
    free(raw_actual); free(raw_initial); free(raw_expected);
    free(value_actual); free(value_initial); free(value_expected);
    free(key_actual); free(key_initial); free(key_expected);
    free(iq_actual); free(iq_expected);
    free(gate_actual); free(gate_expected);
    free(q_actual); free(q_expected);
    free(index_raw); free(value_raw); free(key_raw); free(q_raw);
}

enum {
    QSA_BOUNDARY_CACHE_CAP = 12,
    QSA_BOUNDARY_HEAD_DIM = 256,
    QSA_BOUNDARY_INDEX_DIM = 128,
    QSA_BOUNDARY_RATIO = 4,
    QSA_BOUNDARY_ROPE_DIM = 64,
    QSA_BOUNDARY_Q_NORM_OFFSET = 81000,
    QSA_BOUNDARY_K_NORM_OFFSET = 81600,
    QSA_BOUNDARY_IQ_NORM_OFFSET = 82200,
    QSA_BOUNDARY_IK_NORM_OFFSET = 82600,
};

typedef struct {
    ds4_gpu_tensor *key;
    ds4_gpu_tensor *value;
    ds4_gpu_tensor *raw_index;
    ds4_gpu_tensor *pooled_index;
} qsa_boundary_cache;

static void qsa_boundary_dispatch(
        qsa_boundary_cache *cache,
        ds4_gpu_tensor *q,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *index_q,
        const ds4_gpu_tensor *q_gate_raw,
        const ds4_gpu_tensor *key_raw,
        const ds4_gpu_tensor *value_raw,
        const ds4_gpu_tensor *index_qk_raw,
        uint32_t cache_pos,
        uint32_t rows) {
    require_ok(ds4_gpu_qwen4_qsa_prepare_model(
                   q, gate, index_q, cache->key, cache->value,
                   cache->raw_index, cache->pooled_index,
                   q_gate_raw, key_raw, value_raw, index_qk_raw,
                   model_fixture_allocation, model_fixture_size,
                   QSA_BOUNDARY_Q_NORM_OFFSET,
                   QSA_BOUNDARY_K_NORM_OFFSET,
                   QSA_BOUNDARY_IQ_NORM_OFFSET,
                   QSA_BOUNDARY_IK_NORM_OFFSET,
                   NULL, 0u, cache_pos, rows, QSA_BOUNDARY_CACHE_CAP,
                   1u, 1u, QSA_BOUNDARY_HEAD_DIM, 1u,
                   QSA_BOUNDARY_INDEX_DIM, QSA_BOUNDARY_RATIO,
                   QSA_BOUNDARY_ROPE_DIM, 10000000.0f, 1.0e-6f),
               "QSA partial-commit boundary dispatch");
}

static void test_qsa_partial_commit_pool_boundary(void) {
    enum { VERIFY_ROWS = 4, FOLLOW_ROWS = 1, FUTURE_ROWS = 4 };
    uint16_t *model = model_fixture_allocation;
    require_ok(model_fixture_size >
                   QSA_BOUNDARY_IK_NORM_OFFSET +
                       QSA_BOUNDARY_INDEX_DIM * sizeof(uint16_t),
               "QSA boundary model fixture range");
    for (uint32_t dim = 0; dim < QSA_BOUNDARY_HEAD_DIM; dim++) {
        model[QSA_BOUNDARY_Q_NORM_OFFSET / sizeof(uint16_t) + dim] =
            f32_to_bf16(0.75f + 0.001f * (float)(dim % 43u));
        model[QSA_BOUNDARY_K_NORM_OFFSET / sizeof(uint16_t) + dim] =
            f32_to_bf16(0.85f + 0.001f * (float)(dim % 37u));
    }
    for (uint32_t dim = 0; dim < QSA_BOUNDARY_INDEX_DIM; dim++) {
        model[QSA_BOUNDARY_IQ_NORM_OFFSET / sizeof(uint16_t) + dim] =
            f32_to_bf16(0.9f + 0.001f * (float)(dim % 31u));
        model[QSA_BOUNDARY_IK_NORM_OFFSET / sizeof(uint16_t) + dim] =
            f32_to_bf16(0.8f + 0.001f * (float)(dim % 29u));
    }

    const size_t q_stride = 2u * QSA_BOUNDARY_HEAD_DIM;
    const size_t kv_stride = QSA_BOUNDARY_HEAD_DIM;
    const size_t index_stride = 2u * QSA_BOUNDARY_INDEX_DIM;
    float *verify_q = malloc(VERIFY_ROWS * q_stride * sizeof(float));
    float *verify_k = malloc(VERIFY_ROWS * kv_stride * sizeof(float));
    float *verify_v = malloc(VERIFY_ROWS * kv_stride * sizeof(float));
    float *verify_i = malloc(VERIFY_ROWS * index_stride * sizeof(float));
    float *follow_q = malloc(FOLLOW_ROWS * q_stride * sizeof(float));
    float *follow_k = malloc(FOLLOW_ROWS * kv_stride * sizeof(float));
    float *follow_v = malloc(FOLLOW_ROWS * kv_stride * sizeof(float));
    float *follow_i = malloc(FOLLOW_ROWS * index_stride * sizeof(float));
    float *future_q = malloc(FUTURE_ROWS * q_stride * sizeof(float));
    float *future_k = malloc(FUTURE_ROWS * kv_stride * sizeof(float));
    float *future_v = malloc(FUTURE_ROWS * kv_stride * sizeof(float));
    float *future_i = malloc(FUTURE_ROWS * index_stride * sizeof(float));
    require_ok(verify_q && verify_k && verify_v && verify_i && follow_q &&
                   follow_k && follow_v && follow_i && future_q && future_k &&
                   future_v && future_i,
               "QSA boundary host inputs");
#define FILL_QSA_BOUNDARY(values, count, mul, add, mod, scale) do {            \
    for (size_t fill_i = 0; fill_i < (count); fill_i++)                       \
        (values)[fill_i] = (scale) *                                          \
            (float)((int)((fill_i * (mul) + (add)) % (mod)) -                \
                    (int)((mod) / 2u));                                       \
} while (0)
    FILL_QSA_BOUNDARY(verify_q, VERIFY_ROWS * q_stride,
                      5u, 7u, 47u, 0.009f);
    FILL_QSA_BOUNDARY(verify_k, VERIFY_ROWS * kv_stride,
                      7u, 2u, 41u, 0.011f);
    FILL_QSA_BOUNDARY(verify_v, VERIFY_ROWS * kv_stride,
                      11u, 3u, 43u, 0.013f);
    FILL_QSA_BOUNDARY(verify_i, VERIFY_ROWS * index_stride,
                      13u, 5u, 53u, 0.008f);
    FILL_QSA_BOUNDARY(follow_q, FOLLOW_ROWS * q_stride,
                      17u, 11u, 59u, 0.007f);
    FILL_QSA_BOUNDARY(follow_k, FOLLOW_ROWS * kv_stride,
                      19u, 13u, 61u, 0.006f);
    FILL_QSA_BOUNDARY(follow_v, FOLLOW_ROWS * kv_stride,
                      23u, 17u, 67u, 0.005f);
    FILL_QSA_BOUNDARY(follow_i, FOLLOW_ROWS * index_stride,
                      29u, 19u, 71u, 0.004f);
    FILL_QSA_BOUNDARY(future_q, FUTURE_ROWS * q_stride,
                      31u, 23u, 73u, 0.003f);
    FILL_QSA_BOUNDARY(future_k, FUTURE_ROWS * kv_stride,
                      37u, 29u, 79u, 0.0035f);
    FILL_QSA_BOUNDARY(future_v, FUTURE_ROWS * kv_stride,
                      41u, 31u, 83u, 0.0025f);
    FILL_QSA_BOUNDARY(future_i, FUTURE_ROWS * index_stride,
                      43u, 37u, 89u, 0.002f);
#undef FILL_QSA_BOUNDARY

    /* The first verifier row is the one accepted by both branches. */
    ds4_gpu_tensor *verify_q_t = tensor_from(
        verify_q, VERIFY_ROWS * q_stride * sizeof(float));
    ds4_gpu_tensor *verify_k_t = tensor_from(
        verify_k, VERIFY_ROWS * kv_stride * sizeof(float));
    ds4_gpu_tensor *verify_v_t = tensor_from(
        verify_v, VERIFY_ROWS * kv_stride * sizeof(float));
    ds4_gpu_tensor *verify_i_t = tensor_from(
        verify_i, VERIFY_ROWS * index_stride * sizeof(float));
    ds4_gpu_tensor *follow_q_t = tensor_from(
        follow_q, FOLLOW_ROWS * q_stride * sizeof(float));
    ds4_gpu_tensor *follow_k_t = tensor_from(
        follow_k, FOLLOW_ROWS * kv_stride * sizeof(float));
    ds4_gpu_tensor *follow_v_t = tensor_from(
        follow_v, FOLLOW_ROWS * kv_stride * sizeof(float));
    ds4_gpu_tensor *follow_i_t = tensor_from(
        follow_i, FOLLOW_ROWS * index_stride * sizeof(float));
    ds4_gpu_tensor *future_q_t = tensor_from(
        future_q, FUTURE_ROWS * q_stride * sizeof(float));
    ds4_gpu_tensor *future_k_t = tensor_from(
        future_k, FUTURE_ROWS * kv_stride * sizeof(float));
    ds4_gpu_tensor *future_v_t = tensor_from(
        future_v, FUTURE_ROWS * kv_stride * sizeof(float));
    ds4_gpu_tensor *future_i_t = tensor_from(
        future_i, FUTURE_ROWS * index_stride * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(
        VERIFY_ROWS * QSA_BOUNDARY_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(
        VERIFY_ROWS * QSA_BOUNDARY_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *index_q = ds4_gpu_tensor_alloc(
        VERIFY_ROWS * QSA_BOUNDARY_INDEX_DIM * sizeof(float));
    require_ok(verify_q_t && verify_k_t && verify_v_t && verify_i_t &&
                   follow_q_t && follow_k_t && follow_v_t && follow_i_t &&
                   future_q_t && future_k_t && future_v_t && future_i_t &&
                   q && gate && index_q,
               "QSA boundary device inputs");

    const size_t kv_count =
        (size_t)QSA_BOUNDARY_CACHE_CAP * QSA_BOUNDARY_HEAD_DIM;
    const size_t raw_count =
        (size_t)QSA_BOUNDARY_CACHE_CAP * QSA_BOUNDARY_INDEX_DIM;
    const size_t pooled_count =
        (size_t)(QSA_BOUNDARY_CACHE_CAP / QSA_BOUNDARY_RATIO) *
        QSA_BOUNDARY_INDEX_DIM;
    uint16_t *key_initial = malloc(kv_count * sizeof(uint16_t));
    uint16_t *value_initial = malloc(kv_count * sizeof(uint16_t));
    uint16_t *raw_initial = malloc(raw_count * sizeof(uint16_t));
    uint16_t *pooled_initial = malloc(pooled_count * sizeof(uint16_t));
    uint16_t *actual = malloc((2u * kv_count + raw_count + pooled_count) *
                              sizeof(uint16_t));
    uint16_t *reference = malloc(
        (2u * kv_count + raw_count + pooled_count) * sizeof(uint16_t));
    require_ok(key_initial && value_initial && raw_initial && pooled_initial &&
                   actual && reference,
               "QSA boundary host caches");
    for (size_t i = 0; i < kv_count; i++) {
        key_initial[i] = f32_to_bf16(
            0.001f * (float)((int)(i % 17u) - 8));
        value_initial[i] = f32_to_bf16(
            0.0015f * (float)((int)(i % 19u) - 9));
    }
    for (size_t i = 0; i < raw_count; i++)
        raw_initial[i] = f32_to_bf16(
            0.002f * (float)((int)((i * 3u) % 23u) - 11));
    for (size_t i = 0; i < pooled_count; i++)
        pooled_initial[i] = f32_to_bf16(
            0.003f * (float)((int)((i * 5u) % 29u) - 14));

    qsa_boundary_cache speculative = {
        tensor_from(key_initial, kv_count * sizeof(uint16_t)),
        tensor_from(value_initial, kv_count * sizeof(uint16_t)),
        tensor_from(raw_initial, raw_count * sizeof(uint16_t)),
        tensor_from(pooled_initial, pooled_count * sizeof(uint16_t)),
    };
    qsa_boundary_cache replay = {
        tensor_from(key_initial, kv_count * sizeof(uint16_t)),
        tensor_from(value_initial, kv_count * sizeof(uint16_t)),
        tensor_from(raw_initial, raw_count * sizeof(uint16_t)),
        tensor_from(pooled_initial, pooled_count * sizeof(uint16_t)),
    };
    require_ok(speculative.key && speculative.value && speculative.raw_index &&
                   speculative.pooled_index && replay.key && replay.value &&
                   replay.raw_index && replay.pooled_index,
               "QSA boundary device caches");

    /* At logical position 2, the speculative branch writes positions 2..5,
     * while the replay branch writes only accepted position 2.  Position 3
     * then crosses the ratio-4 boundary, and positions 4..7 overwrite the
     * rejected physical suffix and complete the next pooled row. */
    qsa_boundary_dispatch(&speculative, q, gate, index_q,
                          verify_q_t, verify_k_t, verify_v_t, verify_i_t,
                          2u, VERIFY_ROWS);
    qsa_boundary_dispatch(&replay, q, gate, index_q,
                          verify_q_t, verify_k_t, verify_v_t, verify_i_t,
                          2u, 1u);
    qsa_boundary_dispatch(&speculative, q, gate, index_q,
                          follow_q_t, follow_k_t, follow_v_t, follow_i_t,
                          3u, FOLLOW_ROWS);
    qsa_boundary_dispatch(&replay, q, gate, index_q,
                          follow_q_t, follow_k_t, follow_v_t, follow_i_t,
                          3u, FOLLOW_ROWS);
    qsa_boundary_dispatch(&speculative, q, gate, index_q,
                          future_q_t, future_k_t, future_v_t, future_i_t,
                          4u, FUTURE_ROWS);
    qsa_boundary_dispatch(&replay, q, gate, index_q,
                          future_q_t, future_k_t, future_v_t, future_i_t,
                          4u, FUTURE_ROWS);

    size_t at = 0u;
#define READ_QSA_BOUNDARY_PAIR(field, count, label) do {                       \
    require_ok(ds4_gpu_tensor_read(speculative.field, 0u, actual + at,        \
                                   (count) * sizeof(uint16_t)), label);        \
    require_ok(ds4_gpu_tensor_read(replay.field, 0u, reference + at,          \
                                   (count) * sizeof(uint16_t)), label);        \
    at += (count);                                                            \
} while (0)
    READ_QSA_BOUNDARY_PAIR(key, kv_count, "QSA boundary key readback");
    READ_QSA_BOUNDARY_PAIR(value, kv_count, "QSA boundary value readback");
    READ_QSA_BOUNDARY_PAIR(raw_index, raw_count,
                           "QSA boundary raw-index readback");
    READ_QSA_BOUNDARY_PAIR(pooled_index, pooled_count,
                           "QSA boundary pooled-index readback");
#undef READ_QSA_BOUNDARY_PAIR
    require_ok(!memcmp(actual, reference, at * sizeof(uint16_t)),
               "QSA rejected suffix is overwritten and pooled rows recompute");
    puts("Qwen QSA ratio-4 partial-commit boundary PASS");

    ds4_gpu_tensor_free(replay.pooled_index);
    ds4_gpu_tensor_free(replay.raw_index);
    ds4_gpu_tensor_free(replay.value);
    ds4_gpu_tensor_free(replay.key);
    ds4_gpu_tensor_free(speculative.pooled_index);
    ds4_gpu_tensor_free(speculative.raw_index);
    ds4_gpu_tensor_free(speculative.value);
    ds4_gpu_tensor_free(speculative.key);
    free(reference); free(actual); free(pooled_initial); free(raw_initial);
    free(value_initial); free(key_initial);
    ds4_gpu_tensor_free(index_q);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(future_i_t);
    ds4_gpu_tensor_free(future_v_t);
    ds4_gpu_tensor_free(future_k_t);
    ds4_gpu_tensor_free(future_q_t);
    ds4_gpu_tensor_free(follow_i_t);
    ds4_gpu_tensor_free(follow_v_t);
    ds4_gpu_tensor_free(follow_k_t);
    ds4_gpu_tensor_free(follow_q_t);
    ds4_gpu_tensor_free(verify_i_t);
    ds4_gpu_tensor_free(verify_v_t);
    ds4_gpu_tensor_free(verify_k_t);
    ds4_gpu_tensor_free(verify_q_t);
    free(future_i); free(future_v); free(future_k); free(future_q);
    free(follow_i); free(follow_v); free(follow_k); free(follow_q);
    free(verify_i); free(verify_v); free(verify_k); free(verify_q);
}

static void test_qsa_sparse_attention(void) {
    enum {
        QUERIES = 3,
        CACHE_CAP = 13,
        Q_HEADS = 4,
        KV_HEADS = 2,
        DIM = 256,
        TOP_K = 2,
        RATIO = 4,
    };
    const size_t query_count = (size_t)QUERIES * Q_HEADS * DIM;
    const size_t cache_count = (size_t)CACHE_CAP * KV_HEADS * DIM;
    float *q = malloc(query_count * sizeof(float));
    float *gate = malloc(query_count * sizeof(float));
    uint16_t *key = malloc(cache_count * sizeof(uint16_t));
    uint16_t *value = malloc(cache_count * sizeof(uint16_t));
    float *expected = malloc(query_count * sizeof(float));
    float *actual = malloc(query_count * sizeof(float));
    require_ok(q && gate && key && value && expected && actual,
               "sparse QSA host allocation");
    for (size_t i = 0; i < query_count; i++) {
        q[i] = 0.006f * (float)((int)((i * 7u + 5u) % 31u) - 15);
        gate[i] = 0.025f * (float)((int)((i * 11u) % 23u) - 11);
    }
    for (size_t i = 0; i < cache_count; i++) {
        key[i] = f32_to_bf16(
            0.004f * (float)((int)((i * 13u + 3u) % 37u) - 18));
        value[i] = f32_to_bf16(
            0.007f * (float)((int)((i * 17u + 9u) % 41u) - 20));
    }
    uint32_t selected[QUERIES * TOP_K] = {
        0u, 0u,
        1u, 0u,
        2u, 0u,
    };
    uint32_t counts[QUERIES] = {0u, 2u, 2u};
    uint32_t visible[QUERIES] = {1u, 9u, 13u};
    const uint32_t kv_group = Q_HEADS / KV_HEADS;
    float scores[TOP_K * RATIO + RATIO - 1u];
    uint32_t tokens[TOP_K * RATIO + RATIO - 1u];
    for (uint32_t query = 0; query < QUERIES; query++) {
        const uint32_t complete = visible[query] / RATIO;
        const uint32_t blocks = counts[query] < complete
            ? counts[query] : complete;
        uint32_t token_count = 0;
        for (uint32_t block_rank = 0; block_rank < blocks; block_rank++)
            for (uint32_t item = 0; item < RATIO; item++)
                tokens[token_count++] =
                    selected[query * TOP_K + block_rank] * RATIO + item;
        for (uint32_t token = complete * RATIO;
             token < visible[query]; token++) tokens[token_count++] = token;

        for (uint32_t head = 0; head < Q_HEADS; head++) {
            const uint32_t kv_head = head / kv_group;
            float maximum = -INFINITY;
            for (uint32_t rank = 0; rank < token_count; rank++) {
                float dot = 0.0f;
                const size_t qbase =
                    ((size_t)query * Q_HEADS + head) * DIM;
                const size_t kbase =
                    ((size_t)tokens[rank] * KV_HEADS + kv_head) * DIM;
                for (uint32_t dim = 0; dim < DIM; dim++)
                    dot = fmaf(q[qbase + dim],
                               bf16_to_f32(key[kbase + dim]), dot);
                scores[rank] = dot / sqrtf((float)DIM);
                if (scores[rank] > maximum) maximum = scores[rank];
            }
            float sum = 0.0f;
            for (uint32_t rank = 0; rank < token_count; rank++)
                sum += expf(scores[rank] - maximum);
            const size_t out_base =
                ((size_t)query * Q_HEADS + head) * DIM;
            for (uint32_t dim = 0; dim < DIM; dim++) {
                float result = 0.0f;
                for (uint32_t rank = 0; rank < token_count; rank++) {
                    const size_t vbase =
                        ((size_t)tokens[rank] * KV_HEADS + kv_head) * DIM;
                    result = fmaf(expf(scores[rank] - maximum) / sum,
                                  bf16_to_f32(value[vbase + dim]), result);
                }
                expected[out_base + dim] = result /
                    (1.0f + expf(-gate[out_base + dim]));
            }
        }
    }

    ds4_gpu_tensor *q_t = tensor_from(q, query_count * sizeof(float));
    ds4_gpu_tensor *gate_t = tensor_from(gate, query_count * sizeof(float));
    ds4_gpu_tensor *key_t = tensor_from(key, cache_count * sizeof(uint16_t));
    ds4_gpu_tensor *value_t = tensor_from(
        value, cache_count * sizeof(uint16_t));
    ds4_gpu_tensor *selected_t = tensor_from(selected, sizeof(selected));
    ds4_gpu_tensor *counts_t = tensor_from(counts, sizeof(counts));
    ds4_gpu_tensor *visible_t = tensor_from(visible, sizeof(visible));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(query_count * sizeof(float));
    require_ok(out_t != NULL, "sparse QSA output allocation");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   out_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "sparse QSA attention dispatch");
    require_ok(ds4_gpu_tensor_read(out_t, 0, actual,
                                   query_count * sizeof(float)),
               "sparse QSA output readback");
    require_array_close("Qwen mask-free BF16 sparse QSA attention",
                        actual, expected, query_count, 3e-6f, 3e-5f);

    enum { PAD_QUERIES = 512 };
    const size_t padded_query_count =
        (size_t)PAD_QUERIES * Q_HEADS * DIM;
    float *padded_q = malloc(padded_query_count * sizeof(float));
    float *padded_gate = malloc(padded_query_count * sizeof(float));
    float *padded_out = malloc(padded_query_count * sizeof(float));
    uint32_t *padded_selected = malloc(
        (size_t)PAD_QUERIES * TOP_K * sizeof(uint32_t));
    uint32_t *padded_counts = malloc(
        (size_t)PAD_QUERIES * sizeof(uint32_t));
    uint32_t *padded_visible = malloc(
        (size_t)PAD_QUERIES * sizeof(uint32_t));
    require_ok(padded_q && padded_gate && padded_out && padded_selected &&
                   padded_counts && padded_visible,
               "padded sparse QSA host allocation");
    for (uint32_t query = 0; query < PAD_QUERIES; query++) {
        const uint32_t source = query % QUERIES;
        memcpy(padded_q + (size_t)query * Q_HEADS * DIM,
               q + (size_t)source * Q_HEADS * DIM,
               (size_t)Q_HEADS * DIM * sizeof(float));
        memcpy(padded_gate + (size_t)query * Q_HEADS * DIM,
               gate + (size_t)source * Q_HEADS * DIM,
               (size_t)Q_HEADS * DIM * sizeof(float));
        memcpy(padded_selected + (size_t)query * TOP_K,
               selected + (size_t)source * TOP_K,
               (size_t)TOP_K * sizeof(uint32_t));
        padded_counts[query] = counts[source];
        padded_visible[query] = visible[source];
    }
    ds4_gpu_tensor *padded_q_t = tensor_from(
        padded_q, padded_query_count * sizeof(float));
    ds4_gpu_tensor *padded_gate_t = tensor_from(
        padded_gate, padded_query_count * sizeof(float));
    ds4_gpu_tensor *padded_selected_t = tensor_from(
        padded_selected,
        (size_t)PAD_QUERIES * TOP_K * sizeof(uint32_t));
    ds4_gpu_tensor *padded_counts_t = tensor_from(
        padded_counts, (size_t)PAD_QUERIES * sizeof(uint32_t));
    ds4_gpu_tensor *padded_visible_t = tensor_from(
        padded_visible, (size_t)PAD_QUERIES * sizeof(uint32_t));
    ds4_gpu_tensor *padded_out_t = ds4_gpu_tensor_alloc(
        padded_query_count * sizeof(float));
    require_ok(padded_q_t && padded_gate_t && padded_selected_t &&
                   padded_counts_t && padded_visible_t && padded_out_t,
               "padded sparse QSA device allocation");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   padded_out_t, padded_q_t, padded_gate_t, key_t, value_t,
                   padded_selected_t, padded_counts_t, padded_visible_t,
                   PAD_QUERIES, CACHE_CAP, Q_HEADS, KV_HEADS, DIM, TOP_K,
                   RATIO),
               "padded sparse QSA attention dispatch");
    require_ok(ds4_gpu_tensor_read(
                   padded_out_t, 0, padded_out,
                   padded_query_count * sizeof(float)),
               "padded sparse QSA output readback");
    require_array_close("Qwen sparse QSA padded-query invariance",
                        padded_out, actual, query_count, 0.0f, 0.0f);
    ds4_gpu_tensor_free(padded_out_t);
    ds4_gpu_tensor_free(padded_visible_t);
    ds4_gpu_tensor_free(padded_counts_t);
    ds4_gpu_tensor_free(padded_selected_t);
    ds4_gpu_tensor_free(padded_gate_t);
    ds4_gpu_tensor_free(padded_q_t);
    free(padded_visible);
    free(padded_counts);
    free(padded_selected);
    free(padded_out);
    free(padded_gate);
    free(padded_q);

    float dense_scores[CACHE_CAP];
    for (uint32_t query = 0; query < QUERIES; query++) {
        for (uint32_t head = 0; head < Q_HEADS; head++) {
            const uint32_t kv_head = head / kv_group;
            const size_t qbase =
                ((size_t)query * Q_HEADS + head) * DIM;
            float maximum = -INFINITY;
            for (uint32_t token = 0; token < visible[query]; token++) {
                const size_t kbase =
                    ((size_t)token * KV_HEADS + kv_head) * DIM;
                float dot = 0.0f;
                for (uint32_t dim = 0; dim < DIM; dim++)
                    dot = fmaf(q[qbase + dim],
                               bf16_to_f32(key[kbase + dim]), dot);
                dense_scores[token] = dot / sqrtf((float)DIM);
                if (dense_scores[token] > maximum)
                    maximum = dense_scores[token];
            }
            float sum = 0.0f;
            for (uint32_t token = 0; token < visible[query]; token++)
                sum += expf(dense_scores[token] - maximum);
            for (uint32_t dim = 0; dim < DIM; dim++) {
                float result = 0.0f;
                for (uint32_t token = 0; token < visible[query]; token++) {
                    const size_t vbase =
                        ((size_t)token * KV_HEADS + kv_head) * DIM;
                    result = fmaf(
                        expf(dense_scores[token] - maximum) / sum,
                        bf16_to_f32(value[vbase + dim]), result);
                }
                expected[qbase + dim] = result /
                    (1.0f + expf(-gate[qbase + dim]));
            }
        }
    }
    require_ok(ds4_gpu_qwen4_qsa_dense_attention_bf16(
                   out_t, q_t, gate_t, key_t, value_t, visible_t,
                   QUERIES, CACHE_CAP, Q_HEADS, KV_HEADS, DIM),
               "dense causal QSA attention dispatch");
    require_ok(ds4_gpu_tensor_read(out_t, 0, actual,
                                   query_count * sizeof(float)),
               "dense causal QSA output readback");
    require_array_close("Qwen multimodal BF16 dense causal QSA attention",
                        actual, expected, query_count, 3e-6f, 3e-5f);
    ds4_gpu_tensor_free(out_t);
    ds4_gpu_tensor_free(visible_t);
    ds4_gpu_tensor_free(counts_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(value_t);
    ds4_gpu_tensor_free(key_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(q_t);
    free(actual);
    free(expected);
    free(value);
    free(key);
    free(gate);
    free(q);
}

static void test_qsa_probability_cache_exact(void) {
    enum {
        QUERIES = 3,
        CACHE_CAP = 2051,
        Q_HEADS = 24,
        KV_HEADS = 2,
        DIM = 256,
        TOP_K = 512,
        RATIO = 4,
    };
    const size_t query_count = (size_t)QUERIES * Q_HEADS * DIM;
    const size_t cache_count = (size_t)CACHE_CAP * KV_HEADS * DIM;
    const size_t selected_count = (size_t)QUERIES * TOP_K;
    float *q = malloc(query_count * sizeof(*q));
    float *gate = malloc(query_count * sizeof(*gate));
    uint16_t *key = malloc(cache_count * sizeof(*key));
    uint16_t *value = malloc(cache_count * sizeof(*value));
    uint32_t *selected = malloc(selected_count * sizeof(*selected));
    float *memoized = malloc(query_count * sizeof(*memoized));
    float *legacy = malloc(query_count * sizeof(*legacy));
    require_ok(q && gate && key && value && selected && memoized && legacy,
               "QSA probability-cache host allocation");

    for (size_t i = 0; i < query_count; i++) {
        q[i] = 0.0025f *
            (float)((int)((i * 29u + i / DIM * 7u + 11u) % 97u) - 48);
        gate[i] = 0.03125f *
            (float)((int)((i * 13u + i / DIM * 5u + 3u) % 33u) - 16);
    }
    for (uint32_t token = 0; token < CACHE_CAP; token++) {
        for (uint32_t kv = 0; kv < KV_HEADS; kv++) {
            for (uint32_t dim = 0; dim < DIM; dim++) {
                const size_t at =
                    ((size_t)token * KV_HEADS + kv) * DIM + dim;
                key[at] = f32_to_bf16(
                    0.00175f * (float)((int)(
                        (token * 31u + kv * 43u + dim * 17u + 5u) % 113u) -
                        56));
                value[at] = f32_to_bf16(
                    0.00225f * (float)((int)(
                        (token * 19u + kv * 47u + dim * 23u + 9u) % 109u) -
                        54));
            }
        }
    }
    for (uint32_t rank = 0; rank < TOP_K; rank++) {
        selected[rank] = 0u;
        selected[TOP_K + rank] = rank < 511u
            ? (rank * 37u + 19u) % 511u : 0u;
        selected[2u * TOP_K + rank] = (rank * 37u + 23u) & 511u;
    }
    const uint32_t counts[QUERIES] = {1u, 511u, 512u};
    const uint32_t visible[QUERIES] = {7u, 2047u, 2051u};

    ds4_gpu_tensor *q_t = tensor_from(q, query_count * sizeof(*q));
    ds4_gpu_tensor *gate_t = tensor_from(
        gate, query_count * sizeof(*gate));
    ds4_gpu_tensor *key_t = tensor_from(
        key, cache_count * sizeof(*key));
    ds4_gpu_tensor *value_t = tensor_from(
        value, cache_count * sizeof(*value));
    ds4_gpu_tensor *selected_t = tensor_from(
        selected, selected_count * sizeof(*selected));
    ds4_gpu_tensor *counts_t = tensor_from(counts, sizeof(counts));
    ds4_gpu_tensor *visible_t = tensor_from(visible, sizeof(visible));
    ds4_gpu_tensor *memoized_t = ds4_gpu_tensor_alloc(
        query_count * sizeof(*memoized));
    ds4_gpu_tensor *legacy_t = ds4_gpu_tensor_alloc(
        query_count * sizeof(*legacy));
    require_ok(memoized_t && legacy_t,
               "QSA probability-cache output allocation");

    const char *prior_raw = getenv("DS4_QWEN4_QSA_PROB_CACHE");
    char *prior = prior_raw ? strdup(prior_raw) : NULL;
    require_ok(!prior_raw || prior,
               "QSA probability-cache environment copy");
    require_ok(unsetenv("DS4_QWEN4_QSA_PROB_CACHE") == 0,
               "QSA probability-cache enable");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   memoized_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "QSA probability-cache dispatch");
    require_ok(setenv("DS4_QWEN4_QSA_PROB_CACHE", "0", 1) == 0,
               "QSA probability-cache legacy fallback");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   legacy_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "QSA probability-cache legacy dispatch");
    require_ok(ds4_gpu_tensor_read(
                   memoized_t, 0, memoized,
                   query_count * sizeof(*memoized)) &&
                   ds4_gpu_tensor_read(
                       legacy_t, 0, legacy,
                       query_count * sizeof(*legacy)),
               "QSA probability-cache output readback");
    require_ok(memcmp(memoized, legacy,
                      query_count * sizeof(*memoized)) == 0,
               "QSA probability-cache bitwise legacy parity");
    require_ok(prior
                   ? setenv("DS4_QWEN4_QSA_PROB_CACHE", prior, 1) == 0
                   : unsetenv("DS4_QWEN4_QSA_PROB_CACHE") == 0,
               "QSA probability-cache environment restore");
    free(prior);
    puts("Qwen QSA in-place probability cache exact parity PASS");

    ds4_gpu_tensor_free(legacy_t);
    ds4_gpu_tensor_free(memoized_t);
    ds4_gpu_tensor_free(visible_t);
    ds4_gpu_tensor_free(counts_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(value_t);
    ds4_gpu_tensor_free(key_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(q_t);
    free(legacy);
    free(memoized);
    free(selected);
    free(value);
    free(key);
    free(gate);
    free(q);
}

/* GQA tile-sharing matrix-core attention: F16 operands with F32 accumulation
 * follow the batched-kernel parity policy, so this fixture checks the MMA
 * kernel against the CPU reference at rounding-scale tolerances (rather than
 * the scalar kernel's 3e-6) and additionally cross-checks against the scalar
 * kernel at the same scale. */
static void test_qsa_attention_gqa_mma(void) {
    enum {
        QUERIES = 40,
        CACHE_CAP = 2051,
        Q_HEADS = 24,
        KV_HEADS = 2,
        DIM = 256,
        TOP_K = 512,
        RATIO = 4,
    };
    const size_t query_count = (size_t)QUERIES * Q_HEADS * DIM;
    const size_t cache_count = (size_t)CACHE_CAP * KV_HEADS * DIM;
    const size_t selected_count = (size_t)QUERIES * TOP_K;
    float *q = malloc(query_count * sizeof(*q));
    float *gate = malloc(query_count * sizeof(*gate));
    uint16_t *key = malloc(cache_count * sizeof(*key));
    uint16_t *value = malloc(cache_count * sizeof(*value));
    uint32_t *selected = malloc(selected_count * sizeof(*selected));
    uint32_t *counts = malloc((size_t)QUERIES * sizeof(*counts));
    uint32_t *visible = malloc((size_t)QUERIES * sizeof(*visible));
    float *expected = malloc(query_count * sizeof(*expected));
    float *actual = malloc(query_count * sizeof(*actual));
    float *scalar = malloc(query_count * sizeof(*scalar));
    require_ok(q && gate && key && value && selected && counts && visible &&
                   expected && actual && scalar,
               "QSA GQA MMA host allocation");

    for (size_t i = 0; i < query_count; i++) {
        q[i] = 0.0025f *
            (float)((int)((i * 29u + i / DIM * 7u + 11u) % 97u) - 48);
        gate[i] = 0.03125f *
            (float)((int)((i * 13u + i / DIM * 5u + 3u) % 33u) - 16);
    }
    for (size_t i = 0; i < cache_count; i++) {
        key[i] = f32_to_bf16(
            0.00175f * (float)((int)((i * 31u + 5u) % 113u) - 56));
        value[i] = f32_to_bf16(
            0.00225f * (float)((int)((i * 19u + 9u) % 109u) - 54));
    }
    for (uint32_t query = 0; query < QUERIES; query++) {
        for (uint32_t rank = 0; rank < TOP_K; rank++)
            selected[(size_t)query * TOP_K + rank] =
                (query * 131u + rank * 37u + 19u) % 512u;
    }
    /* Boundary cases first, then deterministic mixed profiles. */
    visible[0] = 7u;    counts[0] = 1u;
    visible[1] = 2047u; counts[1] = 511u;
    visible[2] = 2051u; counts[2] = 512u;
    visible[3] = 3u;    counts[3] = 0u;   /* tail-only selection */
    visible[4] = 2u;    counts[4] = 0u;
    visible[5] = 0u;    counts[5] = 0u;   /* zero selection */
    for (uint32_t query = 6; query < QUERIES; query++) {
        visible[query] = 17u + (query * 173u) % 2051u;
        counts[query] = (query * 61u + 7u) % (TOP_K + 1u);
    }

    /* CPU reference: identical semantics to the scalar-kernel fixture. */
    for (uint32_t query = 0; query < QUERIES; query++) {
        const uint32_t complete = visible[query] / RATIO;
        const uint32_t blocks =
            counts[query] < complete ? counts[query] : complete;
        if (blocks > TOP_K) exit(1);
        uint32_t tokens[TOP_K * RATIO + RATIO - 1u];
        uint32_t token_count = 0;
        for (uint32_t rank = 0; rank < blocks; rank++)
            for (uint32_t item = 0; item < RATIO; item++) {
                const uint32_t token =
                    selected[(size_t)query * TOP_K + rank] * RATIO + item;
                if (token < visible[query]) tokens[token_count++] = token;
            }
        for (uint32_t token = complete * RATIO;
             token < visible[query];
             token++) tokens[token_count++] = token;

        for (uint32_t head = 0; head < Q_HEADS; head++) {
            const uint32_t kv_head = head / (Q_HEADS / KV_HEADS);
            float *scores = malloc(token_count * sizeof(*scores));
            require_ok(scores != NULL, "QSA GQA MMA reference scores");
            float maximum = -INFINITY;
            for (uint32_t rank = 0; rank < token_count; rank++) {
                float dot = 0.0f;
                const size_t qbase = ((size_t)query * Q_HEADS + head) * DIM;
                const size_t kbase =
                    ((size_t)tokens[rank] * KV_HEADS + kv_head) * DIM;
                for (uint32_t dim = 0; dim < DIM; dim++)
                    dot = fmaf(q[qbase + dim], bf16_to_f32(key[kbase + dim]),
                               dot);
                scores[rank] = dot / sqrtf((float)DIM);
                if (scores[rank] > maximum) maximum = scores[rank];
            }
            float sum = 0.0f;
            for (uint32_t rank = 0; rank < token_count; rank++)
                sum += expf(scores[rank] - maximum);
            const size_t out_base = ((size_t)query * Q_HEADS + head) * DIM;
            for (uint32_t dim = 0; dim < DIM; dim++) {
                float result = 0.0f;
                for (uint32_t rank = 0; rank < token_count; rank++) {
                    const size_t vbase =
                        ((size_t)tokens[rank] * KV_HEADS + kv_head) * DIM;
                    result = fmaf(expf(scores[rank] - maximum) / sum,
                                  bf16_to_f32(value[vbase + dim]), result);
                }
                expected[out_base + dim] =
                    result / (1.0f + expf(-gate[out_base + dim]));
            }
            free(scores);
        }
    }

    ds4_gpu_tensor *q_t = tensor_from(q, query_count * sizeof(*q));
    ds4_gpu_tensor *gate_t = tensor_from(gate, query_count * sizeof(*gate));
    ds4_gpu_tensor *key_t = tensor_from(key, cache_count * sizeof(*key));
    ds4_gpu_tensor *value_t =
        tensor_from(value, cache_count * sizeof(*value));
    ds4_gpu_tensor *selected_t =
        tensor_from(selected, selected_count * sizeof(*selected));
    ds4_gpu_tensor *counts_t =
        tensor_from(counts, (size_t)QUERIES * sizeof(*counts));
    ds4_gpu_tensor *visible_t =
        tensor_from(visible, (size_t)QUERIES * sizeof(*visible));
    ds4_gpu_tensor *actual_t =
        ds4_gpu_tensor_alloc(query_count * sizeof(*actual));
    ds4_gpu_tensor *scalar_t =
        ds4_gpu_tensor_alloc(query_count * sizeof(*scalar));
    require_ok(actual_t && scalar_t, "QSA GQA MMA output allocation");

    const char *prior_raw = getenv("DS4_QWEN4_QSA_GQA_MMA");
    char *prior = prior_raw ? strdup(prior_raw) : NULL;
    require_ok(!prior_raw || prior, "QSA GQA MMA environment copy");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA", "1", 1) == 0,
               "QSA GQA MMA enable");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   actual_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "QSA GQA MMA dispatch");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA", "0", 1) == 0,
               "QSA GQA MMA scalar fallback");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   scalar_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "QSA GQA MMA scalar dispatch");
    require_ok(prior
                   ? setenv("DS4_QWEN4_QSA_GQA_MMA", prior, 1) == 0
                   : unsetenv("DS4_QWEN4_QSA_GQA_MMA") == 0,
               "QSA GQA MMA environment restore");
    require_ok(ds4_gpu_tensor_read(actual_t, 0, actual,
                                   query_count * sizeof(*actual)) &&
                   ds4_gpu_tensor_read(scalar_t, 0, scalar,
                                       query_count * sizeof(*scalar)),
               "QSA GQA MMA readback");
    require_array_close("Qwen QSA GQA MMA attention vs scalar kernel",
                        actual, scalar, query_count, 2e-3f, 2e-2f);
    require_array_close("Qwen QSA GQA MMA attention vs CPU reference",
                        actual, expected, query_count, 2e-3f, 2e-2f);

    /* Production-magnitude pass: the tiny-magnitude inputs above keep the
     * softmax nearly uniform, which hid a real bug for sessions (a store
     * clobber produced exactly-uniform weights and still passed these
     * tolerances).  Scale q/k/v so the gathered score spread reaches the
     * +-20 class the full model produces; the tolerance admits F16
     * operand drift at that spread but not a wrong-mixture output. */
    for (size_t i = 0; i < query_count; i++) {
        q[i] *= 12.0f;
        gate[i] *= 12.0f;
    }
    for (size_t i = 0; i < cache_count; i++) {
        const float kf = bf16_to_f32(key[i]);
        key[i] = f32_to_bf16(kf * 12.0f);
        const float vf = bf16_to_f32(value[i]);
        value[i] = f32_to_bf16(vf * 12.0f);
    }
    for (uint32_t query = 0; query < QUERIES; query++) {
        const uint32_t complete = visible[query] / RATIO;
        const uint32_t blocks =
            counts[query] < complete ? counts[query] : complete;
        if (blocks > TOP_K) exit(1);
        uint32_t tokens[TOP_K * RATIO + RATIO - 1u];
        uint32_t token_count = 0;
        for (uint32_t rank = 0; rank < blocks; rank++)
            for (uint32_t item = 0; item < RATIO; item++) {
                const uint32_t token =
                    selected[(size_t)query * TOP_K + rank] * RATIO + item;
                if (token < visible[query]) tokens[token_count++] = token;
            }
        for (uint32_t token = complete * RATIO;
             token < visible[query];
             token++) tokens[token_count++] = token;

        for (uint32_t head = 0; head < Q_HEADS; head++) {
            const uint32_t kv_head = head / (Q_HEADS / KV_HEADS);
            float *scores = malloc(token_count * sizeof(*scores));
            require_ok(scores != NULL,
                       "QSA GQA MMA reference scores (scaled)");
            float maximum = -INFINITY;
            for (uint32_t rank = 0; rank < token_count; rank++) {
                float dot = 0.0f;
                const size_t qbase = ((size_t)query * Q_HEADS + head) * DIM;
                const size_t kbase =
                    ((size_t)tokens[rank] * KV_HEADS + kv_head) * DIM;
                for (uint32_t dim = 0; dim < DIM; dim++)
                    dot = fmaf(q[qbase + dim], bf16_to_f32(key[kbase + dim]),
                               dot);
                scores[rank] = dot / sqrtf((float)DIM);
                if (scores[rank] > maximum) maximum = scores[rank];
            }
            float sum = 0.0f;
            for (uint32_t rank = 0; rank < token_count; rank++)
                sum += expf(scores[rank] - maximum);
            const size_t out_base = ((size_t)query * Q_HEADS + head) * DIM;
            for (uint32_t dim = 0; dim < DIM; dim++) {
                float result = 0.0f;
                for (uint32_t rank = 0; rank < token_count; rank++) {
                    const size_t vbase =
                        ((size_t)tokens[rank] * KV_HEADS + kv_head) * DIM;
                    result = fmaf(expf(scores[rank] - maximum) / sum,
                                  bf16_to_f32(value[vbase + dim]), result);
                }
                expected[out_base + dim] =
                    result / (1.0f + expf(-gate[out_base + dim]));
            }
            free(scores);
        }
    }
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(key_t);
    ds4_gpu_tensor_free(value_t);
    q_t = tensor_from(q, query_count * sizeof(*q));
    gate_t = tensor_from(gate, query_count * sizeof(*gate));
    key_t = tensor_from(key, cache_count * sizeof(*key));
    value_t = tensor_from(value, cache_count * sizeof(*value));
    require_ok(q_t && gate_t && key_t && value_t,
               "QSA GQA MMA scaled upload");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA", "1", 1) == 0,
               "QSA GQA MMA enable (scaled)");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   actual_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "QSA GQA MMA dispatch (scaled)");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA", "0", 1) == 0,
               "QSA GQA MMA scalar fallback (scaled)");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   scalar_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "QSA GQA MMA scalar dispatch (scaled)");
    require_ok(prior
                   ? setenv("DS4_QWEN4_QSA_GQA_MMA", prior, 1) == 0
                   : unsetenv("DS4_QWEN4_QSA_GQA_MMA") == 0,
               "QSA GQA MMA environment restore (scaled)");
    require_ok(ds4_gpu_tensor_read(actual_t, 0, actual,
                                   query_count * sizeof(*actual)) &&
                   ds4_gpu_tensor_read(scalar_t, 0, scalar,
                                       query_count * sizeof(*scalar)),
               "QSA GQA MMA readback (scaled)");
    require_array_close("Qwen QSA GQA MMA production magnitudes vs scalar",
                        actual, scalar, query_count, 5e-2f, 2e-1f);
    require_array_close(
        "Qwen QSA GQA MMA production magnitudes vs CPU reference",
        actual, expected, query_count, 5e-2f, 2e-1f);
    float *plain_prod = malloc(query_count * sizeof(*plain_prod));
    require_ok(plain_prod != NULL, "QSA GQA MMA plain capture");
    memcpy(plain_prod, actual, query_count * sizeof(*plain_prod));

    /* Q-split instantiation on the same production-magnitude data: the
     * hi/lo F16 Q pair restores F32-class operand precision, so it must
     * sit far inside the plain variant's drift. */
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA_QSPLIT", "1", 1) == 0,
               "QSA GQA MMA Q-split enable");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA", "1", 1) == 0,
               "QSA GQA MMA enable (qsplit)");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   actual_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "QSA GQA MMA qsplit dispatch");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA_QSPLIT", "0", 1) == 0,
               "QSA GQA MMA Q-split disable");
    require_ok(prior
                   ? setenv("DS4_QWEN4_QSA_GQA_MMA", prior, 1) == 0
                   : unsetenv("DS4_QWEN4_QSA_GQA_MMA") == 0,
               "QSA GQA MMA environment restore (qsplit)");
    require_ok(ds4_gpu_tensor_read(actual_t, 0, actual,
                                   query_count * sizeof(*actual)),
               "QSA GQA MMA readback (qsplit)");
    /* Per-element maxima at these magnitudes are dominated by near-tie
     * dominant-token flips that single-ulp score differences decide for
     * ANY implementation, and the MEAN drift is dominated by the F16
     * P-tile rounding plus MMA accumulation order — restoring Q to ~22
     * mantissa bits (Q-split) leaves the mean unchanged (measured:
     * 2.630e-4 Q-split vs 2.644e-4 plain), so the bound below is a
     * family check on both variants, not a Q-split improvement claim. */
    {
        double split_mean = 0.0;
        double plain_mean = 0.0;
        for (size_t i = 0; i < query_count; i++) {
            split_mean += fabs((double)actual[i] - (double)scalar[i]);
            plain_mean += fabs((double)plain_prod[i] - (double)scalar[i]);
        }
        split_mean /= (double)query_count;
        plain_mean /= (double)query_count;
        printf("Qwen QSA GQA MMA Q-split drift mean=%.3e (plain %.3e)\n",
               split_mean, plain_mean);
        require_ok(split_mean < 1e-3 && plain_mean < 1e-3,
                   "QSA GQA MMA mean drift out of family");
    }
    require_array_close("Qwen QSA GQA MMA Q-split vs scalar",
                        actual, scalar, query_count, 5e-2f, 2e-1f);
    require_array_close("Qwen QSA GQA MMA Q-split vs CPU reference",
                        actual, expected, query_count, 5e-2f, 2e-1f);

    /* And back at the original tiny magnitudes, where the softmax is
     * nearly uniform and near-tie flips cannot mask a structural break:
     * the Q-split path must match the scalar kernel as tightly as the
     * plain path does (the production-magnitude case above cannot use a
     * tight per-element bound because single-ulp score differences flip
     * near-tied dominant tokens). */
    for (size_t i = 0; i < query_count; i++) {
        q[i] /= 12.0f;
        gate[i] /= 12.0f;
    }
    for (size_t i = 0; i < cache_count; i++) {
        const float kf = bf16_to_f32(key[i]);
        key[i] = f32_to_bf16(kf / 12.0f);
        const float vf = bf16_to_f32(value[i]);
        value[i] = f32_to_bf16(vf / 12.0f);
    }
    for (uint32_t query = 0; query < QUERIES; query++) {
        const uint32_t complete = visible[query] / RATIO;
        const uint32_t blocks =
            counts[query] < complete ? counts[query] : complete;
        uint32_t tokens[TOP_K * RATIO + RATIO - 1u];
        uint32_t token_count = 0;
        for (uint32_t rank = 0; rank < blocks; rank++)
            for (uint32_t item = 0; item < RATIO; item++) {
                const uint32_t token =
                    selected[(size_t)query * TOP_K + rank] * RATIO + item;
                if (token < visible[query]) tokens[token_count++] = token;
            }
        for (uint32_t token = complete * RATIO;
             token < visible[query];
             token++) tokens[token_count++] = token;
        for (uint32_t head = 0; head < Q_HEADS; head++) {
            const uint32_t kv_head = head / (Q_HEADS / KV_HEADS);
            float *scores = malloc(token_count * sizeof(*scores));
            require_ok(scores != NULL,
                       "QSA GQA MMA reference scores (unscaled)");
            float maximum = -INFINITY;
            for (uint32_t rank = 0; rank < token_count; rank++) {
                float dot = 0.0f;
                const size_t qbase = ((size_t)query * Q_HEADS + head) * DIM;
                const size_t kbase =
                    ((size_t)tokens[rank] * KV_HEADS + kv_head) * DIM;
                for (uint32_t dim = 0; dim < DIM; dim++)
                    dot = fmaf(q[qbase + dim], bf16_to_f32(key[kbase + dim]),
                               dot);
                scores[rank] = dot / sqrtf((float)DIM);
                if (scores[rank] > maximum) maximum = scores[rank];
            }
            float sum = 0.0f;
            for (uint32_t rank = 0; rank < token_count; rank++)
                sum += expf(scores[rank] - maximum);
            const size_t out_base = ((size_t)query * Q_HEADS + head) * DIM;
            for (uint32_t dim = 0; dim < DIM; dim++) {
                float result = 0.0f;
                for (uint32_t rank = 0; rank < token_count; rank++) {
                    const size_t vbase =
                        ((size_t)tokens[rank] * KV_HEADS + kv_head) * DIM;
                    result = fmaf(expf(scores[rank] - maximum) / sum,
                                  bf16_to_f32(value[vbase + dim]), result);
                }
                expected[out_base + dim] =
                    result / (1.0f + expf(-gate[out_base + dim]));
            }
            free(scores);
        }
    }
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(key_t);
    ds4_gpu_tensor_free(value_t);
    q_t = tensor_from(q, query_count * sizeof(*q));
    gate_t = tensor_from(gate, query_count * sizeof(*gate));
    key_t = tensor_from(key, cache_count * sizeof(*key));
    value_t = tensor_from(value, cache_count * sizeof(*value));
    require_ok(q_t && gate_t && key_t && value_t,
               "QSA GQA MMA unscaled upload");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA_QSPLIT", "1", 1) == 0,
               "QSA GQA MMA Q-split enable (unscaled)");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA", "1", 1) == 0,
               "QSA GQA MMA enable (unscaled)");
    require_ok(ds4_gpu_qwen4_qsa_attention_bf16(
                   actual_t, q_t, gate_t, key_t, value_t, selected_t,
                   counts_t, visible_t, QUERIES, CACHE_CAP, Q_HEADS,
                   KV_HEADS, DIM, TOP_K, RATIO),
               "QSA GQA MMA qsplit dispatch (unscaled)");
    require_ok(setenv("DS4_QWEN4_QSA_GQA_MMA_QSPLIT", "0", 1) == 0,
               "QSA GQA MMA Q-split disable (unscaled)");
    require_ok(prior
                   ? setenv("DS4_QWEN4_QSA_GQA_MMA", prior, 1) == 0
                   : unsetenv("DS4_QWEN4_QSA_GQA_MMA") == 0,
               "QSA GQA MMA environment restore (unscaled)");
    require_ok(ds4_gpu_tensor_read(actual_t, 0, actual,
                                   query_count * sizeof(*actual)),
               "QSA GQA MMA readback (unscaled)");
    require_array_close("Qwen QSA GQA MMA Q-split unscaled vs CPU",
                        actual, expected, query_count, 2e-3f, 2e-2f);
    free(prior);
    free(plain_prod);

    ds4_gpu_tensor_free(scalar_t);
    ds4_gpu_tensor_free(actual_t);
    ds4_gpu_tensor_free(visible_t);
    ds4_gpu_tensor_free(counts_t);
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(value_t);
    ds4_gpu_tensor_free(key_t);
    ds4_gpu_tensor_free(gate_t);
    ds4_gpu_tensor_free(q_t);
    free(scalar);
    free(actual);
    free(expected);
    free(visible);
    free(counts);
    free(selected);
    free(value);
    free(key);
    free(gate);
    free(q);
}

static void test_qsa_streaming_topk(void) {
    enum {
        QUERIES = 5,
        VISIBLE_BLOCKS = 1500,
        BLOCKS = 2048,
        HEADS = 4,
        DIM = 128,
        TOP_K = 64,
        BLOCK_TILE = 256,
    };
    float *q = malloc((size_t)QUERIES * HEADS * DIM * sizeof(float));
    float *pooled = malloc((size_t)BLOCKS * DIM * sizeof(float));
    uint32_t visible[QUERIES] = {1, 511, 1024, VISIBLE_BLOCKS, 777};
    uint32_t *indices = malloc((size_t)QUERIES * TOP_K * sizeof(uint32_t));
    float *scores = malloc((size_t)QUERIES * TOP_K * sizeof(float));
    uint32_t counts[QUERIES];
    ranked_score *reference = malloc((size_t)BLOCKS * sizeof(*reference));
    require_ok(q && pooled && indices && scores && reference,
               "streaming QSA host allocation");
    for (uint32_t i = 0; i < QUERIES * HEADS * DIM; i++)
        q[i] = 0.0025f * (float)((int)((i * 11u + 5u) % 53u) - 26);
    for (uint32_t b = 0; b < BLOCKS; b++)
        for (uint32_t d = 0; d < DIM; d++)
            pooled[(size_t)b * DIM + d] =
                0.00175f * (float)((int)((b * 29u + d * 7u) % 101u) - 50) +
                2e-6f * (float)b *
                    (float)((int)((d * 5u + 3u) % 11u) - 5);

    ds4_gpu_tensor *qt = tensor_from(
        q, (size_t)QUERIES * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *kt = tensor_from(
        pooled, (size_t)BLOCKS * DIM * sizeof(float));
    uint16_t *pooled_bf16 = malloc((size_t)BLOCKS * DIM * sizeof(uint16_t));
    float *pooled_bf16_f32 = malloc((size_t)BLOCKS * DIM * sizeof(float));
    require_ok(pooled_bf16 && pooled_bf16_f32,
               "streaming QSA BF16 host allocation");
    for (size_t i = 0; i < (size_t)BLOCKS * DIM; i++) {
        pooled_bf16[i] = f32_to_bf16(pooled[i]);
        pooled_bf16_f32[i] = bf16_to_f32(pooled_bf16[i]);
    }
    ds4_gpu_tensor *kt_bf16 = tensor_from(
        pooled_bf16, (size_t)BLOCKS * DIM * sizeof(uint16_t));
    ds4_gpu_tensor *vt = tensor_from(visible, sizeof(visible));
    ds4_gpu_tensor *score_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * TOP_K * sizeof(float));
    ds4_gpu_tensor *index_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * TOP_K * sizeof(uint32_t));
    ds4_gpu_tensor *count_t = ds4_gpu_tensor_alloc(sizeof(counts));
    ds4_gpu_tensor *tile_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * BLOCK_TILE * sizeof(float));
    require_ok(score_t && index_t && count_t && tile_t,
               "streaming QSA device allocation");
    require_ok(ds4_gpu_qwen4_qsa_stream_topk(
                   score_t, index_t, count_t, tile_t, qt, kt, vt,
                   QUERIES, BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "streaming QSA dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)),
               "streaming QSA score readback");
    require_ok(ds4_gpu_tensor_read(index_t, 0, indices,
                                   (size_t)QUERIES * TOP_K * sizeof(uint32_t)),
               "streaming QSA index readback");
    require_ok(ds4_gpu_tensor_read(count_t, 0, counts, sizeof(counts)),
               "streaming QSA count readback");
    float *padded_scores = malloc(
        (size_t)QUERIES * TOP_K * sizeof(float));
    uint32_t *padded_indices = malloc(
        (size_t)QUERIES * TOP_K * sizeof(uint32_t));
    uint32_t padded_counts[QUERIES];
    require_ok(padded_scores && padded_indices,
               "streaming QSA padded-capacity host allocation");
    memcpy(padded_scores, scores,
           (size_t)QUERIES * TOP_K * sizeof(float));
    memcpy(padded_indices, indices,
           (size_t)QUERIES * TOP_K * sizeof(uint32_t));
    memcpy(padded_counts, counts, sizeof(counts));
    require_ok(ds4_gpu_qwen4_qsa_stream_topk(
                   score_t, index_t, count_t, tile_t, qt, kt, vt,
                   QUERIES, VISIBLE_BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "streaming QSA truncated-capacity dispatch");
    require_ok(ds4_gpu_tensor_read(
                   score_t, 0, scores,
                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
               ds4_gpu_tensor_read(
                   index_t, 0, indices,
                   (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
               ds4_gpu_tensor_read(count_t, 0, counts, sizeof(counts)),
               "streaming QSA truncated-capacity readback");
    require_ok(memcmp(padded_scores, scores,
                      (size_t)QUERIES * TOP_K * sizeof(float)) == 0 &&
                   memcmp(padded_indices, indices,
                          (size_t)QUERIES * TOP_K * sizeof(uint32_t)) == 0 &&
                   memcmp(padded_counts, counts, sizeof(counts)) == 0,
               "streaming QSA masked-capacity invariance");

    for (uint32_t query = 0; query < QUERIES; query++) {
        for (uint32_t block = 0; block < visible[query]; block++) {
            float score = 0.0f;
            for (uint32_t h = 0; h < HEADS; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < DIM; d++)
                    dot = fmaf(q[((size_t)query * HEADS + h) * DIM + d],
                               pooled[(size_t)block * DIM + d], dot);
                score += fmaxf(dot, 0.0f);
            }
            reference[block] = (ranked_score){
                .score = score / sqrtf((float)DIM),
                .index = block,
            };
        }
        qsort(reference, visible[query], sizeof(*reference), ranked_desc);
        const uint32_t expected_count = visible[query] < TOP_K
            ? visible[query] : TOP_K;
        require_ok(counts[query] == expected_count,
                   "streaming QSA heap count");
        for (uint32_t rank = 0; rank < expected_count; rank++) {
            const size_t at = (size_t)query * TOP_K + rank;
            if (indices[at] != reference[rank].index) {
                fprintf(stderr,
                        "FAIL: streaming QSA q=%u rank=%u got=%u expected=%u\n",
                        query, rank, indices[at], reference[rank].index);
                exit(1);
            }
            const float tolerance =
                2e-6f + 2e-4f * fabsf(reference[rank].score);
            if (fabsf(scores[at] - reference[rank].score) > tolerance) {
                fprintf(stderr,
                        "FAIL: streaming QSA score q=%u rank=%u got=%g expected=%g\n",
                        query, rank, scores[at], reference[rank].score);
                exit(1);
            }
        }
    }
    puts("Qwen 512-query-microtile streaming QSA ordered top-k PASS");

    /* The batched scorer is the default; the BF16 baselines below capture
     * the original grid's outputs with the restage disabled. */
    require_ok(setenv("DS4_QWEN4_QSA_SCORE_BATCHED", "0", 1) == 0,
               "streaming BF16 baseline scorer env");
    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "streaming BF16 QSA dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)),
               "streaming BF16 QSA score readback");
    require_ok(ds4_gpu_tensor_read(index_t, 0, indices,
                                   (size_t)QUERIES * TOP_K * sizeof(uint32_t)),
               "streaming BF16 QSA index readback");
    require_ok(ds4_gpu_tensor_read(count_t, 0, counts, sizeof(counts)),
               "streaming BF16 QSA count readback");
    memcpy(padded_scores, scores,
           (size_t)QUERIES * TOP_K * sizeof(float));
    memcpy(padded_indices, indices,
           (size_t)QUERIES * TOP_K * sizeof(uint32_t));
    memcpy(padded_counts, counts, sizeof(counts));
    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, VISIBLE_BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "streaming BF16 QSA truncated-capacity dispatch");
    require_ok(ds4_gpu_tensor_read(
                   score_t, 0, scores,
                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
               ds4_gpu_tensor_read(
                   index_t, 0, indices,
                   (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
               ds4_gpu_tensor_read(count_t, 0, counts, sizeof(counts)),
               "streaming BF16 QSA truncated-capacity readback");
    require_ok(memcmp(padded_scores, scores,
                      (size_t)QUERIES * TOP_K * sizeof(float)) == 0 &&
                   memcmp(padded_indices, indices,
                          (size_t)QUERIES * TOP_K * sizeof(uint32_t)) == 0 &&
                   memcmp(padded_counts, counts, sizeof(counts)) == 0,
               "streaming BF16 QSA masked-capacity invariance");

    /* The batched long-context scorer restage must keep every ordered
     * score, index, and count byte-identical to the original grid; the
     * BF16 scalar outputs captured above are the baseline, and the env is
     * honored per dispatch call.  The truncated-capacity rerun exercises
     * the batched grid's partial 32-block tail guards. */
    require_ok(setenv("DS4_QWEN4_QSA_SCORE_BATCHED", "1", 1) == 0,
               "streaming BF16 batched env");
    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "streaming BF16 batched dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
                   ds4_gpu_tensor_read(
                       index_t, 0, indices,
                       (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                   ds4_gpu_tensor_read(count_t, 0, counts, sizeof(counts)),
               "streaming BF16 batched readback");
    require_ok(memcmp(padded_scores, scores,
                      (size_t)QUERIES * TOP_K * sizeof(float)) == 0 &&
                   memcmp(padded_indices, indices,
                          (size_t)QUERIES * TOP_K * sizeof(uint32_t)) == 0 &&
                   memcmp(padded_counts, counts, sizeof(counts)) == 0,
               "streaming BF16 batched scorer byte identity");
    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, VISIBLE_BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "streaming BF16 batched truncated dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
                   ds4_gpu_tensor_read(
                       index_t, 0, indices,
                       (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                   ds4_gpu_tensor_read(count_t, 0, counts, sizeof(counts)),
               "streaming BF16 batched truncated readback");
    require_ok(memcmp(padded_scores, scores,
                      (size_t)QUERIES * TOP_K * sizeof(float)) == 0 &&
                   memcmp(padded_indices, indices,
                          (size_t)QUERIES * TOP_K * sizeof(uint32_t)) == 0 &&
                   memcmp(padded_counts, counts, sizeof(counts)) == 0,
               "streaming BF16 batched masked-capacity invariance");
    require_ok(unsetenv("DS4_QWEN4_QSA_SCORE_BATCHED") == 0,
               "streaming BF16 batched env clear");

    for (uint32_t query = 0; query < QUERIES; query++) {
        for (uint32_t block = 0; block < visible[query]; block++) {
            float score = 0.0f;
            for (uint32_t h = 0; h < HEADS; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < DIM; d++)
                    dot = fmaf(q[((size_t)query * HEADS + h) * DIM + d],
                               pooled_bf16_f32[(size_t)block * DIM + d], dot);
                score += fmaxf(dot, 0.0f);
            }
            reference[block] = (ranked_score){
                .score = score / sqrtf((float)DIM),
                .index = block,
            };
        }
        qsort(reference, visible[query], sizeof(*reference), ranked_desc);
        const uint32_t expected_count = visible[query] < TOP_K
            ? visible[query] : TOP_K;
        require_ok(counts[query] == expected_count,
                   "streaming BF16 QSA heap count");
        for (uint32_t rank = 0; rank < expected_count; rank++) {
            const size_t at = (size_t)query * TOP_K + rank;
            require_ok(indices[at] == reference[rank].index,
                       "streaming BF16 QSA ordered index");
            const float tolerance =
                2e-6f + 2e-4f * fabsf(reference[rank].score);
            require_ok(fabsf(scores[at] - reference[rank].score) <= tolerance,
                       "streaming BF16 QSA score parity");
        }
    }
    puts("Qwen BF16-cache streaming QSA ordered top-k PASS");
    free(padded_indices);
    free(padded_scores);
    ds4_gpu_tensor_free(kt_bf16);
    free(pooled_bf16_f32);
    free(pooled_bf16);
    ds4_gpu_tensor_free(tile_t);
    ds4_gpu_tensor_free(count_t);
    ds4_gpu_tensor_free(index_t);
    ds4_gpu_tensor_free(score_t);
    ds4_gpu_tensor_free(vt);
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(qt);
    free(reference);
    free(scores);
    free(indices);
    free(pooled);
    free(q);
}

static void test_qsa_streaming_topk_merge_select(void) {
    /* Production-geometry companion to test_qsa_streaming_topk: keep 512
     * with 1024-block tiles is the shape the threshold-filtered
     * merge-select kernel is built for.  Every tile's ordered output must
     * be byte-identical to the full-sort bitonic merge, including the
     * first tile (which takes the in-kernel full-sort fallback) and the
     * partial tail tile, and must match the CPU reference. */
    enum {
        QUERIES = 4,
        BLOCKS = 4096,
        TRUNCATED_BLOCKS = 2500,
        HEADS = 4,
        DIM = 128,
        TOP_K = 512,
        BLOCK_TILE = 1024,
    };
    float *q = malloc((size_t)QUERIES * HEADS * DIM * sizeof(float));
    float *pooled = malloc((size_t)BLOCKS * DIM * sizeof(float));
    uint32_t visible[QUERIES] = {BLOCKS, TRUNCATED_BLOCKS, 1024, 513};
    uint32_t *indices = malloc((size_t)QUERIES * TOP_K * sizeof(uint32_t));
    float *scores = malloc((size_t)QUERIES * TOP_K * sizeof(float));
    uint32_t counts[QUERIES];
    ranked_score *reference = malloc((size_t)BLOCKS * sizeof(*reference));
    require_ok(q && pooled && indices && scores && reference,
               "merge-select QSA host allocation");
    for (uint32_t i = 0; i < QUERIES * HEADS * DIM; i++)
        q[i] = 0.0031f * (float)((int)((i * 13u + 7u) % 61u) - 30);
    for (uint32_t b = 0; b < BLOCKS; b++)
        for (uint32_t d = 0; d < DIM; d++)
            pooled[(size_t)b * DIM + d] =
                0.0019f * (float)((int)((b * 31u + d * 11u) % 103u) - 51) +
                3e-6f * (float)b *
                    (float)((int)((d * 7u + 1u) % 13u) - 6);
    uint16_t *pooled_bf16 = malloc((size_t)BLOCKS * DIM * sizeof(uint16_t));
    float *pooled_bf16_f32 = malloc((size_t)BLOCKS * DIM * sizeof(float));
    require_ok(pooled_bf16 && pooled_bf16_f32,
               "merge-select QSA BF16 host allocation");
    for (size_t i = 0; i < (size_t)BLOCKS * DIM; i++) {
        pooled_bf16[i] = f32_to_bf16(pooled[i]);
        pooled_bf16_f32[i] = bf16_to_f32(pooled_bf16[i]);
    }
    ds4_gpu_tensor *qt = tensor_from(
        q, (size_t)QUERIES * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *kt_bf16 = tensor_from(
        pooled_bf16, (size_t)BLOCKS * DIM * sizeof(uint16_t));
    ds4_gpu_tensor *vt = tensor_from(visible, sizeof(visible));
    ds4_gpu_tensor *score_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * TOP_K * sizeof(float));
    ds4_gpu_tensor *index_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * TOP_K * sizeof(uint32_t));
    ds4_gpu_tensor *count_t = ds4_gpu_tensor_alloc(sizeof(counts));
    ds4_gpu_tensor *tile_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * BLOCK_TILE * sizeof(float));
    require_ok(qt && kt_bf16 && vt && score_t && index_t && count_t &&
                   tile_t,
               "merge-select QSA device allocation");
    float *baseline_scores = malloc(
        (size_t)QUERIES * TOP_K * sizeof(float));
    uint32_t *baseline_indices = malloc(
        (size_t)QUERIES * TOP_K * sizeof(uint32_t));
    uint32_t baseline_counts[QUERIES];
    float *full_scores = malloc((size_t)QUERIES * TOP_K * sizeof(float));
    uint32_t *full_indices = malloc(
        (size_t)QUERIES * TOP_K * sizeof(uint32_t));
    uint32_t full_counts[QUERIES];
    require_ok(baseline_scores && baseline_indices && full_scores &&
                   full_indices,
               "merge-select baseline host allocation");

    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "merge-select baseline dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, baseline_scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
                   ds4_gpu_tensor_read(
                       index_t, 0, baseline_indices,
                       (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                   ds4_gpu_tensor_read(count_t, 0, baseline_counts,
                                       sizeof(baseline_counts)),
               "merge-select baseline readback");

    for (int pass = 0; pass < 2; pass++) {
        const uint32_t blocks = pass == 0 ? BLOCKS : TRUNCATED_BLOCKS;
        /* Kernel-equivalence baseline at this capacity: the full-sort
         * bitonic merge with the merge-select env cleared.  Outputs are
         * NOT shared across capacities (q0 sees 4096 blocks in one run
         * and 2500 in the other, which selects different top-512s), so
         * each pass re-runs its own baseline. */
        require_ok(setenv("DS4_QWEN4_QSA_MERGE_SELECT", "0", 1) == 0,
                   "merge-select baseline env");
        require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                       score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                       QUERIES, blocks, HEADS, DIM, TOP_K, BLOCK_TILE),
                   "merge-select baseline dispatch");
        require_ok(ds4_gpu_tensor_read(
                       score_t, 0, baseline_scores,
                       (size_t)QUERIES * TOP_K * sizeof(float)) &&
                       ds4_gpu_tensor_read(
                           index_t, 0, baseline_indices,
                           (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                       ds4_gpu_tensor_read(count_t, 0, baseline_counts,
                                           sizeof(baseline_counts)),
                   "merge-select baseline readback");
        if (pass == 0) {
            /* Preserve the full-capacity outputs for the CPU-reference
             * check below (the truncated pass legitimately differs). */
            memcpy(full_scores, baseline_scores,
                   (size_t)QUERIES * TOP_K * sizeof(float));
            memcpy(full_indices, baseline_indices,
                   (size_t)QUERIES * TOP_K * sizeof(uint32_t));
            memcpy(full_counts, baseline_counts, sizeof(baseline_counts));
        }
        require_ok(setenv("DS4_QWEN4_QSA_MERGE_SELECT", "1", 1) == 0,
                   "merge-select env");
        require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                       score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                       QUERIES, blocks, HEADS, DIM, TOP_K, BLOCK_TILE),
                   "merge-select dispatch");
        require_ok(ds4_gpu_tensor_read(
                       score_t, 0, scores,
                       (size_t)QUERIES * TOP_K * sizeof(float)) &&
                       ds4_gpu_tensor_read(
                           index_t, 0, indices,
                           (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                       ds4_gpu_tensor_read(count_t, 0, counts,
                                           sizeof(counts)),
                   "merge-select readback");
        char label[64];
        snprintf(label, sizeof(label),
                 "merge-select byte identity blocks=%u", blocks);
        require_ok(memcmp(baseline_scores, scores,
                          (size_t)QUERIES * TOP_K * sizeof(float)) == 0 &&
                       memcmp(baseline_indices, indices,
                              (size_t)QUERIES * TOP_K *
                                  sizeof(uint32_t)) == 0 &&
                       memcmp(baseline_counts, counts,
                              sizeof(baseline_counts)) == 0,
                   label);
    }
    require_ok(unsetenv("DS4_QWEN4_QSA_MERGE_SELECT") == 0,
               "merge-select env clear");

    for (uint32_t query = 0; query < QUERIES; query++) {
        for (uint32_t block = 0; block < visible[query]; block++) {
            float score = 0.0f;
            for (uint32_t h = 0; h < HEADS; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < DIM; d++)
                    dot = fmaf(q[((size_t)query * HEADS + h) * DIM + d],
                               pooled_bf16_f32[(size_t)block * DIM + d], dot);
                score += fmaxf(dot, 0.0f);
            }
            reference[block] = (ranked_score){
                .score = score / sqrtf((float)DIM),
                .index = block,
            };
        }
        qsort(reference, visible[query], sizeof(*reference), ranked_desc);
        const uint32_t expected_count = visible[query] < TOP_K
            ? visible[query] : TOP_K;
        require_ok(full_counts[query] == expected_count,
                   "merge-select count");
        for (uint32_t rank = 0; rank < expected_count; rank++) {
            const size_t at = (size_t)query * TOP_K + rank;
            require_ok(full_indices[at] == reference[rank].index,
                       "merge-select ordered index");
            const float tolerance =
                2e-6f + 2e-4f * fabsf(reference[rank].score);
            require_ok(fabsf(full_scores[at] - reference[rank].score) <=
                           tolerance,
                       "merge-select score parity");
        }
    }
    puts("Qwen streaming QSA merge-select byte identity PASS");
    free(full_indices);
    free(full_scores);
    free(baseline_indices);
    free(baseline_scores);
    ds4_gpu_tensor_free(tile_t);
    ds4_gpu_tensor_free(count_t);
    ds4_gpu_tensor_free(index_t);
    ds4_gpu_tensor_free(score_t);
    ds4_gpu_tensor_free(vt);
    ds4_gpu_tensor_free(kt_bf16);
    free(pooled_bf16_f32);
    free(pooled_bf16);
    free(reference);
    free(scores);
    free(indices);
    free(pooled);
    free(q);
}

static void test_qsa_streaming_topk_score_mm(void) {
    /* Tensor-core index-scorer fixture.  The MMA kernel stages the F32
     * query vectors and the BF16 pooled keys to F16 tiles (the mul_mm
     * operand-rounding policy) and accumulates F32 on the matrix units,
     * so the reference models exactly that rounding and accepts the
     * accumulation-order slack as tolerance; rank swaps are allowed only
     * between reference scores within the swap tolerance.  Geometry
     * covers a partial M-tile (70 queries = 280 head rows), ragged merge
     * tiles (2548 = 1024+1024+500, 1025 = 1024+1), masked-capacity
     * invariance across different tail raggedness, and the below-
     * threshold guard (5 queries must stay on the scalar kernel,
     * byte-identical). */
    enum {
        QUERIES = 70,
        BLOCKS = 2548,
        CAP_BLOCKS = 1025,
        HEADS = 4,
        DIM = 128,
        TOP_K = 512,
        BLOCK_TILE = 1024,
        SMALL_QUERIES = 5,
        SMALL_BLOCKS = 37,
        SMALL_TOP_K = 8,
    };
    float *q = malloc((size_t)QUERIES * HEADS * DIM * sizeof(float));
    float *pooled = malloc((size_t)BLOCKS * DIM * sizeof(float));
    uint32_t visible[QUERIES];
    const uint32_t patterns[] = {BLOCKS, 2049, 1500, 1025, 513, 100, 1};
    for (uint32_t query = 0; query < QUERIES; query++)
        visible[query] = patterns[query % 7u];
    uint32_t *indices = malloc((size_t)QUERIES * TOP_K * sizeof(uint32_t));
    float *scores = malloc((size_t)QUERIES * TOP_K * sizeof(float));
    uint32_t counts[QUERIES];
    ranked_score *reference = malloc((size_t)BLOCKS * sizeof(*reference));
    float *scalar_scores = malloc((size_t)QUERIES * TOP_K * sizeof(float));
    uint32_t *scalar_indices =
        malloc((size_t)QUERIES * TOP_K * sizeof(uint32_t));
    uint32_t scalar_counts[QUERIES];
    float *cap_scores = malloc((size_t)QUERIES * TOP_K * sizeof(float));
    uint32_t *cap_indices =
        malloc((size_t)QUERIES * TOP_K * sizeof(uint32_t));
    uint32_t cap_counts[QUERIES];
    require_ok(q && pooled && indices && scores && reference &&
                   scalar_scores && scalar_indices && cap_scores &&
                   cap_indices,
               "score-mm QSA host allocation");
    for (size_t i = 0; i < (size_t)QUERIES * HEADS * DIM; i++)
        q[i] = 0.0021f * (float)((int)((i * 17u + 3u) % 59u) - 29);
    for (uint32_t b = 0; b < BLOCKS; b++)
        for (uint32_t d = 0; d < DIM; d++)
            pooled[(size_t)b * DIM + d] =
                0.0016f * (float)((int)((b * 37u + d * 13u) % 107u) - 53) +
                3e-6f * (float)b *
                    (float)((int)((d * 3u + 2u) % 9u) - 4);
    uint16_t *pooled_bf16 = malloc((size_t)BLOCKS * DIM * sizeof(uint16_t));
    require_ok(pooled_bf16 != NULL, "score-mm QSA BF16 host allocation");
    for (size_t i = 0; i < (size_t)BLOCKS * DIM; i++)
        pooled_bf16[i] = f32_to_bf16(pooled[i]);

    ds4_gpu_tensor *qt = tensor_from(
        q, (size_t)QUERIES * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *kt_bf16 = tensor_from(
        pooled_bf16, (size_t)BLOCKS * DIM * sizeof(uint16_t));
    ds4_gpu_tensor *vt = tensor_from(visible, sizeof(visible));
    ds4_gpu_tensor *score_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * TOP_K * sizeof(float));
    ds4_gpu_tensor *index_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * TOP_K * sizeof(uint32_t));
    ds4_gpu_tensor *count_t = ds4_gpu_tensor_alloc(sizeof(counts));
    ds4_gpu_tensor *tile_t = ds4_gpu_tensor_alloc(
        (size_t)QUERIES * BLOCK_TILE * sizeof(float));
    require_ok(qt && kt_bf16 && vt && score_t && index_t && count_t &&
                   tile_t,
               "score-mm QSA device allocation");

    const char *prior_raw = getenv("DS4_QWEN4_QSA_SCORE_MM");
    char *prior = prior_raw ? strdup(prior_raw) : NULL;
    require_ok(!prior_raw || prior, "score-mm environment copy");

    /* Scalar (batched) baseline at full capacity. */
    require_ok(setenv("DS4_QWEN4_QSA_SCORE_MM", "0", 1) == 0,
               "score-mm scalar env");
    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "score-mm scalar dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, scalar_scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
                   ds4_gpu_tensor_read(
                       index_t, 0, scalar_indices,
                       (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                   ds4_gpu_tensor_read(count_t, 0, scalar_counts,
                                       sizeof(scalar_counts)),
               "score-mm scalar readback");

    /* MMA path at full capacity vs the F16-rounding reference. */
    require_ok(setenv("DS4_QWEN4_QSA_SCORE_MM", "1", 1) == 0,
               "score-mm enable");
    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "score-mm dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
                   ds4_gpu_tensor_read(
                       index_t, 0, indices,
                       (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                   ds4_gpu_tensor_read(count_t, 0, counts, sizeof(counts)),
               "score-mm readback");

    const float score_tol_abs = 2e-4f;
    const float score_tol_rel = 6e-3f;
    const float swap_tol = 8e-3f;
    uint64_t exact_ranks = 0;
    uint64_t compared_ranks = 0;
    for (uint32_t query = 0; query < QUERIES; query++) {
        for (uint32_t block = 0; block < visible[query]; block++) {
            float score = 0.0f;
            for (uint32_t h = 0; h < HEADS; h++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < DIM; d++)
                    dot += (float)(_Float16)
                              q[((size_t)query * HEADS + h) * DIM + d] *
                          (float)(_Float16)
                              bf16_to_f32(
                                  pooled_bf16[(size_t)block * DIM + d]);
                score += fmaxf(dot, 0.0f);
            }
            reference[block] = (ranked_score){
                .score = score / sqrtf((float)DIM),
                .index = block,
            };
        }
        qsort(reference, visible[query], sizeof(*reference), ranked_desc);
        const uint32_t expected_count = visible[query] < TOP_K
            ? visible[query] : TOP_K;
        require_ok(counts[query] == expected_count,
                   "score-mm count");
        require_ok(scalar_counts[query] == expected_count,
                   "score-mm scalar count");
        for (uint32_t rank = 0; rank < expected_count; rank++) {
            const size_t at = (size_t)query * TOP_K + rank;
            require_ok(indices[at] < visible[query],
                       "score-mm index visibility");
            const float tol = score_tol_abs +
                score_tol_rel * fabsf(reference[rank].score);
            char label[96];
            if (indices[at] == reference[rank].index) {
                exact_ranks++;
            } else {
                /* Rank swap: the returned block must sit within the swap
                 * tolerance of the reference score at this rank, and the
                 * returned score must match that block's reference. */
                float swapped = 0.0f;
                bool found = false;
                for (uint32_t b = 0; b < visible[query]; b++)
                    if (reference[b].index == indices[at]) {
                        swapped = reference[b].score;
                        found = true;
                        break;
                    }
                (void)found; /* qsort keeps every visible block */
                snprintf(label, sizeof(label),
                         "score-mm swap q=%u rank=%u", query, rank);
                require_ok(fabsf(swapped - reference[rank].score) <=
                               swap_tol,
                           label);
            }
            compared_ranks++;
            snprintf(label, sizeof(label),
                     "score-mm score q=%u rank=%u got=%g expected=%g",
                     query, rank, scores[at], reference[rank].score);
            require_ok(fabsf(scores[at] - reference[rank].score) <= tol,
                       label);
        }
    }
    /* The F16 rounding is tiny relative to the fixture's score spread:
     * the overwhelming majority of ranks must be exact, and every MMA
     * score must stay in the scalar kernel's neighborhood at the same
     * rank. */
    require_ok(exact_ranks * 100u >= compared_ranks * 95u,
               "score-mm rank agreement with F16 reference");
    /* Observable engagement: the F16 staging must perturb at least some
     * scores relative to the scalar kernel — a path that silently never
     * dispatched would be byte-identical and must fail here. */
    {
        uint64_t changed = 0;
        for (size_t i = 0; i < (size_t)QUERIES * TOP_K; i++)
            if (scores[i] != scalar_scores[i]) changed++;
        require_ok(changed > (uint64_t)QUERIES * TOP_K / 10u,
                   "score-mm observable engagement");
    }
    for (uint32_t query = 0; query < QUERIES; query++) {
        const uint32_t expected_count = visible[query] < TOP_K
            ? visible[query] : TOP_K;
        for (uint32_t rank = 0; rank < expected_count; rank++) {
            const size_t at = (size_t)query * TOP_K + rank;
            char label[96];
            snprintf(label, sizeof(label),
                     "score-mm scalar delta q=%u rank=%u", query, rank);
            require_ok(fabsf(scores[at] - scalar_scores[at]) <= swap_tol,
                       label);
        }
    }

    /* Masked-capacity invariance: with every query's visible count
     * clamped under CAP_BLOCKS, the MMA outputs must be byte-identical
     * between blocks=CAP_BLOCKS (1024+1 ragged tail) and blocks=BLOCKS
     * (1024+1024+500) — different tile tails, same visible content. */
    for (uint32_t query = 0; query < QUERIES; query++)
        visible[query] = visible[query] < CAP_BLOCKS
            ? visible[query] : CAP_BLOCKS;
    require_ok(ds4_gpu_tensor_write(vt, 0, visible, sizeof(visible)) != 0,
               "score-mm capacity writeback");
    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "score-mm capacity full dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, cap_scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
                   ds4_gpu_tensor_read(
                       index_t, 0, cap_indices,
                       (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                   ds4_gpu_tensor_read(count_t, 0, cap_counts,
                                       sizeof(cap_counts)),
               "score-mm capacity full readback");
    require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                   score_t, index_t, count_t, tile_t, qt, kt_bf16, vt,
                   QUERIES, CAP_BLOCKS, HEADS, DIM, TOP_K, BLOCK_TILE),
               "score-mm capacity truncated dispatch");
    require_ok(ds4_gpu_tensor_read(score_t, 0, scores,
                                   (size_t)QUERIES * TOP_K * sizeof(float)) &&
                   ds4_gpu_tensor_read(
                       index_t, 0, indices,
                       (size_t)QUERIES * TOP_K * sizeof(uint32_t)) &&
                   ds4_gpu_tensor_read(count_t, 0, counts, sizeof(counts)),
               "score-mm capacity truncated readback");
    require_ok(memcmp(cap_scores, scores,
                      (size_t)QUERIES * TOP_K * sizeof(float)) == 0 &&
                   memcmp(cap_indices, indices,
                          (size_t)QUERIES * TOP_K * sizeof(uint32_t)) == 0 &&
                   memcmp(cap_counts, counts, sizeof(cap_counts)) == 0,
               "score-mm masked-capacity invariance");

    /* Below the query threshold the MMA env must be ignored: outputs
     * byte-identical to the scalar kernel on a small ragged geometry
     * (37 blocks = one 32+5 tile, 5 queries = 20 head rows). */
    {
        float *small_q =
            malloc((size_t)SMALL_QUERIES * HEADS * DIM * sizeof(float));
        uint32_t small_visible[SMALL_QUERIES] = {
            SMALL_BLOCKS, 33, 8, 1, 17};
        float *small_scores =
            malloc((size_t)SMALL_QUERIES * SMALL_TOP_K * sizeof(float));
        float *small_scalar =
            malloc((size_t)SMALL_QUERIES * SMALL_TOP_K * sizeof(float));
        uint32_t *small_indices = malloc(
            (size_t)SMALL_QUERIES * SMALL_TOP_K * sizeof(uint32_t));
        uint32_t *small_scalar_idx = malloc(
            (size_t)SMALL_QUERIES * SMALL_TOP_K * sizeof(uint32_t));
        uint32_t small_counts[SMALL_QUERIES];
        uint32_t small_scalar_counts[SMALL_QUERIES];
        require_ok(small_q && small_scores && small_scalar &&
                       small_indices && small_scalar_idx,
                   "score-mm small host allocation");
        for (size_t i = 0;
             i < (size_t)SMALL_QUERIES * HEADS * DIM; i++)
            small_q[i] = 0.0017f * (float)((int)((i * 23u + 11u) % 61u) - 30);
        ds4_gpu_tensor *small_qt = tensor_from(
            small_q, (size_t)SMALL_QUERIES * HEADS * DIM * sizeof(float));
        ds4_gpu_tensor *small_vt =
            tensor_from(small_visible, sizeof(small_visible));
        ds4_gpu_tensor *small_score_t = ds4_gpu_tensor_alloc(
            (size_t)SMALL_QUERIES * SMALL_TOP_K * sizeof(float));
        ds4_gpu_tensor *small_index_t = ds4_gpu_tensor_alloc(
            (size_t)SMALL_QUERIES * SMALL_TOP_K * sizeof(uint32_t));
        ds4_gpu_tensor *small_count_t =
            ds4_gpu_tensor_alloc(sizeof(small_counts));
        ds4_gpu_tensor *small_tile_t = ds4_gpu_tensor_alloc(
            (size_t)SMALL_QUERIES * BLOCK_TILE * sizeof(float));
        require_ok(small_qt && small_vt && small_score_t &&
                       small_index_t && small_count_t && small_tile_t,
                   "score-mm small device allocation");
        require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                       small_score_t, small_index_t, small_count_t,
                       small_tile_t, small_qt, kt_bf16, small_vt,
                       SMALL_QUERIES, SMALL_BLOCKS, HEADS, DIM,
                       SMALL_TOP_K, BLOCK_TILE),
                   "score-mm small MMA-env dispatch");
        require_ok(ds4_gpu_tensor_read(
                       small_score_t, 0, small_scores,
                       (size_t)SMALL_QUERIES * SMALL_TOP_K *
                           sizeof(float)) &&
                       ds4_gpu_tensor_read(
                           small_index_t, 0, small_indices,
                           (size_t)SMALL_QUERIES * SMALL_TOP_K *
                               sizeof(uint32_t)) &&
                       ds4_gpu_tensor_read(
                           small_count_t, 0, small_counts,
                           sizeof(small_counts)),
                   "score-mm small readback");
        require_ok(setenv("DS4_QWEN4_QSA_SCORE_MM", "0", 1) == 0,
                   "score-mm small scalar env");
        require_ok(ds4_gpu_qwen4_qsa_stream_topk_bf16(
                       small_score_t, small_index_t, small_count_t,
                       small_tile_t, small_qt, kt_bf16, small_vt,
                       SMALL_QUERIES, SMALL_BLOCKS, HEADS, DIM,
                       SMALL_TOP_K, BLOCK_TILE),
                   "score-mm small scalar dispatch");
        require_ok(ds4_gpu_tensor_read(
                       small_score_t, 0, small_scalar,
                       (size_t)SMALL_QUERIES * SMALL_TOP_K *
                           sizeof(float)) &&
                       ds4_gpu_tensor_read(
                           small_index_t, 0, small_scalar_idx,
                           (size_t)SMALL_QUERIES * SMALL_TOP_K *
                               sizeof(uint32_t)) &&
                       ds4_gpu_tensor_read(
                           small_count_t, 0, small_scalar_counts,
                           sizeof(small_scalar_counts)),
                   "score-mm small scalar readback");
        require_ok(memcmp(small_scores, small_scalar,
                          (size_t)SMALL_QUERIES * SMALL_TOP_K *
                              sizeof(float)) == 0 &&
                       memcmp(small_indices, small_scalar_idx,
                              (size_t)SMALL_QUERIES * SMALL_TOP_K *
                                  sizeof(uint32_t)) == 0 &&
                       memcmp(small_counts, small_scalar_counts,
                              sizeof(small_counts)) == 0,
                   "score-mm threshold guard byte identity");
        ds4_gpu_tensor_free(small_tile_t);
        ds4_gpu_tensor_free(small_count_t);
        ds4_gpu_tensor_free(small_index_t);
        ds4_gpu_tensor_free(small_score_t);
        ds4_gpu_tensor_free(small_vt);
        ds4_gpu_tensor_free(small_qt);
        free(small_scalar_idx);
        free(small_indices);
        free(small_scalar);
        free(small_scores);
        free(small_q);
    }

    require_ok(prior
                   ? setenv("DS4_QWEN4_QSA_SCORE_MM", prior, 1) == 0
                   : unsetenv("DS4_QWEN4_QSA_SCORE_MM") == 0,
               "score-mm environment restore");
    free(prior);
    puts("Qwen streaming QSA tensor-core scorer PASS");
    ds4_gpu_tensor_free(tile_t);
    ds4_gpu_tensor_free(count_t);
    ds4_gpu_tensor_free(index_t);
    ds4_gpu_tensor_free(score_t);
    ds4_gpu_tensor_free(vt);
    ds4_gpu_tensor_free(kt_bf16);
    free(pooled_bf16);
    free(cap_indices);
    free(cap_scores);
    free(scalar_indices);
    free(scalar_scores);
    free(reference);
    free(scores);
    free(indices);
    free(pooled);
    free(q);
}

int main(void) {
    /* Opt the fixture suite into the Metal 4 TensorOps route wherever the
     * platform can compile it: M5-class GPUs get the real TensorOps units
     * and pre-M5 machines exercise the portable fallback, so the mul_mm and
     * grouped-MoE fixtures pin parity for the accelerated prefill kernels on
     * every machine.  An explicit DS4_METAL_ENABLE_TENSOR value (including
     * =0) is respected; DS4_QWEN4_TEST_DISABLE_TENSOR_ROUTE=1 also keeps
     * this run on the legacy kernels only. */
    if (getenv("DS4_QWEN4_TEST_DISABLE_TENSOR_ROUTE") == NULL) {
        const char *raw = getenv("DS4_METAL_ENABLE_TENSOR");
        if (!raw || raw[0] == '\0') {
            setenv("DS4_METAL_ENABLE_TENSOR", "1", 1);
        }
    }
    require_ok(ds4_gpu_init(), "Metal initialization");
    test_fast_path_capabilities();
    if (getenv("DS4_QWEN4_DENSE_MATVEC_BENCH") != NULL) {
        benchmark_dense_q8_vs_q4();
        ds4_gpu_cleanup();
        free(dense_matvec_bench_allocation);
        return 0;
    }
    if (getenv("DS4_QWEN4_MOE_Q4_BENCH") != NULL) {
        benchmark_moe_q4_k_vs_q4_0();
        ds4_gpu_cleanup();
        free(dense_matvec_bench_allocation);
        return 0;
    }
    if (getenv("DS4_QWEN4_GDN_BENCH") != NULL) {
        benchmark_gdn_decode();
        ds4_gpu_cleanup();
        return 0;
    }
    if (getenv("DS4_QWEN4_QSA_PROB_CACHE_FIXTURE_ONLY") != NULL) {
        test_qsa_probability_cache_exact();
        ds4_gpu_cleanup();
        puts("Qwen QSA probability-cache focused fixture PASS");
        return 0;
    }
    if (getenv("DS4_QWEN4_MOE_EXACT_FIXTURE_ONLY") != NULL) {
        test_moe_q4_k();
        test_moe_iq2_xxs_q2_k();
        ds4_gpu_cleanup();
        free(moe_rows8_fixture_allocation);
        free(moe_q2_rows8_fixture_allocation);
        puts("Qwen exact rows8 Q4_K and IQ2_XXS/Q2_K MoE focused fixtures PASS");
        return 0;
    }
    if (getenv("DS4_QWEN4_MOE_Q2_FIXTURE_ONLY") != NULL) {
        test_moe_iq2_xxs_q2_k();
        ds4_gpu_cleanup();
        free(moe_q2_rows8_fixture_allocation);
        puts("Qwen exact rows8 IQ2_XXS/Q2_K MoE focused fixture PASS");
        return 0;
    }
    test_vision_gelu_tail();
    test_q8_0_matmul();
    test_model_q8_0_paths();
    test_model_q8_0_mul_mm_rows();
    test_model_q8_0_mpp_rows();
    test_model_q8_0_mpp_f32stage_rows();
    test_model_bf16_matmul_rows();
    test_vision_fc2_q8_0_stride();
    test_moe_topk();
    test_moe_q4_k();
    test_moe_q4_0();
    test_moe_q4_0_mul_mm_id();
    test_moe_q4_k_mul_mm_id();
    test_moe_iq2_xxs_q2_k();
    test_ple_gate_conv();
    test_ple_long_tiled_schedule();
    test_gdn_prepare_and_output_norm();
    test_gdn_prepare_long_split();
    test_gdn_bf16_state_decode();
    test_gdn_bf16_capture_exact();
    test_gdn_fused_bf16_state_decode();
    test_gdn_capture_exact();
    const uint32_t lengths[] = {1u, 5u, 17u, 2048u, 8192u};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++)
        test_gdn_length(lengths[i]);
    const uint32_t pooled_lengths[] = {512u, 2048u, 8192u, 32768u};
    const qsa_input_type qsa_inputs[] = {
        QSA_INPUT_F32, QSA_INPUT_F16, QSA_INPUT_BF16,
        QSA_INPUT_F32_BF16,
    };
    for (size_t p = 0; p < sizeof(qsa_inputs) / sizeof(qsa_inputs[0]); p++)
        for (size_t i = 0;
             i < sizeof(pooled_lengths) / sizeof(pooled_lengths[0]); i++)
            test_qsa(pooled_lengths[i], qsa_inputs[p]);
    test_qsa_prepare();
    test_qsa_partial_commit_pool_boundary();
    test_qsa_sparse_attention();
    test_qsa_probability_cache_exact();
    test_qsa_attention_gqa_mma();
    test_qsa_streaming_topk();
    test_qsa_streaming_topk_merge_select();
    test_qsa_streaming_topk_score_mm();
    ds4_gpu_cleanup();
    free(moe_rows8_fixture_allocation);
    free(moe_q2_rows8_fixture_allocation);
    free(vision_fc2_fixture_allocation);
    free(model_fixture_allocation);
    puts("Qwen3.8-Flash-Next Metal fixtures PASS");
    return 0;
}
