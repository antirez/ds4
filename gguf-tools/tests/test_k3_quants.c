#include "../quants.h"

#include "ggml-quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef size_t (*quantize_fn)(const float *, void *, int64_t, int64_t,
                              const float *);
typedef void (*dequantize_fn)(const void *, float *, int64_t);

static uint32_t random_state = UINT32_C(0x4b334747);

static float random_float(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return ((float)(random_state & UINT32_C(0x00ffffff)) /
            8388608.0f) - 1.0f;
}

static void fill_values(float *values, int64_t elements) {
    for (int64_t i = 0; i < elements; i++) {
        const float envelope =
            i % 97 == 0 ? 8.0f : (i % 23 == 0 ? 2.5f : 0.7f);
        values[i] = i % 211 == 0 ? 0.0f : random_float() * envelope;
    }
}

static void fill_importance(float *importance, int64_t columns) {
    for (int64_t i = 0; i < columns; i++) {
        const float value = random_float();
        importance[i] = 0.01f + value * value * 7.0f;
    }
}

static int check_quant(const char *name, ds4q_type type,
                       quantize_fn reference_quantize,
                       dequantize_fn reference_dequantize,
                       int64_t rows, int64_t columns,
                       int require_exact,
                       float maximum_relative_rmse) {
    const int64_t elements = rows * columns;
    const size_t row_size = ds4q_row_size(type, columns);
    const size_t bytes = (size_t)rows * row_size;
    float *values = malloc((size_t)elements * sizeof(*values));
    float *importance = malloc((size_t)columns * sizeof(*importance));
    float *decoded = malloc((size_t)elements * sizeof(*decoded));
    float *reference_decoded =
        malloc((size_t)elements * sizeof(*reference_decoded));
    uint8_t *ours = malloc(bytes);
    uint8_t *reference = malloc(bytes);
    if (!values || !importance || !decoded || !reference_decoded ||
        !ours || !reference) {
        fprintf(stderr, "%s: allocation failed\n", name);
        return 0;
    }

    fill_values(values, elements);
    fill_importance(importance, columns);
    ds4q_quantize_init(type);
    const float *weights =
        ds4q_requires_imatrix(type) || type == DS4Q_TYPE_Q2_K
            ? importance : NULL;
    const size_t our_bytes = ds4q_quantize_chunk(
        type, values, ours, 0, rows, columns, weights);
    const size_t reference_bytes = reference_quantize(
        values, reference, rows, columns, weights);
    if (our_bytes != bytes || reference_bytes != bytes) {
        fprintf(stderr,
                "%s: size mismatch: ours=%zu reference=%zu expected=%zu\n",
                name, our_bytes, reference_bytes, bytes);
        return 0;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < bytes; i++) {
        if (ours[i] != reference[i]) mismatches++;
    }
    if (require_exact && mismatches) {
        size_t first = 0;
        while (first < bytes && ours[first] == reference[first]) first++;
        fprintf(stderr,
                "%s: byte mismatch at %zu: ours=%02x reference=%02x\n",
                name, first, ours[first], reference[first]);
        return 0;
    }

    for (int64_t row = 0; row < rows; row++) {
        reference_dequantize(ours + (size_t)row * row_size,
                             decoded + row * columns, columns);
        reference_dequantize(reference + (size_t)row * row_size,
                             reference_decoded + row * columns, columns);
    }
    double signal = 0.0;
    double error = 0.0;
    double weighted_signal = 0.0;
    double weighted_error = 0.0;
    double reference_error = 0.0;
    double reference_weighted_error = 0.0;
    for (int64_t i = 0; i < elements; i++) {
        const double value = values[i];
        const double difference = decoded[i] - value;
        const double weight = importance[i % columns];
        signal += value * value;
        error += difference * difference;
        weighted_signal += weight * value * value;
        weighted_error += weight * difference * difference;
        const double reference_difference = reference_decoded[i] - value;
        reference_error += reference_difference * reference_difference;
        reference_weighted_error +=
            weight * reference_difference * reference_difference;
    }
    const double relative_rmse = sqrt(error / signal);
    const double weighted_relative_rmse =
        sqrt(weighted_error / weighted_signal);
    printf("%-8s byte_delta %.4f%%, rel_rmse %.6f/%.6f, "
           "weighted %.6f/%.6f\n",
           name, 100.0 * (double)mismatches / bytes,
           relative_rmse, sqrt(reference_error / signal),
           weighted_relative_rmse,
           sqrt(reference_weighted_error / weighted_signal));

    free(reference);
    free(ours);
    free(reference_decoded);
    free(decoded);
    free(importance);
    free(values);
    return (!require_exact || mismatches == 0) &&
           isfinite(relative_rmse) &&
           relative_rmse <= maximum_relative_rmse;
}

static size_t quantize_mxfp4_wrapper(
        const float *values, void *output, int64_t rows, int64_t columns,
        const float *importance) {
    (void)importance;
    quantize_row_mxfp4_ref(values, output, rows * columns);
    return (size_t)rows * ds4q_row_size(DS4Q_TYPE_MXFP4, columns);
}

static void dequantize_q8_wrapper(const void *source, float *target,
                                  int64_t elements) {
    dequantize_row_q8_0(source, target, elements);
}

static void dequantize_q2_wrapper(const void *source, float *target,
                                  int64_t elements) {
    dequantize_row_q2_K(source, target, elements);
}

static void dequantize_iq2_wrapper(const void *source, float *target,
                                   int64_t elements) {
    dequantize_row_iq2_xxs(source, target, elements);
}

static void dequantize_mxfp4_wrapper(const void *source, float *target,
                                     int64_t elements) {
    dequantize_row_mxfp4(source, target, elements);
}

static int check_edge_cases(void) {
    float tiny[32];
    uint8_t tiny_ours[17];
    for (uint32_t i = 0; i < 32u; i++) {
        tiny[i] = (i & 1u) ? -0x1p-140f : 0x1p-140f;
    }
    if (ds4q_quantize_chunk(
            DS4Q_TYPE_MXFP4, tiny, tiny_ours, 0, 1, 32, NULL) !=
            sizeof(tiny_ours)) {
        fprintf(stderr, "MXFP4 tiny-block quantization failed\n");
        return 0;
    }
    for (uint32_t i = 0; i < sizeof(tiny_ours); i++) {
        if (tiny_ours[i] != 0u) {
            fprintf(stderr, "MXFP4 tiny-block did not clamp to zero\n");
            return 0;
        }
    }

    float values[256];
    float zero_importance[256] = {0};
    float decoded[256];
    uint8_t quantized[66];
    fill_values(values, 256);
    if (ds4q_quantize_chunk(
            DS4Q_TYPE_IQ2_XXS, values, quantized, 0, 1, 256,
            zero_importance) != sizeof(quantized)) {
        fprintf(stderr, "IQ2_XXS zero-importance quantization failed\n");
        return 0;
    }
    dequantize_row_iq2_xxs(
        (const block_iq2_xxs *)(const void *)quantized, decoded, 256);
    for (uint32_t i = 0; i < 256u; i++) {
        if (!isfinite(decoded[i])) {
            fprintf(stderr,
                    "IQ2_XXS zero-importance produced a non-finite value\n");
            return 0;
        }
    }
    puts("edge cases: ok");
    return 1;
}

int main(void) {
    enum { rows = 5, columns = 3584 };
    iq2xs_init_impl(GGML_TYPE_IQ2_XXS);
    int ok = 1;
    ok &= check_quant("Q8_0", DS4Q_TYPE_Q8_0,
                      quantize_q8_0, dequantize_q8_wrapper,
                      rows, columns, 1, 0.02f);
    ok &= check_quant("Q2_K", DS4Q_TYPE_Q2_K,
                      quantize_q2_K, dequantize_q2_wrapper,
                      rows, columns, 0, 0.45f);
    ok &= check_quant("IQ2_XXS", DS4Q_TYPE_IQ2_XXS,
                      quantize_iq2_xxs, dequantize_iq2_wrapper,
                      rows, columns, 1, 0.65f);
    ok &= check_quant("MXFP4", DS4Q_TYPE_MXFP4,
                      quantize_mxfp4_wrapper,
                      dequantize_mxfp4_wrapper,
                      rows, columns, 1, 0.25f);
    ok &= check_edge_cases();
    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    return ok ? 0 : 1;
}
