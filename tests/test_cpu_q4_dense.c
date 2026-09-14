/* Exercise the production CPU dispatch with synthetic mapped weights.
 * Including the engine keeps the test tied to its actual static kernels. */
#ifndef DS4_NO_GPU
#define DS4_NO_GPU
#endif
#define DS4_TEST_HOOKS
#include "../ds4.c"

static uint32_t test_rng = 1;
static uint32_t next_bits(void) {
    test_rng = test_rng * 1664525u + 1013904223u;
    return test_rng;
}

static int test_shape(uint32_t k, uint32_t groups, uint32_t rank,
                      uint32_t tokens) {
    const size_t rows = (size_t)groups * rank;
    const size_t blocks = k / QK_K;
    const size_t weight_bytes = rows * blocks * sizeof(block_q4_K);
    block_q4_K *weights = xmalloc(weight_bytes);
    for (size_t b = 0; b < rows * blocks; b++) {
        weights[b].d = 0x2400;    /* 1/64 */
        weights[b].dmin = 0x1800; /* 1/512 */
        for (size_t j = 0; j < sizeof(weights[b].scales); j++)
            weights[b].scales[j] = (uint8_t)(next_bits() >> 24);
        for (size_t j = 0; j < sizeof(weights[b].qs); j++)
            weights[b].qs[j] = (uint8_t)(next_bits() >> 24);
    }
    float *input = xmalloc((size_t)tokens * groups * k * sizeof(float));
    for (size_t i = 0; i < (size_t)tokens * groups * k; i++)
        input[i] = ((int)(next_bits() >> 24) - 128) / 64.0f;
    float *output = xmalloc((tokens * rows + 2) * sizeof(float));
    float *reference = xmalloc(tokens * rows * sizeof(float));
    float *scratch_out = xmalloc(rows * sizeof(float));
    output[0] = 12345.0f;
    output[tokens * rows + 1] = -12345.0f;
    ds4_model model = {.map = (const uint8_t *)weights, .size = weight_bytes};
    ds4_tensor tensor = {
        .type = DS4_TENSOR_Q4_K, .ndim = 2, .dim = {k, rows, 0, 0},
        .bytes = weight_bytes, .abs_offset = 0,
    };
    ds4_cpu_decode_scratch scratch = {
        .q8_cap = groups * k,
        .dense_xq = xmalloc((size_t)groups * blocks * sizeof(block_q8_K)),
    };
    int failed = 0;
    for (uint32_t t = 0; t < tokens; t++) {
        const float *x = input + (size_t)t * groups * k;
        float *ref = reference + t * rows;
        if (groups == 1) {
            matvec_any(ref, &model, &tensor, x);
            matvec_any_decode_scratch(scratch_out, &model, &tensor, x, &scratch);
        } else {
            matvec_dense_grouped_rows(ref, &model, &tensor, x, groups, k, rank);
            matvec_dense_grouped_rows_decode_scratch(
                    scratch_out, &model, &tensor, x, groups, k, rank, &scratch);
        }
        if (memcmp(ref, scratch_out, rows * sizeof(float)) != 0) failed = 1;
    }
    if (groups == 1)
        matmul_dense_batch(output + 1, &model, &tensor, input, tokens);
    else
        matmul_dense_grouped_batch(output + 1, &model, &tensor, input,
                                   tokens, groups, k, rank);
    if (memcmp(reference, output + 1, tokens * rows * sizeof(float)) != 0 ||
        output[0] != 12345.0f || output[tokens * rows + 1] != -12345.0f)
        failed = 1;
    if (failed)
        fprintf(stderr, "Q4 CPU mismatch: K=%u groups=%u rank=%u tokens=%u\n",
                k, groups, rank, tokens);
    free(scratch.dense_xq);
    free(scratch_out);
    free(reference);
    free(output);
    free(input);
    free(weights);
    return failed;
}

int main(void) {
    const uint32_t widths[] = {256, 1024, 4096};
    const uint32_t token_counts[] = {1, 2, 3, 8, 33, 65};
    int failures = 0;
    unsigned cases = 0;
    for (size_t k = 0; k < sizeof(widths) / sizeof(widths[0]); k++) {
        for (size_t n = 0; n < sizeof(token_counts) / sizeof(token_counts[0]); n++) {
            failures += test_shape(widths[k], 1, 9, token_counts[n]);
            failures += test_shape(widths[k], 3, 5, token_counts[n]);
            cases += 2;
        }
    }
    if (!ds4_test_indexer_q_type_supported(DS4_TENSOR_Q4_K) ||
        !ds4_test_indexer_q_type_supported(DS4_TENSOR_Q8_0) ||
        !ds4_test_indexer_q_type_supported(DS4_TENSOR_F16) ||
        ds4_test_indexer_q_type_supported(DS4_TENSOR_F32)) failures++;
    printf("CPU Q4 dense/grouped decode, scratch and prefill: %u cases, %s\n",
           cases, failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
