/*
 * planar_quant_test.c -- unit tests for Planar3 512-dim quantization
 *
 * Build:
 *   cc -O2 -Wall -Wextra -std=c99 -I. -o tests/planar_quant_test \
 *      tests/planar_quant_test.c ds4_planar_quant.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ds4_planar_quant.h"

#define DIM 512
#define NROWS_BATCH 100

/* ---- test harness (matches ds4 test style) ---- */

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "planar_quant_test: FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

/* ---- simple deterministic PRNG (xorshift32) ---- */

static uint32_t prng_state;

static void prng_seed(uint32_t s) {
    prng_state = s ? s : 1u;
}

static float prng_float(void) {
    /* xorshift32 */
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    /* map to [-1, 1) */
    return (float)((int32_t)prng_state) / (float)0x7FFFFFFF;
}

/* ---- helpers ---- */

static float vec_norm(const float *v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

/* ---- tests ---- */

static int test_block_size(void) {
    int fail = 0;
    fail += expect(sizeof(ds4_block_planar3) == 50,
                   "sizeof(ds4_block_planar3) should be 50");
    fail += expect(sizeof(ds4_row_planar3) == 200,
                   "sizeof(ds4_row_planar3) should be 200");

    printf("test_block_size: block=%zu row=%zu -- %s\n",
           sizeof(ds4_block_planar3), sizeof(ds4_row_planar3),
           fail ? "FAIL" : "OK");
    return fail;
}

static int test_roundtrip_basis_vector(void) {
    float src[DIM] = {0};
    float dst[DIM];
    ds4_row_planar3 compressed;

    src[0] = 1.0f; /* e0 */

    ds4_planar3_quantize_row(src, &compressed);
    ds4_planar3_dequantize_row(&compressed, dst);

    float cos = ds4_planar3_roundtrip_cosine(src, dst, DIM);
    float mse = ds4_planar3_roundtrip_mse(src, dst, DIM);

    /* Spike vectors (all energy in one dimension) are pathological for
       2-bit rotated quantization: Givens rotation spreads the energy
       across many pairs, and coarse 3-bit centroids lose detail.
       Cosine ~0.74 is expected; we verify roundtrip is non-degenerate. */
    int fail = expect(cos > 0.70f,
                      "basis vector cosine similarity should be > 0.70");

    printf("test_roundtrip_basis_vector: cosine=%.6f mse=%.6f -- %s\n",
           cos, mse, fail ? "FAIL" : "OK");
    return fail;
}

static int test_roundtrip_random(void) {
    const int N = 10;
    float src[DIM], dst[DIM];
    ds4_row_planar3 compressed;

    prng_seed(123);

    float cos_min = 1.0f, cos_max = 0.0f, cos_sum = 0.0f;
    float mse_sum = 0.0f;
    int fail = 0;

    for (int v = 0; v < N; v++) {
        for (int i = 0; i < DIM; i++)
            src[i] = prng_float();

        ds4_planar3_quantize_row(src, &compressed);
        ds4_planar3_dequantize_row(&compressed, dst);

        float cos = ds4_planar3_roundtrip_cosine(src, dst, DIM);
        float mse = ds4_planar3_roundtrip_mse(src, dst, DIM);

        if (cos < cos_min) cos_min = cos;
        if (cos > cos_max) cos_max = cos;
        cos_sum += cos;
        mse_sum += mse;
    }

    float cos_avg = cos_sum / N;
    float mse_avg = mse_sum / N;

    fail += expect(cos_avg > 0.97f,
                   "random vector avg cosine similarity should be > 0.97");

    printf("test_roundtrip_random: %d vectors, cosine min=%.3f avg=%.3f max=%.3f, MSE avg=%.6f -- %s\n",
           N, cos_min, cos_avg, cos_max, mse_avg, fail ? "FAIL" : "OK");
    return fail;
}

static int test_roundtrip_large_norm(void) {
    float src[DIM], dst[DIM];
    ds4_row_planar3 compressed;

    for (int i = 0; i < DIM; i++)
        src[i] = sinf((float)i * 0.1f + 0.5f) * 10.0f;

    float orig_norm = vec_norm(src, DIM);

    ds4_planar3_quantize_row(src, &compressed);
    ds4_planar3_dequantize_row(&compressed, dst);

    float cos = ds4_planar3_roundtrip_cosine(src, dst, DIM);
    float mse = ds4_planar3_roundtrip_mse(src, dst, DIM);
    float recon_norm = vec_norm(dst, DIM);

    float norm_ratio = recon_norm / orig_norm;
    int fail = 0;
    fail += expect(cos > 0.95f,
                   "large-norm cosine similarity should be > 0.95");
    fail += expect(norm_ratio > 0.90f && norm_ratio < 1.10f,
                   "reconstructed norm should be within 10% of original");

    printf("test_roundtrip_large_norm: cosine=%.6f mse=%.6f "
           "norm_orig=%.3f norm_recon=%.3f ratio=%.3f -- %s\n",
           cos, mse, orig_norm, recon_norm, norm_ratio, fail ? "FAIL" : "OK");
    return fail;
}

static int test_batch_quantize(void) {
    float src[NROWS_BATCH * DIM];
    float dst[NROWS_BATCH * DIM];
    ds4_row_planar3 compressed[NROWS_BATCH];

    prng_seed(456);
    for (int i = 0; i < NROWS_BATCH * DIM; i++)
        src[i] = prng_float();

    size_t total = ds4_planar3_quantize(src, compressed, NROWS_BATCH, DIM);
    ds4_planar3_dequantize(compressed, dst, NROWS_BATCH, DIM);

    float cos_sum = 0.0f;
    int fail = 0;

    for (int r = 0; r < NROWS_BATCH; r++) {
        float cos = ds4_planar3_roundtrip_cosine(
            src + r * DIM, dst + r * DIM, DIM);
        cos_sum += cos;
    }

    float cos_avg = cos_sum / NROWS_BATCH;
    fail += expect(cos_avg > 0.97f,
                   "batch avg cosine similarity should be > 0.97");
    fail += expect(total == NROWS_BATCH * sizeof(ds4_row_planar3),
                   "batch total compressed size mismatch");

    printf("test_batch_quantize: %d rows, avg cosine=%.4f, total_bytes=%zu -- %s\n",
           NROWS_BATCH, cos_avg, total, fail ? "FAIL" : "OK");
    return fail;
}

static int test_compression_ratio(void) {
    size_t fp16_bytes = DIM * 2;  /* 2 bytes per FP16 */
    size_t planar_bytes = sizeof(ds4_row_planar3);
    float ratio = (float)fp16_bytes / (float)planar_bytes;

    int fail = expect(planar_bytes == 200,
                      "compressed row should be exactly 200 bytes");

    printf("test_compression_ratio: fp16=%zu planar=%zu ratio=%.2fx -- %s\n",
           fp16_bytes, planar_bytes, ratio, fail ? "FAIL" : "OK");
    return fail;
}

static int test_block_independence(void) {
    float src[DIM] = {0};
    float dst[DIM];
    ds4_row_planar3 compressed;

    /* Block 0: e0 = [1,0,0,...,0], blocks 1-3: all zeros */
    src[0] = 1.0f;
    /* all other dims already 0 */

    ds4_planar3_quantize_row(src, &compressed);
    ds4_planar3_dequantize_row(&compressed, dst);

    /* Check block 3 (indices 384..511) is near-zero */
    float block3_sq = 0.0f;
    for (int i = 384; i < 512; i++)
        block3_sq += dst[i] * dst[i];
    float block3_norm = sqrtf(block3_sq);

    int fail = expect(block3_norm < 1e-3f,
                       "block 3 should be near-zero when only block 0 has data");

    printf("test_block_independence: block3_norm=%.6e -- %s\n",
           block3_norm, fail ? "FAIL" : "OK");
    return fail;
}

static int test_batch_dim_mismatch(void) {
    float src[DIM];
    float dst[DIM];
    ds4_row_planar3 compressed[1];

    for (int i = 0; i < DIM; i++) src[i] = 1.0f;

    /* quantize with wrong dim should return 0 */
    size_t ret = ds4_planar3_quantize(src, compressed, 1, 511);
    int fail = expect(ret == 0,
                      "quantize with n_per_row=511 should return 0");

    /* dequantize with wrong dim should be no-op: fill dst with sentinel */
    memset(compressed, 0, sizeof(ds4_row_planar3));
    for (int i = 0; i < DIM; i++) dst[i] = 42.0f;
    ds4_planar3_dequantize(compressed, dst, 1, 511);
    for (int i = 0; i < DIM; i++) {
        if (dst[i] != 42.0f) {
            fail += expect(0, "dequantize with n_per_row=511 should not write dst");
            break;
        }
    }

    printf("test_batch_dim_mismatch: quantize_ret=%zu dst_unchanged=%s -- %s\n",
           ret, fail ? "no" : "yes", fail ? "FAIL" : "OK");
    return fail;
}

static int test_zero_norm(void) {
    float src[DIM] = {0};
    float dst[DIM];
    ds4_row_planar3 comp;
    memset(&comp, 0, sizeof(comp));

    ds4_planar3_quantize_row(src, &comp);
    ds4_planar3_dequantize_row(&comp, dst);

    int fail = 0;
    for (int i = 0; i < DIM; i++) {
        if (dst[i] != 0.0f) {
            fail += expect(0, "zero-norm dequant should produce zeros");
            break;
        }
    }
    printf("test_zero_norm: all_zeros=%s -- %s\n", fail ? "no" : "yes", fail ? "FAIL" : "OK");
    return fail;
}

static int test_single_element(void) {
    float src[DIM] = {0};
    src[42] = 3.7f;
    float dst[DIM];
    ds4_row_planar3 comp;

    ds4_planar3_quantize_row(src, &comp);
    ds4_planar3_dequantize_row(&comp, dst);

    float cos = ds4_planar3_roundtrip_cosine(src, dst, DIM);
    int fail = expect(cos > 0.5f,
                      "single-element vector should preserve direction");
    if (!fail) {
        printf("test_single_element: cos=%f val[42]=%f -- OK\n", cos, dst[42]);
    }
    return fail;
}

/* ---- main ---- */

int main(void) {
    int fail = 0;

    fail += test_block_size();
    fail += test_roundtrip_basis_vector();
    fail += test_roundtrip_random();
    fail += test_roundtrip_large_norm();
    fail += test_batch_quantize();
    fail += test_compression_ratio();
    fail += test_block_independence();
    fail += test_batch_dim_mismatch();
    fail += test_zero_norm();
    fail += test_single_element();

    if (fail) {
        fprintf(stderr, "\nplanar_quant_test: %d test(s) FAILED\n", fail);
        return 1;
    }

    printf("\nplanar_quant_test: all tests passed\n");
    return 0;
}
