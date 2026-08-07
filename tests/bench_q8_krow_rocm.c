/* Throughput benchmark for the ROCm k-row Q8_0 prequant matvec tier.
 *
 * Compares, at production decode shapes:
 *   A) one k-row batched call  (ds4_gpu_matmul_q8_0_decode_rows_exact_tensor)
 *   B) k sequential one-row calls (ds4_gpu_matmul_q8_0_tensor, n_tok=1)
 * which is what the speculative verifier's span evaluation costs without
 * the tier.  Also verifies the two paths stay bit-identical, so the
 * reported speedup is for the exact-rows contract, not an approximation.
 */

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#define Q8_BLOCK_BYTES 34u
#define MAX_K 6u

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
} shape;

static uint32_t rng_state;
static uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static float rng_float(void) {
    return ((float)(rng_next() >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static void fill_q8_weights(uint8_t *w, uint32_t out_dim, uint32_t blocks) {
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (row * blocks + b) * Q8_BLOCK_BYTES;
            const uint16_t scale_bits = (uint16_t)(0x2c00u + (rng_next() & 0x3ffu));
            blk[0] = (uint8_t)(scale_bits & 0xffu);
            blk[1] = (uint8_t)(scale_bits >> 8);
            for (uint32_t j = 0; j < 32u; j++) {
                blk[2u + j] = (uint8_t)(int8_t)((int)(rng_next() % 255u) - 127);
            }
        }
    }
}

static double bench_rows(ds4_gpu_tensor *out_batch_t, const void *model,
                         uint64_t model_size, uint64_t weight_offset,
                         uint32_t in_dim, uint32_t out_dim,
                         ds4_gpu_tensor *x_batch_t, uint32_t k,
                         float *out_batch, int iters) {
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        const double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            if (!ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                    out_batch_t, model, model_size, weight_offset,
                    in_dim, out_dim, x_batch_t, k)) return -1.0;
        }
        if (!ds4_gpu_tensor_read(out_batch_t, 0u, out_batch, sizeof(float)))
            return -1.0;
        const double el = (now_sec() - t0) * 1000.0 / iters;
        if (el < best) best = el;
    }
    return best;
}

static double bench_seq(ds4_gpu_tensor *out_row_t, const void *model,
                        uint64_t model_size, uint64_t weight_offset,
                        uint32_t in_dim, uint32_t out_dim,
                        ds4_gpu_tensor *x_row_t, float *x, uint32_t k,
                        float *out_row, int iters) {
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        const double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            for (uint32_t r = 0; r < k; r++) {
                if (!ds4_gpu_tensor_write(x_row_t, 0u,
                                          x + (uint64_t)r * in_dim,
                                          (uint64_t)in_dim * sizeof(float)) ||
                    !ds4_gpu_matmul_q8_0_tensor(out_row_t, model, model_size,
                                                weight_offset, in_dim, out_dim,
                                                x_row_t, 1u)) return -1.0;
            }
        }
        if (!ds4_gpu_tensor_read(out_row_t, 0u, out_row, sizeof(float)))
            return -1.0;
        const double el = (now_sec() - t0) * 1000.0 / iters;
        if (el < best) best = el;
    }
    return best;
}

static int run_shape(const shape *sh) {
    const uint32_t in_dim = sh->in_dim;
    const uint32_t out_dim = sh->out_dim;
    const uint32_t blocks = (in_dim + 31u) / 32u;
    const uint64_t weight_bytes = (uint64_t)out_dim * blocks * Q8_BLOCK_BYTES;
    const uint64_t weight_offset = 4096u;
    const uint64_t model_size = weight_offset + weight_bytes;

    int ok = 1;
    FILE *model_file = tmpfile();
    void *model = MAP_FAILED;
    float *x = NULL, *out_batch = NULL, *out_row = NULL;
    ds4_gpu_tensor *x_batch_t = NULL, *x_row_t = NULL;
    ds4_gpu_tensor *out_batch_t = NULL, *out_row_t = NULL;

    if (model_file &&
        ftruncate(fileno(model_file), (off_t)model_size) == 0) {
        model = mmap(NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fileno(model_file), 0);
    }
    x = (float *)malloc((size_t)MAX_K * in_dim * sizeof(float));
    out_batch = (float *)malloc((size_t)MAX_K * out_dim * sizeof(float));
    out_row = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!model_file || model == MAP_FAILED || !x || !out_batch || !out_row) {
        fprintf(stderr, "krow bench: host allocation failed\n");
        ok = 0;
        goto cleanup;
    }
    memset(model, 0, (size_t)model_size);
    rng_state = 0x51c05eedu ^ in_dim ^ (out_dim << 8);
    fill_q8_weights((uint8_t *)model + weight_offset, out_dim, blocks);
    for (uint64_t i = 0; i < (uint64_t)MAX_K * in_dim; i++) x[i] = rng_float();

    if (!ds4_gpu_set_model_map(model, model_size)) {
        fprintf(stderr, "krow bench: model map setup failed\n");
        ok = 0;
        goto cleanup;
    }
    x_batch_t = ds4_gpu_tensor_alloc((uint64_t)MAX_K * in_dim * sizeof(float));
    x_row_t = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    out_batch_t = ds4_gpu_tensor_alloc((uint64_t)MAX_K * out_dim * sizeof(float));
    out_row_t = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    if (!x_batch_t || !x_row_t || !out_batch_t || !out_row_t) {
        fprintf(stderr, "krow bench: tensor allocation failed\n");
        ok = 0;
        goto cleanup;
    }
    if (!ds4_gpu_tensor_write(x_batch_t, 0u, x,
                              (uint64_t)MAX_K * in_dim * sizeof(float))) {
        ok = 0;
        goto cleanup;
    }

    fprintf(stderr, "shape %ux%u (q8_0, best-of-3, 200 iters):\n",
            in_dim, out_dim);
    for (uint32_t k = 2; k <= MAX_K; k++) {
        /* Exactness first: batched vs sequential must stay bit-identical. */
        if (!ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                out_batch_t, model, model_size, weight_offset,
                in_dim, out_dim, x_batch_t, k) ||
            !ds4_gpu_tensor_read(out_batch_t, 0u, out_batch,
                                 (uint64_t)k * out_dim * sizeof(float))) {
            fprintf(stderr, "  k=%u: batched eval failed\n", k);
            ok = 0;
            continue;
        }
        uint64_t mismatches = 0;
        for (uint32_t r = 0; r < k; r++) {
            if (!ds4_gpu_tensor_write(x_row_t, 0u, x + (uint64_t)r * in_dim,
                                      (uint64_t)in_dim * sizeof(float)) ||
                !ds4_gpu_matmul_q8_0_tensor(out_row_t, model, model_size,
                                            weight_offset, in_dim, out_dim,
                                            x_row_t, 1u) ||
                !ds4_gpu_tensor_read(out_row_t, 0u, out_row,
                                     (uint64_t)out_dim * sizeof(float))) {
                ok = 0;
                break;
            }
            for (uint32_t i = 0; i < out_dim; i++) {
                if (memcmp(&out_batch[(uint64_t)r * out_dim + i], &out_row[i],
                           sizeof(float)) != 0) mismatches++;
            }
        }
        const double rows_ms = bench_rows(out_batch_t, model, model_size,
                                          weight_offset, in_dim, out_dim,
                                          x_batch_t, k, out_batch, 200);
        const double seq_ms = bench_seq(out_row_t, model, model_size,
                                        weight_offset, in_dim, out_dim,
                                        x_row_t, x, k, out_row, 200);
        if (rows_ms < 0.0 || seq_ms < 0.0) {
            ok = 0;
            continue;
        }
        fprintf(stderr,
                "  k=%u: rows=%.4f ms  seq=%.4f ms  speedup=%.2fx  %s\n",
                k, rows_ms, seq_ms, seq_ms / rows_ms,
                mismatches == 0 ? "BITEXACT" : "MISMATCH");
        if (mismatches != 0) ok = 0;
    }

cleanup:
    if (x_batch_t) ds4_gpu_tensor_free(x_batch_t);
    if (x_row_t) ds4_gpu_tensor_free(x_row_t);
    if (out_batch_t) ds4_gpu_tensor_free(out_batch_t);
    if (out_row_t) ds4_gpu_tensor_free(out_row_t);
    if (model != MAP_FAILED) munmap(model, (size_t)model_size);
    if (model_file) fclose(model_file);
    free(x);
    free(out_batch);
    free(out_row);
    return ok;
}

int main(void) {
    /* Production decode matvec shapes: square attention-sized, the shared
     * expert pair, and a wide FFN. */
    const shape shapes[] = {
        {4096u, 4096u},
        {2048u, 4096u},
        {4096u, 14336u},
    };
    if (!ds4_gpu_init()) {
        fprintf(stderr, "krow bench: ds4_gpu_init failed\n");
        return 1;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    int ok = 1;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        if (!run_shape(&shapes[i])) ok = 0;
    }
    ds4_gpu_set_model_map(NULL, 0u);
    ds4_gpu_cleanup();
    fprintf(stderr, "krow bench: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
