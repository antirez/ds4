/* Bit-exactness test for the ROCm k-row Q8_0 prequant matvec tier.
 *
 * The decode tier (matmul_q8_0_preq_rows_w32_kernel) defines the reference
 * summation order for one-row evals.  The k-row tier must produce, for
 * every row of an n_tok in 2..5 call, bitwise the same output as a
 * one-row call on the same weights and that row's activations: the exact
 * speculative verifier depends on this equivalence, and it is what the
 * generic batch tiers (different reduction shapes, weights re-read per
 * row) do not provide.
 */

#include "ds4_gpu.h"

#include <math.h>
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
/* 2..5 exercises the k-row kernel tiers; 6..7 exercises the per-row
 * fallback that keeps the exact-rows contract honest for wider spans. */
#define MAX_K 7u

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
} shape;

/* Deterministic PRNG so failures reproduce. */
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
            /* fp16 scale in (0, 0.125]: bias 0x2c00 (~0.0625) plus noise. */
            const uint16_t scale_bits = (uint16_t)(0x2c00u + (rng_next() & 0x3ffu));
            blk[0] = (uint8_t)(scale_bits & 0xffu);
            blk[1] = (uint8_t)(scale_bits >> 8);
            for (uint32_t j = 0; j < 32u; j++) {
                blk[2u + j] = (uint8_t)(int8_t)((int)(rng_next() % 255u) - 127);
            }
        }
    }
}

static int run_shape(const shape *sh) {
    const uint32_t in_dim = sh->in_dim;
    const uint32_t out_dim = sh->out_dim;
    const uint32_t blocks = (in_dim + 31u) / 32u;
    /* Second matrix for the pair entry point: a different out_dim keeps
     * the divergent row guards honest. */
    const uint32_t out1_dim = out_dim / 2u + 3u;
    const uint64_t weight_bytes = (uint64_t)out_dim * blocks * Q8_BLOCK_BYTES;
    const uint64_t weight1_bytes = (uint64_t)out1_dim * blocks * Q8_BLOCK_BYTES;
    /* Keep the weight tensors away from offset zero so range checks see a
     * realistic layout. */
    const uint64_t weight_offset = 4096u;
    const uint64_t weight1_offset = weight_offset + weight_bytes;
    /* F16 router matrix (4096x256) for the router rows entry. */
    const uint64_t router_offset = weight1_offset + weight1_bytes;
    const uint64_t router_bytes = 4096u * 256u * 2u;
    const uint64_t model_size = router_offset + router_bytes;

    int ok = 1;
    FILE *model_file = tmpfile();
    void *model = MAP_FAILED;
    float *x = NULL;
    float *out_batch = NULL;
    float *out_row = NULL;
    ds4_gpu_tensor *x_batch_t = NULL;
    ds4_gpu_tensor *x_row_t = NULL;
    ds4_gpu_tensor *out_batch_t = NULL;
    ds4_gpu_tensor *out_row_t = NULL;

    if (model_file &&
        ftruncate(fileno(model_file), (off_t)model_size) == 0) {
        model = mmap(NULL, (size_t)model_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fileno(model_file), 0);
    }
    x = (float *)malloc((size_t)MAX_K * in_dim * sizeof(float));
    out_batch = (float *)malloc((size_t)MAX_K * out_dim * sizeof(float));
    out_row = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!model_file || model == MAP_FAILED || !x || !out_batch || !out_row) {
        fprintf(stderr, "q8 krow: host allocation failed\n");
        ok = 0;
        goto cleanup;
    }
    memset(model, 0, (size_t)model_size);
    rng_state = 0x51c05eedu ^ in_dim ^ (out_dim << 8);
    fill_q8_weights((uint8_t *)model + weight_offset, out_dim, blocks);
    fill_q8_weights((uint8_t *)model + weight1_offset, out1_dim, blocks);
    {
        uint16_t *rw = (uint16_t *)((uint8_t *)model + router_offset);
        for (uint64_t i = 0; i < 4096u * 256u; i++) {
            rw[i] = (uint16_t)(0x2c00u + (rng_next() & 0x3ffu)) |
                    (uint16_t)((rng_next() & 1u) << 15);
        }
    }
    for (uint64_t i = 0; i < (uint64_t)MAX_K * in_dim; i++) x[i] = rng_float();

    if (!ds4_gpu_set_model_map(model, model_size)) {
        fprintf(stderr, "q8 krow: model map setup failed\n");
        ok = 0;
        goto cleanup;
    }

    x_batch_t = ds4_gpu_tensor_alloc((uint64_t)MAX_K * in_dim * sizeof(float));
    x_row_t = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    out_batch_t = ds4_gpu_tensor_alloc((uint64_t)MAX_K * out_dim * sizeof(float));
    out_row_t = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    if (!x_batch_t || !x_row_t || !out_batch_t || !out_row_t) {
        fprintf(stderr, "q8 krow: tensor allocation failed\n");
        ok = 0;
        goto cleanup;
    }
    if (!ds4_gpu_tensor_write(x_batch_t, 0u, x,
                              (uint64_t)MAX_K * in_dim * sizeof(float))) {
        fprintf(stderr, "q8 krow: activation upload failed\n");
        ok = 0;
        goto cleanup;
    }

    for (uint32_t k = 2; k <= MAX_K; k++) {
        if (!ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                out_batch_t, model, model_size,
                weight_offset, in_dim, out_dim,
                x_batch_t, k) ||
            !ds4_gpu_tensor_read(out_batch_t, 0u, out_batch,
                                 (uint64_t)k * out_dim * sizeof(float))) {
            fprintf(stderr, "q8 krow: batch eval failed (%ux%u k=%u)\n",
                    in_dim, out_dim, k);
            ok = 0;
            continue;
        }
        uint64_t mismatches = 0;
        uint64_t first_at = 0;
        for (uint32_t r = 0; r < k; r++) {
            if (!ds4_gpu_tensor_write(x_row_t, 0u, x + (uint64_t)r * in_dim,
                                      (uint64_t)in_dim * sizeof(float)) ||
                !ds4_gpu_matmul_q8_0_tensor(out_row_t, model, model_size,
                                            weight_offset, in_dim, out_dim,
                                            x_row_t, 1u) ||
                !ds4_gpu_tensor_read(out_row_t, 0u, out_row,
                                     (uint64_t)out_dim * sizeof(float))) {
                fprintf(stderr, "q8 krow: row eval failed (%ux%u k=%u r=%u)\n",
                        in_dim, out_dim, k, r);
                ok = 0;
                break;
            }
            for (uint32_t i = 0; i < out_dim; i++) {
                if (memcmp(&out_batch[(uint64_t)r * out_dim + i], &out_row[i],
                           sizeof(float)) != 0) {
                    if (mismatches == 0) first_at = (uint64_t)r * out_dim + i;
                    mismatches++;
                }
            }
        }
        /* Timed pass: the same call the exactness check exercised.  With
         * DS4_ROCM_DSV4_PREQUANT_DECODE=0 the dispatch reverts to the f32
         * batch tiers, so this doubles as an A/B probe for the tier. */
        const int iters = 100;
        double best = 1e30;
        for (int t = 0; t < 3; t++) {
            const double t0 = now_sec();
            for (int i = 0; i < iters; i++) {
                if (!ds4_gpu_matmul_q8_0_tensor(out_batch_t, model, model_size,
                                                weight_offset, in_dim, out_dim,
                                                x_batch_t, k)) {
                    ok = 0;
                    break;
                }
            }
            if (!ds4_gpu_tensor_read(out_batch_t, 0u, out_batch,
                                     sizeof(float))) ok = 0;
            const double el = (now_sec() - t0) * 1000.0 / iters;
            if (el < best) best = el;
        }
        fprintf(stderr, "q8 krow %ux%u k=%u: %s (%llu mismatches",
                in_dim, out_dim, k,
                mismatches == 0 ? "BITEXACT" : "MISMATCH",
                (unsigned long long)mismatches);
        if (mismatches) {
            fprintf(stderr, ", first at %llu", (unsigned long long)first_at);
            ok = 0;
        }
        fprintf(stderr, ") %.3f ms/call\n", best);
    }

    /* Pair entry point: n_rows in 2..5 must match per-row one-row pair
     * calls, whose reduction order comes from the fused pair kernel. */
    {
        ds4_gpu_tensor *out1_batch_t =
            ds4_gpu_tensor_alloc((uint64_t)MAX_K * out1_dim * sizeof(float));
        ds4_gpu_tensor *out1_row_t =
            ds4_gpu_tensor_alloc((uint64_t)out1_dim * sizeof(float));
        float *out1_batch = (float *)malloc((size_t)MAX_K * out1_dim * sizeof(float));
        float *out1_row = (float *)malloc((size_t)out1_dim * sizeof(float));
        if (!out1_batch_t || !out1_row_t || !out1_batch || !out1_row) {
            fprintf(stderr, "q8 krow pair: allocation failed\n");
            ok = 0;
        } else {
            for (uint32_t k = 2; k <= MAX_K; k++) {
                if (!ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
                        out_batch_t, out1_batch_t, model, model_size,
                        weight_offset, weight1_offset, in_dim,
                        out_dim, out1_dim, x_batch_t, k) ||
                    !ds4_gpu_tensor_read(out_batch_t, 0u, out_batch,
                                         (uint64_t)k * out_dim * sizeof(float)) ||
                    !ds4_gpu_tensor_read(out1_batch_t, 0u, out1_batch,
                                         (uint64_t)k * out1_dim * sizeof(float))) {
                    fprintf(stderr, "q8 krow pair: batch eval failed (%ux%u k=%u)\n",
                            in_dim, out_dim, k);
                    ok = 0;
                    continue;
                }
                uint64_t mism = 0;
                for (uint32_t r = 0; r < k; r++) {
                    if (!ds4_gpu_tensor_write(x_row_t, 0u, x + (uint64_t)r * in_dim,
                                              (uint64_t)in_dim * sizeof(float)) ||
                        !ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
                            out_row_t, out1_row_t, model, model_size,
                            weight_offset, weight1_offset, in_dim,
                            out_dim, out1_dim, x_row_t, 1u) ||
                        !ds4_gpu_tensor_read(out_row_t, 0u, out_row,
                                             (uint64_t)out_dim * sizeof(float)) ||
                        !ds4_gpu_tensor_read(out1_row_t, 0u, out1_row,
                                             (uint64_t)out1_dim * sizeof(float))) {
                        fprintf(stderr, "q8 krow pair: row eval failed (k=%u r=%u)\n", k, r);
                        ok = 0;
                        break;
                    }
                    for (uint32_t i = 0; i < out_dim; i++)
                        if (memcmp(&out_batch[(uint64_t)r * out_dim + i],
                                   &out_row[i], sizeof(float)) != 0) mism++;
                    for (uint32_t i = 0; i < out1_dim; i++)
                        if (memcmp(&out1_batch[(uint64_t)r * out1_dim + i],
                                   &out1_row[i], sizeof(float)) != 0) mism++;
                }
                fprintf(stderr, "q8 krow pair %ux%u+%u k=%u: %s (%llu mismatches)\n",
                        in_dim, out_dim, out1_dim, k,
                        mism == 0 ? "BITEXACT" : "MISMATCH",
                        (unsigned long long)mism);
                if (mism) ok = 0;
            }
        }
        if (out1_batch_t) ds4_gpu_tensor_free(out1_batch_t);
        if (out1_row_t) ds4_gpu_tensor_free(out1_row_t);
        free(out1_batch);
        free(out1_row);
    }

    /* Shared-expert rows entry: per-row bit-equality with the one-row
     * shared gate/up + swiglu entry (both weight matrices point at the
     * first weight region; the contract is about arithmetic order). */
    {
        ds4_gpu_tensor *gate_b = ds4_gpu_tensor_alloc((uint64_t)MAX_K * out_dim * sizeof(float));
        ds4_gpu_tensor *up_b = ds4_gpu_tensor_alloc((uint64_t)MAX_K * out_dim * sizeof(float));
        ds4_gpu_tensor *mid_b = ds4_gpu_tensor_alloc((uint64_t)MAX_K * out_dim * sizeof(float));
        ds4_gpu_tensor *gate_r = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
        ds4_gpu_tensor *up_r = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
        ds4_gpu_tensor *mid_r = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
        float *mid_batch = (float *)malloc((size_t)MAX_K * out_dim * sizeof(float));
        float *mid_row = (float *)malloc((size_t)out_dim * sizeof(float));
        if (gate_b && up_b && mid_b && gate_r && up_r && mid_r &&
            mid_batch && mid_row) {
            for (uint32_t k = 2; k <= MAX_K; k++) {
                if (!ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                        gate_b, up_b, mid_b, model, model_size,
                        weight_offset, weight_offset, in_dim, out_dim,
                        x_batch_t, k, 30.0f) ||
                    !ds4_gpu_tensor_read(mid_b, 0u, mid_batch,
                                         (uint64_t)k * out_dim * sizeof(float))) {
                    fprintf(stderr, "q8 krow swiglu: batch eval failed (k=%u)\n", k);
                    ok = 0;
                    continue;
                }
                uint64_t mism = 0;
                for (uint32_t r = 0; r < k; r++) {
                    if (!ds4_gpu_tensor_write(x_row_t, 0u, x + (uint64_t)r * in_dim,
                                              (uint64_t)in_dim * sizeof(float)) ||
                        !ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                            gate_r, up_r, mid_r, model, model_size,
                            weight_offset, weight_offset, in_dim, out_dim,
                            x_row_t, 30.0f) ||
                        !ds4_gpu_tensor_read(mid_r, 0u, mid_row,
                                             (uint64_t)out_dim * sizeof(float))) {
                        fprintf(stderr, "q8 krow swiglu: row eval failed (k=%u r=%u)\n", k, r);
                        ok = 0;
                        break;
                    }
                    for (uint32_t i = 0; i < out_dim; i++)
                        if (memcmp(&mid_batch[(uint64_t)r * out_dim + i],
                                   &mid_row[i], sizeof(float)) != 0) mism++;
                }
                fprintf(stderr, "q8 krow swiglu %ux%u k=%u: %s (%llu mismatches)\n",
                        in_dim, out_dim, k,
                        mism == 0 ? "BITEXACT" : "MISMATCH",
                        (unsigned long long)mism);
                if (mism) ok = 0;
            }
        } else {
            fprintf(stderr, "q8 krow swiglu: allocation failed\n");
            ok = 0;
        }
        if (gate_b) ds4_gpu_tensor_free(gate_b);
        if (up_b) ds4_gpu_tensor_free(up_b);
        if (mid_b) ds4_gpu_tensor_free(mid_b);
        if (gate_r) ds4_gpu_tensor_free(gate_r);
        if (up_r) ds4_gpu_tensor_free(up_r);
        if (mid_r) ds4_gpu_tensor_free(mid_r);
        free(mid_batch);
        free(mid_row);
    }

    /* Router rows entry: batched router logits must equal per-row one-row
     * calls bitwise, or batch rows can select different experts. */
    if (in_dim == 4096u) {
        ds4_gpu_tensor *lg_b = ds4_gpu_tensor_alloc((uint64_t)MAX_K * 256u * sizeof(float));
        ds4_gpu_tensor *lg_r = ds4_gpu_tensor_alloc(256u * sizeof(float));
        float *lb = (float *)malloc((size_t)MAX_K * 256u * sizeof(float));
        float *lr = (float *)malloc(256u * sizeof(float));
        if (lg_b && lg_r && lb && lr) {
            for (uint32_t k = 2; k <= MAX_K; k++) {
                uint64_t mism = 0;
                if (!ds4_gpu_matmul_f16_router_rows_exact_tensor(
                        lg_b, model, model_size, router_offset, x_batch_t, k) ||
                    !ds4_gpu_tensor_read(lg_b, 0u, lb,
                                         (uint64_t)k * 256u * sizeof(float))) {
                    fprintf(stderr, "q8 krow router: batch eval failed (k=%u)\n", k);
                    ok = 0;
                    continue;
                }
                for (uint32_t r = 0; r < k; r++) {
                    if (!ds4_gpu_tensor_write(x_row_t, 0u, x + (uint64_t)r * in_dim,
                                              (uint64_t)in_dim * sizeof(float)) ||
                        !ds4_gpu_matmul_f16_router_rows_exact_tensor(
                            lg_r, model, model_size, router_offset, x_row_t, 1u) ||
                        !ds4_gpu_tensor_read(lg_r, 0u, lr, 256u * sizeof(float))) {
                        fprintf(stderr, "q8 krow router: row eval failed\n");
                        ok = 0;
                        break;
                    }
                    for (uint32_t i = 0; i < 256u; i++)
                        if (memcmp(&lb[(uint64_t)r * 256u + i], &lr[i],
                                   sizeof(float)) != 0) mism++;
                }
                fprintf(stderr, "q8 krow router 4096x256 k=%u: %s (%llu mismatches)\n",
                        k, mism == 0 ? "BITEXACT" : "MISMATCH",
                        (unsigned long long)mism);
                if (mism) ok = 0;
            }
        } else {
            fprintf(stderr, "q8 krow router: allocation failed\n");
            ok = 0;
        }
        if (lg_b) ds4_gpu_tensor_free(lg_b);
        if (lg_r) ds4_gpu_tensor_free(lg_r);
        free(lb);
        free(lr);
    }

cleanup:
    if (x_batch_t) ds4_gpu_tensor_free(x_batch_t);
    if (x_row_t) ds4_gpu_tensor_free(x_row_t);
    if (out_batch_t) ds4_gpu_tensor_free(out_batch_t);
    if (out_row_t) ds4_gpu_tensor_free(out_row_t);
    if (model != MAP_FAILED) munmap(model, (size_t)model_size);
    if (model_file) fclose(model_file);
    return ok;
}

int main(void) {
    /* Production decode matvec shapes: square attention-sized, the shared
     * expert pair, a wide FFN, and an odd out_dim for row-guard coverage. */
    const shape shapes[] = {
        {4096u, 4096u},
        {2048u, 4096u},
        {4096u, 14336u},
        {4096u, 4099u},
    };
    if (!ds4_gpu_init()) {
        fprintf(stderr, "q8 krow: ds4_gpu_init failed\n");
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
    fprintf(stderr, "Q8_0 k-row ROCm matvec: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
