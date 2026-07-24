/* Numerical check for the wave64-native matmul_q8_0_pair_f32_sharedx_warp_rows_w64_kernel.
 *
 * ds4_gpu_matmul_q8_0_pair_tensor() with n_tok == 1 is the single-token decode
 * hot path profiled this session: on wave64 (CDNA) it now dispatches to a new
 * matmul_q8_0_pair_f32_sharedx_warp_rows_w64_kernel that has every lane of a
 * 64-lane wavefront cooperate on one output row (two Q8_0 blocks per
 * iteration, one 64-wide reduction), instead of the original w32 kernel's two
 * independent 32-lane-half rows. Both kernels operate on f32 activations
 * directly (no f16 staging), so unlike the batched matrix-core test this one
 * compares against a plain double-precision CPU reference with no rounding
 * ambiguity -- any indexing slip in the new kernel's block-pairing or
 * cross-half reduction should show up as a real, non-noise-sized error.
 *
 * DS4_ROCM_DISABLE_PAIR_W64 forces the same call back onto the original w32
 * kernel (unmodified) for a direct kernel-vs-kernel cross-check in addition
 * to the CPU reference.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

#define IN_DIM   2048u
#define OUT0_DIM 777u   /* deliberately not a multiple of 16/32/64 */
#define OUT1_DIM 513u   /* different, smaller, also not block-aligned */

#define QK  32u
#define BLOCK_BYTES 34u

static uint32_t g_rng = 0x9e3779b9u;

static float next_uniform(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)((g_rng >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;

    if (((x >> 23) & 0xffu) == 0xffu) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t half = 1u << (shift - 1);
        uint32_t r = mant + half - 1u + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (r >> shift));
    }
    uint32_t r = mant + 0x0fffu + ((mant >> 13) & 1u);
    if (r & 0x800000u) { r = 0; exp++; if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (r >> 13));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) { x = sign; }
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            x = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static void make_weights(unsigned char *w, uint32_t out_dim, uint32_t n_blocks) {
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t b = 0; b < n_blocks; b++) {
            unsigned char *blk = w + ((uint64_t)r * n_blocks + b) * BLOCK_BYTES;
            const uint16_t d = f32_to_f16(0.01f + 0.02f * fabsf(next_uniform()));
            memcpy(blk, &d, 2);
            for (uint32_t j = 0; j < QK; j++) {
                blk[2 + j] = (unsigned char)(signed char)(int)(next_uniform() * 100.0f);
            }
        }
    }
}

static double cpu_ref_row(const unsigned char *w, uint32_t n_blocks, const float *x, uint32_t row) {
    double acc = 0.0;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = w + ((uint64_t)row * n_blocks + b) * BLOCK_BYTES;
        uint16_t dbits;
        memcpy(&dbits, blk, 2);
        const float d = f16_to_f32(dbits);
        for (uint32_t j = 0; j < QK; j++) {
            const float wv = d * (float)(int)(signed char)blk[2 + j];
            const float xv = x[b * QK + j];
            acc += (double)wv * (double)xv;
        }
    }
    return acc;
}

static int run_once(const unsigned char *w0, const unsigned char *w1,
                     uint32_t n_blocks, const float *x,
                     ds4_gpu_tensor *xt, ds4_gpu_tensor *o0t, ds4_gpu_tensor *o1t,
                     float *out0, float *out1,
                     const void *model_map, uint64_t model_size,
                     uint64_t off0, uint64_t off1) {
    (void)w0; (void)w1; (void)n_blocks; (void)x;
    if (!ds4_gpu_matmul_q8_0_pair_tensor(o0t, o1t, model_map, model_size, off0, off1,
                                         IN_DIM, OUT0_DIM, OUT1_DIM, xt, 1)) {
        fprintf(stderr, "matmul pair failed\n");
        return 0;
    }
    if (!ds4_gpu_tensor_read(o0t, 0, out0, (uint64_t)OUT0_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_read(o1t, 0, out1, (uint64_t)OUT1_DIM * sizeof(float))) {
        fprintf(stderr, "tensor read failed\n");
        return 0;
    }
    return 1;
}

int main(void) {
    const uint32_t n_blocks = IN_DIM / QK;
    const uint64_t w0_bytes = (uint64_t)OUT0_DIM * n_blocks * BLOCK_BYTES;
    const uint64_t w1_bytes = (uint64_t)OUT1_DIM * n_blocks * BLOCK_BYTES;
    const uint64_t model_size = w0_bytes + w1_bytes;

    unsigned char *model = malloc(model_size);
    float *x = malloc((uint64_t)IN_DIM * sizeof(float));
    float *w64_out0 = malloc((uint64_t)OUT0_DIM * sizeof(float));
    float *w64_out1 = malloc((uint64_t)OUT1_DIM * sizeof(float));
    float *w32_out0 = malloc((uint64_t)OUT0_DIM * sizeof(float));
    float *w32_out1 = malloc((uint64_t)OUT1_DIM * sizeof(float));
    if (!model || !x || !w64_out0 || !w64_out1 || !w32_out0 || !w32_out1) {
        fprintf(stderr, "oom\n");
        return 1;
    }

    unsigned char *w0 = model;
    unsigned char *w1 = model + w0_bytes;
    make_weights(w0, OUT0_DIM, n_blocks);
    make_weights(w1, OUT1_DIM, n_blocks);
    for (uint32_t i = 0; i < IN_DIM; i++) x[i] = next_uniform();

    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }
    if (!ds4_gpu_set_model_map(model, model_size)) { fprintf(stderr, "set_model_map failed\n"); return 1; }
    if (!ds4_gpu_cache_model_range(model, model_size, 0, model_size, "test q8 pair weights")) {
        fprintf(stderr, "cache_model_range failed\n");
        return 1;
    }

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)IN_DIM * sizeof(float));
    ds4_gpu_tensor *o0t = ds4_gpu_tensor_alloc((uint64_t)OUT0_DIM * sizeof(float));
    ds4_gpu_tensor *o1t = ds4_gpu_tensor_alloc((uint64_t)OUT1_DIM * sizeof(float));
    if (!xt || !o0t || !o1t) { fprintf(stderr, "tensor alloc failed\n"); return 1; }
    if (!ds4_gpu_tensor_write(xt, 0, x, (uint64_t)IN_DIM * sizeof(float))) {
        fprintf(stderr, "tensor write failed\n");
        return 1;
    }

    unsetenv("DS4_ROCM_DISABLE_PAIR_W64");
    if (!run_once(w0, w1, n_blocks, x, xt, o0t, o1t, w64_out0, w64_out1,
                  model, model_size, 0, w0_bytes)) {
        return 1;
    }

    setenv("DS4_ROCM_DISABLE_PAIR_W64", "1", 1);
    if (!run_once(w0, w1, n_blocks, x, xt, o0t, o1t, w32_out0, w32_out1,
                  model, model_size, 0, w0_bytes)) {
        return 1;
    }
    unsetenv("DS4_ROCM_DISABLE_PAIR_W64");

    double rms = 0.0;
    for (uint32_t i = 0; i < OUT0_DIM; i++) rms += (double)w64_out0[i] * w64_out0[i];
    for (uint32_t i = 0; i < OUT1_DIM; i++) rms += (double)w64_out1[i] * w64_out1[i];
    rms = sqrt(rms / (double)(OUT0_DIM + OUT1_DIM));
    if (!(rms > 0.0)) { fprintf(stderr, "FAIL: output is all zeros\n"); return 1; }

    double worst_ref = 0.0, worst_w32 = 0.0;
    for (uint32_t r = 0; r < OUT0_DIM; r++) {
        const double ref = cpu_ref_row(w0, n_blocks, x, r);
        const double d_ref = fabs((double)w64_out0[r] - ref);
        if (d_ref > worst_ref) worst_ref = d_ref;
        const double d_w32 = fabs((double)w64_out0[r] - (double)w32_out0[r]);
        if (d_w32 > worst_w32) worst_w32 = d_w32;
    }
    for (uint32_t r = 0; r < OUT1_DIM; r++) {
        const double ref = cpu_ref_row(w1, n_blocks, x, r);
        const double d_ref = fabs((double)w64_out1[r] - ref);
        if (d_ref > worst_ref) worst_ref = d_ref;
        const double d_w32 = fabs((double)w64_out1[r] - (double)w32_out1[r]);
        if (d_w32 > worst_w32) worst_w32 = d_w32;
    }

    printf("output rms                 : %.4f\n", rms);
    printf("max |w64 - cpu f64ref|     : %.6f  (%.4f%% of rms)\n",
           worst_ref, 100.0 * worst_ref / rms);
    printf("max |w64 - w32|            : %.6f  (%.4f%% of rms)\n",
           worst_w32, 100.0 * worst_w32 / rms);

    /* Both kernels do the same f32 multiply-adds in different accumulation
     * order (32-wide-per-block-then-combine vs 64-wide-per-row), so only FP
     * summation-order noise should separate them from each other or from the
     * reference -- nowhere near the ~100% a block-pairing/lane-mapping bug
     * would produce. */
    int ok = 1;
    if (!(worst_ref < 1.0e-3 * rms)) {
        fprintf(stderr, "FAIL: w64 kernel disagrees with the CPU reference\n");
        ok = 0;
    }
    if (!(worst_w32 < 1.0e-3 * rms)) {
        fprintf(stderr, "FAIL: w64 kernel disagrees with the w32 kernel\n");
        ok = 0;
    }

    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(o0t);
    ds4_gpu_tensor_free(o1t);
    ds4_gpu_cleanup();
    free(model); free(x);
    free(w64_out0); free(w64_out1); free(w32_out0); free(w32_out1);

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
