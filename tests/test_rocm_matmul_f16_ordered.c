/* Numerical check for the wave64-native matmul_f16_ordered_chunks_w64_kernel.
 *
 * ds4_gpu_matmul_f16_tensor() with n_tok == 1 and the router-projection shape
 * (in_dim=4096, out_dim=256, matching real usage) dispatches to
 * matmul_f16_ordered_chunks_kernel, which the w32 build launches with a
 * 32-thread block -- on wave64 (CDNA) hardware that leaves the other 32
 * lanes of every dispatched wavefront completely idle (not a two-rows
 * tradeoff like the other w32 kernels this session touched, just unused
 * capacity). The w64 counterpart divides the same row's dot product across
 * all 64 threads instead.
 *
 * DS4_ROCM_DISABLE_F16_ORDERED_W64 forces the same call back onto the
 * original w32 kernel (unmodified) for a direct kernel-vs-kernel
 * cross-check in addition to the CPU reference.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

#define IN_DIM  4096u
#define OUT_DIM 256u

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

static double cpu_ref_row(const uint16_t *w, uint32_t in_dim, const float *x, uint32_t row) {
    double acc = 0.0;
    const uint16_t *wr = w + (uint64_t)row * in_dim;
    for (uint32_t i = 0; i < in_dim; i++) acc += (double)f16_to_f32(wr[i]) * (double)x[i];
    return acc;
}

static int run_once(ds4_gpu_tensor *xt, ds4_gpu_tensor *ot, float *out,
                     const void *model_map, uint64_t model_size) {
    if (!ds4_gpu_matmul_f16_tensor(ot, model_map, model_size, 0, IN_DIM, OUT_DIM, xt, 1)) {
        fprintf(stderr, "matmul f16 ordered failed\n");
        return 0;
    }
    if (!ds4_gpu_tensor_read(ot, 0, out, (uint64_t)OUT_DIM * sizeof(float))) {
        fprintf(stderr, "tensor read failed\n");
        return 0;
    }
    return 1;
}

int main(void) {
    const uint64_t model_size = (uint64_t)OUT_DIM * IN_DIM * sizeof(uint16_t);

    uint16_t *w = malloc(model_size);
    float *x = malloc((uint64_t)IN_DIM * sizeof(float));
    float *w64_out = malloc((uint64_t)OUT_DIM * sizeof(float));
    float *w32_out = malloc((uint64_t)OUT_DIM * sizeof(float));
    if (!w || !x || !w64_out || !w32_out) {
        fprintf(stderr, "oom\n");
        return 1;
    }

    for (uint64_t i = 0; i < (uint64_t)OUT_DIM * IN_DIM; i++) w[i] = f32_to_f16(0.02f * next_uniform());
    for (uint32_t i = 0; i < IN_DIM; i++) x[i] = next_uniform();

    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }
    if (!ds4_gpu_set_model_map(w, model_size)) { fprintf(stderr, "set_model_map failed\n"); return 1; }
    if (!ds4_gpu_cache_model_range(w, model_size, 0, model_size, "test f16 ordered weights")) {
        fprintf(stderr, "cache_model_range failed\n");
        return 1;
    }

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)IN_DIM * sizeof(float));
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc((uint64_t)OUT_DIM * sizeof(float));
    if (!xt || !ot) { fprintf(stderr, "tensor alloc failed\n"); return 1; }
    if (!ds4_gpu_tensor_write(xt, 0, x, (uint64_t)IN_DIM * sizeof(float))) {
        fprintf(stderr, "tensor write failed\n");
        return 1;
    }

    unsetenv("DS4_ROCM_DISABLE_F16_ORDERED_W64");
    if (!run_once(xt, ot, w64_out, w, model_size)) return 1;

    setenv("DS4_ROCM_DISABLE_F16_ORDERED_W64", "1", 1);
    if (!run_once(xt, ot, w32_out, w, model_size)) return 1;
    unsetenv("DS4_ROCM_DISABLE_F16_ORDERED_W64");

    double rms = 0.0;
    for (uint32_t i = 0; i < OUT_DIM; i++) rms += (double)w64_out[i] * w64_out[i];
    rms = sqrt(rms / (double)OUT_DIM);
    if (!(rms > 0.0)) { fprintf(stderr, "FAIL: output is all zeros\n"); return 1; }

    double worst_ref = 0.0, worst_w32 = 0.0;
    for (uint32_t r = 0; r < OUT_DIM; r++) {
        const double ref = cpu_ref_row(w, IN_DIM, x, r);
        const double d_ref = fabs((double)w64_out[r] - ref);
        if (d_ref > worst_ref) worst_ref = d_ref;
        const double d_w32 = fabs((double)w64_out[r] - (double)w32_out[r]);
        if (d_w32 > worst_w32) worst_w32 = d_w32;
    }

    printf("output rms                 : %.4f\n", rms);
    printf("max |w64 - cpu f64ref|     : %.6f  (%.4f%% of rms)\n",
           worst_ref, 100.0 * worst_ref / rms);
    printf("max |w64 - w32|            : %.6f  (%.4f%% of rms)\n",
           worst_w32, 100.0 * worst_w32 / rms);

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
    ds4_gpu_tensor_free(ot);
    ds4_gpu_cleanup();
    free(w); free(x); free(w64_out); free(w32_out);

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
