/* Numerical check for the matrix-core Q8_0 batched GEMM on ROCm.
 *
 * ds4_gpu_matmul_q8_0_tensor() routes large prefill batches to a matrix-core
 * kernel -- WMMA on RDNA3/RDNA4, MFMA on CDNA -- and everything else to the
 * portable "sharedx" kernel.  Those kernels distribute a 16x16x16 fragment
 * across lanes very differently, so an indexing slip in either one produces
 * plausible-looking but wrong activations rather than a crash.
 *
 * This drives the public entry point at a shape that selects the matrix-core
 * path, then re-runs the identical call in quality mode, which forces the
 * portable path, and compares.  It also compares against a CPU reference that
 * models the f16 rounding the matrix-core path applies to its operands, so the
 * test stays tight enough to catch a single misplaced lane index.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

#define IN_DIM   2048u
#define OUT_DIM  1024u
#define N_TOK     256u   /* >= 256 is what arms the matrix-core path */

#define QK  32u
#define BLOCK_BYTES 34u

static uint32_t g_rng = 0x2b7e1516u;

static float next_uniform(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)((g_rng >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

/* IEEE binary16 round-to-nearest-even, and back.  The device rounds both the
 * dequantized weights and the staged activations to f16 before the matrix
 * core sees them; the reference has to do the same to be comparable. */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;

    if (((x >> 23) & 0xffu) == 0xffu) {          /* Inf / NaN */
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);   /* overflow -> Inf */
    if (exp <= 0) {                                        /* subnormal / zero */
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

static float round_f16(float f) { return f16_to_f32(f32_to_f16(f)); }

int main(void) {
    const uint32_t n_blocks = IN_DIM / QK;
    const uint64_t w_bytes = (uint64_t)OUT_DIM * n_blocks * BLOCK_BYTES;
    const uint64_t x_count = (uint64_t)N_TOK * IN_DIM;
    const uint64_t o_count = (uint64_t)N_TOK * OUT_DIM;

    unsigned char *w = malloc(w_bytes);
    float *x = malloc(x_count * sizeof(float));
    float *fast = malloc(o_count * sizeof(float));
    float *portable = malloc(o_count * sizeof(float));
    if (!w || !x || !fast || !portable) { fprintf(stderr, "oom\n"); return 1; }

    /* Random Q8_0 weights: one f16 scale plus 32 int8 per block. */
    for (uint32_t r = 0; r < OUT_DIM; r++) {
        for (uint32_t b = 0; b < n_blocks; b++) {
            unsigned char *blk = w + ((uint64_t)r * n_blocks + b) * BLOCK_BYTES;
            const uint16_t d = f32_to_f16(0.01f + 0.02f * fabsf(next_uniform()));
            memcpy(blk, &d, 2);
            for (uint32_t j = 0; j < QK; j++) {
                blk[2 + j] = (unsigned char)(signed char)(int)(next_uniform() * 100.0f);
            }
        }
    }
    for (uint64_t i = 0; i < x_count; i++) x[i] = next_uniform();

    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 1; }
    if (!ds4_gpu_set_model_map(w, w_bytes)) { fprintf(stderr, "set_model_map failed\n"); return 1; }
    if (!ds4_gpu_cache_model_range(w, w_bytes, 0, w_bytes, "test q8 weights")) {
        fprintf(stderr, "cache_model_range failed\n");
        return 1;
    }

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc(o_count * sizeof(float));
    if (!xt || !ot) { fprintf(stderr, "tensor alloc failed\n"); return 1; }
    if (!ds4_gpu_tensor_write(xt, 0, x, x_count * sizeof(float))) {
        fprintf(stderr, "tensor write failed\n");
        return 1;
    }

    /* Matrix-core path. */
    ds4_gpu_set_quality(false);
    if (!ds4_gpu_matmul_q8_0_tensor(ot, w, w_bytes, 0, IN_DIM, OUT_DIM, xt, N_TOK)) {
        fprintf(stderr, "matmul (fast) failed\n");
        return 1;
    }
    if (!ds4_gpu_tensor_read(ot, 0, fast, o_count * sizeof(float))) {
        fprintf(stderr, "tensor read failed\n");
        return 1;
    }

    /* Same call, quality mode: takes the portable kernel instead. */
    ds4_gpu_set_quality(true);
    if (!ds4_gpu_matmul_q8_0_tensor(ot, w, w_bytes, 0, IN_DIM, OUT_DIM, xt, N_TOK)) {
        fprintf(stderr, "matmul (portable) failed\n");
        return 1;
    }
    if (!ds4_gpu_tensor_read(ot, 0, portable, o_count * sizeof(float))) {
        fprintf(stderr, "tensor read failed\n");
        return 1;
    }

    /* CPU reference over f16-rounded operands, on a slice of rows. */
    double worst_ref = 0.0, worst_port = 0.0, rms = 0.0;
    for (uint64_t i = 0; i < o_count; i++) rms += (double)fast[i] * fast[i];
    rms = sqrt(rms / (double)o_count);
    if (!(rms > 0.0)) { fprintf(stderr, "FAIL: output is all zeros\n"); return 1; }

    for (uint32_t t = 0; t < 32u; t++) {
        for (uint32_t r = 0; r < OUT_DIM; r += 7u) {
            double acc = 0.0;
            for (uint32_t b = 0; b < n_blocks; b++) {
                const unsigned char *blk = w + ((uint64_t)r * n_blocks + b) * BLOCK_BYTES;
                uint16_t dbits;
                memcpy(&dbits, blk, 2);
                const float d = f16_to_f32(dbits);
                for (uint32_t j = 0; j < QK; j++) {
                    const float wv = round_f16(d * (float)(int)(signed char)blk[2 + j]);
                    const float xv = round_f16(x[(uint64_t)t * IN_DIM + b * QK + j]);
                    acc += (double)wv * (double)xv;
                }
            }
            const double got = fast[(uint64_t)t * OUT_DIM + r];
            const double d_ref = fabs(got - acc);
            if (d_ref > worst_ref) worst_ref = d_ref;
        }
    }
    for (uint64_t i = 0; i < o_count; i++) {
        const double d = fabs((double)fast[i] - (double)portable[i]);
        if (d > worst_port) worst_port = d;
    }

    printf("output rms                    : %.4f\n", rms);
    printf("max |matrix-core - cpu f16ref|: %.5f  (%.3f%% of rms)\n",
           worst_ref, 100.0 * worst_ref / rms);
    printf("max |matrix-core - portable|  : %.5f  (%.3f%% of rms)\n",
           worst_port, 100.0 * worst_port / rms);

    /* The reference models f16 operands exactly, so only accumulation order
     * differs; 1% of rms is far below the ~100% a layout bug produces.  The
     * portable kernel keeps f32 activations, so it legitimately differs by the
     * f16 quantization of x, which is a larger but still small budget. */
    int ok = 1;
    if (!(worst_ref < 0.01 * rms)) {
        fprintf(stderr, "FAIL: matrix-core kernel disagrees with the CPU reference\n");
        ok = 0;
    }
    if (!(worst_port < 0.05 * rms)) {
        fprintf(stderr, "FAIL: matrix-core kernel disagrees with the portable kernel\n");
        ok = 0;
    }

    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(ot);
    ds4_gpu_cleanup();
    free(w); free(x); free(fast); free(portable);

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
