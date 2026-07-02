#include "ds4_gpu.h"

/* Synthetic ROCm Q8_0 matmul correctness check. The default build runs the
 * CDNA MFMA path, then the Makefile target reruns with that path disabled.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint64_t round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1u) & ~(align - 1u);
}

static uint16_t float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void fill_q8_0_weights(uint8_t *weights,
                              uint32_t in_dim,
                              uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                const float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = float_to_f16(amax / 127.0f);
            const float scale = f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

static void fill_activations(float *x, uint32_t n_tok, uint32_t in_dim) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
}

static void reference_q8_0(const uint8_t *weights,
                           const float *x,
                           float *ref_f32,
                           float *ref_f16,
                           uint32_t n_tok,
                           uint32_t in_dim,
                           uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float acc_f32 = 0.0f;
            float acc_f16 = 0.0f;
            for (uint32_t b = 0; b < blocks; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    const float xv = x[(uint64_t)t * in_dim + b * 32u + i];
                    const float w_f32 = scale * (float)qs[i];
                    const float w_f16 = f16_to_f32(float_to_f16(w_f32));
                    const float x_f16 = f16_to_f32(float_to_f16(xv));
                    acc_f32 += w_f32 * xv;
                    acc_f16 += w_f16 * x_f16;
                }
            }
            ref_f32[(uint64_t)t * out_dim + o] = acc_f32;
            ref_f16[(uint64_t)t * out_dim + o] = acc_f16;
        }
    }
}

static int check_errors(const float *got,
                        const float *ref,
                        uint64_t n,
                        float *out_max_abs,
                        float *out_rms) {
    double sumsq = 0.0;
    float max_abs = 0.0f;
    int bad = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            bad = 1;
            continue;
        }
        const float err = fabsf(got[i] - ref[i]);
        if (err > max_abs) max_abs = err;
        sumsq += (double)err * (double)err;
    }
    *out_max_abs = max_abs;
    *out_rms = (float)sqrt(sumsq / (double)n);
    return bad ? 1 : 0;
}

int main(void) {
    const uint32_t in_dim = 1024;
    const uint32_t out_dim = 1024;
    const uint32_t n_tok = 32;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    if (posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) != 0 ||
        !weights_raw) {
        fprintf(stderr, "rocm-q8-mfma-correctness: failed to allocate weights\n");
        return 1;
    }
    memset(weights_raw, 0, (size_t)weight_alloc);
    fill_q8_0_weights((uint8_t *)weights_raw, in_dim, out_dim);

    float *x_host = (float *)malloc((size_t)x_bytes);
    float *out_host = (float *)malloc((size_t)out_bytes);
    float *ref_f32 = (float *)malloc((size_t)out_bytes);
    float *ref_f16 = (float *)malloc((size_t)out_bytes);
    if (!x_host || !out_host || !ref_f32 || !ref_f16) {
        fprintf(stderr, "rocm-q8-mfma-correctness: failed to allocate host buffers\n");
        free(ref_f16);
        free(ref_f32);
        free(out_host);
        free(x_host);
        free(weights_raw);
        return 1;
    }
    fill_activations(x_host, n_tok, in_dim);
    for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim; i++) out_host[i] = 12345.0f;
    reference_q8_0((const uint8_t *)weights_raw, x_host, ref_f32, ref_f16,
                   n_tok, in_dim, out_dim);

    if (!ds4_gpu_init()) {
        fprintf(stderr, "rocm-q8-mfma-correctness: ROCm backend unavailable\n");
        free(ref_f16);
        free(ref_f32);
        free(out_host);
        free(x_host);
        free(weights_raw);
        return 1;
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    int rc = 1;
    if (x && out &&
        ds4_gpu_tensor_write(x, 0, x_host, x_bytes) &&
        ds4_gpu_tensor_write(out, 0, out_host, out_bytes) &&
        ds4_gpu_set_model_map(weights_raw, weight_alloc)) {
        ds4_gpu_set_quality(false);
        if (ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                       in_dim, out_dim, x, n_tok) &&
            ds4_gpu_tensor_read(out, 0, out_host, out_bytes)) {
            float max_f32 = 0.0f, rms_f32 = 0.0f;
            float max_f16 = 0.0f, rms_f16 = 0.0f;
            const uint64_t n = (uint64_t)n_tok * out_dim;
            int bad_f32 = check_errors(out_host, ref_f32, n, &max_f32, &rms_f32);
            int bad_f16 = check_errors(out_host, ref_f16, n, &max_f16, &rms_f16);
            const int use_f16 = rms_f16 < rms_f32;
            const float best_max = use_f16 ? max_f16 : max_f32;
            const float best_rms = use_f16 ? rms_f16 : rms_f32;
            fprintf(stderr,
                    "rocm-q8-mfma-correctness: n_tok=%u in=%u out=%u ref=%s "
                    "max_abs=%g rms=%g f32(max=%g rms=%g) f16(max=%g rms=%g)\n",
                    n_tok, in_dim, out_dim, use_f16 ? "f16" : "f32",
                    best_max, best_rms, max_f32, rms_f32, max_f16, rms_f16);
            if (!bad_f32 && !bad_f16 && best_max < 0.12f && best_rms < 0.02f) rc = 0;
        }
    }

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(ref_f16);
    free(ref_f32);
    free(out_host);
    free(x_host);
    free(weights_raw);
    return rc;
}
