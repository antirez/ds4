/* Metal kernel tests for Qwen3.8-Flash-Next (metal/qwen4.metal): every kernel
 * at the release and mini model shapes against a double-precision reference.
 * Build: make test-qwen4-kernels */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint32_t g_rng = 0x9e3779b9u;

static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xffffffu) / 8388608.0f) - 1.0f;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void check_close(const char *what, const float *got, const double *ref, uint64_t n, double tol) {
    double worst = 0.0, scale = 1e-6;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "%s: non-finite value at %llu\n", what, (unsigned long long)i);
            exit(1);
        }
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) { worst = d; worst_i = i; }
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    if (worst > tol * scale) {
        fprintf(stderr, "%s: max|d| %.3e (rel %.3e) at %llu: got %.6f ref %.6f\n",
                what, worst, worst / scale, (unsigned long long)worst_i, got[worst_i], ref[worst_i]);
        exit(1);
    }
    printf("  %-44s ok  max|d|=%.2e (scale %.2e)\n", what, worst, scale);
}

static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } v = { f };
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    union { uint32_t u; float f; } v;
    if (exp == 0) {
        if (mant == 0) { v.u = sign; return v.f; }
        exp = 127 - 15 + 1;
        while (!(mant & 0x400u)) { mant <<= 1; exp--; }
        mant &= 0x3ffu;
        v.u = sign | (exp << 23) | (mant << 13);
        return v.f;
    }
    if (exp == 31) { v.u = sign | 0x7f800000u | (mant << 13); return v.f; }
    v.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return v.f;
}

static double sigmoid_d(double x) { return x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x)); }
static double silu_d(double x) { return x * sigmoid_d(x); }
static double softplus_d(double x) { return x > 20.0 ? x : log1p(exp(x)); }

/* ---- weight arena ---- */

/* anonymous mmap standing in for the model map; weights are appended, never freed */
typedef struct {
    uint8_t *base;
    uint64_t size;
    uint64_t used;
} arena_t;

static uint64_t arena_alloc(arena_t *a, uint64_t bytes) {
    const uint64_t off = (a->used + 63u) & ~63ull;
    if (off + bytes > a->size) {
        fprintf(stderr, "arena exhausted\n");
        exit(1);
    }
    a->used = off + bytes;
    return off;
}

/* f32 weights with a double shadow for the reference */
static uint64_t arena_f32(arena_t *a, uint64_t n, double **shadow, float lo, float hi) {
    const uint64_t off = arena_alloc(a, n * sizeof(float));
    float *w = (float *)(a->base + off);
    *shadow = malloc(n * sizeof(double));
    for (uint64_t i = 0; i < n; i++) {
        w[i] = lo + (hi - lo) * (0.5f * frand() + 0.5f);
        (*shadow)[i] = w[i];
    }
    return off;
}

static uint64_t arena_f16(arena_t *a, uint64_t n, double **shadow, float scale) {
    const uint64_t off = arena_alloc(a, n * sizeof(uint16_t));
    uint16_t *w = (uint16_t *)(a->base + off);
    *shadow = malloc(n * sizeof(double));
    for (uint64_t i = 0; i < n; i++) {
        w[i] = f32_to_f16(scale * frand());
        (*shadow)[i] = f16_to_f32(w[i]);
    }
    return off;
}

static uint64_t arena_bf16(arena_t *a, uint64_t n, double **shadow, float scale) {
    const uint64_t off = arena_alloc(a, n * sizeof(uint16_t));
    uint16_t *w = (uint16_t *)(a->base + off);
    *shadow = malloc(n * sizeof(double));
    for (uint64_t i = 0; i < n; i++) {
        const float v = scale * frand();
        uint32_t u;
        memcpy(&u, &v, 4);
        w[i] = (uint16_t)(u >> 16);
        const uint32_t back = (uint32_t)w[i] << 16;
        float r;
        memcpy(&r, &back, 4);
        (*shadow)[i] = r;
    }
    return off;
}

/* q4_0 rows: 18-byte blocks of 32 (f16 scale, 16 nibble bytes; low nibbles first) */
static uint64_t arena_q4_0(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 32;
    const uint64_t off = arena_alloc(a, rows * blocks * 18u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f, maxv = 0.0f;
            for (int j = 0; j < 32; j++) {
                vals[j] = scale * frand();
                if (fabsf(vals[j]) > amax) { amax = fabsf(vals[j]); maxv = vals[j]; }
            }
            const float d = maxv / -8.0f;
            const uint16_t dh = f32_to_f16(d);
            const float dq = f16_to_f32(dh);
            const float id = dq ? 1.0f / dq : 0.0f;
            uint8_t *blk = w + (r * blocks + b) * 18u;
            memcpy(blk, &dh, 2);
            for (int j = 0; j < 16; j++) {
                int q0 = (int)(vals[j] * id + 8.5f), q1 = (int)(vals[j + 16] * id + 8.5f);
                if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
                if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
                blk[2 + j] = (uint8_t)(q0 | (q1 << 4));
                (*shadow)[r * cols + b * 32 + j] = dq * (q0 - 8);
                (*shadow)[r * cols + b * 32 + 16 + j] = dq * (q1 - 8);
            }
        }
    }
    return off;
}

/* q4_K rows: 144-byte super-blocks of 256 (d, dmin, 6-bit scales/mins, nibbles) */
static uint64_t arena_q4_K(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 256;
    const uint64_t off = arena_alloc(a, rows * blocks * 144u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (r * blocks + b) * 144u;
            /* per-group scale/min in 6 bits, block d/dmin as f16 */
            const float d = scale / 63.0f / 15.0f, dmin = d;
            const uint16_t dh = f32_to_f16(d), mh = f32_to_f16(dmin);
            const float dq = f16_to_f32(dh), mq = f16_to_f32(mh);
            memcpy(blk, &dh, 2);
            memcpy(blk + 2, &mh, 2);
            uint8_t sc[8], mn[8];
            for (int g = 0; g < 8; g++) {
                sc[g] = (uint8_t)(1 + (int)(62.0f * (0.5f * frand() + 0.5f)));
                mn[g] = (uint8_t)((int)(63.0f * (0.5f * frand() + 0.5f)));
            }
            uint8_t *s = blk + 4;
            for (int g = 0; g < 4; g++) { s[g] = sc[g] & 63; s[g + 4] = mn[g] & 63; }
            for (int g = 4; g < 8; g++) {
                s[g + 4] = (uint8_t)((sc[g] & 0xF) | ((mn[g] & 0xF) << 4));
                s[g - 4] |= (uint8_t)((sc[g] >> 4) << 6);
                s[g] |= (uint8_t)((mn[g] >> 4) << 6);
            }
            memset(blk + 16, 0, 128);
            for (int g = 0; g < 8; g++) {
                for (int j = 0; j < 32; j++) {
                    const int q = (int)(15.0f * (0.5f * frand() + 0.5f));
                    blk[16 + (g >> 1) * 32 + j] |= (uint8_t)(q << ((g & 1) * 4));
                    (*shadow)[r * cols + b * 256 + g * 32 + j] = dq * sc[g] * q - mq * mn[g];
                }
            }
        }
    }
    return off;
}

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

/* QWEN4_DUMP_QUANT=DIR: write the raw blocks and the shadow values for offline checks */
static void dump_quant(const char *type, const uint8_t *raw, uint64_t raw_bytes, const double *shadow, uint64_t n) {
    const char *dir = getenv("QWEN4_DUMP_QUANT");
    if (!dir) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, type);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(raw, 1, raw_bytes, f); fclose(f); }
    snprintf(path, sizeof(path), "%s/%s.shadow", dir, type);
    f = fopen(path, "wb");
    if (f) { fwrite(shadow, sizeof(double), n, f); fclose(f); }
}

/* q2_K rows: 84-byte super-blocks of 256 (16 scale/min nibbles, 64 packed bytes, d, dmin) */
static uint64_t arena_q2_K(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 256;
    const uint64_t off = arena_alloc(a, rows * blocks * 84u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (r * blocks + b) * 84u;
            const uint16_t dh = f32_to_f16(scale / 15.0f / 3.0f), mh = dh;
            const float dq = f16_to_f32(dh), mq = f16_to_f32(mh);
            memcpy(blk + 80, &dh, 2);
            memcpy(blk + 82, &mh, 2);
            for (int g = 0; g < 16; g++) {
                const int sc = (int)(15.0f * (0.5f * frand() + 0.5f)), mn = (int)(15.0f * (0.5f * frand() + 0.5f));
                blk[g] = (uint8_t)(sc | (mn << 4));
            }
            memset(blk + 16, 0, 64);
            for (int g = 0; g < 16; g++) {
                const int q_base = 32 * (g / 8) + 16 * (g & 1), shift = ((g / 2) & 3) * 2;
                for (int l = 0; l < 16; l++) {
                    const int q = (int)(3.0f * (0.5f * frand() + 0.5f)) & 3;
                    blk[16 + q_base + l] |= (uint8_t)(q << shift);
                    (*shadow)[r * cols + b * 256 + g * 16 + l] = dq * (blk[g] & 0xF) * q - mq * (blk[g] >> 4);
                }
            }
        }
    }
    dump_quant("q2_K", w, rows * blocks * 84u, *shadow, rows * cols);
    return off;
}

/* iq2_xxs rows: 66-byte super-blocks of 256 (d, then per 32: 4 grid indices, 4x7 sign bits, 4-bit scale) */
static uint64_t arena_iq2_xxs(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 256;
    const uint64_t off = arena_alloc(a, rows * blocks * 66u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (r * blocks + b) * 66u;
            const uint16_t dh = f32_to_f16(scale / 16.0f);
            const float dq = f16_to_f32(dh);
            memcpy(blk, &dh, 2);
            for (int ib32 = 0; ib32 < 8; ib32++) {
                uint32_t aux_g = 0, aux_s = 0;
                for (int j = 0; j < 4; j++) {
                    aux_g |= (uint32_t)(rand() & 0xFF) << (8 * j);
                    aux_s |= (uint32_t)(rand() & 0x7F) << (7 * j);
                }
                aux_s |= (uint32_t)(rand() & 0xF) << 28;
                uint16_t q2[4] = { (uint16_t)aux_g, (uint16_t)(aux_g >> 16), (uint16_t)aux_s, (uint16_t)(aux_s >> 16) };
                memcpy(blk + 2 + ib32 * 8, q2, 8);
                const float dl = dq * (0.5f + (float)(aux_s >> 28)) * 0.25f;
                for (int j = 0; j < 4; j++) {
                    const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + ((aux_g >> (8 * j)) & 0xFF));
                    const uint32_t signs = ksigns_iq2xs[(aux_s >> (7 * j)) & 127];
                    for (int i = 0; i < 8; i++) {
                        (*shadow)[r * cols + b * 256 + ib32 * 32 + j * 8 + i] = dl * grid[i] * (((signs >> i) & 1) ? -1.0 : 1.0);
                    }
                }
            }
        }
    }
    dump_quant("iq2_xxs", w, rows * blocks * 66u, *shadow, rows * cols);
    return off;
}

/* q8_0 rows of `cols` elements (cols % 32 == 0), `rows` rows */
static uint64_t arena_q8_0(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 32;
    const uint64_t off = arena_alloc(a, rows * blocks * 34u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                vals[j] = scale * frand();
                if (fabsf(vals[j]) > amax) amax = fabsf(vals[j]);
            }
            const float d = amax / 127.0f;
            const uint16_t dh = f32_to_f16(d);
            const float dq = f16_to_f32(dh);
            uint8_t *blk = w + (r * blocks + b) * 34u;
            memcpy(blk, &dh, 2);
            for (int j = 0; j < 32; j++) {
                int q = (int)lrintf(d > 0 ? vals[j] / d : 0.0f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                ((int8_t *)blk)[2 + j] = (int8_t)q;
                (*shadow)[r * cols + b * 32 + j] = (double)dq * q;
            }
        }
    }
    return off;
}

static ds4_gpu_tensor *upload(const float *data, uint64_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    require_ok(t != NULL, "tensor alloc");
    if (data) require_ok(ds4_gpu_tensor_write(t, 0, data, n * sizeof(float)), "tensor write");
    else require_ok(ds4_gpu_tensor_fill_f32(t, 0.0f, n), "tensor fill");
    return t;
}

static float *download(const ds4_gpu_tensor *t, uint64_t n) {
    float *out = malloc(n * sizeof(float));
    require_ok(ds4_gpu_tensor_read(t, 0, out, n * sizeof(float)), "tensor read");
    return out;
}

static void check_tensor(const char *what, const ds4_gpu_tensor *t, const double *ref, uint64_t n, double tol) {
    float *got = download(t, n);
    check_close(what, got, ref, n, tol);
    free(got);
}

static float *rand_vec(uint64_t n, float scale) {
    float *v = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) v[i] = scale * frand();
    return v;
}

/* ---- hyper-connections ---- */

static void test_hc(arena_t *a, uint32_t E, uint32_t rank, uint32_t T, uint32_t wtype) {
    const bool f16 = wtype == 1u, q8 = wtype == 8u;
    const uint32_t hc = 4, dim = E * hc, CH = DS4_QWEN4_HC_CHUNKS;
    const float eps = 1e-6f;
    double *g_gamma, *g_down, *g_up, *g_inj;
    const uint64_t gamma_off = arena_f32(a, dim, &g_gamma, 0.5f, 1.5f);
    const uint64_t down_off = q8 ? arena_q8_0(a, rank, dim, &g_down, 0.05f)
                            : f16 ? arena_f16(a, (uint64_t)rank * dim, &g_down, 0.05f)
                                  : arena_f32(a, (uint64_t)rank * dim, &g_down, -0.05f, 0.05f);
    /* q8 up rows need rank % 32; smaller ranks keep f16 like the converter does */
    const uint32_t up_type = q8 && (rank % 32) == 0 ? 8u : q8 ? 1u : wtype;
    const uint64_t up_off = up_type == 8u ? arena_q8_0(a, dim, rank, &g_up, 0.2f)
                          : up_type == 1u ? arena_f16(a, (uint64_t)dim * rank, &g_up, 0.2f)
                                          : arena_f32(a, (uint64_t)dim * rank, &g_up, -0.2f, 0.2f);
    const uint64_t inj_off = q8 ? arena_q8_0(a, hc, dim, &g_inj, 0.05f)
                           : f16 ? arena_f16(a, (uint64_t)hc * dim, &g_inj, 0.05f)
                                 : arena_f32(a, (uint64_t)hc * dim, &g_inj, -0.05f, 0.05f);
    float *R = rand_vec((uint64_t)T * dim, 1.0f);
    float *blk = rand_vec((uint64_t)T * E, 1.0f);

    double *xn = malloc((uint64_t)T * dim * sizeof(double));
    double *lo = malloc((uint64_t)T * rank * sizeof(double));
    double *part = malloc((uint64_t)T * hc * CH * hc * sizeof(double));
    double *mixed = malloc((uint64_t)T * E * sizeof(double));
    double *R2 = malloc((uint64_t)T * dim * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < hc; s++) {
            double ss = 0.0;
            for (uint32_t i = 0; i < E; i++) ss += (double)R[t * dim + s * E + i] * R[t * dim + s * E + i];
            const double inv = 1.0 / sqrt(ss / E + eps);
            for (uint32_t i = 0; i < E; i++) xn[t * dim + s * E + i] = R[t * dim + s * E + i] * inv * g_gamma[s * E + i];
        }
        for (uint32_t r = 0; r < rank; r++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < dim; i++) acc += g_down[(uint64_t)r * dim + i] * xn[t * dim + i];
            lo[t * rank + r] = acc;   /* raw: gate_mix applies silu(./hc) */
        }
        double inj[4];
        const uint32_t per = (E + CH - 1) / CH;
        for (uint32_t j = 0; j < hc; j++) {
            double tot = 0.0;
            for (uint32_t s = 0; s < hc; s++) {
                for (uint32_t c = 0; c < CH; c++) {
                    double acc = 0.0;
                    for (uint32_t i = c * per; i < E && i < (c + 1) * per; i++)
                        acc += g_inj[(uint64_t)j * dim + s * E + i] * xn[t * dim + s * E + i];
                    part[((uint64_t)t * hc * CH + s * CH + c) * hc + j] = acc;
                    tot += acc;
                }
            }
            inj[j] = 2.0 * sigmoid_d(tot / hc);
        }
        for (uint32_t d = 0; d < E; d++) {
            double m = 0.0;
            for (uint32_t s = 0; s < hc; s++) {
                double acc = 0.0;
                for (uint32_t r = 0; r < rank; r++) acc += g_up[(uint64_t)(s * E + d) * rank + r] * silu_d(lo[t * rank + r] / hc);
                m += sigmoid_d(acc) * xn[t * dim + s * E + d];
            }
            mixed[t * E + d] = m / hc;
        }
        for (uint32_t s = 0; s < hc; s++)
            for (uint32_t d = 0; d < E; d++)
                R2[t * dim + s * E + d] = R[t * dim + s * E + d] + inj[s] * blk[t * E + d];
    }

    ds4_gpu_tensor *gR = upload(R, (uint64_t)T * dim);
    ds4_gpu_tensor *gxn = upload(NULL, (uint64_t)T * dim);
    ds4_gpu_tensor *glo = upload(NULL, (uint64_t)T * rank);
    ds4_gpu_tensor *ginj = upload(NULL, (uint64_t)T * hc * CH * hc);
    ds4_gpu_tensor *gmixed = upload(NULL, (uint64_t)T * E);
    ds4_gpu_tensor *gblk = upload(blk, (uint64_t)T * E);
    require_ok(ds4_gpu_qwen4_hc_norm_tensor(gxn, ginj, gR, a->base, a->size, gamma_off, inj_off, wtype, T, E, hc, hc, eps),
               "hc norm");
    require_ok(q8 ? ds4_gpu_matmul_q8_0_tensor(glo, a->base, a->size, down_off, dim, rank, gxn, T)
             : f16 ? ds4_gpu_matmul_f16_tensor(glo, a->base, a->size, down_off, dim, rank, gxn, T)
                   : ds4_gpu_matmul_f32_tensor(glo, a->base, a->size, down_off, dim, rank, gxn, T), "hc down gemv");
    require_ok(ds4_gpu_qwen4_hc_gate_mix_tensor(gmixed, gxn, glo, a->base, a->size, up_off, up_type, T, E, hc, rank),
               "hc gate mix");
    require_ok(ds4_gpu_qwen4_hc_combine_tensor(gR, gblk, ginj, T, E, hc), "hc combine");
    char name[96];
    const char *tname = q8 ? "q8_0" : f16 ? "f16" : "f32";
    snprintf(name, sizeof(name), "hc E=%u rank=%u T=%u %s: xn", E, rank, T, tname);
    check_tensor(name, gxn, xn, (uint64_t)T * dim, 1e-5);
    snprintf(name, sizeof(name), "hc E=%u rank=%u T=%u %s: lowrank", E, rank, T, tname);
    check_tensor(name, glo, lo, (uint64_t)T * rank, 2e-4);   /* f16 low-rank weights */
    snprintf(name, sizeof(name), "hc E=%u rank=%u T=%u %s: inject partials", E, rank, T, tname);
    check_tensor(name, ginj, part, (uint64_t)T * hc * CH * hc, 2e-5);
    snprintf(name, sizeof(name), "hc E=%u rank=%u T=%u %s: mixed", E, rank, T, tname);
    check_tensor(name, gmixed, mixed, (uint64_t)T * E, 2e-5);
    snprintf(name, sizeof(name), "hc E=%u rank=%u T=%u %s: combine", E, rank, T, tname);
    check_tensor(name, gR, R2, (uint64_t)T * dim, 2e-5);
    ds4_gpu_tensor_free(gblk); ds4_gpu_tensor_free(gmixed); ds4_gpu_tensor_free(ginj);
    ds4_gpu_tensor_free(glo); ds4_gpu_tensor_free(gxn); ds4_gpu_tensor_free(gR);
    free(R2); free(mixed); free(part); free(lo); free(xn); free(blk); free(R);
    free(g_gamma); free(g_down); free(g_up); free(g_inj);
}

/* ---- gated delta net ---- */

static void gdn_reference(uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t K, uint32_t T,
                          const float *qkv_in, const float *z, const float *ab_in,
                          const double *conv_w, const double *ssm_a, const double *dt, const double *norm_w,
                          double *state, double *hist, double *out) {
    const uint32_t kd = Hk * D, vd = Hv * D, C = 2 * kd + vd;
    double *conv = malloc(C * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t c = 0; c < C; c++) {
            double acc = conv_w[c * K + (K - 1)] * qkv_in[t * C + c];
            for (uint32_t k = 0; k + 1 < K; k++) acc += conv_w[c * K + k] * hist[k * C + c];
            conv[c] = silu_d(acc);
        }
        for (uint32_t k = 0; k + 2 < K; k++) memcpy(hist + k * C, hist + (k + 1) * C, C * sizeof(double));
        for (uint32_t c = 0; c < C; c++) hist[(K - 2) * C + c] = qkv_in[t * C + c];
        for (uint32_t h = 0; h < Hk; h++) {
            double sq = 0.0, sk = 0.0;
            for (uint32_t i = 0; i < D; i++) { sq += conv[h * D + i] * conv[h * D + i]; sk += conv[kd + h * D + i] * conv[kd + h * D + i]; }
            const double qs = 1.0 / sqrt(sq + 1e-6) / sqrt((double)D), ks = 1.0 / sqrt(sk + 1e-6);
            for (uint32_t i = 0; i < D; i++) { conv[h * D + i] *= qs; conv[kd + h * D + i] *= ks; }
        }
        for (uint32_t j = 0; j < Hv; j++) {
            const uint32_t kh = j % Hk;
            const double g = exp(ssm_a[j] * softplus_d((double)ab_in[t * 2 * Hv + j] + dt[j]));
            const double beta = sigmoid_d(ab_in[t * 2 * Hv + Hv + j]);
            double *S = state + (uint64_t)j * D * D;   /* [dv][dk] */
            const double *q = conv + kh * D, *k = conv + kd + kh * D, *v = conv + 2 * kd + j * D;
            double o[128];
            for (uint32_t dv = 0; dv < D; dv++) {
                double u = 0.0;
                for (uint32_t dk = 0; dk < D; dk++) { S[dv * D + dk] *= g; u += S[dv * D + dk] * k[dk]; }
                const double delta = (v[dv] - u) * beta;
                double acc = 0.0;
                for (uint32_t dk = 0; dk < D; dk++) { S[dv * D + dk] += k[dk] * delta; acc += S[dv * D + dk] * q[dk]; }
                o[dv] = acc;
            }
            double ss = 0.0;
            for (uint32_t dv = 0; dv < D; dv++) ss += o[dv] * o[dv];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t dv = 0; dv < D; dv++)
                out[((uint64_t)t * Hv + j) * D + dv] = o[dv] * r * norm_w[dv] * sigmoid_d(z[((uint64_t)t * Hv + j) * D + dv]);
        }
    }
    free(conv);
}

static void test_gdn(arena_t *a, uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t T) {
    const uint32_t K = 4, kd = Hk * D, vd = Hv * D, C = 2 * kd + vd;
    double *conv_w, *ssm_a, *dt, *norm_w;
    const uint64_t conv_off = arena_f32(a, (uint64_t)C * K, &conv_w, -0.5f, 0.5f);
    const uint64_t a_off = arena_f32(a, Hv, &ssm_a, -8.0f, -0.1f);
    const uint64_t dt_off = arena_f32(a, Hv, &dt, 0.2f, 1.5f);
    const uint64_t norm_off = arena_f32(a, D, &norm_w, 0.8f, 1.2f);
    float *qkv = rand_vec((uint64_t)T * C, 1.0f);
    float *z = rand_vec((uint64_t)T * vd, 1.0f);
    /* raw alpha/beta come from Q8 projections of a random input so the
     * fused front kernel can be checked against the same reference */
    const uint32_t E = 2560;
    double *alpha_w, *beta_w;
    const uint64_t alpha_off = arena_q8_0(a, Hv, E, &alpha_w, 0.05f);
    const uint64_t beta_off = arena_q8_0(a, Hv, E, &beta_w, 0.05f);
    float *mixed = rand_vec((uint64_t)T * E, 1.0f);
    float *ab = malloc((uint64_t)T * 2 * Hv * sizeof(float));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t j = 0; j < Hv; j++) {
            double sa = 0.0, sb = 0.0;
            for (uint32_t i = 0; i < E; i++) {
                sa += alpha_w[(uint64_t)j * E + i] * mixed[(uint64_t)t * E + i];
                sb += beta_w[(uint64_t)j * E + i] * mixed[(uint64_t)t * E + i];
            }
            ab[(uint64_t)t * 2 * Hv + j] = (float)sa;
            ab[(uint64_t)t * 2 * Hv + Hv + j] = (float)sb;
        }
    }
    float *av = malloc((uint64_t)T * Hv * sizeof(float)), *bv = malloc((uint64_t)T * Hv * sizeof(float));
    for (uint32_t t = 0; t < T; t++) {
        memcpy(av + (uint64_t)t * Hv, ab + (uint64_t)t * 2 * Hv, Hv * sizeof(float));
        memcpy(bv + (uint64_t)t * Hv, ab + (uint64_t)t * 2 * Hv + Hv, Hv * sizeof(float));
    }
    float *state0 = rand_vec((uint64_t)Hv * D * D, 0.1f);
    float *hist0 = rand_vec((uint64_t)(K - 1) * C, 1.0f);

    double *ref_state = malloc((uint64_t)Hv * D * D * sizeof(double));
    double *ref_hist = malloc((uint64_t)(K - 1) * C * sizeof(double));
    double *ref_out = malloc((uint64_t)T * vd * sizeof(double));
    for (uint64_t i = 0; i < (uint64_t)Hv * D * D; i++) ref_state[i] = state0[i];
    for (uint64_t i = 0; i < (uint64_t)(K - 1) * C; i++) ref_hist[i] = hist0[i];
    gdn_reference(Hk, Hv, D, K, T, qkv, z, ab, conv_w, ssm_a, dt, norm_w, ref_state, ref_hist, ref_out);

    char name[96];
    /* (a) one chunk of T tokens, (b) T single-token calls, (c) fused front
     * kernel (conv + alpha/beta + prep) in pairs of tokens */
    for (int mode = 0; mode < 3; mode++) {
        ds4_gpu_tensor *gqkv = upload(qkv, (uint64_t)T * C);
        ds4_gpu_tensor *gz = upload(z, (uint64_t)T * vd);
        ds4_gpu_tensor *ga = upload(av, (uint64_t)T * Hv);
        ds4_gpu_tensor *gb = upload(bv, (uint64_t)T * Hv);
        ds4_gpu_tensor *gstate = upload(state0, (uint64_t)Hv * D * D);
        ds4_gpu_tensor *ghist = upload(hist0, (uint64_t)(K - 1) * C);
        ds4_gpu_tensor *gout = upload(NULL, (uint64_t)T * vd);
        ds4_gpu_tensor *gmixed = upload(mixed, (uint64_t)T * E);
        const uint32_t base_step = mode == 0 ? T : mode == 1 ? 1 : 2;
        for (uint32_t t0 = 0; t0 < T; t0 += base_step) {
            const uint32_t step = t0 + base_step <= T ? base_step : T - t0;
            ds4_gpu_tensor *vqkv = ds4_gpu_tensor_view(gqkv, (uint64_t)t0 * C * 4, (uint64_t)step * C * 4);
            ds4_gpu_tensor *va = ds4_gpu_tensor_view(ga, (uint64_t)t0 * Hv * 4, (uint64_t)step * Hv * 4);
            ds4_gpu_tensor *vb = ds4_gpu_tensor_view(gb, (uint64_t)t0 * Hv * 4, (uint64_t)step * Hv * 4);
            ds4_gpu_tensor *vz = ds4_gpu_tensor_view(gz, (uint64_t)t0 * vd * 4, (uint64_t)step * vd * 4);
            ds4_gpu_tensor *vout = ds4_gpu_tensor_view(gout, (uint64_t)t0 * vd * 4, (uint64_t)step * vd * 4);
            if (mode == 2) {
                ds4_gpu_tensor *vm = ds4_gpu_tensor_view(gmixed, (uint64_t)t0 * E * 4, (uint64_t)step * E * 4);
                require_ok(ds4_gpu_qwen4_gdn_front_tensor(vqkv, ghist, vm, va, vb, a->base, a->size, conv_off, alpha_off, beta_off,
                                                          a_off, dt_off, 8u, step, Hk, Hv, D, K, E, NULL, 0u), "gdn front");
                ds4_gpu_tensor_free(vm);
            } else {
                require_ok(ds4_gpu_qwen4_conv_stream_tensor(vqkv, ghist, a->base, a->size, conv_off, step, C, K, true), "gdn conv");
                require_ok(ds4_gpu_qwen4_gdn_prep_tensor(vqkv, va, vb, a->base, a->size, a_off, dt_off, step, Hk, Hv, D), "gdn prep");
            }
            require_ok(ds4_gpu_qwen4_gdn_scan_tensor(vout, gstate, vqkv, va, vb, step, Hk, Hv, D, NULL, 0u), "gdn scan");
            require_ok(ds4_gpu_qwen4_gdn_out_tensor(vout, vz, a->base, a->size, norm_off, step, Hv, D, 1e-6f), "gdn out");
            ds4_gpu_tensor_free(vout); ds4_gpu_tensor_free(vz); ds4_gpu_tensor_free(vb); ds4_gpu_tensor_free(va); ds4_gpu_tensor_free(vqkv);
        }
        const char *mname = mode == 0 ? "chunk" : mode == 1 ? "decode" : "fused";
        snprintf(name, sizeof(name), "gdn %u/%u/%u T=%u %s: output", Hk, Hv, D, T, mname);
        check_tensor(name, gout, ref_out, (uint64_t)T * vd, 5e-5);
        snprintf(name, sizeof(name), "gdn %u/%u/%u T=%u %s: state", Hk, Hv, D, T, mname);
        check_tensor(name, gstate, ref_state, (uint64_t)Hv * D * D, 5e-5);
        snprintf(name, sizeof(name), "gdn %u/%u/%u T=%u %s: conv history", Hk, Hv, D, T, mname);
        check_tensor(name, ghist, ref_hist, (uint64_t)(K - 1) * C, 1e-6);
        ds4_gpu_tensor_free(gmixed);
        ds4_gpu_tensor_free(gout); ds4_gpu_tensor_free(ghist); ds4_gpu_tensor_free(gstate);
        ds4_gpu_tensor_free(gb); ds4_gpu_tensor_free(ga); ds4_gpu_tensor_free(gz); ds4_gpu_tensor_free(gqkv);
    }
    free(bv); free(av);
    free(ref_out); free(ref_hist); free(ref_state); free(hist0); free(state0); free(ab); free(z); free(qkv);
    free(conv_w); free(ssm_a); free(dt); free(norm_w);
    free(mixed); free(alpha_w); free(beta_w);
}

/* ---- indexer matrix scorer ---- */

static void test_idx_score_mm(uint32_t T, uint32_t n, uint32_t pos0) {
    const uint32_t Hi = 4, Di = 128, ratio = 4;
    float *q = rand_vec((uint64_t)T * Hi * Di, 1.0f);
    float *keyf = rand_vec((uint64_t)n * Di, 1.0f);
    uint16_t *keyh = malloc((uint64_t)n * Di * 2);
    for (uint64_t i = 0; i < (uint64_t)n * Di; i++) { keyh[i] = f32_to_f16(keyf[i]); keyf[i] = f16_to_f32(keyh[i]); }
    double *ref = malloc((uint64_t)T * n * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t visible = (pos0 + t + 1) / ratio;
        for (uint32_t b = 0; b < n; b++) {
            double sum = -3.0e38;
            if (b < visible) {
                sum = 0.0;
                for (uint32_t h = 0; h < Hi; h++) {
                    double d = 0.0;
                    for (uint32_t x = 0; x < Di; x++) d += (double)q[((uint64_t)t * Hi + h) * Di + x] * keyf[(uint64_t)b * Di + x];
                    sum += d > 0.0 ? d : 0.0;
                }
            }
            ref[(uint64_t)t * n + b] = sum;
        }
    }
    ds4_gpu_tensor *gq = upload(q, (uint64_t)T * Hi * Di);
    ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc((uint64_t)n * Di * 2);
    ds4_gpu_tensor *gs = upload(NULL, (uint64_t)T * n);
    require_ok(gq && gk && gs && ds4_gpu_tensor_write(gk, 0, keyh, (uint64_t)n * Di * 2) &&
               ds4_gpu_qwen4_idx_score_tensor(gs, gq, gk, T, n, Hi, Di, pos0, ratio), "idx score mm");
    float *got = download(gs, (uint64_t)T * n);
    /* invisible entries are -3e38 on both sides: compare them exactly by mapping to 0 */
    for (uint64_t i = 0; i < (uint64_t)T * n; i++) {
        if (ref[i] < -1e37) {
            ref[i] = 0.0;
            got[i] = got[i] < -1e37f ? 0.0f : 1.0f;
        }
    }
    char name[96];
    snprintf(name, sizeof(name), "idx score mm T=%u n=%u pos0=%u", T, n, pos0);
    check_close(name, got, ref, (uint64_t)T * n, 4e-3);   /* queries rounded to half */
    free(got); ds4_gpu_tensor_free(gs); ds4_gpu_tensor_free(gk); ds4_gpu_tensor_free(gq); free(ref); free(keyh); free(keyf); free(q);
}

/* ---- indexer radix select ---- */

static int cmp_desc_idx(void *ctx, const void *a, const void *b) {
    const float *sc = ctx;
    const int32_t ia = *(const int32_t *)a, ib = *(const int32_t *)b;
    if (sc[ia] != sc[ib]) return sc[ia] > sc[ib] ? -1 : 1;
    return ia < ib ? -1 : 1;
}

static void test_idx_select(uint32_t T, uint32_t n, uint32_t k, uint32_t visible) {
    float *sc = rand_vec((uint64_t)T * n, 4.0f);
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t b = 0; b < n; b++) {
            float *v = &sc[(uint64_t)t * n + b];
            if (b >= visible) *v = -3.0e38f;                 /* incomplete blocks */
            else if (*v < 0.0f) *v = 0.0f;                   /* relu sums */
            else if ((b % 7) == 3) *v = sc[(uint64_t)t * n + (b / 7) * 7];   /* duplicates */
        }
    }
    ds4_gpu_tensor *gs = upload(sc, (uint64_t)T * n);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * k * 4);
    require_ok(gs && gsel && ds4_gpu_qwen4_idx_select_tensor(gsel, gs, n, T, k), "idx select");
    int32_t *got = malloc((uint64_t)T * k * 4);
    require_ok(ds4_gpu_tensor_read(gsel, 0, got, (uint64_t)T * k * 4), "idx select read");
    int32_t *order = malloc((uint64_t)n * 4);
    uint64_t bad = 0;
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t b = 0; b < n; b++) order[b] = (int32_t)b;
        qsort_r(order, n, 4, sc + (uint64_t)t * n, cmp_desc_idx);   /* BSD argument order (macOS) */
        /* the k selected must equal the reference top-k as a set */
        int32_t *g = got + (uint64_t)t * k;
        for (uint32_t i = 0; i < k; i++) {
            bool found = false;
            for (uint32_t j = 0; j < k && !found; j++) found = g[j] == order[i];
            bad += !found;
        }
    }
    char name[96];
    snprintf(name, sizeof(name), "idx select T=%u n=%u k=%u visible=%u", T, n, k, visible);
    require_ok(bad == 0, name);
    printf("  %-44s ok\n", name);
    free(order); free(got); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gs); free(sc);
}

/* prefill attention kernel vs the per-token kernel on the same inputs */
static void test_attn_mm(uint32_t T, uint32_t pos0, bool sparse) {
    const uint32_t H = 24, Hkv = 2, D = 256, ratio = 4, k_blocks = 6, sel_stride = k_blocks * ratio + ratio;
    const uint32_t cap = pos0 + T;
    const uint64_t qn = (uint64_t)T * H * D, kvn = (uint64_t)cap * Hkv * D;
    float *q = rand_vec(qn, 1.0f), *gate = rand_vec(qn, 2.0f);
    _Float16 *kc = malloc(kvn * 2), *vc = malloc(kvn * 2);
    for (uint64_t i = 0; i < kvn; i++) { kc[i] = (_Float16)(frand() - 0.5f); vc[i] = (_Float16)(frand() - 0.5f); }
    int32_t *sel = malloc((uint64_t)T * sel_stride * 4);
    uint32_t *cnt = malloc(T * 4);
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t pos = pos0 + t, n_blocks = (pos + 1) / ratio, nb = n_blocks < k_blocks ? n_blocks : k_blocks;
        uint32_t blocks[16], n = 0;
        for (uint32_t i = 0; i < nb; i++) {
            uint32_t b;
            bool dup;
            do {
                b = (uint32_t)((frand() + 1.0f) * 0.5f * n_blocks) % n_blocks;
                dup = false;
                for (uint32_t j = 0; j < i && !dup; j++) dup = blocks[j] == b;
            } while (dup);
            blocks[i] = b;
        }
        for (uint32_t i = 0; i < nb; i++) for (uint32_t r = 0; r < ratio; r++) sel[(uint64_t)t * sel_stride + n++] = (int32_t)(blocks[i] * ratio + r);
        for (uint32_t k = n_blocks * ratio; k <= pos; k++) sel[(uint64_t)t * sel_stride + n++] = (int32_t)k;
        cnt[t] = n;
    }
    ds4_gpu_tensor *gq = upload(q, qn), *ggate = upload(gate, qn);
    ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(kvn * 2), *gv = ds4_gpu_tensor_alloc(kvn * 2);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * sel_stride * 4), *gcnt = ds4_gpu_tensor_alloc(T * 4);
    ds4_gpu_tensor *go_ref = upload(NULL, qn), *go_new = upload(NULL, qn);
    require_ok(gk && gv && gsel && gcnt && ds4_gpu_tensor_write(gk, 0, kc, kvn * 2) && ds4_gpu_tensor_write(gv, 0, vc, kvn * 2) &&
               ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * sel_stride * 4) && ds4_gpu_tensor_write(gcnt, 0, cnt, T * 4), "attn mm setup");
    setenv("DS4_QWEN4_NO_ATTN_MM", "1", 1);
    require_ok(ds4_gpu_qwen4_attn_decode_tensor(go_ref, gq, ggate, gk, gv, gsel, gcnt, NULL, T, H, Hkv, D, pos0, sparse, sel_stride, 0.0625f), "attn reference");
    unsetenv("DS4_QWEN4_NO_ATTN_MM");
    require_ok(ds4_gpu_qwen4_attn_decode_tensor(go_new, gq, ggate, gk, gv, gsel, gcnt, NULL, T, H, Hkv, D, pos0, sparse, sel_stride, 0.0625f), "attn mm");
    float *ref = download(go_ref, qn), *got = download(go_new, qn);
    double worst = 0.0, scale = 0.0;
    for (uint64_t i = 0; i < qn; i++) {
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) worst = d;
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    char name[96];
    snprintf(name, sizeof(name), "attn mm T=%u pos0=%u %s", T, pos0, sparse ? "sparse" : "dense");
    require_ok(worst <= 4e-3 * scale, name);
    printf("  %-44s ok  max|d|=%.2e (scale %.2e)\n", name, worst, scale);
    free(ref); free(got); free(q); free(gate); free(kc); free(vc); free(sel); free(cnt);
    ds4_gpu_tensor_free(gq); ds4_gpu_tensor_free(ggate); ds4_gpu_tensor_free(gk); ds4_gpu_tensor_free(gv);
    ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gcnt); ds4_gpu_tensor_free(go_ref); ds4_gpu_tensor_free(go_new);
}

/* ---- PLE ---- */

static void test_ple(arena_t *a, uint32_t E, uint32_t T) {
    const uint32_t hc = 4, dim = E * hc, K = 4, dil = 3, H = (K - 1) * dil;
    const float eps = 1e-6f;
    double *gk, *gq, *gc, *cw;
    const uint64_t gk_off = arena_f32(a, dim, &gk, 0.5f, 1.5f);
    const uint64_t gq_off = arena_f32(a, dim, &gq, 0.5f, 1.5f);
    const uint64_t gc_off = arena_f32(a, dim, &gc, 0.5f, 1.5f);
    const uint64_t cw_off = arena_f32(a, (uint64_t)dim * K, &cw, -0.4f, 0.4f);
    float *R = rand_vec((uint64_t)T * dim, 1.0f);
    float *key = rand_vec((uint64_t)T * dim, 1.0f);
    float *value = rand_vec((uint64_t)T * E, 1.0f);
    float *hist0 = rand_vec((uint64_t)H * dim, 1.0f);

    double *ref_R = malloc((uint64_t)T * dim * sizeof(double));
    double *ref_hist = malloc((uint64_t)H * dim * sizeof(double));
    double *gated = malloc(dim * sizeof(double)), *normed = malloc(dim * sizeof(double));
    for (uint64_t i = 0; i < (uint64_t)H * dim; i++) ref_hist[i] = hist0[i];
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < hc; s++) {
            double sk = 0.0, sr = 0.0;
            for (uint32_t i = 0; i < E; i++) {
                sk += (double)key[t * dim + s * E + i] * key[t * dim + s * E + i];
                sr += (double)R[t * dim + s * E + i] * R[t * dim + s * E + i];
            }
            const double ik = 1.0 / sqrt(sk / E + eps), ir = 1.0 / sqrt(sr / E + eps);
            double dot = 0.0;
            for (uint32_t i = 0; i < E; i++)
                dot += (key[t * dim + s * E + i] * ik * gk[s * E + i]) * (R[t * dim + s * E + i] * ir * gq[s * E + i]);
            double g = dot / sqrt((double)E);
            const double mag = sqrt(fmax(fabs(g), 1e-6));
            g = sigmoid_d(g > 0 ? mag : (g < 0 ? -mag : 0.0));
            double sg = 0.0;
            for (uint32_t i = 0; i < E; i++) { gated[s * E + i] = g * value[t * E + i]; sg += gated[s * E + i] * gated[s * E + i]; }
            const double ig = 1.0 / sqrt(sg / E + eps);
            for (uint32_t i = 0; i < E; i++) normed[s * E + i] = gated[s * E + i] * ig * gc[s * E + i];
        }
        for (uint32_t c = 0; c < dim; c++) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; k++) {
                const uint32_t back = (K - 1 - k) * dil;
                acc += cw[(uint64_t)c * K + k] * (back == 0 ? normed[c] : ref_hist[(uint64_t)(H - back) * dim + c]);
            }
            ref_R[t * dim + c] = R[t * dim + c] + gated[c] + silu_d(acc);
        }
        memmove(ref_hist, ref_hist + dim, (uint64_t)(H - 1) * dim * sizeof(double));
        memcpy(ref_hist + (uint64_t)(H - 1) * dim, normed, dim * sizeof(double));
    }

    char name[96];
    for (int mode = 0; mode < 2; mode++) {
        ds4_gpu_tensor *gR = upload(R, (uint64_t)T * dim);
        ds4_gpu_tensor *gkey = upload(key, (uint64_t)T * dim);
        ds4_gpu_tensor *gval = upload(value, (uint64_t)T * E);
        ds4_gpu_tensor *ggated = upload(NULL, (uint64_t)T * dim);
        ds4_gpu_tensor *gnormed = upload(NULL, (uint64_t)T * dim);
        ds4_gpu_tensor *ghist = upload(hist0, (uint64_t)H * dim);
        const uint32_t step = mode == 0 ? T : 1;
        for (uint32_t t0 = 0; t0 < T; t0 += step) {
            ds4_gpu_tensor *vR = ds4_gpu_tensor_view(gR, (uint64_t)t0 * dim * 4, (uint64_t)step * dim * 4);
            ds4_gpu_tensor *vkey = ds4_gpu_tensor_view(gkey, (uint64_t)t0 * dim * 4, (uint64_t)step * dim * 4);
            ds4_gpu_tensor *vval = ds4_gpu_tensor_view(gval, (uint64_t)t0 * E * 4, (uint64_t)step * E * 4);
            ds4_gpu_tensor *vg = ds4_gpu_tensor_view(ggated, (uint64_t)t0 * dim * 4, (uint64_t)step * dim * 4);
            ds4_gpu_tensor *vn = ds4_gpu_tensor_view(gnormed, (uint64_t)t0 * dim * 4, (uint64_t)step * dim * 4);
            require_ok(ds4_gpu_qwen4_ple_gate_tensor(vg, vn, vR, vkey, vval, a->base, a->size, gk_off, gq_off, gc_off,
                                                     step, E, hc, eps), "ple gate");
            require_ok(ds4_gpu_qwen4_ple_conv_tensor(vR, vg, vn, ghist, a->base, a->size, cw_off, 0u, step, dim, K, dil, NULL, 0u), "ple conv");
            ds4_gpu_tensor_free(vn); ds4_gpu_tensor_free(vg); ds4_gpu_tensor_free(vval);
            ds4_gpu_tensor_free(vkey); ds4_gpu_tensor_free(vR);
        }
        const char *mname = mode ? "decode" : "chunk";
        snprintf(name, sizeof(name), "ple E=%u T=%u %s: residual", E, T, mname);
        check_tensor(name, gR, ref_R, (uint64_t)T * dim, 2e-5);
        snprintf(name, sizeof(name), "ple E=%u T=%u %s: history", E, T, mname);
        check_tensor(name, ghist, ref_hist, (uint64_t)H * dim, 2e-5);
        ds4_gpu_tensor_free(ghist); ds4_gpu_tensor_free(gnormed); ds4_gpu_tensor_free(ggated);
        ds4_gpu_tensor_free(gval); ds4_gpu_tensor_free(gkey); ds4_gpu_tensor_free(gR);
    }
    free(normed); free(gated); free(ref_hist); free(ref_R); free(hist0); free(value); free(key); free(R);
    free(gk); free(gq); free(gc); free(cw);
}

/* ---- router ---- */

static void test_router(arena_t *a, uint32_t NE, uint32_t k, uint32_t T) {
    const uint32_t E = 2560;
    double *gate_w;
    const uint64_t gate_off = arena_f32(a, E, &gate_w, -0.05f, 0.05f);
    float *x = rand_vec((uint64_t)T * E, 1.0f);
    float *logits = rand_vec((uint64_t)T * NE, 3.0f);
    ds4_gpu_tensor *gl = upload(logits, (uint64_t)T * NE);
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * k * 4);
    ds4_gpu_tensor *gw = upload(NULL, (uint64_t)T * k);
    ds4_gpu_tensor *gsg = upload(NULL, T);
    require_ok(ds4_gpu_qwen4_router_topk_tensor(gsel, gw, gl, gx, a->base, a->size, gate_off, 0u, E, gsg, T, NE, k), "router");
    char name[96];
    {
        double *ref = malloc(T * sizeof(double));
        for (uint32_t t = 0; t < T; t++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < E; i++) acc += gate_w[i] * x[(uint64_t)t * E + i];
            ref[t] = acc;
        }
        snprintf(name, sizeof(name), "router NE=%u k=%u T=%u: shared gate logit", NE, k, T);
        check_tensor(name, gsg, ref, T, 1e-5);
        free(ref);
    }
    ds4_gpu_tensor_free(gsg); ds4_gpu_tensor_free(gx); free(x); free(gate_w);
    int32_t *sel = malloc((uint64_t)T * k * 4);
    require_ok(ds4_gpu_tensor_read(gsel, 0, sel, (uint64_t)T * k * 4), "router read");
    float *w = download(gw, (uint64_t)T * k);
    double worst = 0.0;
    for (uint32_t t = 0; t < T; t++) {
        double *p = malloc(NE * sizeof(double));
        double mx = -1e300, sum = 0.0;
        for (uint32_t e = 0; e < NE; e++) if (logits[t * NE + e] > mx) mx = logits[t * NE + e];
        for (uint32_t e = 0; e < NE; e++) { p[e] = exp(logits[t * NE + e] - mx); sum += p[e]; }
        for (uint32_t e = 0; e < NE; e++) p[e] /= sum;
        double wsum = 0.0;
        int *ref = malloc(k * sizeof(int));
        for (uint32_t i = 0; i < k; i++) {
            int best = -1;
            for (uint32_t e = 0; e < NE; e++) {
                bool used = false;
                for (uint32_t j = 0; j < i; j++) used |= ref[j] == (int)e;
                if (!used && (best < 0 || p[e] > p[best])) best = (int)e;
            }
            ref[i] = best;
            wsum += p[best];
        }
        for (uint32_t i = 0; i < k; i++) {
            if (sel[t * k + i] != ref[i]) {
                fprintf(stderr, "router: token %u slot %u expert %d != %d\n", t, i, sel[t * k + i], ref[i]);
                exit(1);
            }
            const double d = fabs(w[t * k + i] - p[ref[i]] / wsum);
            if (d > worst) worst = d;
        }
        free(ref); free(p);
    }
    if (worst > 1e-6) { fprintf(stderr, "router weights max|d| %.3e\n", worst); exit(1); }
    snprintf(name, sizeof(name), "router NE=%u k=%u T=%u: softmax top-k", NE, k, T);
    printf("  %-44s ok  max|d|=%.2e\n", name, worst);
    free(w); free(sel); ds4_gpu_tensor_free(gw); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gl); free(logits);
}

/* ---- attention + indexer ---- */

static void rope_ref(double *x, uint32_t n_rot, uint32_t pos, double base) {
    const uint32_t nh = n_rot / 2;
    for (uint32_t i = 0; i < nh; i++) {
        const double th = (double)pos * pow(base, -2.0 * i / n_rot);
        const double c = cos(th), s = sin(th), x0 = x[i], x1 = x[i + nh];
        x[i] = x0 * c - x1 * s;
        x[i + nh] = x0 * s + x1 * c;
    }
}

/* Decode n_pos tokens one at a time through prep / block keys / scoring /
 * expand / attention and compare against a reference that recomputes the
 * selection.  k_blocks is small so the sparse path is exercised early. */
static void test_attention(arena_t *a, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t n_rot,
                           uint32_t Hi, uint32_t Di, uint32_t k_blocks, uint32_t n_pos) {
    const uint32_t ratio = 4, cap = n_pos + 4, group = H / Hkv;
    const double base = 1.0e7, eps = 1e-6;
    const float scale = 1.0f / sqrtf((float)D);
    double *gq_w, *gk_w, *giq_w, *gik_w;
    const uint64_t gq_off = arena_f32(a, D, &gq_w, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk_w, 0.5f, 1.5f);
    const uint64_t giq_off = arena_f32(a, Di, &giq_w, 0.5f, 1.5f);
    const uint64_t gik_off = arena_f32(a, Di, &gik_w, 0.5f, 1.5f);

    float *qg = rand_vec((uint64_t)n_pos * H * 2 * D, 1.0f);
    float *kp = rand_vec((uint64_t)n_pos * Hkv * D, 1.0f);
    float *vp = rand_vec((uint64_t)n_pos * Hkv * D, 1.0f);
    float *iq = rand_vec((uint64_t)n_pos * Hi * Di, 1.0f);
    float *ik = rand_vec((uint64_t)n_pos * Di, 1.0f);

    double *kc = malloc((uint64_t)cap * Hkv * D * sizeof(double));
    double *vc = malloc((uint64_t)cap * Hkv * D * sizeof(double));
    double *ref_out = malloc((uint64_t)n_pos * H * D * sizeof(double));
    double *qn = malloc((uint64_t)H * D * sizeof(double));
    double *iqn = malloc((uint64_t)Hi * Di * sizeof(double));
    double *score = malloc((uint64_t)(cap / ratio + 1) * sizeof(double));
    int *sel = malloc((uint64_t)cap * sizeof(int));
    double *p = malloc((uint64_t)cap * sizeof(double));
    for (uint32_t pos = 0; pos < n_pos; pos++) {
        for (uint32_t h = 0; h < H; h++) {
            const float *src = qg + ((uint64_t)pos * H + h) * 2 * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / D + eps);
            for (uint32_t i = 0; i < D; i++) qn[h * D + i] = src[i] * r * gq_w[i];
            rope_ref(qn + h * D, n_rot, pos, base);
        }
        for (uint32_t h = 0; h < Hkv; h++) {
            const float *src = kp + ((uint64_t)pos * Hkv + h) * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / D + eps);
            double *dst = kc + ((uint64_t)pos * Hkv + h) * D, *vdst = vc + ((uint64_t)pos * Hkv + h) * D;
            const float *vsrc = vp + ((uint64_t)pos * Hkv + h) * D;
            double tmp[256];
            /* rope on the f32 value before the f16 store, as the kernel does */
            for (uint32_t i = 0; i < D; i++) tmp[i] = src[i] * r * gk_w[i];
            rope_ref(tmp, n_rot, pos, base);
            for (uint32_t i = 0; i < D; i++) dst[i] = f16_to_f32(f32_to_f16((float)tmp[i]));
            for (uint32_t i = 0; i < D; i++) vdst[i] = f16_to_f32(f32_to_f16(vsrc[i]));
        }
        for (uint32_t h = 0; h < Hi; h++) {
            const float *src = iq + ((uint64_t)pos * Hi + h) * Di;
            double ss = 0.0;
            for (uint32_t i = 0; i < Di; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / Di + eps);
            for (uint32_t i = 0; i < Di; i++) iqn[h * Di + i] = src[i] * r * giq_w[i];
            rope_ref(iqn + h * Di, n_rot, pos, base);
        }
        const uint32_t n_vis = pos + 1, n_blocks = n_vis / ratio;
        uint32_t n_sel = 0;
        if (n_blocks <= k_blocks) {
            for (uint32_t t = 0; t < n_vis; t++) sel[n_sel++] = (int)t;
        } else {
            for (uint32_t b = 0; b < n_blocks; b++) {
                double key[128], ss = 0.0;
                for (uint32_t d = 0; d < Di; d++) {
                    double acc = 0.0;
                    for (uint32_t t = 0; t < ratio; t++) acc += ik[((uint64_t)(b * ratio + t)) * Di + d];
                    key[d] = acc / ratio;
                    ss += key[d] * key[d];
                }
                const double r = 1.0 / sqrt(ss / Di + eps);
                for (uint32_t d = 0; d < Di; d++) key[d] *= r * gik_w[d];
                rope_ref(key, n_rot, b * ratio, base);
                for (uint32_t d = 0; d < Di; d++) key[d] = f16_to_f32(f32_to_f16((float)key[d]));
                double sum = 0.0;
                for (uint32_t h = 0; h < Hi; h++) {
                    double dot = 0.0;
                    for (uint32_t d = 0; d < Di; d++) dot += iqn[h * Di + d] * key[d];
                    if (dot > 0) sum += dot;
                }
                score[b] = sum;
            }
            bool *taken = calloc(n_blocks, sizeof(bool));
            for (uint32_t i = 0; i < k_blocks; i++) {
                int best = -1;
                for (uint32_t b = 0; b < n_blocks; b++) if (!taken[b] && (best < 0 || score[b] > score[best])) best = (int)b;
                taken[best] = true;
            }
            for (uint32_t b = 0; b < n_blocks; b++) if (taken[b]) for (uint32_t t = 0; t < ratio; t++) sel[n_sel++] = (int)(b * ratio + t);
            for (uint32_t t = n_blocks * ratio; t < n_vis; t++) sel[n_sel++] = (int)t;
            free(taken);
        }
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / group;
            double mx = -1e300;
            for (uint32_t i = 0; i < n_sel; i++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < D; d++) dot += qn[h * D + d] * kc[((uint64_t)sel[i] * Hkv + kvh) * D + d];
                p[i] = dot * scale;
                if (p[i] > mx) mx = p[i];
            }
            double sum = 0.0;
            for (uint32_t i = 0; i < n_sel; i++) { p[i] = exp(p[i] - mx); sum += p[i]; }
            const float *gate = qg + ((uint64_t)pos * H + h) * 2 * D + D;
            for (uint32_t d = 0; d < D; d++) {
                double acc = 0.0;
                for (uint32_t i = 0; i < n_sel; i++) acc += p[i] * vc[((uint64_t)sel[i] * Hkv + kvh) * D + d];
                ref_out[((uint64_t)pos * H + h) * D + d] = acc / sum * sigmoid_d(gate[d]);
            }
        }
    }

    /* GPU: decode token by token */
    const uint32_t sel_stride = k_blocks * ratio + ratio;
    ds4_gpu_tensor *gqg = upload(qg, (uint64_t)n_pos * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(kp, (uint64_t)n_pos * Hkv * D);
    ds4_gpu_tensor *gvp = upload(vp, (uint64_t)n_pos * Hkv * D);
    ds4_gpu_tensor *giq = upload(iq, (uint64_t)n_pos * Hi * Di);
    ds4_gpu_tensor *gik = upload(ik, (uint64_t)n_pos * Di);
    ds4_gpu_tensor *gq = upload(NULL, (uint64_t)H * D);
    ds4_gpu_tensor *ggate = upload(NULL, (uint64_t)H * D);
    ds4_gpu_tensor *gkc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2);
    ds4_gpu_tensor *gvc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2);
    ds4_gpu_tensor *giqo = upload(NULL, (uint64_t)Hi * Di);
    ds4_gpu_tensor *gikc = upload(NULL, (uint64_t)cap * Di);
    ds4_gpu_tensor *gbk = ds4_gpu_tensor_alloc((uint64_t)(cap / ratio + 1) * Di * 2);
    ds4_gpu_tensor *gscore = upload(NULL, (uint64_t)(cap / ratio + 1));
    ds4_gpu_tensor *gselb = ds4_gpu_tensor_alloc((uint64_t)k_blocks * 4);
    ds4_gpu_tensor *gselt = ds4_gpu_tensor_alloc((uint64_t)sel_stride * 4);
    ds4_gpu_tensor *gnsel = ds4_gpu_tensor_alloc(4);
    ds4_gpu_tensor *gout = upload(NULL, (uint64_t)n_pos * H * D);
    ds4_gpu_tensor *gpart = upload(NULL, ds4_gpu_qwen4_attn_part_floats(1, H, D));
    ds4_gpu_tensor *gpos3 = ds4_gpu_tensor_alloc((uint64_t)cap * 16);
    require_ok(gkc && gvc && gbk && gselb && gselt && gnsel && gpart && gpos3, "attention allocs");
    {
        uint32_t *p3 = calloc((size_t)cap * 4, sizeof(uint32_t));
        for (uint32_t p = 0; p < cap; p++) p3[p * 4] = p3[p * 4 + 1] = p3[p * 4 + 2] = p;
        require_ok(ds4_gpu_tensor_write(gpos3, 0, p3, (uint64_t)cap * 16), "pos3 write");
        free(p3);
    }
    for (uint32_t pos = 0; pos < n_pos; pos++) {
        ds4_gpu_tensor *vqg = ds4_gpu_tensor_view(gqg, (uint64_t)pos * H * 2 * D * 4, (uint64_t)H * 2 * D * 4);
        ds4_gpu_tensor *vk = ds4_gpu_tensor_view(gkp, (uint64_t)pos * Hkv * D * 4, (uint64_t)Hkv * D * 4);
        ds4_gpu_tensor *vv = ds4_gpu_tensor_view(gvp, (uint64_t)pos * Hkv * D * 4, (uint64_t)Hkv * D * 4);
        ds4_gpu_tensor *viq = ds4_gpu_tensor_view(giq, (uint64_t)pos * Hi * Di * 4, (uint64_t)Hi * Di * 4);
        ds4_gpu_tensor *vik = ds4_gpu_tensor_view(gik, (uint64_t)pos * Di * 4, (uint64_t)Di * 4);
        ds4_gpu_tensor *vout = ds4_gpu_tensor_view(gout, (uint64_t)pos * H * D * 4, (uint64_t)H * D * 4);
        require_ok(ds4_gpu_qwen4_attn_prep_tensor(gq, ggate, gkc, gvc, giqo, gikc, vqg, vk, vv, viq, vik, gpos3,
                                                  a->base, a->size, gq_off, gk_off, giq_off,
                                                  1, H, Hkv, D, n_rot, Hi, Di, pos, cap, (float)base, (float)eps), "attn prep");
        const uint32_t n_blocks = (pos + 1) / ratio;
        if ((pos + 1) % ratio == 0) {
            require_ok(ds4_gpu_qwen4_idx_block_key_tensor(gbk, gikc, gpos3, a->base, a->size, gik_off, n_blocks - 1, 1,
                                                          ratio, Di, n_rot, (float)base, (float)eps), "block key");
        }
        const bool sparse = n_blocks > k_blocks;
        if (sparse) {
            require_ok(ds4_gpu_qwen4_idx_score_tensor(gscore, giqo, gbk, 1, n_blocks, Hi, Di, pos, ratio), "idx score");
            require_ok(ds4_gpu_qwen4_idx_select_tensor(gselb, gscore, n_blocks, 1, k_blocks), "idx select");
            {   /* the radix select must pick the argsort's set (order may differ) */
                ds4_gpu_tensor *gref = ds4_gpu_tensor_alloc((uint64_t)k_blocks * 4);
                require_ok(gref && ds4_gpu_indexer_topk_tensor(gref, gscore, n_blocks, 1, k_blocks), "topk ref");
                int32_t got_sel[64], ref_sel[64];
                require_ok(k_blocks <= 64 && ds4_gpu_tensor_read(gselb, 0, got_sel, k_blocks * 4) &&
                           ds4_gpu_tensor_read(gref, 0, ref_sel, k_blocks * 4), "sel read");
                for (uint32_t i = 0; i < k_blocks; i++) {
                    bool found = false;
                    for (uint32_t j = 0; j < k_blocks; j++) found |= got_sel[j] == ref_sel[i];
                    require_ok(found, "radix select matches argsort set");
                }
                ds4_gpu_tensor_free(gref);
            }
            require_ok(ds4_gpu_qwen4_idx_expand_tensor(gselt, gnsel, gselb, 1, k_blocks, ratio, pos, sel_stride), "idx expand");
        }
        /* even positions exercise the key-split partials (main sets DS4_QWEN4_ATTN_SPLIT_KEYS=8) */
        require_ok(ds4_gpu_qwen4_attn_decode_tensor(vout, gq, ggate, gkc, gvc, gselt, gnsel, (pos % 2) ? NULL : gpart,
                                                    1, H, Hkv, D, pos, sparse, sel_stride, scale), "attn decode");
        ds4_gpu_tensor_free(vout); ds4_gpu_tensor_free(vik); ds4_gpu_tensor_free(viq);
        ds4_gpu_tensor_free(vv); ds4_gpu_tensor_free(vk); ds4_gpu_tensor_free(vqg);
    }
    char name[96];
    snprintf(name, sizeof(name), "attention H=%u/%u D=%u k=%u n=%u", H, Hkv, D, k_blocks, n_pos);
    check_tensor(name, gout, ref_out, (uint64_t)n_pos * H * D, 5e-4);   /* half KV; block order differs */

    ds4_gpu_tensor_free(gout); ds4_gpu_tensor_free(gnsel); ds4_gpu_tensor_free(gselt); ds4_gpu_tensor_free(gselb);
    ds4_gpu_tensor_free(gscore); ds4_gpu_tensor_free(gbk); ds4_gpu_tensor_free(gikc); ds4_gpu_tensor_free(giqo);
    ds4_gpu_tensor_free(gvc); ds4_gpu_tensor_free(gkc); ds4_gpu_tensor_free(ggate); ds4_gpu_tensor_free(gq);
    ds4_gpu_tensor_free(gik); ds4_gpu_tensor_free(giq); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gqg);
    free(p); free(sel); free(score); free(iqn); free(qn); free(ref_out); free(vc); free(kc);
    free(ik); free(iq); free(vp); free(kp); free(qg);
    free(gq_w); free(gk_w); free(giq_w); free(gik_w);
}

/* ---- routed experts ---- */

static uint64_t arena_tier(arena_t *a, uint32_t wtype, uint64_t rows, uint64_t cols, double **shadow) {
    return wtype == 12u ? arena_q4_K(a, rows, cols, shadow, 0.05f) :
           wtype == 10u ? arena_q2_K(a, rows, cols, shadow, 0.05f) :
           wtype == 16u ? arena_iq2_xxs(a, rows, cols, shadow, 0.05f) : arena_q8_0(a, rows, cols, shadow, 0.05f);
}

/* wtype is the gate/up type (0 f32, 8 q8_0, 12 q4_K, 10 q2_K, 16 iq2_xxs); down
 * and the shared expert are q8_0, or f32 with wtype 0 */
static void test_moe(arena_t *a, uint32_t NE, uint32_t slots, uint32_t E, uint32_t F, uint32_t T, uint32_t wtype) {
    double *gate_w, *up_w, *down_w, *sg_w, *su_w, *sd_w;
    uint64_t gate_off, up_off, down_off, sg_off, su_off, sd_off;
    const bool q8 = wtype != 0u;
    const uint32_t dtype = q8 ? 8u : 0u;
    const char *tier_name = wtype == 12u ? "q4_K" : wtype == 10u ? "q2_K" : wtype == 16u ? "iq2_xxs" : q8 ? "q8_0" : "f32";
    if (q8) {
        gate_off = arena_tier(a, wtype, (uint64_t)NE * F, E, &gate_w);
        up_off = arena_tier(a, wtype, (uint64_t)NE * F, E, &up_w);
        down_off = arena_q8_0(a, (uint64_t)NE * E, F, &down_w, 0.05f);
        sg_off = arena_q8_0(a, F, E, &sg_w, 0.05f);
        su_off = arena_q8_0(a, F, E, &su_w, 0.05f);
        sd_off = arena_q8_0(a, E, F, &sd_w, 0.05f);
    } else {
        gate_off = arena_f32(a, (uint64_t)NE * F * E, &gate_w, -0.05f, 0.05f);
        up_off = arena_f32(a, (uint64_t)NE * F * E, &up_w, -0.05f, 0.05f);
        down_off = arena_f32(a, (uint64_t)NE * E * F, &down_w, -0.05f, 0.05f);
        sg_off = arena_f32(a, (uint64_t)F * E, &sg_w, -0.05f, 0.05f);
        su_off = arena_f32(a, (uint64_t)F * E, &su_w, -0.05f, 0.05f);
        sd_off = arena_f32(a, (uint64_t)E * F, &sd_w, -0.05f, 0.05f);
    }
    const uint32_t n_out = slots + 1;   /* + shared expert */
    float *x = rand_vec((uint64_t)T * E, 1.0f);
    float *sgate = rand_vec(T, 2.0f);
    int32_t *sel = malloc((uint64_t)T * slots * 4);
    float *w = malloc((uint64_t)T * slots * 4);
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < slots; s++) {
            sel[t * slots + s] = (int32_t)((t * 7u + s * 3u) % NE);
            w[t * slots + s] = 0.5f * frand() + 0.6f;
        }
    }
    double *mid = malloc((uint64_t)T * n_out * F * sizeof(double));
    double *part = malloc((uint64_t)T * n_out * E * sizeof(double));
    double *out = malloc((uint64_t)T * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t d = 0; d < E; d++) out[t * E + d] = 0.0;
        for (uint32_t s = 0; s < n_out; s++) {
            const bool shared = s == slots;
            const uint32_t e = shared ? 0 : (uint32_t)sel[t * slots + s];
            const double *gw = shared ? sg_w : gate_w + (uint64_t)e * F * E;
            const double *uw = shared ? su_w : up_w + (uint64_t)e * F * E;
            const double *dw = shared ? sd_w : down_w + (uint64_t)e * E * F;
            for (uint32_t f = 0; f < F; f++) {
                double g = 0.0, u = 0.0;
                for (uint32_t i = 0; i < E; i++) {
                    g += gw[(uint64_t)f * E + i] * x[t * E + i];
                    u += uw[(uint64_t)f * E + i] * x[t * E + i];
                }
                mid[((uint64_t)t * n_out + s) * F + f] = silu_d(g) * u;
            }
            const double wgt = shared ? sigmoid_d(sgate[t]) : w[t * slots + s];
            for (uint32_t d = 0; d < E; d++) {
                double acc = 0.0;
                for (uint32_t f = 0; f < F; f++) acc += dw[(uint64_t)d * F + f] * mid[((uint64_t)t * n_out + s) * F + f];
                part[((uint64_t)t * n_out + s) * E + d] = acc;
                out[t * E + d] += wgt * acc;
            }
        }
    }
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    require_ok(ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "sel write");
    ds4_gpu_tensor *gw = upload(w, (uint64_t)T * slots);
    ds4_gpu_tensor *gmid = upload(NULL, (uint64_t)T * n_out * F);
    ds4_gpu_tensor *gpart = upload(NULL, (uint64_t)T * n_out * E);
    ds4_gpu_tensor *gsg = upload(sgate, T);
    ds4_gpu_tensor *gout = upload(NULL, (uint64_t)T * E);
    require_ok(ds4_gpu_qwen4_moe_mid_tensor(gmid, gx, gsel, a->base, a->size, gate_off, up_off, wtype, NE, T, slots, E, F,
                                            sg_off, su_off, dtype), "moe mid");
    require_ok(ds4_gpu_qwen4_moe_down_tensor(gpart, gmid, gsel, a->base, a->size, down_off, dtype, NE, T, slots, F, E,
                                             sd_off, dtype), "moe down");
    char name[96];
    const uint32_t CH = DS4_QWEN4_HC_CHUNKS;
    float *R0 = rand_vec((uint64_t)T * 4 * E, 1.0f);
    float *injv = rand_vec((uint64_t)T * 4 * CH * 4, 1.0f);
    double *R_ref = malloc((uint64_t)T * 4 * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < 4; s++) {
            double tot = 0.0;
            for (uint32_t src = 0; src < 4 * CH; src++) tot += injv[((uint64_t)t * 4 * CH + src) * 4 + s];
            const double wgt = 2.0 * sigmoid_d(tot / 4.0);
            for (uint32_t d = 0; d < E; d++)
                R_ref[((uint64_t)t * 4 + s) * E + d] = (double)R0[((uint64_t)t * 4 + s) * E + d] + wgt * out[t * E + d];
        }
    ds4_gpu_tensor *gR = upload(R0, (uint64_t)T * 4 * E);
    ds4_gpu_tensor *ginj = upload(injv, (uint64_t)T * 4 * CH * 4);
    require_ok(ds4_gpu_qwen4_moe_reduce_tensor(gout, gpart, gw, gsg, NULL, gR, ginj, T, slots, n_out, E, 4), "moe reduce");
    const char *dname = q8 ? "q8_0" : "f32";
    snprintf(name, sizeof(name), "moe %s E=%u F=%u slots=%u T=%u: reduce+combine", dname, E, F, slots, T);
    check_tensor(name, gR, R_ref, (uint64_t)T * 4 * E, 2e-5);
    ds4_gpu_tensor_free(ginj); ds4_gpu_tensor_free(gR); free(R_ref); free(injv); free(R0);
    snprintf(name, sizeof(name), "moe %s E=%u F=%u slots=%u T=%u: mid (+shared)", tier_name, E, F, slots, T);
    check_tensor(name, gmid, mid, (uint64_t)T * n_out * F, 2e-5);
    snprintf(name, sizeof(name), "moe %s E=%u F=%u slots=%u T=%u: down (+shared)", dname, E, F, slots, T);
    check_tensor(name, gpart, part, (uint64_t)T * n_out * E, 2e-5);
    snprintf(name, sizeof(name), "moe %s E=%u F=%u slots=%u T=%u: out", dname, E, F, slots, T);
    check_tensor(name, gout, out, (uint64_t)T * E, 2e-5);
    if (q8 && (E % 64) == 0 && (F % 64) == 0) {
        /* prefill path: lists + tiled GEMMs over routed slots only (f16 tiles) */
        ds4_gpu_tensor *glists = ds4_gpu_tensor_alloc((uint64_t)NE * T * 4);
        ds4_gpu_tensor *gcounts = ds4_gpu_tensor_alloc((uint64_t)NE * 4);
        ds4_gpu_tensor *gmid2 = upload(NULL, (uint64_t)T * slots * F);
        ds4_gpu_tensor *gpart2 = upload(NULL, (uint64_t)T * slots * E);
        require_ok(ds4_gpu_qwen4_moe_build_lists_tensor(glists, gcounts, gsel, T, slots, NE, T), "moe lists");
        require_ok(ds4_gpu_qwen4_moe_mm_mid_tensor(gmid2, gx, glists, gcounts, a->base, a->size, gate_off, up_off, wtype,
                                                   NE, T, slots, slots, E, F, T), "moe mm mid");
        require_ok(ds4_gpu_qwen4_moe_mm_down_tensor(gpart2, gmid2, glists, gcounts, a->base, a->size, down_off, dtype,
                                                    NE, T, slots, slots, F, E, T), "moe mm down");
        double *mid_r = malloc((uint64_t)T * slots * F * sizeof(double));
        double *part_r = malloc((uint64_t)T * slots * E * sizeof(double));
        for (uint32_t t = 0; t < T; t++)
            for (uint32_t s = 0; s < slots; s++) {
                memcpy(mid_r + ((uint64_t)t * slots + s) * F, mid + ((uint64_t)t * n_out + s) * F, F * sizeof(double));
                memcpy(part_r + ((uint64_t)t * slots + s) * E, part + ((uint64_t)t * n_out + s) * E, E * sizeof(double));
            }
        int32_t *cnt = malloc((uint64_t)NE * 4);
        require_ok(ds4_gpu_tensor_read(gcounts, 0, cnt, (uint64_t)NE * 4), "counts read");
        int total = 0;
        for (uint32_t e = 0; e < NE; e++) total += cnt[e];
        if (total != (int)(T * slots)) { fprintf(stderr, "moe lists: %d pairs != %u\n", total, T * slots); exit(1); }
        free(cnt);
        snprintf(name, sizeof(name), "moe mm %s E=%u F=%u slots=%u T=%u: mid", tier_name, E, F, slots, T);
        check_tensor(name, gmid2, mid_r, (uint64_t)T * slots * F, 3e-3);
        snprintf(name, sizeof(name), "moe mm %s E=%u F=%u slots=%u T=%u: down", tier_name, E, F, slots, T);
        check_tensor(name, gpart2, part_r, (uint64_t)T * slots * E, 3e-3);
        free(part_r); free(mid_r);
        ds4_gpu_tensor_free(gpart2); ds4_gpu_tensor_free(gmid2); ds4_gpu_tensor_free(gcounts); ds4_gpu_tensor_free(glists);
    }
    ds4_gpu_tensor_free(gout); ds4_gpu_tensor_free(gsg); ds4_gpu_tensor_free(gpart);
    ds4_gpu_tensor_free(gmid); ds4_gpu_tensor_free(gw); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gx);
    free(out); free(part); free(mid); free(w); free(sel); free(sgate); free(x);
    free(gate_w); free(up_w); free(down_w); free(sg_w); free(su_w); free(sd_w);
}

/* MTP input staging: cat rows [rms(e)*g_e | 0] and [0 | rms(R_s)*g_h_s]
 * (full-row or per-stream RMS), then R_out = proj[0] + proj[1+s]. */
static void test_mtp(arena_t *a, uint32_t E, uint32_t hc) {
    double *g_e, *g_h;
    const uint64_t g_e_off = arena_f32(a, E, &g_e, 0.5f, 1.5f);
    const uint64_t g_h_off = arena_f32(a, (uint64_t)hc * E, &g_h, 0.5f, 1.5f);
    float *e = rand_vec(E, 1.0f);
    float *R = rand_vec((uint64_t)hc * E, 1.0f);
    float *proj = rand_vec((uint64_t)(hc + 1u) * E, 1.0f);
    double *cat = calloc((uint64_t)(hc + 1u) * 2u * E, sizeof(double));
    double *R_ref = malloc((uint64_t)hc * E * sizeof(double));
    const double eps = 1e-6;
    double ss = 0.0;
    for (uint32_t i = 0; i < E; i++) ss += (double)e[i] * e[i];
    double inv = 1.0 / sqrt(ss / E + eps);
    for (uint32_t i = 0; i < E; i++) cat[i] = e[i] * inv * g_e[i];
    double full = 0.0;
    for (uint32_t i = 0; i < hc * E; i++) full += (double)R[i] * R[i];
    inv = 1.0 / sqrt(full / ((double)hc * E) + eps);
    for (uint32_t s = 0; s < hc; s++) {
        for (uint32_t i = 0; i < E; i++) {
            cat[(uint64_t)(s + 1u) * 2u * E + E + i] = R[s * E + i] * inv * g_h[s * E + i];
            R_ref[s * E + i] = (double)proj[i] + proj[(s + 1u) * E + i];
        }
    }
    ds4_gpu_tensor *ge = upload(e, E);
    ds4_gpu_tensor *gR = upload(R, (uint64_t)hc * E);
    ds4_gpu_tensor *gcat = upload(NULL, (uint64_t)(hc + 1u) * 2u * E);
    ds4_gpu_tensor *gproj = upload(proj, (uint64_t)(hc + 1u) * E);
    ds4_gpu_tensor *gout = upload(NULL, (uint64_t)hc * E);
    require_ok(ds4_gpu_qwen4_mtp_stage_tensor(gcat, ge, gR, a->base, a->size, g_e_off, g_h_off, E, hc, (float)eps), "mtp stage");
    require_ok(ds4_gpu_qwen4_mtp_combine_tensor(gout, gproj, E, hc), "mtp combine");
    char name[96];
    snprintf(name, sizeof(name), "mtp stage E=%u hc=%u", E, hc);
    check_tensor(name, gcat, cat, (uint64_t)(hc + 1u) * 2u * E, 1e-5);
    snprintf(name, sizeof(name), "mtp combine E=%u hc=%u", E, hc);
    check_tensor(name, gout, R_ref, (uint64_t)hc * E, 1e-6);
    ds4_gpu_tensor_free(gout); ds4_gpu_tensor_free(gproj); ds4_gpu_tensor_free(gcat); ds4_gpu_tensor_free(gR); ds4_gpu_tensor_free(ge);
    free(R_ref); free(cat); free(proj); free(R); free(e); free(g_e); free(g_h);
}

static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef int (*bench_fn)(void *ud);

static double bench_run(const char *name, bench_fn fn, void *ud, uint32_t reps) {
    const char *only = getenv("QWEN4_BENCH_ONLY");
    if (only && only[0] && !strstr(name, only)) return 0.0;
    if (only && only[0]) { for (uint32_t i = 0; i < 20; i++) fn(ud); }   /* single-bench runs: warm the clocks */
    require_ok(ds4_gpu_begin_commands(), "begin");
    require_ok(fn(ud), name);
    require_ok(ds4_gpu_end_commands(), "end");
    require_ok(ds4_gpu_synchronize(), "sync");
    const double t0 = bench_now();
    require_ok(ds4_gpu_begin_commands(), "begin");
    for (uint32_t r = 0; r < reps; r++) require_ok(fn(ud), name);
    require_ok(ds4_gpu_end_commands(), "end");
    require_ok(ds4_gpu_synchronize(), "sync");
    const double us = 1e6 * (bench_now() - t0) / reps;
    printf("  %-44s %8.1f us\n", name, us);
    return us;
}

typedef struct {
    arena_t *a;
    uint64_t off[12];
    ds4_gpu_tensor *t[43];
    uint32_t n[3];
} bench_ctx;

static int bench_q8_small(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_q8_0_tensor(c->t[1], c->a->base, c->a->size, c->off[0], 2560, 48, c->t[0], 1); }
static int bench_q8_big(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_q8_0_tensor(c->t[1], c->a->base, c->a->size, c->off[1], 2560, 6144, c->t[0], 1); }
static int bench_q8_big2(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_q8_0_tensor(c->t[1], c->a->base, c->a->size, c->off[1], 2560, 6144, c->t[0], 2); }
static int bench_combine(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_hc_combine_tensor(c->t[2], c->t[0], c->t[3], 1, 2560, 4); }
static int bench_hc_norm(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_hc_norm_tensor(c->t[4], c->t[3], c->t[2], c->a->base, c->a->size, c->off[2], c->off[4], 1u, 1, 2560, 4, 4, 1e-6f); }
static int bench_hc_down(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_f16_tensor(c->t[5], c->a->base, c->a->size, c->off[3], 10240, 320, c->t[4], 1); }
static int bench_hc_mix(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_hc_gate_mix_tensor(c->t[0], c->t[4], c->t[5], c->a->base, c->a->size, c->off[5], 1u, 1, 2560, 4, 320); }
static int bench_attn(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_attn_decode_tensor(c->t[6], c->t[7], c->t[7], c->t[8], c->t[9], NULL, NULL, c->n[1] ? c->t[10] : NULL,
                                            1, 24, 2, 256, c->n[0], false, 0, 0.0625f);
}
static int bench_gdn_front(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_gdn_front_tensor(c->t[11], c->t[12], c->t[0], c->t[13], c->t[14], c->a->base, c->a->size,
                                          c->off[6], c->off[0], c->off[0], c->off[7], c->off[7], 8u, 1, 16, 48, 128, 4, 2560, NULL, 0u);
}
static int bench_gdn_unfused(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_matmul_q8_0_tensor(c->t[13], c->a->base, c->a->size, c->off[0], 2560, 48, c->t[0], 1) &&
           ds4_gpu_matmul_q8_0_tensor(c->t[14], c->a->base, c->a->size, c->off[0], 2560, 48, c->t[0], 1) &&
           ds4_gpu_qwen4_conv_stream_tensor(c->t[11], c->t[12], c->a->base, c->a->size, c->off[6], 1, 10240, 4, true) &&
           ds4_gpu_qwen4_gdn_prep_tensor(c->t[11], c->t[13], c->t[14], c->a->base, c->a->size, c->off[7], c->off[7], 1, 16, 48, 128);
}
static int bench_moe_mid(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_moe_mid_tensor(c->t[1], c->t[0], c->t[16], c->a->base, c->a->size, c->off[8], c->off[9], 8u, 16, 1, 10, 2560, 640, c->off[8], c->off[9], 8u);
}
static int bench_moe_down(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_moe_down_tensor(c->t[17], c->t[1], c->t[16], c->a->base, c->a->size, c->off[10], 8u, 16, 1, 10, 640, 2560, c->off[10], 8u);
}
static int bench_router_gemv(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_f32_tensor(c->t[17], c->a->base, c->a->size, c->off[11], 2560, 512, c->t[0], 1); }
static int bench_router_topk(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_router_topk_tensor(c->t[16], c->t[18], c->t[17], c->t[0], c->a->base, c->a->size, c->off[2], 0u, 2560, c->t[18], 1, 512, 10);
}
static int bench_router_topk_nogate(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_router_topk_tensor(c->t[16], c->t[18], c->t[17], NULL, NULL, 0, 0, 0u, 0, NULL, 1, 512, 10);
}
static int bench_router_topk_k1(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_router_topk_tensor(c->t[16], c->t[18], c->t[17], NULL, NULL, 0, 0, 0u, 0, NULL, 1, 512, 1);
}
/* prefill-sized (T=256) variants */
static int bench_p_router_f32(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_f32_tensor(c->t[19], c->a->base, c->a->size, c->off[11], 2560, 512, c->t[18], 256); }
static int bench_p_router_mm(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_dense_mm_tensor(c->t[19], c->t[18], c->a->base, c->a->size, c->off[11], 0u, 256, 2560, 512); }
static int bench_p_topk(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_router_topk_tensor(c->t[16], c->t[1], c->t[19], c->t[18], c->a->base, c->a->size, c->off[2], 0u, 2560, c->t[17], 256, 512, 10); }
static int bench_p_hc_norm(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_hc_norm_tensor(c->t[1], c->t[3], c->t[19], c->a->base, c->a->size, c->off[2], c->off[4], 1u, 256, 2560, 4, 4, 1e-6f); }
static int bench_p_hc_down_f16(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_f16_tensor(c->t[17], c->a->base, c->a->size, c->off[3], 10240, 320, c->t[1], 256); }
static int bench_p_hc_down_mm(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_dense_mm_tensor(c->t[17], c->t[1], c->a->base, c->a->size, c->off[3], 1u, 256, 10240, 320); }
static int bench_p_hc_up_f16(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_f16_tensor(c->t[1], c->a->base, c->a->size, c->off[5], 320, 10240, c->t[17], 256); }
static int bench_p_hc_up_mm(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_dense_mm_tensor(c->t[1], c->t[17], c->a->base, c->a->size, c->off[5], 1u, 256, 320, 10240); }
static int bench_p_hc_mix_rows(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_hc_mix_rows_tensor(c->t[19], c->t[1], c->t[1], 256, 2560, 4); }
static int bench_p_q8_gemm(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_q8_0_tensor(c->t[1], c->a->base, c->a->size, c->off[1], 2560, 6144, c->t[18], 256); }
static int bench_p_q8_mm(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_dense_mm_tensor(c->t[1], c->t[18], c->a->base, c->a->size, c->off[1], 8u, 256, 2560, 6144); }
static int bench_p_idx_score(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_idx_score_tensor(c->t[26], c->t[24], c->t[25], 32, 65536, 4, 128, 262144, 4); }
static int bench_p_idx_argsort(void *ud) { bench_ctx *c = ud; return ds4_gpu_indexer_topk_tensor(c->t[27], c->t[26], 65536, 32, 512); }
static int bench_p_idx_select(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_idx_select_tensor(c->t[27], c->t[26], 65536, 32, 512); }
static int bench_p_q8_gemm_2k(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_q8_0_tensor(c->t[40], c->a->base, c->a->size, c->off[1], 2560, 6144, c->t[39], 2048); }
static int bench_p_f16_gemm_2k(void *ud) { bench_ctx *c = ud; return ds4_gpu_matmul_f16_tensor(c->t[41], c->a->base, c->a->size, c->off[3], 10240, 320, c->t[42], 2048); }
static int bench_p_idx_score_1k(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_idx_score_tensor(c->t[29], c->t[28], c->t[25], 1024, 65536, 4, 128, 262144 - 1024, 4); }
static int bench_p_idx_select_1k(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_idx_select_tensor(c->t[30], c->t[29], 65536, 1024, 512); }
/* sparse prefill attention: 1024 queries at the end of a 256k context, 512 selected blocks each */
static int bench_p_attn_sparse(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_attn_decode_tensor(c->t[37], c->t[33], c->t[34], c->t[31], c->t[32], c->n[2] ? c->t[36] : c->t[35], c->t[38], NULL,
                                            1024, 24, 2, 256, 262144 - 1024, true, 2052, 0.0625f);
}
static int bench_p_gdn_r4(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_gdn_scan_tensor(c->t[15], c->t[1], c->t[11], c->t[13], c->t[14], 1024, 16, 48, 128, NULL, 0u); }
static int bench_p_moe_mm(void *ud) {
    bench_ctx *c = ud;
    return ds4_gpu_qwen4_moe_build_lists_tensor(c->t[20], c->t[21], c->t[16], 256, 10, 16, 256) &&
           ds4_gpu_qwen4_moe_mm_mid_tensor(c->t[22], c->t[18], c->t[20], c->t[21], c->a->base, c->a->size, c->off[8], c->off[9], 8u, 16, 256, 10, 10, 2560, 640, 256) &&
           ds4_gpu_qwen4_moe_mm_down_tensor(c->t[23], c->t[22], c->t[20], c->t[21], c->a->base, c->a->size, c->off[10], 8u, 16, 256, 10, 10, 640, 2560, 256);
}
static int bench_gdn_scan(void *ud) { bench_ctx *c = ud; return ds4_gpu_qwen4_gdn_scan_tensor(c->t[15], c->t[1], c->t[11], c->t[13], c->t[14], 1, 16, 48, 128, NULL, 0u); }

/* QWEN4_BENCH=1: per-dispatch cost of the decode kernels at full-model shapes */
static void bench_dispatch(arena_t *a) {
    bench_ctx c = { .a = a };
    double *sh;
    c.off[0] = arena_q8_0(a, 48, 2560, &sh, 0.05f); free(sh);
    c.off[1] = arena_q8_0(a, 6144, 2560, &sh, 0.05f); free(sh);
    c.off[2] = arena_f32(a, 10240, &sh, 0.5f, 1.5f); free(sh);
    c.off[3] = arena_f16(a, 320ull * 10240, &sh, 0.05f); free(sh);
    c.off[4] = arena_f16(a, 4ull * 10240, &sh, 0.05f); free(sh);
    c.off[5] = arena_f16(a, 10240ull * 320, &sh, 0.05f); free(sh);
    c.off[6] = arena_f32(a, 10240ull * 4, &sh, -0.5f, 0.5f); free(sh);
    c.off[7] = arena_f32(a, 48, &sh, -2.0f, -0.1f); free(sh);
    c.t[0] = upload(NULL, 2 * 2560);
    c.t[2] = upload(NULL, 10240);                   /* R */
    c.t[4] = upload(NULL, 10240);                   /* xn */
    c.t[5] = upload(NULL, 320);                     /* lo */
    c.t[6] = upload(NULL, 24 * 256);                /* attn out */
    c.t[7] = upload(NULL, 24 * 256);                /* q / gate */
    c.t[8] = ds4_gpu_tensor_alloc(4096ull * 512 * 2);  /* k cache f16 */
    c.t[9] = ds4_gpu_tensor_alloc(4096ull * 512 * 2);  /* v cache f16 */
    c.t[10] = upload(NULL, ds4_gpu_qwen4_attn_part_floats(2, 24, 256));
    c.t[12] = upload(NULL, 3 * 10240);              /* conv state */
    c.off[8] = arena_q8_0(a, 16ull * 640, 2560, &sh, 0.05f); free(sh);   /* 16 experts gate */
    c.off[9] = arena_q8_0(a, 16ull * 640, 2560, &sh, 0.05f); free(sh);   /* up */
    c.off[10] = arena_q8_0(a, 16ull * 2560, 640, &sh, 0.05f); free(sh);  /* down */
    c.off[11] = arena_f32(a, 512ull * 2560, &sh, -0.05f, 0.05f); free(sh);
    c.t[18] = upload(NULL, 256ull * 2560);        /* x for 256 tokens */
    c.t[19] = upload(NULL, 256ull * 10240);       /* wide scratch */
    c.t[1] = upload(NULL, 256ull * 10240);        /* scratch (also the GDN state) */
    c.t[3] = upload(NULL, 256ull * 4 * DS4_QWEN4_HC_CHUNKS * 4);   /* inj partials */
    c.t[20] = ds4_gpu_tensor_alloc(16ull * 256 * 4);
    c.t[21] = ds4_gpu_tensor_alloc(16 * 4);
    c.t[22] = upload(NULL, 256ull * 10 * 640);
    c.t[23] = upload(NULL, 256ull * 10 * 2560);
    c.t[17] = upload(NULL, 256ull * 2560);
    c.t[11] = upload(NULL, 1024ull * 10240);             /* qkv for 1024 tokens */
    c.t[13] = upload(NULL, 1024ull * 48);                /* decay */
    c.t[14] = upload(NULL, 1024ull * 48);                /* beta */
    c.t[15] = upload(NULL, 1024ull * 48 * 128);          /* scan output */
    c.t[24] = upload(NULL, 32ull * 512);                 /* indexer q, 32 tokens */
    c.t[25] = ds4_gpu_tensor_alloc(65536ull * 128 * 2);  /* block keys f16 */
    c.t[26] = upload(NULL, 32ull * 65536);               /* scores */
    c.t[27] = ds4_gpu_tensor_alloc(32ull * 512 * 4);     /* selected */
    c.t[28] = upload(NULL, 1024ull * 512);               /* indexer q, 1024 tokens */
    c.t[29] = upload(NULL, 1024ull * 65536);             /* scores, 1024 tokens */
    c.t[30] = ds4_gpu_tensor_alloc(1024ull * 512 * 4);   /* selected, 1024 tokens */
    {   /* 256k K/V caches, 1024 queries, selections: random blocks vs blocks from the last 8k tokens */
        const uint64_t kv_n = 262144ull * 512;
        _Float16 *kv = malloc(kv_n * 2);
        for (uint64_t i = 0; i < kv_n; i++) kv[i] = (_Float16)(frand() - 0.5f);
        c.t[31] = ds4_gpu_tensor_alloc(kv_n * 2);
        require_ok(c.t[31] && ds4_gpu_tensor_write(c.t[31], 0, kv, kv_n * 2), "k cache write");
        for (uint64_t i = 0; i < kv_n; i += 7) kv[i] = (_Float16)(frand() - 0.5f);
        c.t[32] = ds4_gpu_tensor_alloc(kv_n * 2);
        require_ok(c.t[32] && ds4_gpu_tensor_write(c.t[32], 0, kv, kv_n * 2), "v cache write");
        free(kv);
        float *qf = malloc(1024ull * 6144 * 4);
        for (int i = 0; i < 1024 * 6144; i++) qf[i] = frand() - 0.5f;
        c.t[33] = upload(qf, 1024ull * 6144);
        c.t[34] = upload(qf, 1024ull * 6144);
        free(qf);
        int32_t *sel = malloc(1024ull * 2052 * 4);
        uint32_t *cnt = malloc(1024 * 4);
        for (int variant = 0; variant < 2; variant++) {
            for (int t = 0; t < 1024; t++) {
                const uint32_t pos = 262144 - 1024 + t, n_blocks = pos / 4;
                const uint32_t lo = variant ? n_blocks - 2048 : 0, span = n_blocks - lo;
                uint32_t blocks[512];
                for (int i = 0; i < 512; i++) {
                    uint32_t b;
                    bool dup;
                    do {
                        b = lo + (uint32_t)((frand() + 1.0f) * 0.5f * span) % span;
                        dup = false;
                        for (int j = 0; j < i && !dup; j++) dup = blocks[j] == b;
                    } while (dup);
                    blocks[i] = b;
                }
                for (int i = 1; i < 512; i++) {
                    const uint32_t b = blocks[i];
                    int j = i - 1;
                    while (j >= 0 && blocks[j] > b) { blocks[j + 1] = blocks[j]; j--; }
                    blocks[j + 1] = b;
                }
                for (int i = 0; i < 512; i++)
                    for (int r = 0; r < 4; r++) sel[(uint64_t)t * 2052 + i * 4 + r] = (int32_t)(blocks[i] * 4 + r);
                cnt[t] = 2048;
            }
            c.t[35 + variant] = ds4_gpu_tensor_alloc(1024ull * 2052 * 4);
            require_ok(c.t[35 + variant] && ds4_gpu_tensor_write(c.t[35 + variant], 0, sel, 1024ull * 2052 * 4), "sel write");
        }
        c.t[38] = ds4_gpu_tensor_alloc(1024 * 4);
        require_ok(c.t[38] && ds4_gpu_tensor_write(c.t[38], 0, cnt, 1024 * 4), "cnt write");
        c.t[37] = upload(NULL, 1024ull * 6144);
        free(sel); free(cnt);
    }
    c.t[39] = upload(NULL, 2048ull * 2560);               /* x for 2048 tokens */
    c.t[40] = upload(NULL, 2048ull * 6144);
    c.t[41] = upload(NULL, 2048ull * 320);
    c.t[42] = upload(NULL, 2048ull * 10240);               /* hc-wide x for 2048 tokens */
    {   /* random queries and keys: the select bench needs a real score spread */
        float *q = malloc(1024ull * 512 * 4);
        for (int i = 0; i < 1024 * 512; i++) q[i] = frand() - 0.5f;
        require_ok(ds4_gpu_tensor_write(c.t[28], 0, q, 1024ull * 512 * 4), "indexer q write");
        free(q);
        _Float16 *k = malloc(65536ull * 128 * 2);
        for (int i = 0; i < 65536 * 128; i++) k[i] = (_Float16)(frand() - 0.5f);
        require_ok(c.t[25] && ds4_gpu_tensor_write(c.t[25], 0, k, 65536ull * 128 * 2), "block key write");
        free(k);
    }
    {   /* expert lists for the T=256 moe benches; the decode benches use the first 10 */
        int32_t *sel = malloc(256 * 10 * 4);
        for (int i = 0; i < 256 * 10; i++) sel[i] = (i * 7) % 16;
        c.t[16] = ds4_gpu_tensor_alloc(256 * 10 * 4);
        require_ok(ds4_gpu_tensor_write(c.t[16], 0, sel, 256 * 10 * 4), "sel write");
        free(sel);
    }
    require_ok(ds4_gpu_begin_commands(), "begin");
    require_ok(ds4_gpu_end_commands(), "end");
    bench_run("hc_combine (tiny)", bench_combine, &c, 200);
    bench_run("q8 gemv 48x2560", bench_q8_small, &c, 200);
    bench_run("q8 gemv 6144x2560 T=1", bench_q8_big, &c, 200);
    bench_run("q8 gemv 6144x2560 T=2", bench_q8_big2, &c, 200);
    bench_run("hc_norm (+inject partials)", bench_hc_norm, &c, 200);
    bench_run("hc down gemv f16 320x10240", bench_hc_down, &c, 200);
    bench_run("hc_gate_mix f16", bench_hc_mix, &c, 200);
    c.n[0] = 110; c.n[1] = 0; bench_run("attn_decode pos=110 no split", bench_attn, &c, 100);
    c.n[0] = 110; c.n[1] = 1; bench_run("attn_decode pos=110 split", bench_attn, &c, 100);
    c.n[0] = 2000; c.n[1] = 0; bench_run("attn_decode pos=2000 no split", bench_attn, &c, 50);
    c.n[0] = 2000; c.n[1] = 1; bench_run("attn_decode pos=2000 split", bench_attn, &c, 50);
    bench_run("gdn_front (fused)", bench_gdn_front, &c, 100);
    bench_run("gdn gemv a/b + conv + prep (prefill path)", bench_gdn_unfused, &c, 100);
    bench_run("gdn_scan", bench_gdn_scan, &c, 100);
    printf("  --- prefill T=256 ---\n");
    bench_run("router f32 512x2560 T=256 (DS4)", bench_p_router_f32, &c, 20);
    bench_run("router f32 512x2560 T=256 (dense mm)", bench_p_router_mm, &c, 20);
    bench_run("router_topk T=256", bench_p_topk, &c, 20);
    bench_run("hc_norm T=256", bench_p_hc_norm, &c, 20);
    bench_run("hc down f16 320x10240 T=256 (DS4)", bench_p_hc_down_f16, &c, 20);
    bench_run("hc down f16 320x10240 T=256 (dense mm)", bench_p_hc_down_mm, &c, 20);
    bench_run("hc up f16 10240x320 T=256 (DS4)", bench_p_hc_up_f16, &c, 20);
    bench_run("hc up f16 10240x320 T=256 (dense mm)", bench_p_hc_up_mm, &c, 20);
    bench_run("hc_mix_rows T=256", bench_p_hc_mix_rows, &c, 20);
    bench_run("q8 gemm 6144x2560 T=256 (DS4)", bench_p_q8_gemm, &c, 20);
    bench_run("q8 gemm 6144x2560 T=256 (dense mm)", bench_p_q8_mm, &c, 20);
    bench_run("moe mm lists+mid+down 16 experts T=256 (all 2560 pairs)", bench_p_moe_mm, &c, 10);
    {   /* decays in (0,1], betas in (0,1) for the scan benches */
        float *g = malloc(1024 * 48 * 4), *b = malloc(1024 * 48 * 4);
        for (int i = 0; i < 1024 * 48; i++) { g[i] = 0.9f + 0.1f * frand(); b[i] = 0.5f * frand() + 0.25f; }
        require_ok(ds4_gpu_tensor_write(c.t[13], 0, g, 1024 * 48 * 4) && ds4_gpu_tensor_write(c.t[14], 0, b, 1024 * 48 * 4),
                   "scan input write");
        free(g); free(b);
    }
    bench_run("gdn scan r4 T=1024", bench_p_gdn_r4, &c, 5);
    bench_run("idx score n=65536 T=32", bench_p_idx_score, &c, 10);
    bench_run("idx argsort top-512 n=65536 T=32", bench_p_idx_argsort, &c, 5);
    bench_run("idx select top-512 n=65536 T=32", bench_p_idx_select, &c, 10);
    bench_run("q8 gemm 6144x2560 T=2048 (DS4)", bench_p_q8_gemm_2k, &c, 5);
    bench_run("hc down f16 320x10240 T=2048 (DS4)", bench_p_f16_gemm_2k, &c, 5);
    bench_run("idx score n=65536 T=1024", bench_p_idx_score_1k, &c, 5);
    bench_run("idx select top-512 n=65536 T=1024", bench_p_idx_select_1k, &c, 5);
    setenv("DS4_QWEN4_NO_ATTN_MM", "1", 1);
    c.n[2] = 0; bench_run("attn sparse T=1024 ctx=256k random (per-token kernel)", bench_p_attn_sparse, &c, 5);
    unsetenv("DS4_QWEN4_NO_ATTN_MM");
    c.n[2] = 0; bench_run("attn sparse T=1024 ctx=256k random blocks", bench_p_attn_sparse, &c, 5);
    c.n[2] = 1; bench_run("attn sparse T=1024 ctx=256k blocks in last 8k", bench_p_attn_sparse, &c, 5);
    bench_run("router gemv f32 512x2560", bench_router_gemv, &c, 100);
    bench_run("router_topk (+gate logit)", bench_router_topk, &c, 100);
    bench_run("router_topk no gate", bench_router_topk_nogate, &c, 100);
    bench_run("router_topk no gate k=1", bench_router_topk_k1, &c, 100);
    bench_run("moe_mid q8 10+1 slots (37 MB)", bench_moe_mid, &c, 100);
    bench_run("moe_down q8 10+1 slots (19 MB)", bench_moe_down, &c, 100);
}

/* four projections of one input, mixed weight types (q8_0, bf16, q4_0, f16) */
static void test_multi_gemv(arena_t *a, uint32_t E, uint32_t T) {
    const uint32_t rows[4] = { 64, 48, 32, 16 };
    const uint32_t types[4] = { E % 256 == 0 ? 12u : 8u, 30u, 2u, 1u };
    double *sh[4];
    uint64_t offs[4];
    offs[0] = E % 256 == 0 ? arena_q4_K(a, rows[0], E, &sh[0], 0.05f) : arena_q8_0(a, rows[0], E, &sh[0], 0.05f);
    offs[1] = arena_bf16(a, (uint64_t)rows[1] * E, &sh[1], 0.05f);
    offs[2] = arena_q4_0(a, rows[2], E, &sh[2], 0.05f);
    offs[3] = arena_f16(a, (uint64_t)rows[3] * E, &sh[3], 0.05f);
    float *x = rand_vec((uint64_t)T * E, 1.0f);
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *outs[4];
    for (int i = 0; i < 4; i++) outs[i] = upload(NULL, (uint64_t)T * rows[i]);
    require_ok(ds4_gpu_qwen4_multi_gemv_tensor(gx, T, E, 4, outs, a->base, a->size, offs, types, rows), "multi gemv");
    for (int i = 0; i < 4; i++) {
        double *ref = malloc((uint64_t)T * rows[i] * sizeof(double));
        for (uint32_t t = 0; t < T; t++)
            for (uint32_t r = 0; r < rows[i]; r++) {
                double acc = 0.0;
                for (uint32_t k = 0; k < E; k++) acc += sh[i][(uint64_t)r * E + k] * x[(uint64_t)t * E + k];
                ref[(uint64_t)t * rows[i] + r] = acc;
            }
        char name[96];
        snprintf(name, sizeof(name), "multi gemv out%d type %u rows %u T=%u", i, types[i], rows[i], T);
        check_tensor(name, outs[i], ref, (uint64_t)T * rows[i], 3e-5);
        free(ref); free(sh[i]); ds4_gpu_tensor_free(outs[i]);
    }
    ds4_gpu_tensor_free(gx); free(x);
}

/* dense tiled GEMM against a double reference for f32, f16 and q8_0 rows */
static void test_dense_mm(arena_t *a, uint32_t in_dim, uint32_t rows, uint32_t T, uint32_t wtype) {
    double *sh;
    uint64_t off = wtype == 8u ? arena_q8_0(a, rows, in_dim, &sh, 0.05f)
                 : wtype == 1u ? arena_f16(a, (uint64_t)rows * in_dim, &sh, 0.05f)
                               : arena_f32(a, (uint64_t)rows * in_dim, &sh, -0.05f, 0.05f);
    float *x = rand_vec((uint64_t)T * in_dim, 1.0f);
    double *ref = malloc((uint64_t)T * rows * sizeof(double));
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t r = 0; r < rows; r++) {
            double acc = 0.0;
            for (uint32_t k = 0; k < in_dim; k++) acc += sh[(uint64_t)r * in_dim + k] * x[(uint64_t)t * in_dim + k];
            ref[(uint64_t)t * rows + r] = acc;
        }
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * in_dim);
    ds4_gpu_tensor *gout = upload(NULL, (uint64_t)T * rows);
    require_ok(ds4_gpu_qwen4_dense_mm_tensor(gout, gx, a->base, a->size, off, wtype, T, in_dim, rows), "dense mm");
    char name[96];
    snprintf(name, sizeof(name), "dense mm type %u %ux%u T=%u", wtype, rows, in_dim, T);
    check_tensor(name, gout, ref, (uint64_t)T * rows, 3e-5);
    free(ref); free(x); free(sh);
    ds4_gpu_tensor_free(gout); ds4_gpu_tensor_free(gx);
}

int main(void) {
    arena_t arena;
    arena.size = (uint64_t)1536 << 20;
    arena.base = mmap(NULL, arena.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    arena.used = 0;
    if (arena.base == MAP_FAILED) { perror("mmap"); return 1; }
    setenv("DS4_QWEN4_ATTN_SPLIT_KEYS", "8", 1);
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(arena.base, arena.size), "model map registration");

    if (getenv("QWEN4_BENCH")) {
        bench_dispatch(&arena);
        return 0;
    }
    printf("hyper-connections\n");
    test_hc(&arena, 2560, 320, 3, 1u);
    test_hc(&arena, 2560, 320, 1, 0u);
    test_hc(&arena, 2560, 320, 2, 8u);
    test_hc(&arena, 2560, 320, 1, 8u);
    test_hc(&arena, 64, 8, 5, 1u);
    test_hc(&arena, 64, 8, 2, 0u);
    test_hc(&arena, 64, 8, 1, 8u);
    test_hc(&arena, 64, 8, 3, 8u);
    printf("gated delta net\n");
    test_gdn(&arena, 16, 48, 128, 5);
    test_gdn(&arena, 16, 48, 128, 40);
    test_gdn(&arena, 16, 48, 128, 200);
    test_idx_score_mm(37, 3001, 11000);
    test_idx_score_mm(3, 70, 100);
    test_idx_select(4, 70001, 512, 69000);
    test_idx_select(3, 600, 512, 599);
    test_idx_select(2, 3000, 512, 520);
    test_attn_mm(40, 0, false);
    test_attn_mm(37, 3000, true);
    test_attn_mm(3, 100, true);
    test_gdn(&arena, 2, 6, 32, 7);
    printf("ple\n");
    test_ple(&arena, 2560, 3);
    test_ple(&arena, 64, 12);
    printf("router\n");
    test_router(&arena, 512, 10, 3);
    test_router(&arena, 32, 10, 5);
    printf("attention\n");
    test_attention(&arena, 24, 2, 256, 64, 4, 128, 2, 21);
    test_attention(&arena, 4, 2, 32, 8, 4, 32, 2, 30);
    printf("routed experts\n");
    test_moe(&arena, 16, 10, 2560, 640, 2, 8u);
    test_moe(&arena, 16, 10, 2560, 640, 37, 12u);
    test_moe(&arena, 16, 10, 2560, 640, 100, 12u);
    test_moe(&arena, 16, 10, 2560, 640, 37, 10u);
    test_moe(&arena, 16, 10, 2560, 640, 37, 16u);
    test_moe(&arena, 8, 10, 2560, 640, 1, 0u);
    test_moe(&arena, 32, 10, 64, 32, 3, 8u);
    test_moe(&arena, 32, 10, 64, 32, 3, 0u);
    printf("dense mm\n");
    test_dense_mm(&arena, 2560, 512, 37, 0u);
    test_dense_mm(&arena, 10240, 320, 33, 1u);
    test_dense_mm(&arena, 320, 10240, 40, 1u);
    test_dense_mm(&arena, 2560, 100, 9, 8u);
    test_dense_mm(&arena, 64, 32, 70, 0u);
    printf("multi gemv\n");
    test_multi_gemv(&arena, 2560, 2);
    test_multi_gemv(&arena, 64, 3);
    printf("mtp\n");
    test_mtp(&arena, 2560, 4);
    test_mtp(&arena, 64, 4);
    printf("all qwen4 kernel tests passed\n");
    return 0;
}
