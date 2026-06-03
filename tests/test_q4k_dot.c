/*
 * Unit test for Q4_K block layout, scale extraction, and dot product.
 * Build:  cc -O2 -Wall -DDS4_NO_GPU -DDS4_Q4K_DOT_TEST_MAIN -I. -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
 * Run:    ./tests/test_q4k_dot
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) return *(float *)&sign;
    if (exp == 31) return *(float *)(uint32_t[]){ sign | 0x7F800000 | (frac << 13) };
    uint32_t f = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

static inline void q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

/* The corrected dot product, matching ds4.c after the fix. */
static void vec_dot_q4_K_q8_K(int n, float *s, const block_q4_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d  = y[i].d * f16_to_f32(x[i].d);
        const float dm = -y[i].d * f16_to_f32(x[i].dmin);

        const uint8_t *qs = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int8_t  *q8 = y[i].qs;

        int summs = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);
            int32_t gsum = (int32_t)y[i].bsums[j * 2] + (int32_t)y[i].bsums[j * 2 + 1];
            summs += m_val * gsum;
        }

        int isum = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);

            const int byte_off = (j >> 1) * 32;
            const int shift = (j & 1) * 4;

            for (int l = 0; l < 32; l++) {
                isum += ((qs[byte_off + l] >> shift) & 0xF) * (int)q8[j * 32 + l] * sc_val;
            }
        }

        sumf += d * (float)isum + dm * (float)summs;
    }

    *s = sumf;
}

/* Reference: fully dequantize Q4_K to float, then dot with Q8_K's dequantized values. */
static float ref_dot(const block_q4_K *bx, const block_q8_K *by) {
    float x[QK_K];
    const float d  = f16_to_f32(bx->d);
    const float dm = f16_to_f32(bx->dmin);

    for (int j = 0; j < QK_K / 32; j++) {
        uint8_t sc_val, m_val;
        q4_k_get_scale_min(j, bx->scales, &sc_val, &m_val);
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) * 4;
        for (int l = 0; l < 32; l++) {
            int q = (bx->qs[byte_off + l] >> shift) & 0xF;
            x[j * 32 + l] = d * sc_val * (float)q - dm * m_val;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < QK_K; i++) sum += x[i] * by->d * (float)by->qs[i];
    return sum;
}

static void fill_q4_K(block_q4_K *bx, uint32_t seed) {
    uint8_t *p = (uint8_t *)bx;
    uint32_t s = seed;
    for (size_t i = 0; i < sizeof(block_q4_K); i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (uint8_t)(s >> 16);
    }
    /* Ensure d/dmin are valid finite f16 */
    bx->d    &= 0x7BFF;
    bx->dmin &= 0x7BFF;
}

static void fill_q8_K(block_q8_K *by, uint32_t seed) {
    uint32_t s = seed;
    by->d = ((s & 0xFFFF) / 65536.0f) * 2.0f + 0.01f;
    for (int i = 0; i < QK_K; i++) {
        s = s * 1103515245u + 12345u;
        by->qs[i] = (int8_t)((s >> 16) & 0xFF);
    }
    for (int j = 0; j < QK_K / 16; j++) {
        int32_t sum = 0;
        for (int l = 0; l < 16; l++) sum += (int32_t)by->qs[j * 16 + l];
        by->bsums[j] = (int16_t)sum;
    }
}

static int test_block_sizes(void) {
    int ok = (sizeof(block_q4_K) == 144 && sizeof(block_q8_K) == 292);
    printf("  block sizes: Q4_K=%zu (expect 144), Q8_K=%zu (expect 292): %s\n",
           sizeof(block_q4_K), sizeof(block_q8_K), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_scale_extraction(void) {
    uint8_t scales[12] = {0};
    uint8_t expected_sc[8] = {5, 12, 30, 60, 7, 15, 31, 63};
    uint8_t expected_m[8]  = {3, 8, 20, 45, 1, 10, 25, 50};

    for (int j = 0; j < 4; j++) {
        scales[j] = expected_sc[j];
        scales[j + 4] = expected_m[j];
    }
    for (int j = 4; j < 8; j++) {
        scales[j + 4] = (expected_sc[j] & 0xF) | ((expected_m[j] & 0xF) << 4);
        scales[j - 4] |= ((expected_sc[j] >> 4) << 6);
        scales[j - 0] |= ((expected_m[j] >> 4) << 6);
    }

    int ok = 1;
    for (int j = 0; j < 8; j++) {
        uint8_t sc, m;
        q4_k_get_scale_min(j, scales, &sc, &m);
        if (sc != expected_sc[j] || m != expected_m[j]) {
            printf("    j=%d: expected sc=%d m=%d, got sc=%d m=%d\n",
                   j, expected_sc[j], expected_m[j], sc, m);
            ok = 0;
        }
    }

    printf("  scale extraction: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_dot_reference(void) {
    int ok = 1;
    for (uint32_t seed = 1; seed <= 50; seed++) {
        block_q4_K bx;
        block_q8_K by;
        fill_q4_K(&bx, seed);
        fill_q8_K(&by, seed * 7 + 13);

        float result = 0.0f;
        vec_dot_q4_K_q8_K(QK_K, &result, &bx, &by);
        float expected = ref_dot(&bx, &by);

        float err = fabsf(result - expected);
        float rel = fabsf(expected) > 1e-3f ? err / fabsf(expected) : err;
        if (rel > 0.01f) {
            printf("    seed=%u: result=%.6f expected=%.6f rel_err=%.6f\n",
                   seed, result, expected, rel);
            ok = 0;
        }
    }

    printf("  dot vs reference (50 random blocks): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* Test with a hand-crafted known block. */
static int test_dot_known(void) {
    block_q4_K bx;
    block_q8_K by;
    memset(&bx, 0, sizeof(bx));
    memset(&by, 0, sizeof(by));

    /* d=1.0 f16=0x3C00, dmin=0, all scales=1 (sc=1, m=0), all qs=0x88 (value 8) */
    bx.d = 0x3C00;
    bx.dmin = 0;
    /* Groups 0-3: sc=1 m=0 */
    for (int j = 0; j < 4; j++) {
        bx.scales[j] = 1;
        bx.scales[j + 4] = 0;
    }
    /* Groups 4-7: sc=1 m=0, high bits=0 */
    for (int j = 4; j < 8; j++) {
        bx.scales[j + 4] = (1 & 0xF) | (0 << 4);
        /* high bits already 0 */
    }
    /* All qs = 0x88: low nibble=8, high nibble=8 */
    memset(bx.qs, 0x88, sizeof(bx.qs));

    /* Q8: all +1, d=1.0 */
    by.d = 1.0f;
    for (int i = 0; i < QK_K; i++) by.qs[i] = 1;
    for (int j = 0; j < QK_K / 16; j++) by.bsums[j] = 16;

    /* Expected: d=1, dm=0, isum = sum over 8 groups * 32 values * 8 * 1 * sc(=1) = 8*32*8 = 2048 */
    float result = 0.0f;
    vec_dot_q4_K_q8_K(QK_K, &result, &bx, &by);
    float expected = 2048.0f;

    int ok = (fabsf(result - expected) < 0.5f);
    printf("  dot known: result=%.1f expected=%.1f: %s\n", result, expected, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(void) {
    int failures = 0;

    printf("Q4_K unit tests:\n");
    failures += test_block_sizes();
    failures += test_scale_extraction();
    failures += test_dot_known();
    failures += test_dot_reference();

    printf("\n%d/%d tests passed\n", 4 - failures, 4);
    return failures ? 1 : 0;
}
