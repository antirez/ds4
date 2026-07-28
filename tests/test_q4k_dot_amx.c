/*
 * Unit test for AMX-accelerated Q4_K dot product.
 * Compile: cc -O3 -march=native -mamx-tile -mamx-int8 -o tests/test_q4k_dot_amx tests/test_q4k_dot_amx.c -lm
 * Run:     tests/test_q4k_dot_amx
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

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
    if (exp == 31) {
        uint32_t f = sign | 0x7F800000 | (frac << 13);
        float out; memcpy(&out, &f, sizeof(out));
        return out;
    }
    uint32_t f = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    float out; memcpy(&out, &f, sizeof(out));
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

/* AMX-accelerated dot product (same as ds4.c) */
static float vec_dot_amx(const block_q4_K *x, const block_q8_K *y) {
    float sumf = 0.0f;
    const float d  = y->d * f16_to_f32(x->d);
    const float dm = -y->d * f16_to_f32(x->dmin);
    const uint8_t *qs = x->qs;
    const uint8_t *sc = x->scales;
    const int8_t  *q8 = y->qs;

    uint8_t sc_vals[8], m_vals[8];
    for (int j = 0; j < 8; j++)
        q4_k_get_scale_min(j, sc, &sc_vals[j], &m_vals[j]);

    int summs = 0;
    for (int j = 0; j < 8; j++)
        summs += (int)m_vals[j] * ((int32_t)y->bsums[j*2] + (int32_t)y->bsums[j*2+1]);

    uint8_t cfg[64] __attribute__((aligned(64)));
    memset(cfg, 0, sizeof(cfg));
    cfg[0] = 1;
    *(uint16_t*)(cfg + 16) = 4;  cfg[24] = 1;
    *(uint16_t*)(cfg + 18) = 32; cfg[26] = 1;
    *(uint16_t*)(cfg + 20) = 1;  cfg[28] = 32;
    __asm__ volatile("ldtilecfg %0" : : "m"(*(const uint8_t(*)[64])cfg) : "memory");

    int isum = 0;
    uint8_t unpacked[32] __attribute__((aligned(64)));
    for (int j = 0; j < 8; j++) {
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) * 4;
        for (int l = 0; l < 32; l++)
            unpacked[l] = (qs[byte_off + l] >> shift) & 0xF;
        _tile_zero(0);
        _tile_loadd(1, unpacked, 32);
        _tile_loadd(2, q8 + j * 32, 1);
        _tile_dpbusd(0, 1, 2);
        int32_t raw;
        _tile_stored(0, &raw, 4);
        isum += raw * (int)sc_vals[j];
    }
    __asm__ volatile("tilerelease" : : : "memory");
    sumf = d * (float)isum + dm * (float)summs;
    return sumf;
}

/* Scalar dot product (reference) */
static float vec_dot_scalar(const block_q4_K *x, const block_q8_K *y) {
    const float d  = y->d * f16_to_f32(x->d);
    const float dm = -y->d * f16_to_f32(x->dmin);
    const uint8_t *qs = x->qs;
    const uint8_t *sc = x->scales;
    const int8_t  *q8 = y->qs;

    uint8_t sc_vals[8], m_vals[8];
    for (int j = 0; j < 8; j++)
        q4_k_get_scale_min(j, sc, &sc_vals[j], &m_vals[j]);

    int summs = 0;
    for (int j = 0; j < 8; j++)
        summs += (int)m_vals[j] * ((int32_t)y->bsums[j*2] + (int32_t)y->bsums[j*2+1]);

    int isum = 0;
    for (int j = 0; j < 8; j++) {
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) * 4;
        for (int l = 0; l < 32; l++)
            isum += ((qs[byte_off + l] >> shift) & 0xF) * (int)q8[j * 32 + l] * sc_vals[j];
    }
    return d * (float)isum + dm * (float)summs;
}

/* Fully dequantized reference dot product */
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
    for (int i = 0; i < QK_K; i++)
        sum += x[i] * by->d * (float)by->qs[i];
    return sum;
}

static void fill_q4_K(block_q4_K *bx, uint32_t seed) {
    uint8_t *p = (uint8_t *)bx;
    uint32_t s = seed;
    for (size_t i = 0; i < sizeof(block_q4_K); i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (uint8_t)(s >> 16);
    }
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

int main(void) {
    int failures = 0;
    printf("AMX Q4_K dot product test:\n");

    /* Check runtime AMX support first */
    unsigned eax, ebx, ecx, edx;
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
    if (!(edx & (1 << 24)) || !(edx & (1 << 25))) {
        printf("  SKIP: CPU lacks AMX tile/int8 support\n");
        return 0;
    }
    uint32_t xcr0_low, xcr0_high;
    __asm__ volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
    if (!(xcr0_low & (1 << 18)) || !(xcr0_low & (1 << 19))) {
        printf("  SKIP: AMX not enabled in XCR0\n");
        return 0;
    }
    printf("  AMX hardware detected, running tests...\n");

    /* Test 1: known block (all ones vs all ones) */
    {
        block_q4_K bx;
        block_q8_K by;
        memset(&bx, 0, sizeof(bx));
        memset(&by, 0, sizeof(by));
        bx.d = 0x3C00;
        bx.dmin = 0;
        for (int j = 0; j < 4; j++) {
            bx.scales[j] = 1;
            bx.scales[j + 4] = 0;
        }
        for (int j = 4; j < 8; j++) {
            bx.scales[j + 4] = (1 & 0xF) | (0 << 4);
        }
        memset(bx.qs, 0x88, sizeof(bx.qs));
        by.d = 1.0f;
        for (int i = 0; i < QK_K; i++) by.qs[i] = 1;
        for (int j = 0; j < QK_K / 16; j++) by.bsums[j] = 16;

        float amx_res = vec_dot_amx(&bx, &by);
        float ref = ref_dot(&bx, &by);
        float err = fabsf(amx_res - ref);
        printf("  known block: AMX=%.1f ref=%.1f err=%.6f: %s\n",
               amx_res, ref, err, err < 0.5f ? "PASS" : "FAIL");
        if (err >= 0.5f) failures++;
    }

    /* Test 2: random blocks */
    {
        int ok = 1;
        for (uint32_t seed = 1; seed <= 100; seed++) {
            block_q4_K bx;
            block_q8_K by;
            fill_q4_K(&bx, seed);
            fill_q8_K(&by, seed * 7 + 13);

            float amx_res = vec_dot_amx(&bx, &by);
            float sca_res = vec_dot_scalar(&bx, &by);
            float ref = ref_dot(&bx, &by);

            float err_amx = fabsf(amx_res - ref);
            float rel_amx = fabsf(ref) > 1e-3f ? err_amx / fabsf(ref) : err_amx;
            if (rel_amx > 0.01f) {
                printf("    seed=%u: AMX=%f scalar=%f ref=%f rel_err=%f\n",
                       seed, amx_res, sca_res, ref, rel_amx);
                ok = 0;
            }
        }
        printf("  random blocks (100): %s\n", ok ? "PASS" : "FAIL");
        if (!ok) failures++;
    }

    printf("\n%d/%d tests passed\n", 2 - failures, 2);
    return failures ? 1 : 0;
}
