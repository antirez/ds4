/*
 * Unit test for Q2_K / Q4_K / IQ2_XXS block layouts and dot products.
 * Build:  cc -O3 -march=native -Wall -Wextra -std=c99 -o tests/test_quant_dot tests/test_quant_dot.c -lm -pthread
 * Run:    ./tests/test_quant_dot
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#if defined(__AVX2__) && defined(__AVX512F__) && defined(__AVX512BW__)
#define TEST_HAVE_AVX512_QUANT 1
#endif

#define QK_K 256

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
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

static inline int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16; i++) sum += (int32_t)((q2[i] >> shift) & 3) * (int32_t)q8[i];
    return sum;
}

static void vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; j++) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                int isuml = dot_q2_16(q2, q8, shift);
                isum += d * isuml;

                d = sc[is++] & 0x0f;
                isuml = dot_q2_16(q2 + 16, q8 + 16, shift);
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    *s = sumf;
}

#if defined(__AVX2__)
static inline int32_t hsum_i32_8_avx2(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
    return _mm_cvtsi128_si32(sum);
}

static inline __m256i dot_q2_16_avx2_epi32(const uint8_t *q2, const int8_t *q8, int shift) {
    __m128i q2v = _mm_loadu_si128((const __m128i *)q2);
    if (shift != 0) q2v = _mm_srli_epi16(q2v, shift);
    q2v = _mm_and_si128(q2v, _mm_set1_epi8(3));

    const __m256i q2_16 = _mm256_cvtepu8_epi16(q2v);
    const __m256i q8_16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)q8));
    return _mm256_madd_epi16(q2_16, q8_16);
}

static void vec_dot_q2_K_q8_K_avx2(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        const __m128i sc8 = _mm_loadu_si128((const __m128i *)sc);
        const __m256i sc16 = _mm256_cvtepu8_epi16(sc8);
        const __m256i mins16 = _mm256_srli_epi16(sc16, 4);
        const __m256i bsums16 = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m256i min_products = _mm256_madd_epi16(mins16, bsums16);
        const int summs = hsum_i32_8_avx2(min_products);

        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);

        __m256i isumv = _mm256_setzero_si256();
        int is = 0;
        for (int k = 0; k < QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                __m256i p0 = dot_q2_16_avx2_epi32(q2, q8, shift);
                __m256i p1 = dot_q2_16_avx2_epi32(q2 + 16, q8 + 16, shift);
                p0 = _mm256_mullo_epi32(p0, _mm256_set1_epi32(sc[is++] & 0x0f));
                p1 = _mm256_mullo_epi32(p1, _mm256_set1_epi32(sc[is++] & 0x0f));
                isumv = _mm256_add_epi32(isumv, _mm256_add_epi32(p0, p1));
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }

        const int isum = hsum_i32_8_avx2(isumv);
        sumf += dall * (float)isum - dmin * (float)summs;
    }

    *s = sumf;
}
#endif

#if defined(TEST_HAVE_AVX512_QUANT)
static inline __m256i zext_i128_to_i256(__m128i v) {
    return _mm256_inserti128_si256(_mm256_setzero_si256(), v, 0);
}

static inline __m512i dot_q2_32_avx512_epi32(const uint8_t *q2, const int8_t *q8, int shift) {
    __m256i q2v = _mm256_loadu_si256((const __m256i *)q2);
    if (shift != 0) q2v = _mm256_srli_epi16(q2v, shift);
    q2v = _mm256_and_si256(q2v, _mm256_set1_epi8(3));

    const __m512i q2_16 = _mm512_cvtepu8_epi16(q2v);
    const __m512i q8_16 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)q8));
    return _mm512_madd_epi16(q2_16, q8_16);
}

static void vec_dot_q2_K_q8_K_avx512(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        const __m128i sc8 = _mm_loadu_si128((const __m128i *)sc);
        const __m256i sc16 = _mm256_cvtepu8_epi16(sc8);
        const __m256i mins16 = _mm256_srli_epi16(sc16, 4);
        const __m256i bsums16 = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m256i min_products = _mm256_madd_epi16(mins16, bsums16);
        const int summs = hsum_i32_8_avx2(min_products);

        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);

        __m512i isumv = _mm512_setzero_si512();
        int is = 0;
        for (int k = 0; k < QK_K / 128; k++) {
            for (int shift = 0; shift < 8; shift += 2) {
                __m512i p = dot_q2_32_avx512_epi32(q2, q8, shift);
                const __m512i scalev = _mm512_setr_epi32(
                    sc[is] & 0x0f, sc[is] & 0x0f, sc[is] & 0x0f, sc[is] & 0x0f,
                    sc[is] & 0x0f, sc[is] & 0x0f, sc[is] & 0x0f, sc[is] & 0x0f,
                    sc[is + 1] & 0x0f, sc[is + 1] & 0x0f, sc[is + 1] & 0x0f, sc[is + 1] & 0x0f,
                    sc[is + 1] & 0x0f, sc[is + 1] & 0x0f, sc[is + 1] & 0x0f, sc[is + 1] & 0x0f);
                isumv = _mm512_add_epi32(isumv, _mm512_mullo_epi32(p, scalev));
                is += 2;
                q8 += 32;
            }
            q2 += 32;
        }

        const int isum = _mm512_reduce_add_epi32(isumv);
        sumf += dall * (float)isum - dmin * (float)summs;
    }

    *s = sumf;
}

static inline __m512i madd_i8_u8_i32_avx512(__m512i u8v, __m512i s8v) {
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), u8v, s8v);
#else
    const __m512i ones = _mm512_set1_epi16(1);
    return _mm512_madd_epi16(_mm512_maddubs_epi16(u8v, s8v), ones);
#endif
}
#endif

static float ref_dot(const block_q2_K *bx, const block_q8_K *by) {
    const float d = f16_to_f32(bx->d);
    const float dmin = f16_to_f32(bx->dmin);
    float sum = 0.0f;

    for (int j = 0; j < QK_K / 16; j++) {
        const float sc = (float)(bx->scales[j] & 0x0f);
        const float min = (float)(bx->scales[j] >> 4);
        const int q2_off = (j >= 8 ? 32 : 0) + (j & 1) * 16;
        const int shift = (j % 8) / 2 * 2;
        for (int l = 0; l < 16; l++) {
            const int q = (bx->qs[q2_off + l] >> shift) & 3;
            const float x = d * sc * (float)q - dmin * min;
            sum += x * by->d * (float)by->qs[j * 16 + l];
        }
    }

    return sum;
}

static uint32_t lcg_next(uint32_t *s) {
    *s = *s * 1103515245u + 12345u;
    return *s;
}

static void fill_q2_K(block_q2_K *bx, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < sizeof(bx->scales); i++) bx->scales[i] = (uint8_t)(lcg_next(&s) >> 16);
    for (size_t i = 0; i < sizeof(bx->qs); i++) bx->qs[i] = (uint8_t)(lcg_next(&s) >> 16);
    bx->d = (uint16_t)(0x3c00u + (lcg_next(&s) & 0x03ffu));
    bx->dmin = (uint16_t)(0x3400u + (lcg_next(&s) & 0x03ffu));
}

static void fill_q8_K(block_q8_K *by, uint32_t seed) {
    uint32_t s = seed;
    by->d = ((s & 0xffffu) / 65536.0f) * 2.0f + 0.01f;
    for (int i = 0; i < QK_K; i++) {
        by->qs[i] = (int8_t)((lcg_next(&s) >> 16) & 0xffu);
    }
    for (int j = 0; j < QK_K / 16; j++) {
        int32_t sum = 0;
        for (int l = 0; l < 16; l++) sum += (int32_t)by->qs[j * 16 + l];
        by->bsums[j] = (int16_t)sum;
    }
}

static int test_block_sizes(void) {
    const int ok = sizeof(block_q2_K) == 84 && sizeof(block_q8_K) == 292;
    printf("  block sizes: Q2_K=%zu (expect 84), Q8_K=%zu (expect 292): %s\n",
           sizeof(block_q2_K), sizeof(block_q8_K), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_dot_known(void) {
    block_q2_K bx;
    block_q8_K by;
    memset(&bx, 0, sizeof(bx));
    memset(&by, 0, sizeof(by));

    bx.d = 0x3c00u;    /* 1.0 */
    bx.dmin = 0;
    for (int j = 0; j < QK_K / 16; j++) bx.scales[j] = 1;
    memset(bx.qs, 0x55, sizeof(bx.qs)); /* Every 2-bit value is 1. */
    by.d = 1.0f;
    for (int i = 0; i < QK_K; i++) by.qs[i] = 1;
    for (int j = 0; j < QK_K / 16; j++) by.bsums[j] = 16;

    float result = 0.0f;
    vec_dot_q2_K_q8_K(QK_K, &result, &bx, &by);
    const float expected = 256.0f;
    const int ok = fabsf(result - expected) < 0.5f;
    printf("  dot known: result=%.1f expected=%.1f: %s\n", result, expected, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_dot_reference(void) {
    int ok = 1;
    for (uint32_t seed = 1; seed <= 100; seed++) {
        block_q2_K bx;
        block_q8_K by;
        fill_q2_K(&bx, seed);
        fill_q8_K(&by, seed * 7u + 13u);

        float result = 0.0f;
        vec_dot_q2_K_q8_K(QK_K, &result, &bx, &by);
        const float expected = ref_dot(&bx, &by);
#if defined(__AVX2__)
        float avx2_result = 0.0f;
        vec_dot_q2_K_q8_K_avx2(QK_K, &avx2_result, &bx, &by);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
        float avx512_result = 0.0f;
        vec_dot_q2_K_q8_K_avx512(QK_K, &avx512_result, &bx, &by);
#endif
        const float err = fabsf(result - expected);
        const float rel = fabsf(expected) > 1e-3f ? err / fabsf(expected) : err;
        if (rel > 0.01f
#if defined(__AVX2__)
                || fabsf(avx2_result - result) > 0.001f
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
                || fabsf(avx512_result - result) > 0.001f
#endif
        ) {
            printf("    seed=%u: result=%.6f expected=%.6f rel_err=%.6f",
                   seed, result, expected, rel);
#if defined(__AVX2__)
            printf(" avx2=%.6f", avx2_result);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
            printf(" avx512=%.6f", avx512_result);
#endif
            printf("\n");
            ok = 0;
        }
    }

    printf("  dot vs reference (100 random blocks%s%s): %s\n",
#if defined(__AVX2__)
           ", AVX2 checked",
#else
           "",
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
           ", AVX512 checked",
#else
           "",
#endif
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_multi_block(void) {
    block_q2_K bx[3];
    block_q8_K by[3];
    for (uint32_t i = 0; i < 3; i++) {
        fill_q2_K(&bx[i], 101u + i * 13u);
        fill_q8_K(&by[i], 211u + i * 17u);
    }

    float result = 0.0f;
    vec_dot_q2_K_q8_K(3 * QK_K, &result, bx, by);
#if defined(__AVX2__)
    float avx2_result = 0.0f;
    vec_dot_q2_K_q8_K_avx2(3 * QK_K, &avx2_result, bx, by);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
    float avx512_result = 0.0f;
    vec_dot_q2_K_q8_K_avx512(3 * QK_K, &avx512_result, bx, by);
#endif

    float expected = 0.0f;
    for (int i = 0; i < 3; i++) expected += ref_dot(&bx[i], &by[i]);
    const float err = fabsf(result - expected);
    const float rel = fabsf(expected) > 1e-3f ? err / fabsf(expected) : err;
    const int ok = rel <= 0.01f
#if defined(__AVX2__)
        && fabsf(avx2_result - result) <= 0.001f
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
        && fabsf(avx512_result - result) <= 0.001f
#endif
        ;
    printf("  multi-block dot: result=%.6f expected=%.6f rel_err=%.6f",
           result, expected, rel);
#if defined(__AVX2__)
    printf(" avx2=%.6f", avx2_result);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
    printf(" avx512=%.6f", avx512_result);
#endif
    printf(": %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* ========================================================================
 * Q4_K tests
 * ======================================================================== */

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

static inline void q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

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

#if defined(__AVX2__)
static void vec_dot_q4_K_q8_K_avx2(int n, float *s, const block_q4_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i m4   = _mm256_set1_epi8(0x0F);

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

        __m256i isumv = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);

            const int byte_off = (j >> 1) * 32;
            const int shift = (j & 1) * 4;

            __m256i qs32 = _mm256_loadu_si256((const __m256i *)(qs + byte_off));
            if (shift) qs32 = _mm256_srli_epi16(qs32, 4);
            qs32 = _mm256_and_si256(qs32, m4);

            const __m256i q8v = _mm256_loadu_si256((const __m256i *)(q8 + j * 32));
            const __m256i prod16 = _mm256_maddubs_epi16(qs32, q8v);
            const __m256i prod32 = _mm256_madd_epi16(prod16, ones);

            isumv = _mm256_add_epi32(isumv,
                        _mm256_mullo_epi32(prod32, _mm256_set1_epi32((int)sc_val)));
        }

        sumf += d * (float)hsum_i32_8_avx2(isumv) + dm * (float)summs;
    }

    *s = sumf;
}
#endif

#if defined(TEST_HAVE_AVX512_QUANT)
static void vec_dot_q4_K_q8_K_avx512(int n, float *s, const block_q4_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;
    const __m256i m4 = _mm256_set1_epi8(0x0F);

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

        __m512i isumv = _mm512_setzero_si512();
        for (int j = 0; j < QK_K / 64; j++) {
            uint8_t sc0, m0, sc1, m1;
            q4_k_get_scale_min(j * 2, sc, &sc0, &m0);
            q4_k_get_scale_min(j * 2 + 1, sc, &sc1, &m1);

            const __m256i qs32 = _mm256_loadu_si256((const __m256i *)(qs + j * 32));
            const __m256i lo = _mm256_and_si256(qs32, m4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qs32, 4), m4);
            __m512i q4v = _mm512_castsi256_si512(lo);
            q4v = _mm512_inserti64x4(q4v, hi, 1);

            const __m512i q8v = _mm512_loadu_si512((const void *)(q8 + j * 64));
            const __m512i dotv = madd_i8_u8_i32_avx512(q4v, q8v);
            const __m512i scalev = _mm512_setr_epi32(
                sc0, sc0, sc0, sc0, sc0, sc0, sc0, sc0,
                sc1, sc1, sc1, sc1, sc1, sc1, sc1, sc1);
            isumv = _mm512_add_epi32(isumv, _mm512_mullo_epi32(dotv, scalev));
        }

        sumf += d * (float)_mm512_reduce_add_epi32(isumv) + dm * (float)summs;
    }

    *s = sumf;
}
#endif

static void fill_q4_K(block_q4_K *bx, uint32_t seed) {
    uint32_t s = seed;
    bx->d = (uint16_t)(0x3c00u + (lcg_next(&s) & 0x03ffu));
    bx->dmin = (uint16_t)(0x3400u + (lcg_next(&s) & 0x03ffu));
    for (size_t i = 0; i < sizeof(bx->scales); i++) bx->scales[i] = (uint8_t)(lcg_next(&s) >> 16);
    for (size_t i = 0; i < sizeof(bx->qs); i++) bx->qs[i] = (uint8_t)(lcg_next(&s) >> 16);
}

static float ref_dot_q4_K(const block_q4_K *bx, const block_q8_K *by) {
    const float d = f16_to_f32(bx->d);
    const float dmin = f16_to_f32(bx->dmin);
    float sum = 0.0f;

    for (int j = 0; j < QK_K / 32; j++) {
        uint8_t sc_val, m_val;
        q4_k_get_scale_min(j, bx->scales, &sc_val, &m_val);
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) * 4;
        for (int l = 0; l < 32; l++) {
            sum += (d * (float)sc_val * (float)((bx->qs[byte_off + l] >> shift) & 0xF)
                  - dmin * (float)m_val) * by->d * (float)by->qs[j * 32 + l];
        }
    }

    return sum;
}

static int test_q4k_block_size(void) {
    const int ok = sizeof(block_q4_K) == 144;
    printf("  Q4_K block size: %zu (expect 144): %s\n",
           sizeof(block_q4_K), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_q4k_dot_reference(void) {
    int ok = 1;
    for (uint32_t seed = 1; seed <= 100; seed++) {
        block_q4_K bx;
        block_q8_K by;
        fill_q4_K(&bx, seed);
        fill_q8_K(&by, seed * 11u + 7u);

        float result = 0.0f;
        vec_dot_q4_K_q8_K(QK_K, &result, &bx, &by);
        const float expected = ref_dot_q4_K(&bx, &by);
#if defined(__AVX2__)
        float avx2_result = 0.0f;
        vec_dot_q4_K_q8_K_avx2(QK_K, &avx2_result, &bx, &by);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
        float avx512_result = 0.0f;
        vec_dot_q4_K_q8_K_avx512(QK_K, &avx512_result, &bx, &by);
#endif
        const float err = fabsf(result - expected);
        const float rel = fabsf(expected) > 1e-3f ? err / fabsf(expected) : err;
        if (rel > 0.01f
#if defined(__AVX2__)
                || fabsf(avx2_result - result) > 0.001f
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
                || fabsf(avx512_result - result) > 0.001f
#endif
        ) {
            printf("    seed=%u: result=%.6f expected=%.6f rel_err=%.6f",
                   seed, result, expected, rel);
#if defined(__AVX2__)
            printf(" avx2=%.6f", avx2_result);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
            printf(" avx512=%.6f", avx512_result);
#endif
            printf("\n");
            ok = 0;
        }
    }

    printf("  Q4_K dot vs reference (100 random blocks%s%s): %s\n",
#if defined(__AVX2__)
           ", AVX2 checked",
#else
           "",
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
           ", AVX512 checked",
#else
           "",
#endif
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_q4k_nibble_edge(void) {
    block_q4_K bx;
    block_q8_K by;
    memset(&bx, 0, sizeof(bx));
    memset(&by, 0, sizeof(by));

    bx.d = 0x3c00u;    /* 1.0 */
    bx.dmin = 0x3800u; /* 0.5 */
    bx.scales[0] = 5;
    bx.scales[1] = 9;
    bx.scales[4] = 3;
    bx.scales[5] = 2;

    static const uint8_t q4_pattern[4] = {0xF1, 0x2E, 0x80, 0x07};
    for (int i = 0; i < 32; i++) bx.qs[i] = q4_pattern[i & 3];

    by.d = 1.25f;
    for (int i = 0; i < QK_K; i++) by.qs[i] = (int8_t)((i % 17) - 8);
    for (int j = 0; j < QK_K / 16; j++) {
        int32_t sum = 0;
        for (int l = 0; l < 16; l++) sum += (int32_t)by.qs[j * 16 + l];
        by.bsums[j] = (int16_t)sum;
    }

    float result = 0.0f;
    vec_dot_q4_K_q8_K(QK_K, &result, &bx, &by);
    const float expected = ref_dot_q4_K(&bx, &by);
#if defined(__AVX2__)
    float avx2_result = 0.0f;
    vec_dot_q4_K_q8_K_avx2(QK_K, &avx2_result, &bx, &by);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
    float avx512_result = 0.0f;
    vec_dot_q4_K_q8_K_avx512(QK_K, &avx512_result, &bx, &by);
#endif

    const float err = fabsf(result - expected);
    const int ok = err <= 0.001f
#if defined(__AVX2__)
        && fabsf(avx2_result - result) <= 0.001f
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
        && fabsf(avx512_result - result) <= 0.001f
#endif
        ;

    printf("  Q4_K nibble edge: result=%.6f expected=%.6f err=%.6f",
           result, expected, err);
#if defined(__AVX2__)
    printf(" avx2=%.6f", avx2_result);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
    printf(" avx512=%.6f", avx512_result);
#endif
    printf(": %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_q4k_multi_block(void) {
    block_q4_K bx[3];
    block_q8_K by[3];
    for (uint32_t i = 0; i < 3; i++) {
        fill_q4_K(&bx[i], 301u + i * 19u);
        fill_q8_K(&by[i], 401u + i * 23u);
    }

    float result = 0.0f;
    vec_dot_q4_K_q8_K(3 * QK_K, &result, bx, by);
#if defined(__AVX2__)
    float avx2_result = 0.0f;
    vec_dot_q4_K_q8_K_avx2(3 * QK_K, &avx2_result, bx, by);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
    float avx512_result = 0.0f;
    vec_dot_q4_K_q8_K_avx512(3 * QK_K, &avx512_result, bx, by);
#endif

    float expected = 0.0f;
    for (int i = 0; i < 3; i++) expected += ref_dot_q4_K(&bx[i], &by[i]);
    const float err = fabsf(result - expected);
    const float rel = fabsf(expected) > 1e-3f ? err / fabsf(expected) : err;
    const int ok = rel <= 0.01f
#if defined(__AVX2__)
        && fabsf(avx2_result - result) <= 0.001f
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
        && fabsf(avx512_result - result) <= 0.001f
#endif
        ;
    printf("  Q4_K multi-block dot: result=%.6f expected=%.6f rel_err=%.6f",
           result, expected, rel);
#if defined(__AVX2__)
    printf(" avx2=%.6f", avx2_result);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
    printf(" avx512=%.6f", avx512_result);
#endif
    printf(": %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* ========================================================================
 * IQ2_XXS tests
 * ======================================================================== */

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

static const uint8_t iq2_kmask_iq2xs[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

static const uint8_t iq2_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2_grid[256] = {
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

static int8_t iq2_signed_grid[256][128][8];
static int iq2_initialized = 0;

static void iq2_init(void) {
    if (iq2_initialized) return;
    for (uint32_t s = 0; s < 128; s++) {
        const uint8_t signs = iq2_ksigns_iq2xs[s];
        for (uint32_t j = 0; j < 8; j++) {
            const int sign = (signs & iq2_kmask_iq2xs[j]) ? -1 : 1;
            for (uint32_t g = 0; g < 256; g++) {
                const uint8_t *grid = (const uint8_t *)(iq2_grid + g);
                iq2_signed_grid[g][s][j] = (int8_t)(sign * (int)grid[j]);
            }
        }
    }
    iq2_initialized = 1;
}

static void fill_iq2_xxs(block_iq2_xxs *bx, uint32_t seed) {
    uint32_t s = seed;
    bx->d = (uint16_t)(0x3c00u + (lcg_next(&s) & 0x03ffu));
    for (size_t i = 0; i < QK_K / 8; i++) bx->qs[i] = (uint16_t)(lcg_next(&s) >> 16);
}

static void vec_dot_iq2_xxs_q8_K(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
    iq2_init();
    const int nb = n / QK_K;
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                const int8_t *g0 = iq2_signed_grid[aux8[l]][sign_idx0];
                const int8_t *g1 = iq2_signed_grid[aux8[l + 1]][sign_idx1];
                for (int k = 0; k < 8; k++) sumi += (int32_t)g0[k] * (int32_t)q8[k];
                for (int k = 0; k < 8; k++) sumi += (int32_t)g1[k] * (int32_t)q8[8 + k];
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    *s = 0.125f * sumf;
}

#if defined(__AVX2__)
static void vec_dot_iq2_xxs_q8_K_avx2(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
    iq2_init();
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        __m256i bsum = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint8_t *aux8 = (const uint8_t *)aux32;

            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            __m256i sumi = _mm256_setzero_si256();

            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;

                const int8_t *g0 = iq2_signed_grid[aux8[l]][sign_idx0];
                const int8_t *g1 = iq2_signed_grid[aux8[l + 1]][sign_idx1];
                __m128i g_lo = _mm_loadl_epi64((const __m128i *)g0);
                g_lo = _mm_unpacklo_epi64(g_lo, _mm_loadl_epi64((const __m128i *)g1));
                __m256i g16 = _mm256_cvtepi8_epi16(g_lo);

                __m256i q16 = _mm256_cvtepi8_epi16(
                    _mm_loadu_si128((const __m128i *)q8));
                q8 += 16;

                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(g16, q16));
            }

            __m256i ls_v = _mm256_set1_epi32((int32_t)ls);
            bsum = _mm256_add_epi32(bsum, _mm256_mullo_epi32(sumi, ls_v));
        }

        sumf += d * (float)hsum_i32_8_avx2(bsum);
    }

    *s = 0.125f * sumf;
}
#endif

#if defined(TEST_HAVE_AVX512_QUANT)
static void vec_dot_iq2_xxs_q8_K_avx512(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
    iq2_init();
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        __m512i bsum = _mm512_setzero_si512();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            __m512i sumi = _mm512_setzero_si512();

            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                const int8_t *g0 = iq2_signed_grid[aux8[l]][sign_idx0];
                const int8_t *g1 = iq2_signed_grid[aux8[l + 1]][sign_idx1];
                __m128i g_lo = _mm_loadl_epi64((const __m128i *)g0);
                g_lo = _mm_unpacklo_epi64(g_lo, _mm_loadl_epi64((const __m128i *)g1));
                const __m512i g16 = _mm512_cvtepi8_epi16(zext_i128_to_i256(g_lo));
                const __m512i q16 = _mm512_cvtepi8_epi16(
                    zext_i128_to_i256(_mm_loadu_si128((const __m128i *)q8)));
                q8 += 16;
                sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(g16, q16));
            }

            __m512i ls_v = _mm512_set1_epi32((int32_t)ls);
            bsum = _mm512_add_epi32(bsum, _mm512_mullo_epi32(sumi, ls_v));
        }

        sumf += d * (float)_mm512_reduce_add_epi32(bsum);
    }

    *s = 0.125f * sumf;
}
#endif

static int test_iq2xxs_block_size(void) {
    const int ok = sizeof(block_iq2_xxs) == 66;
    printf("  IQ2_XXS block size: %zu (expect 66): %s\n",
           sizeof(block_iq2_xxs), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_iq2xxs_dot_reference(void) {
    int ok = 1;
    for (uint32_t seed = 1; seed <= 100; seed++) {
        block_iq2_xxs bx;
        block_q8_K by;
        fill_iq2_xxs(&bx, seed);
        fill_q8_K(&by, seed * 13u + 5u);

        float scalar_result = 0.0f;
        vec_dot_iq2_xxs_q8_K(QK_K, &scalar_result, &bx, &by);
#if defined(__AVX2__)
        float avx2_result = 0.0f;
        vec_dot_iq2_xxs_q8_K_avx2(QK_K, &avx2_result, &bx, &by);
        if (fabsf(avx2_result - scalar_result) > 0.001f) {
            printf("    seed=%u: scalar=%.6f avx2=%.6f diff=%.6f\n",
                   seed, scalar_result, avx2_result, fabsf(avx2_result - scalar_result));
            ok = 0;
        }
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
        float avx512_result = 0.0f;
        vec_dot_iq2_xxs_q8_K_avx512(QK_K, &avx512_result, &bx, &by);
        if (fabsf(avx512_result - scalar_result) > 0.001f) {
            printf("    seed=%u: scalar=%.6f avx512=%.6f diff=%.6f\n",
                   seed, scalar_result, avx512_result, fabsf(avx512_result - scalar_result));
            ok = 0;
        }
#endif
    }

    printf("  IQ2_XXS dot vs scalar (100 random blocks%s%s): %s\n",
#if defined(TEST_HAVE_AVX512_QUANT)
           ", AVX512 checked",
#elif defined(__AVX2__)
           ", AVX2 checked",
#else
           "",
#endif
           "", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_iq2xxs_multi_block(void) {
    block_iq2_xxs bx[2];
    block_q8_K by[2];
    for (uint32_t i = 0; i < 2; i++) {
        fill_iq2_xxs(&bx[i], 501u + i * 11u);
        fill_q8_K(&by[i], 601u + i * 13u);
    }

    float scalar_result = 0.0f;
    vec_dot_iq2_xxs_q8_K(2 * QK_K, &scalar_result, bx, by);
#if defined(__AVX2__)
    float avx2_result = 0.0f;
    vec_dot_iq2_xxs_q8_K_avx2(2 * QK_K, &avx2_result, bx, by);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
    float avx512_result = 0.0f;
    vec_dot_iq2_xxs_q8_K_avx512(2 * QK_K, &avx512_result, bx, by);
#endif

    float expected = 0.0f;
    for (int i = 0; i < 2; i++) {
        float r = 0.0f;
        vec_dot_iq2_xxs_q8_K(QK_K, &r, &bx[i], &by[i]);
        expected += r;
    }
    const float err = fabsf(scalar_result - expected);
    const int ok = err < 0.001f
#if defined(__AVX2__)
        && fabsf(avx2_result - scalar_result) <= 0.001f
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
        && fabsf(avx512_result - scalar_result) <= 0.001f
#endif
        ;
    printf("  IQ2_XXS multi-block dot: scalar=%.6f expected=%.6f err=%.6f",
           scalar_result, expected, err);
#if defined(__AVX2__)
    printf(" avx2=%.6f", avx2_result);
#endif
#if defined(TEST_HAVE_AVX512_QUANT)
    printf(" avx512=%.6f", avx512_result);
#endif
    printf(": %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(void) {
    int failures = 0;
    int total = 0;

    printf("Q2_K unit tests:\n");
    failures += test_block_sizes();
    total++;
    failures += test_dot_known();
    total++;
    failures += test_dot_reference();
    total++;
    failures += test_multi_block();
    total++;

    printf("\nQ4_K unit tests:\n");
    failures += test_q4k_block_size();
    total++;
    failures += test_q4k_dot_reference();
    total++;
    failures += test_q4k_nibble_edge();
    total++;
    failures += test_q4k_multi_block();
    total++;

    printf("\nIQ2_XXS unit tests:\n");
    failures += test_iq2xxs_block_size();
    total++;
    failures += test_iq2xxs_dot_reference();
    total++;
    failures += test_iq2xxs_multi_block();
    total++;

    printf("\n%d/%d tests passed\n", total - failures, total);
    return failures ? 1 : 0;
}
