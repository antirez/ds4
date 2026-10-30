/* Selected GGML kernels, MIT, copyright (c) 2023-2026 The ggml authors.
 * Pinned source and adaptation notes: VENDOR.md. */
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#if defined(__AVX2__) || defined(__F16C__)
#include <immintrin.h>
#endif
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "qwen35moe-quants.h"
#define GGML_RESTRICT restrict
#define UNUSED(x) (void)(x)
#define GGML_ASSERT assert
#define GGML_FP16_TO_FP32 ds4_qwen35moe_half
#define GGML_CPU_FP16_TO_FP32 ds4_qwen35moe_half
static float ds4_qwen35moe_half_bits(uint16_t h) {
    uint32_t s=(uint32_t)(h & 0x8000)<<16, e=(h>>10)&31, m=h&1023, bits;
    if (!e) {
        if (!m) bits=s;
        else { int exp=-14; while (!(m&1024)) {m<<=1; exp--;} bits=s|((uint32_t)(exp+127)<<23)|((m&1023)<<13); }
    } else bits=s|((e==31 ? 255 : e+112)<<23)|(m<<13);
    float f; memcpy(&f,&bits,4); return f;
}
static float ds4_qwen35moe_half_table[1 << 16];
static pthread_once_t ds4_qwen35moe_half_once = PTHREAD_ONCE_INIT;
static void ds4_qwen35moe_half_table_init(void) {
    for (uint32_t i = 0; i < (1u << 16); i++)
        ds4_qwen35moe_half_table[i] = ds4_qwen35moe_half_bits((uint16_t)i);
}
void ds4_qwen35moe_init(void) {
    pthread_once(&ds4_qwen35moe_half_once, ds4_qwen35moe_half_table_init);
}
static inline float ds4_qwen35moe_half(uint16_t h) {
    return ds4_qwen35moe_half_table[h];
}
float ds4_qwen35moe_half_lookup(uint16_t h) {
    ds4_qwen35moe_init();
    return ds4_qwen35moe_half(h);
}
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
static void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}
static void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}
static void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}
static void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
static void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}
static void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}
static void dequantize_row_iq2_xs(const block_iq2_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}
static uint16_t ds4_qwen35moe_to_half(float f) {
#if defined(__F16C__)
    return (uint16_t)_cvtss_sh(f,0);
#else
    uint32_t u; memcpy(&u,&f,4);
    uint32_t sign=(u>>16)&0x8000; int exp=(int)((u>>23)&255)-112;
    uint32_t mant=u&0x7fffff;
    if(exp<=0) { if(exp < -10) return sign; mant|=0x800000; int shift=14-exp;
        return sign+((mant+((1u<<(shift-1))-1)+((mant>>shift)&1))>>shift); }
    if(exp>=31) return sign|0x7c00;
    return (uint16_t)(sign+((uint32_t)exp<<10)+((mant+4095+((mant>>13)&1))>>13));
#endif
}
#define GGML_FP32_TO_FP16 ds4_qwen35moe_to_half
#define GGML_CPU_FP32_TO_FP16 ds4_qwen35moe_to_half
#define GGML_UNUSED(x) (void)(x)
static void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}
static void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}
static void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}
static void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}
static void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}
static void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}
static void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}
static void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}
static void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}
static void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const uint8_t  * GGML_RESTRICT sc = x[i].scales;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

#if defined(__AVX2__) && defined(__FMA__)
// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    const __m128i hi64 = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#elif defined(__AVXVNNI__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
#else
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
#endif
}
#elif defined(__AVX__)
static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}

static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytesl = _mm_or_si128(bytesl, bit_mask);
    bytesh = _mm_or_si128(bytesh, bit_mask);
    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
    return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
    const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
    const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m128i axl = _mm256_castsi256_si128(ax);
    const __m128i axh = _mm256_extractf128_si256(ax, 1);
    const __m128i syl = _mm256_castsi256_si128(sy);
    const __m128i syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);
    // Get absolute values of x vectors
    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    // Sign the values of the y vectors
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented by four 128-bit vectors
static inline __m256 mul_sum_i8_quad_float(const __m128i x_1_0, const __m128i x_1_1, const __m128i x_2_0, const __m128i x_2_1,
                                           const __m128i y_1_0, const __m128i y_1_1, const __m128i y_2_0, const __m128i y_2_1) {
    const __m128i mone = _mm_set1_epi16(1);

    const __m128i p16_1_0 = mul_add_epi8_sse(x_1_0, y_1_0);
    const __m128i p16_1_1 = mul_add_epi8_sse(x_1_1, y_1_1);
    const __m128i p16_2_0 = mul_add_epi8_sse(x_2_0, y_2_0);
    const __m128i p16_2_1 = mul_add_epi8_sse(x_2_1, y_2_1);
    const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
    const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
    const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
    const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
    const __m128i p_1 = _mm_add_epi32(p_1_0, p_1_1);
    const __m128i p_2 = _mm_add_epi32(p_2_0, p_2_1);
    return _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));
}

// quad fp16 delta calculation
static inline __m256 quad_fp16_delta_float(const float x0, const float y0, const float x1, const float y1) {
    // GGML_CPU_FP16_TO_FP32 is faster than Intel F16C
    return _mm256_set_m128(_mm_set1_ps(GGML_CPU_FP16_TO_FP32(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                           _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}

static inline __m256 quad_mx_delta_float(const uint8_t x0, const float y0, const uint8_t x1, const float y1) {
    return _mm256_set_m128(_mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                           _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}
#endif
#elif defined(__SSSE3__)
static inline __m128i bytes_from_bits_16(const uint8_t * x) {
    uint16_t x16;
    memcpy(&x16, x, sizeof(uint16_t));

    const __m128i shuf_mask = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    __m128i bytes = _mm_shuffle_epi8(_mm_set1_epi16((short) x16), shuf_mask);
    const __m128i bit_mask = _mm_set_epi64x(0x7fbfdfeff7fbfdfe, 0x7fbfdfeff7fbfdfe);
    bytes = _mm_or_si128(bytes, bit_mask);

    return _mm_cmpeq_epi8(bytes, _mm_set1_epi64x(-1));
}

// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 =_mm_hadd_ps(a, b);
    __m128 res_1 =_mm_hadd_ps(c, d);
    __m128 res =_mm_hadd_ps(res_0, res_1);
    res =_mm_hadd_ps(res, res);
    res =_mm_hadd_ps(res, res);

    return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
static void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = GGML_CPU_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}
static void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        const __m128i qx_1_0 = _mm_loadu_si128((const __m128i *)x[ib].qs);
        const __m128i qx_1_1 = _mm_loadu_si128((const __m128i *)x[ib].qs + 1);
        const __m128i qx_2_0 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i qx_2_1 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs + 1);
        const __m128i qy_1_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        const __m128i qy_1_1 = _mm_loadu_si128((const __m128i *)y[ib].qs + 1);
        const __m128i qy_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i qy_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m256 p = mul_sum_i8_quad_float(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}
static void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; ++j) {

            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(0x3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // load mins and scales from block_q2_K.scales[QK_K/16]
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
        const __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));

        // summs = y[i].bsums * (x[i].scales >> 4) in 16bits*8*2 to 32bits*4*2
        const __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((const __m128i*)&y[i].bsums[0]));
        const __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((const __m128i*)&y[i].bsums[8]));

        // sumf += -dmin * summs in 32bits*8
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(MM256_SET_M128I(summs_1, summs_0))), acc);

        const __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
        const __m128i scales[2] = { scales_0, scales_1 };

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K/128; ++j) {

            // load Q8 quants int8*16*8 from block_q8_K.qs[QK_K]
            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            // load 2bits*16*8 from block_q2_K.qs[QK_K/4]
            __m128i q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_0 = _mm_and_si128(q2bits, m3);
            const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_1 = _mm_and_si128(q2bits, m3);
            const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

            // isuml = q8[l] * ((q2[l] >> shift) & 3) in 8bits*16*8 to 16bits*8*8
            __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
            __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
            __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
            __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
            __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
            __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
            __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
            __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

            // isum += (x[i].scales[is++] & 0xF) * isuml in 16bits*8*8 to 32bits*4*8
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

            p0 = _mm_add_epi32(p0, p1);
            p2 = _mm_add_epi32(p2, p3);
            p4 = _mm_add_epi32(p4, p5);
            p6 = _mm_add_epi32(p6, p7);

            // isum in 32bits*4*2
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
        }

        // sumf += dall * isum - dmin * summs in 32bits
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dall), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
static void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = _mm_add_epi32(sumi_1, p16l);

            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = _mm_add_epi32(sumi_1, p16h);

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
static void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc) + summs;

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m128i mone  = _mm_set1_epi8(1);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].qh[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].qh[16]);
        __m128i hmask = mone;

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int bit = 0;

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            const __m128i q5bits_0 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;
            const __m128i q5bits_1 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;

            __m128i q5l_0 = _mm_and_si128(q5bits_0, m4);
            __m128i q5l_1 = _mm_and_si128(q5bits_1, m4);
            __m128i q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            __m128i q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            __m128i q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            __m128i q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_0 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm_madd_epi16(scale_0, p16_0);
            p16_1 = _mm_madd_epi16(scale_0, p16_1);

            q5l_0 = _mm_and_si128(_mm_srli_epi16(q5bits_0, 4), m4);
            q5l_1 = _mm_and_si128(_mm_srli_epi16(q5bits_1, 4), m4);
            q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_2 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_3 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_2 = _mm_madd_epi16(scale_1, p16_2);
            p16_3 = _mm_madd_epi16(scale_1, p16_3);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc) + summs;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
static void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));

        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m15 = _mm_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // handle the q6_k -32 offset separately using bsums
        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)y[i].bsums);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)y[i].bsums + 1);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
        const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        const __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
        const __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i*)qh); qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i*)qh); qh += 16;

            const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
            const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
            const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
            const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
            const __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
            const __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
            const __m128i q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
            const __m128i q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

            const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;

            const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), q4h_0);
            const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), q4h_1);
            const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), q4h_2);
            const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), q4h_3);
            const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), q4h_4);
            const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), q4h_5);
            const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), q4h_6);
            const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), q4h_7);

            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8)), p16_7);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));

        }

        sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
        sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
        const __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
static void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX2__)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]]);
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127]);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
static void ggml_vec_dot_iq2_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX2__)

    const __m256i mone = _mm256_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*)bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_2);

    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i*)k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            const __m256i q2_data = _mm256_loadu_si256((const __m256i*)q2);  q2 += 16;
            aux_gindex = _mm256_and_si256(q2_data, m511);

            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);

            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_4 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            const __m256i q2_1 = _mm256_set_epi64x(iq2xs_grid[gindex[ 3]], iq2xs_grid[gindex[ 2]],
                                                   iq2xs_grid[gindex[ 1]], iq2xs_grid[gindex[ 0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xs_grid[gindex[ 7]], iq2xs_grid[gindex[ 6]],
                                                   iq2xs_grid[gindex[ 5]], iq2xs_grid[gindex[ 4]]);
            const __m256i q2_3 = _mm256_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                   iq2xs_grid[gindex[ 9]], iq2xs_grid[gindex[ 8]]);
            const __m256i q2_4 = _mm256_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                   iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);

            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = MM256_SET_M128I(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = MM256_SET_M128I(full_signs_h, full_signs_h);

            __m256i signs;
            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i dot3  = _mm256_maddubs_epi16(q2_3, q8s_3);
            const __m256i dot4  = _mm256_maddubs_epi16(q2_4, q8s_4);

            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3)));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const __m128i mone = _mm_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i bit_selector_mask_0 = _mm_loadu_si128((const __m128i*)bit_selector_mask_bytes);
    const __m128i bit_selector_mask_1 = _mm_loadu_si128((const __m128i*)bit_selector_mask_bytes + 1);
    const __m128i block_sign_shuffle_1_0 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_1);
    const __m128i block_sign_shuffle_1_1 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_1 + 1);
    const __m128i block_sign_shuffle_2_0 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_2);
    const __m128i block_sign_shuffle_2_1 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_2 + 1);

    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m128i bit_helper_0 = _mm_loadu_si128((const __m128i*)k_bit_helper);
    const __m128i bit_helper_1 = _mm_loadu_si128((const __m128i*)k_bit_helper + 1);
    const __m128i m511 = _mm_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            const __m128i q2_data_0 = _mm_loadu_si128((const __m128i*)q2);
            const __m128i q2_data_1 = _mm_loadu_si128((const __m128i*)q2 + 1);  q2 += 16;
            aux_gindex = MM256_SET_M128I(_mm_and_si128(q2_data_1, m511), _mm_and_si128(q2_data_0, m511));

            const __m128i partial_sign_bits_0 = _mm_srli_epi16(q2_data_0, 9);
            const __m128i partial_sign_bits_1 = _mm_srli_epi16(q2_data_1, 9);
            const __m128i partial_sign_bits_upper_0 = _mm_srli_epi16(q2_data_0, 13);
            const __m128i partial_sign_bits_upper_1 = _mm_srli_epi16(q2_data_1, 13);
            const __m128i partial_sign_bits_for_counting_0 = _mm_xor_si128(partial_sign_bits_0, partial_sign_bits_upper_0);
            const __m128i partial_sign_bits_for_counting_1 = _mm_xor_si128(partial_sign_bits_1, partial_sign_bits_upper_1);

            const __m128i odd_bits_0 = _mm_shuffle_epi8(bit_helper_0, partial_sign_bits_for_counting_0);
            const __m128i odd_bits_1 = _mm_shuffle_epi8(bit_helper_1, partial_sign_bits_for_counting_1);
            const __m128i full_sign_bits_0 = _mm_or_si128(partial_sign_bits_0, odd_bits_0);
            const __m128i full_sign_bits_1 = _mm_or_si128(partial_sign_bits_1, odd_bits_1);

            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i q2_1_0 = _mm_set_epi64x(iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]]);
            const __m128i q2_3_0 = _mm_set_epi64x(iq2xs_grid[gindex[9]], iq2xs_grid[gindex[8]]);
            const __m128i q2_3_1 = _mm_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]]);
            const __m128i q2_4_0 = _mm_set_epi64x(iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
            const __m128i q2_4_1 = _mm_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]]);

            // AVX2 full_signs_1 is full_sign_bits_0 here
            // AVX2 full_signs_2 is full_sign_bits_1 here
            __m128i signs_0, signs_1;
            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_3_0 = _mm_sign_epi8(q8_3_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_3_1 = _mm_sign_epi8(q8_3_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_4_0 = _mm_sign_epi8(q8_4_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_4_1 = _mm_sign_epi8(q8_4_1, _mm_or_si128(signs_1, mone));

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const __m128i dot3_0  = _mm_maddubs_epi16(q2_3_0, q8s_3_0);
            const __m128i dot3_1  = _mm_maddubs_epi16(q2_3_1, q8s_3_1);
            const __m128i dot4_0  = _mm_maddubs_epi16(q2_4_0, q8s_4_0);
            const __m128i dot4_1  = _mm_maddubs_epi16(q2_4_1, q8s_4_1);

            __m128i sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0));
            const __m128i sc1_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc1_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1));
            const __m128i sc2_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc2_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2));
            const __m128i sc3_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc3_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3));
            const __m128i sc4_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc4_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, sc1_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, sc1_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, sc2_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, sc2_1));
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot3_0, sc3_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot3_1, sc3_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot4_0, sc4_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot4_1, sc4_1));
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#endif

size_t ds4_qwen35moe_activation_bytes(int n) { return (size_t)(n/256)*sizeof(block_q8_K); }
void ds4_qwen35moe_quantize(int type, const float *x, void *out, int n) {
    ds4_qwen35moe_init();
    if(type==8) {
#if defined(__AVX2__) && defined(__FMA__)
        quantize_row_q8_0(x,out,n);
#else
        quantize_row_q8_0_ref(x,out,n);
#endif
    } else quantize_row_q8_K_generic(x,out,n);
}
void ds4_qwen35moe_dequant(int type, const void *x, float *out, int n) {
    ds4_qwen35moe_init();
    switch(type) {
    case 10: dequantize_row_q2_K(x,out,n); break;
    case 12: dequantize_row_q4_K(x,out,n); break;
    case 13: dequantize_row_q5_K(x,out,n); break;
    case 14: dequantize_row_q6_K(x,out,n); break;
    case 8: dequantize_row_q8_0(x,out,n); break;
    case 16: dequantize_row_iq2_xxs(x,out,n); break;
    case 17: dequantize_row_iq2_xs(x,out,n); break;
    default: assert(0);
    }
}
ds4_qwen35moe_dot_kernel ds4_qwen35moe_select_dot(int type) {
    ds4_qwen35moe_init();
#if defined(__AVX2__) && defined(__FMA__)
#define QWEN35MOE_SELECTED_DOT(name) ggml_vec_dot_##name
#else
#define QWEN35MOE_SELECTED_DOT(name) ggml_vec_dot_##name##_generic
#endif
    switch(type) {
    case 8:  return QWEN35MOE_SELECTED_DOT(q8_0_q8_0);
    case 10: return QWEN35MOE_SELECTED_DOT(q2_K_q8_K);
    case 12: return QWEN35MOE_SELECTED_DOT(q4_K_q8_K);
    case 13: return QWEN35MOE_SELECTED_DOT(q5_K_q8_K);
    case 14: return QWEN35MOE_SELECTED_DOT(q6_K_q8_K);
    case 16: return QWEN35MOE_SELECTED_DOT(iq2_xxs_q8_K);
    case 17: return QWEN35MOE_SELECTED_DOT(iq2_xs_q8_K);
    default: assert(0); return NULL;
    }
#undef QWEN35MOE_SELECTED_DOT
}
float ds4_qwen35moe_dot(int type, int n, const void *x, const void *activation) {
    float result = 0;
    ds4_qwen35moe_select_dot(type)(n, &result, 0, x, 0, activation, 0, 1);
    return result;
}

/* Same four-accumulator reduction as GGML's AVX float dot product. */
float ds4_qwen35moe_float_dot(int n, const float *x, const float *y) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 sums[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
    int end=n&~31;
    for(int i=0;i<end;i+=32) for(int j=0;j<4;j++)
        sums[j]=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+j*8),_mm256_loadu_ps(y+i+j*8),sums[j]);
    sums[0]=_mm256_add_ps(sums[0],sums[2]); sums[1]=_mm256_add_ps(sums[1],sums[3]);
    sums[0]=_mm256_add_ps(sums[0],sums[1]);
    __m128 a=_mm_add_ps(_mm256_castps256_ps128(sums[0]),_mm256_extractf128_ps(sums[0],1));
    a=_mm_hadd_ps(a,a);float result=_mm_cvtss_f32(_mm_hadd_ps(a,a));
    for(int i=end;i<n;i++) result+=x[i]*y[i];
    return result;
#else
    double sum=0;for(int i=0;i<n;i++)sum+=(double)(x[i]*y[i]);return (float)sum;
#endif
}

#if defined(__AVX2__) && defined(__FMA__)
inline static __m256 ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m256 ggml_v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = ggml_v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}


#endif

void ds4_qwen35moe_silu(int n, float *y, const float *x) {
    int i=0;
#if defined(__AVX2__) && defined(__FMA__)
    for(;i+7<n;i+=8) _mm256_storeu_ps(y+i,ggml_v_silu(_mm256_loadu_ps(x+i)));
#endif
    for(;i<n;i++)y[i]=x[i]/(1+expf(-x[i]));
}
double ds4_qwen35moe_softmax(int n, float *y, const float *x, float max) {
    int i=0;double sum=0;
#if defined(__AVX2__) && defined(__FMA__)
    for(;i+7<n;i+=8) {
        __m256 val=ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x+i),_mm256_set1_ps(max)));
        _mm256_storeu_ps(y+i,val);
        __m128 val2=_mm_add_ps(_mm256_extractf128_ps(val,1),_mm256_castps256_ps128(val));
        val2=_mm_add_ps(val2,_mm_movehl_ps(val2,val2));
        val2=_mm_add_ss(val2,_mm_movehdup_ps(val2));
        sum+=(double)_mm_cvtss_f32(val2);
    }
#endif
    for(;i<n;i++){float val=expf(x[i]-max);sum+=val;y[i]=val;}
    return sum;
}

/* Selected AVX2 form of pinned ggml-cpu/simd-gemm.h, 6 query rows by 16
 * columns. Keeping K in the inner loop preserves flash attention's FMA order. */
#if defined(__AVX2__) && defined(__FMA__)
static void qwen35moe_gemm_6x16(float *c, const float *a, const float *b, int k, int n,
                          int rows, int cols) {
    __m256 acc[6][2];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            acc[i][j] = _mm256_loadu_ps(c + i * n + j * 8);
    for (int x = 0; x < k; x++) {
        __m256 bv0 = _mm256_loadu_ps(b + x * n);
        __m256 bv1 = cols == 2 ? _mm256_loadu_ps(b + x * n + 8) : _mm256_setzero_ps();
        for (int i = 0; i < rows; i++) {
            __m256 av = _mm256_set1_ps(a[i * k + x]);
            acc[i][0] = _mm256_fmadd_ps(bv0, av, acc[i][0]);
            if (cols == 2) acc[i][1] = _mm256_fmadd_ps(bv1, av, acc[i][1]);
        }
    }
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            _mm256_storeu_ps(c + i * n + j * 8, acc[i][j]);
}
#endif
void ds4_qwen35moe_gemm_f32(float *c, const float *a, const float *b, int m, int k, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    for (int i = 0; i < m; i += 6) {
        int rows = m - i < 6 ? m - i : 6;
        int j = 0;
        for (; j + 16 <= n; j += 16)
            qwen35moe_gemm_6x16(c + i * n + j, a + i * k, b + j, k, n, rows, 2);
        for (; j + 8 <= n; j += 8)
            qwen35moe_gemm_6x16(c + i * n + j, a + i * k, b + j, k, n, rows, 1);
        for (; j < n; j++)
            for (int r = 0; r < rows; r++)
                for (int x = 0; x < k; x++)
                    c[(i + r) * n + j] = fmaf(a[(i + r) * k + x], b[x * n + j], c[(i + r) * n + j]);
    }
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int x = 0; x < k; x++)
                c[i * n + j] = fmaf(a[i * k + x], b[x * n + j], c[i * n + j]);
#endif
}
void ds4_qwen35moe_f16_to_f32(int n, const unsigned short *src, float *dst) {
    ds4_qwen35moe_init();
    int i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8) {
        __m128i half = _mm_loadu_si128((const __m128i *)(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(half));
    }
#endif
    for (; i < n; i++) dst[i] = ds4_qwen35moe_half(src[i]);
}

#if defined(__AVX2__) && defined(__FMA__)
/* IQ2_XXS/IQ2_XS panel GEMM selected from pinned ggml-cpu/iqp.cpp. The Ryzen
 * path uses signed AVX2 pair sums (no VNNI bias). Eight quantized weight rows
 * share their decode across four Q8_K activations. */
struct qwen35moe_iqp_panel {
    float dfac[8];
    int32_t bias[8];
    int8_t iscales[16 * 8];
    int8_t qs[256 * 8];
};
static uint8_t qwen35moe_iqp_signs(uint32_t v) {
    uint32_t p = v ^ (v >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    return (uint8_t)(v ^ ((p & 1) << 7));
}
static void qwen35moe_iqp_store8(int8_t *dst, uint64_t g0, uint64_t g1,
                           uint64_t g2, uint64_t g3, uint32_t signs) {
    const __m256i grid = _mm256_set_epi64x((int64_t)g3, (int64_t)g2,
                                           (int64_t)g1, (int64_t)g0);
    const __m256i broadcast = _mm256_setr_epi8(
        0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i signbytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int32_t)signs), broadcast);
    const __m256i selector = _mm256_set1_epi64x((int64_t)0x8040201008040201ULL);
    const __m256i mask = _mm256_cmpeq_epi8(_mm256_and_si256(signbytes, selector), selector);
    _mm256_storeu_si256((__m256i *)dst,
                       _mm256_sub_epi8(_mm256_xor_si256(grid, mask), mask));
}
static void qwen35moe_iqp_decode16(int type, const void *raw, int8_t vals[256],
                              int8_t scales[16], float *dfac) {
    if (type == 16) {
        const block_iq2_xxs *x = raw;
        *dfac = ds4_qwen35moe_half(x->d) * 0.125f;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux[2];
            memcpy(aux, x->qs + 4 * ib, sizeof(aux));
            const uint8_t *indices = (const uint8_t *)aux;
            const int8_t scale = (int8_t)(2 * (aux[1] >> 28) + 1);
            scales[2 * ib] = scales[2 * ib + 1] = scale;
            const uint32_t signs = (uint32_t)qwen35moe_iqp_signs((aux[1] >> 0) & 127) |
                (uint32_t)qwen35moe_iqp_signs((aux[1] >> 7) & 127) << 8 |
                (uint32_t)qwen35moe_iqp_signs((aux[1] >> 14) & 127) << 16 |
                (uint32_t)qwen35moe_iqp_signs((aux[1] >> 21) & 127) << 24;
            qwen35moe_iqp_store8(vals + 32 * ib, iq2xxs_grid[indices[0]], iq2xxs_grid[indices[1]],
                           iq2xxs_grid[indices[2]], iq2xxs_grid[indices[3]], signs);
        }
    } else {
        const block_iq2_xs *x = raw;
        *dfac = ds4_qwen35moe_half(x->d) * 0.125f;
        for (int ib = 0; ib < 8; ib++) {
            scales[2 * ib] = (int8_t)(2 * (x->scales[ib] & 15) + 1);
            scales[2 * ib + 1] = (int8_t)(2 * (x->scales[ib] >> 4) + 1);
            const uint16_t *q = x->qs + 4 * ib;
            const uint32_t signs = (uint32_t)qwen35moe_iqp_signs(q[0] >> 9) |
                (uint32_t)qwen35moe_iqp_signs(q[1] >> 9) << 8 |
                (uint32_t)qwen35moe_iqp_signs(q[2] >> 9) << 16 |
                (uint32_t)qwen35moe_iqp_signs(q[3] >> 9) << 24;
            qwen35moe_iqp_store8(vals + 32 * ib, iq2xs_grid[q[0] & 511], iq2xs_grid[q[1] & 511],
                           iq2xs_grid[q[2] & 511], iq2xs_grid[q[3] & 511], signs);
        }
    }
}
static void qwen35moe_iqp_interleave(int8_t *dst, const int8_t vals[8][256], int off) {
    __m256i v[8], a[8], b[8];
    for (int r = 0; r < 8; r++) v[r] = _mm256_loadu_si256((const __m256i *)(vals[r] + off));
    a[0] = _mm256_unpacklo_epi32(v[0], v[1]); a[1] = _mm256_unpackhi_epi32(v[0], v[1]);
    a[2] = _mm256_unpacklo_epi32(v[2], v[3]); a[3] = _mm256_unpackhi_epi32(v[2], v[3]);
    a[4] = _mm256_unpacklo_epi32(v[4], v[5]); a[5] = _mm256_unpackhi_epi32(v[4], v[5]);
    a[6] = _mm256_unpacklo_epi32(v[6], v[7]); a[7] = _mm256_unpackhi_epi32(v[6], v[7]);
    b[0] = _mm256_unpacklo_epi64(a[0], a[2]); b[1] = _mm256_unpackhi_epi64(a[0], a[2]);
    b[2] = _mm256_unpacklo_epi64(a[1], a[3]); b[3] = _mm256_unpackhi_epi64(a[1], a[3]);
    b[4] = _mm256_unpacklo_epi64(a[4], a[6]); b[5] = _mm256_unpackhi_epi64(a[4], a[6]);
    b[6] = _mm256_unpacklo_epi64(a[5], a[7]); b[7] = _mm256_unpackhi_epi64(a[5], a[7]);
    for (int i = 0; i < 4; i++) {
        _mm256_storeu_si256((__m256i *)(dst + i * 32), _mm256_permute2x128_si256(b[i], b[i + 4], 0x20));
        _mm256_storeu_si256((__m256i *)(dst + (i + 4) * 32), _mm256_permute2x128_si256(b[i], b[i + 4], 0x31));
    }
}
static void qwen35moe_iqp_decode_panel(int type, const uint8_t *weights, size_t stride,
                                  int blocks, struct qwen35moe_iqp_panel *panel) {
    const size_t block_bytes = type == 16 ? sizeof(block_iq2_xxs) : sizeof(block_iq2_xs);
    for (int block = 0; block < blocks; block++) {
        int8_t vals[8][256], scales[8][16];
        float dfac[8];
        for (int r = 0; r < 8; r++)
            qwen35moe_iqp_decode16(type, weights + r * stride + block * block_bytes,
                              vals[r], scales[r], &dfac[r]);
        for (int r = 0; r < 8; r++) {
            panel[block].dfac[r] = dfac[r];
            for (int sb = 0; sb < 16; sb++) panel[block].iscales[sb * 8 + r] = scales[r][sb];
        }
        for (int group = 0; group < 8; group++)
            qwen35moe_iqp_interleave(panel[block].qs + group * 256, vals, group * 32);
    }
}
static __m256i qwen35moe_iqp_dot4(__m256i acc, __m256i weights, __m256i activations) {
    const __m256i abs_weights = _mm256_sign_epi8(weights, weights);
    const __m256i signed_activations = _mm256_sign_epi8(activations, weights);
    const __m256i pairs = _mm256_maddubs_epi16(abs_weights, signed_activations);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}
static void qwen35moe_iqp_gemm4(const struct qwen35moe_iqp_panel *panel,
                           const block_q8_K *const activation[4], int blocks, float out[4][8]) {
    __m256 sums[4];
    for (int m = 0; m < 4; m++) sums[m] = _mm256_setzero_ps();
    for (int block = 0; block < blocks; block++) {
        __m256i isum[4];
        for (int m = 0; m < 4; m++) isum[m] = _mm256_setzero_si256();
        for (int sb = 0; sb < 16; sb++) {
            const int8_t *weights = panel[block].qs + sb * 128;
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(weights + 0));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(weights + 32));
            __m256i w2 = _mm256_loadu_si256((const __m256i *)(weights + 64));
            __m256i w3 = _mm256_loadu_si256((const __m256i *)(weights + 96));
            __m256i scale = _mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(panel[block].iscales + sb * 8)));
            for (int m = 0; m < 4; m++) {
                __m128i y = _mm_loadu_si128((const __m128i *)(activation[m][block].qs + sb * 16));
                __m256i by = _mm256_broadcastsi128_si256(y);
                __m256i dot = _mm256_setzero_si256();
                dot = qwen35moe_iqp_dot4(dot, w0, _mm256_shuffle_epi32(by, 0x00));
                dot = qwen35moe_iqp_dot4(dot, w1, _mm256_shuffle_epi32(by, 0x55));
                dot = qwen35moe_iqp_dot4(dot, w2, _mm256_shuffle_epi32(by, 0xaa));
                dot = qwen35moe_iqp_dot4(dot, w3, _mm256_shuffle_epi32(by, 0xff));
                isum[m] = _mm256_add_epi32(isum[m], _mm256_mullo_epi32(dot, scale));
            }
        }
        __m256 dfac = _mm256_loadu_ps(panel[block].dfac);
        for (int m = 0; m < 4; m++)
            sums[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum[m]),
                                       _mm256_mul_ps(dfac, _mm256_set1_ps(activation[m][block].d)),
                                       sums[m]);
    }
    for (int m = 0; m < 4; m++) _mm256_storeu_ps(out[m], sums[m]);
}
#endif
void ds4_qwen35moe_iqp_matbatch(int type, int n, const void *weights, size_t weight_stride,
                           const void *activations, size_t activation_stride,
                           float *out, size_t output_stride, int rows, int count) {
    ds4_qwen35moe_init();
#if defined(__AVX2__) && defined(__FMA__)
    assert((type == 16 || type == 17) && n % 256 == 0 && n <= 4096 && rows % 8 == 0);
    const int blocks = n / 256;
    const uint8_t *w = weights, *a = activations;
    for (int r = 0; r < rows; r += 8) {
        struct qwen35moe_iqp_panel panel[16];
        qwen35moe_iqp_decode_panel(type, w + r * weight_stride, weight_stride, blocks, panel);
        for (int token = 0; token < count; token += 4) {
            const block_q8_K *packed[4];
            int used = count - token < 4 ? count - token : 4;
            for (int m = 0; m < 4; m++) {
                int b = token + (m < used ? m : used - 1);
                packed[m] = (const block_q8_K *)(a + (size_t)b * activation_stride);
            }
            float result[4][8];
            qwen35moe_iqp_gemm4(panel, packed, blocks, result);
            for (int m = 0; m < used; m++)
                memcpy(out + (size_t)(token + m) * output_stride + r, result[m], 8 * sizeof(float));
        }
    }
#else
    (void)type; (void)n; (void)weights; (void)weight_stride;
    (void)activations; (void)activation_stride; (void)out; (void)output_stride;
    (void)rows; (void)count;
#endif
}
/* Eight mmap Q4_K rows are unpacked into one short-lived panel. The
 * per-64-value integer reductions and separate min accumulator follow the
 * selected AVX2 Q4_Kx8/Q8_Kx4 GEMM without retaining a second weight copy. */
#if defined(__AVX2__) && defined(__FMA__)
struct qwen35moe_q4k_panel {
    float d[8], dmin[8];
    uint8_t scale[8][8], min[8][8];
    uint8_t q[8][8][32]; /* [32-value subblock][4-value group][row][value] */
};
static void qwen35moe_q4k_pack8(struct qwen35moe_q4k_panel *dst, const uint8_t *weights,
                          size_t stride, int block) {
    int8_t unpacked[8][256];
    const __m256i nibble = _mm256_set1_epi8(15);
    for (int row = 0; row < 8; row++) {
        const block_q4_K *src = (const block_q4_K *)(weights + row * stride +
                                                     block * sizeof(block_q4_K));
        dst->d[row] = ds4_qwen35moe_half(src->d);
        dst->dmin[row] = ds4_qwen35moe_half(src->dmin);
        for (int sb = 0; sb < 8; sb++)
            get_scale_min_k4(sb, src->scales, &dst->scale[sb][row], &dst->min[sb][row]);
        for (int group = 0; group < 4; group++) {
            __m256i packed = _mm256_loadu_si256((const __m256i *)(src->qs + group * 32));
            __m256i low = _mm256_and_si256(packed, nibble);
            __m256i high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble);
            _mm256_storeu_si256((__m256i *)(unpacked[row] + group * 64), low);
            _mm256_storeu_si256((__m256i *)(unpacked[row] + group * 64 + 32), high);
        }
    }
    for (int sb = 0; sb < 8; sb++)
        qwen35moe_iqp_interleave((int8_t *)dst->q[sb], unpacked, sb * 32);
}
static inline __m256i qwen35moe_q4k_dot32(const struct qwen35moe_q4k_panel *panel,
                                     const block_q8_K *activation, int sb) {
    __m256i dots = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    for (int group = 0; group < 8; group++) {
        __m256i w = _mm256_loadu_si256((const __m256i *)panel->q[sb][group]);
        uint32_t packed;
        memcpy(&packed, activation->qs + sb * 32 + group * 4, sizeof(packed));
        __m256i a = _mm256_set1_epi32((int)packed);
        dots = _mm256_add_epi32(dots,
            _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), ones));
    }
    return dots;
}
static void qwen35moe_q4k_panel_dot(const struct qwen35moe_q4k_panel *panel, int blocks,
                              const block_q8_K *activation, float out[8], int gemv) {
    __m256 acc = _mm256_setzero_ps(), min_acc = _mm256_setzero_ps();
    for (int block = 0; block < blocks; block++) {
        __m256i block_sum = _mm256_setzero_si256();
        __m256i block_min = _mm256_setzero_si256();
        const __m256 d = _mm256_mul_ps(_mm256_loadu_ps(panel[block].d),
                                       _mm256_set1_ps(activation[block].d));
        const __m256 dmin = _mm256_mul_ps(_mm256_loadu_ps(panel[block].dmin),
                                          _mm256_set1_ps(activation[block].d));
        for (int pair = 0; pair < 4; pair++) {
            const int first = 2 * pair, second = first + 1;
            __m256i s0 = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)panel[block].scale[first]));
            __m256i s1 = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)panel[block].scale[second]));
            __m256i dot0 = qwen35moe_q4k_dot32(panel + block, activation + block, first);
            __m256i dot1 = qwen35moe_q4k_dot32(panel + block, activation + block, second);
            __m256i scaled = _mm256_add_epi32(_mm256_mullo_epi32(dot0, s0),
                                               _mm256_mullo_epi32(dot1, s1));
            const int qsum0 = activation[block].bsums[first * 2] +
                              activation[block].bsums[first * 2 + 1];
            const int qsum1 = activation[block].bsums[second * 2] +
                              activation[block].bsums[second * 2 + 1];
            __m256i m0 = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)panel[block].min[first]));
            __m256i m1 = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)panel[block].min[second]));
            __m256i mins = _mm256_add_epi32(_mm256_mullo_epi32(m0, _mm256_set1_epi32(qsum0)),
                                             _mm256_mullo_epi32(m1, _mm256_set1_epi32(qsum1)));
            if (gemv) {
                block_sum = _mm256_add_epi32(block_sum, scaled);
                block_min = _mm256_add_epi32(block_min, mins);
            } else {
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(scaled), d, acc);
                min_acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mins), dmin, min_acc);
            }
        }
        if (gemv) {
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(block_sum), d, acc);
            min_acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(block_min), dmin, min_acc);
        }
    }
    _mm256_storeu_ps(out, _mm256_sub_ps(acc, min_acc));
}
#endif
void ds4_qwen35moe_q4k_matbatch(int n, const void *weights, size_t weight_stride,
                           const void *activations, size_t activation_stride,
                           float *out, size_t output_stride, int rows, int count) {
#if defined(__AVX2__) && defined(__FMA__)
    assert(n > 0 && n <= 4096 && n % 256 == 0 && rows % 8 == 0);
    const int blocks = n / 256;
    for (int row = 0; row < rows; row += 8) {
        struct qwen35moe_q4k_panel panel[16];
        for (int block = 0; block < blocks; block++)
            qwen35moe_q4k_pack8(panel + block,
                (const uint8_t *)weights + row * weight_stride, weight_stride, block);
        for (int token = 0; token < count; token++) {
            const block_q8_K *a = (const block_q8_K *)((const uint8_t *)activations +
                                                        (size_t)token * activation_stride);
            float value[8];
            qwen35moe_q4k_panel_dot(panel, blocks, a, value,
                               token >= count - count % 4);
            memcpy(out + (size_t)token * output_stride + row, value, sizeof(value));
        }
    }
#else
    (void)n; (void)weights; (void)weight_stride; (void)activations;
    (void)activation_stride; (void)out; (void)output_stride; (void)rows; (void)count;
#endif
}
/* One-query repacked-GEMV reduction directly over the original mmap row.
 * GGML accumulates quantized dot/min terms across a whole Q4_K block before
 * applying one FMA per block; this avoids repacking every decode token. */
void ds4_qwen35moe_q4k_gemv_dot(int n, float *out, size_t out_stride,
                           const void *weight, size_t weight_stride,
                           const void *activation, size_t activation_stride, int count) {
    assert(n > 0 && n % 256 == 0 && count == 1);
    (void)out_stride; (void)weight_stride; (void)activation_stride;
    const block_q4_K *w = weight;
    const block_q8_K *a = activation;
    float sum = 0, min_sum = 0;
    for (int block = 0; block < n / 256; block++) {
        int32_t integer_sum = 0, integer_min = 0;
        for (int group = 0; group < 4; group++) {
            uint8_t scale_low, min_low, scale_high, min_high;
            get_scale_min_k4(group * 2, w[block].scales, &scale_low, &min_low);
            get_scale_min_k4(group * 2 + 1, w[block].scales, &scale_high, &min_high);
            int dot_low = 0, dot_high = 0;
#if defined(__AVX2__)
            const __m256i raw = _mm256_loadu_si256((const __m256i *)(w[block].qs + group * 32));
            const __m256i mask = _mm256_set1_epi8(15);
            const __m256i low = _mm256_and_si256(raw, mask);
            const __m256i high = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask);
            const __m256i av0 = _mm256_loadu_si256((const __m256i *)(a[block].qs + group * 64));
            const __m256i av1 = _mm256_loadu_si256((const __m256i *)(a[block].qs + group * 64 + 32));
            const __m256i ones = _mm256_set1_epi16(1);
            __m256i dots0 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, av0), ones);
            __m256i dots1 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, av1), ones);
            int32_t lanes0[8], lanes1[8];
            _mm256_storeu_si256((__m256i *)lanes0, dots0);
            _mm256_storeu_si256((__m256i *)lanes1, dots1);
            for (int lane = 0; lane < 8; lane++) {
                dot_low += lanes0[lane];
                dot_high += lanes1[lane];
            }
#else
            for (int i = 0; i < 32; i++) {
                uint8_t packed = w[block].qs[group * 32 + i];
                dot_low += (packed & 15) * a[block].qs[group * 64 + i];
                dot_high += (packed >> 4) * a[block].qs[group * 64 + 32 + i];
            }
#endif
            integer_sum += dot_low * scale_low + dot_high * scale_high;
            int bsum0 = a[block].bsums[group * 4] + a[block].bsums[group * 4 + 1];
            int bsum1 = a[block].bsums[group * 4 + 2] + a[block].bsums[group * 4 + 3];
            integer_min += bsum0 * min_low + bsum1 * min_high;
        }
        const float d = ds4_qwen35moe_half(w[block].d) * a[block].d;
        const float dmin = ds4_qwen35moe_half(w[block].dmin) * a[block].d;
        sum = fmaf((float)integer_sum, d, sum);
        min_sum = fmaf((float)integer_min, dmin, min_sum);
    }
    *out = sum - min_sum;
}
/* Per-output reduction of pinned llamafile tinyBLAS<8, f32>. Several Qwen
 * control matrices use this path only when GGML sees two or more token rows. */
#if defined(__AVX2__) && defined(__FMA__)
static inline float qwen35moe_sgemm_float_reduce(__m256 accum) {
    __m128 sum = _mm_add_ps(_mm256_extractf128_ps(accum, 1), _mm256_castps256_ps128(accum));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}
#endif
float ds4_qwen35moe_sgemm_float_dot(int n, const float *weights, const float *activation) {
#if defined(__AVX2__) && defined(__FMA__)
    assert(n % 8 == 0);
    __m256 accum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8)
        accum = _mm256_fmadd_ps(_mm256_loadu_ps(weights + i),
                                _mm256_loadu_ps(activation + i), accum);
    return qwen35moe_sgemm_float_reduce(accum);
#else
    return ds4_qwen35moe_float_dot(n, weights, activation);
#endif
}
/* Keep the same K-order FMA and horizontal reduction for four output rows
 * against two activations, while loading each weight/activation vector once. */
void ds4_qwen35moe_sgemm_float_tile4x2(int n, const float *weights, size_t weight_stride,
                                   const float *activation, size_t activation_stride,
                                   float *out, size_t output_stride, int rows, int columns) {
    assert(rows >= 1 && rows <= 4 && columns >= 1 && columns <= 2);
#if defined(__AVX2__) && defined(__FMA__)
    assert(n % 8 == 0);
    __m256 accum[2][4];
    for (int c = 0; c < columns; c++)
        for (int r = 0; r < rows; r++) accum[c][r] = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 x[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int c = 0; c < columns; c++)
            x[c] = _mm256_loadu_ps(activation + (size_t)c * activation_stride + i);
        for (int r = 0; r < rows; r++) {
            __m256 w = _mm256_loadu_ps(weights + (size_t)r * weight_stride + i);
            for (int c = 0; c < columns; c++)
                accum[c][r] = _mm256_fmadd_ps(w, x[c], accum[c][r]);
        }
    }
    for (int c = 0; c < columns; c++)
        for (int r = 0; r < rows; r++)
            out[(size_t)c * output_stride + r] = qwen35moe_sgemm_float_reduce(accum[c][r]);
#else
    for (int c = 0; c < columns; c++)
        for (int r = 0; r < rows; r++)
            out[(size_t)c * output_stride + r] = ds4_qwen35moe_sgemm_float_dot(
                n, weights + (size_t)r * weight_stride,
                activation + (size_t)c * activation_stride);
#endif
}
/* F16C forms of pinned ggml_vec_scale_f16 / ggml_vec_mad_f16. Each update
 * rounds back to F16, as required by single-query flash attention. */
void ds4_qwen35moe_f16_scale(int n, unsigned short *accumulator, float scale) {
    ds4_qwen35moe_init();
    int i = 0;
#if defined(__AVX2__) && defined(__F16C__)
    __m256 factor = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        __m128i old = _mm_loadu_si128((const __m128i *)(accumulator + i));
        __m256 result = _mm256_mul_ps(_mm256_cvtph_ps(old), factor);
        _mm_storeu_si128((__m128i *)(accumulator + i), _mm256_cvtps_ph(result, 0));
    }
#endif
    for (; i < n; i++) accumulator[i] = ds4_qwen35moe_to_half(ds4_qwen35moe_half(accumulator[i]) * scale);
}
void ds4_qwen35moe_f16_mad(int n, unsigned short *accumulator,
                      const unsigned short *value, float scale) {
    ds4_qwen35moe_init();
    int i = 0;
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
    __m256 factor = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        __m128i old = _mm_loadu_si128((const __m128i *)(accumulator + i));
        __m128i val = _mm_loadu_si128((const __m128i *)(value + i));
        __m256 result = _mm256_fmadd_ps(_mm256_cvtph_ps(val), factor, _mm256_cvtph_ps(old));
        _mm_storeu_si128((__m128i *)(accumulator + i), _mm256_cvtps_ph(result, 0));
    }
#endif
    for (; i < n; i++) accumulator[i] = ds4_qwen35moe_to_half(
        fmaf(ds4_qwen35moe_half(value[i]), scale, ds4_qwen35moe_half(accumulator[i])));
}

/* GGML argsorts all 256 route probabilities before taking eight. Equal keys
 * retain the partition order of the pinned GCC std::sort implementation; a
 * stable index-order top-k changes MoE accumulation at tied boundaries. This
 * independent C introsort follows that observable partition/tail behavior. */
static inline int qwen35moe_route_before(const float *p, int a, int b) {
    return p[a] > p[b];
}
static inline void qwen35moe_route_swap(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}
static int qwen35moe_route_partition(int ids[256], int lo, int hi, const float *p) {
    int a = lo + 1, b = lo + (hi - lo) / 2, c = hi - 1;
    int pivot;
    if (qwen35moe_route_before(p, ids[a], ids[b])) {
        if (qwen35moe_route_before(p, ids[b], ids[c])) pivot = b;
        else pivot = qwen35moe_route_before(p, ids[a], ids[c]) ? c : a;
    } else {
        if (qwen35moe_route_before(p, ids[a], ids[c])) pivot = a;
        else pivot = qwen35moe_route_before(p, ids[b], ids[c]) ? c : b;
    }
    qwen35moe_route_swap(&ids[lo], &ids[pivot]);
    int left = lo + 1, right = hi;
    for (;;) {
        while (qwen35moe_route_before(p, ids[left], ids[lo])) left++;
        right--;
        while (qwen35moe_route_before(p, ids[lo], ids[right])) right--;
        if (left >= right) return left;
        qwen35moe_route_swap(&ids[left], &ids[right]);
        left++;
    }
}
static void qwen35moe_route_quicksort(int ids[256], int lo, int hi, int depth, const float *p) {
    while (hi - lo > 16) {
        if (!depth) {
            /* A bounded fallback for adversarial scores. Ordinary model rows
             * stay inside the median-partition depth budget. */
            for (int i = lo; i < hi; i++)
                for (int j = i + 1; j < hi; j++)
                    if (qwen35moe_route_before(p, ids[j], ids[i])) qwen35moe_route_swap(&ids[i], &ids[j]);
            return;
        }
        depth--;
        int middle = qwen35moe_route_partition(ids, lo, hi, p);
        qwen35moe_route_quicksort(ids, middle, hi, depth, p);
        hi = middle;
    }
}
void ds4_qwen35moe_router_top8(const float probabilities[256], int chosen[8]) {
    int ids[256];
    for (int i = 0; i < 256; i++) ids[i] = i;
    qwen35moe_route_quicksort(ids, 0, 256, 16, probabilities);
    for (int i = 1; i < 256; i++) {
        int value = ids[i], j = i;
        if (i < 16 && qwen35moe_route_before(probabilities, value, ids[0])) {
            memmove(ids + 1, ids, (size_t)i * sizeof(int));
            ids[0] = value;
            continue;
        }
        while (j > 0 && qwen35moe_route_before(probabilities, value, ids[j - 1])) {
            ids[j] = ids[j - 1];
            j--;
        }
        ids[j] = value;
    }
    memcpy(chosen, ids, 8 * sizeof(int));
}
