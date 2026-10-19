/* TurboQuant (TQ-MSE) KV codec for the Qwen3.8-Flash-Next attention caches.
 * CPU reference implementation; the GPU kernels mirror this arithmetic.
 * See ds4_qwen4_tq.h for the format contract.  Build WITHOUT -ffast-math. */
#include "ds4_qwen4_tq.h"

#include <math.h>
#include <string.h>

#define DS4_QWEN4_TQ_TABLES_DEFINITION
#include "ds4_qwen4_tq_tables.h"

#define TQ ds4_qwen4_tq_table

static float tq_sign(uint32_t d) {
    return (TQ.signs[d >> 5] >> (d & 31u)) & 1u ? 1.0f : -1.0f;
}

/* Per-width views of the concatenated tables (DS4_QWEN4_TQ_TAB_BASE). */
const float *ds4_qwen4_tq_cb(uint32_t bits) { return TQ.cb + DS4_QWEN4_TQ_TAB_BASE(bits); }
const float *ds4_qwen4_tq_mid(uint32_t bits) { return TQ.mid + DS4_QWEN4_TQ_TAB_BASE(bits); }

bool ds4_qwen4_tq_bits_supported(uint32_t bits) {
    return bits >= DS4_QWEN4_TQ_MIN_BITS && bits <= DS4_QWEN4_TQ_MAX_BITS;
}

/* Canonical in-place 256-point Walsh-Hadamard (Sylvester), stages ascending,
 * then the exact 1/16 scale.  Reads both pair members before writing. */
static void wht256(float *y) {
    for (uint32_t stride = 1; stride < DS4_QWEN4_TQ_DIM; stride <<= 1) {
        for (uint32_t base = 0; base < DS4_QWEN4_TQ_DIM; base += 2u * stride) {
            for (uint32_t i = 0; i < stride; i++) {
                const float a = y[base + i], b = y[base + i + stride];
                y[base + i] = a + b;
                y[base + i + stride] = a - b;
            }
        }
    }
    for (uint32_t i = 0; i < DS4_QWEN4_TQ_DIM; i++) y[i] *= 1.0f / 16.0f;
}

void ds4_qwen4_tq_rht(const float *x, float *y) {
    for (uint32_t d = 0; d < DS4_QWEN4_TQ_DIM; d++) y[d] = x[d] * tq_sign(d);
    wht256(y);
}

void ds4_qwen4_tq_rht_inverse(const float *y, float *x) {
    memmove(x, y, DS4_QWEN4_TQ_DIM * sizeof(float));
    wht256(x);
    for (uint32_t d = 0; d < DS4_QWEN4_TQ_DIM; d++) x[d] *= tq_sign(d);
}

/* The f16 pair mirrors tests/test_qwen4_kernels.c (keep them in sync). */
static uint16_t f32_to_f16_rne(float f) {
    union { float f; uint32_t u; } v = { f };
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint16_t half = (uint16_t)(sign | (mant >> shift));
        if ((mant >> (shift - 1)) & 1u) half++;
        return half;
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint16_t half = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if (mant & 0x1000u) half++;
    return half;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    union { float f; uint32_t u; } v;
    if (exp == 0) {
        if (mant == 0) { v.u = sign; return v.f; }
        uint32_t e = 127u - 15u + 1u, m = mant;
        while (!(m & 0x400u)) { m <<= 1; e--; }
        v.u = sign | (e << 23) | ((m & 0x3ffu) << 13);
        return v.f;
    }
    if (exp == 31u) { v.u = sign | 0x7f800000u | (mant << 13); return v.f; }
    v.u = sign | ((exp + 112u) << 23) | (mant << 13);
    return v.f;
}

uint32_t ds4_qwen4_tq_packed_width(uint32_t bits) {
    return bits * (DS4_QWEN4_TQ_DIM / 32u);   /* 256*bits/32 == bits*8 */
}

uint64_t ds4_qwen4_tq_norms_offset(uint64_t cap, uint32_t bits) {
    const uint64_t codes = cap * 2ull * ds4_qwen4_tq_packed_width(bits) * 4ull;
    return (codes + 15ull) / 16ull * 16ull;
}

uint64_t ds4_qwen4_tq_blob_bytes(uint64_t cap, uint32_t bits) {
    return ds4_qwen4_tq_norms_offset(cap, bits) + cap * 2ull * 2ull;
}

void ds4_qwen4_tq_quantize_row(const float *row, uint32_t bits,
                               uint32_t *codes, uint16_t *norm_f16) {
    float ss = 0.0f;
    for (uint32_t d = 0; d < DS4_QWEN4_TQ_DIM; d++) ss += row[d] * row[d];
    const float norm = sqrtf(ss);
    const float denom = norm > 1e-6f ? norm : 1e-6f;
    float rot[DS4_QWEN4_TQ_DIM];
    for (uint32_t d = 0; d < DS4_QWEN4_TQ_DIM; d++) rot[d] = row[d] / denom;
    ds4_qwen4_tq_rht(rot, rot);
    const float *mid = ds4_qwen4_tq_mid(bits);
    const uint32_t n_mid = (1u << bits) - 1u;
    memset(codes, 0, ds4_qwen4_tq_packed_width(bits) * sizeof(uint32_t));
    for (uint32_t d = 0; d < DS4_QWEN4_TQ_DIM; d++) {
        uint32_t lo = 0u, hi = n_mid;              /* #midpoints < rot[d] */
        while (lo < hi) {
            const uint32_t m = (lo + hi) / 2u;
            if (rot[d] > mid[m]) lo = m + 1u; else hi = m;
        }
        const uint32_t off = d * bits, w = off >> 5, s = off & 31u;
        codes[w] |= lo << s;
        if (s + bits > 32u) codes[w + 1u] |= lo >> (32u - s);
    }
    *norm_f16 = f32_to_f16_rne(norm);
}

void ds4_qwen4_tq_dequantize_row(const uint32_t *codes, uint16_t norm_f16,
                                 uint32_t bits, float *row) {
    const float *cb = ds4_qwen4_tq_cb(bits);
    const uint32_t mask = (1u << bits) - 1u;
    for (uint32_t d = 0; d < DS4_QWEN4_TQ_DIM; d++) {
        const uint32_t off = d * bits, w = off >> 5, s = off & 31u;
        uint32_t v = codes[w] >> s;
        if (s + bits > 32u) v |= codes[w + 1u] << (32u - s);
        row[d] = cb[v & mask];
    }
    ds4_qwen4_tq_rht_inverse(row, row);
    const float norm = f16_to_f32(norm_f16);
    for (uint32_t d = 0; d < DS4_QWEN4_TQ_DIM; d++) row[d] *= norm;
}

bool ds4_qwen4_tq_layer_want(uint32_t il, uint32_t n_layer, uint32_t n_nextn,
                             uint32_t attn_interval) {
    if (attn_interval == 0u || n_nextn >= n_layer) return false;
    const uint32_t n_trunk = n_layer - n_nextn;
    if (il >= n_trunk) return false;                    /* nextn/MTP: f16 */
    if ((il + 1u) % attn_interval != 0u) return false;  /* GDN/linear: no KV cache */
    if (n_trunk / attn_interval <= 2u) return true;     /* shallow stack: all layers */
    return il != n_trunk - 1u;                          /* deep: last attention stays f16 */
}
