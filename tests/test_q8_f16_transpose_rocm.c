/* Bit-exact test for the ROCm kernel that builds the transposed F16 cache of
 * Q8_0 weights (cuda_q8_f16_transpose_ptr).
 *
 * The CPU oracle follows the Q8_0 block layout (an F16 scale d followed by 32
 * int8 quants, value = d * q): the product of an F16 scale and an int8 is
 * exact in FP32, so rounding it once to F16 (round to nearest even, with
 * subnormals) is the only correctly rounded F16 result.  Shapes cover every
 * dimension pair the cache accepts, plus ragged ones that do not fill a tile
 * and rows whose length is not a multiple of the Q8_0 block.  Scales include
 * zeros, subnormals, the largest finite F16, infinities and NaN.
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK8_0 32u
#define Q8_0_BLOCK_BYTES 34u

static uint32_t rng_state = 0x9e3779b9u;

static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    float v;
    if (exp == 0u) {
        v = ldexpf((float)mant, -24);
    } else if (exp == 31u) {
        v = mant ? NAN : INFINITY;
    } else {
        v = ldexpf((float)(mant | 0x400u), (int)exp - 25);
    }
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    bits |= sign;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

/* Round an FP32 value to F16, nearest even, keeping subnormals. */
static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    const int e = (int)exp - 127;
    if (e > 15) return (uint16_t)(sign | 0x7c00u);
    if (exp == 0u) return sign; /* FP32 subnormals are far below F16 range */
    mant |= 0x800000u;
    /* Number of low mantissa bits to drop: 13 for normals, more below 2^-14. */
    int shift = e >= -14 ? 13 : 13 + (-14 - e);
    if (shift > 24) return sign;
    const uint32_t kept = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    uint32_t r = kept + (rem > half || (rem == half && (kept & 1u)));
    if (e >= -14) {
        /* r holds 1.mantissa in 11 bits; a carry may bump the exponent. */
        uint32_t he = (uint32_t)(e + 15);
        if (r >= 0x800u) {
            r >>= 1;
            he++;
        }
        if (he >= 31u) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | (he << 10) | (r & 0x3ffu));
    }
    /* Subnormal: r is the mantissa; rounding up to 0x400 yields the smallest normal. */
    return (uint16_t)(sign | r);
}

static bool f16_is_nan(uint16_t h) {
    return (h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0u;
}

static const uint16_t special_scales[] = {
    0x0000u, 0x8000u, 0x0001u, 0x83ffu, 0x0400u, 0x7bffu, 0xfbffu, 0x7c00u, 0xfc00u, 0x7e00u,
};

static uint16_t random_scale(void) {
    /* Mostly the magnitudes real Q8_0 weights use, with every special value
     * showing up a few times per matrix. */
    if ((rng_next() & 63u) == 0u) {
        return special_scales[rng_next() % (sizeof(special_scales) / sizeof(special_scales[0]))];
    }
    const float mag = ldexpf(1.0f + (float)(rng_next() & 0xffffu) / 65536.0f, -(int)(rng_next() % 20u));
    return f32_to_f16((rng_next() & 1u) ? -mag : mag);
}

static bool run_shape(uint64_t in_dim, uint64_t out_dim, bool report_time) {
    const uint64_t blocks = (in_dim + QK8_0 - 1u) / QK8_0;
    const uint64_t q8_bytes = out_dim * blocks * Q8_0_BLOCK_BYTES;
    const uint64_t n = in_dim * out_dim;
    unsigned char *q8 = malloc((size_t)q8_bytes);
    uint16_t *got = malloc((size_t)n * sizeof(uint16_t));
    if (!q8 || !got) {
        fprintf(stderr, "Q8_0 F16 transpose: host allocation failed for %llux%llu\n",
                (unsigned long long)in_dim, (unsigned long long)out_dim);
        free(q8);
        free(got);
        return false;
    }
    for (uint64_t b = 0; b < out_dim * blocks; b++) {
        unsigned char *blk = q8 + b * Q8_0_BLOCK_BYTES;
        const uint16_t d = random_scale();
        memcpy(blk, &d, sizeof(d));
        for (uint32_t j = 0; j < QK8_0; j++) blk[2u + j] = (unsigned char)rng_next();
    }
    /* Pin both ends of the quant range in the first block. */
    q8[2] = 0x80u;
    if (in_dim > 1u) q8[3] = 0x7fu;

    double ms = 0.0;
    bool ok = ds4_gpu_test_q8_0_f16_transpose(q8, in_dim, out_dim, got, &ms) != 0;
    uint64_t bad = 0;
    for (uint64_t row = 0; ok && row < out_dim; row++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            const unsigned char *blk = q8 + (row * blocks + i / QK8_0) * Q8_0_BLOCK_BYTES;
            uint16_t d;
            memcpy(&d, blk, sizeof(d));
            const int8_t q = (int8_t)blk[2u + i % QK8_0];
            const uint16_t want = f32_to_f16(f16_to_f32(d) * (float)q);
            const uint16_t have = got[i * out_dim + row];
            if (have == want || (f16_is_nan(have) && f16_is_nan(want))) continue;
            if (bad < 8u) {
                fprintf(stderr,
                        "  %llux%llu row %llu col %llu: scale 0x%04x q %d want 0x%04x got 0x%04x\n",
                        (unsigned long long)in_dim, (unsigned long long)out_dim,
                        (unsigned long long)row, (unsigned long long)i, d, q, want, have);
            }
            bad++;
        }
    }
    if (bad) ok = false;
    fprintf(stderr, "Q8_0 F16 transpose in_dim %5llu out_dim %5llu: %s",
            (unsigned long long)in_dim, (unsigned long long)out_dim, ok ? "ok" : "FAIL");
    if (bad) fprintf(stderr, " (%llu wrong values)", (unsigned long long)bad);
    if (report_time) {
        const double bytes = (double)q8_bytes + (double)n * 2.0;
        fprintf(stderr, ", kernel %.3f ms, %.1f GB/s", ms, bytes / (ms * 1e-3) / 1e9);
    }
    fprintf(stderr, "\n");
    free(q8);
    free(got);
    return ok;
}

int main(void) {
    static const uint64_t shapes[][2] = {
        /* in_dim, out_dim: DS4 Flash attn_output_b (8192 x 4096), the only
         * tensor the prefill caches transposed, then every pair
         * cuda_q8_f16_cache_allowed accepts by shape. */
        {8192u, 4096u},
        {1024u, 32768u},
        {4096u, 2048u},
        {2048u, 4096u},
        {4096u, 1024u},
        {4096u, 512u},
        /* Ragged shapes: partial tiles in both directions, a single row or
         * column, and rows that end inside a Q8_0 block. */
        {32u, 1u},
        {1u, 77u},
        {96u, 100u},
        {160u, 129u},
        {1056u, 70u},
        {200u, 333u},
        {4095u, 65u},
    };
    if (!ds4_gpu_init()) {
        fprintf(stderr, "Q8_0 F16 transpose: ds4_gpu_init failed\n");
        return 1;
    }
    bool ok = true;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        ok = run_shape(shapes[s][0], shapes[s][1], false) && ok;
    }
    /* Time attn_output_b again, now that the kernel is loaded: the first
     * launch in a process also pays for loading the code object. */
    ok = run_shape(shapes[0][0], shapes[0][1], true) && ok;
    ds4_gpu_cleanup();
    fprintf(stderr, "Q8_0 F16 transpose ROCm: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
