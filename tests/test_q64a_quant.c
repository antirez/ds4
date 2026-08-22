#include "../gguf-tools/quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NROWS = 64, NCOLS = 64 };

static uint32_t f32_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static float bf16_effective(float v) {
    return ds4q_bf16_to_f32((uint16_t)(f32_bits(v) >> 16));
}

static int clamp_round(float value, int maxq) {
    int q = (int)roundf(value);
    if (q < 0) return 0;
    if (q > maxq) return maxq;
    return q;
}

static int packed_code(const uint8_t *block, int index, int bits) {
    if (bits == 4) {
        return (block[index >> 1] >> ((index & 1) * 4)) & 0x0f;
    }
    return (block[index >> 2] >> ((index & 3) * 2)) & 0x03;
}

static int check_type(ds4q_type type, int bits, int block_bytes) {
    float src[NROWS * NCOLS];
    uint32_t state = UINT32_C(0x12345678);
    for (int row = 0; row < NROWS; row++) {
        const float amplitude = 0.013f + 0.037f * (float)(row + 1);
        const float offset = 0.0073f * (float)((row % 9) - 4);
        for (int col = 0; col < NCOLS; col++) {
            state = state * UINT32_C(1664525) + UINT32_C(1013904223);
            const float unit = (float)(state >> 8) * (1.0f / 16777216.0f);
            src[row * NCOLS + col] =
                offset + amplitude * (2.0f * unit - 1.0f) +
                0.011f * sinf((float)(row * NCOLS + col) * 0.17f);
        }
    }

    const size_t row_bytes = ds4q_row_size(type, NCOLS);
    if (row_bytes != (size_t)block_bytes) {
        fprintf(stderr, "FAIL: type %d row bytes=%zu expected=%d\n",
                type, row_bytes, block_bytes);
        return 1;
    }
    uint8_t *encoded = calloc(NROWS, row_bytes);
    if (!encoded) return 1;
    const size_t written = ds4q_quantize_chunk(
        type, src, encoded, 0, NROWS, NCOLS, NULL);
    if (written != NROWS * row_bytes) {
        fprintf(stderr, "FAIL: type %d wrote=%zu expected=%zu\n",
                type, written, NROWS * row_bytes);
        free(encoded);
        return 1;
    }

    const int maxq = (1 << bits) - 1;
    const int scale_off = bits == 4 ? 32 : 16;
    const int bias_off = scale_off + 2;
    int mismatches = 0;
    int differs_from_old = 0;
    for (int row = 0; row < NROWS; row++) {
        const float *values = src + row * NCOLS;
        float mn = values[0], mx = values[0];
        for (int col = 1; col < NCOLS; col++) {
            if (values[col] < mn) mn = values[col];
            if (values[col] > mx) mx = values[col];
        }
        const float old_scale = mx == mn ? 0.0f : (mx - mn) / (float)maxq;
        const float old_bias = mn;
        const uint8_t *block = encoded + (size_t)row * row_bytes;
        uint16_t scale_bits, bias_bits;
        memcpy(&scale_bits, block + scale_off, 2);
        memcpy(&bias_bits, block + bias_off, 2);
        const float scale = ds4q_bf16_to_f32(scale_bits);
        const float bias = ds4q_bf16_to_f32(bias_bits);
        if (scale != bf16_effective(old_scale) ||
            bias != bf16_effective(old_bias)) {
            fprintf(stderr, "FAIL: type %d row %d stored parameter mismatch\n",
                    type, row);
            mismatches++;
            continue;
        }
        for (int col = 0; col < NCOLS; col++) {
            const int expected = scale == 0.0f ? 0 :
                clamp_round((values[col] - bias) / scale, maxq);
            const int actual = packed_code(block, col, bits);
            const int old = old_scale == 0.0f ? 0 :
                clamp_round((values[col] - old_bias) / old_scale, maxq);
            if (actual != expected) mismatches++;
            if (old != expected) differs_from_old++;
        }
    }
    free(encoded);
    if (mismatches != 0) {
        fprintf(stderr, "FAIL: type %d has %d codes not derived from stored bf16\n",
                type, mismatches);
        return 1;
    }
    if (differs_from_old == 0) {
        fprintf(stderr, "FAIL: type %d fixture does not distinguish old f32 indexing\n",
                type);
        return 1;
    }
    fprintf(stderr,
            "ok: type %d matches stored bf16 parameters (%d pre-fix codes differ)\n",
            type, differs_from_old);
    return 0;
}

int main(void) {
    int failures = 0;
    failures += check_type(DS4Q_TYPE_Q4_64A, 4, 36);
    failures += check_type(DS4Q_TYPE_Q2_64A, 2, 20);
    return failures == 0 ? 0 : 1;
}
