/* Unit tests for the GGUF MXFP4 block layout and scalar reference dot. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define QK_MXFP4 32

typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2];
} block_mxfp4;

static const float mxfp4_values[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static float e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float vec_dot_mxfp4_f32(int n, const block_mxfp4 *x, const float *y) {
    float sum = 0.0f;
    for (int ib = 0; ib < n / QK_MXFP4; ib++) {
        const float d = e8m0_to_f32(x[ib].e);
        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            const uint8_t q = x[ib].qs[j];
            sum += d * mxfp4_values[q & 0x0f] * y[ib * QK_MXFP4 + j];
            sum += d * mxfp4_values[q >> 4] * y[ib * QK_MXFP4 + j + QK_MXFP4 / 2];
        }
    }
    return sum;
}

static float reference_value(uint8_t e, uint8_t q) {
    static const int8_t twice_value[16] = {
         0,  1,  2,  3,  4,  6,  8,  12,
         0, -1, -2, -3, -4, -6, -8, -12,
    };
    const float scale = e == 0 ? ldexpf(1.0f, -127) : ldexpf(1.0f, (int)e - 127);
    return 0.5f * (float)twice_value[q] * scale;
}

static int test_block_layout(void) {
    const int ok = sizeof(block_mxfp4) == 17;
    printf("  block size: %zu (expect 17): %s\n", sizeof(block_mxfp4), ok ? "PASS" : "FAIL");
    return !ok;
}

static int test_e8m0(void) {
    const uint8_t exponents[] = {0, 1, 2, 126, 127, 128, 254};
    int ok = 1;
    for (size_t i = 0; i < sizeof(exponents); i++) {
        const uint8_t e = exponents[i];
        const float expected = e == 0 ? ldexpf(1.0f, -127) : ldexpf(1.0f, (int)e - 127);
        if (e8m0_to_f32(e) != expected) ok = 0;
    }
    printf("  E8M0 scale conversion: %s\n", ok ? "PASS" : "FAIL");
    return !ok;
}

static int test_nibble_order(void) {
    block_mxfp4 block = {.e = 127};
    float y[QK_MXFP4];
    memset(block.qs, 0xf6, sizeof(block.qs));
    for (int i = 0; i < QK_MXFP4; i++) y[i] = 1.0f;
    const float got = vec_dot_mxfp4_f32(QK_MXFP4, &block, y);
    const int ok = got == -32.0f;
    printf("  nibble order: %.1f (expect -32.0): %s\n", got, ok ? "PASS" : "FAIL");
    return !ok;
}

static int test_random_reference(void) {
    enum { N_BLOCK = 7, N = N_BLOCK * QK_MXFP4 };
    block_mxfp4 blocks[N_BLOCK];
    float y[N];
    uint32_t state = 0x12345678u;
    int ok = 1;

    for (int trial = 0; trial < 100; trial++) {
        for (int ib = 0; ib < N_BLOCK; ib++) {
            state = state * 1664525u + 1013904223u;
            blocks[ib].e = (uint8_t)(110u + state % 35u);
            for (int j = 0; j < QK_MXFP4 / 2; j++) {
                state = state * 1664525u + 1013904223u;
                blocks[ib].qs[j] = (uint8_t)(state >> 24);
            }
        }
        for (int i = 0; i < N; i++) {
            state = state * 1664525u + 1013904223u;
            y[i] = (float)(int16_t)(state >> 16) / 4096.0f;
        }

        const float got = vec_dot_mxfp4_f32(N, blocks, y);
        float expected = 0.0f;
        for (int ib = 0; ib < N_BLOCK; ib++) {
            for (int j = 0; j < QK_MXFP4 / 2; j++) {
                const uint8_t q = blocks[ib].qs[j];
                expected += reference_value(blocks[ib].e, q & 0x0f) * y[ib * QK_MXFP4 + j];
                expected += reference_value(blocks[ib].e, q >> 4) * y[ib * QK_MXFP4 + j + QK_MXFP4 / 2];
            }
        }
        const float tolerance = 2.0e-6f * fmaxf(1.0f, fabsf(expected));
        if (fabsf(got - expected) > tolerance) {
            printf("    trial %d: got %.9g expected %.9g\n", trial, got, expected);
            ok = 0;
            break;
        }
    }
    printf("  dot vs independent reference: %s\n", ok ? "PASS" : "FAIL");
    return !ok;
}

int main(void) {
    int failures = 0;
    printf("MXFP4 unit tests:\n");
    failures += test_block_layout();
    failures += test_e8m0();
    failures += test_nibble_order();
    failures += test_random_reference();
    printf("\n%d/4 tests passed\n", 4 - failures);
    return failures != 0;
}
