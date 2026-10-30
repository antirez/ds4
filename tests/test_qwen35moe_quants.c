/* Check quantized dot products against decoded weights and activations. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if defined(__F16C__)
#include <immintrin.h>
#endif
#define GGML_COMMON_DECL_C
#include "cpu/ggml/ggml-common.h"
#include "cpu/ggml/qwen35moe-quants.h"

int main(void) {
#if defined(__F16C__)
    /* Exhaust every finite half pattern against hardware conversion before
     * quantized kernels use the once-initialized lookup table. */
    for (uint32_t h = 0; h < (1u << 16); h++) {
        if ((h & 0x7c00u) == 0x7c00u) continue;
        float actual = ds4_qwen35moe_half_lookup((uint16_t)h);
        float expected = _cvtsh_ss((uint16_t)h);
        assert(memcmp(&actual, &expected, sizeof(float)) == 0);
    }
#endif
    const int types[] = {8, 10, 12, 13, 14, 16, 17};
    const size_t sizes[] = {sizeof(block_q8_0),  sizeof(block_q2_K), sizeof(block_q4_K),
                            sizeof(block_q5_K),  sizeof(block_q6_K), sizeof(block_iq2_xxs),
                            sizeof(block_iq2_xs)};
    float input[1024], decoded[1024], activation[1024];
    uint64_t data[1024] = {0}, quantized[1024] = {0};
    for (int i = 0; i < 1024; i++)
        input[i] = sinf((float)i * 0.17f) * 0.75f;
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        int block = types[t] == 8 ? 32 : 256;
        size_t bytes = 1024 / (size_t)block * sizes[t];
        unsigned char *raw = (unsigned char *)data;
        for (size_t i = 0; i < bytes; i++)
            raw[i] = (unsigned char)(i * 29 + 7);
        for (int b = 0; b < 1024 / block; b++) {
            unsigned char *p = raw + (size_t)b * sizes[t];
            uint16_t d = 0x2e00, dm = 0x2900;
            size_t off = types[t] == 10 ? 80 : types[t] == 14 ? 208 : 0;
            memcpy(p + off, &d, 2);
            if (types[t] == 10)
                memcpy(p + 82, &dm, 2);
            if (types[t] == 12 || types[t] == 13)
                memcpy(p + 2, &dm, 2);
        }
        ds4_qwen35moe_dequant(types[t], data, decoded, 1024);
        ds4_qwen35moe_quantize(types[t], input, quantized, 1024);
        if (types[t] == 8) {
            ds4_qwen35moe_dequant(8, quantized, activation, 1024);
        } else {
            block_q8_K *q = (block_q8_K *)quantized;
            for (int i = 0; i < 1024; i++)
                activation[i] = q[i / 256].d * q[i / 256].qs[i % 256];
        }
        double expected = 0;
        for (int i = 0; i < 1024; i++)
            expected += (double)decoded[i] * activation[i];
        float actual = ds4_qwen35moe_dot(types[t], 1024, data, quantized);
        assert(isfinite(actual));
        assert(fabs((double)actual - expected) < 1e-4 * (1 + fabs(expected)));
    }
    memset(input, 0, sizeof(input));
    ds4_qwen35moe_quantize(16, input, quantized, 1024);
    block_q8_K *zero = (block_q8_K *)quantized;
    for (int i = 0; i < 4; i++)
        assert(zero[i].d == 0);
    /* The flash-attention GEMM must accumulate in K order, including edge
     * rows and columns, because a later quantized expert can magnify a ULP. */
    float a[64 * 256], b[256 * 64], c[64 * 64], expected_c[64 * 64];
    for (int i = 0; i < 64 * 256; i++) a[i] = sinf(i * 0.013f);
    for (int i = 0; i < 256 * 64; i++) b[i] = cosf(i * 0.019f);
    for (int m = 1; m <= 64; m += 63)
        for (int n = 9; n <= 64; n += 55) {
            memset(c, 0, sizeof(c));
            memset(expected_c, 0, sizeof(expected_c));
            ds4_qwen35moe_gemm_f32(c, a, b, m, 256, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < 256; k++)
                        expected_c[i * n + j] = fmaf(a[i * 256 + k], b[k * n + j],
                                                       expected_c[i * n + j]);
            assert(memcmp(c, expected_c, (size_t)m * n * sizeof(float)) == 0);
        }
    float fw[4 * 32], fx[2 * 32], fo[2 * 4];
    for (int i = 0; i < 4 * 32; i++) fw[i] = cosf(i * 0.031f);
    for (int i = 0; i < 2 * 32; i++) fx[i] = sinf(i * 0.027f);
    for (int rows = 1; rows <= 4; rows++)
        for (int columns = 1; columns <= 2; columns++) {
            ds4_qwen35moe_sgemm_float_tile4x2(32, fw, 32, fx, 32, fo, 4, rows, columns);
            for (int col = 0; col < columns; col++)
                for (int row = 0; row < rows; row++) {
                    float expected = ds4_qwen35moe_sgemm_float_dot(32, fw + row * 32, fx + col * 32);
                    assert(memcmp(fo + col * 4 + row, &expected, sizeof(float)) == 0);
                }
        }
#if defined(__F16C__)
    unsigned short half_acc[256], half_val[256], half_expect[256];
    for (int i = 0; i < 256; i++) {
        half_acc[i] = (unsigned short)_cvtss_sh(sinf(i * 0.19f), 0);
        half_val[i] = (unsigned short)_cvtss_sh(cosf(i * 0.13f), 0);
        half_expect[i] = half_acc[i];
    }
    ds4_qwen35moe_f16_scale(256, half_acc, 0.731f);
    for (int i = 0; i < 256; i++)
        half_expect[i] = (unsigned short)_cvtss_sh(_cvtsh_ss(half_expect[i]) * 0.731f, 0);
    assert(memcmp(half_acc, half_expect, sizeof(half_acc)) == 0);
    ds4_qwen35moe_f16_mad(256, half_acc, half_val, -0.413f);
    for (int i = 0; i < 256; i++)
        half_expect[i] = (unsigned short)_cvtss_sh(
            fmaf(_cvtsh_ss(half_val[i]), -0.413f, _cvtsh_ss(half_expect[i])), 0);
    assert(memcmp(half_acc, half_expect, sizeof(half_acc)) == 0);
#endif
    /* Equal-key order is the observable behavior of the pinned GCC argsort.
     * These oracle results cover all-equal, boundary-tie, and dense-tie rows. */
    float route[256];
    int chosen[8];
    const int expected_equal[8] = {176, 161, 162, 163, 164, 165, 166, 167};
    memset(route, 0, sizeof(route));
    ds4_qwen35moe_router_top8(route, chosen);
    assert(memcmp(chosen, expected_equal, sizeof(chosen)) == 0);
    for (int i = 0; i < 256; i++) route[i] = -(float)(i + 1);
    route[88] = 10; route[160] = 9; route[80] = 8; route[28] = 7;
    route[35] = 6; route[254] = 5; route[59] = route[235] = 4;
    const int expected_boundary[8] = {88, 160, 80, 28, 35, 254, 235, 59};
    ds4_qwen35moe_router_top8(route, chosen);
    assert(memcmp(chosen, expected_boundary, sizeof(chosen)) == 0);
    for (int i = 0; i < 256; i++) route[i] = (float)((i * 37) % 17) / 100.0f;
    const int expected_dense[8] = {198, 45, 62, 79, 232, 96, 28, 113};
    ds4_qwen35moe_router_top8(route, chosen);
    assert(memcmp(chosen, expected_dense, sizeof(chosen)) == 0);
    puts("qwen35moe CPU primitives: quantized block layouts, decoded-dot agreement, and zero activations: PASS");
    return 0;
}
