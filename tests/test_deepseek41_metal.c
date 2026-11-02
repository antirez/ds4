#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t seed = 7919;
static float random_value(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return ((int)(seed % 65537) - 32768) / 8192.0f;
}

static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, 4);
    return value;
}

static float nearest(float value, int fp4) {
    const float fp4_values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float best_value = 0, best_error = INFINITY;
    int best = 0;
    for (int i = 0; i < (fp4 ? 8 : 127); i++) {
        const float v = fp4 ? fp4_values[i] : i < 8 ? ldexpf(i, -9) :
                        ldexpf(1.0f + (i & 7) / 8.0f, (i >> 3) - 7);
        const float error = fabsf(fabsf(value) - v);
        if (error < best_error || (error == best_error && !(i & 1) && (best & 1))) {
            best = i; best_value = v; best_error = error;
        }
    }
    return copysignf(best_value, value);
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_managed(bytes);
    if (t && data && !ds4_gpu_tensor_write(t, 0, data, bytes)) {
        ds4_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

static int check_router(void) {
    enum { ROWS = 129, MAX_EXPERTS = 384, USED = 6 };
    const size_t bias_bytes = (size_t)getpagesize();
    float *bias = NULL;
    CHECK(bias_bytes >= MAX_EXPERTS * sizeof(float));
    CHECK(posix_memalign((void **)&bias, bias_bytes, bias_bytes) == 0);
    memset(bias, 0, bias_bytes);
    float *logits = malloc(ROWS * MAX_EXPERTS * sizeof(float));
    int32_t tokens[ROWS] = {0};
    CHECK(bias && logits);
    for (unsigned e = 0; e < MAX_EXPERTS; e++) bias[e] = (e % 7) * 0.125f;
    CHECK(ds4_gpu_set_model_map(bias, bias_bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *p = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *ids = upload(NULL, ROWS * USED * sizeof(int32_t));
    ds4_gpu_tensor *weights = upload(NULL, ROWS * USED * sizeof(float));
    ds4_gpu_tensor *tok = upload(tokens, sizeof(tokens));
    CHECK(x && p && ids && weights && tok);
    const unsigned counts[] = {1, 3, 17, ROWS};
    for (unsigned n = 128; n <= MAX_EXPERTS; n += 128) {
        const unsigned USED = n == 128 ? 3 : 6;
        for (unsigned mode = 0; mode < 3; mode++) {
            for (unsigned i = 0; i < ROWS * n; i++) logits[i] = mode == 1 ? 0 :
                mode == 2 ? (i % 2 ? -80 : 40) : random_value() * 8;
            CHECK(ds4_gpu_tensor_write(x, 0, logits, ROWS * n * sizeof(float)));
            for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const unsigned rows = counts[ci];
                CHECK(ds4_gpu_router_select_batch_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, 0, true, false,
                    x, tok, n, USED, 1.5f, rows));
                CHECK(ds4_gpu_synchronize());
                const int32_t *selected = ds4_gpu_tensor_contents(ids);
                const float *actual = ds4_gpu_tensor_contents(weights);
                const float *probs = ds4_gpu_tensor_contents(p);
                CHECK(selected && actual && probs);
                for (unsigned t = 0; t < rows; t++) {
                    double ref[MAX_EXPERTS];
                    int best[USED];
                    for (unsigned k = 0; k < USED; k++) best[k] = -1;
                    for (unsigned e = 0; e < n; e++) {
                        const double v = logits[t * n + e];
                        ref[e] = sqrt(v > 20 ? v : v < -20 ? exp(v) : log1p(exp(v)));
                        if (!(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e])))
                            fprintf(stderr, "router n=%u mode=%u rows=%u row=%u expert=%u logit=%.9g actual=%.9g ref=%.9g\n",
                                    n, mode, rows, t, e, v, probs[t * n + e], ref[e]);
                        CHECK(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e]));
                        const double score = ref[e] + bias[e];
                        for (unsigned k = 0; k < USED; k++) {
                            if (best[k] < 0 || score > ref[best[k]] + bias[best[k]]) {
                                for (unsigned j = USED - 1; j > k; j--) best[j] = best[j - 1];
                                best[k] = (int)e;
                                break;
                            }
                        }
                    }
                    double sum = 0;
                    for (unsigned k = 0; k < USED; k++) {
                        const int id = selected[t * USED + k];
                        CHECK(id >= 0 && id < (int)n);
                        for (unsigned j = 0; j < k; j++) CHECK(id != selected[t * USED + j]);
#ifdef __APPLE__
                        /* Bitonic selection does not promise CUDA's lower-ID
                         * tie break. Still require the correct top-k scores. */
                        CHECK(fabs((ref[id] + bias[id]) - (ref[best[k]] + bias[best[k]])) < 3e-6);
#else
                        CHECK(id == best[k]);
#endif
                        sum += ref[id];
                    }
                    for (unsigned k = 0; k < USED; k++) {
                        CHECK(fabs(actual[t * USED + k] - 1.5 * ref[selected[t * USED + k]] /
                              fmax(sum, 0x1p-14)) < 3e-6);
                    }
                }
                int32_t first_ids[USED]; float first_weights[USED];
                memcpy(first_ids, selected, sizeof(first_ids));
                memcpy(first_weights, actual, sizeof(first_weights));
                CHECK(ds4_gpu_router_select_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, n, USED, 1.5f,
                    0, 0, true, false, x));
                CHECK(ds4_gpu_synchronize());
                const int32_t *scalar_ids = ds4_gpu_tensor_contents(ids);
                const float *scalar_weights = ds4_gpu_tensor_contents(weights);
#ifdef __APPLE__
                for (unsigned k = 0; k < USED; k++) {
                    CHECK(scalar_ids[k] >= 0 && scalar_ids[k] < (int32_t)n);
                    CHECK(probs[first_ids[k]] + bias[first_ids[k]] ==
                          probs[scalar_ids[k]] + bias[scalar_ids[k]]);
                    CHECK(fabsf(first_weights[k] - scalar_weights[k]) < 3e-6f);
                }
#else
                CHECK(!memcmp(first_ids, scalar_ids, sizeof(first_ids)));
                CHECK(!memcmp(first_weights, scalar_weights, sizeof(first_weights)));
#endif
            }
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(p); ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(tok);
    ds4_gpu_cleanup();
    free(bias); free(logits);
    CHECK(ds4_gpu_init());
    fprintf(stderr, "256/384-expert routing, ties/extremes and scalar/batch oracle: PASS\n");
    return 1;
}

static int check_quantization(void) {
    enum { WIDTH = 512, ROWS = 33, N = WIDTH * ROWS };
    float *source = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    CHECK(source && actual);
    ds4_gpu_tensor *t = upload(NULL, N * sizeof(float));
    CHECK(t);
    for (int mode = 0; mode < 4; mode++) {
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i++) source[i] = random_value() * (1u << ((i / block) % 4));
        for (int i = 0; i < WIDTH; i++) source[i] = copysignf(0.0f, i & 1 ? -1.0f : 1.0f);
        CHECK(ds4_gpu_tensor_write(t, 0, source, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_quantize(t, WIDTH, ROWS, (ds4_v41_activation_format)mode));
        CHECK(ds4_gpu_tensor_read(t, 0, actual, N * sizeof(float)));
        for (int start = 0; start < N; start += block) {
            float amax = 0, scale = 1;
            for (int i = 0; i < block; i++) amax = fmaxf(amax, fabsf(bf16(source[start + i])));
            if (mode == DS4_V41_FP8_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f))));
            if (mode == DS4_V41_FP4_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f))));
            if (mode == DS4_V41_FP4_E4M3) scale = nearest(fmaxf(amax, 6.0f / 512.0f) / 6.0f, 0);
            for (int i = 0; i < block; i++) {
                float expected = bf16(source[start + i]);
                if (mode) expected = bf16(nearest(expected / scale, mode != 1) * scale);
                if (memcmp(&expected, actual + start + i, 4)) {
                    fprintf(stderr, "quantization mode=%d index=%d: %.9g != %.9g\n",
                            mode, start + i, actual[start + i], expected);
                    return 0;
                }
            }
        }
    }
    CHECK(ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_FP8_E8M0));
    CHECK(!ds4_gpu_dsv41_quantize(t, UINT32_MAX, UINT32_MAX, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 32, 1, (ds4_v41_activation_format)4));
    ds4_gpu_tensor_free(t);
    free(source); free(actual);
    fprintf(stderr, "V4.1 BF16/FP8/FP4 round trips: exact\n");
    return 1;
}

static double monotonic_seconds(void);

static uint32_t bf16_bits(uint32_t bits) {
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return bits & 0xffff0000u;
}

static uint32_t bf16_input(size_t i) {
    const uint32_t low[] = {0, 0x7fffu, 0x8000u, 0x8001u, 0xffffu};
    if (i < 65536u * 5u) return ((uint32_t)(i / 5u) << 16u) | low[i % 5u];
    return (uint32_t)i * 2654435761u;
}

static int check_bf16_linear(void) {
    const struct { uint32_t width, rows, offset; } shapes[] = {
        {1, 1, 0}, {3, 1, 0}, {5, 1, 0}, {24, 1, 0}, {31, 3, 0},
        {32, 7, 1}, {129, 3, 4}, {1023, 1, 0}, {1024, 1, 0}, {1025, 1, 0},
        {65536u * 5u, 1, 0}, {5120, 1, 0}, {20480, 512, 0},
        {32768, 4096, 0}, {32768, 8192, 0}
    };
    const unsigned modes[] = {0, 1, 1, 0};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        const size_t count = (size_t)shapes[s].width * shapes[s].rows;
        const size_t offset = shapes[s].offset;
        ds4_gpu_tensor *storage = upload(NULL, (count + 8) * sizeof(uint32_t));
        CHECK(storage);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(storage,
            offset * sizeof(uint32_t), count * sizeof(uint32_t));
        CHECK(view);
        uint32_t *bits = ds4_gpu_tensor_contents(storage);
        CHECK(bits);
        for (unsigned pass = 0; pass < 4; pass++) {
            const unsigned mode = modes[pass];
            if (mode) unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
            else setenv("DS4_METAL_DISABLE_V41_LINEAR_BF16", "1", 1);
            for (size_t i = 0; i < count + 8; i++) bits[i] = 0x12345678;
            for (size_t i = 0; i < count; i++) bits[offset + i] = bf16_input(i);
            double elapsed = 0;
            for (unsigned repeat = 0; repeat < 5; repeat++) {
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_dsv41_quantize(view, shapes[s].width, shapes[s].rows, DS4_V41_BF16));
                CHECK(ds4_gpu_synchronize());
                if (repeat) elapsed += (monotonic_seconds() - begin) * 250;
            }
            for (size_t i = 0; i < count; i++) CHECK(bits[offset + i] == bf16_bits(bf16_input(i)));
            for (size_t i = 0; i < offset; i++) CHECK(bits[i] == 0x12345678);
            for (size_t i = offset + count; i < count + 8; i++) CHECK(bits[i] == 0x12345678);
            fprintf(stderr, "BF16 width=%u rows=%u offset=%zu mode=%u pass=%u %.3f ms: exact\n",
                shapes[s].width, shapes[s].rows, offset, mode, pass, elapsed);
        }
        ds4_gpu_tensor_free(view);
        ds4_gpu_tensor_free(storage);
    }
    unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
    return 1;
}

#ifdef __APPLE__
static int check_hc_scaled(void) {
    enum { WIDTH = 20480, OUT = 24, ROWS = 8192 };
    const size_t weight_bytes = WIDTH * OUT * sizeof(_Float16);
    const size_t page = (size_t)getpagesize();
    const size_t mapped_bytes = (weight_bytes + page - 1) / page * page;
    void *model = NULL;
    CHECK(!posix_memalign(&model, page, mapped_bytes));
    _Float16 *weights = model;
    for (size_t i = 0; i < WIDTH * OUT; i++) weights[i] = (_Float16)(random_value() / 64);
    CHECK(ds4_gpu_set_model_map(model, mapped_bytes));
    ds4_gpu_tensor *x = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *norm = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *scales = upload(NULL, (ROWS + 1) * 4);
    ds4_gpu_tensor *out = upload(NULL, (OUT * ROWS + 1) * 4);
    ds4_gpu_tensor *ref = upload(NULL, (OUT * ROWS + 1) * 4);
    CHECK(x && norm && scales && out && ref);
    float *input = ds4_gpu_tensor_contents(x), *y = ds4_gpu_tensor_contents(out);
    float *expected = ds4_gpu_tensor_contents(ref), *s = ds4_gpu_tensor_contents(scales);
    for (size_t i = 0; i < (size_t)WIDTH * ROWS; i++)
        input[i] = bf16(random_value() * ((i / WIDTH) % 7 ? 1 : 0x1p-16f));
    const unsigned rows[] = {9, 31, 32, 33, 127, 512, 2048, 4096, 8192};
    for (size_t shape = 0; shape < sizeof(rows) / sizeof(*rows); shape++) {
        const uint32_t n = rows[shape];
        double elapsed[2] = {0};
        for (unsigned repeat = 0; repeat < 7; repeat++) {
            y[n * OUT] = expected[n * OUT] = s[n] = 12345;
            for (unsigned j = 0; j < 2; j++) {
                const unsigned mode = j ^ (repeat & 1);
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_begin_commands());
                if (mode) {
                    CHECK(ds4_gpu_hc_rms_scale_project_f16_tensor(out, n >= 512u ? scales : norm,
                        model, mapped_bytes, 0, WIDTH, OUT, x, n, 1e-6f));
                } else {
                    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(norm, x, WIDTH, n, 1e-6f));
                    CHECK(ds4_gpu_matmul_f16_tensor(ref, model, mapped_bytes, 0,
                        WIDTH, OUT, norm, n));
                }
                CHECK(ds4_gpu_end_commands());
                if (repeat) elapsed[mode] += (monotonic_seconds() - begin) * (1000.0 / 6);
            }
            CHECK(!memcmp(y, expected, n * OUT * 4));
            CHECK(y[n * OUT] == 12345 && expected[n * OUT] == 12345 && s[n] == 12345);
        }
        fprintf(stderr, "HC scaled rows=%u: exact %.3f -> %.3f ms\n", n, elapsed[0], elapsed[1]);
    }
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(scales);
    ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup(); free(model);
    return 1;
}

#endif

static int check_engram(void) {
    enum { D = 5120, ROWS = 5, N = ROWS * 4 * D };
    float *x = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    float *kv = malloc(ROWS * 5 * D * sizeof(float));
    float *qw = malloc(4 * D * sizeof(float)), *kw = malloc(4 * D * sizeof(float));
    uint8_t mask[] = {1, 1, 0, 1, 0};
    CHECK(x && actual && kv && qw && kw);
    for (int i = 0; i < N; i++) x[i] = bf16(random_value());
    for (int i = 0; i < ROWS * 5 * D; i++) kv[i] = bf16(random_value());
    for (int i = 0; i < 4 * D; i++) { qw[i] = random_value(); kw[i] = random_value(); }
    memset(x, 0, D * sizeof(float));
    memset(kv + D, 0, D * sizeof(float));
    ds4_gpu_tensor *xt = upload(x, N * sizeof(float));
    ds4_gpu_tensor *kt = upload(kv, ROWS * 5 * D * sizeof(float));
    ds4_gpu_tensor *qwt = upload(qw, 4 * D * sizeof(float));
    ds4_gpu_tensor *kwt = upload(kw, 4 * D * sizeof(float));
    ds4_gpu_tensor *mt = upload(mask, sizeof(mask));
    CHECK(xt && kt && qwt && kwt && mt);
    size_t rounded_differently = 0;
    double error2 = 0, norm2 = 0;
    for (int masked = 0; masked < 2; masked++) {
        CHECK(ds4_gpu_tensor_write(xt, 0, x, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, masked ? mt : NULL, D, ROWS, 1e-20f));
        CHECK(ds4_gpu_tensor_read(xt, 0, actual, N * sizeof(float)));
        for (int row = 0; row < ROWS; row++) for (int h = 0; h < 4; h++) {
            double dot = 0, h2 = 0, k2 = 0;
            for (int i = 0; i < D; i++) {
                const double a = x[(row * 4 + h) * D + i], b = kv[(row * 5 + h) * D + i];
                h2 += a * a; k2 += b * b;
                dot += a * (float)(qw[h * D + i] * kw[h * D + i]) * b;
            }
            dot /= sqrt(h2 / D + 1e-20) * sqrt(k2 / D + 1e-20) * sqrt(D);
            double gate = 1 / (1 + exp(-copysign(sqrt(fmax(fabs(dot), 1e-6)), dot)));
            if (masked && !mask[row]) gate = 0;
            for (int i = 0; i < D; i++) {
                const int off = (row * 4 + h) * D + i;
                const float expected = bf16(x[off] + (float)gate * kv[(row * 5 + 4) * D + i]);
                const double error = actual[off] - expected;
                CHECK(isfinite(actual[off]));
                CHECK(fabs(error) <= fmax(1e-6, fabs(expected) / 128));
                if (masked && !mask[row]) CHECK(actual[off] == x[off]);
                rounded_differently += actual[off] != expected;
                error2 += error * error; norm2 += (double)expected * expected;
            }
        }
    }
    CHECK(rounded_differently < N / 1000);
    CHECK(sqrt(error2 / norm2) < 1e-4);
    fprintf(stderr, "V4.1 Engram gate: %zu BF16 boundary differences, relative RMS %.8g\n",
            rounded_differently, sqrt(error2 / norm2));
#ifdef __APPLE__
    {   /* the decode step's gate over 8 SIMD groups (unmasked, width 5120, <= 16 rows) gives the 32-lane bits */
        float *spread = malloc(N * sizeof(float));
        CHECK(spread);
        ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
        for (int off = 0; off < 2; off++) {
            const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_engram_add_tg256");
            ds4_gpu_test_set_v41_off(off ? DS4_GPU_V41_SPLIT_REDUCE : 0);
            const int ok = ds4_gpu_tensor_write(xt, 0, x, N * sizeof(float)) &&
                ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, NULL, D, ROWS, 1e-20f);
            ds4_gpu_test_set_v41_off(0);
            CHECK(ok && ds4_gpu_tensor_read(xt, 0, off ? actual : spread, N * sizeof(float)));
            CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_engram_add_tg256") == runs + !off);
        }
        CHECK(!memcmp(actual, spread, N * sizeof(float)));
        free(spread);
        fprintf(stderr, "V4.1 Engram gate over 8 SIMD groups == 32 lanes, bitwise PASS\n");
    }
#endif
    CHECK(!ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, mt, D, ROWS + 1, 1e-20f));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qwt);
    ds4_gpu_tensor_free(kwt); ds4_gpu_tensor_free(mt);
    free(x); free(actual); free(kv); free(qw); free(kw);
    return 1;
}

static int check_rope_stride(void) {
    enum { WIDTH = 512, HEADS = 2, ROWS = 129, N = WIDTH * HEADS * ROWS };
    ds4_gpu_tensor *a = upload(NULL, (N + 1u) * sizeof(float));
    ds4_gpu_tensor *b = upload(NULL, (N + 1u) * sizeof(float));
    CHECK(a && b);
    float *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    const uint32_t starts[] = {0, 126, 32766, 1048318};
    for (uint32_t kind = 0; kind < 2; kind++) for (uint32_t inverse = 0; inverse < 2; inverse++) {
        for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
            for (size_t j = 0; j < N; j++) x[j] = y[j] = bf16(random_value());
            x[N] = y[N] = 12345;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, starts[i], 2, kind, inverse));
            for (uint32_t row = 0; row < ROWS; row++) {
                ds4_gpu_tensor *v = ds4_gpu_tensor_view(b, (uint64_t)row * WIDTH * HEADS * 4u,
                                                       WIDTH * HEADS * 4u);
                CHECK(v && ds4_gpu_dsv41_rope(v, WIDTH, HEADS, 1, starts[i] + row * 2u, kind, inverse));
                ds4_gpu_tensor_free(v);
            }
            CHECK(ds4_gpu_end_commands());
            CHECK(!memcmp(x, y, (N + 1u) * sizeof(float)) && x[N] == 12345);
        }
    }
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, 0, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 1048320, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS + 1, 0, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, UINT32_MAX, true, false));
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(a);
    fprintf(stderr, "V4.1 strided RoPE: exact row parity through 1M positions, guards PASS\n");
    return 1;
}

#ifdef __APPLE__
/* The fused RoPE, quantize and store kernel against the three separate steps. */
static int check_rope_quantize(void) {
    enum { WIDTH = 512, TRIALS = 300 };
    ds4_gpu_tensor *a = upload(NULL, WIDTH * sizeof(float));
    ds4_gpu_tensor *b = upload(NULL, WIDTH * sizeof(float));
    ds4_gpu_tensor *c = upload(NULL, (WIDTH + 1u) * sizeof(float));
    CHECK(a && b && c);
    float *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    float *z = ds4_gpu_tensor_contents(c);
    for (uint32_t t = 0; t < TRIALS; t++) {
        const ds4_v41_activation_format format = (ds4_v41_activation_format)(1 + t % 3);
        const uint32_t width = format == DS4_V41_FP4_E8M0 ? 128 : WIDTH;
        const bool compressed = (t / 3) & 1;
        const uint32_t pos = (uint32_t)(fabsf(random_value()) * 15000);
        for (uint32_t i = 0; i < width; i++) x[i] = y[i] = ldexpf(random_value(), (int)(t % 11) - 5);
        z[width] = 12345;
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_rope(a, width, 1, 1, pos, compressed, false));
        CHECK(ds4_gpu_dsv41_quantize(a, width, 1, format));
        CHECK(ds4_gpu_dsv41_rope_quantize(b, c, 0, width, pos, compressed, format));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(x, z, width * sizeof(float)) && z[width] == 12345);
    }
    ds4_gpu_tensor_free(c); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(a);
    fprintf(stderr, "V4.1 fused RoPE quantize: bitwise match with the separate steps PASS\n");
    return 1;
}
#else
static int check_rope_quantize(void) { return 1; }
#endif

#ifdef __APPLE__
/* Single-Mac decode's deferred hyper-connection weights against the original order, bitwise, on random weights and
 * fresh inputs per draw. Attention: the collapse, then the weights, query and KV row in one dispatch, against the
 * block input, the roped query and the KV row's rope and store. FFN: the collapse, then the block input into scratch,
 * against the block input. With DS4_GPU_V41_DEFERRED_HC off the collapse declines. */
static int check_deferred_hc(void) {
    const uint64_t qo = 0, qb = 32768ull * 40 * 34, ho = qo + qb, hb = 24ull * 20480 * 2;
    const uint64_t so = ho + hb, bo = so + 64, no = bo + 128, wb = no + 5120 * 4;
    char *model = NULL;
    CHECK(posix_memalign((void **)&model, getpagesize(), wb) == 0);
    for (uint64_t i = 0; i < qb / 34; i++) {
        uint8_t *block = (uint8_t *)model + qo + i * 34;
        const uint16_t d = (uint16_t)(0x1c00 + (int)(fabsf(random_value()) * 128));
        memcpy(block, &d, 2);
        for (int j = 0; j < 32; j++) block[2 + j] = (uint8_t)(int8_t)(random_value() * 31);
    }
    uint16_t *fn = (uint16_t *)(model + ho);
    for (uint64_t i = 0; i < hb / 2; i++)
        fn[i] = (uint16_t)((random_value() < 0 ? 0x8000 : 0) | (0x2000 + (int)(fabsf(random_value()) * 1536)));
    float *scale = (float *)(model + so), *base = (float *)(model + bo), *nw = (float *)(model + no);
    scale[0] = 0.04f; scale[1] = 0.03f; scale[2] = 0.02f;
    for (int i = 0; i < 24; i++) base[i] = random_value() / 32;
    for (int i = 0; i < 5120; i++) nw[i] = 1.0f + random_value() / 32;
    CHECK(ds4_gpu_set_model_map(model, wb));
    ds4_gpu_tensor *res = upload(NULL, 20480 * 4), *pre = upload(NULL, 16), *qr = upload(NULL, 1280 * 4);
    ds4_gpu_tensor *kv = upload(NULL, 512 * 4), *scratch = upload(NULL, 5120 * 4);
    enum { X, NORM, MIX, SPLIT, Q, KV_ROW, OUTS };
    const uint32_t widths[OUTS] = {5120, 5120, 24, 24, 32768, 512};
    ds4_gpu_tensor *out[2][OUTS];
    CHECK(res && pre && qr && kv && scratch);
    for (int a = 0; a < 2; a++)
        for (int i = 0; i < OUTS; i++) CHECK((out[a][i] = upload(NULL, widths[i] * 4)) != NULL);
    float *r = ds4_gpu_tensor_contents(res), *p = ds4_gpu_tensor_contents(pre);
    float *x = ds4_gpu_tensor_contents(qr), *k = ds4_gpu_tensor_contents(kv);
    const float eps = 1e-6f;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (uint32_t draw = 0; draw < 6; draw++) {
        const int ffn = draw & 1;
        const uint32_t pos = draw * 104729u % 1048576u;
        const bool compressed = (draw >> 1) & 1;
        for (int i = 0; i < 20480; i++) r[i] = random_value() * (draw % 3 == 2 ? 16.0f : 1.0f);
        for (int i = 0; i < 4; i++) p[i] = random_value() / 4;
        for (int i = 0; i < 1280; i++) x[i] = random_value();
        for (int i = 0; i < 512; i++) k[i] = random_value();
        for (int a = 0; a < 2; a++)
            for (int i = 0; i < OUTS; i++) memset(ds4_gpu_tensor_contents(out[a][i]), 0xff, widths[i] * 4);
        ds4_gpu_tensor **o = out[0], **d = out[1];
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_hc_block_input_rows(o[MIX], o[X], o[NORM], o[SPLIT], res, pre, model, wb, ho, so, bo, no,
                                                20480, 24, 5120, 4, 20, eps, eps, 1, 4));
        if (!ffn) {
            CHECK(ds4_gpu_dsv41_project_q(o[Q], model, wb, qo, 1280, 32768, qr, pos, compressed));
            CHECK(ds4_gpu_dsv41_rope_quantize(kv, o[KV_ROW], 0, 512, pos, compressed, DS4_V41_FP8_E8M0));
        }
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        const uint64_t collapses = ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse");
        const uint64_t queries = ds4_gpu_test_kernel_runs("kernel_dsv41_query_deferred_hc");
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_deferred_collapse(d[X], d[NORM], res, pre, model, wb, no, eps));
        if (ffn) {
            CHECK(ds4_gpu_dsv41_hc_block_input_rows(d[MIX], d[X], scratch, d[SPLIT], res, pre, model, wb, ho, so, bo,
                                                    no, 20480, 24, 5120, 4, 20, eps, eps, 1, 4));
        } else {
            CHECK(ds4_gpu_dsv41_query_deferred_hc(d[Q], d[MIX], d[SPLIT], qr, res, model, wb, qo, ho, so, bo, pos,
                                                  compressed, 20, eps, eps, d[X], d[NORM], pre, no, kv, d[KV_ROW], 0));
        }
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse") == collapses + 1);
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_query_deferred_hc") == queries + !ffn);
        for (int i = 0; i < (ffn ? Q : OUTS); i++)
            CHECK(!memcmp(ds4_gpu_tensor_contents(o[i]), ds4_gpu_tensor_contents(d[i]), widths[i] * 4));
        if (ffn) CHECK(!memcmp(ds4_gpu_tensor_contents(o[NORM]), ds4_gpu_tensor_contents(scratch), 5120 * 4));
    }
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_DEFERRED_HC);
    const int declined = !ds4_gpu_dsv41_deferred_collapse(out[1][X], out[1][NORM], res, pre, model, wb, no, eps);
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    CHECK(declined);
    for (int a = 0; a < 2; a++)
        for (int i = 0; i < OUTS; i++) ds4_gpu_tensor_free(out[a][i]);
    ds4_gpu_tensor_free(scratch); ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(qr);
    ds4_gpu_tensor_free(pre); ds4_gpu_tensor_free(res);
    free(model);
    fprintf(stderr, "V4.1 deferred HC weights (attention and FFN) == original order, bitwise PASS\n");
    return 1;
}
#else
static int check_deferred_hc(void) { return 1; }
#endif

static int check_pool(void) {
    enum { D = 512, ROWS = 257, PAIRS = ROWS / 2 };
    float *kv = malloc(ROWS * D * sizeof(float)), *scores = malloc(ROWS * D * sizeof(float));
    float *got = malloc(PAIRS * D * sizeof(float)), *reference = malloc(PAIRS * D * sizeof(float));
    CHECK(kv && scores && got && reference);
    for (int i = 0; i < ROWS * D; i++) { kv[i] = random_value(); scores[i] = random_value() * 25; }
    ds4_gpu_tensor *kt = upload(kv, ROWS * D * sizeof(float));
    ds4_gpu_tensor *st = upload(scores, ROWS * D * sizeof(float));
    ds4_gpu_tensor *pk = upload(NULL, D * sizeof(float)), *ps = upload(NULL, D * sizeof(float));
    ds4_gpu_tensor *out = upload(NULL, PAIRS * D * sizeof(float));
    CHECK(kt && st && pk && ps && out);
    const uint32_t chunks[] = {257, 1, 2, 3, 17, 127, 128, 129};
    for (size_t c = 0; c < sizeof(chunks) / sizeof(*chunks); c++) {
        CHECK(ds4_gpu_tensor_fill_f32(pk, NAN, D));
        CHECK(ds4_gpu_tensor_fill_f32(ps, NAN, D));
        CHECK(ds4_gpu_begin_commands());
        for (uint32_t start = 0; start < ROWS;) {
            uint32_t n = chunks[c] < ROWS - start ? chunks[c] : ROWS - start;
            uint32_t pairs = (n + (start & 1u)) / 2;
            ds4_gpu_tensor *k = ds4_gpu_tensor_view(kt, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *s = ds4_gpu_tensor_view(st, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *o = pairs ? ds4_gpu_tensor_view(out, (uint64_t)(start / 2) * D * 4,
                                                         (uint64_t)pairs * D * 4) : NULL;
            CHECK(k && s && (!pairs || o));
            CHECK(ds4_gpu_dsv41_pool2(o, k, s, pk, ps, D, n, start));
            ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(o);
            start += n;
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(out, 0, got, PAIRS * D * sizeof(float)));
        if (!c) {
            memcpy(reference, got, PAIRS * D * sizeof(float));
            for (int p = 0; p < PAIRS; p++) for (int i = 0; i < D; i++) {
                const int a = 2 * p * D + i, b = a + D;
                const double gate = 1 / (1 + exp((double)scores[b] - scores[a]));
                const float expected = bf16((float)(kv[a] * gate + kv[b] * (1 - gate)));
                CHECK(isfinite(got[p * D + i]));
                CHECK(fabsf(got[p * D + i] - expected) <= fmaxf(1e-6f, fabsf(expected) / 128));
            }
        } else CHECK(!memcmp(reference, got, PAIRS * D * sizeof(float)));
        float tail[D];
        CHECK(ds4_gpu_tensor_read(pk, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, kv + (ROWS - 1) * D, sizeof(tail)));
        CHECK(ds4_gpu_tensor_read(ps, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, scores + (ROWS - 1) * D, sizeof(tail)));
    }
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_pool2(out, kt, st, pk, ps, D, ROWS - 1, 0));
    CHECK(ds4_gpu_end_commands());
    CHECK(!memcmp(ds4_gpu_tensor_contents(pk), kv + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(ps), scores + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(out), reference, PAIRS * D * sizeof(float)));
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(ps); ds4_gpu_tensor_free(out);
    free(kv); free(scores); free(got); free(reference);
    fprintf(stderr, "V4.1 pair pooling: exact across eight chunk sizes\n");
    return 1;
}

typedef struct { float score; uint32_t index; } candidate;
static int candidate_desc(const void *a, const void *b) {
    const candidate *x = a, *y = b;
    return x->score > y->score ? -1 : x->score < y->score ? 1 :
           x->index < y->index ? -1 : x->index > y->index;
}

static int check_candidates(void) {
    const uint32_t widths[] = {1, 7, 8, 9, 127, 16385, 17017};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7) / 8, rows = 17;
        const uint32_t top = blocks < 2048 ? blocks : 2048;
        float *scores = malloc((size_t)n * rows * 4), *got = malloc((size_t)n * rows * 4);
        float *maxima = malloc((size_t)blocks * rows * 4);
        candidate *sorted = malloc(blocks * sizeof(candidate));
        uint8_t *kept = malloc(blocks);
        CHECK(scores && got && maxima && sorted && kept);
        /* Unique finite values avoid unspecified top-k tie ordering. */
        for (uint32_t r = 0; r < rows; r++) for (uint32_t i = 0; i < n; i++)
            scores[(size_t)r * n + i] = (float)((i * 7919u + r * 1009u) % 104729u) - 50000;
        ds4_gpu_tensor *s = upload(NULL, (size_t)n * rows * 4);
        ds4_gpu_tensor *b = upload(NULL, (size_t)blocks * rows * 4);
        ds4_gpu_tensor *t = upload(NULL, (size_t)top * rows * 4);
        ds4_gpu_tensor *m = upload(NULL, (size_t)blocks * rows * 4);
        CHECK(s && b && t && m);
        for (uint32_t ratio = 1; ratio <= 2; ratio++) for (int late = 0; late < 2; late++) {
            const uint32_t start = late ? n * ratio - 1 : 0;
            CHECK(ds4_gpu_tensor_write(s, 0, scores, (size_t)n * rows * 4));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, start, ratio));
            CHECK(ds4_gpu_indexer_topk_tensor(t, b, blocks, rows, top));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(m, t, blocks, rows, top));
            CHECK(ds4_gpu_dsv41_candidate_filter(s, m, n, rows, start, ratio, 0));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(b, 0, maxima, (size_t)blocks * rows * 4));
            CHECK(ds4_gpu_tensor_read(s, 0, got, (size_t)n * rows * 4));
            for (uint32_t r = 0; r < rows; r++) {
                uint32_t visible = (start + r + 1) / ratio;
                if (visible > n) visible = n;
                for (uint32_t j = 0; j < blocks; j++) {
                    float best = -INFINITY;
                    for (uint32_t i = j * 8; i < (j + 1) * 8 && i < visible; i++)
                        best = fmaxf(best, scores[(size_t)r * n + i]);
                    if (visible && j == (visible - 1) / 8) best = INFINITY;
                    CHECK(maxima[(size_t)r * blocks + j] == best);
                    sorted[j] = (candidate){best, j};
                }
                qsort(sorted, blocks, sizeof(*sorted), candidate_desc);
                memset(kept, 0, blocks);
                for (uint32_t j = 0; j < top; j++)
                    if (sorted[j].score > -INFINITY) kept[sorted[j].index] = 1;
                for (uint32_t i = 0; i < n; i++) {
                    float expected = i < visible && kept[i / 8] ? scores[(size_t)r * n + i] : -INFINITY;
                    CHECK(got[(size_t)r * n + i] == expected);
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, UINT32_MAX, 1));
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, 0, 0));
        CHECK(!ds4_gpu_dsv41_candidate_filter(s, m, n, rows + 1, 0, 1, 0));
        CHECK(blocks == 1 || !ds4_gpu_dsv41_candidate_filter(s, m, n, rows, 0, 1, blocks - 1));
        ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(t); ds4_gpu_tensor_free(m);
        free(scores); free(got); free(maxima); free(sorted); free(kept);
    }
    fprintf(stderr, "V4.1 causal candidate blocks and filtering: exact\n");
    return 1;
}

/* Batched block top-k, strided masks and filtering against the per-row calls. */
static int check_candidates_batch(uint32_t n, uint32_t rows, uint32_t start) {
    const uint32_t blocks = (n + 7) / 8, stride = blocks + 3;
    float *scores = malloc((size_t)n * rows * 4), *got = malloc((size_t)n * rows * 4);
    float *mask = malloc((size_t)stride * rows * 4);
    CHECK(scores && got && mask);
    for (uint32_t r = 0; r < rows; r++) for (uint32_t i = 0; i < n; i++)
        scores[(size_t)r * n + i] = (float)((i * 7919u + r * 1009u) % 104729u) - 50000;
    for (size_t i = 0; i < (size_t)stride * rows; i++) mask[i] = 7.0f;
    ds4_gpu_tensor *s = upload(scores, (size_t)n * rows * 4);
    ds4_gpu_tensor *b = upload(NULL, (size_t)blocks * rows * 4);
    ds4_gpu_tensor *t = upload(NULL, (size_t)2048 * rows * 4);
    ds4_gpu_tensor *ref = upload(NULL, (size_t)2048 * rows * 4);
    ds4_gpu_tensor *m = upload(mask, (size_t)stride * rows * 4);
    CHECK(s && b && t && ref && m);
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, start, 1));
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t visible = (start + r + 8) / 8;
        if (visible > blocks) visible = blocks;
        ds4_gpu_tensor *br = ds4_gpu_tensor_view(b, (size_t)r * blocks * 4, (size_t)visible * 4);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(ref, (size_t)r * 2048 * 4, 2048 * 4);
        CHECK(br && out && ds4_gpu_indexer_topk_tensor(out, br, visible, 1, 2048));
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(br);
    }
    CHECK(ds4_gpu_dsv41_candidate_topk_batch(t, b, blocks, rows, start));
    CHECK(ds4_gpu_dsv41_candidate_mask_batch(m, t, blocks, rows, stride));
    CHECK(ds4_gpu_dsv41_candidate_filter(s, m, n, rows, start, 1, stride));
    CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    const int32_t *ids = ds4_gpu_tensor_contents(t), *expected = ds4_gpu_tensor_contents(ref);
    CHECK(ids && expected && !memcmp(ids, expected, (size_t)2048 * rows * 4));
    CHECK(ds4_gpu_tensor_read(m, 0, mask, (size_t)stride * rows * 4));
    CHECK(ds4_gpu_tensor_read(s, 0, got, (size_t)n * rows * 4));
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t visible = start + r + 1;
        if (visible > n) visible = n;
        for (uint32_t j = 0; j < blocks; j++) {
            bool kept = false;
            for (uint32_t k = 0; k < 2048 && !kept; k++) kept = ids[(size_t)r * 2048 + k] == (int32_t)j;
            CHECK(mask[(size_t)r * stride + j] == (kept ? 0.0f : -INFINITY));
        }
        for (uint32_t j = blocks; j < stride; j++) CHECK(mask[(size_t)r * stride + j] == 7.0f);
        for (uint32_t i = 0; i < n; i++) {
            const float expect = i < visible && mask[(size_t)r * stride + i / 8] == 0.0f ?
                scores[(size_t)r * n + i] : -INFINITY;
            CHECK(got[(size_t)r * n + i] == expect);
        }
    }
    CHECK(!ds4_gpu_dsv41_candidate_topk_batch(t, b, blocks, rows, 16376));
    CHECK(!ds4_gpu_dsv41_candidate_mask_batch(m, t, blocks, rows, blocks - 1));
    ds4_gpu_tensor_free(m); ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(t);
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(s);
    free(scores); free(got); free(mask);
    fprintf(stderr, "V4.1 batched candidates n=%u start=%u: exact\n", n, start);
    return 1;
}

static int check_sparse_gather(void) {
    const uint32_t sizes[] = {1, 3, 511, 512, 513, 8193, 17017};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
        const uint32_t rows = sizes[i], selected = rows < 512 ? rows : 512;
        ds4_gpu_tensor *source = upload(NULL, (uint64_t)rows * 512 * 4);
        ds4_gpu_tensor *ids = upload(NULL, selected * 4);
        ds4_gpu_tensor *out = upload(NULL, (uint64_t)selected * 512 * 4);
        CHECK(source && ids && out);
        float *input = ds4_gpu_tensor_contents(source);
        int32_t *indices = ds4_gpu_tensor_contents(ids);
        for (uint32_t r = 0; r < rows; r++) for (uint32_t c = 0; c < 512; c++)
            input[(size_t)r * 512 + c] = (float)r + (float)c / 512.0f;
        for (uint32_t r = 0; r < selected; r++) indices[r] = (int32_t)(rows - 1u - r);
        CHECK(ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected));
        const float *result = ds4_gpu_tensor_contents(out);
        for (uint32_t r = 0; r < selected; r++)
            CHECK(!memcmp(result + r * 512, input + (size_t)indices[r] * 512, 512 * 4));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows + 1, selected));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected + 1));
        ds4_gpu_tensor_free(source); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(out);
    }
    fprintf(stderr, "V4.1 bounded sparse KV gather: exact\n");
    return 1;
}

#ifdef __APPLE__
/* Random Q8_0 weights [rows][cols] at page-aligned memory (scales around 2^-8). */
static void *q8_weights(uint32_t rows, uint32_t cols, uint64_t *bytes) {
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    *bytes = (uint64_t)rows * cols / 32 * sizeof(q8_block);
    void *model = NULL;
    if (posix_memalign(&model, getpagesize(), *bytes) != 0) return NULL;
    q8_block *w = model;
    for (uint64_t i = 0; i < *bytes / sizeof(*w); i++) {
        w[i].d = (uint16_t)(0x1c00 + (int)(random_value() * 8192) % 1024);
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(random_value() * 8192) % 127);
    }
    return model;
}

/* Run one call as built, where kernel must run once, then with the V4.1 paths in off; the outputs must match. */
#define LITERAL_PAIR(off, kernel, out, bytes, call) do { \
        const uint64_t runs_ = ds4_gpu_test_kernel_runs(kernel); \
        CHECK(ds4_gpu_begin_commands() && (call) && ds4_gpu_end_commands() && ds4_gpu_synchronize()); \
        CHECK(ds4_gpu_test_kernel_runs(kernel) == runs_ + 1); \
        void *lit_ = malloc(bytes); \
        CHECK(lit_); \
        memcpy(lit_, ds4_gpu_tensor_contents(out), bytes); \
        ds4_gpu_test_set_v41_off(off); \
        const int gen_ok_ = ds4_gpu_begin_commands() && (call) && ds4_gpu_end_commands() && ds4_gpu_synchronize(); \
        ds4_gpu_test_set_v41_off(0); \
        CHECK(gen_ok_ && ds4_gpu_test_kernel_runs(kernel) == runs_ + 1); \
        CHECK(!memcmp(lit_, ds4_gpu_tensor_contents(out), bytes)); \
        free(lit_); \
    } while (0)

/* One RoPE table per decode token. From the same attention partials: the reduce that writes the heads inverse-roped
 * and bf16-rounded, then the plain low (literal and generic), gives the plain reduce and inline-rope low's bits; the
 * indexer query's one-dispatch rope + quantize gives rope then quantize. Raw and compressed, up to the 1M limit. */
static int check_rope_tab(void) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, NROWS = 64, DV = 512, NWG = 32, QN = 32 * 128 };
    const uint64_t hb = (uint64_t)NROWS * DV * 4, lb = (uint64_t)GROUPS * RANK * 4;
    const uint64_t tn = (uint64_t)NROWS * (DV * NWG + 2u * NWG);
    uint64_t wb = 0;
    void *model = q8_weights(GROUPS * RANK, GROUP, &wb);
    float *t = malloc(tn * 4), *qref = malloc(QN * 4);
    CHECK(model && t && qref && ds4_gpu_set_model_map(model, wb));
    for (uint64_t i = 0; i < (uint64_t)NROWS * DV * NWG; i++) t[i] = random_value();
    for (uint64_t i = (uint64_t)NROWS * DV * NWG; i < tn; i += 2) {
        t[i] = fabsf(random_value()) + 0.25f;   /* S */
        t[i + 1] = random_value();              /* M */
    }
    ds4_gpu_tensor *tmp = upload(t, tn * 4), *plain = upload(NULL, hb), *once = upload(NULL, hb);
    ds4_gpu_tensor *q = upload(NULL, QN * 4), *tab = upload(NULL, 512);
    ds4_gpu_tensor *low[3] = {upload(NULL, lb), upload(NULL, lb), upload(NULL, lb)};
    CHECK(tmp && plain && once && q && tab && low[0] && low[1] && low[2]);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint32_t positions[] = {0, 1, 8191, 1048575};
    for (size_t pi = 0; pi < sizeof(positions) / sizeof(*positions); pi++) for (int compressed = 0; compressed < 2; compressed++) {
        const uint32_t pos = positions[pi];
        const uint64_t reduce = ds4_gpu_test_kernel_runs("kernel_dsv41_flash_attn_vec_reduce_heads");
        const uint64_t literal = ds4_gpu_test_kernel_runs("kernel_dsv4_attn_out_low_q8_0_f32_static");
        CHECK(ds4_gpu_begin_commands() && ds4_gpu_dsv41_rope_coeff_table(tab, pos));
        CHECK(ds4_gpu_dsv41_flash_attn_reduce_check(plain, tmp, NROWS, 0, NULL, false));
        CHECK(ds4_gpu_dsv41_flash_attn_reduce_check(once, tmp, NROWS, 1, tab, compressed));
        CHECK(ds4_gpu_dsv41_attention_low_plain(low[1], model, wb, 0, GROUP, RANK, GROUPS, once));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
        /* the indexer query from the plain heads */
        CHECK(ds4_gpu_tensor_write(q, 0, ds4_gpu_tensor_contents(plain), QN * 4) && ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_rope(q, 128, 32, 1, pos, compressed, false));
        CHECK(ds4_gpu_dsv41_quantize(q, 128, 32, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_tensor_read(q, 0, qref, QN * 4));
        CHECK(ds4_gpu_tensor_write(q, 0, ds4_gpu_tensor_contents(plain), QN * 4) && ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_rope_quantize_heads_tab(q, 128, 32, pos, compressed, DS4_V41_FP4_E8M0, tab));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize() && !memcmp(qref, ds4_gpu_tensor_contents(q), QN * 4));
        /* the inline-rope low on the plain heads, and the generic plain low on the transformed ones */
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_attention_low(low[0], model, wb, 0, GROUP, RANK, GROUPS, plain, pos, compressed));
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_LITERAL_SHAPES);
        const int ok = ds4_gpu_dsv41_attention_low_plain(low[2], model, wb, 0, GROUP, RANK, GROUPS, once);
        ds4_gpu_test_set_v41_off(0);
        CHECK(ok && ds4_gpu_end_commands() && ds4_gpu_synchronize());
        CHECK(!memcmp(ds4_gpu_tensor_contents(low[0]), ds4_gpu_tensor_contents(low[1]), lb));
        CHECK(!memcmp(ds4_gpu_tensor_contents(low[0]), ds4_gpu_tensor_contents(low[2]), lb));
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_flash_attn_vec_reduce_heads") == reduce + 1);
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv4_attn_out_low_q8_0_f32_static") == literal + 1);
    }
    ds4_gpu_tensor *tensors[] = {tmp, plain, once, q, tab, low[0], low[1], low[2]};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(t); free(qref); free(model);
    fprintf(stderr, "V4.1 RoPE table: transformed heads + plain low == inline rope low; indexer query rope + quantize "
            "in one dispatch == two PASS\n");
    return 1;
}

/* The V4.1 decode calls that take literal-shape kernels give the generic kernels' bits. */
static int check_literal_shapes(void) {
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    {   /* attention output-B into the HC streams: 8192 -> 5120, one row, no add */
        uint64_t wb = 0;
        void *model = q8_weights(5120, 8192, &wb);
        CHECK(model && ds4_gpu_set_model_map(model, wb));
        ds4_gpu_tensor *x = upload(NULL, 8192 * 4), *res = upload(NULL, 4 * 5120 * 4), *split = upload(NULL, 24 * 4);
        ds4_gpu_tensor *out = upload(NULL, 4 * 5120 * 4);
        CHECK(x && res && split && out);
        float *h = ds4_gpu_tensor_contents(x), *r = ds4_gpu_tensor_contents(res), *sp = ds4_gpu_tensor_contents(split);
        for (int i = 0; i < 8192; i++) h[i] = random_value();
        for (int i = 0; i < 4 * 5120; i++) r[i] = random_value();
        for (int i = 0; i < 24; i++) sp[i] = random_value() / 4.0f;
        LITERAL_PAIR(DS4_GPU_V41_LITERAL_SHAPES, "kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_static", out, 4 * 5120 * 4,
                     ds4_gpu_dsv41_matmul_expand_rows(out, model, wb, 0, 8192, 5120, x, NULL, res, split, 4, 1));
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(res); ds4_gpu_tensor_free(split); ds4_gpu_tensor_free(out);
        free(model);
    }
    {   /* the serial FFN join: shared + routed into the HC streams, one token, bf16 rounding */
        ds4_gpu_tensor *blk = upload(NULL, 5120 * 4), *add = upload(NULL, 5120 * 4), *res = upload(NULL, 4 * 5120 * 4);
        ds4_gpu_tensor *split = upload(NULL, 24 * 4), *out = upload(NULL, 4 * 5120 * 4);
        CHECK(blk && add && res && split && out);
        float *b = ds4_gpu_tensor_contents(blk), *a = ds4_gpu_tensor_contents(add);
        float *r = ds4_gpu_tensor_contents(res), *sp = ds4_gpu_tensor_contents(split);
        for (int i = 0; i < 5120; i++) { b[i] = random_value(); a[i] = random_value(); }
        for (int i = 0; i < 4 * 5120; i++) r[i] = random_value();
        for (int i = 0; i < 24; i++) sp[i] = random_value() / 4.0f;
        LITERAL_PAIR(DS4_GPU_V41_LITERAL_SHAPES, "kernel_dsv4_hc_expand4_serial_add_bf16", out, 4 * 5120 * 4,
                     ds4_gpu_hc_expand_split_add_bf16_tensor(out, blk, add, res, split, 5120, 4));
        ds4_gpu_tensor_free(blk); ds4_gpu_tensor_free(add); ds4_gpu_tensor_free(res);
        ds4_gpu_tensor_free(split); ds4_gpu_tensor_free(out);
    }
    {   /* the attention q_a/kv pair: 5120 -> 1280 and 5120 -> 512, one row, bf16 outputs */
        uint64_t wb = 0;
        void *model = q8_weights(1280 + 512, 5120, &wb);
        CHECK(model && ds4_gpu_set_model_map(model, wb));
        const uint64_t off_b = 1280ull * (5120 / 32) * 34;
        ds4_gpu_tensor *x = upload(NULL, 5120 * 4), *qa = upload(NULL, 1280 * 4), *kv = upload(NULL, 512 * 4);
        CHECK(x && qa && kv);
        float *h = ds4_gpu_tensor_contents(x);
        for (int i = 0; i < 5120; i++) h[i] = random_value();
        for (int i = 0; i < 2; i++)
            LITERAL_PAIR(DS4_GPU_V41_LITERAL_SHAPES, "kernel_mul_mv_q8_0_f32_bf16_pair_qa_kv", i ? kv : qa,
                         i ? 512 * 4 : 1280 * 4, ds4_gpu_dsv41_project_pair_q8(qa, kv, model, wb, 0, off_b, 5120, 1280, 512, x));
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(qa); ds4_gpu_tensor_free(kv);
        free(model);
    }
    if (ds4_gpu_device_is_pre_m5_apple_silicon()) {
        /* MoE gate/up of one token, 6 of 384 MXFP4 experts (5120 -> 2304), into the down projection. Only the routed
         * experts are written: up reads the gate experts one row on, down (5120 x 1224 bytes) from byte 0. */
        const uint64_t row = 2720, expert = 2304 * row, page = (uint64_t)getpagesize();
        const uint64_t map_bytes = (384 * expert + row + page - 1) / page * page;
        unsigned char *map = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        CHECK(map != MAP_FAILED && ds4_gpu_set_model_map(map, map_bytes));
        const int32_t sel[6] = {3, 77, 150, 200, 301, 383};
        const float weights[6] = {0.3f, 0.25f, 0.2f, 0.12f, 0.08f, 0.05f};
        for (int e = 0; e < 6; e++)
            for (uint64_t i = (uint64_t)sel[e] * expert; i < (uint64_t)(sel[e] + 1) * expert + row; i++) {
                (void)random_value();
                map[i] = i % 17 ? (unsigned char)(seed >> 5) : (unsigned char)(118 + seed % 8);   /* e8m0 scales */
            }
        ds4_gpu_tensor *x = upload(NULL, 5120 * 4), *st = upload(sel, sizeof(sel)), *wt = upload(weights, sizeof(weights));
        ds4_gpu_tensor *t[5];
        for (int i = 0; i < 5; i++) t[i] = upload(NULL, 6 * 5120 * 4);
        CHECK(x && st && wt && t[0] && t[1] && t[2] && t[3] && t[4]);
        float *h = ds4_gpu_tensor_contents(x);
        for (int i = 0; i < 5120; i++) h[i] = random_value();
        LITERAL_PAIR(DS4_GPU_V41_LITERAL_SHAPES, "kernel_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_v41_f32",
                     t[2], 6 * 2304 * 4, ds4_gpu_routed_moe_one_tensor(t[4], t[0], t[1], t[2], t[3], map, map_bytes,
                     0, row, 0, 39, 39, expert, row, expert, 1224, 5120, 2304, 5120, st, wt, 384, 6, 10.0f, x, NULL,
                     0, true));
        for (int i = 0; i < 5; i++) ds4_gpu_tensor_free(t[i]);
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
        /* map stays: the model views may wrap it */
    }
    fprintf(stderr, "V4.1 literal shapes (output-B HC, serial HC join, q_a/kv pair, MoE gate/up) == generic PASS\n");
    return 1;
}

/* The HC collapse over 32 + 10 groups against one group, bitwise, fresh inputs on every call: the collapsed row and
 * its RMS norm. */
static int check_split_collapse(void) {
    const uint64_t wb = 65536;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
    ds4_gpu_tensor *res = upload(NULL, 4 * 5120 * 4), *pre = upload(NULL, 16);
    ds4_gpu_tensor *x[2] = {upload(NULL, 5120 * 4), upload(NULL, 5120 * 4)};
    ds4_gpu_tensor *n[2] = {upload(NULL, 5120 * 4), upload(NULL, 5120 * 4)};
    CHECK(res && pre && x[0] && x[1] && n[0] && n[1] && ds4_gpu_set_model_map(model, wb));
    float *w = model, *r = ds4_gpu_tensor_contents(res), *p = ds4_gpu_tensor_contents(pre);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (int call = 0; call < 8; call++) {
        for (int i = 0; i < 5120; i++) w[i] = 1.0f + random_value() / 8.0f;
        for (int i = 0; i < 4 * 5120; i++) r[i] = random_value() * (call % 4 == 3 ? 64.0f : 1.0f);
        for (int i = 0; i < 4; i++) p[i] = random_value();
        const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse_part");
        for (int off = 0; off < 2; off++) {
            ds4_gpu_test_set_v41_off(off ? DS4_GPU_V41_SPLIT_REDUCE : 0);
            const int ok = ds4_gpu_begin_commands() && ds4_gpu_dsv41_deferred_collapse(x[off], n[off], res, pre,
                model, wb, 0, 1e-6f) && ds4_gpu_end_commands() && ds4_gpu_synchronize();
            ds4_gpu_test_set_v41_off(0);
            CHECK(ok);
        }
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse_part") == runs + 1);
        CHECK(!memcmp(ds4_gpu_tensor_contents(x[0]), ds4_gpu_tensor_contents(x[1]), 5120 * 4));
        CHECK(!memcmp(ds4_gpu_tensor_contents(n[0]), ds4_gpu_tensor_contents(n[1]), 5120 * 4));
    }
    ds4_gpu_tensor_free(res); ds4_gpu_tensor_free(pre);
    for (int i = 0; i < 2; i++) { ds4_gpu_tensor_free(x[i]); ds4_gpu_tensor_free(n[i]); }
    free(model);
    fprintf(stderr, "V4.1 split HC collapse == one group, bitwise PASS\n");
    return 1;
}
#else
static int check_rope_tab(void) { return 1; }
static int check_literal_shapes(void) { return 1; }
static int check_split_collapse(void) { return 1; }
#endif

static int check_attention_output(bool large) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, OUT = 5120 };
    const uint32_t ROWS = large ? 8192u : 513u;
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * GROUP / 32 * sizeof(q8_block);
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * RANK / 32 * sizeof(q8_block);
    const uint64_t xb = (uint64_t)ROWS * GROUPS * GROUP * 4;
    const uint64_t lb = (uint64_t)ROWS * GROUPS * RANK * 4, ob = (uint64_t)ROWS * OUT * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), a_bytes + b_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < (a_bytes + b_bytes) / sizeof(*w); i++) {
        w[i].d = 0x2000; /* Exact 1/128 scale and binary-fraction inputs. */
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int)(random_value() * 8192) % 8;
    }
    float *x = malloc(xb), *reference = malloc(ob), *reference_low = malloc(lb);
    CHECK(x && reference && reference_low);
    for (uint64_t i = 0; i < xb / 4; i++) x[i] = (int)(random_value() * 8192) % 51 / 256.0f;
    ds4_gpu_tensor *xt = upload(x, xb), *low = upload(NULL, lb), *out = upload(NULL, ob);
    CHECK(xt && low && out && ds4_gpu_set_model_map(model, a_bytes + b_bytes));
    CHECK(ds4_gpu_begin_commands());
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)r * GROUPS * GROUP * 4,
                                                GROUPS * GROUP * 4);
        ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low, (uint64_t)r * GROUPS * RANK * 4,
                                                GROUPS * RANK * 4);
        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(out, (uint64_t)r * OUT * 4, OUT * 4);
        CHECK(xr && lr && yr);
        CHECK(ds4_gpu_attention_output_low_q8_tensor(lr, model, a_bytes + b_bytes,
            0, GROUP, RANK, GROUPS, xr));
        CHECK(ds4_gpu_dsv41_quantize(lr, GROUPS * RANK, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_tensor(yr, model, a_bytes + b_bytes,
            a_bytes, GROUPS * RANK, OUT, lr, 1));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(lr); ds4_gpu_tensor_free(yr);
    }
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(out, 0, reference, ob));
    CHECK(ds4_gpu_tensor_read(low, 0, reference_low, lb));
    const uint32_t sizes[] = {1, 2, 3, 5, 6, 8, 31, 32, 63, 64, 65, 257, 512, 513, 8191, 8192};
    for (unsigned n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        const uint32_t rows = sizes[n];
        if (rows > ROWS) break;
        CHECK(ds4_gpu_tensor_fill_f32(out, NAN, ob / 4));
        CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
        CHECK(ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
            0, a_bytes, xt, rows));
        const float *got = ds4_gpu_tensor_contents(out);
        const float *got_low = ds4_gpu_tensor_contents(low);
        CHECK(!memcmp(got_low, reference_low, (uint64_t)rows * GROUPS * RANK * 4));
        for (uint64_t i = 0; i < (uint64_t)rows * OUT; i++) {
            if (!isfinite(got[i]) || fabsf(got[i] - reference[i]) > 2e-5f * (1 + fabsf(reference[i])))
                fprintf(stderr, "attention output rows=%u index=%llu actual=%.9g reference=%.9g\n",
                    rows, (unsigned long long)i, got[i], reference[i]);
            CHECK(isfinite(got[i]) && fabsf(got[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        }
        if (rows < ROWS) CHECK(isnan(got[(uint64_t)rows * OUT]));
        fprintf(stderr, "V4.1 batched Q8 output, BF16 boundary, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, xt, ROWS));
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, xt, ROWS + 1));
#ifndef DS4_ROCM_BUILD
    ds4_gpu_tensor *packed = upload(NULL, xb / 2);
    ds4_gpu_tensor *partial[2] = {upload(NULL, ob), upload(NULL, ob)};
    CHECK(packed && partial[0] && partial[1]);
    const uint32_t tp_sizes[] = {1, 2, 3, 5, 6, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 512, 513};
    const uint32_t q_half = GROUPS / 2 * GROUP, low_half = GROUPS / 2 * RANK;
    for (size_t n = 0; n < sizeof(tp_sizes) / sizeof(*tp_sizes); n++) {
        const uint32_t rows = tp_sizes[n];
        for (uint32_t rank = 0; rank < 2; rank++) {
            float *px = ds4_gpu_tensor_contents(packed);
            for (uint32_t t = 0; t < rows; t++)
                memcpy(px + (size_t)t * q_half, x + (size_t)t * 2 * q_half + rank * q_half,
                       q_half * sizeof(float));
            CHECK(ds4_gpu_tensor_fill_f32(partial[rank], NAN, ob / 4));
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_dsv41_attention_output_tp_batch(partial[rank], low,
                model, a_bytes + b_bytes, 0, a_bytes, packed, rows, rank));
            const float *got_low = ds4_gpu_tensor_contents(low);
            const float *got = ds4_gpu_tensor_contents(partial[rank]);
            for (uint32_t t = 0; t < rows; t++) {
                CHECK(!memcmp(got_low + (size_t)t * low_half,
                    reference_low + (size_t)t * 2 * low_half + rank * low_half,
                    low_half * sizeof(float)));
                /* Independent double sums sample both ends and interior output
                 * rows, including rank one's strided input slice. */
                const uint32_t columns[] = {0, 1, 31, 64, 997, 4095, OUT - 1};
                for (size_t c = 0; c < sizeof(columns) / sizeof(*columns); c++) {
                    const uint32_t col = columns[c];
                    const q8_block *bw = (const q8_block *)((const char *)model + a_bytes) +
                        (size_t)col * 2 * low_half / 32 + rank * low_half / 32;
                    double sum = 0;
                    double rounding_bound = 0;
#ifndef __APPLE__
                    /* CUDA's established Q8 dot first rounds each activation
                     * block to signed bytes. TP must keep that same boundary. */
                    for (uint32_t k = 0; k < low_half; k += 32) {
                        const float *v = reference_low + (size_t)t * 2 * low_half + rank * low_half + k;
                        float amax = 0;
                        for (uint32_t j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(v[j]));
                        const float scale = amax / 127.0f;
                        const float inv = scale ? 1.0f / scale : 0;
                        int dot = 0;
                        for (uint32_t j = 0; j < 32; j++) {
                            dot += bw[k / 32].qs[j] * (int)lrintf(v[j] * inv);
                            /* The fast GPU reciprocal can approach an exact
                             * half-integer from the other side. Bound only
                             * those ambiguous bins, not arbitrary dot error. */
                            const double q = amax ? fabs((double)v[j] * 127.0 / amax) : 0;
                            if (q - floor(q) == 0.5)
                                rounding_bound += abs(bw[k / 32].qs[j]) * (double)scale / 128.0;
                        }
                        sum += dot * (double)scale / 128.0;
                    }
#else
                    for (uint32_t k = 0; k < low_half; k++)
                        sum += bw[k / 32].qs[k % 32] / 128.0 *
                            reference_low[(size_t)t * 2 * low_half + rank * low_half + k];
#endif
                    if (!isfinite(got[(size_t)t * OUT + col]) ||
                        fabs(got[(size_t)t * OUT + col] - sum) > rounding_bound + 2e-5 * (1 + fabs(sum)))
                        fprintf(stderr, "TP output rank=%u rows=%u row=%u col=%u actual=%.9g oracle=%.9g tie_bound=%.9g\n",
                            rank, rows, t, col, got[(size_t)t * OUT + col], sum, rounding_bound);
                    CHECK(isfinite(got[(size_t)t * OUT + col]) &&
                        fabs(got[(size_t)t * OUT + col] - sum) <= rounding_bound + 2e-5 * (1 + fabs(sum)));
                }
            }
            CHECK(isnan(got_low[(size_t)rows * low_half]));
            if (rows < ROWS) CHECK(isnan(got[(size_t)rows * OUT]));
        }
        const float *a = ds4_gpu_tensor_contents(partial[0]);
        const float *b = ds4_gpu_tensor_contents(partial[1]);
        for (size_t i = 0; i < (size_t)rows * OUT; i++)
            CHECK(isfinite(a[i]) && isfinite(b[i]) &&
                fabsf(a[i] + b[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        fprintf(stderr, "V4.1 TP Q8 output, both ranks, BF16 and double oracle, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, 1, 2));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, packed, 1, 1));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, ROWS + 1, 0));
    ds4_gpu_tensor_free(partial[0]); ds4_gpu_tensor_free(partial[1]); ds4_gpu_tensor_free(packed);
#else
    fprintf(stderr, "V4.1 TP output projection: skipped (ROCm has no network TP)\n");
#endif
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(x); free(reference); free(reference_low); free(model);
    return 1;
}

#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
static int check_indexer_grouped_keys(const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights) {
    enum { KEYS = 8197, BLOCKS = (KEYS + 7) / 8, DIM = 128 };
    ds4_gpu_tensor *keys = upload(NULL, KEYS * DIM * sizeof(float));
    ds4_gpu_tensor *out = upload(NULL, (KEYS + 1) * sizeof(float));
    ds4_gpu_tensor *reference = upload(NULL, KEYS * sizeof(float));
    ds4_gpu_tensor *mask = upload(NULL, BLOCKS * sizeof(float));
    CHECK(keys && out && reference && mask);
    float *k = ds4_gpu_tensor_contents(keys), *m = ds4_gpu_tensor_contents(mask);
    float *s = ds4_gpu_tensor_contents(out), *ref = ds4_gpu_tensor_contents(reference);
    for (size_t i = 0; i < (size_t)KEYS * DIM; i++) k[i] = random_value();
    CHECK(ds4_gpu_dsv41_quantize(keys, DIM, KEYS, DS4_V41_FP4_E8M0));
    ds4_gpu_set_quality(true);
    CHECK(ds4_gpu_glm_indexer_score_one_tensor(reference, q, weights, keys, KEYS, 32, DIM,
                                               1.0f / 64.0f, false));
    CHECK(ds4_gpu_synchronize());
    ds4_gpu_set_quality(false);
    s[KEYS] = 12345;
    CHECK(ds4_gpu_dsv41_indexer_scores_batch(out, q, weights, keys, KEYS, 1, 2 * KEYS - 1, 2));
    CHECK(ds4_gpu_synchronize() && s[KEYS] == 12345 && !memcmp(s, ref, KEYS * sizeof(float)));
    for (unsigned phase = 0; phase < 2; phase++) {
        for (unsigned i = 0; i < BLOCKS; i++) m[i] = (i + phase) % 2u ? -INFINITY : 0.0f;
        CHECK(ds4_gpu_dsv41_indexer_scores_masked_one(out, q, weights, keys, mask, KEYS));
        CHECK(ds4_gpu_synchronize() && s[KEYS] == 12345);
        for (unsigned i = 0; i < KEYS; i++)
            CHECK(m[i / 8] ? s[i] == -INFINITY : !memcmp(s + i, ref + i, sizeof(float)));
    }
    ds4_gpu_tensor_free(mask); ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(keys);
    return 1;
}
#endif

static int check_indexer_batch(void) {
    enum { KEYS = 1025, ROWS = 33, HEADS = 32, DIM = 128 };
    ds4_gpu_tensor *qt = upload(NULL, (size_t)ROWS * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)KEYS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)ROWS * HEADS * sizeof(float));
    ds4_gpu_tensor *st = upload(NULL, (size_t)(ROWS * KEYS + 1) * sizeof(float));
    const uint64_t packed_bytes = ds4_gpu_dsv41_indexer_packed_bytes(KEYS, ROWS);
    ds4_gpu_tensor *pt = upload(NULL, (size_t)packed_bytes + 4);
    CHECK(qt && kt && wt && st && pt);
#ifndef __APPLE__
    ds4_gpu_tensor *reference = upload(NULL, (size_t)KEYS * sizeof(float));
    CHECK(reference);
#endif
    float *q = ds4_gpu_tensor_contents(qt), *k = ds4_gpu_tensor_contents(kt);
    float *w = ds4_gpu_tensor_contents(wt), *s = ds4_gpu_tensor_contents(st);
    CHECK(q && k && w && s);
    const bool packed_available = ds4_gpu_dsv41_tensor_ops_available();
    for (uint32_t range = 0; range < 3; range++) {
        for (size_t i = 0; i < (size_t)ROWS * HEADS * DIM; i++)
            q[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)KEYS * DIM; i++)
            k[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)ROWS * HEADS; i++) w[i] = bf16(random_value());
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, ROWS * HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, KEYS, DS4_V41_FP4_E8M0));
        if (range == 2) { q[0] = 1.0001f; k[67u * DIM] = 1.0003f; }
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        /* The fused rejection must leave the same original score slots,
         * including a partial final block whether retained or rejected. */
        float mask[(KEYS + 7u) / 8u];
        for (size_t i = 0; i < sizeof(mask) / sizeof(mask[0]); i++)
            mask[i] = (i + range) % 3u ? -INFINITY : 0.0f;
        ds4_gpu_tensor *mt = upload(mask, sizeof(mask));
        CHECK(mt && ds4_gpu_dsv41_indexer_scores_batch(reference, qt, wt, kt, KEYS, 1, KEYS - 1, 1));
        CHECK(ds4_gpu_dsv41_candidate_filter(reference, mt, KEYS, 1, KEYS - 1, 1, 0));
        s[KEYS] = 12345;
        CHECK(ds4_gpu_dsv41_indexer_scores_masked_one(st, qt, wt, kt, mt, KEYS));
        CHECK(ds4_gpu_synchronize() && s[KEYS] == 12345);
        CHECK(!memcmp(s, ds4_gpu_tensor_contents(reference), KEYS * sizeof(float)));
        ds4_gpu_tensor_free(mt);
#endif
        uint32_t *packed_guard = (uint32_t *)((char *)ds4_gpu_tensor_contents(pt) + packed_bytes);
        *packed_guard = 0xabcdef01;
        if (packed_available) CHECK(ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS));
        CHECK(*packed_guard == 0xabcdef01);
        const uint32_t counts[] = {1, 7, 8, 9, 31, 32, 33};
        for (uint32_t ratio = 1; ratio <= 2; ratio++) {
            for (uint32_t early = 0; early <= 1; early++) {
                for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                    const uint32_t rows = counts[ci];
                    const uint32_t start = early ? 0 : KEYS * ratio - rows;
                    for (uint32_t mode = 0; mode < (packed_available ? 3u : 1u); mode++) {
                        const uint32_t offset = mode == 2 && rows < ROWS ? 1u : 0u;
                        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(qt, (uint64_t)offset * HEADS * DIM * 4u,
                            (uint64_t)rows * HEADS * DIM * 4u);
                        ds4_gpu_tensor *wv = ds4_gpu_tensor_view(wt, (uint64_t)offset * HEADS * 4u,
                            (uint64_t)rows * HEADS * 4u);
                        CHECK(qv && wv);
                        s[(size_t)rows * KEYS] = 12345;
                        CHECK(mode ? ds4_gpu_dsv41_indexer_scores_packed(st, qv, wv, kt, pt,
                            KEYS, rows, start, ratio, ROWS, offset) :
                            ds4_gpu_dsv41_indexer_scores_batch(st, qv, wv, kt, KEYS, rows, start, ratio));
                        CHECK(ds4_gpu_synchronize());
                        ds4_gpu_tensor_free(qv); ds4_gpu_tensor_free(wv);
                        CHECK(s[(size_t)rows * KEYS] == 12345);
                        CHECK(*packed_guard == 0xabcdef01);
#ifndef __APPLE__
                        /* Compare the CUDA warp scorer with the original
                         * shared-memory reduction, not only a loose oracle. */
                        ds4_gpu_set_quality(true);
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            if (!visible) continue;
                            ds4_gpu_tensor *qr = ds4_gpu_tensor_view(qt,
                                (uint64_t)t * HEADS * DIM * 4u, HEADS * DIM * 4u);
                            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt,
                                (uint64_t)t * HEADS * 4u, HEADS * 4u);
                            CHECK(qr && wr && ds4_gpu_glm_indexer_score_one_tensor(
                                reference, qr, wr, kt, visible, HEADS, DIM, 1.0f / 64.0f, false));
                            CHECK(ds4_gpu_synchronize());
                            CHECK(!memcmp(s + (size_t)t * KEYS,
                                ds4_gpu_tensor_contents(reference), visible * sizeof(float)));
                            ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(wr);
                        }
                        ds4_gpu_set_quality(false);
#endif
                        double worst = 0;
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            for (uint32_t j = 0; j < KEYS; j++) {
                                const float actual = s[(size_t)t * KEYS + j];
                                if (j >= visible) { CHECK(actual == -INFINITY); continue; }
                                double expected = 0, magnitude = 0;
                                for (uint32_t h = 0; h < HEADS; h++) {
                                    double dot = 0;
                                    for (uint32_t d = 0; d < DIM; d++)
                                        dot += (double)q[((size_t)(t + offset) * HEADS + h) * DIM + d] *
                                               k[(size_t)j * DIM + d];
                                    const double term = fmax(dot / 64.0, 0) * w[(t + offset) * HEADS + h];
                                    expected += term;
                                    magnitude += fabs(term);
                                }
                                CHECK(isfinite(actual));
                                const double error = fabs(actual - expected) / fmax(magnitude, 1);
                                worst = fmax(worst, error);
                                CHECK(error < 0.00001);
                            }
                        }
                        fprintf(stderr, "V4.1 index scores mode=%u range=%u ratio=%u start=%u rows=%u error=%.9g\n",
                                mode, range, ratio, start, rows, worst);
                    }
                }
            }
        }
    }
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    if (ds4_gpu_device_is_spark()) CHECK(check_indexer_grouped_keys(qt, wt));
#endif
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 4));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, UINT32_MAX, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS + 1, 0, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, KEYS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_pack(st, qt, kt, KEYS, ROWS));
    CHECK(!ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS + 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 0, ROWS, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, ROWS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, UINT32_MAX, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, st,
        KEYS, ROWS, 0, 1, ROWS, 0));
    ds4_gpu_tensor_free(pt);
#ifndef __APPLE__
    ds4_gpu_tensor_free(reference);
#endif
    ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 causal batched index scores: double-precision oracle PASS\n");
    return 1;
}

static double monotonic_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int check_embedding(void) {
    enum { VOCAB = 19, WIDTH = 5120, HC = 4, ROWS = 257 };
    const size_t row_bytes = WIDTH * HC * sizeof(float);
    const size_t weight_bytes = VOCAB * WIDTH * sizeof(uint16_t);
    uint16_t *model = NULL;
    CHECK(posix_memalign((void **)&model, getpagesize(), weight_bytes) == 0);
    for (unsigned i = 0; i < VOCAB * WIDTH; i++)
        model[i] = (uint16_t)(0x2000u | (i % 1024u) | ((i & 1u) ? 0x8000u : 0));
    int32_t ids[ROWS];
    for (unsigned i = 0; i < ROWS; i++) ids[i] = (int32_t)((i * 7u) % VOCAB);
    CHECK(ds4_gpu_set_model_map(model, weight_bytes));
    ds4_gpu_tensor *tokens = upload(ids, sizeof(ids));
    ds4_gpu_tensor *out = upload(NULL, ROWS * row_bytes + sizeof(float));
    ds4_gpu_tensor *ref = upload(NULL, VOCAB * row_bytes);
    CHECK(tokens && out && ref);
    for (unsigned token = 0; token < VOCAB; token++) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(ref, token * row_bytes, row_bytes);
        CHECK(row && ds4_gpu_embed_token_hc_tensor(row, model, weight_bytes, 0,
                                                   VOCAB, token, WIDTH, HC));
        ds4_gpu_tensor_free(row);
    }
    const unsigned counts[] = {1, 7, 127, 128, ROWS};
    const float marker = 12345;
    float *actual = malloc(ROWS * row_bytes + sizeof(float));
    float *expected = malloc(VOCAB * row_bytes);
    CHECK(actual && expected && ds4_gpu_tensor_read(ref, 0, expected, VOCAB * row_bytes));
    for (unsigned i = 0; i < sizeof(counts) / sizeof(*counts); i++) {
        const unsigned rows = counts[i];
        CHECK(ds4_gpu_tensor_write(out, rows * row_bytes, &marker, sizeof(marker)));
        CHECK(ds4_gpu_embed_tokens_hc_tensor(out, tokens, model, weight_bytes, 0,
                                             VOCAB, rows, WIDTH, HC));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, rows * row_bytes + sizeof(float)));
        CHECK(actual[rows * WIDTH * HC] == marker);
        for (unsigned row = 0; row < rows; row++) {
            CHECK(!memcmp(actual + row * WIDTH * HC, expected + ids[row] * WIDTH * HC, row_bytes));
            for (unsigned hc = 0; hc < HC; hc++) for (unsigned d = 0; d < WIDTH; d++) {
                const unsigned w = (unsigned)ids[row] * WIDTH + d;
                const float value = (1.0f + (w % 1024u) / 1024.0f) / 128.0f * ((w & 1u) ? -1 : 1);
                CHECK(actual[(row * HC + hc) * WIDTH + d] == value);
            }
        }
        fprintf(stderr, "V4.1 batched embedding rows=%u: exact scalar and independent oracle PASS\n", rows);
    }
    free(expected); free(actual);
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(tokens);
    ds4_gpu_cleanup(); free(model);
    CHECK(ds4_gpu_init());
    return 1;
}

static int check_index_projection(void) {
    enum { MAX_ROWS = 4096 };
    const uint32_t widths[] = {1280, 5120, 5120, 512}, outputs[] = {4096, 32, 512, 128};
    const uint32_t counts[] = {1, 31, 64, 65, 513, 2048, MAX_ROWS};
    for (uint32_t shape = 0; shape < 4; shape++) {
        const uint32_t width = widths[shape], output = outputs[shape];
        const size_t weight_bytes = (size_t)width * output * sizeof(uint16_t);
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), weight_bytes) == 0);
        uint16_t *weights = model;
        for (size_t i = 0; i < weight_bytes / sizeof(*weights); i++) {
            const int value = (int)(random_value() * 8192);
            weights[i] = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u) |
                                    (value < 0 ? 0x8000u : 0));
        }
        CHECK(ds4_gpu_set_model_map(model, weight_bytes));
        ds4_gpu_tensor *in = upload(NULL, (size_t)MAX_ROWS * width * sizeof(float));
        ds4_gpu_tensor *out = upload(NULL, ((size_t)MAX_ROWS * output + 1) * sizeof(float));
        ds4_gpu_tensor *ref = upload(NULL, (size_t)MAX_ROWS * output * sizeof(float));
        CHECK(in && out && ref);
        float *x = ds4_gpu_tensor_contents(in), *y = ds4_gpu_tensor_contents(out);
        const float *expected = ds4_gpu_tensor_contents(ref);
        for (size_t i = 0; i < (size_t)MAX_ROWS * width; i++) x[i] = bf16(random_value());
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t rows = counts[ci];
            y[(size_t)rows * output] = 12345;
            CHECK(ds4_gpu_begin_commands());
            for (uint32_t t = 0; t < rows; t++) {
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(in, (size_t)t * width * 4, width * 4);
                ds4_gpu_tensor *yr = ds4_gpu_tensor_view(ref, (size_t)t * output * 4, output * 4);
                CHECK(xr && yr && ds4_gpu_matmul_f16_tensor(yr, model, weight_bytes, 0, width, output, xr, 1));
                ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
            }
            CHECK(ds4_gpu_end_commands());
            const double start = monotonic_seconds();
            CHECK(ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, rows, in));
            CHECK(ds4_gpu_synchronize());
            const double seconds = monotonic_seconds() - start;
            CHECK(y[(size_t)rows * output] == 12345);
            double error = 0, norm = 0;
            for (size_t i = 0; i < (size_t)rows * output; i++) {
                CHECK(isfinite(y[i]) && isfinite(expected[i]));
                const double d = (double)y[i] - expected[i];
                error += d * d; norm += (double)expected[i] * expected[i];
            }
            const double relative = sqrt(error / fmax(norm, 1e-30));
            fprintf(stderr, "V4.1 exact F16 projection width=%u output=%u rows=%u RMS=%.9g %.3f ms\n",
                width, output, rows, relative, seconds * 1000);
            CHECK(memcmp(y, expected, (size_t)rows * output * sizeof(float)) == 0);
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t o = j * (output / 8), t = rows - 1;
                double sum = 0, magnitude = 0;
                for (uint32_t k = 0; k < width; k++) {
                    const uint16_t w = weights[(size_t)o * width + k];
                    const float value = (1.0f + (w & 1023) / 1024.0f) / 128.0f *
                                        (w & 0x8000 ? -1 : 1);
                    const double term = (double)value * x[(size_t)t * width + k];
                    sum += term; magnitude += fabs(term);
                }
                CHECK(fabs(y[(size_t)t * output + o] - sum) < 0.000001 * fmax(magnitude, 1));
            }
        }
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes - 1, 0, width, output, 1, in));
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, MAX_ROWS + 1, in));
        ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(in);
        ds4_gpu_cleanup(); free(model);
        CHECK(ds4_gpu_init());
    }
    return 1;
}

typedef struct { float value; uint32_t id; } topk_entry;

static int compare_topk_entry(const void *a, const void *b) {
    const topk_entry *x = a, *y = b;
    if (x->value > y->value) return -1;
    if (x->value < y->value) return 1;
    return (x->id > y->id) - (x->id < y->id);
}

static int check_general_topk(void) {
    const uint32_t widths[] = {1, 2, 31, 127, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi], rows = 3;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * rows * 4);
        ds4_gpu_tensor *selected = upload(NULL, ((size_t)width * rows + 1) * 4);
        topk_entry *reference = malloc((size_t)width * rows * sizeof(*reference));
        bool *seen = calloc(width, sizeof(*seen));
        CHECK(scores && selected && reference && seen);
        float *s = ds4_gpu_tensor_contents(scores);
        uint32_t *ids = ds4_gpu_tensor_contents(selected);
        const uint32_t counts[] = {1, 2, 31, 128, 511, 512, 513, 2048, width};
        for (uint32_t pattern = 0; pattern < 4; pattern++) {
            for (uint32_t t = 0; t < rows; t++) {
                for (uint32_t i = 0; i < width; i++) {
                    const float value = pattern == 0 ? random_value() :
                        pattern == 1 ? (float)(i % 7) :
                        pattern == 2 ? -INFINITY : (i & 1 ? -0.0f : 0.0f);
                    const float v = pattern == 1 && i % 17 == 0 ? INFINITY : value;
                    s[t * width + i] = v;
                    reference[t * width + i] = (topk_entry){v, i};
                }
                qsort(reference + t * width, width, sizeof(*reference), compare_topk_entry);
            }
            for (size_t ki = 0; ki < sizeof(counts) / sizeof(*counts); ki++) {
                const uint32_t k = counts[ki];
                if (k > width) continue;
                ids[rows * k] = 0xabcdef01;
                CHECK(ds4_gpu_indexer_topk_tensor(selected, scores, width, rows, k));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * k] == 0xabcdef01);
                for (uint32_t t = 0; t < rows; t++) {
                    memset(seen, 0, width * sizeof(*seen));
                    for (uint32_t i = 0; i < k; i++) {
                        const uint32_t id = ids[t * k + i];
                        CHECK(id < width && !seen[id]);
                        seen[id] = true;
#ifdef __APPLE__
                        /* Metal's bitonic order may permute exactly tied keys. */
                        CHECK(s[t * width + id] == reference[t * width + i].value);
#else
                        if (ids[t * k + i] != reference[t * width + i].id) {
                            fprintf(stderr, "top-k width=%u k=%u pattern=%u row=%u rank=%u: %u vs %u\n",
                                width, k, pattern, t, i, ids[t * k + i], reference[t * width + i].id);
                            CHECK(0);
                        }
#endif
                    }
                }
            }
        }
        fprintf(stderr, "V4.1 variable-k sort width=%u: independent sorted-value/unique-ID oracle PASS\n", width);
        ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores); free(reference); free(seen);
    }
    return 1;
}

/* The radix-select path against the merge ladder: same values by rank, same
 * ids and order when the selected values are unique, ties by ascending id.
 * For k <= 512 the winners' 512-wide sort gives the 2048-wide sort's ids. */
static int check_topk_select(void) {
    const uint32_t widths[] = {16384, 65536, 131072, 262144};
    const uint32_t counts[] = {1, 512, 2048};
#ifdef __APPLE__
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
#endif
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi], rows = 2;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * rows * 4);
        ds4_gpu_tensor *fast = upload(NULL, (size_t)2048 * rows * 4);
        ds4_gpu_tensor *slow = upload(NULL, (size_t)2048 * rows * 4);
        topk_entry *reference = malloc((size_t)width * rows * sizeof(*reference));
        bool *seen = calloc(width, sizeof(*seen));
        CHECK(scores && fast && slow && reference && seen);
        float *s = ds4_gpu_tensor_contents(scores);
        for (uint32_t pattern = 0; pattern < 4; pattern++) {
            for (uint32_t t = 0; t < rows; t++) {
                for (uint32_t i = 0; i < width; i++) {
                    /* 0: coarse random values with many ties; 1: unique; 2: unique
                     * with seven of eight blocks masked like the candidate filter;
                     * 3: fewer finite values than 2048, so ties at -inf are selected */
                    const float unique = (float)((i * 7919u + t * 1009u) % 1048573u) - 500000.0f;
                    s[(size_t)t * width + i] = pattern == 0 ? random_value() : pattern == 1 ? unique :
                        pattern == 2 ? (i / 8 % 8 ? -INFINITY : unique) : (i % 1361 ? -INFINITY : unique);
                    reference[(size_t)t * width + i] = (topk_entry){s[(size_t)t * width + i], i};
                }
                qsort(reference + (size_t)t * width, width, sizeof(*reference), compare_topk_entry);
            }
            for (size_t ki = 0; ki < sizeof(counts) / sizeof(*counts); ki++) {
                const uint32_t k = counts[ki];
                const uint32_t n = ki == 1 ? 1u : rows;   /* the scratch changes row layout between calls */
                CHECK(unsetenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == 0);
#ifdef __APPLE__
                const uint64_t sort512 = ds4_gpu_test_kernel_runs("kernel_topk_select_sort512");
#endif
                CHECK(ds4_gpu_dsv41_indexer_topk_tensor(fast, scores, width, n, k));
#ifdef __APPLE__
                CHECK(ds4_gpu_synchronize() && ds4_gpu_test_kernel_runs("kernel_topk_select_sort512") == sort512 + (k <= 512));
                if (k <= 512) {
                    ds4_gpu_test_set_v41_off(DS4_GPU_V41_DECODE_TOPK);
                    const int ok = ds4_gpu_dsv41_indexer_topk_tensor(slow, scores, width, n, k) && ds4_gpu_synchronize();
                    ds4_gpu_test_set_v41_off(0);
                    CHECK(ok && ds4_gpu_test_kernel_runs("kernel_topk_select_sort512") == sort512 + 1);
                    CHECK(!memcmp(ds4_gpu_tensor_contents(fast), ds4_gpu_tensor_contents(slow), (size_t)n * k * 4));
                }
#endif
                CHECK(setenv("DS4_METAL_DISABLE_V41_TOPK_SELECT", "1", 1) == 0);
                CHECK(ds4_gpu_indexer_topk_tensor(slow, scores, width, n, k));
                CHECK(unsetenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == 0);
                CHECK(ds4_gpu_synchronize());
                const int32_t *a = ds4_gpu_tensor_contents(fast), *b = ds4_gpu_tensor_contents(slow);
                for (uint32_t t = 0; t < n; t++) {
                    const float *row = s + (size_t)t * width;
                    const topk_entry *ref = reference + (size_t)t * width;
                    memset(seen, 0, width * sizeof(*seen));
                    for (uint32_t i = 0; i < k; i++) {
                        const int32_t id = a[t * k + i];
                        CHECK(id >= 0 && (uint32_t)id < width && !seen[id]);
                        seen[id] = true;
                        CHECK(row[id] == ref[i].value);
                        CHECK(row[id] == row[b[t * k + i]]);
                        if (i && row[id] == row[a[t * k + i - 1]]) CHECK(a[t * k + i - 1] < id);
                    }
                    if (pattern == 1 || pattern == 2)
                        CHECK(!memcmp(a + t * k, b + t * k, k * sizeof(*a)));
                }
            }
        }
        fprintf(stderr, "V4.1 radix top-k width=%u: values by rank, unique-ID and tie-order oracle PASS\n", width);
        ds4_gpu_tensor_free(scores); ds4_gpu_tensor_free(fast); ds4_gpu_tensor_free(slow);
        free(reference); free(seen);
    }
    return 1;
}

static int check_causal_topk(void) {
    const uint32_t frontiers[] = {513, 514, 1023, 1024, 1025, 1535, 2047, 2048, 4095, 4096, 8191, 8192,
                                  16383, 32767, 65535};
    const uint32_t counts[] = {1, 2, 31, 32, 33, 127, 128, 129};
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
    for (size_t fi = 0; fi < sizeof(frontiers) / sizeof(*frontiers); fi++) {
        const uint32_t start = frontiers[fi] * ratio - 1u;
        const uint32_t width = (start + 129u) / ratio + 1024u;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * 129u * sizeof(float));
        ds4_gpu_tensor *selected = upload(NULL, (129u * 512u + 1u) * sizeof(int32_t));
        ds4_gpu_tensor *reference = upload(NULL, 129u * 512u * sizeof(int32_t));
        CHECK(scores && selected && reference);
        float *s = ds4_gpu_tensor_contents(scores);
        int32_t *ids = ds4_gpu_tensor_contents(selected);
        const int32_t *ref = ds4_gpu_tensor_contents(reference);
        for (uint32_t pattern = 0; pattern < 3; pattern++) {
            for (uint32_t t = 0; t < 129u; t++) {
                const uint32_t visible = (start + t + 1u) / ratio;
                for (uint32_t j = 0; j < width; j++) {
                    /* Future entries are deliberately attractive. Causality
                     * must come from each row's bounds, not score contents. */
                    float v = pattern == 0 ? random_value() : pattern == 1 ? (float)(j % 7u) : -INFINITY;
                    s[(size_t)t * width + j] = j < visible ? v : 12345.0f;
                }
            }
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const uint32_t rows = counts[ci];
                /* The causal kernels are the ladder's variants and agree with it
                 * tie for tie; the radix path orders ties by id, so the reference
                 * stays on the ladder. */
                CHECK(setenv("DS4_METAL_DISABLE_V41_TOPK_SELECT", "1", 1) == 0);
                CHECK(ds4_gpu_begin_commands());
                for (uint32_t t = 0; t < rows; t++) {
                    const uint32_t visible = (start + t + 1u) / ratio;
                    ds4_gpu_tensor *sr = ds4_gpu_tensor_view(scores, (size_t)t * width * 4u, visible * 4u);
                    ds4_gpu_tensor *out = ds4_gpu_tensor_view(reference, (size_t)t * 512u * 4u, 512u * 4u);
                    CHECK(sr && out && ds4_gpu_indexer_topk_tensor(out, sr, visible, 1, 512));
                    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(sr);
                }
                CHECK(ds4_gpu_end_commands());
                CHECK(unsetenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == 0);
                ids[rows * 512u] = 123456;
                CHECK(ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, rows, start, ratio));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * 512u] == 123456);
                for (uint32_t t = 0; t < rows; t++) {
                    CHECK(s[(size_t)t * width + width - 1u] == 12345.0f);
                    for (uint32_t k = 0; k < 512u; k++) {
                        if (ids[t * 512u + k] != ref[t * 512u + k]) {
                            fprintf(stderr, "top-k mismatch ratio=%u visible=%u rows=%u pattern=%u row=%u rank=%u: %d vs %d\n",
                                ratio, frontiers[fi], rows, pattern, t, k, ids[t * 512u + k], ref[t * 512u + k]);
                            CHECK(0);
                        }
                        CHECK(ids[t * 512u + k] >= 0 && (uint32_t)ids[t * 512u + k] < (start + t + 1u) / ratio);
                    }
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 130, start, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, UINT32_MAX, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, 0, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, 512u * ratio - 1u, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, start, 0));
        fprintf(stderr, "V4.1 causal top-k ratio=%u visible=%u: exact IDs including ties PASS\n", ratio, frontiers[fi]);
        ds4_gpu_tensor_free(reference); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores);
    }
    }
    return 1;
}


#ifdef __APPLE__
/* The V4.1 decode top-512 of one row (shuffle block sort, staged then fused merge) against itself without the fused
 * merge on random, tied with +inf, -inf-tailed, signed-zero and NaN rows, and that against the plain ladder (V4 Flash's
 * call) where no NaN is present: the same ids in the same order. */
static int check_decode_topk(void) {
    const uint32_t widths[] = {513, 1024, 1025, 10240, 10241, 16383, 65536};
    ds4_gpu_tensor *scores = upload(NULL, 65536u * sizeof(float)), *ids[3];
    for (int i = 0; i < 3; i++) ids[i] = upload(NULL, 513u * sizeof(int32_t));
    CHECK(scores && ids[0] && ids[1] && ids[2]);
    float *s = ds4_gpu_tensor_contents(scores);
    int32_t *a = ds4_gpu_tensor_contents(ids[0]), *b = ds4_gpu_tensor_contents(ids[1]);
    int32_t *c = ds4_gpu_tensor_contents(ids[2]);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    CHECK(setenv("DS4_METAL_DISABLE_V41_TOPK_SELECT", "1", 1) == 0);   /* 65536 stays off the radix path */
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi];
        for (uint32_t pattern = 0; pattern < 5; pattern++) {
            for (uint32_t i = 0; i < width; i++)
                s[i] = pattern == 0 ? random_value() : pattern == 1 ? (i % 17 == 0 ? INFINITY : (float)(i % 7)) :
                       pattern == 2 ? (i % 3 ? -INFINITY : random_value()) : pattern == 3 ? (i & 1 ? -0.0f : 0.0f) :
                       (i % 5 == 0 ? NAN : random_value());
            a[512] = b[512] = c[512] = 0x5eed;
            const uint64_t shuffle = ds4_gpu_test_kernel_runs("kernel_argsort_f32_i32_desc");
            const uint64_t fused = ds4_gpu_test_kernel_runs("kernel_argsort_merge_f32_i32_desc_prefix_fused");
            CHECK(ds4_gpu_dsv41_indexer_topk_tensor(ids[0], scores, width, 1, 512) && ds4_gpu_synchronize());
            ds4_gpu_test_set_v41_off(DS4_GPU_V41_UNFUSED_TOPK);
            const int ok = ds4_gpu_dsv41_indexer_topk_tensor(ids[1], scores, width, 1, 512) && ds4_gpu_synchronize();
            ds4_gpu_test_set_v41_off(0);
            CHECK(ok && ds4_gpu_indexer_topk_tensor(ids[2], scores, width, 1, 512) && ds4_gpu_synchronize());
            CHECK(ds4_gpu_test_kernel_runs("kernel_argsort_f32_i32_desc") == shuffle + 3);
            CHECK(ds4_gpu_test_kernel_runs("kernel_argsort_merge_f32_i32_desc_prefix_fused") == fused + (width > 1024));
            CHECK(a[512] == 0x5eed && b[512] == 0x5eed && c[512] == 0x5eed);
            for (uint32_t k = 0; k < 512u; k++) {
                if (a[k] != b[k] || (pattern != 4 && b[k] != c[k])) {
                    fprintf(stderr, "decode top-k width=%u pattern=%u rank=%u: %d, unfused %d, ladder %d\n",
                            width, pattern, k, a[k], b[k], c[k]);
                    CHECK(0);
                }
            }
        }
    }
    CHECK(unsetenv("DS4_METAL_DISABLE_V41_TOPK_SELECT") == 0);
    ds4_gpu_tensor_free(scores);
    for (int i = 0; i < 3; i++) ds4_gpu_tensor_free(ids[i]);
    fprintf(stderr, "V4.1 decode top-512: fused == staged == plain ladder ids, ties included PASS\n");
    return 1;
}
#else
static int check_decode_topk(void) { return 1; }
#endif

/* Rows with at most 512 visible keys select every key; other slots stay. */
static int check_indexer_all(void) {
    const uint32_t rows = 129u;
    ds4_gpu_tensor *selected = upload(NULL, (rows * 512u + 1u) * sizeof(int32_t));
    CHECK(selected);
    int32_t *ids = ds4_gpu_tensor_contents(selected);
    CHECK(ids);
    for (uint32_t ratio = 1; ratio <= 2; ratio++) for (uint32_t start = 0; start <= 300u; start += 300u) {
        for (uint32_t i = 0; i <= rows * 512u; i++) ids[i] = -1;
        CHECK(ds4_gpu_dsv41_indexer_all_batch(selected, rows, start, ratio));
        CHECK(ds4_gpu_synchronize());
        for (uint32_t t = 0; t < rows; t++) {
            const uint32_t visible = (start + t + 1u) / ratio;
            for (uint32_t i = 0; i < 512u; i++)
                CHECK(ids[t * 512u + i] == (i < visible ? (int32_t)i : -1));
        }
        CHECK(ids[rows * 512u] == -1);
    }
    CHECK(!ds4_gpu_dsv41_indexer_all_batch(selected, rows, 512u, 1));
    CHECK(!ds4_gpu_dsv41_indexer_all_batch(selected, rows, 0, 0));
    ds4_gpu_tensor_free(selected);
    fprintf(stderr, "V4.1 indexer select-all rows: exact\n");
    return 1;
}

#ifdef __APPLE__
static int compare_score_entry(const void *a, const void *b) {
    const float *x = *(const float *const *)a, *y = *(const float *const *)b;
    return *x < *y ? 1 : *x > *y ? -1 : 0;
}

/* The per-key and the wide single-row index scoring kernels on the same random
 * query and keys: both against a double oracle, then their top-2048 sets, where
 * any difference must be a near-tie at the 2048th score. */
static int check_index_score_wide(void) {
    enum { HEADS = 32, DIM = 128, TOP = 2048, ROUNDS = 8 };
    const uint32_t counts[] = {4096, 131072};
    const uint32_t max_keys = counts[1];
    ds4_gpu_tensor *qt = upload(NULL, (size_t)HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)HEADS * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)max_keys * DIM * sizeof(float));
    ds4_gpu_tensor *sd = upload(NULL, ((size_t)max_keys + 1) * sizeof(float));
    ds4_gpu_tensor *sw = upload(NULL, ((size_t)max_keys + 1) * sizeof(float));
    CHECK(qt && wt && kt && sd && sw);
    float *q = ds4_gpu_tensor_contents(qt), *w = ds4_gpu_tensor_contents(wt);
    float *k = ds4_gpu_tensor_contents(kt);
    float *direct = ds4_gpu_tensor_contents(sd), *wide = ds4_gpu_tensor_contents(sw);
    const float **order = malloc((size_t)max_keys * sizeof(*order));
    unsigned char *in_direct = calloc(max_keys, 1), *in_wide = calloc(max_keys, 1);
    CHECK(q && w && k && direct && wide && order && in_direct && in_wide);
    for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const uint32_t keys = counts[ci];
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++)
            q[i] = ldexpf(random_value(), (int)((i / 32) % 41) - 20);
        for (size_t i = 0; i < (size_t)keys * DIM; i++)
            k[i] = ldexpf(random_value(), (int)((i / 32) % 41) - 20);
        for (size_t i = 0; i < HEADS; i++) w[i] = bf16(random_value());
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, keys, DS4_V41_FP4_E8M0));
        direct[keys] = wide[keys] = 12345;
        double seconds[2] = {0, 0};
        for (int mode = 0; mode < 2; mode++) {
            ds4_gpu_test_set_flags(mode ? DS4_GPU_TEST_INDEX_SCORE_WIDE : DS4_GPU_TEST_INDEX_SCORE_DIRECT);
            for (int round = 0; round < ROUNDS; round++) {
                const double t0 = monotonic_seconds();
                CHECK(ds4_gpu_glm_indexer_score_one_tensor(mode ? sw : sd, qt, wt, kt,
                    keys, HEADS, DIM, 1.0f / 64.0f, false));
                CHECK(ds4_gpu_synchronize());
                if (round) seconds[mode] += monotonic_seconds() - t0;
            }
        }
        ds4_gpu_test_set_flags(0);
        CHECK(direct[keys] == 12345 && wide[keys] == 12345);
        double worst[2] = {0, 0}, gap = 0;
        for (uint32_t j = 0; j < keys; j++) {
            double expected = 0, magnitude = 0;
            for (uint32_t h = 0; h < HEADS; h++) {
                double dot = 0;
                for (uint32_t d = 0; d < DIM; d++)
                    dot += (double)q[(size_t)h * DIM + d] * k[(size_t)j * DIM + d];
                const double term = fmax(dot / 64.0, 0) * w[h];
                expected += term;
                magnitude += fabs(term);
            }
            CHECK(isfinite(direct[j]) && isfinite(wide[j]));
            worst[0] = fmax(worst[0], fabs(direct[j] - expected) / fmax(magnitude, 1));
            worst[1] = fmax(worst[1], fabs(wide[j] - expected) / fmax(magnitude, 1));
            gap = fmax(gap, fabs((double)direct[j] - wide[j]) / fmax(magnitude, 1));
        }
        CHECK(worst[0] < 0.00001 && worst[1] < 0.00001);
        /* top-2048 membership of both kernels */
        for (uint32_t j = 0; j < keys; j++) order[j] = direct + j;
        qsort(order, keys, sizeof(*order), compare_score_entry);
        const float threshold = *order[TOP - 1];
        memset(in_direct, 0, keys);
        for (uint32_t i = 0; i < TOP; i++) in_direct[order[i] - direct] = 1;
        for (uint32_t j = 0; j < keys; j++) order[j] = wide + j;
        qsort(order, keys, sizeof(*order), compare_score_entry);
        memset(in_wide, 0, keys);
        for (uint32_t i = 0; i < TOP; i++) in_wide[order[i] - wide] = 1;
        uint32_t differing = 0;
        for (uint32_t j = 0; j < keys; j++) {
            if (in_direct[j] == in_wide[j]) continue;
            differing++;
            CHECK(fabs(direct[j] - threshold) <= 0.00001 * fmax(fabs(threshold), 1));
        }
        fprintf(stderr, "V4.1 index score wide keys=%u oracle error direct=%.3g wide=%.3g "
                "kernel gap=%.3g top-%d symmetric difference=%u (near-ties) "
                "us per call direct=%.0f wide=%.0f\n",
                keys, worst[0], worst[1], gap, TOP, differing,
                seconds[0] / (ROUNDS - 1) * 1e6, seconds[1] / (ROUNDS - 1) * 1e6);
    }
    free(in_wide); free(in_direct); free(order);
    ds4_gpu_tensor_free(sw); ds4_gpu_tensor_free(sd); ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(wt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 wide index score: double-precision oracle PASS\n");
    return 1;
}

/* Candidate-only index rows against the wide row they replace (wide scores,
 * candidate filter, radix top-512): the same ids in the same order, with
 * realistic and wide FP4 block scales, partial newest blocks and ties. */
static int check_index_candidates(void) {
    enum { HEADS = 32, DIM = 128, TOP = 512, ROUNDS = 6 };
    const uint32_t widths[] = {16385, 16391, 20007, 65536, 131072, 131075, 262147};
    const uint32_t max_keys = 262147, max_blocks = (max_keys + 7) / 8;
    ds4_gpu_tensor *qt = upload(NULL, (size_t)HEADS * DIM * 4), *wt = upload(NULL, HEADS * 4);
    ds4_gpu_tensor *kt = upload(NULL, (size_t)max_keys * DIM * 4);
    ds4_gpu_tensor *full = upload(NULL, (size_t)max_keys * 4), *plain = upload(NULL, (size_t)max_keys * 4);
    ds4_gpu_tensor *source = upload(NULL, (size_t)max_keys * 4), *cand = upload(NULL, 16384 * 4);
    ds4_gpu_tensor *list = upload(NULL, 2048 * 4), *blocks_t = upload(NULL, (size_t)max_blocks * 4);
    ds4_gpu_tensor *picked = upload(NULL, 2048 * 4), *mask = upload(NULL, (size_t)max_blocks * 4);
    ds4_gpu_tensor *old_sel = upload(NULL, TOP * 4), *new_sel = upload(NULL, TOP * 4);
    CHECK(qt && wt && kt && full && plain && source && cand && list && blocks_t && picked && mask &&
          old_sel && new_sel);
    float *q = ds4_gpu_tensor_contents(qt), *w = ds4_gpu_tensor_contents(wt);
    float *k = ds4_gpu_tensor_contents(kt), *s = ds4_gpu_tensor_contents(source);
    const float *p = ds4_gpu_tensor_contents(plain), *c = ds4_gpu_tensor_contents(cand);
    const float *m = ds4_gpu_tensor_contents(mask);
    const uint32_t *l = ds4_gpu_tensor_contents(list);
    const int32_t *a = ds4_gpu_tensor_contents(old_sel), *b = ds4_gpu_tensor_contents(new_sel);
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7) / 8, count = 16384 - (blocks * 8 - n);
        double seconds[2] = {0, 0};
        uint32_t ties = 0;
        for (int mode = 0; mode < 4; mode++) {
            /* bit 0: block exponents over +-20 instead of +-2; bit 1: ties */
            const int range = mode & 1 ? 20 : 2;
            for (size_t i = 0; i < (size_t)HEADS * DIM; i++)
                q[i] = ldexpf(random_value(), (int)((i / 32) % (2 * range + 1)) - range);
            for (size_t i = 0; i < (size_t)n * DIM; i++)
                k[i] = ldexpf(random_value(), (int)((i / 32 + n) % (2 * range + 1)) - range);
            for (size_t i = 0; i < HEADS; i++) w[i] = random_value() * 0.37139f;
            for (uint32_t i = 0; i < n; i++) s[i] = (float)((i * 7919u + mode) % 1048573u);
            CHECK(ds4_gpu_dsv41_quantize(qt, DIM, HEADS, DS4_V41_FP4_E8M0));
            CHECK(ds4_gpu_dsv41_quantize(kt, DIM, n, DS4_V41_FP4_E8M0));
            CHECK(ds4_gpu_synchronize());
            if (mode & 2) {
                /* copy the best key over 1/16 of all keys: the 512th score ties */
                CHECK(ds4_gpu_dsv41_indexer_score_one_tensor(plain, qt, wt, kt, n, HEADS, DIM,
                    1.0f / 64.0f, false) && ds4_gpu_synchronize());
                uint32_t best = 0;
                for (uint32_t i = 1; i < n; i++) if (p[i] > p[best]) best = i;
                for (uint32_t i = 0; i < n; i += 16)
                    memcpy(k + (size_t)(i ^ 5) % n * DIM, k + (size_t)best * DIM, DIM * 4);
            }
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(blocks_t, source, n, 1, n - 1, 1));
            CHECK(ds4_gpu_dsv41_indexer_topk_tensor(picked, blocks_t, blocks, 1, 2048));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(mask, picked, blocks, 1, 2048));
            CHECK(ds4_gpu_dsv41_indexer_score_one_tensor(plain, qt, wt, kt, n, HEADS, DIM, 1.0f / 64.0f, false));
            CHECK(ds4_gpu_end_commands());
            ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
            for (int round = 0; round < ROUNDS; round++) {
                /* path 1 builds a zeroed list, 2 reuses it, 3 rebuilds it with the decode top-k off */
                for (int path = 0; path < 4; path++) {
                    const uint64_t reused = ds4_gpu_test_kernel_runs("candidate list reused");
                    if (path == 1 || path == 3) memset(ds4_gpu_tensor_contents(list), 0, 2048 * 4);
                    if (path == 3) ds4_gpu_test_set_v41_off(DS4_GPU_V41_DECODE_TOPK);
                    const double t0 = monotonic_seconds();
                    CHECK(ds4_gpu_begin_commands());
                    if (path) {
                        CHECK(ds4_gpu_dsv41_indexer_candidates_tensor(new_sel, cand, list, qt, wt, kt, mask,
                            n, 1.0f / 64.0f, TOP, path >= 2));
                    } else {
                        CHECK(ds4_gpu_dsv41_indexer_score_one_tensor(full, qt, wt, kt, n, HEADS, DIM,
                            1.0f / 64.0f, false));
                        CHECK(ds4_gpu_dsv41_candidate_filter(full, mask, n, 1, n - 1, 1, 0));
                        CHECK(ds4_gpu_dsv41_indexer_topk_tensor(old_sel, full, n, 1, TOP));
                    }
                    CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
                    ds4_gpu_test_set_v41_off(0);
                    if (round && path < 2) seconds[path] += monotonic_seconds() - t0;
                    CHECK(ds4_gpu_test_kernel_runs("candidate list reused") == reused + (path == 2));
                    if (path) CHECK(!memcmp(a, b, TOP * sizeof(*a)));
                }
            }
            /* the list is the mask's candidates in order; each score is the wide one */
            uint32_t at = 0;
            for (uint32_t j = 0; j < blocks; j++) if (m[j] == 0.0f) CHECK(at < 2048 && l[at++] == j);
            CHECK(at == 2048 && l[2047] == blocks - 1);
            for (uint32_t i = 0; i < count; i++) {
                const uint32_t key = l[i / 8] * 8 + i % 8;
                CHECK(key < n && !memcmp(c + i, p + key, sizeof(*c)));
            }
            for (uint32_t i = 0; i < TOP; i++) ties += i && p[a[i]] == p[a[i - 1]];
        }
        fprintf(stderr, "V4.1 candidate index rows keys=%u candidates=%u: ids, order and scores "
                "identical (%u tied neighbours) us per row full=%.0f candidates=%.0f\n",
                n, count, ties, seconds[0] / (4 * (ROUNDS - 1)) * 1e6, seconds[1] / (4 * (ROUNDS - 1)) * 1e6);
    }
    {   /* the engine's record (the list was built on this mask): a row with another mask rebuilds the list */
        const uint32_t n = max_keys, blocks = max_blocks;
        ds4_gpu_tensor *masks[2] = {mask, upload(NULL, (size_t)max_blocks * 4)};
        CHECK(masks[1]);
        int32_t ref[2][TOP];
        for (int i = 0; i < 2; i++) {
            for (uint32_t j = 0; j < n; j++) s[j] = (float)((j * (7919u + 104u * (uint32_t)i) + (uint32_t)i) % 1048573u);
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(blocks_t, source, n, 1, n - 1, 1));
            CHECK(ds4_gpu_dsv41_indexer_topk_tensor(picked, blocks_t, blocks, 1, 2048));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(masks[i], picked, blocks, 1, 2048));
            CHECK(ds4_gpu_dsv41_indexer_score_one_tensor(full, qt, wt, kt, n, HEADS, DIM, 1.0f / 64.0f, false));
            CHECK(ds4_gpu_dsv41_candidate_filter(full, masks[i], n, 1, n - 1, 1, 0));
            CHECK(ds4_gpu_dsv41_indexer_topk_tensor(old_sel, full, n, 1, TOP));
            CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
            memcpy(ref[i], a, sizeof(ref[i]));
        }
        CHECK(memcmp(ref[0], ref[1], sizeof(ref[0])));
        const ds4_gpu_tensor *record = NULL;
        const int steps[] = {0, 0, 1, 1, 0};
        for (size_t i = 0; i < sizeof(steps) / sizeof(*steps); i++) {
            const ds4_gpu_tensor *m_i = masks[steps[i]];
            const uint64_t reused = ds4_gpu_test_kernel_runs("candidate list reused");
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_indexer_candidates_tensor(new_sel, cand, list, qt, wt, kt, m_i, n, 1.0f / 64.0f, TOP,
                record == m_i));
            CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
            CHECK(ds4_gpu_test_kernel_runs("candidate list reused") == reused + (record == m_i));
            CHECK(!memcmp(b, ref[steps[i]], sizeof(ref[0])));
            record = m_i;
        }
        ds4_gpu_tensor_free(masks[1]);
    }
    CHECK(!ds4_gpu_dsv41_indexer_candidates_tensor(new_sel, cand, list, qt, wt, kt, mask,
        16384, 1.0f / 64.0f, TOP, 0));
    ds4_gpu_tensor *tensors[] = {qt, wt, kt, full, plain, source, cand, list, blocks_t, picked, mask,
                                 old_sel, new_sel};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    return 1;
}

/* The bf16 key view and its scorers against the wide scorer on the f32 cache: the same bits for every key. Keys pass
 * the FP4_E8M0 quantizer with block scales spread over 41 exponents and over 12 (where summation order shows in the
 * bits); the view converts in two pieces at an odd boundary. The decode row entry reads the view where the wide
 * scorer runs and scores the f32 keys without it. */
static int check_index_score_bf16(void) {
    enum { HEADS = 32, DIM = 128 };
    const uint32_t counts[] = {8200, 65535, 65536, 98303, 131077};
    const uint32_t max_keys = 131077;
    const char *kernels[] = {"kernel_dsv41_indexer_score_rows_split4_bf16", "kernel_dsv41_indexer_score_rows_split2_bf16",
                             "kernel_dsv41_indexer_score_bf16"};
    ds4_gpu_tensor *qt = upload(NULL, (size_t)HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)HEADS * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)max_keys * DIM * sizeof(float));
    ds4_gpu_tensor *vt = upload(NULL, (size_t)max_keys * DIM * sizeof(uint16_t));
    ds4_gpu_tensor *sw = upload(NULL, ((size_t)max_keys + 1) * sizeof(float));
    ds4_gpu_tensor *sb = upload(NULL, ((size_t)max_keys + 1) * sizeof(float));
    ds4_gpu_tensor *ft = upload(NULL, sizeof(uint32_t));
    CHECK(qt && wt && kt && vt && sw && sb && ft);
    float *q = ds4_gpu_tensor_contents(qt), *w = ds4_gpu_tensor_contents(wt);
    float *k = ds4_gpu_tensor_contents(kt);
    float *wide = ds4_gpu_tensor_contents(sw), *fast = ds4_gpu_tensor_contents(sb);
    uint32_t *flag = ds4_gpu_tensor_contents(ft);
    for (int spread = 0; spread < 2; spread++) {
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++)
            q[i] = spread == 0 ? ldexpf(random_value(), (int)((i / 32) % 41) - 20) :
                                 ldexpf(random_value(), (int)((i / 32 * 7) % 13) - 6);
        for (size_t i = 0; i < HEADS; i++) w[i] = bf16(random_value());
        for (size_t i = 0; i < (size_t)max_keys * DIM; i++)
            k[i] = spread == 0 ? ldexpf(random_value(), (int)((i / 32) % 41) - 20) :
                                 ldexpf(random_value(), (int)((i / 32 * 5 + i / DIM) % 13) - 6);
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, max_keys, DS4_V41_FP4_E8M0));
        *flag = 0;
        CHECK(ds4_gpu_dsv41_index_keys_bf16(vt, kt, 0, 77777, ft));
        CHECK(ds4_gpu_dsv41_index_keys_bf16(vt, kt, 77777, max_keys - 77777, ft));
        CHECK(ds4_gpu_synchronize() && *flag == 0);
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t keys = counts[ci];
            const char *kernel = kernels[keys < 10240u ? 0 : keys < 65536u ? 1 : 2];
            wide[keys] = fast[keys] = 12345;
            ds4_gpu_test_set_flags(DS4_GPU_TEST_INDEX_SCORE_WIDE);
            CHECK(ds4_gpu_glm_indexer_score_one_tensor(sw, qt, wt, kt, keys, HEADS, DIM, 1.0f / 64.0f, false));
            ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
            for (int view = 1; view >= 0; view--) {
                const uint64_t runs = ds4_gpu_test_kernel_runs(kernel);
                memset(fast, 0xff, (size_t)keys * sizeof(float));
                CHECK(ds4_gpu_dsv41_indexer_score_row(sb, qt, wt, kt, view ? vt : NULL, keys) && ds4_gpu_synchronize());
                CHECK(ds4_gpu_test_kernel_runs(kernel) == runs + (uint64_t)view);
                CHECK(!memcmp(wide, fast, (size_t)keys * sizeof(float)) && fast[keys] == 12345);
            }
            uint32_t positive = 0, rounded = 0;
            for (uint32_t j = 0; j < keys; j++) positive += wide[j] > 0;
            /* sensitivity: keys whose f32 score is not the exact sum, so order shows */
            for (uint32_t j = 0; ci == 0 && j < 2048; j++) {
                double expected = 0;
                for (uint32_t h = 0; h < HEADS; h++) {
                    double dot = 0;
                    for (uint32_t d = 0; d < DIM; d++)
                        dot += (double)q[(size_t)h * DIM + d] * k[(size_t)j * DIM + d];
                    expected += fmax(dot / 64.0, 0) * w[h];
                }
                rounded += (double)wide[j] != expected;
            }
            CHECK(wide[keys] == 12345 && positive > keys / 16 && (ci != 0 || rounded > 0));
        }
    }
    /* a key that is not bf16-exact raises the flag */
    uint32_t bits;
    memcpy(&bits, &k[(size_t)123456 * DIM + 77], 4);
    bits |= 1u;
    memcpy(&k[(size_t)123456 * DIM + 77], &bits, 4);
    CHECK(ds4_gpu_dsv41_index_keys_bf16(vt, kt, 123456, 1, ft) && ds4_gpu_synchronize() && *flag != 0);
    ds4_gpu_tensor_free(ft); ds4_gpu_tensor_free(sb); ds4_gpu_tensor_free(sw);
    ds4_gpu_tensor_free(vt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(wt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 bf16 index score (split4, split2, full) == wide scorer bits PASS\n");
    return 1;
}

/* The FP8 copy reader (ds4_gpu_matmul_e4m3_bf16_tensor), with its blend kernels and with the narrow readers, gives
 * the BF16-rounded F16 projection's bits at the Engram (6144 -> 256, two-row blend) and indexer q_b (1280 -> 4096,
 * block-owner blend) shapes: power-of-two 32x32 block scales 2^-15..2^-8 (every weight exact in F16), normal and
 * small codes. The BF16 router's blend reader (5120 -> 384) gives the narrow reader's bits. */
static int check_e4m3_copies(void) {
    const uint32_t ins[2] = {6144, 1280}, outs[2] = {256, 4096};
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (int which = 0; which < 2; which++) {
        const uint32_t IN = ins[which], OUT = outs[which];
        const char *blend = which ? "kernel_e4m3_blend_owner_r2_bf16" : "kernel_e4m3_blend_r2_bf16";
        const size_t f16_bytes = (size_t)IN * OUT * 2, code_bytes = (size_t)IN * OUT;
        const size_t nscale = (size_t)(IN / 32) * (OUT / 32), page = (size_t)getpagesize();
        const size_t map_bytes = (f16_bytes + page - 1) / page * page;
        unsigned char *map = NULL, *codes = malloc(code_bytes);
        float *scales = malloc(nscale * sizeof(float)), *x = malloc((size_t)IN * sizeof(float));
        CHECK(posix_memalign((void **)&map, page, map_bytes) == 0 && codes && scales && x);
        memset(map, 0, map_bytes);
        _Float16 *w16 = (_Float16 *)map;
        CHECK(ds4_gpu_set_model_map(map, map_bytes));
        for (uint32_t i = 0; i < IN; i++) x[i] = random_value() / 4;
        ds4_gpu_tensor *xt = upload(x, (size_t)IN * sizeof(float));
        ds4_gpu_tensor *o16 = upload(NULL, OUT * 4), *o8 = upload(NULL, OUT * 4), *o8n = upload(NULL, OUT * 4);
        CHECK(xt && o16 && o8 && o8n);
        for (int pass = 0; pass < 2; pass++) {
            for (size_t i = 0; i < nscale; i++) { (void)random_value(); scales[i] = ldexpf(1.0f, -15 + (int)(seed % 8u)); }
            for (size_t i = 0; i < code_bytes; i++) {
                (void)random_value();
                unsigned char c = (unsigned char)(seed >> 3);
                if ((c & 0x7f) == 0x7f) c ^= 1;   /* no NaN codes */
                if (pass) c = (unsigned char)(c & 0x87);   /* subnormal and small codes */
                codes[i] = c;
            }
            for (size_t r = 0; r < OUT; r++)
                for (size_t k = 0; k < IN; k++) {
                    const unsigned char c = codes[r * IN + k], a = c & 0x7f;
                    const uint32_t ex = (a >> 3) & 15u, m = a & 7u;
                    float v = a == 0 ? 0.0f : ex == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, (int)ex - 7);
                    if (c & 0x80) v = -v;
                    w16[r * IN + k] = (_Float16)(v * scales[(r / 32) * (IN / 32) + k / 32]);
                }
            ds4_gpu_tensor *ct = upload(codes, code_bytes), *st = upload(scales, nscale * sizeof(float));
            CHECK(ct && st);
            const uint64_t runs = ds4_gpu_test_kernel_runs(blend);
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_project_f16_bf16(o16, map, map_bytes, 0, IN, OUT, xt));
            CHECK(ds4_gpu_matmul_e4m3_bf16_tensor(o8, ct, st, IN, OUT, xt));
            ds4_gpu_test_set_v41_off(DS4_GPU_V41_PLAIN_READERS);
            const int ok = ds4_gpu_matmul_e4m3_bf16_tensor(o8n, ct, st, IN, OUT, xt);
            ds4_gpu_test_set_v41_off(0);
            CHECK(ok && ds4_gpu_end_commands() && ds4_gpu_synchronize());
            CHECK(ds4_gpu_test_kernel_runs(blend) == runs + 1);
            CHECK(!memcmp(ds4_gpu_tensor_contents(o16), ds4_gpu_tensor_contents(o8), OUT * sizeof(float)));
            CHECK(!memcmp(ds4_gpu_tensor_contents(o16), ds4_gpu_tensor_contents(o8n), OUT * sizeof(float)));
            ds4_gpu_tensor_free(ct); ds4_gpu_tensor_free(st);
        }
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(o16); ds4_gpu_tensor_free(o8); ds4_gpu_tensor_free(o8n);
        free(codes); free(scales); free(x);   /* map stays: the model views may wrap it */
    }
    {   /* BF16 router 5120 -> 384 */
        const uint32_t in = 5120, out = 384;
        ds4_gpu_tensor *wt = upload(NULL, (size_t)in * out * 2), *xt = upload(NULL, (size_t)in * 4), *o = upload(NULL, out * 4);
        CHECK(wt && xt && o);
        uint16_t *w = ds4_gpu_tensor_contents(wt);
        float *x = ds4_gpu_tensor_contents(xt);
        for (size_t i = 0; i < (size_t)in * out; i++) {
            const float v = random_value() / 64;
            uint32_t bits;
            memcpy(&bits, &v, 4);
            w[i] = (uint16_t)(bits >> 16);
        }
        for (uint32_t i = 0; i < in; i++) x[i] = random_value();
        LITERAL_PAIR(DS4_GPU_V41_PLAIN_READERS, "kernel_router_blend_r1", o, out * 4,
                     ds4_gpu_matmul_bf16_weights_tensor(o, wt, in, out, xt));
        ds4_gpu_tensor_free(wt); ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(o);
    }
    fprintf(stderr, "V4.1 FP8 copy reader (blend and narrow) == F16 projection; router blend == narrow PASS\n");
    return 1;
}
#else
static int check_index_score_wide(void) { return 1; }
static int check_index_candidates(void) { return 1; }
static int check_index_score_bf16(void) { return 1; }
static int check_e4m3_copies(void) { return 1; }
#endif

static int check_compact_carry(void) {
    const uint32_t widths[] = {1, 31, 32, 33, 127, 128, 129, 20480};
    const uint32_t rows = 129, offset = 2;
    for (uint32_t format = 0; format <= DS4_V41_CARRY_MASK; format++) {
        for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
            const uint32_t width = widths[wi];
            const uint32_t words = format == DS4_V41_CARRY_BF16 ? (width + 1) / 2 : (width + 31) / 32;
            const size_t count = (size_t)width * rows;
            const size_t packed_bytes = (size_t)(rows + 4) * words * sizeof(uint32_t);
            ds4_gpu_tensor *plain = upload(NULL, (count + 1) * sizeof(float));
            ds4_gpu_tensor *packed = upload(NULL, packed_bytes);
            uint32_t *expected = malloc(count * sizeof(uint32_t));
            CHECK(plain && packed && expected);
            uint32_t *bits = ds4_gpu_tensor_contents(plain);
            unsigned char *storage = ds4_gpu_tensor_contents(packed);
            for (size_t i = 0; i < count; i++) {
                const uint32_t b = (uint32_t)(i * 40503u + 32768u) & 0xffffu;
                expected[i] = format == DS4_V41_CARRY_BF16 ? b << 16 : (b & 1u ? 0xff800000u : 0u);
            }
            memcpy(bits, expected, count * sizeof(uint32_t));
            bits[count] = 0x12345678u;
            memset(storage, 0xa5, packed_bytes);
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, true));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)));
            memset(bits, 0, count * sizeof(uint32_t));
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, false));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)) && bits[count] == 0x12345678u);
            for (size_t i = 0; i < (size_t)offset * words * 4u; i++) CHECK(storage[i] == 0xa5);
            for (size_t i = (size_t)(offset + rows) * words * 4u; i < packed_bytes; i++)
                CHECK(storage[i] == 0xa5);
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, UINT32_MAX, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset + 3, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, 0, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, 0, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, DS4_V41_CARRY_F32, true));
            fprintf(stderr, "V4.1 compact carry format=%u width=%u: exact round-trip and bounds PASS\n", format, width);
            free(expected);
            ds4_gpu_tensor_free(plain);
            ds4_gpu_tensor_free(packed);
        }
    }
    return 1;
}

/* Exercise the full-head and compact TP layouts with the same causal keys.
 * Selected rows are shuffled; include masked future rows at odd frontiers. */
static int check_tp_attention(void) {
    enum { D = 512, H = 64, K = 512, C = 2048 };
    const struct { uint32_t rows, selected; } cases[] = {
        {1, K}, {31, K}, {32, K}, {33, K}, {129, K}, {257, K}, {2048, K},
        {1, 0}, {1, 1}, {2, K}, {6, K}, {8, K},
        {2, 0}, {2, 1}, {6, 0}
    };
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < H; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    ds4_gpu_set_quality(false);
    for (unsigned s = 0; s < sizeof(cases) / sizeof(*cases); s++) {
        const uint32_t n = cases[s].rows, nr = n + 127, start = 3073;
        const uint32_t ratio = 1u + s % 2u;
        const size_t nq = (size_t)n * H * D;
        float *q = malloc(nq * 4), *actual = malloc(nq * 4);
        float *compact = malloc(nq * 2), *part = malloc(nq * 2);
        float *raw = malloc((size_t)nr * D * 4), *comp = malloc(C * D * 4);
        int32_t *ids = malloc((size_t)n * K * 4);
        CHECK(q && actual && compact && part && raw && comp && ids);
        for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < nr * D; i++) raw[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < C * D; i++) comp[i] = bf16(random_value() / 4);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t j = 0; j < K; j++)
                ids[t * K + j] = j < cases[s].selected &&
                    (cases[s].selected != K || j % 29)
                    ? (int32_t)((j * 127u + t * 17u) % C) : -1;
        ds4_gpu_tensor *qt = upload(q, nq * 4), *rt = upload(raw, (size_t)nr * D * 4);
        ds4_gpu_tensor *ct = upload(comp, C * D * 4), *it = upload(ids, (size_t)n * K * 4);
        ds4_gpu_tensor *out = upload(NULL, nq * 4), *qp = upload(NULL, nq * 2);
        ds4_gpu_tensor *op = upload(NULL, nq * 2);
        CHECK(qt && rt && ct && it && out && qp && op);
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out, sinks, getpagesize(),
            0, qt, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H, D));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, nq * 4));
        double max_split = 0, max_oracle = 0;
        for (uint32_t rank = 0; rank < 2; rank++) {
            for (uint32_t t = 0; t < n; t++)
                memcpy(compact + (size_t)t * H/2 * D,
                    q + ((size_t)t * H + rank * H/2) * D, H/2 * D * 4);
            CHECK(ds4_gpu_tensor_write(qp, 0, compact, nq * 2));
            CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
            CHECK(ds4_gpu_tensor_read(op, 0, part, nq * 2));
#ifdef __APPLE__
            if (n > 1) {
                ds4_gpu_set_quality(true); /* Existing eight-head kernel. */
                CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                    rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
                CHECK(!memcmp(part, ds4_gpu_tensor_contents(op), nq * 2));
                ds4_gpu_set_quality(false);
            }
#endif
            if (n == 2048 && rank == 0) {
                const bool controls[] = {true, false, false, true};
                for (unsigned pass = 0; pass < 4; pass++) {
                    ds4_gpu_set_quality(controls[pass]);
                    const double begin = monotonic_seconds();
                    for (int repeat = 0; repeat < 4; repeat++)
                        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op,
                            sinks, getpagesize(), 0, qp, rt, ct, 0, it, n, start,
                            nr, nr, 0, C, K, 128, ratio, H/2, D));
                    CHECK(ds4_gpu_synchronize());
                    fprintf(stderr, "TP indexed attention control=%u %.3f ms\n",
                        controls[pass], (monotonic_seconds() - begin) * 250);
                }
                ds4_gpu_set_quality(false);
            }
            for (uint32_t t = 0; t < n; t++) {
                for (uint32_t h = 0; h < H/2; h++) {
                    const float *got = part + ((size_t)t * H/2 + h) * D;
                    const float *ref = actual + ((size_t)t * H + rank * H/2 + h) * D;
                    for (int d = 0; d < D; d++) {
                        const double err = fabs((double)got[d] - ref[d]);
                        max_split = fmax(max_split, err);
                        CHECK(isfinite(got[d]) && err < 3e-5 * (1 + fabs(ref[d])));
                    }
                    if ((t != 0 && t != n/2 && t + 1 != n) || h % 15) continue;
                    const float *query = compact + ((size_t)t * H/2 + h) * D;
                    double logits[128 + K + 1], values[128 + K + 1];
                    const uint32_t col = (t * 71u + h * 19u) % D;
                    unsigned count = 0;
                    for (uint32_t j = 0; j < 128 + K; j++) {
                        const int32_t id = j < 128 ? (int32_t)(t + j) : ids[t * K + j - 128];
                        if (j >= 128 && (uint32_t)id >= (start + t + 1) / ratio) continue;
                        const float *key = (j < 128 ? raw : comp) + (size_t)id * D;
                        double dot = 0;
                        for (int d = 0; d < D; d++) dot += (double)query[d] * key[d];
                        logits[count] = dot / sqrt(512.0);
                        values[count++] = key[col];
                    }
                    logits[count] = sinks[rank * H/2 + h]; values[count++] = 0;
                    double max = -INFINITY, den = 0, num = 0;
                    for (unsigned j = 0; j < count; j++) max = fmax(max, logits[j]);
                    for (unsigned j = 0; j < count; j++) {
                        const double p = exp(logits[j] - max);
                        den += p; num += p * values[j];
                    }
                    const double err = fabs(got[col] - num / den);
                    max_oracle = fmax(max_oracle, err);
                    CHECK(err < 3e-5 * (1 + fabs(num / den)));
                }
            }
        }
        fprintf(stderr, "V4.1 TP attention rows=%u selected=%u ratio=%u max split=%g oracle=%g: PASS\n",
            n, cases[s].selected, ratio, max_split, max_oracle);
        ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(rt); ds4_gpu_tensor_free(ct);
        ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(qp);
        ds4_gpu_tensor_free(op);
        free(q); free(actual); free(compact); free(part); free(raw); free(comp); free(ids);
    }
    ds4_gpu_cleanup();
    free(sinks);
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--embedding")) {
        const int ok = ds4_gpu_init() && check_embedding();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--router")) {
        const int ok = ds4_gpu_init() && check_router();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--tp-attention")) {
        const int ok = ds4_gpu_init() && check_tp_attention();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && (!strcmp(argv[1], "--attention-output") ||
                      !strcmp(argv[1], "--attention-output-large"))) {
        const int ok = ds4_gpu_init() && check_attention_output(
            !strcmp(argv[1], "--attention-output-large"));
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--hc-scaled")) {
        const int ok = ds4_gpu_init() && check_hc_scaled();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--bf16-linear")) {
        const int ok = ds4_gpu_init() && check_bf16_linear();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--compact-carry")) {
        const int ok = ds4_gpu_init() && check_compact_carry();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--rope-quantize")) {
        const int ok = ds4_gpu_init() && check_rope_quantize();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--deferred-hc")) {
        const int ok = ds4_gpu_init() && check_deferred_hc();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-scores")) {
        const int ok = ds4_gpu_init() && check_indexer_batch();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-score-wide")) {
        const int ok = ds4_gpu_init() && check_index_score_wide();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-candidates")) {
        const int ok = ds4_gpu_init() && check_index_candidates();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-topk")) {
        const int ok = ds4_gpu_init() && check_causal_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--topk-select")) {
        const int ok = ds4_gpu_init() && check_topk_select();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--general-topk")) {
        const int ok = ds4_gpu_init() && check_general_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-projection")) {
        const int ok = ds4_gpu_init() && check_index_projection();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-score-bf16")) {
        const int ok = ds4_gpu_init() && check_index_score_bf16();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--e4m3-copies")) {
        const int ok = ds4_gpu_init() && check_e4m3_copies();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--engram")) {
        const int ok = ds4_gpu_init() && check_engram();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--rope-tab")) {
        const int ok = ds4_gpu_init() && check_rope_tab();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--literal-shapes")) {
        const int ok = ds4_gpu_init() && check_literal_shapes();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--decode-topk")) {
        const int ok = ds4_gpu_init() && check_decode_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--split-collapse")) {
        const int ok = ds4_gpu_init() && check_split_collapse();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc != 1) return 2;
    int ok = ds4_gpu_init() && check_router() && check_quantization() && check_engram() && check_rope_stride() &&
             check_rope_quantize() && check_deferred_hc() && check_pool() && check_candidates() && check_candidates_batch(16449, 17, 16400) &&
             check_candidates_batch(17017, 17, 17000) && check_sparse_gather() && check_indexer_batch() &&
             check_index_score_wide() && check_index_candidates() && check_embedding() && check_index_projection() && check_general_topk() &&
             check_topk_select() && check_causal_topk() && check_indexer_all() && check_compact_carry() &&
             check_attention_output(false) && check_tp_attention();
    ok = ok && check_literal_shapes();
    ok = ok && check_e4m3_copies();
    ok = ok && check_rope_tab();
    ok = ok && check_split_collapse();
    ok = ok && check_index_score_bf16();
    ok = ok && check_decode_topk();
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
