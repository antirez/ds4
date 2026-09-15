/* ROCm V4.1 production-shape correctness harness. CPU oracles derive from
 * test_deepseek41_metal.c; device access is exclusively explicit copies.
 * The parent process never initializes HIP: each shape execs a fresh child,
 * stops at the first failure, and synchronizes after every risky stage. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)
#define RUN(x) do { fprintf(stderr, "stage: %s\n", #x); CHECK(x); CHECK(sync_guards()); } while (0)
static unsigned requested_shape;

enum { GUARD_BYTES = 64 };
typedef struct allocation {
    ds4_gpu_tensor *storage, *view;
    size_t bytes;
    struct allocation *next;
} allocation;
static allocation *allocations;

static int sync_guards(void) {
    CHECK(ds4_gpu_synchronize());
    for (allocation *a = allocations; a; a = a->next) {
        unsigned char before[GUARD_BYTES], after[GUARD_BYTES];
        CHECK(ds4_gpu_tensor_read(a->storage, 0, before, sizeof(before)));
        CHECK(ds4_gpu_tensor_read(a->storage, GUARD_BYTES + a->bytes, after, sizeof(after)));
        for (unsigned i = 0; i < GUARD_BYTES; i++) CHECK(before[i] == 0xa5 && after[i] == 0xa5);
    }
    return 1;
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    allocation *a = calloc(1, sizeof(*a));
    if (!a || bytes > SIZE_MAX - 2 * GUARD_BYTES) { free(a); return NULL; }
    a->bytes = bytes;
    a->storage = ds4_gpu_tensor_alloc(bytes + 2 * GUARD_BYTES);
    if (!a->storage) { free(a); return NULL; }
    unsigned char guard[GUARD_BYTES];
    memset(guard, 0xa5, sizeof(guard));
    if (!ds4_gpu_tensor_write(a->storage, 0, guard, sizeof(guard)) ||
        !ds4_gpu_tensor_write(a->storage, GUARD_BYTES + bytes, guard, sizeof(guard))) goto fail;
    a->view = ds4_gpu_tensor_view(a->storage, GUARD_BYTES, bytes);
    if (!a->view || (data && !ds4_gpu_tensor_write(a->view, 0, data, bytes))) goto fail;
    a->next = allocations;
    allocations = a;
    return a->view;
fail:
    ds4_gpu_tensor_free(a->view);
    ds4_gpu_tensor_free(a->storage);
    free(a);
    return NULL;
}

static void guarded_free(ds4_gpu_tensor *t) {
    for (allocation **link = &allocations; *link; link = &(*link)->next) {
        allocation *a = *link;
        if (a->view == t) {
            *link = a->next;
            ds4_gpu_tensor_free(a->view);
            ds4_gpu_tensor_free(a->storage);
            free(a);
            return;
        }
    }
    ds4_gpu_tensor_free(t);
}
#define ds4_gpu_tensor_free guarded_free

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

static int check_quantization(void) {
    enum { WIDTH = 512, ROWS = 33, N = WIDTH * ROWS };
    float *source = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    CHECK(source && actual);
    ds4_gpu_tensor *t = upload(NULL, N * sizeof(float));
    CHECK(t);
    for (int mode = 0; mode < 4; mode++) {
        if ((unsigned)mode != requested_shape) continue;
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i++) source[i] = random_value() * (1u << ((i / block) % 4));
        for (int i = 0; i < WIDTH; i++) source[i] = copysignf(0.0f, i & 1 ? -1.0f : 1.0f);
        CHECK(ds4_gpu_tensor_write(t, 0, source, N * sizeof(float)));
        RUN(ds4_gpu_dsv41_quantize(t, WIDTH, ROWS, (ds4_v41_activation_format)mode));
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
    RUN(ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_FP8_E8M0));
    CHECK(!ds4_gpu_dsv41_quantize(t, UINT32_MAX, UINT32_MAX, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 32, 1, (ds4_v41_activation_format)4));
    ds4_gpu_tensor_free(t);
    free(source); free(actual);
    fprintf(stderr, "V4.1 BF16/FP8/FP4 round trips: exact\n");
    return 1;
}

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
        RUN(ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, masked ? mt : NULL, D, ROWS, 1e-20f));
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
    CHECK(!ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, mt, D, ROWS + 1, 1e-20f));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qwt);
    ds4_gpu_tensor_free(kwt); ds4_gpu_tensor_free(mt);
    free(x); free(actual); free(kv); free(qw); free(kw);
    return 1;
}

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
        if (c != requested_shape) continue;
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
            RUN(ds4_gpu_dsv41_pool2(o, k, s, pk, ps, D, n, start));
            ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(o);
            start += n;
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(out, 0, got, PAIRS * D * sizeof(float)));
        {
            for (int p = 0; p < PAIRS; p++) for (int i = 0; i < D; i++) {
                const int a = 2 * p * D + i, b = a + D;
                const double gate = 1 / (1 + exp((double)scores[b] - scores[a]));
                const float expected = bf16((float)(kv[a] * gate + kv[b] * (1 - gate)));
                CHECK(isfinite(got[p * D + i]));
                CHECK(fabsf(got[p * D + i] - expected) <= fmaxf(1e-6f, fabsf(expected) / 128));
            }
        }
        float tail[D];
        CHECK(ds4_gpu_tensor_read(pk, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, kv + (ROWS - 1) * D, sizeof(tail)));
        CHECK(ds4_gpu_tensor_read(ps, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, scores + (ROWS - 1) * D, sizeof(tail)));
    }
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(ps); ds4_gpu_tensor_free(out);
    free(kv); free(scores); free(got); free(reference);
    fprintf(stderr, "V4.1 pair pooling: independent full-output and odd-tail oracle\n");
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
        if (wi != requested_shape) continue;
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
            RUN(ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, start, ratio));
            RUN(ds4_gpu_indexer_topk_tensor(t, b, blocks, rows, top));
            RUN(ds4_gpu_dsv4_topk_mask_tensor(m, t, blocks, rows, top));
            RUN(ds4_gpu_dsv41_candidate_filter(s, m, n, rows, start, ratio));
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
        CHECK(!ds4_gpu_dsv41_candidate_filter(s, m, n, rows + 1, 0, 1));
        ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(t); ds4_gpu_tensor_free(m);
        free(scores); free(got); free(maxima); free(sorted); free(kept);
    }
    fprintf(stderr, "V4.1 causal candidate blocks and filtering: exact\n");
    return 1;
}



static int check_bf16_boundaries(void) {
    static const uint32_t widths[] = {1, 3, 24, 31, 32, 33, 129, 1023, 1024, 1025, 5120, 20480, 65536u * 5u};
    const uint32_t width = widths[requested_shape], rows = width > 20480 ? 1 : 3;
    const size_t count = (size_t)width * rows;
    uint32_t *input = malloc(count * 4), *actual = malloc(count * 4);
    CHECK(input && actual);
    const uint32_t low[] = {0, 0x7fff, 0x8000, 0x8001, 0xffff};
    for (size_t i = 0; i < count; i++) input[i] = ((uint32_t)(i / 5u) << 16) | low[i % 5u];
    ds4_gpu_tensor *t = upload(input, count * 4);
    CHECK(t);
    RUN(ds4_gpu_dsv41_quantize(t, width, rows, DS4_V41_BF16));
    CHECK(ds4_gpu_tensor_read(t, 0, actual, count * 4));
    for (size_t i = 0; i < count; i++) {
        uint32_t bits = input[i];
        if ((bits & 0x7f800000u) != 0x7f800000u) bits += 0x7fffu + ((bits >> 16u) & 1u);
        CHECK(actual[i] == (bits & 0xffff0000u));
    }
    ds4_gpu_tensor_free(t); free(actual); free(input);
    return 1;
}

static int check_rope(void) {
    enum { WIDTH = 512, HEADS = 2, ROWS = 129, COUNT = WIDTH * HEADS * ROWS };
    const uint32_t starts[] = {0, 126, 32766, 1048318};
    const uint32_t start = starts[requested_shape % 4u];
    const bool compressed = (requested_shape / 4u) & 1u;
    const bool inverse = requested_shape / 8u;
    float *input = malloc(COUNT * 4), *got = malloc(COUNT * 4);
    CHECK(input && got);
    for (size_t i = 0; i < COUNT; i++) input[i] = bf16(random_value());
    ds4_gpu_tensor *t = upload(input, COUNT * 4);
    CHECK(t);
    RUN(ds4_gpu_dsv41_rope_stride(t, WIDTH, HEADS, ROWS, start, 2, compressed, inverse));
    CHECK(ds4_gpu_tensor_read(t, 0, got, COUNT * 4));
    const double pi = acos(-1.0);
    const float base = compressed ? 160000.0f : 10000.0f;
    const float low = floor(64.0 * log(65536.0 / (32.0 * 2.0 * pi)) / (2.0 * log(base)));
    const float high = ceil(64.0 * log(65536.0 / (2.0 * pi)) / (2.0 * log(base)));
    size_t rounded = 0;
    double error2 = 0, norm2 = 0;
    for (uint32_t row = 0; row < ROWS; row++) for (uint32_t head = 0; head < HEADS; head++) {
        const size_t off = ((size_t)row * HEADS + head) * WIDTH;
        CHECK(!memcmp(got + off, input + off, (WIDTH - 64) * 4));
        for (uint32_t i = 0; i < 32; i++) {
            /* Match the released reference's F32 frequency and phase arithmetic;
             * evaluate trig independently on the CPU, then round outputs to BF16. */
            float frequency = 1.0f / powf(base, (float)i / 32.0f);
            if (compressed) {
                const float ramp = fminf(1, fmaxf(0, (i - low) / (high - low)));
                const float smooth = 1.0f - ramp;
                frequency = (frequency / 16.0f) * (1.0f - smooth) + frequency * smooth;
            }
            const float phase = (float)(start + row * 2u) * frequency;
            const float c = cosf(phase), si = (inverse ? -1.0f : 1.0f) * sinf(phase);
            const size_t at = off + WIDTH - 64 + i * 2;
            const float expected[] = {bf16(input[at] * c - input[at + 1] * si),
                                      bf16(input[at] * si + input[at + 1] * c)};
            for (unsigned j = 0; j < 2; j++) {
                const double error = (double)got[at + j] - expected[j];
                CHECK(isfinite(got[at + j]));
                CHECK(fabs(error) <= fmax(1e-6, fabs(expected[j]) / 128));
                rounded += got[at + j] != expected[j];
                error2 += error * error; norm2 += (double)expected[j] * expected[j];
            }
        }
    }
    CHECK(sqrt(error2 / fmax(norm2, 1e-30)) < 1e-4);
    fprintf(stderr, "RoPE start=%u compressed=%u inverse=%u BF16 differences=%zu relative_RMS=%.9g\n",
            start, compressed, inverse, rounded, sqrt(error2 / fmax(norm2, 1e-30)));
    CHECK(!ds4_gpu_dsv41_rope_stride(t, WIDTH, HEADS, ROWS, 0, 0, compressed, inverse));
    CHECK(!ds4_gpu_dsv41_rope_stride(t, WIDTH, HEADS, ROWS, 1048320, 2, compressed, inverse));
    CHECK(!ds4_gpu_dsv41_rope_stride(t, WIDTH, HEADS, ROWS + 1, 0, 2, compressed, inverse));
    ds4_gpu_tensor_free(t); free(got); free(input);
    return 1;
}

static int check_sparse_gather(void) {
    const uint32_t sizes[] = {1, 3, 511, 512, 513, 8193, 17017};
    const uint32_t rows = sizes[requested_shape], selected = rows < 512 ? rows : 512;
    const size_t count = (size_t)rows * 512, outputs = (size_t)selected * 512;
    float *input = malloc(count * 4), *result = malloc(outputs * 4);
    int32_t *indices = malloc(selected * 4);
    CHECK(input && result && indices);
    for (uint32_t r = 0; r < rows; r++) for (uint32_t c = 0; c < 512; c++)
        input[(size_t)r * 512 + c] = (float)r + (float)c / 512.0f;
    for (uint32_t r = 0; r < selected; r++) indices[r] = (int32_t)(rows - 1u - r);
    ds4_gpu_tensor *source = upload(input, count * 4), *ids = upload(indices, selected * 4);
    ds4_gpu_tensor *out = upload(NULL, outputs * 4);
    CHECK(source && ids && out);
    RUN(ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected));
    CHECK(ds4_gpu_tensor_read(out, 0, result, outputs * 4));
    for (uint32_t r = 0; r < selected; r++)
        CHECK(!memcmp(result + (size_t)r * 512, input + (size_t)indices[r] * 512, 512 * 4));
    CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows + 1, selected));
    CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected + 1));
    ds4_gpu_tensor_free(source); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(out);
    free(input); free(result); free(indices);
    return 1;
}

static int check_compact_carry(void) {
    const uint32_t widths[] = {1, 31, 32, 33, 127, 128, 129, 20480};
    const uint32_t format = requested_shape / 8, width = widths[requested_shape % 8];
    const uint32_t rows = 129, offset = 2;
    const uint32_t words = format == DS4_V41_CARRY_BF16 ? (width + 1) / 2 : (width + 31) / 32;
    const size_t count = (size_t)width * rows, bytes = (size_t)(rows + 4) * words * 4;
    uint32_t *expected = malloc(count * 4), *actual = malloc(count * 4);
    unsigned char *storage = malloc(bytes), *packed_ref = malloc(bytes);
    CHECK(expected && actual && storage && packed_ref);
    memset(storage, 0xa5, bytes); memset(packed_ref, 0xa5, bytes);
    for (size_t i = 0; i < count; i++) {
        const uint32_t b = (uint32_t)(i * 40503u + 32768u) & 0xffffu;
        expected[i] = format == DS4_V41_CARRY_BF16 ? b << 16 : (b & 1u ? 0xff800000u : 0u);
    }
    for (uint32_t r = 0; r < rows; r++) {
        if (format == DS4_V41_CARRY_BF16) {
            for (uint32_t c = 0; c < width; c++) {
                const uint16_t b = expected[(size_t)r * width + c] >> 16;
                memcpy(packed_ref + (size_t)(r + offset) * words * 4 + c * 2, &b, 2);
            }
        } else {
            for (uint32_t w = 0; w < words; w++) {
                uint32_t bits = 0;
                for (uint32_t bit = 0; bit < 32 && w * 32 + bit < width; bit++)
                    if (!expected[(size_t)r * width + w * 32 + bit]) bits |= 1u << bit;
                memcpy(packed_ref + ((size_t)(r + offset) * words + w) * 4, &bits, 4);
            }
        }
    }
    ds4_gpu_tensor *plain = upload(expected, count * 4), *packed = upload(storage, bytes);
    CHECK(plain && packed);
    RUN(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, true));
    CHECK(ds4_gpu_tensor_read(plain, 0, actual, count * 4));
    CHECK(!memcmp(actual, expected, count * 4));
    CHECK(ds4_gpu_tensor_read(packed, 0, storage, bytes));
    CHECK(!memcmp(storage, packed_ref, bytes));
    memset(actual, 0, count * 4);
    CHECK(ds4_gpu_tensor_write(plain, 0, actual, count * 4));
    RUN(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, false));
    CHECK(ds4_gpu_tensor_read(plain, 0, actual, count * 4));
    CHECK(!memcmp(actual, expected, count * 4));
    CHECK(!ds4_gpu_dsv41_carry_copy(packed, UINT32_MAX, plain, width, rows, format, true));
    CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset + 3, plain, width, rows, format, true));
    CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, 0, rows, format, true));
    CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, DS4_V41_CARRY_F32, true));
    ds4_gpu_tensor_free(plain); ds4_gpu_tensor_free(packed);
    free(expected); free(actual); free(storage); free(packed_ref);
    return 1;
}

static int check_causal_topk(void) {
    const uint32_t frontiers[] = {1024, 1025, 2047, 2048, 16383, 32767, 65535};
    const uint32_t frontier = frontiers[requested_shape % 7u], ratio = requested_shape / 7u + 1u;
    const uint32_t rows = 33, start = frontier * ratio - 1, width = (start + rows) / ratio + 129;
    const size_t count = (size_t)width * rows;
    float *s = malloc(count * 4);
    int32_t *ids = malloc((size_t)rows * 512 * 4);
    candidate *sorted = malloc(width * sizeof(*sorted));
    CHECK(s && ids && sorted);
    ds4_gpu_tensor *scores = upload(NULL, count * 4), *selected = upload(NULL, (size_t)rows * 512 * 4);
    CHECK(scores && selected);
    for (uint32_t pattern = 0; pattern < 3; pattern++) {
        for (uint32_t row = 0; row < rows; row++) {
            const uint32_t visible = (start + row + 1u) / ratio;
            for (uint32_t j = 0; j < width; j++) {
                const float v = pattern == 0 ? random_value() : pattern == 1 ? (float)(j % 7) : -INFINITY;
                s[(size_t)row * width + j] = j < visible ? v : 12345;
            }
        }
        CHECK(ds4_gpu_tensor_write(scores, 0, s, count * 4));
        RUN(ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, rows, start, ratio));
        CHECK(ds4_gpu_tensor_read(selected, 0, ids, (size_t)rows * 512 * 4));
        for (uint32_t row = 0; row < rows; row++) {
            const uint32_t visible = (start + row + 1u) / ratio;
            for (uint32_t j = 0; j < visible; j++) sorted[j] = (candidate){s[(size_t)row * width + j], j};
            qsort(sorted, visible, sizeof(*sorted), candidate_desc);
            for (uint32_t j = 0; j < 512; j++) CHECK(ids[(size_t)row * 512 + j] == (int32_t)sorted[j].index);
        }
    }
    CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, rows + 1, start, ratio));
    CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, UINT32_MAX, ratio));
    CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, 1, 2, 0, 1));
    CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, start, 0));
    ds4_gpu_tensor_free(scores); ds4_gpu_tensor_free(selected);
    free(s); free(ids); free(sorted);
    return 1;
}

static int check_indexer_scores(void) {
    enum { HEADS = 32, DIM = 128 };
    const struct { uint32_t keys, rows; } shapes[] = {{129,1}, {1025,7}, {1025,8}, {1025,9}, {1025,31}, {1025,32}, {1025,33}, {16385,3}};
    const uint32_t keys = shapes[requested_shape % 8u].keys, rows = shapes[requested_shape % 8u].rows;
    const uint32_t ratio = requested_shape / 8u + 1u;
    const size_t nq = (size_t)rows * HEADS * DIM, nk = (size_t)keys * DIM, nw = (size_t)rows * HEADS;
    float *q = malloc(nq * 4), *k = malloc(nk * 4), *w = malloc(nw * 4), *s = malloc((size_t)rows * keys * 4);
    CHECK(q && k && w && s);
    for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value());
    for (size_t i = 0; i < nk; i++) k[i] = bf16(random_value());
    for (size_t i = 0; i < nw; i++) w[i] = bf16(random_value());
    /* Construct legal FP4 values using the independent codebook oracle. */
    for (size_t i = 0; i < nq; i++) q[i] = nearest(q[i], 1) * ldexpf(1, (int)((i / 32) % 17) - 8);
    for (size_t i = 0; i < nk; i++) k[i] = nearest(k[i], 1) * ldexpf(1, (int)((i / 32) % 17) - 8);
    /* Non-BF16 values detect any extra cast in the F32 fallback. */
    q[0] = 1.0001f; k[67u * DIM] = 1.0003f;
    ds4_gpu_tensor *qt = upload(q, nq * 4), *kt = upload(k, nk * 4), *wt = upload(w, nw * 4);
    ds4_gpu_tensor *st = upload(NULL, (size_t)rows * keys * 4);
    CHECK(qt && kt && wt && st);
    for (unsigned early = 0; early < (ratio == 2 ? 3u : 2u); early++) {
        /* The last unpaired CSA2 token still reuses all existing pooled keys. */
        const uint32_t start = early == 2 ? (keys + 1u) * ratio - rows - 1u :
                               early ? 0 : keys * ratio - rows;
        RUN(ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, keys, rows, start, ratio));
        CHECK(ds4_gpu_tensor_read(st, 0, s, (size_t)rows * keys * 4));
        double worst = 0;
        for (uint32_t row = 0; row < rows; row++) {
            const uint32_t visible = (start + row + 1u) / ratio;
            for (uint32_t j = 0; j < keys; j++) {
                const float actual = s[(size_t)row * keys + j];
                if (j >= visible) { CHECK(actual == -INFINITY); continue; }
                double expected = 0, magnitude = 0;
                for (uint32_t h = 0; h < HEADS; h++) {
                    double dot = 0;
                    for (uint32_t d = 0; d < DIM; d++) dot += (double)q[((size_t)row * HEADS + h) * DIM + d] * k[(size_t)j * DIM + d];
                    const double term = fmax(dot / 64.0, 0) * w[row * HEADS + h];
                    expected += term; magnitude += fabs(term);
                }
                const double error = fabs(actual - expected) / fmax(magnitude, 1);
                CHECK(isfinite(actual) && error < 1e-5);
                worst = fmax(worst, error);
            }
        }
        fprintf(stderr, "Indexer keys=%u rows=%u ratio=%u start=%u worst_relative=%.9g\n", keys, rows, ratio, start, worst);
    }
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, keys, rows, 0, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, keys, rows, 0, 4));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, keys, rows, UINT32_MAX, ratio));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, keys, rows + 1, 0, ratio));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, keys, rows,
        (keys + 1u) * ratio - rows, ratio));
    ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(wt); ds4_gpu_tensor_free(st);
    free(q); free(k); free(w); free(s);
    return 1;
}


static float fixture_half(uint16_t bits) {
    return (1.0f + (bits & 1023) / 1024.0f) / 128.0f * (bits & 0x8000 ? -1.0f : 1.0f);
}

static int check_projection(void) {
    const struct { uint32_t width, out, rows; } shapes[] = {
        {1280,4096,1}, {1280,4096,33}, {5120,32,31}, {5120,512,9}, {512,128,513}, {20480,24,33}
    };
    const uint32_t width = shapes[requested_shape].width, output = shapes[requested_shape].out, rows = shapes[requested_shape].rows;
    const size_t weight_bytes = (size_t)width * output * 2, nx = (size_t)width * rows, ny = (size_t)output * rows;
    void *model = NULL;
    CHECK(!posix_memalign(&model, (size_t)sysconf(_SC_PAGESIZE), weight_bytes));
    uint16_t *weights = model;
    float *input = malloc(nx * 4), *actual = malloc(ny * 4), *scalar = malloc(ny * 4);
    CHECK(input && actual && scalar);
    for (size_t i = 0; i < weight_bytes / 2; i++) {
        const int value = (int)(random_value() * 8192);
        weights[i] = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u) | (value < 0 ? 0x8000u : 0));
    }
    for (size_t i = 0; i < nx; i++) input[i] = i % 511 ? bf16(random_value()) : 1.0001f;
    CHECK(ds4_gpu_set_model_map(model, weight_bytes));
    ds4_gpu_tensor *xt = upload(input, nx * 4), *out = upload(NULL, ny * 4), *ref = upload(NULL, ny * 4);
    CHECK(xt && out && ref);
    RUN(ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, rows, xt));
    CHECK(ds4_gpu_tensor_read(out, 0, actual, ny * 4));
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (size_t)row * width * 4, width * 4);
        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(ref, (size_t)row * output * 4, output * 4);
        CHECK(xr && yr);
        RUN(ds4_gpu_matmul_f16_tensor(yr, model, weight_bytes, 0, width, output, xr, 1));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(yr);
    }
    CHECK(ds4_gpu_tensor_read(ref, 0, scalar, ny * 4));
    CHECK(!memcmp(actual, scalar, ny * 4));
    double worst = 0;
    for (uint32_t row = 0; row < rows; row++) for (uint32_t o = 0; o < output; o++) {
        double sum = 0, magnitude = 0;
        for (uint32_t k = 0; k < width; k++) {
            const double term = (double)fixture_half(weights[(size_t)o * width + k]) * input[(size_t)row * width + k];
            sum += term; magnitude += fabs(term);
        }
        const float got = actual[(size_t)row * output + o];
        const double error = fabs(got - sum) / fmax(magnitude, 1);
        CHECK(isfinite(got) && error < 1e-6);
        worst = fmax(worst, error);
    }
    fprintf(stderr, "F16 projection width=%u out=%u rows=%u scalar exact, double worst=%.9g\n", width, output, rows, worst);
    CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes - 1, 0, width, output, rows, xt));
    CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, rows + 1, xt));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(ref);
    ds4_gpu_cleanup(); free(model); free(input); free(actual); free(scalar);
    return ds4_gpu_init();
}

static int check_hc_scaled(void) {
    enum { WIDTH = 20480, OUT = 24 };
    const uint32_t counts[] = {1,9,33,513}, rows = counts[requested_shape];
    const size_t weight_bytes = (size_t)WIDTH * OUT * 2, nx = (size_t)WIDTH * rows, ny = (size_t)OUT * rows;
    void *model = NULL;
    CHECK(!posix_memalign(&model, (size_t)sysconf(_SC_PAGESIZE), weight_bytes));
    uint16_t *weights = model;
    float *input = malloc(nx * 4), *actual = malloc(ny * 4);
    CHECK(input && actual);
    for (size_t i = 0; i < weight_bytes / 2; i++) weights[i] = (uint16_t)(0x2000 | ((i * 7919u) & 1023u) | (i & 1 ? 0x8000 : 0));
    for (size_t i = 0; i < nx; i++) input[i] = bf16(random_value() * ((i / WIDTH) % 7 ? 1 : 0x1p-16f));
    CHECK(ds4_gpu_set_model_map(model, weight_bytes));
    ds4_gpu_tensor *xt = upload(input, nx * 4), *scratch = upload(NULL, nx * 4), *out = upload(NULL, ny * 4);
    CHECK(xt && scratch && out);
    RUN(ds4_gpu_hc_rms_scale_project_f16_tensor(out, scratch, model, weight_bytes, 0, WIDTH, OUT, xt, rows, 1e-6f));
    CHECK(ds4_gpu_tensor_read(out, 0, actual, ny * 4));
    double worst = 0;
    for (uint32_t row = 0; row < rows; row++) {
        double sumsq = 0;
        for (uint32_t k = 0; k < WIDTH; k++) {
            const double x = input[(size_t)row * WIDTH + k];
            sumsq += x * x;
        }
        const double scale = 1.0 / sqrt(sumsq / WIDTH + 1e-6);
        for (uint32_t o = 0; o < OUT; o++) {
            double sum = 0, magnitude = 0;
            for (uint32_t k = 0; k < WIDTH; k++) {
                const double term = input[(size_t)row * WIDTH + k] * scale * fixture_half(weights[(size_t)o * WIDTH + k]);
                sum += term; magnitude += fabs(term);
            }
            const float got = actual[(size_t)row * OUT + o];
            const double error = fabs(got - sum) / fmax(magnitude, 1);
            CHECK(isfinite(got) && error < 1e-6);
            worst = fmax(worst, error);
        }
    }
    fprintf(stderr, "HC RMS projection rows=%u full double oracle worst=%.9g\n", rows, worst);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(scratch); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup(); free(model); free(input); free(actual);
    return ds4_gpu_init();
}

typedef struct { uint16_t scale; int8_t q[32]; } fixture_q8;
static uint32_t low_column(uint32_t group, uint32_t output, uint32_t term) {
    return (output * 131u + group * 17u + term * 503u) % 4096u;
}
static int low_coefficient(uint32_t group, uint32_t output, uint32_t term) {
    return (int)((output + group + term) % 7u) - 3;
}
static uint32_t output_column(uint32_t output, uint32_t term) {
    return (output * 97u + term * 947u) % 8192u;
}
static int output_coefficient(uint32_t output, uint32_t term) {
    return (int)((output + term * 3u) % 7u) - 3;
}

/* Independent operand rounding and the standard forward-error bound already
 * used to qualify the F16-operand/F32-accumulator bulk projection. */
static float reference_f16(float value) {
    const double magnitude = fabs((double)value);
    if (!magnitude) return value;
    int exponent;
    (void)frexp(magnitude, &exponent);
    const double step = ldexp(1.0, exponent < -13 ? -24 : exponent - 11);
    return (float)copysign(nearbyint(magnitude / step) * step, (double)value);
}

static double projection_roundoff_bound(double magnitude, unsigned width) {
    const double f32 = (2.0 * width + 1.0) * 0x1p-24;
    const double f64 = (width + 1.0) * 0x1p-53;
    return (f32 / (1.0 - f32) + f64 / (1.0 - f64)) * magnitude;
}

static int check_attention_output(void) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, OUT = 5120, TERMS = 8 };
    const uint32_t counts[] = {1,31,32,33,65,513}, rows = counts[requested_shape];
    const size_t a_blocks = (size_t)GROUPS * RANK * GROUP / 32;
    const size_t b_blocks = (size_t)OUT * GROUPS * RANK / 32;
    const size_t a_bytes = a_blocks * sizeof(fixture_q8), bytes = (a_blocks + b_blocks) * sizeof(fixture_q8);
    const size_t nx = (size_t)rows * GROUPS * GROUP, nl = (size_t)rows * GROUPS * RANK, ny = (size_t)rows * OUT;
    void *model = NULL;
    CHECK(!posix_memalign(&model, (size_t)sysconf(_SC_PAGESIZE), bytes));
    memset(model, 0, bytes);
    fixture_q8 *a = model, *b = a + a_blocks;
    /* Dense production layouts with independently specified sparse coefficients
     * keep a complete CPU oracle cheap, including all groups and output rows. */
    for (uint32_t group = 0; group < GROUPS; group++) for (uint32_t o = 0; o < RANK; o++) {
        for (uint32_t j = 0; j < TERMS; j++) {
            const uint32_t k = low_column(group, o, j);
            fixture_q8 *block = &a[((size_t)group * RANK + o) * GROUP / 32 + k / 32];
            block->scale = 0x2000;
            block->q[k % 32] = (int8_t)low_coefficient(group, o, j);
        }
    }
    for (uint32_t o = 0; o < OUT; o++) for (uint32_t j = 0; j < TERMS; j++) {
        const uint32_t k = output_column(o, j);
        fixture_q8 *block = &b[(size_t)o * GROUPS * RANK / 32 + k / 32];
        block->scale = 0x2000;
        block->q[k % 32] = (int8_t)output_coefficient(o, j);
    }
    float *input = malloc(nx * 4), *low_ref = malloc(nl * 4), *low_got = malloc(nl * 4), *out_got = malloc(ny * 4);
    CHECK(input && low_ref && low_got && out_got);
    for (size_t i = 0; i < nx; i++) input[i] = ((int)((i * 37u) % 257u) - 128) / 32.0f;
    CHECK(ds4_gpu_set_model_map(model, bytes));
    ds4_gpu_tensor *xt = upload(input, nx * 4), *low = upload(NULL, nl * 4), *out = upload(NULL, ny * 4);
    CHECK(xt && low && out);
    RUN(ds4_gpu_dsv41_attention_output_batch(out, low, model, bytes, 0, a_bytes, xt, rows));
    CHECK(ds4_gpu_tensor_read(low, 0, low_got, nl * 4));
    CHECK(ds4_gpu_tensor_read(out, 0, out_got, ny * 4));
    const int bulk = rows >= 32u && rows <= 2048u;
    double worst_low = 0, worst_output = 0;
    CHECK(reference_f16(0x1.002p0f) == 1.0f);
    CHECK(reference_f16(0x1.006p0f) == 0x1.008p0f);
    CHECK(reference_f16(0x1p-25f) == 0.0f);
    CHECK(reference_f16(-0x1.8p-24f) == -0x1p-23f);
    for (uint32_t row = 0; row < rows; row++) for (uint32_t group = 0; group < GROUPS; group++) {
        for (uint32_t o = 0; o < RANK; o++) {
            double sum = 0, magnitude = 0;
            for (uint32_t j = 0; j < TERMS; j++) {
                /* These fixture inputs and scaled coefficients are exactly F16. */
                const double term = input[((size_t)row * GROUPS + group) * GROUP + low_column(group, o, j)] * low_coefficient(group, o, j) / 128.0;
                sum += term; magnitude += fabs(term);
            }
            const size_t at = ((size_t)row * GROUPS + group) * RANK + o;
            low_ref[at] = bf16((float)sum);
            CHECK(isfinite(low_got[at]) && low_got[at] == bf16(low_got[at]));
            if (bulk) {
                /* Propagate accumulation error through the required BF16 boundary,
                 * including cancellation and sums on a rounding midpoint. */
                const double bound = projection_roundoff_bound(magnitude, GROUP);
                CHECK(low_got[at] >= bf16((float)(sum - bound)) &&
                      low_got[at] <= bf16((float)(sum + bound)));
            } else {
                CHECK(low_got[at] == low_ref[at]);
            }
            worst_low = fmax(worst_low, fabs(low_got[at] - low_ref[at]));
        }
    }
    for (uint32_t row = 0; row < rows; row++) for (uint32_t o = 0; o < OUT; o++) {
        double sum = 0, magnitude = 0;
        for (uint32_t j = 0; j < TERMS; j++) {
            const float low_value = low_got[(size_t)row * GROUPS * RANK + output_column(o, j)];
            const double term = (bulk ? reference_f16(low_value) : low_value) * output_coefficient(o, j) / 128.0;
            sum += term; magnitude += fabs(term);
        }
        const float got = out_got[(size_t)row * OUT + o];
        const double bound = bulk ? projection_roundoff_bound(magnitude, GROUPS * RANK) : 2e-5 * (1 + fabs(sum));
        CHECK(isfinite(got) && fabs(got - sum) <= bound);
        worst_output = fmax(worst_output, fabs(got - sum));
    }
    fprintf(stderr, "attention-output full reference rows=%u low_values=%zu output_values=%zu bulk=%d max_low_drift=%.9g max_output_error=%.9g\n",
            rows, nl, ny, bulk, worst_low, worst_output);

    /* Exercise the graph's direct Q8 projection helper independently, using
     * values that an accidental F16/BF16 activation cast would change. */
    for (size_t i = 0; i < nl; i++) low_ref[i] += ((i & 1u) ? -1.0f : 1.0f) * 0.00012345f;
    CHECK(ds4_gpu_tensor_write(low, 0, low_ref, nl * 4));
    RUN(ds4_gpu_dsv41_q8_projection_rows(out, model, bytes, a_bytes, GROUPS * RANK, OUT, rows, low));
    CHECK(ds4_gpu_tensor_read(out, 0, out_got, ny * 4));
    size_t q8_mismatches = 0, q8_worst_at = 0;
    double q8_worst_absolute = 0, q8_worst_fraction = 0;
    /* The fixture's scaled coefficients are exact, with one nonzero product
     * per lane: one rounded product plus five F32 tree additions gives gamma6. */
    const double q8_unit_roundoff = FLT_EPSILON / 2.0;
    const double q8_gamma6 = 6 * q8_unit_roundoff / (1 - 6 * q8_unit_roundoff);
    for (uint32_t row = 0; row < rows; row++) for (uint32_t o = 0; o < OUT; o++) {
        double sum = 0, magnitude = 0;
        for (uint32_t j = 0; j < TERMS; j++) {
            const double term = (double)low_ref[(size_t)row * GROUPS * RANK + output_column(o, j)] * output_coefficient(o, j) / 128.0;
            sum += term; magnitude += fabs(term);
        }
        const float got = out_got[(size_t)row * OUT + o];
        const double error = fabs(got - sum), tolerance = q8_gamma6 * fmax(magnitude, 0.001);
        const double fraction = isfinite(got) ? error / tolerance : INFINITY;
        if (fraction > q8_worst_fraction) {
            q8_worst_fraction = fraction;
            q8_worst_at = (size_t)row * OUT + o;
        }
        q8_worst_absolute = fmax(q8_worst_absolute, isfinite(got) ? error : INFINITY);
        if (!(isfinite(got) && error <= tolerance)) {
            if (!q8_mismatches) {
                fprintf(stderr, "Q8 projection first mismatch shape=%u rows=%u row=%u output=%u index=%zu actual=%.17g (%a) sum=%.17g (%a) magnitude=%.17g error=%.17g tolerance=%.17g error_over_tolerance=%.17g\n",
                        requested_shape, rows, row, o, (size_t)row * OUT + o, (double)got, (double)got, sum, sum, magnitude, error, tolerance, fraction);
                for (uint32_t j = 0; j < TERMS; j++) {
                    const uint32_t k = output_column(o, j);
                    fprintf(stderr, "Q8 projection first mismatch term=%u column=%u input=%.17g (%a) coefficient=%d scale=1/128\n",
                            j, k, (double)low_ref[(size_t)row * GROUPS * RANK + k], (double)low_ref[(size_t)row * GROUPS * RANK + k], output_coefficient(o, j));
                }
            }
            q8_mismatches++;
        }
    }
    fprintf(stderr, "Q8 projection full scan outputs=%zu mismatches=%zu worst_absolute=%.17g worst_error_over_tolerance=%.17g worst_index=%zu\n",
            ny, q8_mismatches, q8_worst_absolute, q8_worst_fraction, q8_worst_at);
    CHECK(q8_mismatches == 0);
    /* Scalar decode uses a different F32 reduction from batched rows. Check
     * every scalar output against the same independent oracle and unchanged
     * gamma6 bound above; retain bit equality for repeated scalar calls. */
    ds4_gpu_tensor *one = upload(NULL, OUT * 4);
    CHECK(one);
    const uint32_t probes[] = {0, rows / 2, rows - 1};
    for (unsigned i = 0; i < sizeof(probes) / sizeof(*probes); i++) {
        if (i && probes[i] == probes[i - 1]) continue;
        const uint32_t row = probes[i];
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(low, (size_t)row * GROUPS * RANK * 4, GROUPS * RANK * 4);
        CHECK(view);
        RUN(ds4_gpu_dsv41_q8_projection_rows(one, model, bytes, a_bytes, GROUPS * RANK, OUT, 1, view));
        CHECK(ds4_gpu_tensor_read(one, 0, low_got, OUT * 4));
        double scalar_worst_fraction = 0, scalar_batch_drift = 0;
        for (uint32_t o = 0; o < OUT; o++) {
            double sum = 0, magnitude = 0;
            for (uint32_t j = 0; j < TERMS; j++) {
                const double term = (double)low_ref[(size_t)row * GROUPS * RANK + output_column(o, j)] * output_coefficient(o, j) / 128.0;
                sum += term;
                magnitude += fabs(term);
            }
            const double error = fabs((double)low_got[o] - sum);
            const double tolerance = q8_gamma6 * fmax(magnitude, 0.001);
            CHECK(isfinite(low_got[o]) && error <= tolerance);
            scalar_worst_fraction = fmax(scalar_worst_fraction, error / tolerance);
            scalar_batch_drift = fmax(scalar_batch_drift, fabs((double)low_got[o] - out_got[(size_t)row * OUT + o]));
        }
        if (rows == 1u) CHECK(!memcmp(low_got, out_got, OUT * 4));
        fprintf(stderr, "Q8 scalar full oracle row=%u outputs=%u worst_error_over_unchanged_bound=%.9g scalar_batch_maxabs=%.9g\n",
                row, OUT, scalar_worst_fraction, scalar_batch_drift);
        ds4_gpu_tensor_free(view);
    }
    ds4_gpu_tensor_free(one);
    CHECK(!ds4_gpu_dsv41_q8_projection_rows(out, model, bytes - 1, a_bytes, GROUPS * RANK, OUT, rows, low));
    CHECK(!ds4_gpu_dsv41_q8_projection_rows(out, model, bytes, a_bytes, GROUPS * RANK, OUT, rows + 1, low));
    CHECK(!ds4_gpu_dsv41_q8_projection_rows(out, model, bytes, a_bytes, GROUPS * RANK - 1, OUT, rows, low));
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, bytes - 1, 0, a_bytes, xt, rows));
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, bytes, 0, a_bytes, xt, rows + 1));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup(); free(model); free(input); free(low_ref); free(low_got); free(out_got);
    return ds4_gpu_init();
}

typedef struct { const char *name; unsigned shapes; int (*run)(void); } test_case;
static const test_case cases[] = {
    {"quantization", 4, check_quantization},
    {"engram", 1, check_engram},
    {"pool", 8, check_pool},
    {"candidates", 7, check_candidates},
    {"bf16", 13, check_bf16_boundaries},
    {"rope", 16, check_rope},
    {"gather", 7, check_sparse_gather},
    {"carry", 16, check_compact_carry},
    {"topk", 14, check_causal_topk},
    {"indexer", 16, check_indexer_scores},
    {"projection", 6, check_projection},
    {"hc", 4, check_hc_scaled},
    {"attention-output", 6, check_attention_output},
};

int main(int argc, char **argv) {
    const char *only = NULL;
    if (argc == 2 && !strcmp(argv[1], "--list")) {
        for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++)
            printf("%s %u shapes\n", cases[i].name, cases[i].shapes);
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "--case")) only = argv[2];
    else if (argc == 5 && !strcmp(argv[1], "--case") && !strcmp(argv[3], "--shape")) {
        char *end;
        unsigned long shape = strtoul(argv[4], &end, 10);
        if (!*argv[4] || *end || shape > UINT32_MAX) return 2;
        for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
            if (strcmp(cases[i].name, argv[2])) continue;
            if (shape >= cases[i].shapes) return 2;
            requested_shape = (unsigned)shape;
            fprintf(stderr, "V4.1 ROCm case=%s shape=%u pid=%ld\n", cases[i].name, requested_shape, (long)getpid());
            if (!ds4_gpu_init() || !cases[i].run() || !sync_guards()) _exit(1);
            if (allocations) { fprintf(stderr, "unreleased guarded allocation\n"); _exit(1); }
            ds4_gpu_cleanup();
            fprintf(stderr, "PASS %s shape=%u\n", cases[i].name, requested_shape);
            return 0;
        }
        return 2;
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--list | --case NAME [--shape N]]\n", argv[0]);
        return 2;
    }
    unsigned tested = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        if (only && strcmp(only, cases[i].name)) continue;
        for (unsigned shape = 0; shape < cases[i].shapes; shape++) {
            char number[24];
            snprintf(number, sizeof(number), "%u", shape);
            pid_t child = fork();
            if (child < 0) { perror("fork"); return 1; }
            if (!child) {
                execlp(argv[0], argv[0], "--case", cases[i].name, "--shape", number, (char *)NULL);
                perror("exec harness");
                _exit(127);
            }
            int status;
            while (waitpid(child, &status, 0) < 0) {
                if (errno != EINTR) { perror("waitpid"); return 1; }
            }
            if (!WIFEXITED(status) || WEXITSTATUS(status)) {
                fprintf(stderr, "FAIL %s shape=%u child_status=%d; stopping at first failure\n", cases[i].name, shape, status);
                return 1;
            }
            tested++;
        }
    }
    if (!tested) return 2;
    fprintf(stderr, "PASS %u isolated V4.1 ROCm shapes\n", tested);
    return 0;
}
