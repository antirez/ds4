/* Synthetic CUDA HC fusion oracle; no model or distributed runtime.
 * make test-cuda-v41-hc
 * ./tests/test_cuda_v41_hc --bench
 * Compile the CPU oracle without fast-math: legacy CUDA accumulates with FMA. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { E = 5120, HC = 4, PAD = 16 };
/* Explicit tensor indices keep the slab's guards and original inputs together. */
enum { BLOCK, ADD, RESIDUAL, SPLIT, WORK, OUTPUT, REFERENCE, TENSORS };
typedef struct {
    uint32_t rows;
    uint64_t bytes, offset[TENSORS], count[TENSORS];
    float *input, *actual;
    ds4_gpu_tensor *slab, *t[TENSORS];
} fixture;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float value(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static float bf16(float x) {
    uint32_t u = bits(x);
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16u) & 1u);
    return value(u & 0xffff0000u);
}
static float add_rn(float a, float b) { volatile float x = a + b; return x; }
static float mul_rn(float a, float b) { volatile float x = a * b; return x; }
static uint32_t width(unsigned t) {
    return t == SPLIT ? 24 : t == RESIDUAL || t == OUTPUT || t == REFERENCE ? HC * E : E;
}
static float *data(fixture *f, unsigned t) { return f->input + f->offset[t]; }

static int initialize(fixture *f, uint32_t rows, unsigned pattern) {
    f->rows = rows;
    uint64_t words = 0;
    for (unsigned t = 0; t < TENSORS; t++) {
        f->count[t] = (uint64_t)rows * width(t);
        f->offset[t] = words + PAD;
        words += f->count[t] + 2u * PAD;
    }
    f->bytes = words * 4;
    f->input = malloc(f->bytes); f->actual = malloc(f->bytes);
    CHECK(f->input && f->actual);
    for (uint64_t i = 0; i < words; i++) f->input[i] = value(0x7fc12345u);
    const uint32_t ties[] = {0x3f807fff,0x3f808000,0x3f808001,0x3f818000,
        0xbf808000,0xbf818000,0x00000000,0x80000000,0x411f8000,0xc11f8000};
    for (unsigned t = 0; t <= SPLIT; t++) {
        float *x = data(f, t);
        for (uint64_t i = 0; i < f->count[t]; i++) {
            uint32_t z = (uint32_t)i * 747796405u + (t + 7u) * 2891336453u;
            z ^= z >> 13;
            x[i] = ((int)(z % 65537u) - 32768) / 8192.0f;
            if (pattern == 1 && t != SPLIT)
                x[i] = value(ties[(i + 3u * t) % (sizeof(ties) / sizeof(*ties))]);
        }
    }
    if (pattern >= 2) for (uint32_t r = 0; r < rows; r++) {
        float *split = data(f, SPLIT) + r * 24u;
        for (unsigned i = 0; i < 24; i++) split[i] = 0;
        for (unsigned dst = 0; dst < HC; dst++) {
            split[4 + dst] = 1;
            for (unsigned src = 0; src < HC; src++)
                split[8 + src * HC + dst] = pattern == 2 ? (src ? 0 : 1.0f - 0x1p-13f) : 1;
        }
        for (unsigned d = 0; d < E; d++) {
            const uint64_t i = (uint64_t)r * E + d, h = (uint64_t)r * HC * E + d;
            data(f, BLOCK)[i] = pattern == 2 ? -1 : (d & 1 ? 1.00390625f : -1.01171875f);
            data(f, ADD)[i] = pattern == 2 ? 0 : (d & 1 ? 0x1p-16f : -0x1p-16f);
            data(f, RESIDUAL)[h] = pattern == 2 ? 1.0f + 0x1p-13f : 8192;
            data(f, RESIDUAL)[h + E] = pattern == 2 ? 0 : -8192;
            data(f, RESIDUAL)[h + 2 * E] = pattern == 2 ? 0 : (d & 1 ? 0x1p-8f : -0x1p-8f);
            data(f, RESIDUAL)[h + 3 * E] = pattern == 2 ? 0 : 0x1p-16f;
        }
    }
    f->slab = ds4_gpu_tensor_alloc(f->bytes);
    CHECK(f->slab && ds4_gpu_tensor_write(f->slab, 0, f->input, f->bytes));
    for (unsigned t = 0; t < TENSORS; t++) {
        f->t[t] = ds4_gpu_tensor_view(f->slab, f->offset[t] * 4, f->count[t] * 4);
        CHECK(f->t[t]);
    }
    return 1;
}

static void destroy(fixture *f) {
    for (unsigned t = 0; t < TENSORS; t++) ds4_gpu_tensor_free(f->t[t]);
    ds4_gpu_tensor_free(f->slab); free(f->actual); free(f->input);
}

static int legacy_expand(fixture *f) {
    CHECK(ds4_gpu_dsv41_quantize(f->t[WORK], E, f->rows, DS4_V41_BF16));
    CHECK(ds4_gpu_hc_expand_split_tensor(f->t[REFERENCE], f->t[WORK],
        f->t[RESIDUAL], f->t[SPLIT], E, HC));
    return ds4_gpu_dsv41_quantize(f->t[REFERENCE], HC * E, f->rows, DS4_V41_BF16);
}

static int sequence(fixture *f, unsigned add, int fused) {
    ds4_gpu_tensor *a = add == 2 ? f->t[BLOCK] : add ? f->t[ADD] : NULL;
    if (fused) return ds4_gpu_dsv41_hc_expand_bf16(f->t[OUTPUT], f->t[BLOCK], a,
        f->t[RESIDUAL], f->t[SPLIT], f->rows);
    CHECK(a ? ds4_gpu_add_tensor(f->t[WORK], f->t[BLOCK], a, f->rows * E) :
        ds4_gpu_tensor_copy(f->t[WORK], 0, f->t[BLOCK], 0, f->count[BLOCK] * 4));
    return legacy_expand(f);
}

/* The initial product is rounded before the four ordered CUDA FMAs. In
 * particular, rounding each residual product separately is a different graph. */
static float cpu_value(fixture *f, uint32_t row, unsigned dst, unsigned d, unsigned add, int contracted) {
    const uint64_t i = (uint64_t)row * E + d;
    const float *s = data(f, SPLIT) + row * 24u;
    float block = data(f, BLOCK)[i];
    if (add) block = add_rn(block, data(f, add == 2 ? BLOCK : ADD)[i]);
    float acc = mul_rn(bf16(block), s[4 + dst]);
    for (unsigned src = 0; src < HC; src++) {
        const float c = s[8 + src * HC + dst];
        const float r = data(f, RESIDUAL)[(uint64_t)row * HC * E + src * E + d];
        acc = contracted ? fmaf(c, r, acc) : add_rn(acc, mul_rn(c, r));
    }
    return bf16(acc);
}

static int compare(fixture *f, unsigned add) {
    CHECK(ds4_gpu_tensor_read(f->slab, 0, f->actual, f->bytes));
    const float *out = f->actual + f->offset[OUTPUT], *ref = f->actual + f->offset[REFERENCE];
    for (uint32_t r = 0; r < f->rows; r++) for (unsigned h = 0; h < HC; h++)
        for (unsigned d = 0; d < E; d++) {
            const uint64_t i = ((uint64_t)r * HC + h) * E + d;
            const float cpu = cpu_value(f, r, h, d, add, 1);
            /* --fmad=false builds use the other exact candidate. Full-array
             * GPU legacy parity above remains mandatory in either build. */
            if (bits(out[i]) != bits(ref[i]) || (bits(out[i]) != bits(cpu) &&
                bits(out[i]) != bits(cpu_value(f, r, h, d, add, 0)))) {
                fprintf(stderr, "HC rows=%u add=%u row=%u hc=%u dim=%u fused=%08x legacy=%08x CPU=%08x\n",
                    f->rows, add, r, h, d, bits(out[i]), bits(ref[i]), bits(cpu));
                return 0;
            }
            CHECK(isfinite(out[i]) && !(bits(out[i]) & 0xffffu));
        }
    for (unsigned t = 0; t < TENSORS; t++) {
        if (t <= SPLIT) CHECK(!memcmp(f->actual + f->offset[t], data(f, t), f->count[t] * 4));
        for (unsigned p = 0; p < PAD; p++) {
            CHECK(bits(f->actual[f->offset[t] - PAD + p]) == 0x7fc12345u);
            CHECK(bits(f->actual[f->offset[t] + f->count[t] + p]) == 0x7fc12345u);
        }
    }
    return 1;
}

static int rejected(fixture *f) {
    CHECK(ds4_gpu_tensor_read(f->slab, 0, f->actual, f->bytes));
    CHECK(!ds4_gpu_dsv41_hc_expand_bf16(f->t[OUTPUT], f->t[BLOCK], NULL, f->t[RESIDUAL], f->t[SPLIT], 0));
    CHECK(!ds4_gpu_dsv41_hc_expand_bf16(f->t[OUTPUT], f->t[BLOCK], NULL, f->t[RESIDUAL], f->t[SPLIT], UINT32_MAX));
    const unsigned slots[] = {OUTPUT, BLOCK, ADD, RESIDUAL, SPLIT};
    for (unsigned i = 0; i < sizeof(slots) / sizeof(*slots); i++) {
        ds4_gpu_tensor *args[] = {f->t[OUTPUT], f->t[BLOCK], f->t[ADD], f->t[RESIDUAL], f->t[SPLIT]};
        const unsigned t = slots[i];
        ds4_gpu_tensor *short_t = ds4_gpu_tensor_view(f->t[t], 0, f->count[t] * 4 - 4);
        CHECK(short_t);
        args[i] = short_t;
        CHECK(!ds4_gpu_dsv41_hc_expand_bf16(args[0], args[1], args[2], args[3], args[4], f->rows));
        ds4_gpu_tensor_free(short_t);
        if (t != ADD) {
            args[i] = NULL;
            CHECK(!ds4_gpu_dsv41_hc_expand_bf16(args[0], args[1], args[2], args[3], args[4], f->rows));
        }
    }
    /* Out must be disjoint even when a view only partially overlaps an input. */
    for (unsigned t = BLOCK; t <= SPLIT; t++) {
        ds4_gpu_tensor *overlap = ds4_gpu_tensor_view(f->slab, f->offset[t] * 4, f->count[OUTPUT] * 4);
        CHECK(overlap && !ds4_gpu_dsv41_hc_expand_bf16(overlap, f->t[BLOCK], f->t[ADD],
            f->t[RESIDUAL], f->t[SPLIT], f->rows));
        ds4_gpu_tensor_free(overlap);
    }
    ds4_gpu_tensor *bad = ds4_gpu_tensor_view(f->slab, f->offset[BLOCK] * 4 + 1, f->count[BLOCK] * 4);
    CHECK(bad && !ds4_gpu_dsv41_hc_expand_bf16(f->t[OUTPUT], bad, NULL, f->t[RESIDUAL], f->t[SPLIT], f->rows));
    ds4_gpu_tensor_free(bad);
    ds4_gpu_set_quality(true);
    CHECK(!ds4_gpu_dsv41_hc_expand_bf16(f->t[OUTPUT], f->t[BLOCK], NULL, f->t[RESIDUAL], f->t[SPLIT], f->rows));
    ds4_gpu_set_quality(false);
    CHECK(ds4_gpu_synchronize());
    /* Reuse the original host buffer only after all numerical/input checks. */
    CHECK(ds4_gpu_tensor_read(f->slab, 0, f->input, f->bytes));
    CHECK(!memcmp(f->actual, f->input, f->bytes));
    return 1;
}

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int time_order(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static int benchmark(fixture *f, unsigned add) {
    double samples[2][7];
    /* The no-add block rounds in place idempotently, so restore its fixture
     * once outside all timed intervals. Add mode writes WORK directly. */
    CHECK(ds4_gpu_tensor_copy(f->t[WORK], 0, f->t[BLOCK], 0, f->count[BLOCK] * 4));
    for (unsigned run = 0; run < 9; run++) for (unsigned turn = 0; turn < 2; turn++) {
        const unsigned fused = (run + turn) & 1u;
        CHECK(ds4_gpu_synchronize());
        const double start = seconds();
        for (unsigned i = 0; i < 20; i++) {
            if (fused) CHECK(sequence(f, add, 1));
            else {
                CHECK(!add || ds4_gpu_add_tensor(f->t[WORK], f->t[BLOCK],
                    f->t[add == 2 ? BLOCK : ADD], f->rows * E));
                CHECK(legacy_expand(f));
            }
        }
        CHECK(ds4_gpu_synchronize());
        if (run >= 2) samples[fused][run - 2] = (seconds() - start) * 1e6 / 20;
    }
    for (unsigned m = 0; m < 2; m++) qsort(samples[m], 7, sizeof(double), time_order);
    printf("CUDA HC rows=%u add=%u median_us legacy=%.3f fused=%.3f speedup=%.2fx\n",
        f->rows, add, samples[0][3], samples[1][3], samples[0][3] / samples[1][3]);
    return 1;
}

int main(int argc, char **argv) {
    const int bench = argc == 2 && !strcmp(argv[1], "--bench");
    if (argc != 1 && !bench) { fprintf(stderr, "usage: %s [--bench]\n", argv[0]); return 2; }
    if (!ds4_gpu_init()) return 1;
    const uint32_t rows[] = {1, 2, 8, 32, 65, 128};
    int ok = 1;
    for (unsigned n = 0; ok && n < sizeof(rows) / sizeof(*rows); n++)
        for (unsigned pattern = 0; ok && pattern < 4; pattern++) {
            fixture f = {0};
            ok = initialize(&f, rows[n], pattern);
            for (unsigned add = 0; ok && add < 3; add++) {
                ok = sequence(&f, add, 0) && sequence(&f, add, 1) &&
                     ds4_gpu_synchronize() && compare(&f, add);
                if (ok && bench && !pattern && add < 2) ok = benchmark(&f, add);
            }
            if (ok && rows[n] == 1 && !pattern) ok = rejected(&f);
            destroy(&f);
            if (ok) fprintf(stderr, "CUDA HC rows=%u pattern=%u optional-add/alias: bitwise legacy + CPU FMA PASS\n", rows[n], pattern);
        }
    ds4_gpu_cleanup();
    if (ok) puts("CUDA V4.1 HC: BF16 boundaries, FMA/cancellation, inputs/guards and rejections PASS");
    return ok ? 0 : 1;
}
