#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)
enum { E = 5120, F = 2304, PAD = 64 };
enum { A, B, H, S, G, U, AR, W, GR, UR, ER, EF, SR, SF, MR, MF, NT };
enum { SWIGLU, EXPAND, ADD_EXPAND, SUM_PRE, SUM_SPLIT, OPS };
static const char *op_names[] = {"swiglu", "expand", "add-expand", "sum-pre", "sum-split"};
typedef struct {
    uint32_t rows;
    uint64_t bytes[NT], offset[NT];
    ds4_gpu_tensor *slab, *t[NT];
    float clamp;
} fixture;
static void *model;
static uint64_t model_bytes;

static uint32_t width(unsigned i) {
    if (i == H || i == ER || i == EF) return 4 * E;
    if (i == S) return 24;
    if (i == G || i == U || i == GR || i == UR || i == MR || i == MF) return F;
    return E;
}

static int initialize(fixture *f, uint32_t rows, unsigned pattern) {
    f->rows = rows;
    f->clamp = 10;
    uint64_t total = 0;
    for (unsigned i = 0; i < NT; i++) {
        f->bytes[i] = (uint64_t)width(i) * rows * 4;
        f->offset[i] = total + PAD * 4;
        total += f->bytes[i] + 2 * PAD * 4;
    }
    f->slab = ds4_gpu_tensor_alloc(total);
    CHECK(f->slab && ds4_gpu_tensor_contents(f->slab));
    uint32_t *p = ds4_gpu_tensor_contents(f->slab);
    for (uint64_t i = 0; i < total / 4; i++) p[i] = 0x7fc12345u;
    const uint32_t ties[] = {0x3f807fff,0x3f808000,0x3f808001,0x3f818000,
        0xbf808000,0xbf818000,0x00000000,0x80000000,0x00800000,0x80800000,
        0x411f8000,0xc11f8000,0x41208000,0xc1208000};
    for (unsigned i = 0; i < NT; i++) {
        f->t[i] = ds4_gpu_tensor_view(f->slab, f->offset[i], f->bytes[i]);
        CHECK(f->t[i]);
        if (i > U) continue;
        float *v = ds4_gpu_tensor_contents(f->t[i]);
        for (uint64_t j = 0; j < f->bytes[i] / 4; j++) {
            if (pattern == 1 && i != S) {
                const uint32_t bits = ties[(j + 3u * i) % (sizeof(ties)/sizeof(*ties))];
                memcpy(v + j, &bits, 4);
            } else {
                uint32_t z = (uint32_t)j * 747796405u + (i + pattern * 7u) * 2891336453u;
                z ^= z >> 13;
                v[j] = ((int)(z % 65537u) - 32768) / 8192.0f;
            }
        }
    }
    if (pattern == 2) {
        float *a = ds4_gpu_tensor_contents(f->t[A]);
        float *b = ds4_gpu_tensor_contents(f->t[B]);
        float *h = ds4_gpu_tensor_contents(f->t[H]);
        float *s = ds4_gpu_tensor_contents(f->t[S]);
        for (uint32_t t = 0; t < rows; t++) {
            for (unsigned j = 0; j < 24; j++) s[t * 24u + j] = 1.0f;
            for (unsigned d = 0; d < E; d++) {
                a[(uint64_t)t * E + d] = d & 1u ? 1.00390625f : -1.01171875f;
                b[(uint64_t)t * E + d] = d & 1u ? 0x1p-16f : -0x1p-16f;
                const uint64_t base = (uint64_t)t * 4u * E + d;
                h[base] = 8192.0f;
                h[base + E] = -8192.0f;
                h[base + 2 * E] = d & 1u ? 0x1p-8f : -0x1p-8f;
                h[base + 3 * E] = 0x1p-16f;
            }
        }
    }
    memcpy(ds4_gpu_tensor_contents(f->t[AR]), ds4_gpu_tensor_contents(f->t[A]), f->bytes[A]);
    memcpy(ds4_gpu_tensor_contents(f->t[GR]), ds4_gpu_tensor_contents(f->t[G]), f->bytes[G]);
    memcpy(ds4_gpu_tensor_contents(f->t[UR]), ds4_gpu_tensor_contents(f->t[U]), f->bytes[U]);
    return 1;
}

static void destroy(fixture *f) {
    for (unsigned i = 0; i < NT; i++) ds4_gpu_tensor_free(f->t[i]);
    ds4_gpu_tensor_free(f->slab);
}

static int sequence(fixture *f, unsigned op, int fused) {
    const uint32_t rows = f->rows;
    if (fused) {
        if (op == SWIGLU) return ds4_gpu_dsv41_swiglu_bf16(f->t[MF], f->t[G], f->t[U], rows, f->clamp);
        if (op == EXPAND || op == ADD_EXPAND)
            return ds4_gpu_dsv41_hc_expand_bf16(f->t[EF], f->t[A],
                op == ADD_EXPAND ? f->t[B] : NULL, f->t[H], f->t[S], rows);
        return ds4_gpu_dsv41_hc_sum_bf16(f->t[SF], f->t[H], f->t[S], rows, op == SUM_SPLIT);
    }
    if (op == SWIGLU)
        return ds4_gpu_dsv41_quantize(f->t[GR], F, rows, DS4_V41_BF16) &&
            ds4_gpu_dsv41_quantize(f->t[UR], F, rows, DS4_V41_BF16) &&
            ds4_gpu_swiglu_tensor(f->t[MR], f->t[GR], f->t[UR], rows * F, f->clamp, 1.0f) &&
            ds4_gpu_dsv41_quantize(f->t[MR], F, rows, DS4_V41_BF16);
    if (op == EXPAND || op == ADD_EXPAND) {
        /* The graph permits block=routed. The sum is deliberately in-place
         * in W after its caller has copied the routed output there. */
        ds4_gpu_tensor *block = op == ADD_EXPAND ? f->t[W] : f->t[AR];
        return (op != ADD_EXPAND || ds4_gpu_add_tensor(block, block, f->t[B], rows * E)) &&
            ds4_gpu_dsv41_quantize(block, E, rows, DS4_V41_BF16) &&
            ds4_gpu_hc_expand_split_tensor(f->t[ER], block, f->t[H], f->t[S], E, 4) &&
            ds4_gpu_dsv41_quantize(f->t[ER], 4 * E, rows, DS4_V41_BF16);
    }
    return (op == SUM_SPLIT ?
            ds4_gpu_hc_weighted_sum_split_tensor(f->t[SR], f->t[H], f->t[S], E, 4) :
            ds4_gpu_hc_weighted_sum_tensor(f->t[SR], f->t[H], f->t[S], E, 4)) &&
        ds4_gpu_dsv41_quantize(f->t[SR], E, rows, DS4_V41_BF16);
}

static uint64_t hash(const void *ptr, uint64_t bytes) {
    const uint32_t *p = ptr;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint64_t i = 0; i < bytes / 4; i++) h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

static int compare(fixture *f, unsigned op) {
    const unsigned r = op == SWIGLU ? MR : op <= ADD_EXPAND ? ER : SR;
    const unsigned v = r + 1;
    const uint32_t *a = ds4_gpu_tensor_contents(f->t[r]);
    const uint32_t *b = ds4_gpu_tensor_contents(f->t[v]);
    for (uint64_t i = 0; i < f->bytes[r] / 4; i++) {
        if (a[i] != b[i] || (b[i] & 0xffffu)) {
            fprintf(stderr, "%s rows=%u clamp=%g word=%llu expected=%08x got=%08x\n",
                op_names[op], f->rows, f->clamp, (unsigned long long)i, a[i], b[i]);
            return 0;
        }
    }
    const uint32_t *p = ds4_gpu_tensor_contents(f->slab);
    for (unsigned i = 0; i < NT; i++) for (unsigned j = 0; j < PAD; j++) {
        CHECK(p[f->offset[i]/4 - PAD + j] == 0x7fc12345u);
        CHECK(p[(f->offset[i] + f->bytes[i])/4 + j] == 0x7fc12345u);
    }
    return 1;
}

static int parity(fixture *f, unsigned op) {
    uint64_t input_hash[U + 1];
    for (unsigned i = 0; i <= U; i++) input_hash[i] = hash(ds4_gpu_tensor_contents(f->t[i]), f->bytes[i]);
    CHECK(ds4_gpu_begin_commands());
    if (op == ADD_EXPAND)
        CHECK(ds4_gpu_tensor_copy(f->t[W], 0, f->t[A], 0, f->bytes[A]));
    CHECK(sequence(f, op, 0) && sequence(f, op, 1));
    CHECK(ds4_gpu_end_commands() && compare(f, op));
    for (unsigned i = 0; i <= U; i++)
        CHECK(hash(ds4_gpu_tensor_contents(f->t[i]), f->bytes[i]) == input_hash[i]);
    return 1;
}

static int rejected(fixture *f) {
    const uint64_t before = hash(ds4_gpu_tensor_contents(f->slab), ds4_gpu_tensor_bytes(f->slab));
    CHECK(!ds4_gpu_dsv41_swiglu_bf16(f->t[MF], f->t[G], f->t[U], 0, 10));
    CHECK(!ds4_gpu_dsv41_swiglu_bf16(f->t[MF], f->t[G], f->t[U], UINT32_MAX, 10));
    CHECK(!ds4_gpu_dsv41_swiglu_bf16(f->t[MF], f->t[G], f->t[U], f->rows, NAN));
    CHECK(!ds4_gpu_dsv41_swiglu_bf16(f->t[MF], f->t[G], f->t[U], f->rows, -1));
    CHECK(!ds4_gpu_dsv41_swiglu_bf16(f->t[G], f->t[G], f->t[U], f->rows, 10));
    CHECK(!ds4_gpu_dsv41_hc_expand_bf16(f->t[H], f->t[A], f->t[B], f->t[H], f->t[S], f->rows));
    CHECK(!ds4_gpu_dsv41_hc_expand_bf16(f->t[EF], NULL, NULL, f->t[H], f->t[S], f->rows));
    CHECK(!ds4_gpu_dsv41_hc_sum_bf16(f->t[SF], f->t[H], f->t[S], UINT32_MAX, true));
    CHECK(!ds4_gpu_dsv41_hc_sum_bf16(f->t[H], f->t[H], f->t[S], f->rows, true));
    ds4_gpu_tensor *bad = ds4_gpu_tensor_view(f->slab, f->offset[G] + 4, f->bytes[G]);
    CHECK(bad && !ds4_gpu_dsv41_swiglu_bf16(f->t[MF], bad, f->t[U], f->rows, 10));
    ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(f->slab, f->offset[G] + 16, f->bytes[G]);
    CHECK(bad && !ds4_gpu_dsv41_swiglu_bf16(bad, f->t[G], f->t[U], f->rows, 10));
    ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(f->slab, f->offset[EF], f->bytes[EF] - 4);
    CHECK(bad && !ds4_gpu_dsv41_hc_expand_bf16(bad, f->t[A], NULL, f->t[H], f->t[S], f->rows));
    ds4_gpu_tensor_free(bad);
    ds4_gpu_set_quality(true);
    CHECK(!ds4_gpu_dsv41_swiglu_bf16(f->t[MF], f->t[G], f->t[U], f->rows, 10));
    CHECK(!ds4_gpu_dsv41_hc_expand_bf16(f->t[EF], f->t[A], NULL, f->t[H], f->t[S], f->rows));
    CHECK(!ds4_gpu_dsv41_hc_sum_bf16(f->t[SF], f->t[H], f->t[S], f->rows, true));
    ds4_gpu_set_quality(false);
    CHECK(!ds4_gpu_commands_active());
    CHECK(before == hash(ds4_gpu_tensor_contents(f->slab), ds4_gpu_tensor_bytes(f->slab)));
    return 1;
}

static int q8_boundaries(void) {
    typedef struct { uint16_t d; int8_t qs[32]; } q8;
    CHECK(sizeof(q8) == 34);
    const uint64_t matrix = (uint64_t)E * F / 32u * sizeof(q8);
    model_bytes = 128 + 2 * matrix;
    model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(model != MAP_FAILED);
    q8 *w = (q8 *)((char *)model + 128);
    for (uint64_t i = 0; i < 2 * matrix / sizeof(q8); i++) {
        w[i].d = 0x1800u + i % 0x1800u;
        for (unsigned j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((i * 17 + j * 31) % 255 - 127);
    }
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    for (uint32_t rows = 2; rows <= 8; rows++) {
        fixture f = {0};
        CHECK(initialize(&f, rows, rows & 1u));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_quantize(f.t[A], E, rows, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(f.t[G], model, model_bytes, 128, E, F, f.t[A], rows));
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(f.t[U], model, model_bytes, 128 + matrix, E, F, f.t[A], rows));
        CHECK(ds4_gpu_tensor_copy(f.t[GR], 0, f.t[G], 0, f.bytes[G]));
        CHECK(ds4_gpu_tensor_copy(f.t[UR], 0, f.t[U], 0, f.bytes[U]));
        CHECK(ds4_gpu_end_commands() && parity(&f, SWIGLU));
        destroy(&f);
    }
    return 1;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int benchmark(uint32_t rows) {
    enum { SAMPLES = 7, ITER = 20 };
    fixture f = {0};
    CHECK(initialize(&f, rows, 0));
    for (unsigned op = 0; op < OPS; op++) {
        CHECK(parity(&f, op));
        double samples[2][SAMPLES];
        for (unsigned sample = 0; sample < SAMPLES + 1; sample++) {
            double elapsed[2] = {0,0};
            for (unsigned turn = 0; turn < 4; turn++) {
                const unsigned fused = (turn == 1 || turn == 2) ^ (sample & 1u);
                const double start = seconds();
                CHECK(ds4_gpu_begin_commands());
                for (unsigned i = 0; i < ITER; i++) {
                    /* This copy models an upstream routed producer equally
                     * in both timing arms; it is outside the fusion itself. */
                    if (op == ADD_EXPAND)
                        CHECK(ds4_gpu_tensor_copy(f.t[W], 0, f.t[A], 0, f.bytes[A]));
                    CHECK(sequence(&f, op, fused));
                }
                CHECK(ds4_gpu_end_commands());
                elapsed[fused] += seconds() - start;
            }
            if (sample) for (unsigned mode = 0; mode < 2; mode++)
                samples[mode][sample - 1] = elapsed[mode] * 1e6 / (2 * ITER);
        }
        for (unsigned mode = 0; mode < 2; mode++) qsort(samples[mode], SAMPLES, sizeof(double), cmp_double);
        CHECK(compare(&f, op));
        fprintf(stderr, "V4.1 epilogue %s rows%u median us: old=%.3f fused=%.3f "
            "(alternating ABBA, %u samples)\n", op_names[op], rows,
            samples[0][SAMPLES/2], samples[1][SAMPLES/2], SAMPLES);
    }
    destroy(&f);
    return 1;
}

static int check_all(uint32_t bench_rows) {
    const uint32_t rows[] = {1,2,7,8,31,32,33,128,437,1024};
    for (unsigned i = 0; i < sizeof(rows)/sizeof(*rows); i++) {
        for (unsigned pattern = 0; pattern < 3; pattern++) {
            fixture f = {0};
            CHECK(initialize(&f, rows[i], pattern));
            for (unsigned op = 0; op < OPS; op++) CHECK(parity(&f, op));
            if (rows[i] <= 8) {
                const float limits[] = {0,1.0e-6f,1.001e-6f,1.00390625f};
                for (unsigned k = 0; k < sizeof(limits)/sizeof(*limits); k++) {
                    f.clamp = limits[k];
                    CHECK(parity(&f, SWIGLU));
                }
            }
            if (i == 0) CHECK(rejected(&f));
            destroy(&f);
        }
    }
    CHECK(q8_boundaries());
    fprintf(stderr, "V4.1 epilogues: bitwise BF16/SwiGLU/HC parity, ties/cancellation, "
        "row tails, slab views, block=routed, Q8 rows2..8, guards and immutable inputs PASS\n");
    if (bench_rows) CHECK(benchmark(bench_rows));
    ds4_gpu_cleanup();
    CHECK(munmap(model, model_bytes) == 0);
    return 1;
}

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--bench") &&
                     strcmp(argv[1], "--bench-scalar"))) {
        fprintf(stderr, "usage: %s [--bench | --bench-scalar]\n", argv[0]);
        return 1;
    }
    ds4_gpu_set_quality(false);
    const uint32_t bench_rows = argc == 1 ? 0 : !strcmp(argv[1], "--bench-scalar") ? 1 : 437;
    return ds4_gpu_init() && check_all(bench_rows) ? 0 : 1;
}
