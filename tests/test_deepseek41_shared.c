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

enum { IN = 5120, OUT = 2304, PAD = 4 };
enum { X, GATE, UP, REF, FUSED, NT };
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
typedef struct {
    void *model;
    uint64_t bytes, gate_offset, up_offset;
    ds4_gpu_tensor *storage[NT], *tensor[NT];
    float limit;
} fixture;

static uint32_t seed = 7919;
static uint32_t random_bits(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return seed;
}

static uint32_t width(unsigned i) { return i == X ? IN : OUT; }

static int sequence(fixture *f, int fused) {
    if (fused) return ds4_gpu_dsv41_shared_swiglu(f->tensor[FUSED],
        f->model, f->bytes, f->gate_offset, f->up_offset,
        IN, OUT, f->tensor[X], f->limit) > 0;
    /* This is the current V4.1 graph, including both projection boundaries. */
    return ds4_gpu_dsv41_q8_bf16_rows(f->tensor[GATE], f->model, f->bytes,
        f->gate_offset, IN, OUT, f->tensor[X], 1) &&
        ds4_gpu_dsv41_q8_bf16_rows(f->tensor[UP], f->model, f->bytes,
        f->up_offset, IN, OUT, f->tensor[X], 1) &&
        ds4_gpu_swiglu_tensor(f->tensor[REF], f->tensor[GATE], f->tensor[UP],
            OUT, f->limit, 1.0f) &&
        ds4_gpu_dsv41_quantize(f->tensor[REF], OUT, 1, DS4_V41_BF16);
}

static int initialize(fixture *f) {
    CHECK(sizeof(q8_block) == 34);
    const uint64_t matrix = (uint64_t)IN * OUT / 32u * sizeof(q8_block);
    f->gate_offset = 128;
    f->up_offset = f->gate_offset + matrix + 128;
    f->bytes = f->up_offset + matrix + 128;
    f->model = mmap(NULL, f->bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(f->model != MAP_FAILED);
    for (unsigned i = 0; i < NT; i++) {
        f->storage[i] = ds4_gpu_tensor_alloc((uint64_t)(width(i) + 2 * PAD) * 4);
        CHECK(f->storage[i]);
        f->tensor[i] = ds4_gpu_tensor_view(f->storage[i], PAD * 4, width(i) * 4);
        CHECK(f->tensor[i] && ds4_gpu_tensor_contents(f->tensor[i]));
    }
    CHECK(ds4_gpu_set_model_map(f->model, f->bytes));
    return 1;
}

static void fill(fixture *f, unsigned pattern) {
    float *x = ds4_gpu_tensor_contents(f->tensor[X]);
    const uint32_t edge[] = {
        0x3f807fffu, 0x3f808000u, 0x3f808001u, 0x3f817fffu,
        0x3f818000u, 0x3f818001u, 0xbf808000u, 0xbf818000u,
        0x00000000u, 0x80000000u, 0x00800000u, 0x80800000u,
        0x40df7fffu, 0x40df8000u, 0x40df8001u, 0xc0df8000u,
        0x41207fffu, 0x41208000u, 0x41208001u, 0xc1208000u,
    };
    for (unsigned i = 0; i < IN; i++) {
        if (pattern == 1) {
            uint32_t bits = edge[i % (sizeof(edge) / sizeof(*edge))];
            memcpy(x + i, &bits, 4);
        } else if (pattern == 2) {
            x[i] = i % 32u < 2u ? 8192.0f : 0x1p-12f;
        } else {
            x[i] = (int)(random_bits() % 65537u) / 32768.0f - 1.0f;
        }
    }
    for (unsigned side = 0; side < 2; side++) {
        q8_block *w = (q8_block *)((char *)f->model +
            (side ? f->up_offset : f->gate_offset));
        for (unsigned row = 0; row < OUT; row++) {
            for (unsigned b = 0; b < IN / 32; b++) {
                q8_block *q = w + (size_t)row * (IN / 32) + b;
                q->d = pattern ? 0x3c00u : (uint16_t)(0x1800u + random_bits() % 0x1800u);
                for (unsigned j = 0; j < 32; j++)
                    q->qs[j] = pattern ? 0 : (int8_t)(random_bits() & 255u);
                if (pattern == 1 && !b) q->qs[(row + 7u * side) % 20u] = 1;
                if (pattern == 2) {
                    /* Exact large cancellations precede independently signed
                     * tiny residues in both matrices and every K partition. */
                    q->qs[0] = (row + side + b) % 2 ? 127 : -127;
                    q->qs[1] = -q->qs[0];
                    q->qs[2] = (int)((row + 3 * side + b) % 7) - 3;
                    q->qs[3] = side ? -2 : 1;
                }
            }
        }
    }
}

static int parity(fixture *f) {
    uint32_t input[IN];
    memcpy(input, ds4_gpu_tensor_contents(f->tensor[X]), sizeof(input));
    for (unsigned i = 0; i < NT; i++) {
        uint32_t *p = ds4_gpu_tensor_contents(f->storage[i]);
        for (unsigned j = 0; j < PAD; j++) p[j] = p[PAD + width(i) + j] = 0xdeadbeefu;
    }
    CHECK(ds4_gpu_tensor_fill_f32(f->tensor[FUSED], NAN, OUT));
    CHECK(ds4_gpu_begin_commands());
    CHECK(sequence(f, 0) && sequence(f, 1));
    CHECK(ds4_gpu_end_commands());
    const uint32_t *a = ds4_gpu_tensor_contents(f->tensor[REF]);
    const uint32_t *b = ds4_gpu_tensor_contents(f->tensor[FUSED]);
    for (unsigned i = 0; i < OUT; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "shared row=%u limit=%g expected=%08x got=%08x\n",
                    i, f->limit, a[i], b[i]);
            return 0;
        }
        CHECK((b[i] & 0xffffu) == 0);
    }
    CHECK(!memcmp(input, ds4_gpu_tensor_contents(f->tensor[X]), sizeof(input)));
    for (unsigned i = 0; i < NT; i++) {
        const uint32_t *p = ds4_gpu_tensor_contents(f->storage[i]);
        for (unsigned j = 0; j < PAD; j++)
            CHECK(p[j] == 0xdeadbeefu && p[PAD + width(i) + j] == 0xdeadbeefu);
    }
    return 1;
}

static int rejected(fixture *f) {
    uint32_t input[IN];
    memcpy(input, ds4_gpu_tensor_contents(f->tensor[X]), sizeof(input));
    uint32_t *p = ds4_gpu_tensor_contents(f->storage[FUSED]);
    for (unsigned i = 0; i < OUT + 2 * PAD; i++) p[i] = 0xdeadbeefu;
#define TRY(y, bytes, go, uo, k, m, x, limit) \
    ds4_gpu_dsv41_shared_swiglu(y, f->model, bytes, go, uo, k, m, x, limit)
#define BASE(y, x, limit) TRY(y, f->bytes, f->gate_offset, f->up_offset, IN, OUT, x, limit)
    CHECK(BASE(NULL, f->tensor[X], 1) == 0);
    CHECK(BASE(f->tensor[FUSED], NULL, 1) == 0);
    CHECK(BASE(f->tensor[X], f->tensor[X], 1) == 0);
    CHECK(BASE(f->tensor[FUSED], f->tensor[X], NAN) == 0);
    CHECK(BASE(f->tensor[FUSED], f->tensor[X], INFINITY) == 0);
    CHECK(BASE(f->tensor[FUSED], f->tensor[X], -1) == 0);
    CHECK(TRY(f->tensor[FUSED], f->up_offset, f->gate_offset, f->up_offset,
        IN, OUT, f->tensor[X], 1) == 0);
    CHECK(TRY(f->tensor[FUSED], f->bytes, f->gate_offset + 1, f->up_offset,
        IN, OUT, f->tensor[X], 1) == 0);
    CHECK(TRY(f->tensor[FUSED], f->bytes, f->gate_offset, f->up_offset,
        IN - 32, OUT, f->tensor[X], 1) == 0);
    CHECK(TRY(f->tensor[FUSED], f->bytes, f->gate_offset, f->up_offset,
        IN, OUT - 2, f->tensor[X], 1) == 0);
    ds4_gpu_tensor *bad = ds4_gpu_tensor_view(f->storage[X], 4, IN * 4);
    CHECK(bad && BASE(f->tensor[FUSED], bad, 1) == 0);
    ds4_gpu_tensor_free(bad);
    /* Different view objects still alias the same input allocation. */
    bad = ds4_gpu_tensor_view(f->storage[X], (PAD + 4) * 4, OUT * 4);
    CHECK(bad && BASE(bad, f->tensor[X], 1) == 0);
    ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(f->storage[FUSED], PAD * 4, OUT * 4 - 4);
    CHECK(bad && BASE(bad, f->tensor[X], 1) == 0);
    ds4_gpu_tensor_free(bad);
    const char *ns[] = {"1", "2", "8"};
    for (unsigned i = 0; i < sizeof(ns) / sizeof(*ns); i++) {
        CHECK(setenv("DS4_METAL_Q8_MV_NSG", ns[i], 1) == 0);
        CHECK(BASE(f->tensor[FUSED], f->tensor[X], 1) == 0);
    }
    CHECK(setenv("DS4_METAL_Q8_MV_NSG", "4", 1) == 0);
    ds4_gpu_set_quality(true);
    CHECK(BASE(f->tensor[FUSED], f->tensor[X], 1) == 0);
    ds4_gpu_set_quality(false);
    CHECK(!memcmp(input, ds4_gpu_tensor_contents(f->tensor[X]), sizeof(input)));
    for (unsigned i = 0; i < OUT + 2 * PAD; i++) CHECK(p[i] == 0xdeadbeefu);
#undef BASE
#undef TRY
    return 1;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int benchmark(fixture *f) {
    enum { SAMPLES = 11, ITERATIONS = 32 };
    double samples[2][SAMPLES];
    for (unsigned sample = 0; sample < SAMPLES + 2; sample++) {
        double elapsed[2] = {0, 0};
        for (unsigned turn = 0; turn < 4; turn++) {
            const unsigned mode = (turn == 1 || turn == 2) ^ (sample & 1u);
            const double start = seconds();
            CHECK(ds4_gpu_begin_commands());
            for (unsigned i = 0; i < ITERATIONS; i++) CHECK(sequence(f, mode));
            CHECK(ds4_gpu_end_commands());
            elapsed[mode] += seconds() - start;
        }
        if (sample >= 2) for (unsigned mode = 0; mode < 2; mode++)
            samples[mode][sample - 2] = elapsed[mode] * 1e6 / (2 * ITERATIONS);
    }
    for (unsigned mode = 0; mode < 2; mode++)
        qsort(samples[mode], SAMPLES, sizeof(double), compare_double);
    fprintf(stderr, "V4.1 shared Q8 5120->2304 batched wall median us: "
        "four_dispatch=%.3f fused=%.3f speedup=%.4f (alternating ABBA, %u samples)\n",
        samples[0][SAMPLES / 2], samples[1][SAMPLES / 2],
        samples[0][SAMPLES / 2] / samples[1][SAMPLES / 2], SAMPLES);
    return 1;
}

static int check_all(int bench) {
    fixture f = {0};
    CHECK(initialize(&f));
    const float limits[] = {0, 1.0e-6f, 1.001e-6f, 1.001f, 7.0f, 10.0f};
    for (unsigned pattern = 0; pattern < 3; pattern++) {
        fill(&f, pattern);
        for (unsigned j = 0; j < sizeof(limits) / sizeof(*limits); j++) {
            f.limit = limits[j];
            CHECK(parity(&f));
        }
    }
    CHECK(rejected(&f));
    fprintf(stderr, "V4.1 shared Q8: exact four-dispatch parity; BF16 ties, cancellation, clamp, views, guards, rejection PASS\n");
    if (bench) {
        fill(&f, 0);
        f.limit = 10;
        CHECK(parity(&f) && benchmark(&f));
    }
    for (unsigned i = 0; i < NT; i++) {
        ds4_gpu_tensor_free(f.tensor[i]);
        ds4_gpu_tensor_free(f.storage[i]);
    }
    ds4_gpu_cleanup();
    CHECK(munmap(f.model, f.bytes) == 0);
    return 1;
}

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--bench"))) {
        fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 1;
    }
    if (setenv("DS4_METAL_Q8_MV_NSG", "4", 1)) return 1;
    ds4_gpu_set_quality(false);
    return ds4_gpu_init() && check_all(argc == 2) ? 0 : 1;
}
