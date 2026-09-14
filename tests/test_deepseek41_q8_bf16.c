#include "ds4_gpu.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
typedef struct { uint32_t in, out; } shape;
static const shape shapes[] = {
    {5120, 1280}, {1280, 32768}, {5120, 512}, {5120, 2304}, {2304, 5120}
};
static uint32_t seed = 7919;
static uint32_t random_bits(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return seed;
}

static int project(ds4_gpu_tensor *out, const void *model, uint64_t bytes,
                   shape s, const ds4_gpu_tensor *in, uint32_t rows, int fused) {
    if (fused) return ds4_gpu_dsv41_q8_bf16_rows(out, model, bytes, 128,
                                               s.in, s.out, in, rows);
    const int ok = rows == 1 ?
        ds4_gpu_matmul_q8_0_tensor(out, model, bytes, 128, s.in, s.out, in, 1) :
        ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out, model, bytes, 128,
                                                   s.in, s.out, in, rows);
    return ok && ds4_gpu_dsv41_quantize(out, s.out, rows, DS4_V41_BF16);
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

static int benchmark(ds4_gpu_tensor *out, const void *model, uint64_t bytes,
                     shape s, const ds4_gpu_tensor *in, uint32_t rows) {
    enum { REPEATS = 7, ITERATIONS = 16 };
    double samples[2][REPEATS];
    for (unsigned run = 0; run < REPEATS + 2; run++) {
        /* Reverse the order each round, and exclude both warmup rounds. */
        for (unsigned turn = 0; turn < 2; turn++) {
            const unsigned mode = (run + turn) & 1u;
            const double start = seconds();
            CHECK(ds4_gpu_begin_commands());
            for (unsigned i = 0; i < ITERATIONS; i++)
                CHECK(project(out, model, bytes, s, in, rows, mode));
            CHECK(ds4_gpu_end_commands());
            if (run >= 2) samples[mode][run - 2] =
                (seconds() - start) * 1e6 / ITERATIONS;
        }
    }
    for (unsigned mode = 0; mode < 2; mode++)
        qsort(samples[mode], REPEATS, sizeof(double), compare_double);
    fprintf(stderr, "V4.1 Q8 BF16 %u->%u rows=%u median us: separate=%.3f fused=%.3f speedup=%.4f\n",
            s.in, s.out, rows, samples[0][REPEATS / 2], samples[1][REPEATS / 2],
            samples[0][REPEATS / 2] / samples[1][REPEATS / 2]);
    return 1;
}

static int check_all(int bench) {
    enum { MAX_IN = 5120, MAX_OUT = 32768, MAX_ROWS = 8, PAD = 3 };
    const uint64_t model_bytes = 128u + (uint64_t)1280u * MAX_OUT / 32u * sizeof(q8_block);
    void *model = NULL;
    CHECK(sizeof(q8_block) == 34);
    CHECK(posix_memalign(&model, getpagesize(), model_bytes) == 0);
    memset(model, 0, model_bytes);
    q8_block *weights = (q8_block *)((char *)model + 128);
    const size_t x_count = MAX_ROWS * MAX_IN, y_count = MAX_ROWS * MAX_OUT;
    ds4_gpu_tensor *x_storage = ds4_gpu_tensor_alloc((x_count + 2 * PAD) * sizeof(float));
    ds4_gpu_tensor *y_storage = ds4_gpu_tensor_alloc((y_count + 2 * PAD) * sizeof(float));
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(y_count * sizeof(float));
    CHECK(x_storage && y_storage && reference);
    ds4_gpu_tensor *in = ds4_gpu_tensor_view(x_storage, PAD * sizeof(float), x_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_view(y_storage, PAD * sizeof(float), y_count * sizeof(float));
    CHECK(x_storage && y_storage && reference && in && out);
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    float *x = ds4_gpu_tensor_contents(in);
    uint32_t *guard = ds4_gpu_tensor_contents(y_storage);
    CHECK(x && guard);
    const uint32_t ns[] = {2, 4, 8}, counts[] = {1, 2, 8};
    const char *prior_nsg = getenv("DS4_METAL_Q8_MV_NSG");
    char *saved_nsg = prior_nsg ? strdup(prior_nsg) : NULL;
    CHECK(!prior_nsg || saved_nsg);
    for (unsigned si = 0; si < sizeof(shapes) / sizeof(*shapes); si++) {
        const shape s = shapes[si];
        const size_t blocks = (size_t)s.in * s.out / 32u;
        for (unsigned pattern = 0; pattern < 2; pattern++) {
            for (size_t i = 0; i < blocks; i++) {
                weights[i].d = pattern ? 0x3c00u : (uint16_t)(0x2000u + random_bits() % 0x1400u);
                for (unsigned j = 0; j < 32; j++)
                    weights[i].qs[j] = pattern ? 0 : (int8_t)(random_bits() & 255u);
            }
            for (size_t i = 0; i < (size_t)MAX_ROWS * s.in; i++) {
                if (pattern) {
                    /* Single-term dot products land below/on/above even and
                     * odd BF16 ties, on both sides of zero. */
                    const uint32_t ties[] = {0x3f807fffu, 0x3f808000u, 0x3f808001u,
                        0x3f817fffu, 0x3f818000u, 0x3f818001u, 0xbf808000u, 0xbf818000u};
                    const uint32_t bits = ties[i % 8u];
                    memcpy(x + i, &bits, sizeof(bits));
                } else {
                    x[i] = (int32_t)(random_bits() % 65537u) / 8192.0f - 4.0f;
                    if (i % 31u == 0) x[i] *= 0x1p-20f;
                }
            }
            if (pattern) for (uint32_t row = 0; row < s.out; row++)
                weights[(size_t)row * (s.in / 32u)].qs[row % 8u] = 1;
            for (unsigned ni = 0; ni < sizeof(ns) / sizeof(*ns); ni++) {
                char nsg[8];
                snprintf(nsg, sizeof(nsg), "%u", ns[ni]);
                CHECK(setenv("DS4_METAL_Q8_MV_NSG", nsg, 1) == 0);
                for (unsigned ri = 0; ri < sizeof(counts) / sizeof(*counts); ri++) {
                    const uint32_t rows = counts[ri];
                    const size_t count = (size_t)rows * s.out;
                    for (size_t i = 0; i < y_count + 2 * PAD; i++) guard[i] = 0xdeadbeefu;
                    CHECK(ds4_gpu_begin_commands());
                    /* The original single-row API is the independent oracle
                     * for batched addressing and for its chosen reduction. */
                    for (uint32_t r = 0; r < rows; r++) {
                        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(in,
                            (uint64_t)r * s.in * sizeof(float), s.in * sizeof(float));
                        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(reference,
                            (uint64_t)r * s.out * sizeof(float), s.out * sizeof(float));
                        CHECK(xr && yr && ds4_gpu_matmul_q8_0_tensor(yr, model, model_bytes,
                                                                  128, s.in, s.out, xr, 1));
                        ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
                    }
                    CHECK(ds4_gpu_dsv41_quantize(reference, s.out, rows, DS4_V41_BF16));
                    CHECK(project(out, model, model_bytes, s, in, rows, 1));
                    CHECK(ds4_gpu_end_commands());
                    CHECK(memcmp(ds4_gpu_tensor_contents(reference), guard + PAD, count * 4u) == 0);
                    for (unsigned i = 0; i < PAD; i++) CHECK(guard[i] == 0xdeadbeefu);
                    for (size_t i = PAD + count; i < y_count + 2 * PAD; i++)
                        CHECK(guard[i] == 0xdeadbeefu);
                    if (bench && !pattern && ns[ni] == 4)
                        CHECK(benchmark(out, model, model_bytes, s, in, rows));
                }
            }
        }
        fprintf(stderr, "V4.1 Q8 BF16 %u->%u: exact rows=1,2,8 NSG=2,4,8; ties/views/guards PASS\n",
                s.in, s.out);
    }
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, model_bytes, 128, 5120, 511, in, 1));
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, model_bytes, 129, 5120, 512, in, 1));
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, model_bytes, 128, 5119, 512, in, 1));
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, model_bytes, 128, 5120, 512, in, 0));
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, model_bytes, 128, 5120, 512, in, MAX_ROWS + 1));
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, 128, 128, 5120, 512, in, 1));
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, model_bytes, 128, UINT32_MAX, 512, in, 1));
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, model_bytes, 128, 5120, UINT32_MAX - 1u, in, 1));
    ds4_gpu_tensor *unaligned = ds4_gpu_tensor_view(x_storage, 1, 5120 * sizeof(float));
    CHECK(unaligned);
    CHECK(!ds4_gpu_dsv41_q8_bf16_rows(out, model, model_bytes, 128, 5120, 512, unaligned, 1));
    ds4_gpu_tensor_free(unaligned);
    if (saved_nsg) { CHECK(setenv("DS4_METAL_Q8_MV_NSG", saved_nsg, 1) == 0); free(saved_nsg); }
    else CHECK(unsetenv("DS4_METAL_Q8_MV_NSG") == 0);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(in); ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(y_storage); ds4_gpu_tensor_free(x_storage);
    ds4_gpu_cleanup(); free(model);
    return 1;
}

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--bench"))) {
        fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 1;
    }
    return ds4_gpu_init() && check_all(argc == 2) ? 0 : 1;
}
