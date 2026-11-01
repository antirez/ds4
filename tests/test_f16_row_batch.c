/*
 * F16 exact-row batch matvec gate.
 *
 * ds4_gpu_dsv41_projection_rows() and its strided sibling serve 2..8
 * activation rows (DSpark verification, the batched step, and the draft
 * attention-output groups).  This test pins the contract that every row
 * matches the single-row kernel bit for bit, including the strided grouped
 * layout, partial NR0 tiles and K sizes that do not fill a whole NSG*NF
 * block window.  --bench prints the per-shape legacy/rows timings used to
 * pick the host threshold.
 */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ds4_metal.o references the runtime's TTY probe; this test does not need it. */
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static const float guard_value = 12345.0f;

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "F16 row batch test failed: %s\n", what);
        exit(1);
    }
}

static uint32_t test_seed = 0x5eed1234u;

static uint32_t rnd(void) {
    test_seed ^= test_seed << 13;
    test_seed ^= test_seed >> 17;
    test_seed ^= test_seed << 5;
    return test_seed;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

/* Exact half patterns in [0.5, 1.0]: exponent 14, any 10-bit mantissa. */
static void fill_weights(void *map, uint64_t weight_bytes) {
    uint16_t *p = (uint16_t *)map;
    const uint64_t count = weight_bytes / sizeof(uint16_t);
    for (uint64_t i = 0; i < count; i++)
        p[i] = (uint16_t)(0x3800u | (rnd() & 0x3ffu));
}

static void *alloc_model_map(uint64_t weight_bytes, uint64_t *map_bytes_out) {
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t map_bytes = align_up(weight_bytes, page);
    void *map = NULL;
    require_ok(posix_memalign(&map, (size_t)page, (size_t)map_bytes) == 0,
               "posix_memalign model map");
    memset(map, 0, (size_t)map_bytes);
    fill_weights(map, weight_bytes);
    require_ok(ds4_gpu_set_model_map(map, map_bytes) != 0, "set model map");
    *map_bytes_out = map_bytes;
    return map;
}

static int run_rows(uint32_t in_dim, uint32_t out_dim, uint32_t rows,
                    ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                    const void *map, uint64_t map_bytes) {
    return ds4_gpu_dsv41_projection_rows_strided(
        out, map, map_bytes, 0, in_dim, out_dim, rows, x,
        0, in_dim, 0, out_dim);
}

static void check_bits(const float *ref, const float *act, uint32_t rows,
                       uint32_t out_dim, uint32_t in_dim, const char *what) {
    const uint64_t total = (uint64_t)rows * out_dim;
    for (uint64_t i = 0; i < total; i++) {
        if (memcmp(&ref[i], &act[i], sizeof(float)) != 0) {
            fprintf(stderr,
                    "Mismatch %s in=%u out=%u rows=%u index=%llu ref=%.9g act=%.9g\n",
                    what, in_dim, out_dim, rows, (unsigned long long)i,
                    (double)ref[i], (double)act[i]);
            exit(1);
        }
    }
}

static void run_shape(uint32_t in_dim, uint32_t out_dim) {
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    uint64_t map_bytes = 0;
    void *map = alloc_model_map(weight_bytes, &map_bytes);

    for (uint32_t rows = 1; rows <= 8; rows++) {
        const uint64_t x_floats = (uint64_t)rows * in_dim;
        const uint64_t out_floats = (uint64_t)rows * out_dim;
        const uint64_t guard = 64u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_floats * sizeof(float));
        ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc((out_floats + guard) * sizeof(float));
        ds4_gpu_tensor *act = ds4_gpu_tensor_alloc((out_floats + guard) * sizeof(float));
        ds4_gpu_tensor *contig = ds4_gpu_tensor_alloc((out_floats + guard) * sizeof(float));
        require_ok(x && ref && act && contig, "tensor alloc");

        float *input = (float *)malloc((size_t)(x_floats * sizeof(float)));
        float *ref_host = (float *)malloc((size_t)((out_floats + guard) * sizeof(float)));
        float *act_host = (float *)malloc((size_t)((out_floats + guard) * sizeof(float)));
        float *contig_host = (float *)malloc((size_t)((out_floats + guard) * sizeof(float)));
        require_ok(input && ref_host && act_host && contig_host, "host alloc");

        for (uint64_t i = 0; i < x_floats; i++)
            input[i] = (float)((int)(rnd() % 2049u) - 1024) / 1024.0f;
        require_ok(ds4_gpu_tensor_write(x, 0, input, x_floats * sizeof(float)) != 0,
                   "write activations");
        require_ok(ds4_gpu_tensor_fill_f32(ref, guard_value, out_floats + guard) != 0, "fill ref guard");
        require_ok(ds4_gpu_tensor_fill_f32(act, guard_value, out_floats + guard) != 0, "fill act guard");
        require_ok(ds4_gpu_tensor_fill_f32(contig, guard_value, out_floats + guard) != 0,
                   "fill contig guard");

        /* Reference: the ordinary single-row kernel, once per activation row. */
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(
                x, (uint64_t)r * in_dim * sizeof(float),
                (uint64_t)in_dim * sizeof(float));
            ds4_gpu_tensor *orr = ds4_gpu_tensor_view(
                ref, (uint64_t)r * out_dim * sizeof(float),
                (uint64_t)out_dim * sizeof(float));
            require_ok(xr && orr, "row views");
            require_ok(ds4_gpu_matmul_f16_tensor(
                           orr, map, map_bytes, 0, in_dim, out_dim, xr, 1) != 0,
                       "single-row reference");
            ds4_gpu_tensor_free(xr);
            ds4_gpu_tensor_free(orr);
        }

        /* Actual: the strided exact-row entry point (forced rows kernel). */
        require_ok(run_rows(in_dim, out_dim, rows, act, x, map, map_bytes) != 0,
                   "strided row-batch matvec");

        /* The contiguous entry point too: it picks rows by the host threshold. */
        require_ok(ds4_gpu_dsv41_projection_rows(
                       contig, map, map_bytes, 0, in_dim, out_dim, rows, x) != 0,
                   "contiguous row-batch matvec");

        require_ok(ds4_gpu_tensor_read(ref, 0, ref_host,
                                       (out_floats + guard) * sizeof(float)) != 0,
                   "read ref");
        require_ok(ds4_gpu_tensor_read(act, 0, act_host,
                                       (out_floats + guard) * sizeof(float)) != 0,
                   "read act");
        require_ok(ds4_gpu_tensor_read(contig, 0, contig_host,
                                       (out_floats + guard) * sizeof(float)) != 0,
                   "read contig");

        check_bits(ref_host, act_host, rows, out_dim, in_dim, "strided");
        check_bits(ref_host, contig_host, rows, out_dim, in_dim, "contiguous");
        for (uint64_t i = out_floats; i < out_floats + guard; i++) {
            require_ok(ref_host[i] == guard_value && act_host[i] == guard_value &&
                       contig_host[i] == guard_value, "output guard overwritten");
        }

        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref);
        ds4_gpu_tensor_free(act);
        ds4_gpu_tensor_free(contig);
        free(input);
        free(ref_host);
        free(act_host);
        free(contig_host);
    }

    printf("PASS F16 row batch in=%u out=%u rows=1..8\n", in_dim, out_dim);
    free(map);
}

/* The DSpark attention-output groups are row-major [row][group][width]; the
 * production call passes one group at a time with these strides. */
static void run_grouped_stride(void) {
    const uint32_t groups = 8, width = 4096, rank = 1024, rows = 5;
    const uint64_t out_dim = (uint64_t)groups * rank;
    const uint64_t x_elems = (uint64_t)rows * groups * width;
    const uint64_t out_elems = (uint64_t)rows * groups * rank;
    const uint64_t weight_bytes = out_dim * width * sizeof(uint16_t);
    uint64_t map_bytes = 0;
    void *map = alloc_model_map(weight_bytes, &map_bytes);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_elems * sizeof(float));
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc((out_elems + 64u) * sizeof(float));
    require_ok(x && low, "grouped tensor alloc");
    float *input = (float *)malloc((size_t)(x_elems * sizeof(float)));
    float *got = (float *)malloc((size_t)((out_elems + 64u) * sizeof(float)));
    float *want = (float *)malloc((size_t)((out_elems + 64u) * sizeof(float)));
    require_ok(input && got && want, "grouped host alloc");

    for (uint64_t i = 0; i < x_elems; i++)
        input[i] = (float)((int)(rnd() % 2049u) - 1024) / 1024.0f;
    require_ok(ds4_gpu_tensor_write(x, 0, input, x_elems * sizeof(float)) != 0,
               "grouped write");
    require_ok(ds4_gpu_tensor_fill_f32(low, guard_value, out_elems + 64u) != 0,
               "grouped guard");

    for (uint32_t group = 0; group < groups; group++) {
        require_ok(ds4_gpu_dsv41_projection_rows_strided(
                       low, map, map_bytes, (uint64_t)group * rank * width * sizeof(uint16_t),
                       width, rank, rows, x,
                       (uint64_t)group * width, (uint64_t)groups * width,
                       (uint64_t)group * rank, out_dim) != 0,
                   "grouped strided matvec");
    }

    /* Reference: a contiguous copy of each row/group through one-row matvecs. */
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t group = 0; group < groups; group++) {
            float *src = (float *)malloc((size_t)width * sizeof(float));
            require_ok(src != NULL, "grouped src");
            memcpy(src, input + ((uint64_t)row * groups + group) * width,
                   (size_t)width * sizeof(float));
            ds4_gpu_tensor *xr = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
            ds4_gpu_tensor *yr = ds4_gpu_tensor_alloc((uint64_t)rank * sizeof(float));
            require_ok(xr && yr, "grouped row tensors");
            require_ok(ds4_gpu_tensor_write(xr, 0, src, (uint64_t)width * sizeof(float)) != 0,
                       "grouped row write");
            require_ok(ds4_gpu_matmul_f16_tensor(
                           yr, map, map_bytes, (uint64_t)group * rank * width * sizeof(uint16_t),
                           width, rank, xr, 1) != 0,
                       "grouped row reference");
            require_ok(ds4_gpu_tensor_read(
                           yr, 0, want + ((uint64_t)row * groups + group) * rank,
                           (uint64_t)rank * sizeof(float)) != 0,
                       "grouped row read");
            ds4_gpu_tensor_free(xr);
            ds4_gpu_tensor_free(yr);
            free(src);
        }
    }

    require_ok(ds4_gpu_tensor_read(low, 0, got, (out_elems + 64u) * sizeof(float)) != 0,
               "grouped read");
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t group = 0; group < groups; group++) {
            const uint64_t base = ((uint64_t)row * groups + group) * rank;
            for (uint32_t i = 0; i < rank; i++) {
                if (memcmp(&want[base + i], &got[base + i], sizeof(float)) != 0) {
                    fprintf(stderr,
                            "Grouped mismatch row=%u group=%u col=%u ref=%.9g act=%.9g\n",
                            row, group, i, (double)want[base + i],
                            (double)got[base + i]);
                    exit(1);
                }
            }
        }
    }
    for (uint64_t i = out_elems; i < out_elems + 64u; i++)
        require_ok(got[i] == guard_value, "grouped guard overwritten");

    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(low);
    free(input);
    free(got);
    free(want);
    free(map);
    printf("PASS F16 grouped stride rows=%u groups=%u width=%u rank=%u\n",
           rows, groups, width, rank);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int bench_entry(uint32_t in_dim, uint32_t out_dim, uint32_t rows,
                       ds4_gpu_tensor *out, ds4_gpu_tensor *x,
                       const void *map, uint64_t map_bytes) {
    if (out_dim >= 2048u)
        return ds4_gpu_dsv41_projection_rows(out, map, map_bytes, 0,
                                             in_dim, out_dim, rows, x);
    return ds4_gpu_dsv41_projection_rows_strided(
        out, map, map_bytes, 0, in_dim, out_dim, rows, x,
        0, in_dim, 0, out_dim);
}

static void bench_shape(uint32_t in_dim, uint32_t out_dim) {
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    uint64_t map_bytes = 0;
    void *map = alloc_model_map(weight_bytes, &map_bytes);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)8 * in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)8 * out_dim * sizeof(float));
    require_ok(x && out, "bench tensor alloc");
    float *input = (float *)malloc((size_t)8 * in_dim * sizeof(float));
    require_ok(input != NULL, "bench input");
    for (uint64_t i = 0; i < (uint64_t)8 * in_dim; i++)
        input[i] = (float)((int)(rnd() % 2049u) - 1024) / 1024.0f;
    require_ok(ds4_gpu_tensor_write(x, 0, input, (uint64_t)8 * in_dim * sizeof(float)) != 0,
               "bench write");
    free(input);

    const int iters = (uint64_t)in_dim * out_dim > 50000000ull ? 10 : 20;
    printf("bench in=%u out=%u (entry=%s)\n", in_dim, out_dim,
           out_dim >= 2048u ? "contiguous" : "strided");

    for (uint32_t rows = 2; rows <= 6; rows++) {
        double legacy_ms = 0.0, rows_ms = 0.0;
        setenv("DS4_METAL_F16_ROWS_LEGACY", "1", 1);
        for (int warm = 0; warm < 2; warm++)
            require_ok(bench_entry(in_dim, out_dim, rows, out, x, map, map_bytes) != 0,
                       "bench legacy warm");
        require_ok(ds4_gpu_synchronize() != 0, "bench sync");
        double t0 = now_ms();
        for (int i = 0; i < iters; i++)
            require_ok(bench_entry(in_dim, out_dim, rows, out, x, map, map_bytes) != 0,
                       "bench legacy");
        require_ok(ds4_gpu_synchronize() != 0, "bench sync");
        legacy_ms = (now_ms() - t0) / iters;

        unsetenv("DS4_METAL_F16_ROWS_LEGACY");
        for (int warm = 0; warm < 2; warm++)
            require_ok(bench_entry(in_dim, out_dim, rows, out, x, map, map_bytes) != 0,
                       "bench rows warm");
        require_ok(ds4_gpu_synchronize() != 0, "bench sync");
        t0 = now_ms();
        for (int i = 0; i < iters; i++)
            require_ok(bench_entry(in_dim, out_dim, rows, out, x, map, map_bytes) != 0,
                       "bench rows");
        require_ok(ds4_gpu_synchronize() != 0, "bench sync");
        rows_ms = (now_ms() - t0) / iters;

        printf("  rows=%u legacy=%8.3f ms rows=%8.3f ms ratio=%5.2fx\n",
               rows, legacy_ms, rows_ms, legacy_ms / rows_ms);
        fflush(stdout);
    }

    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(map);
}

/* Production-like measurement for the DSpark attention-output group merge:
 * one stage encodes all eight out=1024 group calls into the same open command
 * batch, so the per-row fallback and the exact-row path differ only in the
 * kernels they enqueue. */
static void bench_strided_group(void) {
    const uint32_t groups = 8, width = 4096, rank = 1024, rows = 5;
    const uint64_t out_stride = (uint64_t)groups * rank;
    const uint64_t x_elems = (uint64_t)rows * groups * width;
    const uint64_t weight_bytes = (uint64_t)groups * rank * width * sizeof(uint16_t);
    uint64_t map_bytes = 0;
    void *map = alloc_model_map(weight_bytes, &map_bytes);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_elems * sizeof(float));
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc((uint64_t)rows * out_stride * sizeof(float));
    require_ok(x && low, "strided bench tensors");
    float *input = (float *)malloc((size_t)(x_elems * sizeof(float)));
    require_ok(input != NULL, "strided bench input");
    for (uint64_t i = 0; i < x_elems; i++)
        input[i] = (float)((int)(rnd() % 2049u) - 1024) / 1024.0f;
    require_ok(ds4_gpu_tensor_write(x, 0, input, x_elems * sizeof(float)) != 0,
               "strided bench write");
    free(input);

    const int stages = 50;
    double ms[2] = {0.0, 0.0};
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 0) setenv("DS4_METAL_F16_ROWS_LEGACY", "1", 1);
        else unsetenv("DS4_METAL_F16_ROWS_LEGACY");
        for (int warm = 0; warm < 3; warm++) {
            require_ok(ds4_gpu_begin_commands() != 0, "bench begin");
            for (uint32_t group = 0; group < groups; group++)
                require_ok(ds4_gpu_dsv41_projection_rows_strided(
                               low, map, map_bytes,
                               (uint64_t)group * rank * width * sizeof(uint16_t),
                               width, rank, rows, x,
                               (uint64_t)group * width, (uint64_t)groups * width,
                               (uint64_t)group * rank, out_stride) != 0,
                           "bench grouped");
            require_ok(ds4_gpu_end_commands() != 0, "bench end");
        }
        require_ok(ds4_gpu_synchronize() != 0, "bench sync");
        const double t0 = now_ms();
        for (int stage = 0; stage < stages; stage++) {
            require_ok(ds4_gpu_begin_commands() != 0, "bench begin");
            for (uint32_t group = 0; group < groups; group++)
                require_ok(ds4_gpu_dsv41_projection_rows_strided(
                               low, map, map_bytes,
                               (uint64_t)group * rank * width * sizeof(uint16_t),
                               width, rank, rows, x,
                               (uint64_t)group * width, (uint64_t)groups * width,
                               (uint64_t)group * rank, out_stride) != 0,
                           "bench grouped");
            require_ok(ds4_gpu_end_commands() != 0, "bench end");
        }
        require_ok(ds4_gpu_synchronize() != 0, "bench sync");
        ms[mode] = (now_ms() - t0) / stages;
    }
    printf("bench strided 1024x5 groups=%u: legacy %.3f ms/stage, rows %.3f ms/stage, ratio %.2fx\n",
           groups, ms[0], ms[1], ms[0] / ms[1]);

    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(low);
    free(map);
}

static void run_bench(void) {
    static const struct { uint32_t in_dim, out_dim; } shapes[] = {
        {1280, 32768},   /* draft q_b */
        {2304,  5120},   /* shared down */
        {5120,  2304},   /* shared gate/up */
        {8192,  5120},   /* draft attn_output_b */
        {4096,  8192},   /* draft attn_output_a (grouped in production) */
        {6144, 25600},   /* target Engram KV */
        {5120,  1280},   /* draft q_a */
        {5120,   512},   /* draft kv */
        {4096,  1024},   /* draft attention-output group */
        {1280,  4096},   /* target indexer q_b */
        {20480,   24},   /* target HC mix (expected to stay on the old path) */
    };
    for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        bench_shape(shapes[i].in_dim, shapes[i].out_dim);
}

int main(int argc, char **argv) {
    require_ok(ds4_gpu_init() != 0, "Metal init");

    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        run_bench();
        ds4_gpu_cleanup();
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--bench-strided") == 0) {
        bench_strided_group();
        ds4_gpu_cleanup();
        return 0;
    }

    static const struct { uint32_t in_dim, out_dim; } shapes[] = {
        {   32,     1},
        {   32,     3},
        {   64,     5},
        {   96,     7},
        {  128,    17},
        {  160,    31},
        {  256,    63},
        {  512,    65},
        { 1024,   127},
        { 2560,   129},
        { 4096,   513},
        { 2560,  4097},
        /* Real V4.1 / DSpark F16 shapes, the partial last tile, and K
         * smaller than one NSG*NF block window. */
        { 1280, 32768},
        { 2304,  5120},
        { 5120,  2304},
        { 6144, 25600},
        { 8192,  5120},
        { 4096,  8192},
        { 5120,  1280},
        { 5120,   512},
        { 4096,  1024},
        { 1280,  4096},
        {20480,    24},
        {   32,  4097},
        {   96,  2304},
    };

    for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        run_shape(shapes[i].in_dim, shapes[i].out_dim);

    run_grouped_stride();

    ds4_gpu_cleanup();
    return 0;
}
