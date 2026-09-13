#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static uint32_t state = 1743;
static uint32_t random_bits(void) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const unsigned char *p = data;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; i++)
        hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

static void check(uint32_t blocks, uint32_t outputs, uint32_t rows) {
    const uint32_t width = blocks * 32u;
    const size_t bytes = (size_t)outputs * blocks * 34u;
    unsigned char *model = malloc(bytes);
    float *input = malloc((size_t)rows * width * sizeof(float));
    float *reference = malloc((size_t)rows * outputs * sizeof(float));
    float *actual = malloc((size_t)rows * outputs * sizeof(float));
    CHECK(model && input && reference && actual);
    for (uint32_t b = 0; b < outputs * blocks; b++) {
        const uint16_t scale = 0x2000u + random_bits() % 1024u;
        memcpy(model + (size_t)b * 34u, &scale, 2);
        for (uint32_t k = 0; k < 32u; k++) model[(size_t)b * 34u + 2u + k] = random_bits() >> 24;
    }
    for (uint32_t i = 0; i < rows * width; i++)
        input[i] = ((int32_t)(random_bits() >> 16) - 32768) / 8192.0f;
    CHECK(ds4_gpu_set_model_map(model, bytes));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((size_t)rows * width * sizeof(float));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc((size_t)rows * outputs * sizeof(float));
    CHECK(x && y && ds4_gpu_tensor_write(x, 0, input, (size_t)rows * width * sizeof(float)));
    for (uint32_t r = 0; r < rows; r++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, (size_t)r * width * 4u, width * 4u);
        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(y, (size_t)r * outputs * 4u, outputs * 4u);
        CHECK(xr && yr && ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            yr, model, bytes, 0, width, outputs, xr, 1));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(yr);
    }
    CHECK(ds4_gpu_tensor_read(y, 0, reference, (size_t)rows * outputs * 4u));
    const uint32_t counts[] = {1, 3, 8, 16, 127, 128, rows};
    for (size_t c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        if (counts[c] > rows) continue;
        CHECK(ds4_gpu_tensor_fill_f32(y, NAN, (size_t)rows * outputs));
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(y, model, bytes,
            0, width, outputs, x, counts[c]));
        CHECK(ds4_gpu_tensor_read(y, 0, actual, (size_t)rows * outputs * 4u));
        CHECK(!memcmp(actual, reference, (size_t)counts[c] * outputs * 4u));
        for (uint32_t i = counts[c] * outputs; i < rows * outputs; i++) CHECK(isnan(actual[i]));
    }
    if (outputs > 1) {
        const uint32_t first[] = {0, outputs / 2};
        const uint32_t count[] = {outputs / 2, outputs - outputs / 2};
        for (uint32_t rank = 0; rank < 2; rank++) {
            CHECK(ds4_gpu_tensor_fill_f32(y, NAN, (size_t)rows * outputs));
            CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(y, model, bytes,
                (uint64_t)first[rank] * blocks * 34u, width, count[rank], x, rows));
            CHECK(ds4_gpu_tensor_read(y, 0, actual, (size_t)rows * outputs * 4u));
            for (uint32_t r = 0; r < rows; r++)
                CHECK(!memcmp(actual + (size_t)r * count[rank],
                    reference + (size_t)r * outputs + first[rank], count[rank] * 4u));
            for (size_t i = (size_t)rows * count[rank]; i < (size_t)rows * outputs; i++)
                CHECK(isnan(actual[i]));
        }
    }
    fprintf(stderr, "CUDA Q8 exact rows blocks=%u outputs=%u rows=%u, unpadded weights: PASS hash=%016" PRIx64 "\n",
        blocks, outputs, rows, hash_bytes(reference, (size_t)rows * outputs * 4u));
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(y);
    ds4_gpu_cleanup();
    free(model); free(input); free(reference); free(actual);
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int compare_time(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void bench(uint32_t width, uint32_t outputs, uint32_t rows) {
    const uint32_t blocks = width / 32;
    const size_t bytes = (size_t)outputs * blocks * 34;
    unsigned char *model = malloc(bytes);
    float *input = malloc((size_t)rows * width * 4);
    CHECK(model && input && ds4_gpu_init());
    for (size_t b = 0; b < (size_t)outputs * blocks; b++) {
        const uint16_t scale = 0x2000u + random_bits() % 1024u;
        memcpy(model + b * 34, &scale, 2);
        for (uint32_t k = 0; k < 32; k++) model[b * 34 + 2 + k] = random_bits() >> 24;
    }
    for (size_t i = 0; i < (size_t)rows * width; i++)
        input[i] = ((int32_t)(random_bits() >> 16) - 32768) / 8192.0f;
    CHECK(ds4_gpu_set_model_map(model, bytes));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((size_t)rows * width * 4);
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc((size_t)rows * outputs * 4);
    CHECK(x && y && ds4_gpu_tensor_write(x, 0, input, (size_t)rows * width * 4));
    double elapsed[9];
    for (unsigned i = 0; i < 11; i++) {
        const double start = seconds();
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(y, model, bytes, 0, width, outputs, x, rows));
        CHECK(ds4_gpu_synchronize());
        if (i >= 2) elapsed[i - 2] = seconds() - start;
    }
    qsort(elapsed, 9, sizeof(*elapsed), compare_time);
    printf("Q8 exact matmul width=%u outputs=%u rows=%u median=%.3f ms\n",
        width, outputs, rows, elapsed[4] * 1000);
    ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(x); ds4_gpu_cleanup();
    free(input); free(model);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--tp-head")) {
        CHECK(ds4_gpu_init());
        check(160, 129280, 8);
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "--bench")) {
        bench(4096, 1024, 2048);
        bench(8192, 5120, 2048);
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "--bench-decode")) {
        bench(5120, 2048, 1);
        bench(1024, 32768, 1);
        bench(4096, 4096, 1);
        bench(8192, 5120, 1);
        bench(5120, 129280, 1);
        return 0;
    }
    if (argc != 1) return 2;
    const uint32_t blocks[] = {1, 3, 31, 32, 33, 128, 256};
    const uint32_t outputs[] = {1, 3, 63, 64, 65};
    for (size_t b = 0; b < sizeof(blocks) / sizeof(*blocks); b++)
        for (size_t o = 0; o < sizeof(outputs) / sizeof(*outputs); o++) {
            CHECK(ds4_gpu_init());
            check(blocks[b], outputs[o], 17);
        }
    for (uint32_t b = 128; b <= 256; b *= 2) {
        CHECK(ds4_gpu_init()); check(b, 4096, 129);
        CHECK(ds4_gpu_init()); check(b, 4161, 257);
    }
    for (uint32_t b = 31; b <= 33; b++) {
        CHECK(ds4_gpu_init()); check(b, 1025, 3);
    }
    CHECK(ds4_gpu_init()); check(160, 129280, 8);
    return 0;
}
