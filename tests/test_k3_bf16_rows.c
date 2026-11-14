/* Compare batched BF16 projections with the unchanged single-token kernel.
 * Anonymous synthetic weights keep the mapping below 3 MiB; no model is used. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"
#include "k3_metal.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static uint32_t mix(uint32_t value) {
    value ^= value >> 16;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15;
    value *= UINT32_C(0x846ca68b);
    return value ^ (value >> 16);
}

static int guards_intact(const uint8_t *bytes, uint64_t size,
                         uint64_t offset, uint64_t active) {
    for (uint64_t i = 0; i < size; i++) {
        if ((i < offset || i >= offset + active) && bytes[i] != 0xa5)
            return 0;
    }
    return 1;
}

static int test_shape(uint32_t in_dim, uint32_t out_dim) {
    enum { MAX_ROWS = 8, INPUT_OFFSET = 64, OUTPUT_OFFSET = 128, GUARD = 128 };
    const uint64_t input_row = (uint64_t)in_dim * sizeof(float);
    const uint64_t output_row = (uint64_t)out_dim * sizeof(float);
    const uint64_t input_size = INPUT_OFFSET + MAX_ROWS * input_row + GUARD;
    const uint64_t output_size = OUTPUT_OFFSET + MAX_ROWS * output_row + GUARD;
    const uint64_t weight_size = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const long page = sysconf(_SC_PAGESIZE);
    uint64_t mapping_size = 0;
    uint8_t *mapping = MAP_FAILED;
    uint8_t *inputs = NULL, *input_check = NULL, *actual = NULL, *expected = NULL;
    ds4_gpu_tensor *input_base = NULL, *output_base = NULL, *reference_base = NULL;
    ds4_gpu_tensor *input = NULL, *output = NULL;
    ds4_gpu_tensor *input_rows[MAX_ROWS] = {0}, *output_rows[MAX_ROWS] = {0};
    int ok = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "K3 BF16 rows %ux%u: line %d: %s\n", \
                in_dim, out_dim, __LINE__, #condition); \
        goto cleanup; \
    } \
} while (0)

    CHECK(page > 0);
    const uint64_t weight_offset = (uint64_t)page + 128u;
    mapping_size = (weight_offset + weight_size + GUARD + (uint64_t)page - 1u)
        / (uint64_t)page * (uint64_t)page;
    CHECK(mapping_size < UINT64_C(3) * 1024u * 1024u);
    mapping = mmap(NULL, (size_t)mapping_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(mapping != MAP_FAILED);
    memset(mapping, 0xa5, (size_t)mapping_size);
    uint16_t *weights = (uint16_t *)(mapping + weight_offset);
    for (uint64_t i = 0; i < weight_size / sizeof(*weights); i++) {
        const uint32_t hash = mix((uint32_t)i + UINT32_C(0x9e3779b9));
        weights[i] = (uint16_t)((hash & 0x8000u) |
            ((119u + ((hash >> 8) % 5u)) << 7) | (hash & 127u));
    }
    inputs = malloc((size_t)input_size);
    input_check = malloc((size_t)input_size);
    actual = malloc((size_t)output_size);
    expected = malloc((size_t)output_size);
    CHECK(inputs && input_check && actual && expected);
    memset(inputs, 0xa5, (size_t)input_size);
    for (uint64_t i = 0; i < MAX_ROWS * input_row / sizeof(float); i++) {
        const uint32_t hash = mix((uint32_t)i + UINT32_C(0x13579bdf));
        /* Mixed signs and full FP32 mantissas expose cancellation and any
         * accidental narrowing of activations to half precision. */
        const uint32_t bits = (hash & UINT32_C(0x80000000)) |
            ((120u + ((hash >> 24) % 6u)) << 23) |
            (hash & UINT32_C(0x7fffff));
        memcpy(inputs + INPUT_OFFSET + i * sizeof(float), &bits, sizeof(bits));
    }
    input_base = ds4_gpu_tensor_alloc_managed(input_size);
    output_base = ds4_gpu_tensor_alloc_managed(output_size);
    reference_base = ds4_gpu_tensor_alloc_managed(output_size);
    CHECK(input_base && output_base && reference_base);
    input = ds4_gpu_tensor_view(input_base, INPUT_OFFSET, MAX_ROWS * input_row);
    output = ds4_gpu_tensor_view(output_base, OUTPUT_OFFSET, MAX_ROWS * output_row);
    CHECK(input && output);
    for (uint32_t row = 0; row < MAX_ROWS; row++) {
        input_rows[row] = ds4_gpu_tensor_view(input, row * input_row, input_row);
        output_rows[row] = ds4_gpu_tensor_view(reference_base,
            OUTPUT_OFFSET + row * output_row, output_row);
        CHECK(input_rows[row] && output_rows[row]);
    }
    CHECK(ds4_gpu_tensor_write(input_base, 0, inputs, input_size));
    CHECK(ds4_gpu_set_model_map_spans(mapping, mapping_size,
        &weight_offset, &weight_size, 1, weight_size));

    for (uint32_t rows = 4; rows <= MAX_ROWS; rows++) {
        memset(actual, 0xa5, (size_t)output_size);
        CHECK(ds4_gpu_tensor_write(output_base, 0, actual, output_size));
        CHECK(ds4_gpu_tensor_write(reference_base, 0, actual, output_size));
        CHECK(ds4_gpu_begin_commands());
        CHECK(k3_gpu_matmul_bf16(output, mapping, mapping_size, weight_offset,
                                 in_dim, out_dim, input, rows));
        for (uint32_t row = 0; row < rows; row++) {
            CHECK(k3_gpu_matmul_bf16(output_rows[row], mapping, mapping_size,
                weight_offset, in_dim, out_dim, input_rows[row], 1));
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(output_base, 0, actual, output_size));
        CHECK(ds4_gpu_tensor_read(reference_base, 0, expected, output_size));
        CHECK(guards_intact(actual, output_size, OUTPUT_OFFSET, rows * output_row));
        CHECK(guards_intact(expected, output_size, OUTPUT_OFFSET, rows * output_row));
        for (uint64_t i = 0; i < (uint64_t)rows * out_dim; i++) {
            uint32_t a, b;
            memcpy(&a, actual + OUTPUT_OFFSET + i * sizeof(float), sizeof(a));
            memcpy(&b, expected + OUTPUT_OFFSET + i * sizeof(float), sizeof(b));
            if (a != b || (a & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
                fprintf(stderr, "BF16 %ux%u batch=%u row=%" PRIu64
                    " column=%" PRIu64 ": %08" PRIx32 " != %08" PRIx32 "\n",
                    in_dim, out_dim, rows, i / out_dim, i % out_dim, a, b);
                goto cleanup;
            }
        }
    }
    /* A few independent double-precision dots also catch shared addressing
     * mistakes that a comparison between two GPU calls could miss. */
    const uint32_t columns[] = {0, out_dim / 2u, out_dim - 1u};
    for (uint32_t row = 0; row < MAX_ROWS; row++) {
        for (uint32_t sample = 0; sample < 3u; sample++) {
            const uint32_t column = columns[sample];
            double sum = 0, absolute_sum = 0;
            for (uint32_t k = 0; k < in_dim; k++) {
                const uint32_t bits = (uint32_t)weights[(uint64_t)column * in_dim + k] << 16;
                float weight, x;
                memcpy(&weight, &bits, sizeof(weight));
                memcpy(&x, inputs + INPUT_OFFSET + row * input_row + k * sizeof(float), sizeof(x));
                const double product = (double)weight * x;
                sum += product;
                absolute_sum += fabs(product);
            }
            float value;
            memcpy(&value, actual + OUTPUT_OFFSET + row * output_row + column * sizeof(float), sizeof(value));
            CHECK(fabs((double)value - sum) <= 4e-6 * absolute_sum + 1e-7);
        }
    }
    CHECK(ds4_gpu_tensor_read(input_base, 0, input_check, input_size));
    CHECK(memcmp(input_check, inputs, (size_t)input_size) == 0);
    CHECK(guards_intact(mapping, mapping_size, weight_offset, weight_size));
    ok = 1;

cleanup:
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = 0;
    const int retired = ds4_gpu_synchronize() && ds4_gpu_release_model_views();
    if (!retired) ok = 0;
    for (uint32_t row = 0; row < MAX_ROWS; row++) {
        ds4_gpu_tensor_free(output_rows[row]);
        ds4_gpu_tensor_free(input_rows[row]);
    }
    ds4_gpu_tensor_free(output);
    ds4_gpu_tensor_free(input);
    ds4_gpu_tensor_free(reference_base);
    ds4_gpu_tensor_free(output_base);
    ds4_gpu_tensor_free(input_base);
    /* A failed GPU cleanup must not retire a mapping still in use. */
    if (retired && mapping != MAP_FAILED &&
        munmap(mapping, (size_t)mapping_size) != 0) ok = 0;
    free(expected);
    free(actual);
    free(input_check);
    free(inputs);
    return ok;
#undef CHECK
}

int main(void) {
    /* Fix the dispatch configuration so an inherited tuning override cannot
     * silently bypass the candidate on supported hardware. */
    if (setenv("K3_BF16_MV_UNROLL", "16", 1) ||
        setenv("K3_BF16_MV_NSG", "4", 1)) return 1;
    ds4_gpu_set_ssd_streaming(true);
    if (!ds4_gpu_init()) return 1;
    const int ok = test_shape(1024, 1024) && test_shape(1025, 1031) &&
                   test_shape(128, 17) && test_shape(128, 1031) &&
                   test_shape(1024, 96);
    ds4_gpu_cleanup();
    if (ok) puts("K3 BF16 batched rows match single-token projections exactly");
    return ok ? 0 : 1;
}
