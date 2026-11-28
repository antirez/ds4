/* Compare batched BF16 projections with independent single-token calls.
 * The grouped prefill case uses one synthetic matrix at the K3 projection shape. */
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

static int test_shape_offset(uint32_t in_dim, uint32_t out_dim, bool misaligned,
                             uint32_t tensorops_rows) {
    const bool tiled = k3_gpu_can_tile_bf16_prefill(in_dim, out_dim, 256);
    const uint32_t max_rows = tensorops_rows ? tensorops_rows :
        tiled ? (misaligned ? 45u : 256u) : 8u;
    enum { MAX_ROWS = 256, INPUT_OFFSET = 64, OUTPUT_OFFSET = 128, GUARD = 128 };
    const uint64_t input_row = (uint64_t)in_dim * sizeof(float);
    const uint64_t output_row = (uint64_t)out_dim * sizeof(float);
    const uint64_t input_size = INPUT_OFFSET + max_rows * input_row + GUARD;
    /* Keep a whole tile of sentinels after a partial TensorOps output. */
    const uint64_t output_guard = tensorops_rows ? 32u * output_row : GUARD;
    const uint64_t output_size = OUTPUT_OFFSET + max_rows * output_row + output_guard;
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
    /* A four-byte displacement still permits scalar BF16 access, and
     * exercises the fallback for an unaligned vector of four weights. */
    const uint64_t weight_offset = (uint64_t)page + (misaligned ? 132u : 128u);
    mapping_size = (weight_offset + weight_size + GUARD + (uint64_t)page - 1u)
        / (uint64_t)page * (uint64_t)page;
    CHECK(mapping_size < (tiled ? UINT64_C(176) : UINT64_C(12)) * 1024u * 1024u);
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
    for (uint64_t i = 0; i < max_rows * input_row / sizeof(float); i++) {
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
    input = ds4_gpu_tensor_view(input_base, INPUT_OFFSET, max_rows * input_row);
    output = ds4_gpu_tensor_view(output_base, OUTPUT_OFFSET, max_rows * output_row);
    CHECK(input && output);
    for (uint32_t row = 0; row < max_rows; row++) {
        input_rows[row] = ds4_gpu_tensor_view(input, row * input_row, input_row);
        output_rows[row] = ds4_gpu_tensor_view(reference_base,
            OUTPUT_OFFSET + row * output_row, output_row);
        CHECK(input_rows[row] && output_rows[row]);
    }
    CHECK(ds4_gpu_tensor_write(input_base, 0, inputs, input_size));
    CHECK(ds4_gpu_set_model_map_spans(mapping, mapping_size,
        &weight_offset, &weight_size, 1, weight_size));

    const char *prefill_setting = getenv("K3_EXPERIMENTAL_PREFILL_KERNELS");
    const bool short_tiles = !prefill_setting || strcmp(prefill_setting, "1") == 0;
    const uint32_t tile_rows[] = {16, 40, 44, 45, 46, 47, 48, 64, 256};
    CHECK(k3_gpu_can_tile_bf16_prefill(in_dim, out_dim, 45) == (tiled && short_tiles));
    const uint32_t cases = tensorops_rows ? 1u :
        tiled ? sizeof(tile_rows) / sizeof(tile_rows[0]) : max_rows;
    for (uint32_t i = 0; i < cases; ++i) {
        const uint32_t rows = tensorops_rows ? tensorops_rows : tiled ? tile_rows[i] : i + 1u;
        if (misaligned && rows != max_rows) continue;
        if (tiled && rows != 256u) {
            CHECK(k3_gpu_can_tile_bf16_prefill(in_dim, out_dim, rows) == short_tiles);
            if (!short_tiles) continue;
        }
        memset(actual, 0xa5, (size_t)output_size);
        CHECK(ds4_gpu_tensor_write(output_base, 0, actual, output_size));
        CHECK(ds4_gpu_tensor_write(reference_base, 0, actual, output_size));
        CHECK(ds4_gpu_begin_commands());
        if (tiled) {
            CHECK(k3_gpu_matmul_bf16_prefill_tiles(output, mapping, mapping_size, weight_offset,
                in_dim, out_dim, input, rows));
        } else {
            CHECK(k3_gpu_matmul_bf16(output, mapping, mapping_size, weight_offset,
                in_dim, out_dim, input, rows));
        }
        if (tensorops_rows) {
            /* Quality mode selects the existing simdgroup kernel for this
             * projection with the same weights, activations and row count. */
            ds4_gpu_set_quality(true);
            ds4_gpu_tensor *reference = ds4_gpu_tensor_view(reference_base, OUTPUT_OFFSET,
                                                            rows * output_row);
            const int reference_ok = reference && k3_gpu_matmul_bf16(reference, mapping,
                mapping_size, weight_offset, in_dim, out_dim, input, rows);
            ds4_gpu_tensor_free(reference);
            ds4_gpu_set_quality(false);
            CHECK(reference_ok);
        } else {
            for (uint32_t row = 0; row < rows; row++) {
                CHECK(k3_gpu_matmul_bf16(output_rows[row], mapping, mapping_size,
                    weight_offset, in_dim, out_dim, input_rows[row], 1));
            }
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(output_base, 0, actual, output_size));
        CHECK(ds4_gpu_tensor_read(reference_base, 0, expected, output_size));
        CHECK(guards_intact(actual, output_size, OUTPUT_OFFSET, rows * output_row));
        CHECK(guards_intact(expected, output_size, OUTPUT_OFFSET, rows * output_row));
        if (tensorops_rows) continue; /* Both batched results are checked below. */
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
    for (uint32_t row = 0; row < max_rows; row++) {
        for (uint32_t sample = 0; sample < (tensorops_rows ? out_dim : 3u); sample++) {
            const uint32_t column = tensorops_rows ? sample : columns[sample];
            double sum = 0, absolute_sum = 0;
            for (uint32_t k = 0; k < in_dim; k++) {
                const uint32_t bits = (uint32_t)weights[(uint64_t)column * in_dim + k] << 16;
                float weight, x;
                memcpy(&weight, &bits, sizeof(weight));
                memcpy(&x, inputs + INPUT_OFFSET + row * input_row + k * sizeof(float), sizeof(x));
                /* Both batched kernels narrow the operands to half and
                 * accumulate in float. The generated BF16 weights fit half. */
                if (tensorops_rows) x = (float)(__fp16)x;
                const double product = (double)weight * x;
                sum += product;
                absolute_sum += fabs(product);
            }
            float value;
            memcpy(&value, actual + OUTPUT_OFFSET + row * output_row + column * sizeof(float), sizeof(value));
            CHECK(fabs((double)value - sum) <= 4e-6 * absolute_sum + 1e-7);
            if (tensorops_rows) {
                float legacy;
                memcpy(&legacy, expected + OUTPUT_OFFSET + row * output_row +
                       column * sizeof(float), sizeof(legacy));
                CHECK(fabs((double)legacy - sum) <= 4e-6 * absolute_sum + 1e-7);
                CHECK(fabs((double)value - legacy) <= 4e-6 * absolute_sum + 1e-7);
            }
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
    for (uint32_t row = 0; row < max_rows; row++) {
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

static int test_shape(uint32_t in_dim, uint32_t out_dim) {
    return test_shape_offset(in_dim, out_dim, false, 0);
}

int main(int argc, char **argv) {
    const bool resident = argc == 2 && strcmp(argv[1], "--resident") == 0;
    uint32_t tensorops_rows = 0;
    if (argc == 3 && !strcmp(argv[1], "--tensorops-rows")) {
        if (!strcmp(argv[2], "32")) tensorops_rows = 32;
        if (!strcmp(argv[2], "45")) tensorops_rows = 45;
        if (!strcmp(argv[2], "64")) tensorops_rows = 64;
    }
    if (argc != 1 && !resident && !tensorops_rows) {
        fprintf(stderr, "usage: %s [--resident | --tensorops-rows 32|45|64]\n", argv[0]);
        return 2;
    }
    /* Fix the dispatch configuration so an inherited tuning override cannot
     * silently bypass the candidate on supported hardware. */
    if (setenv("K3_BF16_MV_UNROLL", "16", 1) ||
        setenv("K3_BF16_MV_NSG", "4", 1)) return 1;
    ds4_gpu_set_ssd_streaming(!resident);
    if (!ds4_gpu_init()) return 1;
    if (tensorops_rows) {
        const int ok = test_shape_offset(256, 192, false, tensorops_rows);
        ds4_gpu_cleanup();
        if (ok) printf("K3 BF16 rows=%u: batched outputs and guards verified\n", tensorops_rows);
        return ok ? 0 : 1;
    }
    const int ok = test_shape(1024, 1024) && test_shape(1025, 1031) &&
                   test_shape(1024, 4096) && test_shape(1025, 4097) &&
                   test_shape(128, 17) && test_shape(128, 1031) &&
                   test_shape(1024, 96) &&
                   (!k3_gpu_can_tile_bf16_prefill(1536, 18432, 256) ||
                    (test_shape(1536, 18432) && test_shape(7168, 12288) &&
                     test_shape(12288, 7168) &&
                     (!k3_gpu_can_tile_bf16_prefill(7168, 12288, 44) ||
                      test_shape_offset(7168, 12288, true, 0))));
    ds4_gpu_cleanup();
    if (ok) puts("K3 BF16 batched rows match single-token projections exactly");
    return ok ? 0 : 1;
}
