#include "ds4_float_compare.h"
#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    const char *name;
    uint32_t bits;
} bit_pattern;

static const bit_pattern patterns[] = {
    {"+1.0",                  UINT32_C(0x3f800000)},
    {"-1.0",                  UINT32_C(0xbf800000)},
    {"positive normal",       UINT32_C(0x40490fdb)},
    {"negative normal",       UINT32_C(0xc2f6e979)},
    {"+0",                    UINT32_C(0x00000000)},
    {"-0",                    UINT32_C(0x80000000)},
    {"+Inf",                  UINT32_C(0x7f800000)},
    {"-Inf",                  UINT32_C(0xff800000)},
    {"qNaN payload 1",        UINT32_C(0x7fc00001)},
    {"qNaN payload 2",        UINT32_C(0x7fc00002)},
    {"negative qNaN",         UINT32_C(0xffc01234)},
    {"min +subnormal",        UINT32_C(0x00000001)},
    {"max +subnormal",        UINT32_C(0x007fffff)},
    {"min -subnormal",        UINT32_C(0x80000001)},
    {"max -subnormal",        UINT32_C(0x807fffff)},
    {"min +normal",           UINT32_C(0x00800000)},
    {"next +normal",          UINT32_C(0x00800001)},
    {"min -normal",           UINT32_C(0x80800000)},
    {"max +finite",           UINT32_C(0x7f7fffff)},
    {"max -finite",           UINT32_C(0xff7fffff)},
};

static float float_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int main(void) {
    float source_before[ARRAY_LEN(patterns)];
    float source_after[ARRAY_LEN(patterns)];
    float destination_before[ARRAY_LEN(patterns)];
    float destination_after[ARRAY_LEN(patterns)];
    const uint64_t bytes = sizeof(source_before);
    ds4_gpu_tensor *source = NULL;
    ds4_gpu_tensor *destination = NULL;
    ds4_float_compare_result destination_result;
    ds4_float_compare_result source_result;
    int initialized = 0;
    int exit_code = 1;

#define REQUIRE(condition)                                                     \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,         \
                    #condition);                                               \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

    REQUIRE(sizeof(float) == sizeof(uint32_t));
    for (size_t i = 0; i < ARRAY_LEN(patterns); ++i) {
        source_before[i] = float_from_bits(patterns[i].bits);
        destination_before[i] = float_from_bits(UINT32_C(0xdeadbeef));
    }

    /* Validate the production-default Metal compilation mode. Strict shader
     * math must not mask object-representation changes in the typed copy. */
    unsetenv("DS4_METAL_MATH_SAFE");
    REQUIRE(ds4_gpu_init());
    initialized = 1;

    source = ds4_gpu_tensor_alloc(bytes);
    destination = ds4_gpu_tensor_alloc(bytes);
    REQUIRE(source != NULL && destination != NULL && source != destination);
    REQUIRE(ds4_gpu_tensor_write(source, 0, source_before, bytes));
    REQUIRE(ds4_gpu_tensor_write(destination, 0, destination_before, bytes));

    /* A strict diagnostic copy cannot run outside an active batch and cannot
     * accept an unaligned byte count. Either case must fail, never blit. */
    REQUIRE(!ds4_gpu_tensor_copy_f32_inline(
            destination, 0, source, 0, bytes));
    REQUIRE(ds4_gpu_begin_commands());
    REQUIRE(ds4_gpu_commands_active());
    REQUIRE(!ds4_gpu_tensor_copy_f32_inline(
            destination, 0, source, 0, bytes - 1));

    /* The first dispatch creates the cached compute encoder. The second must
     * reuse that exact encoder; the API's success contract checks both cases. */
    REQUIRE(ds4_gpu_tensor_copy_f32_inline(
            destination, 0, source, 0, bytes));
    REQUIRE(ds4_gpu_tensor_copy_f32_inline(
            destination, 0, source, 0, bytes));
    REQUIRE(ds4_gpu_commands_active());
    REQUIRE(ds4_gpu_end_commands());
    REQUIRE(!ds4_gpu_commands_active());

    REQUIRE(ds4_gpu_tensor_read(source, 0, source_after, bytes));
    REQUIRE(ds4_gpu_tensor_read(destination, 0, destination_after, bytes));
    REQUIRE(ds4_float_compare_exact(destination_after, source_before,
                                    ARRAY_LEN(patterns), &destination_result));
    REQUIRE(ds4_float_compare_exact(source_after, source_before,
                                    ARRAY_LEN(patterns), &source_result));

    puts("| Pattern | Input bits | Output bits | Exact? |");
    puts("|---|---:|---:|:---:|");
    for (size_t i = 0; i < ARRAY_LEN(patterns); ++i) {
        const uint32_t output = float_bits(destination_after[i]);
        printf("| %s | `0x%08x` | `0x%08x` | %s |\n",
               patterns[i].name,
               patterns[i].bits,
               output,
               output == patterns[i].bits ? "yes" : "NO");
    }

    REQUIRE(destination_result.bit_exact);
    REQUIRE(source_result.bit_exact);
    puts("test_metal_f32_inline_copy: PASS (destination exact, source preserved, same encoder retained)");
    exit_code = 0;

cleanup:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    ds4_gpu_tensor_free(destination);
    ds4_gpu_tensor_free(source);
    if (initialized) ds4_gpu_cleanup();
    return exit_code;
}
