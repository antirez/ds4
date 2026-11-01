/*
 * Q8_0 exact-row batch matvec gate.
 *
 * ds4_gpu_matmul_q8_0_decode_rows_exact_tensor() serves 2..8 activation rows
 * (DSpark verification and other small exact batches).  This test pins the
 * contract that every row matches the single-row kernel bit for bit, for odd
 * output dimensions (the final NR0 tile is partial) and for K sizes that do
 * not fill a whole NSG*NQ block window.
 */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "Q8 row batch test failed: %s\n", what);
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

static void fill_weights(void *map, uint64_t weight_bytes) {
    uint8_t *p = (uint8_t *)map;
    for (uint64_t off = 0; off < weight_bytes; off += 34u) {
        /* Finite positive half scales in a narrow, exact-power-of-two band. */
        const uint16_t scale = (uint16_t)(0x1800u + (rnd() % 32u) * 0x100u);
        memcpy(p + off, &scale, sizeof(scale));
        for (unsigned i = 2; i < 34; i++) p[off + i] = (uint8_t)(rnd() & 0xffu);
    }
}

static void run_shape(uint32_t in_dim, uint32_t out_dim) {
    const uint64_t blocks = in_dim / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t map_bytes = align_up(weight_bytes + row_bytes, page);

    void *map = NULL;
    require_ok(posix_memalign(&map, (size_t)page, (size_t)map_bytes) == 0,
               "posix_memalign model map");
    memset(map, 0, (size_t)map_bytes);
    fill_weights(map, weight_bytes);
    require_ok(ds4_gpu_set_model_map(map, map_bytes) != 0, "set model map");

    for (uint32_t rows = 1; rows <= 8; rows++) {
        const uint64_t x_floats = (uint64_t)rows * in_dim;
        const uint64_t out_floats = (uint64_t)rows * out_dim;
        const uint64_t guard_floats = 64u;
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_floats * sizeof(float));
        ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc((out_floats + guard_floats) * sizeof(float));
        ds4_gpu_tensor *act = ds4_gpu_tensor_alloc((out_floats + guard_floats) * sizeof(float));
        require_ok(x && ref && act, "tensor alloc");

        float *input = (float *)malloc((size_t)(x_floats * sizeof(float)));
        float *ref_host = (float *)malloc((size_t)((out_floats + guard_floats) * sizeof(float)));
        float *act_host = (float *)malloc((size_t)((out_floats + guard_floats) * sizeof(float)));
        require_ok(input && ref_host && act_host, "host alloc");

        for (uint64_t i = 0; i < x_floats; i++)
            input[i] = (float)((int)(rnd() % 2049u) - 1024) / 1024.0f;
        require_ok(ds4_gpu_tensor_write(x, 0, input, x_floats * sizeof(float)) != 0,
                   "write activations");
        require_ok(ds4_gpu_tensor_fill_f32(ref, NAN, out_floats + guard_floats) != 0,
                   "fill ref guard");
        require_ok(ds4_gpu_tensor_fill_f32(act, NAN, out_floats + guard_floats) != 0,
                   "fill act guard");

        /* Reference: the ordinary single-row kernel, once per activation row. */
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(
                    x, (uint64_t)r * in_dim * sizeof(float),
                    (uint64_t)in_dim * sizeof(float));
            ds4_gpu_tensor *orr = ds4_gpu_tensor_view(
                    ref, (uint64_t)r * out_dim * sizeof(float),
                    (uint64_t)out_dim * sizeof(float));
            require_ok(xr && orr, "row views");
            require_ok(ds4_gpu_matmul_q8_0_tensor(
                    orr, map, map_bytes, 0, in_dim, out_dim, xr, 1) != 0,
                    "single-row reference");
            ds4_gpu_tensor_free(xr);
            ds4_gpu_tensor_free(orr);
        }

        /* Actual: the exact-row batch entry point. */
        require_ok(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                act, map, map_bytes, 0, in_dim, out_dim, x, rows) != 0,
                "row batch matvec");

        require_ok(ds4_gpu_tensor_read(
                ref, 0, ref_host, (out_floats + guard_floats) * sizeof(float)) != 0,
                "read ref");
        require_ok(ds4_gpu_tensor_read(
                act, 0, act_host, (out_floats + guard_floats) * sizeof(float)) != 0,
                "read act");

        for (uint32_t r = 0; r < rows; r++) {
            const uint64_t base = (uint64_t)r * out_dim;
            for (uint32_t i = 0; i < out_dim; i++) {
                if (memcmp(&ref_host[base + i], &act_host[base + i],
                           sizeof(float)) != 0) {
                    fprintf(stderr,
                            "Mismatch in=%u out=%u rows=%u row=%u col=%u "
                            "ref=%.9g act=%.9g\n",
                            in_dim, out_dim, rows, r, i,
                            (double)ref_host[base + i],
                            (double)act_host[base + i]);
                    exit(1);
                }
            }
        }
        for (uint64_t i = out_floats; i < out_floats + guard_floats; i++) {
            require_ok(isnan(ref_host[i]) && isnan(act_host[i]),
                       "output guard overwritten");
        }

        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref);
        ds4_gpu_tensor_free(act);
        free(input);
        free(ref_host);
        free(act_host);
    }

    printf("PASS Q8_0 row batch in=%u out=%u rows=1..8\n", in_dim, out_dim);
    free(map);
}

int main(void) {
    require_ok(ds4_gpu_init() != 0, "Metal init");

    static const struct {
        uint32_t in_dim;
        uint32_t out_dim;
    } shapes[] = {
        {  32,    1},
        {  32,    3},
        {  64,    5},
        {  96,    7},
        { 128,   17},
        { 160,   31},
        { 256,   63},
        { 512,   65},
        {1024,  127},
        {2560,  129},
        {4096,  513},
        {2560, 4097},
        /* The host enables the shared-load kernel at out_dim >= 2048.  Cover
         * the real V4.1 projection shapes, the partial last tile, and K
         * smaller than one NSG*NQ block window. */
        {1280, 32768},
        {2304,  5120},
        {5120,  2304},
        {  32,  4097},
        {  96,  2304},
    };

    for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        run_shape(shapes[i].in_dim, shapes[i].out_dim);

    ds4_gpu_cleanup();
    return 0;
}
