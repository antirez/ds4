#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"
#include "k3_metal.h"
#include "k3_safetensors.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum { IN = 64, OUT = 128, ROWS = 2, WEIGHT_BYTES = IN * OUT * 2 };

static int write_exact(int fd, const void *data, size_t bytes) {
    const uint8_t *p = data;
    while (bytes != 0) {
        const ssize_t n = write(fd, p, bytes);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += n;
        bytes -= (size_t)n;
    }
    return 1;
}

/* All inputs are small binary fractions, exactly representable in BF16. */
static float weight_value(uint32_t matrix, uint32_t row, uint32_t col) {
    return (float)((int)((row * 3u + col * 5u + matrix * 7u) % 11u) - 5) / 4.0f;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

int main(void) {
    int ok = 0, initialized = 0, fd = -1;
    char path[] = "/tmp/ds4-k3-stream-map-XXXXXX";
    char error[256] = {0};
    char *header = NULL;
    k3_st_shard *canonical = NULL, *temporary = NULL;
    ds4_gpu_tensor *input = NULL, *output = NULL;
    const k3_st_tensor *weights[2] = {NULL, NULL};
    const void *stable_map = NULL;
    uint16_t matrices[2][IN * OUT];
    float x[ROWS * IN], actual[ROWS * OUT];

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "test_k3_stream_mapping:%d: %s failed%s%s\n", \
                __LINE__, #condition, error[0] ? ": " : "", error); \
        goto cleanup; \
    } \
} while (0)

    const long page = sysconf(_SC_PAGESIZE);
    CHECK(page >= 4096 && page <= WEIGHT_BYTES && WEIGHT_BYTES % page == 0);
    const size_t header_bytes = (size_t)page - 8u;
    header = malloc(header_bytes);
    CHECK(header != NULL);
    memset(header, ' ', header_bytes);
    char json[512];
    const int json_bytes = snprintf(json, sizeof(json),
        "{\"first\":{\"dtype\":\"BF16\",\"shape\":[%u,%u],"
        "\"data_offsets\":[0,%u]},"
        "\"second\":{\"dtype\":\"BF16\",\"shape\":[%u,%u],"
        "\"data_offsets\":[%u,%u]}}",
        OUT, IN, WEIGHT_BYTES, OUT, IN, WEIGHT_BYTES, 2u * WEIGHT_BYTES);
    CHECK(json_bytes > 0 && (size_t)json_bytes < sizeof(json) &&
          (size_t)json_bytes <= header_bytes);
    memcpy(header, json, (size_t)json_bytes);
    for (uint32_t m = 0; m < 2; m++) {
        for (uint32_t row = 0; row < OUT; row++) {
            for (uint32_t col = 0; col < IN; col++) {
                matrices[m][row * IN + col] = to_bf16(weight_value(m, row, col));
            }
        }
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t col = 0; col < IN; col++) {
            x[row * IN + col] = (float)((int)((col + row * 2u) % 7u) - 3) / 8.0f;
        }
    }
    fd = mkstemp(path);
    CHECK(fd >= 0);
    uint8_t length[8];
    for (uint32_t i = 0; i < 8; i++)
        length[i] = (uint8_t)((uint64_t)header_bytes >> (8u * i));
    CHECK(write_exact(fd, length, sizeof(length)) &&
          write_exact(fd, header, header_bytes) &&
          write_exact(fd, matrices, sizeof(matrices)));
    CHECK(close(fd) == 0);
    fd = -1;
    CHECK(k3_st_open(&canonical, path, error, sizeof(error)) &&
          k3_st_open(&temporary, path, error, sizeof(error)) &&
          k3_st_map(canonical, error, sizeof(error)));
    CHECK(k3_st_file_size(canonical) <= 1024u * 1024u);
    stable_map = k3_st_map_base(canonical);
    weights[0] = k3_st_find(temporary, "first");
    weights[1] = k3_st_find(temporary, "second");
    CHECK(stable_map && weights[0] && weights[1]);
    CHECK(weights[0]->size == WEIGHT_BYTES && weights[1]->size == WEIGHT_BYTES);

    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_init());
    initialized = 1;
    input = ds4_gpu_tensor_alloc_managed(sizeof(x));
    output = ds4_gpu_tensor_alloc_managed(sizeof(actual));
    CHECK(input && output && ds4_gpu_tensor_write(input, 0, x, sizeof(x)));
    CHECK(ds4_gpu_set_model_fd(k3_st_fd(canonical)));

    /* A canonical mapping remains stable for expert-cache keys while each
     * static-weight mapping is registered, consumed, released and unmapped.
     * Alternating tensors catches stale-view reuse at a different offset. */
    for (uint32_t cycle = 0; cycle < 4; cycle++) {
        const uint32_t m = cycle & 1u;
        CHECK(k3_st_map(temporary, error, sizeof(error)));
        const void *layer_map = k3_st_map_base(temporary);
        CHECK(layer_map && layer_map != stable_map);
        const uint64_t offset = weights[m]->offset;
        const uint64_t size = weights[m]->size;
        CHECK(ds4_gpu_set_model_fd_for_map(k3_st_fd(temporary), layer_map) &&
              ds4_gpu_set_model_map_spans(layer_map, k3_st_file_size(temporary),
                                          &offset, &size, 1, size));
        CHECK(ds4_gpu_begin_commands() &&
              k3_gpu_matmul_bf16(output, layer_map, k3_st_file_size(temporary),
                                 offset, IN, OUT, input, ROWS) &&
              ds4_gpu_end_commands() &&
              ds4_gpu_tensor_read(output, 0, actual, sizeof(actual)));
        for (uint32_t row = 0; row < ROWS; row++) {
            for (uint32_t out = 0; out < OUT; out++) {
                float expected = 0.0f;
                for (uint32_t col = 0; col < IN; col++)
                    expected += x[row * IN + col] * weight_value(m, out, col);
                CHECK(fabsf(actual[row * OUT + out] - expected) <= 1e-5f);
            }
        }
        CHECK(ds4_gpu_release_model_views());
        k3_st_unmap(temporary);
        CHECK(k3_st_map_base(temporary) == NULL);
        CHECK(k3_st_map_base(canonical) == stable_map);
        for (uint32_t i = 0; i < 2; i++) {
            CHECK(memcmp((const uint8_t *)stable_map + weights[i]->offset,
                         matrices[i], WEIGHT_BYTES) == 0);
        }
    }
    ok = 1;

cleanup:
    if (initialized) {
        (void)ds4_gpu_synchronize();
        (void)ds4_gpu_release_model_views();
    }
    if (output) ds4_gpu_tensor_free(output);
    if (input) ds4_gpu_tensor_free(input);
    ds4_gpu_cleanup();
    k3_st_close(temporary);
    k3_st_close(canonical);
    if (fd >= 0) close(fd);
    unlink(path);
    free(header);
    if (ok) puts("test_k3_stream_mapping: ok");
    return ok ? 0 : 1;
#undef CHECK
}
