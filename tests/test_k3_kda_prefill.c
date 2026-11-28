/* Synthetic KDA chunk-boundary regression; no checkpoint and <16 MiB of
 * explicit fixtures. Compare all outputs and both persistent states exactly. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include "k3_metal.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
enum { OFFSET = 128, GUARD = 128, Q = 0, K, V, DECAY, BETA, GATE, OUT, CONV, STATE, BUFFERS };

typedef struct {
    ds4_gpu_tensor *base, *view;
    uint8_t *host;
    uint64_t bytes, total;
} test_buffer;

static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= UINT32_C(0x7feb352d);
    x ^= x >> 15; x *= UINT32_C(0x846ca68b);
    return x ^ (x >> 16);
}
static float value(uint32_t buffer, uint64_t index) {
    const uint32_t h = mix((uint32_t)index + UINT32_C(0x9e3779b9) * (buffer + 1u));
    const uint32_t bits = (h & UINT32_C(0x807fffff)) |
        ((buffer == STATE ? 117u : 122u) << 23);
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}
static uint64_t hash_bytes(const void *data, uint64_t count) {
    const uint8_t *p = data;
    uint64_t h = UINT64_C(14695981039346656037);
    for (uint64_t i = 0; i < count; i++) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static uint8_t *data(const test_buffer *b) { return b->host + OFFSET; }
static int allocate(test_buffer *b, uint64_t bytes) {
    b->bytes = bytes; b->total = OFFSET + bytes + GUARD;
    b->host = malloc((size_t)b->total);
    b->base = ds4_gpu_tensor_alloc_managed(b->total);
    if (!b->host || !b->base) return 0;
    memset(b->host, 0xa5, (size_t)b->total);
    b->view = ds4_gpu_tensor_view(b->base, OFFSET, bytes);
    return b->view != NULL;
}
static int guards_ok(const test_buffer *b) {
    for (uint32_t i = 0; i < OFFSET; i++) if (b->host[i] != 0xa5) return 0;
    for (uint64_t i = OFFSET + b->bytes; i < b->total; i++)
        if (b->host[i] != 0xa5) return 0;
    return 1;
}

static int dispatch(test_buffer b[BUFFERS], const void *map, uint64_t map_size,
                    const uint64_t offsets[6], uint32_t heads, uint32_t rows,
                    uint32_t first, uint32_t count) {
    ds4_gpu_tensor *views[7] = {0};
    const uint32_t buffers[] = {Q, K, V, DECAY, BETA, GATE, OUT};
    int ok = 1;
    for (uint32_t i = 0; i < 7; i++) {
        const uint32_t n = buffers[i];
        const uint64_t stride = b[n].bytes / rows;
        views[i] = ds4_gpu_tensor_view(b[n].view, first * stride, count * stride);
        if (!views[i]) ok = 0;
    }
    if (ok) ok = k3_gpu_kda_prefill(views[6], b[CONV].view, b[STATE].view,
        views[0], views[1], views[2], views[3], views[4], views[5],
        map, map_size, offsets[0], offsets[1], offsets[2], offsets[3],
        offsets[4], offsets[5], heads, count, -5.0f, 1e-5f);
    for (uint32_t i = 0; i < 7; i++) ds4_gpu_tensor_free(views[i]);
    return ok;
}

static int run_case(uint32_t heads, uint32_t rows, uint32_t split) {
    const uint64_t projection = (uint64_t)heads * 128;
    const uint64_t activation = rows * projection * sizeof(float);
    const uint64_t sizes[BUFFERS] = {
        [Q] = activation, [K] = activation, [V] = activation,
        [DECAY] = activation, [BETA] = (uint64_t)rows * heads * sizeof(float),
        [GATE] = activation, [OUT] = activation,
        [CONV] = 9 * projection * sizeof(float),
        [STATE] = 128 * projection * sizeof(float),
    };
    const uint64_t weights[] = {4 * projection * sizeof(float),
        4 * projection * sizeof(float), 4 * projection * sizeof(float),
        heads * sizeof(float), projection * sizeof(float), 128 * sizeof(float)};
    uint64_t offsets[6] = {0}, map_size = 0, weight_hash = 0;
    uint64_t input_hashes[2] = {0};
    test_buffer buffers[2][BUFFERS] = {{{0}}};
    uint8_t *mapping = MAP_FAILED;
    int ok = 0, retired = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "KDA heads=%u rows=%u split=%u line%d: %s\n", \
        heads, rows, split, __LINE__, #c); goto cleanup; } } while (0)
    const long page = sysconf(_SC_PAGESIZE);
    CHECK(page > 0 && split > 0 && split < rows && rows >= 3);
    uint64_t end = (uint64_t)page + OFFSET;
    for (uint32_t i = 0; i < 6; i++) {
        offsets[i] = end;
        end = (end + weights[i] + 15u) & ~UINT64_C(15);
    }
    map_size = (end + GUARD + page - 1u) / page * page;
    uint64_t fixture_bytes = map_size;
    for (uint32_t b = Q; b < BUFFERS; b++) fixture_bytes += 4 * (OFFSET + sizes[b] + GUARD);
    CHECK(fixture_bytes < UINT64_C(16) * 1024 * 1024);
    mapping = mmap(NULL, (size_t)map_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(mapping != MAP_FAILED);
    memset(mapping, 0xa5, (size_t)map_size);
    for (uint32_t w = 0; w < 6; w++) {
        float *p = (float *)(mapping + offsets[w]);
        for (uint64_t i = 0; i < weights[w] / sizeof(float); i++)
            p[i] = w == 3 ? -0.75f + value(w + 16, i) :
                   w == 5 ? 1.0f + value(w + 16, i) : value(w + 16, i);
    }
    weight_hash = hash_bytes(mapping, map_size);
    for (uint32_t b = Q; b < BUFFERS; b++) {
        CHECK(allocate(&buffers[0][b], sizes[b]));
        CHECK(allocate(&buffers[1][b], sizes[b]));
        for (uint64_t i = 0; i < sizes[b] / sizeof(float); i++)
            ((float *)data(&buffers[0][b]))[i] = b == OUT ? NAN : value(b, i);
        memcpy(buffers[1][b].host, buffers[0][b].host, (size_t)buffers[0][b].total);
        for (uint32_t v = 0; v < 2; v++)
            CHECK(ds4_gpu_tensor_write(buffers[v][b].base, 0,
                buffers[v][b].host, buffers[v][b].total));
    }
    input_hashes[0] = hash_bytes(buffers[0][BETA].host, buffers[0][BETA].total);
    input_hashes[1] = hash_bytes(buffers[0][GATE].host, buffers[0][GATE].total);
    const uint64_t span_size = end - offsets[0];
    CHECK(ds4_gpu_set_model_map_spans(mapping, map_size, offsets, &span_size, 1, span_size));
    CHECK(ds4_gpu_begin_commands());
    CHECK(dispatch(buffers[0], mapping, map_size, offsets, heads, rows, 0, rows));
    for (uint32_t row = 0; row < rows; row += split) {
        const uint32_t count = rows - row < split ? rows - row : split;
        CHECK(dispatch(buffers[1], mapping, map_size, offsets, heads, rows, row, count));
    }
    CHECK(ds4_gpu_end_commands());
    for (uint32_t b = Q; b < BUFFERS; b++) {
        for (uint32_t v = 0; v < 2; v++) {
            CHECK(ds4_gpu_tensor_read(buffers[v][b].base, 0,
                buffers[v][b].host, buffers[v][b].total));
            CHECK(guards_ok(&buffers[v][b]));
        }
        CHECK(memcmp(buffers[0][b].host, buffers[1][b].host,
                     (size_t)buffers[0][b].total) == 0);
        for (uint64_t i = 0; i < sizes[b] / sizeof(float); i++)
            CHECK(isfinite(((float *)data(&buffers[0][b]))[i]));
    }
    CHECK(hash_bytes(buffers[0][BETA].host, buffers[0][BETA].total) == input_hashes[0]);
    CHECK(hash_bytes(buffers[0][GATE].host, buffers[0][GATE].total) == input_hashes[1]);
    /* History is the last three unmodified projection rows, independent of
     * the nonlinear convolution and recurrence calculations. */
    for (uint32_t qkv = 0; qkv < 3; qkv++) for (uint32_t h = 0; h < 3; h++)
        for (uint64_t c = 0; c < projection; c++) {
            const float expected = value(Q + qkv, (rows - 3u + h) * projection + c);
            const float actual = ((float *)data(&buffers[0][CONV]))[(qkv * 3u + h) * projection + c];
            CHECK(memcmp(&expected, &actual, sizeof(float)) == 0);
        }
    CHECK(hash_bytes(mapping, map_size) == weight_hash);
    ok = 1;
cleanup:
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = 0;
    retired = ds4_gpu_synchronize() && ds4_gpu_release_model_views();
    if (!retired) ok = 0;
    for (uint32_t v = 0; v < 2; v++) for (uint32_t b = Q; b < BUFFERS; b++) {
        if (retired) {
            ds4_gpu_tensor_free(buffers[v][b].view);
            ds4_gpu_tensor_free(buffers[v][b].base);
        }
        free(buffers[v][b].host);
    }
    if (retired && mapping != MAP_FAILED && munmap(mapping, (size_t)map_size) != 0) ok = 0;
    return ok;
#undef CHECK
}

int main(int argc, char **argv) {
    const bool resident = argc == 2 && !strcmp(argv[1], "--resident");
    if (argc > 2 || (argc == 2 && !resident)) return 2;
    ds4_gpu_set_ssd_streaming(!resident);
    if (!ds4_gpu_init()) return 1;
    const int ok = run_case(1, 32, 16) && run_case(5, 44, 32) && run_case(5, 64, 32) &&
        run_case(5, 128, 64) && run_case(3, 256, 64) && run_case(3, 249, 64);
    ds4_gpu_cleanup();
    if (ok) puts("KDA prefill: exact full/chunked outputs, convolution history and recurrent state");
    return ok ? 0 : 1;
}
