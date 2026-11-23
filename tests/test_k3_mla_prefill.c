/* Synthetic MLA prefill regression: no checkpoint; explicit CPU/GPU fixtures
 * stay below 32 MiB. Compare chunked and single-token API calls exactly. */
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
enum { OFFSET = 128, GUARD = 128, MAX_ROWS = 128,
       Q = 0, KV = 1, GATE = 2, IDS = 3, INPUTS = 4,
       CACHE = 0, LATENT = 1, OUT = 2, STATES = 3 };
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
static float from_bits(uint32_t bits) {
    float x; memcpy(&x, &bits, sizeof(x)); return x;
}
static float bf16(uint16_t bits) { return from_bits((uint32_t)bits << 16); }
static uint16_t round_bf16(float x) {
    uint32_t u; memcpy(&u, &x, sizeof(u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint64_t hash_bytes(const void *bytes, uint64_t length) {
    const uint8_t *p = bytes;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint64_t i = 0; i < length; i++)
        hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
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
static int write_buffer(const test_buffer *b) {
    return ds4_gpu_tensor_write(b->base, 0, b->host, b->total);
}
static int read_buffer(test_buffer *b) {
    return ds4_gpu_tensor_read(b->base, 0, b->host, b->total);
}
static int guards_ok(const test_buffer *b) {
    for (uint32_t i = 0; i < OFFSET; i++) if (b->host[i] != 0xa5) return 0;
    for (uint64_t i = OFFSET + b->bytes; i < b->total; i++)
        if (b->host[i] != 0xa5) return 0;
    return 1;
}

/* Independent projections and double-precision softmax, sampled over heads
 * and tokens; the complete GPU outputs and cache are compared separately. */
static int oracle(const uint16_t *weights, const test_buffer input[INPUTS],
                  const test_buffer state[STATES], uint32_t heads,
                  uint32_t rows, uint32_t start, bool indexed) {
    const float *q = (const float *)data(&input[Q]);
    const float *gate = (const float *)data(&input[GATE]);
    const int32_t *ids = (const int32_t *)data(&input[IDS]);
    const uint16_t *cache = (const uint16_t *)data(&state[CACHE]);
    const float *actual_latent = (const float *)data(&state[LATENT]);
    const float *actual_out = (const float *)data(&state[OUT]);
    const uint32_t tokens[] = {0, rows - 1u}, head_ids[] = {0, heads - 1u};
    for (uint32_t t = 0; t < 2; t++) for (uint32_t h = 0; h < 2; h++) {
        const uint32_t token = tokens[t], head = head_ids[h], last = start + token;
        const uint64_t base = (uint64_t)token * heads + head;
        float query[512];
        for (uint32_t j = 0; j < 512; j++) {
            float sum = 0;
            for (uint32_t d = 0; d < 128; d++)
                sum = fmaf(q[base * 192 + d],
                    bf16(weights[((uint64_t)head * 256 + d) * 512 + j]), sum);
            query[j] = sum;
        }
        const bool sparse = indexed && last >= 2048u;
        const uint32_t keys = sparse ? 2048u : last + 1u;
        double scores[4096], maximum = -INFINITY;
        if (keys > 4096u) return 0;
        for (uint32_t i = 0; i < keys; i++) {
            const uint32_t key = sparse ? (uint32_t)ids[(uint64_t)token * 2048 + i] : i;
            double score = 0;
            for (uint32_t j = 0; j < 512; j++)
                score += (double)query[j] * bf16(cache[(uint64_t)key * 576 + j]);
            for (uint32_t j = 0; j < 64; j++)
                score += (double)q[base * 192 + 128 + j] * bf16(cache[(uint64_t)key * 576 + 512 + j]);
            scores[i] = score / sqrt(192.0);
            if (scores[i] > maximum) maximum = scores[i];
        }
        double latent[512] = {0}, absolute[512] = {0}, denominator = 0;
        for (uint32_t i = 0; i < keys; i++) {
            const uint32_t key = sparse ? (uint32_t)ids[(uint64_t)token * 2048 + i] : i;
            const double scale = exp(scores[i] - maximum);
            denominator += scale;
            for (uint32_t j = 0; j < 512; j++) {
                const double product = scale * bf16(cache[(uint64_t)key * 576 + j]);
                latent[j] += product; absolute[j] += fabs(product);
            }
        }
        for (uint32_t j = 0; j < 512; j++) {
            const double expected = latent[j] / denominator;
            if (fabs((double)actual_latent[base * 512 + j] - expected) >
                2e-5 * absolute[j] / denominator + 1e-6) return 0;
        }
        for (uint32_t value = 0; value < 128; value++) {
            double sum = 0, absolute_sum = 0;
            for (uint32_t k = 0; k < 512; k++) {
                const double product = (double)actual_latent[base * 512 + k] *
                    bf16(weights[((uint64_t)head * 256 + 128 + value) * 512 + k]);
                sum += product; absolute_sum += fabs(product);
            }
            const double divisor = 1.0 + exp(-(double)gate[base * 128 + value]);
            if (fabs((double)actual_out[base * 128 + value] - sum / divisor) >
                4e-6 * absolute_sum / divisor + 1e-6) return 0;
        }
    }
    return 1;
}

static int run_case(uint32_t heads, uint32_t rows, uint32_t start, bool indexed) {
    const uint32_t capacity = start + rows + 5u;
    const uint64_t weight_bytes = (uint64_t)heads * 256 * 512 * sizeof(uint16_t);
    const uint64_t input_bytes[] = {(uint64_t)rows * heads * 192 * 4,
        (uint64_t)rows * 576 * 4, (uint64_t)rows * heads * 128 * 4, (uint64_t)rows * 2048 * 4};
    const uint64_t state_bytes[] = {(uint64_t)capacity * 576 * 2,
        (uint64_t)rows * heads * 512 * 4, (uint64_t)rows * heads * 128 * 4};
    test_buffer input[INPUTS] = {{0}}, state[2][STATES] = {{{0}}};
    ds4_gpu_tensor *slices[MAX_ROWS][6] = {{0}};
    uint8_t *mapping = MAP_FAILED;
    uint16_t *expected_cache = NULL;
    uint64_t mapping_size = 0, weight_offset = 0, weight_hash = 0, hashes[INPUTS] = {0};
    int ok = 0, retired = 0;
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "MLA heads=%u rows=%u start=%u indexed=%u line%d: %s\n", \
        heads, rows, start, indexed, __LINE__, #condition); goto cleanup; } } while (0)
    const long page = sysconf(_SC_PAGESIZE);
    CHECK(page > 0 && rows > 0 && rows <= MAX_ROWS);
    weight_offset = (uint64_t)page + OFFSET;
    mapping_size = ((weight_offset + weight_bytes + GUARD + page - 1u) / page) * page;
    uint64_t allocation = mapping_size + state_bytes[CACHE];
    for (uint32_t i = 0; i < INPUTS; i++) allocation += 2 * (OFFSET + input_bytes[i] + GUARD);
    for (uint32_t s = 0; s < STATES; s++) allocation += 4 * (OFFSET + state_bytes[s] + GUARD);
    CHECK(allocation < UINT64_C(32) * 1024 * 1024);
    mapping = mmap(NULL, (size_t)mapping_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(mapping != MAP_FAILED);
    memset(mapping, 0xa5, (size_t)mapping_size);
    uint16_t *weights = (uint16_t *)(mapping + weight_offset);
    for (uint64_t i = 0; i < weight_bytes / 2; i++) {
        const uint32_t h = mix((uint32_t)i + UINT32_C(0x9e3779b9));
        weights[i] = (uint16_t)((h & 0x8000u) | ((119u + ((h >> 8) % 5u)) << 7) | (h & 127u));
    }
    weight_hash = hash_bytes(mapping, mapping_size);
    for (uint32_t i = 0; i < INPUTS; i++) CHECK(allocate(&input[i], input_bytes[i]));
    for (uint32_t v = 0; v < 2; v++)
        for (uint32_t s = 0; s < STATES; s++) CHECK(allocate(&state[v][s], state_bytes[s]));
    for (uint32_t b = Q; b <= GATE; b++) for (uint64_t i = 0; i < input[b].bytes / 4; i++) {
        const uint32_t h = mix((uint32_t)i + UINT32_C(0x13579bdf) * (b + 1u));
        ((float *)data(&input[b]))[i] = from_bits((h & 0x807fffffu) | ((120u + ((h >> 24) % 6u)) << 23));
    }
    for (uint32_t row = 0; row < rows; row++) {
        float *kv = (float *)data(&input[KV]) + (uint64_t)row * 576;
        kv[0] = from_bits(0x3f808000u); kv[1] = from_bits(0x3f818000u); kv[2] = from_bits(0xbf808000u);
        for (uint32_t i = 0; i < 2048; i++) {
            const uint32_t last = start + row;
            ((int32_t *)data(&input[IDS]))[(uint64_t)row * 2048 + i] =
                (int32_t)(last >= 2047u ? last - 2047u + i : i % (last + 1u));
        }
    }
    expected_cache = malloc((size_t)state_bytes[CACHE]); CHECK(expected_cache);
    for (uint64_t i = 0; i < state_bytes[CACHE] / 2; i++) {
        const uint32_t h = mix((uint32_t)i + UINT32_C(0xa54ff53a));
        expected_cache[i] = (uint16_t)((h & 0x8000u) | ((120u + ((h >> 8) % 5u)) << 7) | (h & 127u));
    }
    for (uint32_t v = 0; v < 2; v++) {
        memcpy(data(&state[v][CACHE]), expected_cache, (size_t)state_bytes[CACHE]);
        for (uint32_t s = 0; s < STATES; s++) CHECK(write_buffer(&state[v][s]));
    }
    for (uint64_t i = 0; i < input_bytes[KV] / 4; i++)
        expected_cache[(uint64_t)start * 576 + i] = round_bf16(((float *)data(&input[KV]))[i]);
    for (uint32_t i = 0; i < INPUTS; i++) {
        hashes[i] = hash_bytes(input[i].host, input[i].total); CHECK(write_buffer(&input[i]));
    }
    CHECK(ds4_gpu_set_model_map_spans(mapping, mapping_size, &weight_offset, &weight_bytes, 1, weight_bytes));
    CHECK(ds4_gpu_begin_commands());
    CHECK(k3_gpu_mla_prefill(state[0][OUT].view, state[0][CACHE].view, state[0][LATENT].view,
        input[Q].view, input[KV].view, input[GATE].view, mapping, mapping_size, weight_offset,
        heads, start, rows, capacity, indexed ? input[IDS].view : NULL));
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t i = 0; i < INPUTS; i++) {
            const uint64_t bytes = input_bytes[i] / rows;
            slices[row][i] = ds4_gpu_tensor_view(input[i].view, (uint64_t)row * bytes, bytes);
            CHECK(slices[row][i]);
        }
        for (uint32_t s = LATENT; s <= OUT; s++) {
            const uint64_t bytes = state_bytes[s] / rows;
            slices[row][s + 3] = ds4_gpu_tensor_view(state[1][s].view, (uint64_t)row * bytes, bytes);
            CHECK(slices[row][s + 3]);
        }
        CHECK(k3_gpu_mla_prefill(slices[row][5], state[1][CACHE].view, slices[row][4],
            slices[row][Q], slices[row][KV], slices[row][GATE], mapping, mapping_size, weight_offset,
            heads, start + row, 1, capacity, indexed ? slices[row][IDS] : NULL));
    }
    CHECK(ds4_gpu_end_commands());
    for (uint32_t s = 0; s < STATES; s++) {
        CHECK(read_buffer(&state[0][s]) && read_buffer(&state[1][s]));
        CHECK(guards_ok(&state[0][s]) && guards_ok(&state[1][s]));
        CHECK(memcmp(state[0][s].host, state[1][s].host, (size_t)state[0][s].total) == 0);
        if (s != CACHE) for (uint64_t i = 0; i < state_bytes[s] / 4; i++)
            CHECK(isfinite(((float *)data(&state[0][s]))[i]));
    }
    CHECK(memcmp(data(&state[0][CACHE]), expected_cache, (size_t)state_bytes[CACHE]) == 0);
    CHECK(oracle(weights, input, state[0], heads, rows, start, indexed));
    for (uint32_t i = 0; i < INPUTS; i++) {
        CHECK(read_buffer(&input[i]));
        CHECK(hash_bytes(input[i].host, input[i].total) == hashes[i]);
    }
    CHECK(hash_bytes(mapping, mapping_size) == weight_hash);
    ok = 1;
cleanup:
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = 0;
    retired = ds4_gpu_synchronize() && ds4_gpu_release_model_views();
    if (!retired) ok = 0;
    if (retired) {
        for (uint32_t row = 0; row < MAX_ROWS; row++)
            for (uint32_t i = 0; i < 6; i++) ds4_gpu_tensor_free(slices[row][i]);
        for (uint32_t i = 0; i < INPUTS; i++) {
            ds4_gpu_tensor_free(input[i].view); ds4_gpu_tensor_free(input[i].base);
        }
        for (uint32_t v = 0; v < 2; v++) for (uint32_t s = 0; s < STATES; s++) {
            ds4_gpu_tensor_free(state[v][s].view); ds4_gpu_tensor_free(state[v][s].base);
        }
        if (mapping != MAP_FAILED && munmap(mapping, (size_t)mapping_size) != 0) ok = 0;
    }
    for (uint32_t i = 0; i < INPUTS; i++) free(input[i].host);
    for (uint32_t v = 0; v < 2; v++) for (uint32_t s = 0; s < STATES; s++) free(state[v][s].host);
    free(expected_cache);
    return ok;
#undef CHECK
}

int main(int argc, char **argv) {
    const bool resident = argc == 2 && !strcmp(argv[1], "--resident");
    if (argc > 2 || (argc == 2 && !resident)) return 2;
    ds4_gpu_set_ssd_streaming(!resident);
    if (!ds4_gpu_init()) return 1;
    const int ok = run_case(1, 1, 0, false) && run_case(5, 7, 31, false) &&
        run_case(5, 32, 0, false) && run_case(5, 44, 0, false) &&
        run_case(5, 64, 31, false) && run_case(5, 128, 31, false) &&
        run_case(5, 3, 2047, true) &&
        run_case(3, 1, 2048, false) && run_case(1, 1, 4095, true);
    ds4_gpu_cleanup();
    if (ok) puts("MLA prefill: exact chunk/token outputs and cache; CPU oracles and guards passed");
    return ok ? 0 : 1;
}
