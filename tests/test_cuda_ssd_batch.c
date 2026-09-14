/* Small SSD batches must retain isolated decode's quantization and slot sum. */
#include "ds4_gpu.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
enum { LAYERS = 2, MAX_SLOTS = 6 };
static uint32_t rng = 41;
static bool owned_mmq_test;
static unsigned char random_byte(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 24;
}
static ds4_gpu_tensor *tensor(size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t);
    return t;
}

static void poison_experts(ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
                           ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
                           unsigned rows, unsigned slots, unsigned dim, unsigned mid_dim) {
    CHECK(ds4_gpu_tensor_fill_f32(gate, NAN, rows * slots * mid_dim));
    CHECK(ds4_gpu_tensor_fill_f32(up, NAN, rows * slots * mid_dim));
    CHECK(ds4_gpu_tensor_fill_f32(mid, NAN, rows * slots * mid_dim));
    CHECK(ds4_gpu_tensor_fill_f32(down, NAN, rows * slots * dim));
}

static void exact(const float *actual, const float *expected, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(actual[i]) || memcmp(actual + i, expected + i, sizeof(float))) {
            fprintf(stderr, "SSD batch differs at %zu: %.9g != %.9g\n",
                    i, actual[i], expected[i]);
            exit(1);
        }
    }
}

static void run(unsigned dim, unsigned mid_dim, unsigned slots, bool scalar_only,
                bool owned, bool q4, unsigned ROWS, unsigned EXPERTS) {
    const unsigned gate_type = q4 ? 12 : 16, down_type = q4 ? 12 : 10;
    const size_t gate_row = (q4 ? 144 : 66) * (dim / 256);
    const size_t down_row = (q4 ? 144 : 84) * (mid_dim / 256);
    const size_t gate_bytes = mid_dim * gate_row, down_bytes = dim * down_row;
    const size_t layer_bytes = EXPERTS * (2 * gate_bytes + down_bytes);
    const size_t model_bytes = LAYERS * layer_bytes;
    unsigned char *model = malloc(model_bytes);
    float *input = malloc(ROWS * dim * sizeof(float));
    float *expected = malloc(LAYERS * ROWS * dim * sizeof(float));
    float *expected_mid = malloc(LAYERS * ROWS * slots * mid_dim * sizeof(float));
    float *actual = malloc((ROWS + 1) * dim * sizeof(float));
    float *actual_mid = malloc(ROWS * slots * mid_dim * sizeof(float));
    CHECK(model && input && expected && expected_mid && actual && actual_mid);
    for (unsigned layer = 0; layer < LAYERS; layer++) {
        for (unsigned part = 0; part < 3; part++) {
            const size_t unit = q4 ? 144 : part == 2 ? 84 : 66;
            const size_t bytes = EXPERTS * (part == 2 ? down_bytes : gate_bytes);
            unsigned char *w = model + layer * layer_bytes + part * EXPERTS * gate_bytes;
            for (size_t i = 0; i < bytes; i++) w[i] = random_byte();
            for (size_t i = 0; i < bytes; i += unit) {
                const size_t d = i + (!q4 && part == 2 ? 80 : 0);
                w[d] = 0; w[d + 1] = 0x14;
                if (q4 || part == 2) { w[d + 2] = 0; w[d + 3] = 0; }
            }
        }
    }
    char path[] = "/tmp/ds4-ssd-batch-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0 && unlink(path) == 0);
    for (size_t off = 0; off < model_bytes;) {
        const ssize_t n = write(fd, model + off, model_bytes - off);
        if (n < 0 && errno == EINTR) continue;
        CHECK(n > 0);
        off += (size_t)n;
    }
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_fd_for_map(fd, model) &&
          ds4_gpu_set_model_map(model, model_bytes));
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * gate_bytes + down_bytes);
    ds4_gpu_tensor *x = tensor(ROWS * dim * sizeof(float));
    ds4_gpu_tensor *ids = tensor(ROWS * slots * sizeof(int32_t));
    ds4_gpu_tensor *weights = tensor(ROWS * slots * sizeof(float));
    ds4_gpu_tensor *out = tensor((ROWS + 1) * dim * sizeof(float));
    ds4_gpu_tensor *gate = tensor(ROWS * slots * mid_dim * sizeof(float));
    ds4_gpu_tensor *up = tensor(ROWS * slots * mid_dim * sizeof(float));
    ds4_gpu_tensor *mid = tensor(ROWS * slots * mid_dim * sizeof(float));
    ds4_gpu_tensor *down = tensor(ROWS * slots * dim * sizeof(float));
    int32_t selected[ROWS * MAX_SLOTS];
    float sw[ROWS * MAX_SLOTS];
    const unsigned budgets[] = {0, 3, 12, 24};
    const unsigned counts[] = {1, 2, 4, 7, 8, 9, 16, 17, 31, 32, 127, 128, 129,
        8191, 8192, 8193};
    uint64_t scalar_hash = UINT64_C(1469598103934665603);
    for (unsigned pass = 0; pass < 2; pass++) {
        const float clamp = pass ? 0.02f : 10.0f;
        for (unsigned i = 0; i < ROWS * dim; i++)
            input[i] = ((int)random_byte() - 128) / 256.0f;
        for (unsigned row = 0; row < ROWS; row++) for (unsigned j = 0; j < slots; j++) {
            const unsigned i = row * slots + j;
            selected[i] = owned && pass ? (row % 2) * (EXPERTS / 2) + j % (EXPERTS / 2) :
                (row * 3 + (pass ? slots - 1 - j : j)) % EXPERTS;
            sw[i] = (1.0f + random_byte()) / (256 * slots);
        }
        ds4_gpu_set_streaming_expert_cache_budget(12);
        /* Large owned-MMQ cases use unsplit MMQ below as their oracle. */
        const unsigned scalar_rows = owned_mmq_test && ROWS > 129 ? 129 : ROWS;
        for (unsigned layer = 0; layer < LAYERS; layer++) for (unsigned row = 0; row < scalar_rows; row++) {
            CHECK(ds4_gpu_tensor_write(x, 0, input + row * dim, dim * sizeof(float)));
            CHECK(ds4_gpu_tensor_write(ids, 0, selected + row * slots, slots * sizeof(int32_t)));
            CHECK(ds4_gpu_tensor_write(weights, 0, sw + row * slots, slots * sizeof(float)));
            const size_t g = layer * layer_bytes, u = g + EXPERTS * gate_bytes, d = u + EXPERTS * gate_bytes;
            bool half = false;
            CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, model_bytes,
                g, u, d, gate_type, down_type, gate_bytes, gate_row, down_bytes, down_row,
                dim, mid_dim, dim, ids, weights, EXPERTS, slots, clamp, x, layer, 1, &half, false));
            CHECK(!half && ds4_gpu_tensor_read(out, 0, expected + (layer * ROWS + row) * dim, dim * sizeof(float)));
            CHECK(ds4_gpu_tensor_read(mid, 0, expected_mid + (layer * ROWS + row) * slots * mid_dim,
                                      slots * mid_dim * sizeof(float)));
        }
        for (unsigned layer = 0; layer < LAYERS; layer++) {
            const unsigned char *bytes = (const unsigned char *)(expected + layer * ROWS * dim);
            for (size_t i = 0; i < scalar_rows * dim * sizeof(float); i++) {
                scalar_hash ^= bytes[i];
                scalar_hash *= UINT64_C(1099511628211);
            }
        }
        if (owned) {
            float *sum = calloc(ROWS * dim, sizeof(float));
            float *one = malloc(dim * sizeof(float));
            CHECK(sum && one);
            ds4_gpu_set_ssd_streaming(false);
            if (slots == 6) {
                CHECK(ds4_gpu_tensor_write(x, 0, input, dim * sizeof(float)));
                CHECK(ds4_gpu_tensor_write(ids, 0, selected, slots * sizeof(int32_t)));
                CHECK(ds4_gpu_tensor_write(weights, 0, sw, slots * sizeof(float)));
                for (unsigned layer = 0; layer < LAYERS; layer++) {
                    const size_t g = layer * layer_bytes, u = g + EXPERTS * gate_bytes, d = u + EXPERTS * gate_bytes;
                    memset(sum, 0, dim * sizeof(float));
                    for (unsigned rank = 0; rank < 2; rank++) {
                        poison_experts(gate, up, mid, down, 1, slots, dim, mid_dim);
                        CHECK(ds4_gpu_routed_moe_one_owned_tensor(out, gate, up, mid, down,
                            model, model_bytes, g, u, d, gate_type, down_type, gate_bytes, gate_row,
                            down_bytes, down_row, dim, mid_dim, dim, ids, weights, EXPERTS, slots,
                            rank * (EXPERTS / 2), EXPERTS / 2, clamp, x, NULL, false, NULL));
                        CHECK(ds4_gpu_tensor_read(down, 0, actual, slots * dim * sizeof(float)));
                        for (unsigned j = 0; j < slots; j++) {
                            if ((unsigned)selected[j] / (EXPERTS / 2) != rank) continue;
                            for (unsigned i = 0; i < dim; i++) sum[i] += actual[j * dim + i];
                        }
                    }
                    for (unsigned i = 0; i < dim; i++) {
                        const float reference = expected[layer * ROWS * dim + i];
                        CHECK(isfinite(sum[i]) && fabsf(sum[i] - reference) <= 3e-5f * (1 + fabsf(reference)));
                    }
                }
            }
            for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const unsigned rows = counts[ci];
                if (rows > ROWS) continue;
                CHECK(ds4_gpu_tensor_write(x, 0, input, rows * dim * sizeof(float)));
                for (unsigned layer = 0; layer < LAYERS; layer++) {
                    const size_t g = layer * layer_bytes, u = g + EXPERTS * gate_bytes, d = u + EXPERTS * gate_bytes;
                    if (owned_mmq_test && !q4 && rows >= 128 && EXPERTS / 2 >= slots) {
                        CHECK(ds4_gpu_tensor_write(ids, 0, selected, rows * slots * sizeof(int32_t)));
                        CHECK(ds4_gpu_tensor_write(weights, 0, sw, rows * slots * sizeof(float)));
                        bool half = false;
                        CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
                            model, model_bytes, g, u, d, gate_type, down_type, gate_bytes, gate_row,
                            down_bytes, down_row, dim, mid_dim, dim, ids, weights, EXPERTS, slots,
                            clamp, x, layer, rows, &half, false));
                        CHECK(!half && ds4_gpu_tensor_read(out, 0, expected + layer * ROWS * dim,
                            rows * dim * sizeof(float)));
                    }
                    memset(sum, 0, rows * dim * sizeof(float));
                    for (unsigned rank = 0; rank < 2; rank++) {
                        poison_experts(gate, up, mid, down, rows, slots, dim, mid_dim);
                        CHECK(ds4_gpu_tensor_write(ids, 0, selected, rows * slots * sizeof(int32_t)));
                        CHECK(ds4_gpu_tensor_write(weights, 0, sw, rows * slots * sizeof(float)));
                        CHECK(ds4_gpu_tensor_fill_f32(out, 12345, (rows + 1) * dim));
                        bool half = false;
                        CHECK(ds4_gpu_routed_moe_batch_owned_tensor(out, gate, up, mid, down,
                            model, model_bytes, g, u, d, gate_type, down_type, gate_bytes, gate_row,
                            down_bytes, down_row, dim, mid_dim, dim, ids, weights, EXPERTS, slots,
                            rank * (EXPERTS / 2), EXPERTS / 2, clamp, x, layer, rows, &half));
                        CHECK(!half && ds4_gpu_tensor_read(out, 0, actual, (rows * dim + 1) * sizeof(float)));
                        CHECK(actual[rows * dim] == 12345);
                        for (unsigned i = 0; i < rows * dim; i++) {
                            CHECK(isfinite(actual[i]));
                            sum[i] += actual[i];
                            if (pass && (i / dim) % 2 != rank) CHECK(actual[i] == 0);
                        }
                        if (!q4 && dim == 5120 && mid_dim == 2304 && rows <= 8) {
                            /* Each local small-batch row must match its own
                             * scalar partial, not just the two-rank sum. */
                            for (unsigned row = 0; row < rows; row++) {
                                poison_experts(gate, up, mid, down, 1, slots, dim, mid_dim);
                                CHECK(ds4_gpu_tensor_write(x, 0, input + row * dim, dim * sizeof(float)));
                                CHECK(ds4_gpu_tensor_write(ids, 0, selected + row * slots, slots * sizeof(int32_t)));
                                CHECK(ds4_gpu_tensor_write(weights, 0, sw + row * slots, slots * sizeof(float)));
                                CHECK(ds4_gpu_routed_moe_batch_owned_tensor(out, gate, up, mid, down,
                                    model, model_bytes, g, u, d, gate_type, down_type, gate_bytes, gate_row,
                                    down_bytes, down_row, dim, mid_dim, dim, ids, weights, EXPERTS, slots,
                                    rank * (EXPERTS / 2), EXPERTS / 2, clamp, x, layer, 1, &half));
                                CHECK(!half && ds4_gpu_tensor_read(out, 0, one, dim * sizeof(float)));
                                exact(actual + row * dim, one, dim);
                            }
                            CHECK(ds4_gpu_tensor_write(x, 0, input, rows * dim * sizeof(float)));
                        }
                    }
                    for (unsigned i = 0; i < rows * dim; i++) {
                        const float reference = expected[layer * ROWS * dim + i];
                        if (fabsf(sum[i] - reference) > 3e-5f * (1 + fabsf(reference)))
                            fprintf(stderr, "owned %s sum rows=%u index=%u actual=%g reference=%g\n",
                                q4 ? "Q4_K" : "IQ2", rows, i, sum[i], reference);
                        CHECK(fabsf(sum[i] - reference) <= 3e-5f * (1 + fabsf(reference)));
                    }
                }
            }
            ds4_gpu_set_ssd_streaming(true);
            free(one);
            free(sum);
        }
        for (unsigned bi = 0; !owned && !scalar_only && bi < sizeof(budgets) / sizeof(*budgets); bi++) {
            ds4_gpu_set_streaming_expert_cache_budget(budgets[bi]);
            for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const unsigned rows = counts[ci];
                if (rows > ROWS) continue;
                const unsigned first = ROWS - rows;
                CHECK(ds4_gpu_tensor_write(x, 0, input + first * dim, rows * dim * sizeof(float)));
                CHECK(ds4_gpu_tensor_write(ids, 0, selected + first * slots, rows * slots * sizeof(int32_t)));
                CHECK(ds4_gpu_tensor_write(weights, 0, sw + first * slots, rows * slots * sizeof(float)));
                for (unsigned k = 0; k < LAYERS; k++) {
                    const unsigned layer = pass ? LAYERS - 1 - k : k;
                    const size_t g = layer * layer_bytes, u = g + EXPERTS * gate_bytes, d = u + EXPERTS * gate_bytes;
                    const float canary = 12345.0f;
                    CHECK(ds4_gpu_tensor_write(out, rows * dim * sizeof(float), &canary, sizeof(canary)));
                    bool half = false;
                    CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, model_bytes,
                        g, u, d, gate_type, down_type, gate_bytes, gate_row, down_bytes, down_row,
                        dim, mid_dim, dim, ids, weights, EXPERTS, slots, clamp, x, layer, rows, &half, false));
                    CHECK(!half && ds4_gpu_tensor_read(out, 0, actual, (rows * dim + 1) * sizeof(float)));
                    CHECK(actual[rows * dim] == canary);
                    CHECK(ds4_gpu_tensor_read(mid, 0, actual_mid, rows * slots * mid_dim * sizeof(float)));
                    exact(actual, expected + (layer * ROWS + first) * dim, rows * dim);
                    exact(actual_mid, expected_mid + (layer * ROWS + first) * slots * mid_dim, rows * slots * mid_dim);
                }
            }
        }
    }
    fprintf(stderr, "CUDA %s %s: width=%u mid=%u slots=%u rows=%u experts=%u scalar_hash=%016llx PASS\n",
            q4 ? "Q4_K" : "IQ2/Q2_K", owned ? "owned partials" : scalar_only ? "scalar" : "SSD scalar/batch exact",
            dim, mid_dim, slots, ROWS, EXPERTS,
            (unsigned long long)scalar_hash);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(down);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(0);
    ds4_gpu_cleanup();
    close(fd);
    free(model); free(input); free(expected); free(expected_mid); free(actual); free(actual_mid);
}

int main(int argc, char **argv) {
    const bool scalar_only = argc == 2 && !strcmp(argv[1], "--scalar-only");
    const bool large = argc == 2 && !strcmp(argv[1], "--owned-mmq-large");
    owned_mmq_test = large || (argc == 2 && !strcmp(argv[1], "--owned-mmq"));
    const bool owned = owned_mmq_test || (argc == 2 && !strcmp(argv[1], "--owned"));
    if (argc > 1 && !scalar_only && !owned) {
        fprintf(stderr, "usage: %s [--scalar-only|--owned|--owned-mmq|--owned-mmq-large]\n", argv[0]);
        return 2;
    }
    if (owned && !owned_mmq_test) CHECK(setenv("DS4_CUDA_MOE_NO_OWNED_MMQ", "1", 1) == 0);
    if (large) {
        run(512, 512, 6, false, true, false, 8193, 16);
        return 0;
    }
    run(512, 512, 3, scalar_only, owned, false, 8, 8);
    run(512, 512, 6, scalar_only, owned, false, 8, 8);
    run(5120, 2304, 6, scalar_only, owned, false, 8, 8);
    run(4096, 2048, 6, scalar_only, owned, false, 8, 8);
    if (owned) {
        run(5120, 2304, 6, false, true, false, 129, owned_mmq_test ? 16 : 8);
        run(4352, 512, 3, false, true, false, 129, 8);
        run(512, 512, 3, false, true, true, 8, 8);
        run(4096, 2048, 6, false, true, true, 8, 8);
        run(512, 512, 6, false, true, false, 129, 384);
        run(512, 512, 6, false, true, true, 129, 384);
    }
    return 0;
}
