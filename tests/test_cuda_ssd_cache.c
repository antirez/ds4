#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
enum { EXPERTS = 8, LAYERS = 3, ROWS = 129 };
static uint32_t random_state = 17;
static int test_prefetch_exit;
static unsigned char random_byte(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state >> 24;
}
static ds4_gpu_tensor *tensor(size_t bytes, const void *data) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t && (!data || ds4_gpu_tensor_write(t, 0, data, bytes)));
    return t;
}

static void run(unsigned gate_type, unsigned down_type, unsigned dim, unsigned selected) {
    const size_t gate_unit = gate_type == 39 ? 136 : gate_type == 12 ? 144 : 66;
    const size_t down_unit = down_type == 39 ? 136 : down_type == 12 ? 144 : 84;
    const size_t gate_block = gate_unit * (dim / 256);
    const size_t down_block = down_unit * (dim / 256);
    const size_t gate_bytes = dim * gate_block, down_bytes = dim * down_block;
    const size_t layer_bytes = EXPERTS * (gate_bytes * 2 + down_bytes);
    const size_t model_bytes = LAYERS * layer_bytes;
    unsigned char *model = malloc(model_bytes);
    CHECK(model);
    for (unsigned l = 0; l < LAYERS; l++) {
        for (unsigned part = 0; part < 3; part++) {
            const size_t block = part == 2 ? down_block : gate_block;
            const size_t offset = l * layer_bytes + part * EXPERTS * gate_bytes;
            for (unsigned r = 0; r < EXPERTS * dim; r++) {
                unsigned char *w = model + offset + r * block;
                for (size_t j = 0; j < block; j++) w[j] = random_byte();
                if (gate_type == 39) {
                    for (size_t j = 0; j < block; j += 17) w[j] = 118;
                    continue;
                }
                const size_t unit = part == 2 ? down_unit : gate_unit;
                for (size_t j = 0; j < block; j += unit) {
                    const size_t d = j + (unit == 84 ? 80 : 0);
                    w[d] = 0; w[d + 1] = 0x14;
                    if (unit != 66) { w[d + 2] = 0; w[d + 3] = 0; }
                }
            }
        }
    }
    char path[] = "/tmp/ds4-ssd-cache-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0 && unlink(path) == 0 && write(fd, model, model_bytes) == (ssize_t)model_bytes);
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model));
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    float input[ROWS * dim], weights[ROWS * selected];
    int32_t ids[ROWS * selected];
    for (unsigned i = 0; i < ROWS * dim; i++) input[i] = ((int)random_byte() - 128) / 512.0f;
    for (unsigned i = 0; i < ROWS * selected; i++) weights[i] = 1.0f / selected;
    ds4_gpu_tensor *x = tensor(sizeof(input), input), *sw = tensor(sizeof(weights), weights);
    ds4_gpu_tensor *si = tensor(sizeof(ids), NULL);
    ds4_gpu_tensor *out = tensor(ROWS * dim * 4, NULL);
    ds4_gpu_tensor *gate = tensor(ROWS * selected * dim * 4, NULL);
    ds4_gpu_tensor *up = tensor(ROWS * selected * dim * 4, NULL);
    ds4_gpu_tensor *mid = tensor(ROWS * selected * dim * 4, NULL);
    ds4_gpu_tensor *down = tensor(ROWS * selected * dim * 4, NULL);
    float reference[LAYERS][ROWS * dim], actual[ROWS * dim];
    const unsigned counts[] = {1, 7, 33, ROWS};
    for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
        const unsigned rows = counts[ci];
        for (unsigned i = 0; i < rows * selected; i++) ids[i] = (i * 3u + ci) % EXPERTS;
        CHECK(ds4_gpu_tensor_write(si, 0, ids, rows * selected * 4));
        ds4_gpu_set_ssd_streaming(false);
        for (unsigned l = 0; l < LAYERS; l++) {
            const uint64_t g = l * layer_bytes, u = g + EXPERTS * gate_bytes, d = u + EXPERTS * gate_bytes;
            bool half = false;
            CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, model_bytes,
                g, u, d, gate_type, down_type, gate_bytes, gate_block, down_bytes, down_block,
                dim, dim, dim, si, sw, EXPERTS, selected, 10, x, l, rows, &half, true));
            CHECK(!half && ds4_gpu_tensor_read(out, 0, reference[l], rows * dim * 4));
        }
        ds4_gpu_set_ssd_streaming(true);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * gate_bytes + down_bytes);
        /* More slots than one layer: selected slot IDs must not be bounded by EXPERTS. */
        const unsigned budgets[] = {12, 3, 24, 0, 10000};
        for (unsigned bi = 0; bi < sizeof(budgets) / sizeof(*budgets); bi++) {
            ds4_gpu_set_streaming_expert_cache_budget(budgets[bi]);
            CHECK(ds4_gpu_stream_expert_cache_configured_count() == budgets[bi]);
            unsigned prefetched = 0;
            for (unsigned pass = 0; pass < 3; pass++) for (unsigned k = 0; k < LAYERS; k++) {
                const unsigned l = pass & 1 ? LAYERS - 1 - k : k;
                const uint64_t g = l * layer_bytes, u = g + EXPERTS * gate_bytes, d = u + EXPERTS * gate_bytes;
                const ds4_gpu_stream_expert_table current = {
                    model, model_bytes, l, EXPERTS, g, u, d, gate_bytes, down_bytes};
                const unsigned next_layer = (l + 1u) % LAYERS;
                const uint64_t next_gate = next_layer * layer_bytes;
                const ds4_gpu_stream_expert_table next = {
                    model, model_bytes, next_layer, EXPERTS, next_gate,
                    next_gate + EXPERTS * gate_bytes, next_gate + 2 * EXPERTS * gate_bytes,
                    gate_bytes, down_bytes};
                const int started = ds4_gpu_stream_expert_cache_prefetch(&current, &next);
                if (started && test_prefetch_exit) {
                    fprintf(stderr, "CUDA SSD early exit with an outstanding reader\n");
                    exit(0);
                }
                if (budgets[bi] < 2u * EXPERTS) CHECK(!started);
                prefetched += started != 0;
                bool half = false;
                CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, model_bytes,
                    g, u, d, gate_type, down_type, gate_bytes, gate_block, down_bytes, down_block,
                    dim, dim, dim, si, sw, EXPERTS, selected, 10, x, l, rows, &half, false));
                CHECK(!half && ds4_gpu_tensor_read(out, 0, actual, rows * dim * 4));
                for (unsigned i = 0; i < rows * dim; i++)
                    CHECK(isfinite(actual[i]) && fabsf(actual[i] - reference[l][i]) <=
                          2e-5f * (1 + fabsf(reference[l][i])));
            }
            ds4_gpu_stream_expert_cache_prefetch_finish(true);
            if (budgets[bi] >= 2u * EXPERTS) CHECK(prefetched > 0);
        }
        fprintf(stderr, "CUDA SSD cache types=%u/%u width=%u selected=%u rows=%u eviction/remap/budgets: PASS\n",
                gate_type, down_type, dim, selected, rows);
    }
    const ds4_gpu_stream_expert_table table = {
        model, model_bytes, 0, EXPERTS, 0, EXPERTS * gate_bytes, 2 * EXPERTS * gate_bytes,
        gate_bytes, down_bytes};
    ds4_gpu_stream_expert_table next = table;
    next.layer = 1;
    next.gate_offset += layer_bytes;
    next.up_offset += layer_bytes;
    next.down_offset += layer_bytes;
    const int32_t one[] = {0};
    int32_t every[EXPERTS];
    for (unsigned i = 0; i < EXPERTS; i++) every[i] = i;
    ds4_gpu_set_streaming_expert_cache_budget(2u * EXPERTS);
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, one, 1));
    CHECK(ds4_gpu_stream_expert_cache_prefetch(&table, &next));
    /* Fill the rest of the active layer while the next layer's slots are
     * reserved. Neither the GPU's inputs nor the pending writes may be evicted. */
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, every, EXPERTS));
    ds4_gpu_stream_expert_cache_prefetch_finish(false);
    CHECK(ftruncate(fd, 0) == 0);
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&next, every, EXPERTS));
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, every, EXPERTS));
    fprintf(stderr, "CUDA SSD prefetch protects current inputs and publishes complete next layer: PASS\n");
    CHECK(pwrite(fd, model, model_bytes, 0) == (ssize_t)model_bytes);
    ds4_gpu_stream_expert_cache_release_resident();
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, every, EXPERTS));
    CHECK(ds4_gpu_stream_expert_cache_prefetch(&table, &next));
    ds4_gpu_stream_expert_cache_prefetch_finish(false);
    ds4_gpu_stream_expert_table third = next;
    third.layer++;
    third.gate_offset += layer_bytes;
    third.up_offset += layer_bytes;
    third.down_offset += layer_bytes;
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&third, one, 1));
    CHECK(ftruncate(fd, 0) == 0);
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, every, EXPERTS));
    CHECK(pwrite(fd, model, model_bytes, 0) == (ssize_t)model_bytes);
    fprintf(stderr, "CUDA SSD unused read-ahead evicts before demand-loaded experts: PASS\n");
    for (unsigned action = 0; action < 4; action++) {
        ds4_gpu_stream_expert_cache_release_resident();
        ds4_gpu_set_streaming_expert_cache_budget(2u * EXPERTS);
        CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, one, 1));
        if (action == 1) CHECK(ftruncate(fd, 0) == 0);
        CHECK(ds4_gpu_stream_expert_cache_prefetch(&table, &next));
        if (action == 0) ds4_gpu_stream_expert_cache_prefetch_finish(true);
        if (action == 1) ds4_gpu_stream_expert_cache_prefetch_finish(false);
        if (action == 2) ds4_gpu_set_streaming_expert_cache_budget(3);
        if (action == 3) CHECK(ds4_gpu_set_model_fd_for_map(fd, model));
        CHECK(ftruncate(fd, 0) == 0);
        CHECK(!ds4_gpu_stream_expert_cache_begin_selected_load(&next, one, 1));
        CHECK(pwrite(fd, model, model_bytes, 0) == (ssize_t)model_bytes);
        CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&next, every, EXPERTS));
    }
    fprintf(stderr, "CUDA SSD prefetch cancellation/read failure/budget/FD teardown do not publish partial slots: PASS\n");
    ds4_gpu_set_streaming_expert_cache_budget(3);
    const int32_t first[] = {0, 1}, protected_hits[] = {2, 0, 1}, hit[] = {0}, miss[] = {7};
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, first, 2));
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, protected_hits, 3));
    CHECK(ds4_gpu_stream_expert_cache_current_count() == 3);
    CHECK(ftruncate(fd, 0) == 0);
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, hit, 1));
    CHECK(!ds4_gpu_stream_expert_cache_begin_selected_load(&table, miss, 1));
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, hit, 1));
    const int32_t invalid[] = {-1, EXPERTS};
    CHECK(!ds4_gpu_stream_expert_cache_begin_selected_load(&table, invalid, 1));
    CHECK(!ds4_gpu_stream_expert_cache_begin_selected_load(&table, invalid + 1, 1));
    fprintf(stderr, "CUDA SSD cache hits survive unreadable source; failed misses and invalid IDs: PASS\n");
    /* Fail after an earlier miss has queued its copies. Its completed slot
     * must remain usable, and no upload may outlive the failed request. */
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_streaming_expert_cache_budget(3);
    CHECK(pwrite(fd, model, model_bytes, 0) == (ssize_t)model_bytes);
    CHECK(ftruncate(fd, 2 * EXPERTS * gate_bytes + down_bytes) == 0);
    const int32_t partial[] = {0, 7};
    CHECK(!ds4_gpu_stream_expert_cache_begin_selected_load(&table, partial, 2));
    CHECK(ftruncate(fd, 0) == 0);
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&table, hit, 1));
    fprintf(stderr, "CUDA SSD cache partial upload failure drains earlier copies: PASS\n");
    /* A fresh model identity clears the whole-tensor cache. Once the selected
     * experts are loaded, prefill must not read their original tensors again. */
    unsigned char *alias = calloc(1, model_bytes);
    CHECK(alias && pwrite(fd, model, model_bytes, 0) == (ssize_t)model_bytes);
    CHECK(ds4_gpu_set_model_fd_for_map(fd, alias));
    CHECK(ds4_gpu_set_model_map(alias, model_bytes));
    ds4_gpu_set_streaming_expert_cache_budget(24);
    ds4_gpu_stream_expert_table cached = table;
    cached.model_map = alias;
    int32_t all[EXPERTS];
    for (unsigned i = 0; i < EXPERTS; i++) all[i] = i;
    CHECK(ds4_gpu_stream_expert_cache_begin_selected_load(&cached, all, EXPERTS));
    CHECK(ftruncate(fd, 0) == 0);
    bool half = false;
    CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, alias, model_bytes,
        0, EXPERTS * gate_bytes, 2 * EXPERTS * gate_bytes,
        gate_type, down_type, gate_bytes, gate_block, down_bytes, down_block,
        dim, dim, dim, si, sw, EXPERTS, selected, 10, x, 0, ROWS, &half, false));
    CHECK(!half && ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
    for (unsigned i = 0; i < ROWS * dim; i++)
        CHECK(isfinite(actual[i]) && fabsf(actual[i] - reference[0][i]) <=
              2e-5f * (1 + fabsf(reference[0][i])));
    fprintf(stderr, "CUDA SSD prefill uses only cached experts with unreadable source: PASS\n");
    /* Consecutive full-layer chunks may reuse packed weights, but never the
     * previous token IDs. Shift the input rows and cross the 128-row boundary. */
    const unsigned chunks[] = {ROWS - 1, ROWS, ROWS - 1};
    for (unsigned step = 0; step < sizeof(chunks) / sizeof(*chunks); step++) {
        const unsigned rows = chunks[step], first_row = ROWS - rows;
        CHECK(ds4_gpu_tensor_write(x, 0, input + first_row * dim, rows * dim * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(si, 0, ids + first_row * selected, rows * selected * sizeof(int32_t)));
        const float marker = 12345;
        if (rows < ROWS) CHECK(ds4_gpu_tensor_write(out, rows * dim * sizeof(float), &marker, sizeof(marker)));
        CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, alias, model_bytes,
            0, EXPERTS * gate_bytes, 2 * EXPERTS * gate_bytes,
            gate_type, down_type, gate_bytes, gate_block, down_bytes, down_block,
            dim, dim, dim, si, sw, EXPERTS, selected, 10, x, 0, rows, &half, false));
        CHECK(!half && ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
        if (rows < ROWS) CHECK(actual[rows * dim] == marker);
        for (unsigned i = 0; i < rows * dim; i++)
            CHECK(isfinite(actual[i]) && fabsf(actual[i] - reference[0][first_row * dim + i]) <=
                  2e-5f * (1 + fabsf(reference[0][first_row * dim + i])));
    }
    fprintf(stderr, "CUDA SSD consecutive prefill chunks refresh IDs and preserve output bounds: PASS\n");
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(sw); ds4_gpu_tensor_free(si);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(down);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(0);
    ds4_gpu_cleanup();
    close(fd); free(alias); free(model);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--prefetch-exit")) test_prefetch_exit = 1;
    else CHECK(argc == 1);
    run(16, 10, 512, 2);
    run(12, 12, 512, 2);
    run(39, 39, 512, 2);
    /* Cross both the aligned-width and fused-prefill assignment thresholds. */
    run(16, 10, 1024, 8);
    return 0;
}
