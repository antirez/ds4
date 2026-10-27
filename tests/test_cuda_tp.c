/* CUDA network gates: device visibility, bounded staging and failed peers. */
#include "ds4_gpu.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

static void attention_ranges(unsigned dim) {
    enum { TOKENS = 129, HEADS = 4, RATIO = 4, COMPS = TOKENS / RATIO };
    const unsigned DIM = dim;
    assert(ds4_gpu_init());
    const uint64_t row_bytes = HEADS * DIM * sizeof(float);
    float sinks[HEADS] = {-0.3f, 0.1f, 0.7f, -1.0f};
    char path[] = "/tmp/ds4-tp-attention-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0 && write(fd, sinks, sizeof(sinks)) == sizeof(sinks));
    const uint64_t offset = 0, size = sizeof(sinks);
    assert(ds4_gpu_set_model_map_spans(sinks, sizeof(sinks), &offset, &size, 1, 0));
    assert(ds4_gpu_set_model_fd_for_map(fd, sinks));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(TOKENS * row_bytes);
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(TOKENS * DIM * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(COMPS * DIM * sizeof(float));
    ds4_gpu_tensor *full = ds4_gpu_tensor_alloc(TOKENS * row_bytes);
    ds4_gpu_tensor *partial = ds4_gpu_tensor_alloc(TOKENS * row_bytes);
    float *input = malloc(TOKENS * row_bytes);
    float *expected = malloc(TOKENS * row_bytes);
    float *actual = malloc(TOKENS * row_bytes);
    assert(q && raw && comp && full && partial && input && expected && actual);
    for (unsigned i = 0; i < TOKENS * HEADS * DIM; i++)
        input[i] = ((int)(i * 17u % 43u) - 21) / 32.0f;
    assert(ds4_gpu_tensor_write(q, 0, input, TOKENS * row_bytes));
    assert(ds4_gpu_tensor_write(raw, 0, input, TOKENS * DIM * sizeof(float)));
    assert(ds4_gpu_tensor_write(comp, 0, input + 19, COMPS * DIM * sizeof(float)));
    for (unsigned mixed = 0; mixed < 2; mixed++) {
        if (mixed) assert(ds4_gpu_attention_prefill_static_mixed_heads_tensor(
            full, sinks, sizeof(sinks), 0, q, raw, comp, 0, TOKENS, COMPS,
            16, RATIO, HEADS, DIM));
        else assert(ds4_gpu_attention_prefill_raw_heads_tensor(
            full, sinks, sizeof(sinks), 0, q, raw, TOKENS, 16, HEADS, DIM));
        assert(ds4_gpu_tensor_read(full, 0, expected, TOKENS * row_bytes));
        const unsigned starts[] = {0, 1, 64, 128};
        const unsigned counts[] = {1, 63, 64, 1};
        assert(ds4_gpu_tensor_fill_f32(partial, NAN, TOKENS * HEADS * DIM));
        for (unsigned i = 0; i < 4; i++) {
            const unsigned first = starts[i], count = counts[i];
            ds4_gpu_tensor *qs = ds4_gpu_tensor_view(q, first * row_bytes, count * row_bytes);
            ds4_gpu_tensor *ys = ds4_gpu_tensor_view(partial, first * row_bytes, count * row_bytes);
            assert(qs && ys);
            if (mixed) assert(ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
                ys, sinks, sizeof(sinks), 0, qs, raw, comp, 0, first, count,
                TOKENS, COMPS, 16, RATIO, HEADS, DIM));
            else assert(ds4_gpu_attention_prefill_raw_heads_range_tensor(
                ys, sinks, sizeof(sinks), 0, qs, raw, first, count, TOKENS,
                16, HEADS, DIM));
            ds4_gpu_tensor_free(ys); ds4_gpu_tensor_free(qs);
        }
        assert(ds4_gpu_tensor_read(partial, 0, actual, TOKENS * row_bytes));
        for (unsigned i = 0; i < TOKENS * HEADS * DIM; i++)
            assert(isfinite(actual[i]) && fabsf(actual[i] - expected[i]) < 0.00002f);
    }
    assert(!ds4_gpu_attention_prefill_raw_heads_range_tensor(
        partial, sinks, sizeof(sinks), 0, q, raw, TOKENS, 1, TOKENS, 16, HEADS, DIM));
    ds4_gpu_tensor_free(partial); ds4_gpu_tensor_free(full);
    ds4_gpu_tensor_free(comp); ds4_gpu_tensor_free(raw); ds4_gpu_tensor_free(q);
    free(actual); free(expected); free(input);
    ds4_gpu_cleanup();
    assert(close(fd) == 0 && unlink(path) == 0);
    printf("CUDA TP attention row slices, causal/raw/compressed masks (dim=%u): PASS\n", dim);
}

enum { WIDTH = 5120, ROWS = 513 };
typedef struct {
    ds4_gpu_tensor *slab;
    float *host;
    uint64_t row_seq, batch_seq;
    unsigned calls;
    int fail;
} gate_test;

static int row(void *ud, uint32_t layer, uint32_t gate, uint64_t seq) {
    gate_test *t = ud;
    assert(layer == 3 && gate == 1 && seq == ++t->row_seq);
    t->calls++;
    if (t->fail) return 0;
    assert(ds4_gpu_tensor_read(t->slab, 0, t->host, WIDTH * sizeof(float)));
    for (unsigned i = 0; i < WIDTH; i++) {
        assert(t->host[i] == (float)i);
        t->host[i] += 1;
    }
    return ds4_gpu_tensor_write(t->slab, WIDTH * sizeof(float), t->host, WIDTH * sizeof(float));
}

static int batch(void *ud, uint32_t layer, uint32_t rows, uint64_t seq) {
    gate_test *t = ud;
    assert(layer == 5 && rows == 3 && seq == ++t->batch_seq);
    t->calls++;
    return !t->fail;
}

static int big(void *ud, uint32_t layer, uint64_t seq, const void *out, void *in, uint64_t bytes) {
    gate_test *t = ud;
    assert(layer == 7 && seq == ++t->batch_seq && bytes % (WIDTH * sizeof(float)) == 0);
    t->calls++;
    if (t->fail) return 0;
    const float *a = out;
    float *b = in;
    for (uint64_t i = 0; i < bytes / sizeof(float); i++) {
        assert(a[i] == (float)(i % WIDTH));
        b[i] = a[i] + 2;
    }
    return 1;
}

static void attention_head_layout(void) {
    enum { GROUPS = 4, GROUP_DIM = 64, RANK = 32, OUTPUT = 32,
           A_BYTES = GROUPS * RANK * (GROUP_DIM / 32) * 34,
           B_BYTES = OUTPUT * (GROUPS * RANK / 32) * 34 };
    unsigned char model[A_BYTES + B_BYTES];
    for (unsigned b = 0; b < sizeof(model) / 34; b++) {
        const uint16_t scale = 0x2000;
        memcpy(model + b * 34, &scale, 2);
        for (unsigned k = 0; k < 32; k++)
            model[b * 34 + 2 + k] = (unsigned char)((b * 7 + k * 3) % 127);
    }
    char path[] = "/tmp/ds4-tp-heads-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0 && write(fd, model, sizeof(model)) == sizeof(model));
    const uint64_t offset = 0, size = sizeof(model);
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map_spans(model, size, &offset, &size, 1, 0));
    assert(ds4_gpu_set_model_fd_for_map(fd, model));
    float input[GROUPS * GROUP_DIM], expected[OUTPUT], actual[OUTPUT];
    for (unsigned i = 0; i < GROUPS * GROUP_DIM; i++) input[i] = (int)(i % 23) - 11;
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(GROUPS * RANK * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(expected));
    ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(2 * sizeof(expected));
    assert(heads && low && out && slab);
    assert(ds4_gpu_tensor_write(heads, 0, input, sizeof(input)));
    for (unsigned rank = 0; rank < 2; rank++) {
        assert(ds4_gpu_attention_output_q8_tp_tensor(out, low, model, size,
            0, A_BYTES, GROUP_DIM, RANK, GROUPS, rank * (GROUPS / 2),
            GROUPS / 2, OUTPUT, heads));
        assert(ds4_gpu_tensor_read(out, 0, expected, sizeof(expected)));
        ds4_gpu_tensor *compact = ds4_gpu_tensor_view(heads,
            rank * sizeof(input) / 2, sizeof(input) / 2);
        assert(compact && ds4_gpu_tp_init(rank, slab, sizeof(expected), 0,
            sizeof(expected), row, NULL));
        assert(ds4_gpu_attention_output_q8_tp_tensor(out, low, model, size,
            0, A_BYTES, GROUP_DIM, RANK, GROUPS, rank * (GROUPS / 2),
            GROUPS / 2, OUTPUT, compact));
        assert(ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)));
        assert(memcmp(expected, actual, sizeof(actual)) == 0);
        ds4_gpu_tp_shutdown();
        ds4_gpu_tensor_free(compact);
    }
    ds4_gpu_tensor_free(slab); ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
    assert(close(fd) == 0 && unlink(path) == 0);
    puts("CUDA TP compact and intra-host attention head layouts: exact PASS");
}

int main(void) {
    attention_ranges(32);
    attention_ranges(512);
    attention_head_layout();
    assert(ds4_gpu_init());
    const uint64_t vec = WIDTH * sizeof(float), bytes = ROWS * vec;
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(bytes + vec);
    ds4_gpu_tensor *in = ds4_gpu_tensor_alloc(bytes + vec);
    ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(2 * vec);
    float *host = malloc((size_t)bytes + vec);
    assert(out && in && slab && host);
    for (uint64_t i = 0; i < bytes / sizeof(float); i++) host[i] = (float)(i % WIDTH);
    assert(ds4_gpu_tensor_write(out, 0, host, bytes));
    assert(ds4_gpu_tensor_write(slab, 0, host, vec));
    assert(!ds4_gpu_tp_gate_encode(0, 0));
    for (unsigned cycle = 0; cycle < 2; cycle++) {
        gate_test t = {.slab = slab, .host = host};
        assert(!ds4_gpu_tp_init(2, slab, vec, 0, vec, row, &t));
        assert(ds4_gpu_tp_init(cycle, slab, vec, 0, vec, row, &t));
        assert(!ds4_gpu_tp_init(cycle, slab, vec, 0, vec, row, &t));
        ds4_gpu_tp_set_batch_exchange(batch);
        ds4_gpu_tp_set_big_exchange(big);
        assert(ds4_gpu_tp_gate_encode(3, 1));
        assert(ds4_gpu_tensor_read(slab, vec, host, vec));
        for (unsigned i = 0; i < WIDTH; i++) assert(host[i] == (float)i + 1);
        assert(ds4_gpu_tp_batch_gate_encode(5, 3));
        const unsigned counts[] = {1, 7, ROWS, 3};
        for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
            const uint64_t used = counts[c] * vec;
            assert(ds4_gpu_tensor_fill_f32(in, -100, (bytes + vec) / sizeof(float)));
            assert(ds4_gpu_tp_big_gate_encode(7, counts[c], out, in, used));
            assert(ds4_gpu_tensor_read(in, 0, host, used + sizeof(float)));
            for (uint64_t i = 0; i < used / sizeof(float); i++)
                assert(host[i] == (float)(i % WIDTH) + 2);
            assert(host[used / sizeof(float)] == -100);
        }
        assert(!ds4_gpu_tp_big_gate_encode(7, 1, out, in, vec - 1));
        t.fail = 1;
        assert(!ds4_gpu_tp_gate_encode(3, 1) && ds4_gpu_tp_failed());
        const unsigned calls = t.calls;
        assert(!ds4_gpu_tp_gate_encode(3, 1));
        assert(!ds4_gpu_tp_batch_gate_encode(5, 3));
        assert(!ds4_gpu_tp_big_gate_encode(7, 1, out, in, vec));
        assert(t.calls == calls);
        ds4_gpu_tp_shutdown();
        assert(!ds4_gpu_tp_failed());
    }
    gate_test final = {.slab = slab, .host = host};
    assert(ds4_gpu_tp_init(0, slab, vec, 0, vec, row, &final));
    ds4_gpu_tp_set_big_exchange(big);
    assert(ds4_gpu_tp_big_gate_encode(7, 1, out, in, vec));
    ds4_gpu_tensor_free(slab); ds4_gpu_tensor_free(in); ds4_gpu_tensor_free(out);
    free(host);
    ds4_gpu_cleanup();
    assert(!ds4_gpu_tp_gate_encode(3, 1));
    puts("CUDA TP row/batch/bulk visibility, canaries, failure and rebind: PASS");
    return 0;
}
