/* CUDA network gates: device visibility, bounded staging and failed peers. */
#include "ds4_gpu.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(void) {
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
