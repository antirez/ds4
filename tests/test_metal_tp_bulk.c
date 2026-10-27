#include "ds4_gpu.h"
#include "ds4_tp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, err); goto done; \
} } while (0)

static int exchange(void *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    return ds4_tp_gate_exchange(tp, layer, gate, seq);
}

static int exchange_big(void *tp, uint32_t layer, uint64_t seq,
                         const void *out, void *in, uint64_t bytes) {
    return ds4_tp_big_gate_exchange(tp, layer, seq, out, in, bytes);
}

static int exchange_batch(void *tp, uint32_t layer, uint32_t rows, uint64_t seq) {
    return ds4_tp_batch_gate_exchange(tp, layer, rows, seq);
}

static int small_gates(ds4_tp *tp, ds4_gpu_tensor *slab, ds4_gpu_tensor *x,
                       ds4_gpu_tensor *out, unsigned shape, int rank,
                       bool batch) {
    const uint32_t rows = batch ? 1u + shape % 8u : 1u;
    const uint32_t count = rows * 5120u;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (batch && !ds4_tp_batch_block_begin(tp, rows, 40)) return 0;
    for (uint32_t slot = 0; slot < (batch ? 40u : 80u); slot++) {
        const uint32_t layer = batch ? slot : slot / 2u, gate = slot % 2u;
        const uint32_t epoch = (shape * 80u + slot) * 37u;
        float *input = ds4_gpu_tensor_contents(x);
        for (uint32_t j = 0; j < count; j++) input[j] = epoch + j % 127u + rank * 1000;
        const uint64_t oo = batch ? ds4_tp_slab_batch_out_offset(tp, layer) :
            ds4_tp_slab_out_offset(tp, layer, gate);
        const uint64_t io = batch ? ds4_tp_slab_batch_in_offset(tp, layer) :
            ds4_tp_slab_in_offset(tp, layer, gate);
        ds4_gpu_tensor *a = ds4_gpu_tensor_view(slab, oo, bytes);
        ds4_gpu_tensor *b = ds4_gpu_tensor_view(slab, io, bytes);
        int ok = a && b && ds4_gpu_begin_commands() &&
            ds4_gpu_tensor_copy(a, 0, x, 0, bytes) &&
            (batch ? ds4_gpu_tp_batch_gate_encode(layer, rows) :
                     ds4_gpu_tp_gate_encode(layer, gate)) &&
            ds4_gpu_add_tensor(out, a, b, count);
        if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = 0;
        if (ds4_gpu_tp_failed()) ok = 0;
        const float *actual = ds4_gpu_tensor_contents(out);
        for (uint32_t j = 0; ok && j < count; j++)
            if (actual[j] != 2u * (epoch + j % 127u) + 1000u) ok = 0;
        ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
        if (!ok) return 0;
    }
    return !batch || ds4_tp_batch_block_end(tp);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s RANK COORDINATOR PORT RDMA_DEVICE GID\n", argv[0]);
        return 2;
    }
    const int rank = atoi(argv[1]);
    if (rank != 0 && rank != 1) return 2;
    ds4_tp_options opt = {.role = rank ? DS4_TP_WORKER : DS4_TP_LEADER,
        .listen_host = argv[2], .leader_host = argv[2],
        .listen_port = atoi(argv[3]), .leader_port = atoi(argv[3]),
        .transport = DS4_TP_TRANSPORT_RDMA, .rdma_device = argv[4],
        .rdma_gid_index = atoi(argv[5]), .rdma_gid_index_set = true};
    ds4_tp_identity id = {.gguf_bytes = 1, .n_layer = 40, .n_embd = 5120,
        .n_vocab = 16, .ctx_size = 8192};
    char err[256] = "";
    ds4_tp *tp = NULL;
    ds4_gpu_tensor *slab = NULL, *x = NULL, *out = NULL, *in = NULL;
    const uint64_t vec = 5120u * sizeof(float), size = 8192u * vec;
    int rc = 1;
    CHECK(ds4_gpu_init());
    CHECK(ds4_tp_create(&tp, &opt, &id, err, sizeof(err)));
    slab = ds4_gpu_tensor_alloc(ds4_tp_slab_bytes(40, 5120));
    x = ds4_gpu_tensor_alloc(size);
    out = ds4_gpu_tensor_alloc(size);
    in = ds4_gpu_tensor_alloc(size);
    CHECK(slab && x && out && in);
    memset(ds4_gpu_tensor_contents(slab), 0, ds4_tp_slab_bytes(40, 5120));
    CHECK(ds4_tp_attach_slab(tp, ds4_gpu_tensor_contents(slab), err, sizeof(err)));
    CHECK(ds4_gpu_tp_init(rank, slab, ds4_tp_slab_gpu_flags_offset(tp),
        ds4_tp_slab_out_offset(tp, 0, 0), vec, exchange, tp));
    ds4_gpu_tp_set_big_exchange(exchange_big);
    ds4_gpu_tp_set_batch_exchange(exchange_batch);
    float *input = ds4_gpu_tensor_contents(x);
    const uint32_t rows[] = {256, 2048, 2047, 4096, 8192, 2048, 513};
    for (unsigned shape = 0; shape < sizeof(rows) / sizeof(*rows); shape++) {
        /* A newly armed UC receive window must tolerate unequal arrival times,
         * both at startup and after bulk prefill returns to decode. */
        if (rank == (int)(shape % 2u)) usleep(200000);
        CHECK(small_gates(tp, slab, x, out, shape, rank, false));
        CHECK(small_gates(tp, slab, x, out, shape, rank, true));
        const uint32_t count = rows[shape] * 5120u;
        for (uint32_t layer = 0; layer < 40; layer++) {
            const uint32_t epoch = (shape * 40u + layer) * 37u;
            for (uint32_t j = 0; j < count; j++)
                input[j] = epoch + j % 127u + rank * 1000;
            CHECK(ds4_gpu_begin_commands());
            /* Incoming storage was just written by the GPU, as with a dead
             * activation buffer reused for a TP peer's partial output. */
            CHECK(ds4_gpu_tensor_copy(in, 0, x, 0, count * sizeof(float)));
            CHECK(ds4_gpu_add_tensor(out, x, x, count));
            CHECK(ds4_gpu_tp_big_gate_encode(layer, rows[shape], out, in, count * sizeof(float)));
            CHECK(ds4_gpu_add_tensor(out, in, x, count));
            CHECK(ds4_gpu_tensor_copy(in, 0, x, 0, count * sizeof(float)));
            CHECK(ds4_gpu_tp_big_gate_encode(layer, rows[shape], out, in, count * sizeof(float)));
            CHECK(ds4_gpu_add_tensor(out, in, x, count));
            CHECK(ds4_gpu_end_commands() && !ds4_gpu_tp_failed());
            const float *actual = ds4_gpu_tensor_contents(out);
            for (uint32_t j = 0; j < count; j++) {
                const float expected = 4u * (epoch + j % 127u) + (1 + 2 * rank) * 1000;
                if (actual[j] != expected) {
                    fprintf(stderr, "rank=%d rows=%u layer=%u index=%u expected=%g actual=%g\n",
                        rank, rows[shape], layer, j, expected, actual[j]);
                    goto done;
                }
            }
        }
        fprintf(stderr, "RDMA decode/verify/bulk GPU reuse rank=%d rows=%u: exact PASS\n", rank, rows[shape]);
    }
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tp_shutdown();
    ds4_tp_free(tp);
    ds4_gpu_tensor_free(in); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(slab); ds4_gpu_cleanup();
    return rc;
}
