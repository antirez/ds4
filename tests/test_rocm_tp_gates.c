#include "ds4_gpu.h"
#include "../ds4_tp.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static float *cpu_input, *cpu_output;
static int inject_failure;

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, err); goto done; \
} } while (0)

static ds4_gpu_tensor *guard_storage[4];
static uint64_t guard_bytes[4];
static unsigned guard_count;
static ds4_gpu_tensor *guarded_alloc(uint64_t bytes, bool coherent) {
    if (guard_count == 4 || bytes > UINT64_MAX - 128) return NULL;
    ds4_gpu_tensor *storage = coherent ? ds4_gpu_tensor_alloc_coherent(bytes + 128) :
                                         ds4_gpu_tensor_alloc(bytes + 128);
    if (!storage) return NULL;
    unsigned char guard[64]; memset(guard, 0xa5, sizeof(guard));
    ds4_gpu_tensor *view = NULL;
    if (ds4_gpu_tensor_write(storage, 0, guard, 64) &&
        ds4_gpu_tensor_write(storage, bytes + 64, guard, 64))
        view = ds4_gpu_tensor_view(storage, 64, bytes);
    if (!view) { ds4_gpu_tensor_free(storage); return NULL; }
    guard_storage[guard_count] = storage; guard_bytes[guard_count++] = bytes;
    return view;
}
static int check_guards(void) {
    if (!ds4_gpu_synchronize()) return 0;
    for (unsigned j=0;j<guard_count;++j) {
        unsigned char a[64],b[64];
        if (!ds4_gpu_tensor_read(guard_storage[j],0,a,64) ||
            !ds4_gpu_tensor_read(guard_storage[j],guard_bytes[j]+64,b,64)) return 0;
        for (unsigned i=0;i<64;++i) if(a[i]!=0xa5 || b[i]!=0xa5) return 0;
    }
    return 1;
}

static int exchange(void *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    if (inject_failure && seq == 3) { ds4_tp_mark_failed(tp); return 0; }
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
        float *input = cpu_input;
        for (uint32_t j = 0; j < count; j++) input[j] = epoch + j % 127u + rank * 1000;
        if (!ds4_gpu_tensor_write(x, 0, input, bytes)) return 0;
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
            ds4_gpu_tp_add_tensor(out, a, b, count);
        if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = 0;
        if (ds4_gpu_tp_failed()) ok = 0;
        if (!ds4_gpu_tensor_read(out, 0, cpu_output, bytes) || ds4_gpu_tp_failed()) ok = 0;
        const float *actual = cpu_output;
        for (uint32_t j = 0; ok && j < count; j++)
            if (actual[j] != 2u * (epoch + j % 127u) + 1000u) ok = 0;
        ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
        if (!ok) return 0;
    }
    return !batch || ds4_tp_batch_block_end(tp);
}

int main(int argc, char **argv) {
    if (argc < 6 || argc > 9) {
        fprintf(stderr, "usage: %s RANK COORDINATOR PORT tcp|usb4stream DEVICE [fail]\n", argv[0]);
        return 2;
    }
    const int rank = atoi(argv[1]);
    if (rank != 0 && rank != 1) return 2;
    ds4_tp_options opt = {.role = rank ? DS4_TP_WORKER : DS4_TP_LEADER,
        .listen_host = argv[2], .leader_host = argv[2],
        .listen_port = atoi(argv[3]), .leader_port = atoi(argv[3]),
        .transport = !strcmp(argv[4], "usb4stream") ? DS4_TP_TRANSPORT_USB4STREAM : DS4_TP_TRANSPORT_TCP,
        .usb4stream_device = !strcmp(argv[4], "usb4stream") ? argv[5] : NULL};
    const bool roce = !strcmp(argv[4], "rdma");
    const int base_argc = roce ? 8 : 6;
    if ((argc != base_argc && argc != base_argc + 1) ||
        (!roce && strcmp(argv[4], "usb4stream") && strcmp(argv[4], "tcp"))) return 2;
    if (roce) {
        opt.transport = DS4_TP_TRANSPORT_RDMA;
        opt.rdma_device = argv[5]; opt.rdma_port = atoi(argv[6]);
        opt.rdma_gid_index = atoi(argv[7]); opt.rdma_gid_index_set = true;
    }
    inject_failure = argc == base_argc + 1 && !strcmp(argv[base_argc], "fail");
    ds4_tp_identity id = {.gguf_bytes = 1, .n_layer = 40, .n_embd = 5120,
        .n_vocab = 16, .ctx_size = 8192};
    char err[256] = "";
    ds4_tp *tp = NULL;
    ds4_gpu_tensor *slab = NULL, *x = NULL, *out = NULL, *in = NULL;
    const uint64_t vec = 5120u * sizeof(float), size = 2048u * vec;
    int rc = 1;
    CHECK(ds4_gpu_init());
    ds4_gpu_set_deepseek41_model(true);
    cpu_input = malloc(size); cpu_output = malloc(size);
    CHECK(cpu_input && cpu_output);
    CHECK(ds4_tp_create(&tp, &opt, &id, err, sizeof(err)));
    slab = guarded_alloc(ds4_tp_slab_bytes(40, 5120), true);
    x = guarded_alloc(size, false);
    out = guarded_alloc(size, false);
    in = guarded_alloc(size, false);
    CHECK(slab && x && out && in);
    memset(ds4_gpu_tensor_contents(slab), 0, ds4_tp_slab_bytes(40, 5120));
    CHECK(ds4_tp_attach_slab(tp, ds4_gpu_tensor_contents(slab), err, sizeof(err)));
    CHECK(ds4_gpu_tp_init(rank, slab, ds4_tp_slab_gpu_flags_offset(tp),
        ds4_tp_slab_out_offset(tp, 0, 0), vec, exchange, tp));
    ds4_gpu_tp_set_big_exchange(exchange_big);
    ds4_gpu_tp_set_batch_exchange(exchange_batch);
    float *input = cpu_input;
    if (inject_failure) {
        CHECK(!small_gates(tp, slab, x, out, 0, rank, false));
        CHECK(ds4_gpu_tp_failed() && ds4_tp_failed(tp));
        CHECK(!ds4_gpu_tp_gate_encode(0, 0));
        CHECK(check_guards());
        fprintf(stderr, "failed gate remains failed; no later completion PASS\n");
        rc = 0; goto done;
    }
    const uint32_t rows[] = {1, 8, 65, 2048};
    for (unsigned shape = 0; shape < sizeof(rows) / sizeof(*rows); shape++) {
        /* Exercise unequal arrival and decode/batch/bulk mode transitions. */
        if (rank == (int)(shape % 2u)) usleep(200000);
        CHECK(small_gates(tp, slab, x, out, shape, rank, false));
        CHECK(small_gates(tp, slab, x, out, shape, rank, true));
        const uint32_t count = rows[shape] * 5120u;
        for (uint32_t layer = 0; layer < 4; layer++) {
            const uint32_t epoch = (shape * 40u + layer) * 37u;
            for (uint32_t j = 0; j < count; j++)
                input[j] = epoch + j % 127u + rank * 1000;
            CHECK(ds4_gpu_tensor_write(x, 0, input, count * sizeof(float)));
            CHECK(ds4_gpu_begin_commands());
            /* Incoming storage was just written by the GPU, as with a dead
             * activation buffer reused for a TP peer's partial output. */
            CHECK(ds4_gpu_tensor_copy(in, 0, x, 0, count * sizeof(float)));
            CHECK(ds4_gpu_add_tensor(out, x, x, count));
            CHECK(ds4_gpu_tp_big_gate_encode(layer, rows[shape], out, in, count * sizeof(float)));
            CHECK(ds4_gpu_tp_add_tensor(out, in, x, count));
            CHECK(ds4_gpu_tensor_copy(in, 0, x, 0, count * sizeof(float)));
            CHECK(ds4_gpu_tp_big_gate_encode(layer, rows[shape], out, in, count * sizeof(float)));
            CHECK(ds4_gpu_tp_add_tensor(out, in, x, count));
            CHECK(ds4_gpu_end_commands() && !ds4_gpu_tp_failed());
            CHECK(ds4_gpu_tensor_read(out, 0, cpu_output, count * sizeof(float)));
            const float *actual = cpu_output;
            for (uint32_t j = 0; j < count; j++) {
                const float expected = 4u * (epoch + j % 127u) + (1 + 2 * rank) * 1000;
                if (actual[j] != expected) {
                    fprintf(stderr, "rank=%d rows=%u layer=%u index=%u expected=%g actual=%g\n",
                        rank, rows[shape], layer, j, expected, actual[j]);
                    goto done;
                }
            }
        }
        CHECK(check_guards());
        fprintf(stderr, "ROCm decode/verify/bulk GPU reuse rank=%d rows=%u: exact PASS\n", rank, rows[shape]);
    }
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tp_shutdown();
    ds4_tp_free(tp);
    ds4_gpu_tensor_free(in); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(slab);
    for(unsigned i=0;i<guard_count;++i) ds4_gpu_tensor_free(guard_storage[i]);
    ds4_gpu_cleanup();
    free(cpu_input); free(cpu_output);
    return rc;
}
