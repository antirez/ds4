/* Two-host CUDA/RDMA handoff test, without model weights. */
#include "ds4_gpu.h"
#include "ds4_gpu_tp.h"
#include "ds4_tp.h"
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    ds4_tp *tp;
    ds4_gpu_tensor *slab;
    void *staging;
    uint64_t vec;
} handoff;

static int row_exchange(void *ud, uint32_t layer, uint32_t gate, uint64_t seq) {
    handoff *h = ud;
    uint64_t out = ds4_tp_slab_out_offset(h->tp, layer, gate);
    uint64_t in = ds4_tp_slab_in_offset(h->tp, layer, gate);
    return (!h->staging || ds4_gpu_tensor_read(h->slab, out,
                (char *)h->staging + out, h->vec)) &&
        ds4_tp_gate_exchange(h->tp, layer, gate, seq) &&
        (!h->staging || ds4_gpu_tensor_write(h->slab, in,
                (char *)h->staging + in, h->vec));
}

static int batch_exchange(void *ud, uint32_t layer, uint32_t rows, uint64_t seq) {
    handoff *h = ud;
    uint64_t out = ds4_tp_slab_batch_out_offset(h->tp, layer);
    uint64_t in = ds4_tp_slab_batch_in_offset(h->tp, layer);
    return (!h->staging || ds4_gpu_tensor_read(h->slab, out,
                (char *)h->staging + out, rows * h->vec)) &&
        ds4_tp_batch_gate_exchange(h->tp, layer, rows, seq) &&
        (!h->staging || ds4_gpu_tensor_write(h->slab, in,
                (char *)h->staging + in, rows * h->vec));
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", \
    __FILE__, __LINE__, #x, error); goto done; } } while (0)

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s RANK COORDINATOR PORT tcp|rdma DEVICE GID\n", argv[0]);
        return 2;
    }
    const int rank = atoi(argv[1]), port = atoi(argv[3]);
    if ((rank != 0 && rank != 1) || port <= 0 || port > 65535 ||
        (strcmp(argv[4], "tcp") && strcmp(argv[4], "rdma"))) return 2;
    ds4_tp_options opt = {.role = rank ? DS4_TP_WORKER : DS4_TP_LEADER,
        .listen_host = argv[2], .leader_host = argv[2],
        .listen_port = port, .leader_port = port,
        .transport = !strcmp(argv[4], "rdma") ? DS4_TP_TRANSPORT_RDMA : DS4_TP_TRANSPORT_TCP,
        .rdma_device = argv[5], .rdma_gid_index = atoi(argv[6]), .rdma_gid_index_set = true};
    ds4_tp_identity id = {.gguf_bytes = 1, .n_layer = 40, .n_embd = 5120,
        .n_vocab = 16, .ctx_size = 8192, .gate_slot_step = 1, .gates_per_token = 80};
    handoff h = {.vec = (uint64_t)id.n_embd * sizeof(float)};
    const uint64_t slab_bytes = ds4_tp_slab_bytes(id.n_layer, id.n_embd);
    ds4_gpu_tensor *input = NULL, *result = NULL;
    ds4_gpu_tensor *out[80] = {0}, *in[80] = {0};
    ds4_gpu_tensor *bout[80] = {0}, *bin[80] = {0};
    float *host = NULL;
    char error[256] = "";
    int rc = 1;
    CHECK(ds4_gpu_init());
    CHECK(ds4_tp_create(&h.tp, &opt, &id, error, sizeof(error)));
    h.slab = ds4_gpu_tp_slab_alloc(slab_bytes);
    const int shared = h.slab != NULL;
    if (!shared) {
        h.slab = ds4_gpu_tensor_alloc(slab_bytes);
        h.staging = calloc(1, slab_bytes);
        CHECK(h.staging);
    }
    input = ds4_gpu_tensor_alloc(h.vec * 8);
    result = ds4_gpu_tensor_alloc(h.vec * 8 * 80);
    host = malloc(h.vec * 8 * 80);
    CHECK(h.slab && input && result && host);
    CHECK(ds4_gpu_tensor_fill_f32(h.slab, 0, slab_bytes / 4));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_tp_attach_slab(h.tp, shared ? ds4_gpu_tensor_contents(h.slab) : h.staging,
                            error, sizeof(error)));
    for (uint32_t i = 0; i < 80; i++) {
        out[i] = ds4_gpu_tensor_view(h.slab, ds4_tp_slab_out_offset(h.tp, i / 2, i % 2), h.vec);
        in[i] = ds4_gpu_tensor_view(h.slab, ds4_tp_slab_in_offset(h.tp, i / 2, i % 2), h.vec);
        CHECK(out[i] && in[i]);
    }
    for (uint32_t i = 0; i < 80; i++) {
        bout[i] = ds4_gpu_tensor_view(h.slab, ds4_tp_slab_batch_out_offset(h.tp, i), 8 * h.vec);
        bin[i] = ds4_gpu_tensor_view(h.slab, ds4_tp_slab_batch_in_offset(h.tp, i), 8 * h.vec);
        CHECK(bout[i] && bin[i]);
    }
    CHECK(ds4_gpu_tp_init(rank, h.slab, ds4_tp_slab_gpu_flags_offset(h.tp),
        ds4_tp_slab_out_offset(h.tp, 0, 0), h.vec, row_exchange, &h));
    ds4_gpu_tp_set_batch_exchange(batch_exchange);
    const uint32_t batches[] = {1, 2, 8, 3, 1};
    for (unsigned phase = 0; phase < sizeof(batches) / sizeof(*batches); phase++) {
        const uint32_t rows = batches[phase];
        const uint64_t bytes = rows * h.vec;
        double start = now();
        unsigned total_gates = 0;
        for (uint32_t epoch = 0; epoch < 40; epoch++) {
            uint32_t gates = epoch % 3 == 0 ? 2 : epoch % 3 == 1 ? 1u + rank : 1;
            if (rows > 1) {
                CHECK(ds4_tp_batch_block_begin_gates(h.tp, rows, 40, &gates));
                CHECK(gates == (epoch % 3 == 0 ? 2u : 1u));
            }
            const uint32_t slots = rows == 1 ? 80 : 40 * gates;
            total_gates += slots;
            for (uint32_t i = 0; i < slots; i++) {
                const float value = phase * 10000u + epoch * 100u + i;
                ds4_gpu_tensor *a = rows == 1 ? out[i] : bout[i];
                ds4_gpu_tensor *b = rows == 1 ? in[i] : bin[i];
                CHECK(ds4_gpu_tensor_fill_f32(input, value + rank * 1000u, rows * id.n_embd));
                CHECK(ds4_gpu_tensor_copy(a, 0, input, 0, bytes));
                CHECK(rows == 1 ? ds4_gpu_tp_gate_encode(i / 2, i % 2) :
                    ds4_gpu_tp_batch_gate_encode(i, rows));
                CHECK(ds4_gpu_add_tensor(input, rank ? b : a, rank ? a : b, rows * id.n_embd));
                CHECK(ds4_gpu_tensor_copy(result, i * bytes, input, 0, bytes));
            }
            CHECK(ds4_gpu_synchronize() && !ds4_gpu_tp_failed());
            if (rows > 1) CHECK(ds4_tp_batch_block_end(h.tp));
            CHECK(ds4_gpu_tensor_read(result, 0, host, slots * bytes));
            for (uint32_t i = 0; i < slots; i++) {
                const float expected = 2u * (phase * 10000u + epoch * 100u + i) + 1000u;
                for (uint32_t j = 0; j < rows * id.n_embd; j++) {
                    CHECK(host[i * rows * id.n_embd + j] == expected);
                }
            }
            if (!phase && !epoch && rank == 1 && getenv("DS4_TEST_TP_LINK_STALL")) {
                fprintf(stderr, "TP_LINK_STALL_READY\n"); fflush(stderr);
                raise(SIGSTOP);
            }
        }
        fprintf(stderr, "rank=%d shared=%d rows=%u GPU exchange/check %.2f us/gate: PASS\n",
            rank, shared, rows, (now() - start) * 1e6 / total_gates);
    }
    rc = 0;
done:
    ds4_gpu_tp_shutdown();
    if (h.tp) ds4_tp_detach_slab(h.tp);
    for (unsigned i = 0; i < 80; i++) { ds4_gpu_tensor_free(out[i]); ds4_gpu_tensor_free(in[i]); }
    for (unsigned i = 0; i < 80; i++) { ds4_gpu_tensor_free(bout[i]); ds4_gpu_tensor_free(bin[i]); }
    ds4_gpu_tensor_free(result); ds4_gpu_tensor_free(input); ds4_gpu_tensor_free(h.slab);
    ds4_tp_free(h.tp); free(h.staging); free(host);
    ds4_gpu_cleanup();
    return rc;
}
