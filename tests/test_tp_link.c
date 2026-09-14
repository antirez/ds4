/* Physical two-host transport test; no model or GPU is needed. */
#include "ds4_tp.h"
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", \
    __FILE__, __LINE__, #x, error); goto done; } } while (0)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void fill(float *out, uint32_t n, unsigned rank, uint32_t epoch) {
    for (uint32_t i = 0; i < n; i++) out[i] = epoch + i % 127u + rank * 1000u;
}

static int equal(const float *in, uint32_t n, unsigned peer, uint32_t epoch) {
    for (uint32_t i = 0; i < n; i++) {
        if (in[i] != epoch + i % 127u + peer * 1000u) {
            fprintf(stderr, "payload mismatch epoch=%u index=%u value=%g\n", epoch, i, in[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 7) {
        fprintf(stderr, "usage: %s RANK COORDINATOR PORT tcp|rdma [DEVICE GID]\n", argv[0]);
        return 2;
    }
    const int rank = atoi(argv[1]), port = atoi(argv[3]);
    if ((rank != 0 && rank != 1) || port <= 0 || port > 65535 ||
        (strcmp(argv[4], "rdma") && strcmp(argv[4], "tcp"))) return 2;
    ds4_tp_options opt = {.role = rank ? DS4_TP_WORKER : DS4_TP_LEADER,
        .listen_host = argv[2], .leader_host = argv[2],
        .listen_port = port, .leader_port = port,
        .transport = !strcmp(argv[4], "rdma") ? DS4_TP_TRANSPORT_RDMA : DS4_TP_TRANSPORT_TCP};
    if (argc == 7) {
        opt.rdma_device = argv[5];
        opt.rdma_gid_index = atoi(argv[6]);
        opt.rdma_gid_index_set = true;
    }
    ds4_tp_identity id = {.gguf_bytes = 1, .n_layer = 40, .n_embd = 5120,
        .n_vocab = 16, .ctx_size = 8192, .gate_slot_step = 1, .gates_per_token = 80};
    char error[256] = "";
    ds4_tp *tp = NULL;
    void *slab = NULL;
    float *out = NULL, *in = NULL;
    const uint32_t width = id.n_embd, capacity = 8192u * width;
    const uint64_t vec = (uint64_t)width * sizeof(float);
    int rc = 1;
    CHECK(ds4_tp_create(&tp, &opt, &id, error, sizeof(error)));
    CHECK(ds4_tp_is_rdma(tp) == (opt.transport == DS4_TP_TRANSPORT_RDMA));
    slab = calloc(1, ds4_tp_slab_bytes(id.n_layer, width));
    out = malloc((size_t)capacity * sizeof(float));
    in = malloc((size_t)capacity * sizeof(float));
    CHECK(slab && out && in && ds4_tp_attach_slab(tp, slab, error, sizeof(error)));
    uint64_t seq = 0;
    const uint32_t rows[] = {7, 512, 2049, 8192};
    for (unsigned phase = 0; phase < sizeof(rows) / sizeof(*rows); phase++) {
        if (rank == (int)(phase % 2u)) usleep(100000);
        double start = now();
        for (unsigned step = 0; step < 160; step++) {
            const unsigned slot = step % 80u, layer = slot / 2u, gate = slot % 2u;
            const uint32_t epoch = phase * 10000u + step;
            float *a = (float *)((char *)slab + ds4_tp_slab_out_offset(tp, layer, gate));
            const float *b = (const float *)((char *)slab + ds4_tp_slab_in_offset(tp, layer, gate));
            fill(a, width, (unsigned)rank, epoch);
            CHECK(ds4_tp_gate_exchange(tp, layer, gate, ++seq));
            CHECK(equal(b, width, 1u - (unsigned)rank, epoch));
            if (!phase && !step && getenv("DS4_TEST_TP_LINK_STALL") && rank == 1) {
                fprintf(stderr, "TP_LINK_STALL_READY\n");
                fflush(stderr);
                raise(SIGSTOP);
            }
        }
        fprintf(stderr, "rank=%d phase=%u 20KiB decode exchange %.2f us: PASS\n",
            rank, phase, (now() - start) * 1e6 / 160);
        const unsigned small_rows = phase * 2u + 1u;
        CHECK(ds4_tp_batch_block_begin(tp, small_rows, 40));
        for (unsigned layer = 0; layer < 40; layer++) {
            const uint32_t epoch = phase * 10000u + layer + 200u;
            float *a = (float *)((char *)slab + ds4_tp_slab_batch_out_offset(tp, layer));
            const float *b = (const float *)((char *)slab + ds4_tp_slab_batch_in_offset(tp, layer));
            fill(a, small_rows * width, (unsigned)rank, epoch);
            CHECK(ds4_tp_batch_gate_exchange(tp, layer, small_rows, phase * 40u + layer));
            CHECK(equal(b, small_rows * width, 1u - (unsigned)rank, epoch));
        }
        CHECK(ds4_tp_batch_block_end(tp));
        const uint32_t count = rows[phase] * width;
        fill(out, count, (unsigned)rank, phase);
        start = now();
        for (unsigned round = 0; round < 8; round++) {
            memset(in, 0, count * sizeof(float));
            CHECK(ds4_tp_big_gate_exchange(tp, round, phase * 8u + round, out, in, rows[phase] * vec));
            CHECK(equal(in, count, 1u - (unsigned)rank, phase));
        }
        fprintf(stderr, "rank=%d phase=%u rows=%u bidirectional bulk %.2f GiB/s: PASS\n",
            rank, phase, rows[phase], 16.0 * rows[phase] * vec / ((now() - start) * 1073741824.0));
    }
    rc = 0;
done:
    ds4_tp_free(tp);
    free(in); free(out); free(slab);
    return rc;
}
