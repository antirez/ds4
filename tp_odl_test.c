/* tp_odl_test.c — transport-level A/B for the ds4 TP data plane, no model.
 *
 * The engine normally reaches ds4_tp_create() only after loading a GGUF
 * (the hello identity comes from the engine).  This tool feeds a synthetic
 * identity instead, so the whole transport path — hello negotiation, slab
 * attach barrier, gate exchanges — runs without any weights, which is all
 * that is needed to validate and benchmark a transport.
 *
 *   worker:   ./tp_odl_test worker  <leader-host> <port> <odl|tcp|rdma> [iters] [n_embd]
 *   leader:   ./tp_odl_test leader  <listen-host> <port> <odl|tcp|rdma> [iters] [n_embd]
 *
 * Both ranks exchange `iters` sequential gates with the DS4 identity gate
 * mapping (gates_per_token = 0), verify the peer's payload pattern after
 * every receive, and the leader prints per-gate round-trip stats.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4_tp.h"

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s leader|worker <host> <port> <odl|tcp|rdma> "
                "[iters=2000] [n_embd=6144]\n", argv[0]);
        return 2;
    }
    const int leader = !strcmp(argv[1], "leader");
    const char *host = argv[2];
    const int port = atoi(argv[3]);
    const char *transport = argv[4];
    const int iters = argc > 5 ? atoi(argv[5]) : 2000;
    const uint32_t n_embd = argc > 6 ? (uint32_t)atoi(argv[6]) : 6144;
    const uint32_t n_layer = 92;

    ds4_tp_options opt = {0};
    opt.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    if (leader) {
        opt.listen_host = host;
        opt.listen_port = port;
    } else {
        opt.leader_host = host;
        opt.leader_port = port;
    }
    if (!strcmp(transport, "odl")) opt.transport = DS4_TP_TRANSPORT_ODL;
    else if (!strcmp(transport, "tcp")) opt.transport = DS4_TP_TRANSPORT_TCP;
    else if (!strcmp(transport, "rdma")) opt.transport = DS4_TP_TRANSPORT_RDMA;
    else if (!strcmp(transport, "auto")) opt.transport = DS4_TP_TRANSPORT_AUTO;
    else { fprintf(stderr, "bad transport %s\n", transport); return 2; }

    /* DS4-style identity mapping: every slot fires in order. */
    ds4_tp_identity id = {0};
    id.gguf_bytes = 1;
    id.model_id = 1;
    id.n_layer = n_layer;
    id.n_embd = n_embd;
    id.n_vocab = 2;
    id.quant_bits = 4;
    id.ctx_size = 512;
    id.gate_slot_start = 0;
    id.gate_slot_step = 1;
    id.gates_per_token = 0;

    char err[256] = "";
    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        fprintf(stderr, "create failed: %s\n", err);
        return 1;
    }
    const uint64_t slab_bytes = ds4_tp_slab_bytes(n_layer, n_embd);
    uint8_t *slab = malloc(slab_bytes);
    if (!slab) { fprintf(stderr, "slab malloc failed\n"); return 1; }
    memset(slab, 0, slab_bytes);
    /* Fill every out vector with a per-slot pattern; the peer checks it. */
    const uint64_t vec = (uint64_t)n_embd * sizeof(float);
    for (uint32_t s = 0; s < n_layer * DS4_TP_GATES_PER_LAYER; s++) {
        float *out = (float *)(slab + ds4_tp_slab_out_offset(
                tp, s / DS4_TP_GATES_PER_LAYER, s % DS4_TP_GATES_PER_LAYER));
        for (uint32_t j = 0; j < n_embd; j++) out[j] = (float)(s * 1000u + j);
    }
    if (!ds4_tp_attach_slab(tp, slab, err, sizeof(err))) {
        fprintf(stderr, "attach failed: %s\n", err);
        return 1;
    }
    fprintf(stderr, "tp-odl-test: %s attached (%s transport), %u iters, "
            "vec=%llu bytes\n", leader ? "leader" : "worker", transport,
            iters, (unsigned long long)vec);

    double *rtt = leader ? malloc(sizeof(double) * iters) : NULL;
    int rc = 0;
    for (int i = 0; i < iters; i++) {
        /* seq s+1 must land in slot s under the identity mapping. */
        const uint64_t seq = (uint64_t)(i % (n_layer * DS4_TP_GATES_PER_LAYER)) + 1;
        const uint32_t slot = (uint32_t)(seq - 1);
        const uint32_t layer = slot / DS4_TP_GATES_PER_LAYER;
        const uint32_t gate = slot % DS4_TP_GATES_PER_LAYER;
        const double t0 = leader ? now_us() : 0.0;
        if (!ds4_tp_gate_exchange(tp, layer, gate, seq)) {
            fprintf(stderr, "gate %d exchange failed\n", i);
            rc = 1;
            break;
        }
        if (leader) rtt[i] = now_us() - t0;
        /* Verify the peer's pattern landed whole in the in vector. */
        const float *in = (const float *)(slab + ds4_tp_slab_in_offset(tp, layer, gate));
        for (uint32_t j = 0; j < n_embd; j++) {
            if (in[j] != (float)(slot * 1000u + j)) {
                fprintf(stderr, "payload corrupt: gate %d slot %u j %u: "
                        "%g != %u\n", i, slot, j, in[j], slot * 1000u + j);
                rc = 1;
                break;
            }
        }
        if (rc) break;
    }
    if (!rc && leader) {
        qsort(rtt, iters, sizeof(double), cmp_d);
        const double sum = 0;
        double mean = 0;
        for (int i = 0; i < iters; i++) mean += rtt[i];
        mean /= iters;
        (void)sum;
        printf("RESULT transport=%s vec=%lluB iters=%d "
               "min=%.1f p50=%.1f p90=%.1f p99=%.1f max=%.1f mean=%.1f us\n",
               transport, (unsigned long long)vec, iters,
               rtt[0], rtt[iters / 2], rtt[(int)(0.9 * (iters - 1))],
               rtt[(int)(0.99 * (iters - 1))], rtt[iters - 1], mean);
    }
    /* Verify-block batch gate (rows * vec through the slab batch region). */
    if (!rc) {
        const uint32_t rows = DS4_TP_BATCH_MAX_ROWS;
        const uint64_t batch_bytes = (uint64_t)rows * vec;
        float *bout = (float *)(slab + ds4_tp_slab_batch_out_offset(tp, 0));
        for (uint64_t j = 0; j < batch_bytes / sizeof(float); j++)
            bout[j] = (float)(j % 977u);
        const double t0 = leader ? now_us() : 0.0;
        if (!ds4_tp_batch_gate_exchange(tp, 0, rows, 1)) {
            fprintf(stderr, "batch exchange failed\n");
            rc = 1;
        } else if (leader) {
            printf("BATCH transport=%s rows=%u bytes=%llu rtt=%.1f us\n",
                   transport, rows, (unsigned long long)batch_bytes,
                   now_us() - t0);
        }
        const float *bin = (const float *)(slab +
                ds4_tp_slab_batch_in_offset(tp, 0));
        for (uint64_t j = 0; !rc && j < batch_bytes / sizeof(float); j++) {
            if (bin[j] != (float)(j % 977u)) {
                fprintf(stderr, "batch payload corrupt at %llu\n",
                        (unsigned long long)j);
                rc = 1;
            }
        }
    }
    /* Prefill-style big gate: 2 MiB symmetric swap through caller buffers
     * (outside the registered slab, which the odl path allows). */
    if (!rc) {
        const uint64_t big = 2ull * 1024 * 1024;
        uint8_t *bout = malloc(big), *bin = malloc(big);
        if (!bout || !bin) { fprintf(stderr, "big malloc failed\n"); rc = 1; }
        else {
            for (uint64_t j = 0; j < big; j++) bout[j] = (uint8_t)(j * 31 + 7);
            memset(bin, 0, big);
            const double t0 = leader ? now_us() : 0.0;
            if (!ds4_tp_big_gate_exchange(tp, 3, 9, bout, bin, big)) {
                fprintf(stderr, "big exchange failed\n");
                rc = 1;
            } else if (leader) {
                const double dt = now_us() - t0;
                printf("BIG transport=%s bytes=%llu rtt=%.1f us "
                       "(%.2f GB/s per direction)\n",
                       transport, (unsigned long long)big, dt, big / dt);
            }
            for (uint64_t j = 0; !rc && j < big; j++) {
                if (bin[j] != (uint8_t)(j * 31 + 7)) {
                    fprintf(stderr, "big payload corrupt at %llu\n",
                            (unsigned long long)j);
                    rc = 1;
                }
            }
            free(bout);
            free(bin);
        }
    }
    ds4_tp_send_stop(tp);
    ds4_tp_free(tp);
    free(slab);
    free(rtt);
    return rc;
}
