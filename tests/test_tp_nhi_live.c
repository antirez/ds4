/* Two-node live test for the NHI tensor-parallel exchange path (stage 2).
 *
 * Run one rank per machine over a shared thunderbolt-stream device with
 * the patch-14/15 zero-copy module loaded:
 *
 *   rank 0:  test_tp_nhi_live --rank 0 --device /run/ds4-tbstream/device \
 *              --listen 5599 [--iters 2000] [--hidden 7168]
 *   rank 1:  test_tp_nhi_live --rank 1 --device /run/ds4-tbstream/device \
 *              --connect 10.99.0.2 5599 [--iters 2000] [--hidden 7168]
 *
 * Each iteration mirrors one TP gate: launch the spin-combine on the
 * expected receive slot, fill and release the local partial into the
 * transmit slot, submit, and wait for the combine.  Startup runs an
 * out-of-band TCP barrier between both sides' enable and the first
 * submit (contract rule 13), and a second barrier before teardown so
 * the peer's CLOSE never races outstanding traffic.
 *
 * Verification is bit-exact: partials are a deterministic function of
 * (rank, sequence, index), so each rank computes the peer's contribution
 * on the CPU and compares the final accumulator byte for byte.  Timing
 * reports sustained microseconds per exchange for comparison with the
 * transport probe (35.2 us/exchange with reduce at this shape).
 */

#ifdef __linux__

#include "ds4_gpu.h"
#include "ds4_tp_nhi.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float partial_value(int rank, uint64_t seq, uint32_t i) {
    /* Deterministic, exactly representable steps so both sides agree. */
    return (float)((int)(((seq * 2654435761u) ^ (i * 40503u) ^
                          ((uint32_t)rank << 16)) & 0x3ffu) - 512) * 0.03125f;
}

static uint32_t role_stamp(int rank, uint64_t seq0) {
    return (rank == 0 ? 0x5ca16e39u : 0xc35a91e7u) ^ (uint32_t)seq0;
}

static int control_barrier(int fd, uint32_t tag) {
    uint32_t mine = htonl(tag), theirs = 0;
    if (write(fd, &mine, sizeof(mine)) != (ssize_t)sizeof(mine)) return 0;
    size_t got = 0;
    while (got < sizeof(theirs)) {
        ssize_t n = read(fd, (char *)&theirs + got, sizeof(theirs) - got);
        if (n <= 0) return 0;
        got += (size_t)n;
    }
    return ntohl(theirs) == tag;
}

static int control_listen(int port) {
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) return -1;
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(lfd, 1) != 0) {
        close(lfd);
        return -1;
    }
    int fd = accept(lfd, NULL, NULL);
    close(lfd);
    if (fd >= 0) {
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    return fd;
}

static int control_connect(const char *host, int port) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port);
        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
            close(fd);
            return -1;
        }
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            return fd;
        }
        close(fd);
        struct timespec pause = {0, 200 * 1000 * 1000};
        nanosleep(&pause, NULL);
    }
    return -1;
}

int main(int argc, char **argv) {
    const char *device = NULL, *peer_host = NULL;
    int rank = -1, listen_port = 0, peer_port = 0;
    uint32_t iters = 2000, hidden = 7168, ring_frames = 4096;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = argv[++i];
        else if (!strcmp(argv[i], "--listen") && i + 1 < argc) listen_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--connect") && i + 2 < argc) {
            peer_host = argv[++i];
            peer_port = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) hidden = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ring-frames") && i + 1 < argc) ring_frames = (uint32_t)atoi(argv[++i]);
        else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (rank != 0 && rank != 1) { fprintf(stderr, "--rank 0|1 required\n"); return 2; }
    if (!device) { fprintf(stderr, "--device required\n"); return 2; }
    if (rank == 0 && listen_port <= 0) { fprintf(stderr, "rank 0 needs --listen PORT\n"); return 2; }
    if (rank == 1 && (!peer_host || peer_port <= 0)) { fprintf(stderr, "rank 1 needs --connect HOST PORT\n"); return 2; }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "no ROCm device\n");
        return 1;
    }

    /* Control channel first: it also sequences NHI startup (rule 13). */
    int ctl = rank == 0 ? control_listen(listen_port)
                        : control_connect(peer_host, peer_port);
    if (ctl < 0) { fprintf(stderr, "control channel failed\n"); return 1; }

    char err[256] = "";
    ds4_tp_nhi *nhi = NULL;
    if (!ds4_tp_nhi_open(&nhi, device, ring_frames, err, sizeof(err))) {
        fprintf(stderr, "TP NHI open failed: %s\n", err);
        return 1;
    }
    fprintf(stderr, "rank %d: pools imported, device enabled; waiting for peer\n", rank);
    if (!control_barrier(ctl, 0x5450u)) { /* 'TP' */
        fprintf(stderr, "startup barrier failed\n");
        ds4_tp_nhi_close(nhi);
        return 1;
    }

    float *acc_init = malloc((size_t)hidden * sizeof(float));
    float *acc_ref = malloc((size_t)hidden * sizeof(float));
    float *acc_out = malloc((size_t)hidden * sizeof(float));
    float *src = malloc((size_t)hidden * sizeof(float));
    void *acc_dev = NULL;
    int pass = acc_init && acc_ref && acc_out && src;
    for (uint32_t i = 0; pass && i < hidden; i++) {
        acc_init[i] = partial_value(rank ^ 1, 0, i) * 0.5f;
        acc_ref[i] = acc_init[i];
    }
    if (pass) pass = ds4_gpu_tp_dev_buf_create(acc_init, hidden, &acc_dev);

    /* Pre-generate every iteration's partial and the final reference
     * OUTSIDE the timed loop: the harness's hash math and reference
     * accumulation are not part of the exchange path being measured.
     * (The per-iteration host upload of src stays inside: stage-3 removes
     * it entirely by producing partials on the GPU.) */
    float *src_all = malloc((size_t)iters * hidden * sizeof(float));
    if (!src_all) pass = 0;
    for (uint64_t seq = 1; pass && seq <= iters; seq++) {
        float *src_seq = src_all + (size_t)(seq - 1) * hidden;
        for (uint32_t i = 0; i < hidden; i++) {
            src_seq[i] = partial_value(rank, seq, i);
            acc_ref[i] += partial_value(rank ^ 1, seq, i);
        }
    }

    const double t0 = now_sec();
    uint64_t done = 0;
    for (uint64_t seq = 1; pass && seq <= iters; seq++) {
        const uint64_t seq0 = seq - 1;
        const uint32_t local_stamp = role_stamp(rank, seq0);
        const uint32_t peer_stamp = role_stamp(rank ^ 1, seq0);
        if (!ds4_tp_nhi_acquire_tx(nhi, seq0, err, sizeof(err))) {
            fprintf(stderr, "TX slot acquire failed at seq %llu: %s\n",
                    (unsigned long long)seq, err);
            pass = 0;
            break;
        }
        if (!ds4_gpu_tp_spin_combine_start(acc_dev,
                                           ds4_tp_nhi_rx_slot(nhi, seq0),
                                           hidden, peer_stamp, 800000000ull)) {
            fprintf(stderr, "spin start failed at seq %llu\n",
                    (unsigned long long)seq);
            pass = 0;
            break;
        }
        if (!ds4_gpu_tp_fill_release(ds4_tp_nhi_tx_slot(nhi, seq0),
                                     src_all + (size_t)seq0 * hidden,
                                     hidden, local_stamp) ||
            !ds4_tp_nhi_submit(nhi, seq0, err, sizeof(err))) {
            fprintf(stderr, "fill/submit failed at seq %llu: %s\n",
                    (unsigned long long)seq, err);
            pass = 0;
            break;
        }
        int timed_out = 1;
        if (!ds4_gpu_tp_spin_combine_wait(&timed_out) || timed_out) {
            fprintf(stderr, "combine %s at seq %llu\n",
                    timed_out ? "timed out" : "failed",
                    (unsigned long long)seq);
            pass = 0;
            break;
        }
        if (!ds4_tp_nhi_consumed(nhi, seq0, err, sizeof(err))) {
            fprintf(stderr, "repost failed at seq %llu: %s\n",
                    (unsigned long long)seq, err);
            pass = 0;
            break;
        }
        /* Ring bookkeeping off the critical cadence (contract rule 11). */
        if ((seq & 15u) == 0 &&
            (!ds4_tp_nhi_reap(nhi, err, sizeof(err)) ||
             ds4_tp_nhi_peer_closed(nhi))) {
            fprintf(stderr, "ring maintenance failed at seq %llu: %s\n",
                    (unsigned long long)seq, err);
            pass = 0;
            break;
        }
        done = seq;
    }
    const double t1 = now_sec();
    if (pass && !ds4_tp_nhi_reap(nhi, err, sizeof(err))) {
        fprintf(stderr, "final reap failed: %s\n", err);
        pass = 0;
    }
    free(src_all);

    if (pass) {
        pass = ds4_gpu_tp_dev_buf_read(acc_dev, acc_out, hidden) &&
               memcmp(acc_out, acc_ref, (size_t)hidden * sizeof(float)) == 0;
        if (!pass) fprintf(stderr, "accumulator mismatch after %llu exchanges\n",
                           (unsigned long long)done);
    }

    if (pass && !ds4_tp_nhi_quiesce(nhi, err, sizeof(err))) {
        fprintf(stderr, "quiesce failed: %s\n", err);
        pass = 0;
    }
    /* Teardown barrier: nobody closes while peer traffic may be in flight. */
    (void)control_barrier(ctl, 0x444eu); /* 'DN' */
    ds4_tp_nhi_close(nhi);
    close(ctl);
    ds4_gpu_tp_dev_buf_free(acc_dev);
    free(acc_init); free(acc_ref); free(acc_out); free(src);

    if (pass && done) {
        printf("test_tp_nhi_live rank %d: PASS  %llu exchanges of %u floats, "
               "%.1f us/exchange sustained (%.2f ms per 86-gate token)\n",
               rank, (unsigned long long)done, hidden,
               (t1 - t0) * 1e6 / (double)done,
               (t1 - t0) * 1e3 / (double)done * 86.0);
    } else {
        printf("test_tp_nhi_live rank %d: FAIL\n", rank);
    }
    return pass ? 0 : 1;
}

#else
#include <stdio.h>
int main(void) {
    fprintf(stderr, "test_tp_nhi_live is Linux-only\n");
    return 0;
}
#endif
