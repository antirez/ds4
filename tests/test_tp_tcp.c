/* TCP gates must not depend on the requested socket-buffer size. */
#include "../ds4_tp.c"
#include <assert.h>

typedef struct {
    ds4_tp tp;
    unsigned rank;
    bool nonblocking;
    int result;
} tcp_peer;

static void pattern(void *buffer, size_t bytes, unsigned rank) {
    unsigned char *p = buffer;
    for (size_t i = 0; i < bytes; i++) p[i] = (unsigned char)(i * 37 + rank);
}

static void check_pattern(const void *buffer, size_t bytes, unsigned rank) {
    const unsigned char *p = buffer;
    for (size_t i = 0; i < bytes; i++) assert(p[i] == (unsigned char)(i * 37 + rank));
}

static void *exchange(void *arg) {
    tcp_peer *p = arg;
    ds4_tp *tp = &p->tp;
    if (p->rank) usleep(20000);
    for (unsigned step = 0; step < 160; step++) {
        const unsigned layer = (step % 80) / 2, gate = step % 2;
        pattern(tp->slab + ds4_tp_slab_out_offset(tp, layer, gate), tp->vec_bytes, p->rank);
        if (!ds4_tp_gate_exchange(tp, layer, gate, step + 1)) return NULL;
        check_pattern(tp->slab + ds4_tp_slab_in_offset(tp, layer, gate), tp->vec_bytes, 1 - p->rank);
    }
    for (unsigned rows = 1; rows <= DS4_TP_BATCH_MAX_ROWS; rows++) {
        const size_t bytes = tp->vec_bytes * rows;
        pattern(tp->slab + ds4_tp_slab_batch_out_offset(tp, 0), bytes, p->rank);
        if (!ds4_tp_batch_gate_exchange(tp, 0, rows, rows)) return NULL;
        check_pattern(tp->slab + ds4_tp_slab_batch_in_offset(tp, 0), bytes, 1 - p->rank);
    }
    const size_t capacity = 16 * 1024 * 1024 + 13;
    void *out = malloc(capacity), *in = malloc(capacity);
    assert(out && in);
    const size_t lengths[] = {1, 7, 16385, 2 * 1024 * 1024, capacity};
    p->result = 1;
    for (unsigned i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        pattern(out, lengths[i], p->rank);
        /* Bulk framing requires blocking I/O. An already-nonblocking socket
         * only exercises the exchange helper's descriptor-mode preservation. */
        const int ok = p->nonblocking ?
            tp_tcp_exchange(tp, NULL, NULL, out, in, lengths[i]) :
            ds4_tp_big_gate_exchange(tp, 0, i, out, in, lengths[i]);
        if (!ok) {
            p->result = 0;
            break;
        }
        check_pattern(in, lengths[i], 1 - p->rank);
    }
    free(in); free(out);
    return NULL;
}

static void pair(int fd[2], bool tcp) {
    if (tcp) {
        int listener = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in addr = {.sin_family = AF_INET,
            .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
        socklen_t len = sizeof(addr);
        assert(listener >= 0 && bind(listener, (struct sockaddr *)&addr, len) == 0);
        assert(getsockname(listener, (struct sockaddr *)&addr, &len) == 0);
        assert(listen(listener, 1) == 0);
        fd[0] = socket(AF_INET, SOCK_STREAM, 0);
        assert(fd[0] >= 0 && connect(fd[0], (struct sockaddr *)&addr, len) == 0);
        fd[1] = accept(listener, NULL, NULL);
        assert(fd[1] >= 0);
        close(listener);
    } else {
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    }
    for (unsigned i = 0; i < 2; i++) {
        int small = 1024;
        assert(setsockopt(fd[i], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small)) == 0);
        assert(setsockopt(fd[i], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small)) == 0);
#ifdef SO_NOSIGPIPE
        int one = 1;
        assert(setsockopt(fd[i], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) == 0);
#endif
        assert(tp_socket_set_gate_timeout(fd[i], 1000));
        if (tcp) {
            int one = 1;
            assert(setsockopt(fd[i], IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0);
        }
    }
}

static void check_stall_and_close(bool tcp) {
    int fd[2];
    pair(fd, tcp);
    ds4_tp tp = {.data_fd = fd[0], .gate_timeout_ms = 50};
    const int flags = fcntl(fd[0], F_GETFL);
    assert(flags >= 0 && !(flags & O_NONBLOCK));
    char out[32768] = {0}, in[32768];
    double start = tp_now_sec();
    assert(!tp_tcp_exchange(&tp, NULL, NULL, out, in, sizeof(out)));
    assert(errno == ETIMEDOUT && tp_now_sec() - start < 1.0);
    assert(fcntl(fd[0], F_GETFL) == flags);
    close(fd[0]); close(fd[1]);
    pair(fd, tcp);
    tp.data_fd = fd[0];
    assert(shutdown(fd[1], SHUT_WR) == 0);
    assert(!tp_tcp_exchange(&tp, NULL, NULL, out, in, sizeof(out)));
    assert(errno == ECONNRESET);
    assert(fcntl(fd[0], F_GETFL) == flags);
    close(fd[1]);
    assert(!tp_tcp_exchange(&tp, NULL, NULL, out, in, sizeof(out)));
    close(fd[0]);
    puts("TCP stalled, half-closed and disconnected peers: PASS");
}

static void check_exchange(bool tcp, bool nonblocking) {
    int fd[2];
    pair(fd, tcp);
    int flags[2];
    tcp_peer peer[2] = {0};
    pthread_t threads[2];
    for (unsigned i = 0; i < 2; i++) {
        flags[i] = fcntl(fd[i], F_GETFL);
        assert(flags[i] >= 0);
        if (nonblocking) {
            flags[i] |= O_NONBLOCK;
            assert(fcntl(fd[i], F_SETFL, flags[i]) == 0);
        }
        peer[i].rank = i;
        peer[i].nonblocking = nonblocking;
        peer[i].tp = (ds4_tp){.data_fd = fd[i], .n_layer = 40, .n_slots = 80,
            .n_embd = 5120, .vec_bytes = 5120 * 4, .gate_timeout_ms = 1000};
        tp_slab_layout(&peer[i].tp);
        peer[i].tp.slab = calloc(1, peer[i].tp.slab_bytes);
        assert(peer[i].tp.slab);
        assert(pthread_create(&threads[i], NULL, exchange, &peer[i]) == 0);
    }
    for (unsigned i = 0; i < 2; i++) {
        assert(pthread_join(threads[i], NULL) == 0 && peer[i].result);
        assert(fcntl(fd[i], F_GETFL) == flags[i]);
        free(peer[i].tp.slab);
        close(fd[i]);
    }
    printf("%s tiny-buffer decode, 1..8-row gates and bulk payloads, nonblocking=%d: PASS\n",
           tcp ? "TCP" : "Unix", nonblocking);
}

int main(void) {
    for (int tcp = 0; tcp < 2; tcp++) {
        check_exchange(tcp, false);
        check_exchange(tcp, true);
        check_stall_and_close(tcp);
    }
    return 0;
}
