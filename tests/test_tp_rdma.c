/* Exercise RDMA completion accounting without a device or model. */
#include "../ds4_tp.c"
#include <assert.h>

#ifdef DS4_TP_HAVE_VERBS
static atomic_uint decode_barriers;

typedef struct {
    struct ibv_wc recv[1024], send[1024];
    unsigned nr, dr, ns, ds, poll_batch, unsignaled;
    const uint8_t *expected;
    uint64_t expected_bytes, offset;
    unsigned fault;
    unsigned required_decode_barriers;
    bool injected;
} fake_rdma;

static int fake_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                      struct ibv_recv_wr **bad) {
    fake_rdma *f = qp->qp_context;
    (void)bad;
    if (f->dr == f->nr && f->ds == f->ns)
        f->nr = f->dr = f->ns = f->ds = 0;
    for (; wr; wr = wr->next) {
        assert(wr->num_sge == 1 && f->nr < 1024);
        const struct ibv_sge *s = wr->sg_list;
        assert(s->length <= f->expected_bytes - f->offset);
        memcpy((void *)(uintptr_t)s->addr, f->expected + f->offset, s->length);
        f->offset += s->length;
        f->recv[f->nr++] = (struct ibv_wc){.wr_id = wr->wr_id,
            .status = IBV_WC_SUCCESS, .opcode = IBV_WC_RECV,
            .byte_len = s->length};
    }
    return 0;
}

static int fake_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                      struct ibv_send_wr **bad) {
    fake_rdma *f = qp->qp_context;
    (void)bad;
    assert(atomic_load(&decode_barriers) >= f->required_decode_barriers);
    for (; wr; wr = wr->next) {
        assert(wr->num_sge == 1 && f->ns < 1024);
        if (!(wr->send_flags & IBV_SEND_SIGNALED)) f->unsignaled++;
        /* Apple reports these even when IBV_SEND_SIGNALED is absent. */
        f->send[f->ns++] = (struct ibv_wc){.wr_id = wr->wr_id,
            .status = IBV_WC_SUCCESS, .opcode = IBV_WC_SEND};
    }
    return 0;
}

static int fake_poll(struct ibv_cq *cq, int cap, struct ibv_wc *wc) {
    fake_rdma *f = cq->cq_context;
    unsigned n = 0;
    if ((unsigned)cap > f->poll_batch) cap = (int)f->poll_batch;
    /* Receiving the peer's payload does not complete our outstanding sends. */
    while (n < (unsigned)cap && (f->dr < f->nr || f->ds < f->ns)) {
        if (f->dr < f->nr) {
            wc[n] = f->recv[f->dr++];
            if (f->fault && f->fault <= 3 && !f->injected) {
                f->injected = true;
                if (f->fault == 1) f->dr--; /* duplicate completion */
                if (f->fault == 2) wc[n].byte_len++;
                if (f->fault == 3) wc[n].status = IBV_WC_GENERAL_ERR;
            }
        } else {
            wc[n] = f->send[f->ds++];
            if (f->fault == 4 && !f->injected) {
                f->injected = true;
                wc[n].wr_id = DS4_TP_RDMA_BULK_WR_TAG | 999u;
            }
        }
        n++;
    }
    return (int)n;
}

static void *echo_barriers(void *arg) {
    int fd = *(int *)arg;
    uint8_t tag;
    while (read(fd, &tag, 1) == 1) {
        if (tag == 0xD1u) atomic_fetch_add(&decode_barriers, 1u);
        assert(write(fd, &tag, 1) == 1);
    }
    return NULL;
}

static void check_rdma(void) {
    const uint64_t bytes = 3u * 4u * 1024u * 1024u + 4u;
    uint8_t *out = malloc(bytes), *in = malloc(bytes);
    assert(out && in);
    for (uint64_t i = 0; i < bytes; i++) out[i] = (uint8_t)(i * 17u + (i >> 13));
    fake_rdma f = {.expected = out, .expected_bytes = bytes};
    struct ibv_context ctx = {.ops = {.post_recv = fake_recv,
        .post_send = fake_send, .poll_cq = fake_poll}};
    struct ibv_qp qp = {.context = &ctx, .qp_context = &f};
    struct ibv_cq cq = {.context = &ctx, .cq_context = &f};
    struct ibv_mr mr = {.lkey = 1};
    ds4_tp tp = {.n_layer = 40, .n_slots = 80, .vec_bytes = 5120 * 4,
        .gate_timeout_ms = 1000};
    tp_slab_layout(&tp);
    tp.slab = calloc(1, tp.slab_bytes);
    assert(tp.slab);
    tp.rdma.qp = &qp; tp.rdma.cq = &cq; tp.rdma.mr = &mr;
    tp.rdma.recv_depth = tp.rdma.send_depth = 1024;
    assert(pthread_mutex_init(&tp.rdma.post_lock, NULL) == 0);
    int fd[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    tp.control_fd = tp.data_fd = fd[0];
    assert(tp_socket_set_gate_timeout(fd[0], 1000));
    pthread_t echo;
    assert(pthread_create(&echo, NULL, echo_barriers, &fd[1]) == 0);

    const unsigned polls[] = {1, 7, 64};
    for (unsigned p = 0; p < 3; p++) {
        for (unsigned depth = 64; depth <= 1024; depth *= 4) {
            f = (fake_rdma){.expected = out, .expected_bytes = bytes,
                .poll_batch = polls[p]};
            tp.rdma.send_depth = depth;
            memset(in, 0, bytes);
            assert(tp_rdma_big_gate_exchange(&tp, out, in, bytes));
            assert(f.dr == f.nr && f.ds == f.ns && !f.unsignaled);
            assert(!memcmp(out, in, bytes));
        }
    }
    for (unsigned fault = 1; fault <= 4; fault++) {
        f = (fake_rdma){.expected = out, .expected_bytes = bytes,
            .poll_batch = 7, .fault = fault};
        assert(!tp_rdma_big_gate_exchange(&tp, out, in, bytes));
    }

    /* Decode has two messages per gate at hidden width 5120. */
    f = (fake_rdma){.expected = out, .expected_bytes = bytes, .poll_batch = 1,
        .required_decode_barriers = 1};
    tp.gates_per_token = 80; tp.gate_slot_step = 1;
    assert(tp_rdma_gate_exchange(&tp, 0, 0, 1));
    assert(atomic_load(&decode_barriers) == 1u);
    assert(tp.rdma.send_outstanding == f.ns - f.ds);
    while (tp.rdma.send_outstanding) assert(tp_rdma_drain_cq(&tp));
    assert(f.ns == 2 && f.ds == 2 && !f.unsignaled);

    /* Drain the posted decode receives before using the slab for bulk. */
    f = (fake_rdma){.expected = out, .expected_bytes = bytes, .poll_batch = 1};
    tp.rdma.last_gate_seq = 1;
    for (uint64_t seq = 2; seq < 2 + DS4_TP_RDMA_RECV_WINDOW; seq++)
        assert(tp_rdma_post_gate_recv(&tp, seq));
    assert(tp_rdma_drain_decode_window(&tp));
    assert(!tp.rdma.recv_window_active && f.ds == f.ns && !f.unsignaled);

    /* Speculative blocks count each chunk, not each linked send list. */
    f = (fake_rdma){.expected = out, .expected_bytes = bytes, .poll_batch = 1};
    free(tp.rdma.win_sge); free(tp.rdma.win_rwr); free(tp.rdma.win_swr);
    tp.rdma.win_sge = NULL; tp.rdma.win_rwr = NULL; tp.rdma.win_swr = NULL;
    tp.rdma_active = true;
    /* A short prompt can reach verification without any previous bulk gate. */
    assert(ds4_tp_batch_block_begin(&tp, 8, 1));
    assert(tp_rdma_block_gate_exchange(&tp, 0, 8));
    assert(tp.rdma.send_outstanding == f.ns - f.ds);
    assert(ds4_tp_batch_block_end(&tp));
    assert(f.ns == 16 && f.ds == 16 && !f.unsignaled);

    shutdown(fd[0], SHUT_RDWR);
    assert(pthread_join(echo, NULL) == 0);
    close(fd[0]); close(fd[1]);
    pthread_mutex_destroy(&tp.rdma.post_lock);
    free(tp.rdma.win_sge); free(tp.rdma.win_rwr); free(tp.rdma.win_swr);
    free(tp.slab); free(in); free(out);
    puts("RDMA per-message completion accounting, bulk/decode/drain/verify: PASS");
}
#endif

int main(void) {
#ifdef DS4_TP_HAVE_VERBS
    check_rdma();
#else
    puts("SKIP: RDMA completion tests require the Apple verbs headers");
#endif
    return 0;
}
