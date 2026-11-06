/* Exercise RDMA completion accounting without a device or model. */
#include "../ds4_tp.c"
#include <assert.h>

#ifdef DS4_TP_HAVE_VERBS
static atomic_uint decode_barriers;

#ifdef __linux__
static int fake_gid(struct ibv_context *ctx, uint32_t port, uint32_t index,
                     struct ibv_gid_entry *entry, uint32_t flags, size_t size) {
    (void)ctx;
    assert(port == 1 && !flags && size == sizeof(*entry) && index < 5);
    memset(entry, 0, sizeof(*entry));
    entry->gid_type = index < 2 ? IBV_GID_TYPE_ROCE_V1 : IBV_GID_TYPE_ROCE_V2;
    if (index == 4) return 0;
    entry->gid.raw[10] = entry->gid.raw[11] = 0xff;
    entry->gid.raw[12] = 127;
    entry->gid.raw[15] = index == 2 ? 2 : 1;
    return 0;
}

static void check_gid(void) {
    ds4_tp tp = {0};
    tp.rdma.api.query_gid_ex = fake_gid;
    tp.control_fd = socket(AF_INET, SOCK_STREAM, 0);
    assert(tp.control_fd >= 0);
    struct sockaddr_in address = {.sin_family = AF_INET,
        .sin_addr = {.s_addr = htonl(INADDR_LOOPBACK)}};
    assert(bind(tp.control_fd, (struct sockaddr *)&address, sizeof(address)) == 0);
    struct ibv_port_attr port = {.gid_tbl_len = 5};
    union ibv_gid gid;
    int index = -1;
    assert(tp_rdma_linux_gid(&tp, NULL, &port, &gid, &index) == 2 && index == 3);
    tp.opt.rdma_gid_index_set = true;
    tp.opt.rdma_gid_index = 2;
    assert(tp_rdma_linux_gid(&tp, NULL, &port, &gid, &index) == 1 && index == 2);
    const int bad[] = {-1, 1, 4, 5, INT_MAX};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        tp.opt.rdma_gid_index = bad[i];
        assert(!tp_rdma_linux_gid(&tp, NULL, &port, &gid, &index));
    }
    close(tp.control_fd);
    puts("RoCEv2 GID selection, direct address and invalid indices: PASS");
}
#endif

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
    if (f->fault == 6) { errno = EIO; return EIO; }
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
    if (f->fault == 5) { errno = EIO; return EIO; }
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
    if (f->fault == 7) return -1;
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

static void check_rdma(uint32_t max_msg) {
    atomic_store(&decode_barriers, 0u);
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
    tp.rdma.max_msg = max_msg;
    const uint32_t chunks = (tp.vec_bytes + tp_rdma_max_msg(&tp) - 1u) / tp_rdma_max_msg(&tp);
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

    /* 5120-wide rows use two legacy messages or one negotiated RC message. */
    f = (fake_rdma){.expected = out, .expected_bytes = bytes, .poll_batch = 1,
        .required_decode_barriers = 1};
    tp.gates_per_token = 80; tp.gate_slot_step = 1;
    assert(tp_rdma_gate_exchange(&tp, 0, 0, 1));
    assert(atomic_load(&decode_barriers) == 1u);
    assert(tp.rdma.send_outstanding == f.ns - f.ds);
    while (tp.rdma.send_outstanding) assert(tp_rdma_drain_cq(&tp));
    assert(f.ns == chunks && f.ds == chunks && !f.unsignaled);

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
    assert(f.ns == 8u * chunks && f.ds == 8u * chunks && !f.unsignaled);

    for (unsigned rows = 2; rows <= 8; rows += 2) {
        f = (fake_rdma){.expected = out, .expected_bytes = bytes, .poll_batch = 1};
        uint32_t gates = 2;
        assert(ds4_tp_batch_block_begin_gates(&tp, rows, 3, &gates));
        assert(gates == 2);
        for (unsigned slot = 0; slot < 6; slot++)
            assert(ds4_tp_batch_gate_exchange(&tp, slot, rows, slot + 1));
        assert(ds4_tp_batch_block_end(&tp));
        assert(f.ns == 6u * rows * chunks && f.ds == f.ns && !f.unsignaled);
    }

    shutdown(fd[0], SHUT_RDWR);
    assert(pthread_join(echo, NULL) == 0);
    close(fd[0]); close(fd[1]);
    pthread_mutex_destroy(&tp.rdma.post_lock);
    free(tp.rdma.win_sge); free(tp.rdma.win_rwr); free(tp.rdma.win_swr);
    free(tp.slab); free(in); free(out);
    puts("RDMA per-message completion accounting, bulk/decode/drain/verify: PASS");
}

static void check_failed_rdma(uint32_t max_msg) {
    for (unsigned fault = 5; fault <= 9; fault++) {
        fake_rdma f = {.fault = fault == 8 ? 7 : fault};
        struct ibv_context ctx = {.ops = {.post_recv = fake_recv,
            .post_send = fake_send, .poll_cq = fake_poll}};
        struct ibv_qp qp = {.context = &ctx, .qp_context = &f};
        struct ibv_cq cq = {.context = &ctx, .cq_context = &f};
        struct ibv_mr mr = {.lkey = 1};
        ds4_tp tp = {.n_layer = 40, .n_slots = 80, .vec_bytes = 5120 * 4,
            .rdma_active = true, .gate_timeout_ms = 50, .control_fd = -1, .data_fd = -1};
        tp_slab_layout(&tp);
        tp.slab = calloc(1, tp.slab_bytes);
        assert(tp.slab);
        tp.rdma.qp = &qp; tp.rdma.cq = &cq; tp.rdma.mr = &mr;
        tp.rdma.max_msg = max_msg;
        tp.rdma.recv_depth = tp.rdma.send_depth = 1024;
        assert(pthread_mutex_init(&tp.rdma.post_lock, NULL) == 0);
        if (fault == 8) {
            tp.rdma.block_active = true;
            tp.rdma.block_layers = 1;
            tp.rdma.block_rows = 2;
            assert(!ds4_tp_batch_block_end(&tp));
        } else if (fault == 9) {
            /* A mismatched head row poisons only the leader. It must not
             * send a replay or enter another decode gate afterward. */
            tp.rdma.mbox_active = true;
            tp.rdma.begin_count = 1;
            const ds4_tp_head_row_header h = {4, 99, 1};
            memcpy(tp.slab + tp.head_row_in_off, &h, sizeof(h));
            memcpy(tp.slab + tp.head_row_in_off + DS4_TP_HEAD_ROW_SEQ_OFF,
                   &h.block, sizeof(h.block));
            float row[4] = {123, 123, 123, 123};
            assert(!ds4_tp_head_row_recv(&tp, 0, row, 4));
            for (unsigned i = 0; i < 4; i++) assert(row[i] == 123);
        } else {
            tp.rdma.recv_window_active = fault != 6;
            assert(!ds4_tp_batch_block_begin(&tp, 2, 1));
        }
        assert(ds4_tp_failed(&tp));
        const unsigned posted = f.ns + f.nr;
        assert(!ds4_tp_send_verify_commit(&tp, DS4_TP_VERIFY_ROLLBACK_REPLAY, 1));
        assert(!ds4_tp_send_eval(&tp, 1, 1, 7));
        assert(!ds4_tp_gate_exchange(&tp, 0, 0, 1));
        assert(!ds4_tp_batch_block_begin(&tp, 2, 1));
        assert(f.ns + f.nr == posted && ds4_tp_failed(&tp));
        pthread_mutex_destroy(&tp.rdma.post_lock);
        free(tp.rdma.win_sge); free(tp.rdma.win_rwr); free(tp.rdma.win_swr);
        free(tp.slab);
    }
    puts("RDMA drain/post/end and one-sided head failures remain fatal: PASS");
}
#endif

int main(void) {
#ifdef DS4_TP_HAVE_VERBS
#ifdef __linux__
    check_gid();
#endif
    assert(tp_rdma_negotiate_max_msg(0, 0) == 16384u);
    assert(tp_rdma_negotiate_max_msg(65536, 0) == 16384u);
    assert(tp_rdma_negotiate_max_msg(0, 65536) == 16384u);
    assert(tp_rdma_negotiate_max_msg(65536, 16384) == 16384u);
    assert(tp_rdma_negotiate_max_msg(65536, 32768) == 32768u);
    assert(tp_rdma_negotiate_max_msg(65536, 65536) == 65536u);
    assert(!tp_rdma_negotiate_max_msg(65536, 8192));
    assert(!tp_rdma_negotiate_max_msg(65536, 24576));
    check_rdma(0);
    check_rdma(32768);
    check_rdma(65536);
    check_failed_rdma(0);
    check_failed_rdma(32768);
    check_failed_rdma(65536);
#else
    puts("SKIP: RDMA completion tests require verbs headers");
#endif
    return 0;
}
