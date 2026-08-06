/* =========================================================================
 * ds4_distributed_rdma.c - RDMA-over-Thunderbolt data plane (Apple only).
 * =========================================================================
 *
 * Isolates every libibverbs dependency for the distributed engine: the rest of
 * the code drives the protocol through ds4_distributed_rdma.h / ds4_dist_dchan.h
 * and never sees verbs, so the TCP path is unaffected.
 *
 * The provider (librdma) is loaded at runtime via ds4_rdma_verbs.h - no
 * link-time dependency and no build flag. On a machine without it the entry
 * points report "unavailable" and the engine stays on TCP. Compiled only where
 * <infiniband/verbs.h> exists (DS4_RDMA_HAVE_VERBS); elsewhere this file is
 * stubs.
 *
 * Apple's provider is UC-only with two-sided SEND/RECV and no RDMA-CM, so
 * endpoints (GID/QPN/PSN) are exchanged out of band over the TCP bootstrap and
 * MTU/GID are discovered at runtime. See Apple TN3205 for the transport model.
 */

#include "ds4_distributed_rdma.h"
#include "ds4_dist_dchan.h"
#include "ds4_rdma_verbs.h"

#include <stdio.h>
#include <string.h>

#ifdef DS4_RDMA_HAVE_VERBS

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* librdma resolved once; NULL if the provider is absent (fall back to TCP). The
 * setup verbs go through this table; ibv_post_send/post_recv/poll_cq are header
 * inlines over context->ops and are still called directly. */
static ds4_rdma_verbs_api g_verbs;
static int g_verbs_ok;
static pthread_once_t g_verbs_once = PTHREAD_ONCE_INIT;
static void dist_rdma_verbs_init(void) { g_verbs_ok = ds4_rdma_verbs_load(&g_verbs); }
static const ds4_rdma_verbs_api *dist_rdma_verbs(void) {
    pthread_once(&g_verbs_once, dist_rdma_verbs_init);
    return g_verbs_ok ? &g_verbs : NULL;
}

/* Initial packet sequence number for UC QPs. UC has no hardware retransmit, so
 * the value is not load-bearing; both ends just agree (relayed in the endpoint).
 * jaccl uses a small constant; so do we. */
#define DS4_DIST_RDMA_PSN 0u

/* =========================================================================
 * Ring geometry
 * =========================================================================
 *
 * Per Apple TN3205 the UC provider needs send size == recv size and accounts
 * queues in 4 KiB FRAMES, not work requests, up to 4095. A segment is therefore
 * a fixed STRIDE (32 frames): every SEND transfers a full STRIDE and the header
 * `len` gives the valid payload (the rest is padding the receiver drops). Queues
 * are sized to hold the whole ring at once - sizing by WR count instead would
 * leave room for one 128 KiB send in flight and serialize the link. Ring depth
 * (2 MiB/direction) covers the bandwidth-delay product with a prefill chunk in
 * flight; it is not raised because every ring byte is IOMMU-mapped for the life
 * of the channel. */
#define DIST_RDMA_SEG_MAGIC     0x52534547u  /* "RSEG" */
#define DIST_RDMA_NBUF          16                /* data ring depth, per direction */
#define DIST_RDMA_STRIDE        (128u * 1024u)    /* 32 x 4 KiB frames              */
#define DIST_RDMA_FRAME         4096u             /* provider queue accounting unit */
#define DIST_RDMA_QUEUE_FRAMES  ((DIST_RDMA_NBUF * DIST_RDMA_STRIDE) / DIST_RDMA_FRAME)
#define DIST_RDMA_IDLE_FRAMES   32u               /* the direction a half never uses */
#define DIST_RDMA_RECV_WR_FLAG  (1ull << 20)

/* TN3205: at most 4095 frames per queue. Keep the ring inside that budget. */
typedef char dist_rdma_queue_frames_fit[(DIST_RDMA_QUEUE_FRAMES <= 4095) ? 1 : -1];

typedef struct {
    uint32_t magic;
    uint32_t len;     /* valid payload bytes in this segment */
    uint32_t seq;     /* monotonic per direction; catches a dropped UC frame */
    uint32_t reserved;
} dist_rdma_seg_hdr;

#define DIST_RDMA_PAYLOAD (DIST_RDMA_STRIDE - sizeof(dist_rdma_seg_hdr))

/* One direction of the connection: a UC QP plus everything it owns (device
 * context, PD, CQ, QP, ring), so the two directions share no verbs object.
 *
 * The ring is registered while the QP is still in INIT: Apple's provider maps a
 * QP's MR set once, on the RTR transition, so an MR registered after RTR is
 * never mapped (the source of map-saturation and teardown-drop failures). The
 * per-half PD keeps each QP's mapped set down to the one ring it uses. */
typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    uint8_t port_num;
    int gid_index;
    enum ibv_mtu path_mtu;
    void *data_mem;             /* NBUF x STRIDE: sends on tx, recvs on rx */
    struct ibv_mr *data_mr;
} dist_rdma_half;

struct ds4_dist_rdma_conn {
    dist_rdma_half tx;
    dist_rdma_half rx;
};

int ds4_dist_rdma_available(void) {
    const ds4_rdma_verbs_api *V = dist_rdma_verbs();
    if (!V) return 0;
    int n = 0;
    struct ibv_device **list = V->get_device_list(&n);
    if (!list) return 0;
    V->free_device_list(list);
    return n > 0 ? 1 : 0;
}

int ds4_dist_rdma_describe(char *buf, size_t buflen) {
    const ds4_rdma_verbs_api *V = dist_rdma_verbs();
    if (!V) {
        if (buflen) snprintf(buf, buflen, "no RDMA provider (librdma) available");
        return 0;
    }
    int n = 0;
    struct ibv_device **list = V->get_device_list(&n);
    if (!list || n <= 0) {
        if (list) V->free_device_list(list);
        if (buflen) snprintf(buf, buflen, "no RDMA devices found");
        return 0;
    }
    if (buflen) {
        int w = snprintf(buf, buflen, "%d RDMA device(s):", n);
        size_t used = (w > 0) ? (size_t)w : 0;
        if (used >= buflen) used = buflen - 1;
        for (int i = 0; i < n && used + 1 < buflen; i++) {
            const char *name = V->get_device_name(list[i]);
            int w2 = snprintf(buf + used, buflen - used, " %s", name ? name : "?");
            if (w2 <= 0) break;
            used += (size_t)w2;
            if (used >= buflen) break;
        }
    }
    V->free_device_list(list);
    return n;
}

/* Allocate and register a half's ring while its QP is still in INIT (see the
 * dist_rdma_half comment for why the timing matters). */
static int dist_rdma_half_alloc_ring(dist_rdma_half *h, char *err, size_t errlen) {
    const ds4_rdma_verbs_api *V = dist_rdma_verbs();
    if (!V) return -1;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    const size_t data_bytes = (size_t)DIST_RDMA_NBUF * DIST_RDMA_STRIDE;
    if (posix_memalign(&h->data_mem, (size_t)page, data_bytes) != 0) h->data_mem = NULL;
    if (!h->data_mem) {
        if (errlen) snprintf(err, errlen, "RDMA bounce-buffer alloc failed (%zu bytes)", data_bytes);
        return -1;
    }
    /* Apple's provider requires the remote access flags on the MR even for
     * two-sided SEND/RECV (matches jaccl); LOCAL_WRITE alone is rejected. */
    const int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    h->data_mr = V->reg_mr(h->pd, h->data_mem, data_bytes, mr_flags);
    if (!h->data_mr) {
        if (errlen) snprintf(err, errlen, "ibv_reg_mr(data, %zu) failed: %s", data_bytes, strerror(errno));
        return -1;
    }
    return 0;
}

/* Release one half. Order is load-bearing: destroy the QP first (nothing then
 * references the MRs), deregister the MRs while their PD is alive, then PD/CQ/
 * context. ibv_dereg_mr clears the MR's DART PTEs synchronously with no drain,
 * so this is the forward-compatible order. Safe on a partially built half. */
static void dist_rdma_half_release(dist_rdma_half *h) {
    const ds4_rdma_verbs_api *V = dist_rdma_verbs();
    if (!V) return;
    if (h->qp) { V->destroy_qp(h->qp); h->qp = NULL; }
    if (h->data_mr) { V->dereg_mr(h->data_mr); h->data_mr = NULL; }
    free(h->data_mem);
    h->data_mem = NULL;
    if (h->cq) { V->destroy_cq(h->cq); h->cq = NULL; }
    if (h->pd) { V->dealloc_pd(h->pd); h->pd = NULL; }
    if (h->ctx) { V->close_device(h->ctx); h->ctx = NULL; }
}

/* Bring one half from nothing to INIT with its ring registered, and report the
 * endpoint the peer needs. Each half opens the device separately so the two
 * directions share no verbs object at all. is_tx selects which queue is sized
 * for real traffic. */
static int dist_rdma_half_create(dist_rdma_half *h,
                                 const char *dev_name,
                                 int is_tx,
                                 ds4_dist_rdma_endpoint *ep_out,
                                 char *err,
                                 size_t errlen) {
    const ds4_rdma_verbs_api *V = dist_rdma_verbs();
    if (!V) {
        if (errlen) snprintf(err, errlen, "no RDMA provider (librdma) available");
        return -1;
    }
    memset(h, 0, sizeof(*h));

    struct ibv_port_attr pa;
    union ibv_gid gid;
    char resolved_name[32];
    if (ds4_rdma_open_ipv4_link(V, dev_name, -1, &h->ctx, &pa, &gid, &h->gid_index,
                                resolved_name, sizeof(resolved_name), err, errlen) != 0) {
        return -1;
    }
    h->port_num = 1;              /* one port per Thunderbolt device */
    h->path_mtu = pa.active_mtu;  /* reported to the peer; RTR uses IBV_MTU_1024 */

    h->pd = V->alloc_pd(h->ctx);
    if (!h->pd) {
        if (errlen) snprintf(err, errlen, "ibv_alloc_pd failed");
        goto fail;
    }
    h->cq = V->create_cq(h->ctx, DIST_RDMA_QUEUE_FRAMES + 1, NULL, NULL, 0);
    if (!h->cq) {
        if (errlen) snprintf(err, errlen, "ibv_create_cq failed");
        goto fail;
    }

    /* Depths are in 4 KiB frames (see the ring geometry comment). Only the
     * direction this half actually drives is sized for the whole ring; the other
     * is left nominal. That matters because the provider's outstanding-frame
     * budget is per PORT, not per QP or per context - two halves both claiming a
     * full ring on the same cable would compete for it. */
    struct ibv_qp_init_attr ia;
    memset(&ia, 0, sizeof(ia));
    ia.send_cq = h->cq;
    ia.recv_cq = h->cq;
    ia.cap.max_send_wr = is_tx ? DIST_RDMA_QUEUE_FRAMES : DIST_RDMA_IDLE_FRAMES;
    ia.cap.max_recv_wr = is_tx ? DIST_RDMA_IDLE_FRAMES : DIST_RDMA_QUEUE_FRAMES;
    ia.cap.max_send_sge = 1;
    ia.cap.max_recv_sge = 1;
    ia.qp_type = IBV_QPT_UC;
    h->qp = V->create_qp(h->pd, &ia);
    if (!h->qp) {
        /* TN3205 caps a device at 10 UC QPs; a channel takes two, so a busy node
         * can genuinely run out. Say so rather than blaming the attributes. */
        if (errlen) snprintf(err, errlen,
                             "ibv_create_qp(UC) failed: %s (device limit is 10 UC QPs; "
                             "a channel uses 2)", strerror(errno));
        goto fail;
    }

    /* Register the ring (below) while the QP is in INIT: the provider maps a
     * QP's MR set once, on the RTR transition. */
    if (ds4_rdma_qp_to_init(V, h->qp, h->port_num, err, errlen) != 0) goto fail;

    /* The provider may hand back shallower queues than requested; a ring that
     * does not fit would silently serialize the link, so report what we got. */
    struct ibv_qp_attr qa;
    struct ibv_qp_init_attr qia;
    memset(&qa, 0, sizeof(qa));
    memset(&qia, 0, sizeof(qia));
    if (V->query_qp(h->qp, &qa, IBV_QP_CAP, &qia) == 0) {
        const uint32_t got = is_tx ? qa.cap.max_send_wr : qa.cap.max_recv_wr;
        if (got < DIST_RDMA_QUEUE_FRAMES) {
            fprintf(stderr,
                    "ds4 rdma: %s queue depth capped at %u frames (asked %u); "
                    "in-flight bytes limited to %u KiB\n",
                    is_tx ? "send" : "recv", got, (unsigned)DIST_RDMA_QUEUE_FRAMES,
                    (unsigned)((uint64_t)got * DIST_RDMA_FRAME / 1024u));
        }
    }

    if (dist_rdma_half_alloc_ring(h, err, errlen) != 0) goto fail;

    if (ep_out) {
        memset(ep_out, 0, sizeof(*ep_out));
        memcpy(ep_out->gid, gid.raw, 16);
        ep_out->qp_num = h->qp->qp_num;
        ep_out->psn = DS4_DIST_RDMA_PSN;
        ep_out->dev_port = h->port_num;
        ep_out->mtu = (uint32_t)h->path_mtu;
        ep_out->lid = pa.lid;
        snprintf(ep_out->dev_name, sizeof(ep_out->dev_name), "%s", resolved_name);
    }
    return 0;

fail:
    dist_rdma_half_release(h);
    return -1;
}

static int dist_rdma_half_connect(dist_rdma_half *h,
                                  const ds4_dist_rdma_endpoint *remote,
                                  char *err,
                                  size_t errlen) {
    const ds4_rdma_verbs_api *V = dist_rdma_verbs();
    if (!V) return -1;
    return ds4_rdma_qp_to_rtr_rts(V, h->qp, h->port_num, h->gid_index,
                                  DS4_DIST_RDMA_PSN, remote->gid, remote->qp_num,
                                  remote->psn, remote->lid, err, errlen);
}

int ds4_dist_rdma_conn_create(ds4_dist_rdma_conn **out,
                              const char *dev_name,
                              ds4_dist_rdma_endpoints *local_out,
                              char *err,
                              size_t errlen) {
    if (out) *out = NULL;
    ds4_dist_rdma_conn *conn = calloc(1, sizeof(*conn));
    if (!conn) {
        if (errlen) snprintf(err, errlen, "out of memory creating RDMA connection");
        return -1;
    }
    ds4_dist_rdma_endpoints eps;
    memset(&eps, 0, sizeof(eps));
    if (dist_rdma_half_create(&conn->tx, dev_name, 1, &eps.tx, err, errlen) != 0) {
        free(conn);
        return -1;
    }
    /* Pin the second half to the device the first one resolved: with dev_name
     * unset the two halves must still land on the same Thunderbolt link. */
    if (dist_rdma_half_create(&conn->rx, eps.tx.dev_name, 0, &eps.rx, err, errlen) != 0) {
        dist_rdma_half_release(&conn->tx);
        free(conn);
        return -1;
    }
    if (local_out) *local_out = eps;
    *out = conn;
    return 0;
}

void ds4_dist_rdma_conn_destroy(ds4_dist_rdma_conn *conn) {
    if (!conn) return;
    dist_rdma_half_release(&conn->tx);
    dist_rdma_half_release(&conn->rx);
    free(conn);
}

int ds4_dist_rdma_conn_connect(ds4_dist_rdma_conn *conn,
                               const ds4_dist_rdma_endpoints *remote,
                               char *err,
                               size_t errlen) {
    if (!conn || !remote) {
        if (errlen) snprintf(err, errlen, "invalid RDMA connect arguments");
        return -1;
    }
    /* Crossed: our outbound QP is wired to the peer's inbound QP. Both peers run
     * this same line, which is what makes the pairing agree without a role. */
    if (dist_rdma_half_connect(&conn->tx, &remote->rx, err, errlen) != 0) return -1;
    if (dist_rdma_half_connect(&conn->rx, &remote->tx, err, errlen) != 0) return -1;
    return 0;
}

/* =========================================================================
 * Reliable in-order byte stream over UC SEND/RECV
 * =========================================================================
 *
 * Gives the protocol its byte-stream contract (write: 0/-1, read: 1/0/-1) on top
 * of two-sided SEND/RECV, over fixed MR-registered rings with in-order reassembly.
 *
 * Flow control is the Thunderbolt controller's (Apple TN3205: a SEND is held,
 * not dropped, until the peer has posted a matching RECV, via hardware credits).
 * So the only backpressure here is the send ring: a write blocks when every
 * buffer is in flight, and a buffer frees when its SEND completes - which, by
 * that handshake, means the peer took it.
 *
 * QP pair for lock-free full duplex. The channel must behave like a TCP socket:
 * a writer and a reader making progress at once (pipelined prefill, worker
 * prefetch, the worker->worker relay). Rather than lock one QP, each direction
 * gets its own half; with credits in hardware each half is purely one-way:
 *
 *     writer thread   conn->tx  -- SEND -->  peer's conn->rx   reader thread
 *     reader thread   conn->rx  <-- SEND --  peer's conn->tx   writer thread
 *
 * The writer touches only conn->tx, the reader only conn->rx - no verbs object,
 * ring or CQ is shared, so neither blocks the other. tx_mu/rx_mu only serialize
 * multiple writers or multiple readers and are never held together (no lock
 * order). This independence is what lets pipelined prefill run over RDMA.
 *
 * Framing. write() accumulates into a reserved buffer and posts it when full, so
 * many small writes become few full frames; the partial tail goes out when the
 * message declared by begin_message() completes (undeclared writes flush at
 * once). It does not wait for a read() at turnaround: with an independent reader
 * thread nothing guarantees one follows, and a pending message would hang the peer.
 *
 * The link is lossless (Thunderbolt) but UC has no retransmit, so each frame
 * carries a sequence number: a gap (or a CQ error) latches the channel broken
 * and surfaces -1, dropping the route, rather than handing over garbage.
 */

typedef struct {
    ds4_dist_rdma_conn *conn;    /* owns both halves: QP/PD/CQ/context/ring */

    /* ---- tx half: data out. Writers only, under tx_mu. ---- */
    pthread_mutex_t tx_mu;
    int tx_nbuf;                        /* send ring depth                     */
    int tx_busy[DIST_RDMA_NBUF];        /* send buffer still in flight         */
    int tx_pend;                        /* send buffer accumulating, or -1     */
    uint32_t tx_pend_len;               /* payload bytes accumulated in it     */
    uint64_t tx_msg_left;               /* bytes left in the declared message  */
    uint32_t tx_seq;                    /* next frame sequence number          */

    /* ---- rx half: data in. Readers only, under rx_mu. ---- */
    pthread_mutex_t rx_mu;
    int rx_nbuf;                        /* recv ring depth                     */
    struct {
        int buf;
        uint32_t off;   /* payload bytes already consumed */
        uint32_t len;   /* payload length                 */
    } inq[DIST_RDMA_NBUF];
    int inq_head;
    int inq_count;
    uint32_t rx_seq;                    /* next expected frame sequence number */

    int liveness_fd;    /* TCP control fd of the peer; polled for HUP so a read
                         * returns EOF when the peer exits (UC has no FIN). -1 = none */
    /* Set by whichever thread first sees a failure, and by shutdown() from a
     * third thread; every loop below rechecks it, so a relaxed atomic is enough. */
    volatile int broken;
} dist_rdma_chan;

static int dist_rdma_trace(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("DS4_DIST_RDMA_TRACE") ? 1 : 0;
    return cached;
}

/* ---- posting helpers: each names the half it drives ---- */

static int dist_rdma_post_data_recv(dist_rdma_chan *c, int i) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)((char *)c->conn->rx.data_mem + (size_t)i * DIST_RDMA_STRIDE);
    sge.length = (uint32_t)DIST_RDMA_STRIDE;
    sge.lkey = c->conn->rx.data_mr->lkey;
    struct ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = DIST_RDMA_RECV_WR_FLAG | (uint64_t)i;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    struct ibv_recv_wr *bad = NULL;
    return ibv_post_recv(c->conn->rx.qp, &wr, &bad);
}

static int dist_rdma_post_data_send(dist_rdma_chan *c, int i) {
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)((char *)c->conn->tx.data_mem + (size_t)i * DIST_RDMA_STRIDE);
    sge.length = (uint32_t)DIST_RDMA_STRIDE;   /* full stride: send size == recv size */
    sge.lkey = c->conn->tx.data_mr->lkey;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)i;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    struct ibv_send_wr *bad = NULL;
    return ibv_post_send(c->conn->tx.qp, &wr, &bad);
}

/* True if the peer's TCP control fd has hung up (peer process gone). UC never
 * signals a disconnect, so this is how a blocked read or a write waiting on send
 * completions notices the peer left. A peer process exit sends a FIN, which shows up as the
 * control fd becoming readable (POLLIN, recv would return 0) rather than POLLHUP
 * (full close). No real control data flows after bootstrap, so any readability
 * == disconnect. poll() on a shared fd is safe from either thread. */
static int dist_rdma_liveness_gone(dist_rdma_chan *c) {
    if (c->liveness_fd < 0) return 0;
    struct pollfd pfd = { .fd = c->liveness_fd, .events = POLLIN, .revents = 0 };
    if (poll(&pfd, 1, 0) > 0 &&
        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) return 1;
    return 0;
}

/* Busy-poll hard for a short burst (lowest latency for the active
 * request/response), then sleep between polls so an idle thread does not pin a
 * core. The ~60us wait adds <0.1ms to noticing a frame, negligible against
 * per-token compute, and drops idle CPU to ~0. */
static void dist_rdma_idle_backoff(unsigned idle) {
    if (idle > 64u) {
        struct timespec ts = { 0, 60000 };  /* 60 us */
        nanosleep(&ts, NULL);
    }
}

/* ---- tx half (writer thread; tx_mu held) ---- */

/* Drain the tx CQ once, freeing send buffers whose frames the peer has taken.
 * Returns completions processed, or -1 on error. */
static int dist_rdma_tx_progress(dist_rdma_chan *c) {
    struct ibv_wc wc[DIST_RDMA_NBUF];
    int n = ibv_poll_cq(c->conn->tx.cq, (int)(sizeof(wc) / sizeof(wc[0])), wc);
    if (n < 0) { fprintf(stderr, "ds4 rdma: ibv_poll_cq(tx) failed\n"); c->broken = 1; return -1; }
    for (int j = 0; j < n; j++) {
        if (wc[j].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "ds4 rdma: send completion error: status=%d wr_id=0x%llx\n",
                    (int)wc[j].status, (unsigned long long)wc[j].wr_id);
            c->broken = 1;
            return -1;
        }
        c->tx_busy[(int)(wc[j].wr_id & 0xffff)] = 0;
    }
    return n;
}

/* Reserve a free send buffer to accumulate into. Blocks until one frees, which
 * is the channel's entire backpressure story: a send only completes once the
 * controller has handed the frame to a recv the peer posted, so a full ring
 * means the peer is genuinely behind. Gives up if the peer process is gone
 * (nothing will ever complete then). Returns 0, or -1 if the channel broke. */
static int dist_rdma_tx_acquire(dist_rdma_chan *c) {
    if (c->tx_pend >= 0) return 0;
    unsigned idle = 0;
    for (;;) {
        if (c->broken) return -1;
        for (int k = 0; k < c->tx_nbuf; k++) {
            if (!c->tx_busy[k]) { c->tx_pend = k; c->tx_pend_len = 0; return 0; }
        }
        int n = dist_rdma_tx_progress(c);
        if (n < 0) return -1;
        if (n == 0) {
            if ((++idle & 255u) == 0 && dist_rdma_liveness_gone(c)) {
                fprintf(stderr, "ds4 rdma: peer gone while waiting for a free send buffer\n");
                c->broken = 1;
                return -1;
            }
            dist_rdma_idle_backoff(idle);
        } else {
            idle = 0;
        }
    }
}

/* Put the pending frame on the wire, padded to STRIDE (send size == recv size).
 * The buffer is already reserved and the controller queues the send until the
 * peer has a recv for it, so this returns as soon as the post is accepted. */
static int dist_rdma_tx_flush(dist_rdma_chan *c) {
    if (c->tx_pend < 0) return 0;
    if (c->broken) return -1;
    const int i = c->tx_pend;
    dist_rdma_seg_hdr *h =
        (dist_rdma_seg_hdr *)((char *)c->conn->tx.data_mem + (size_t)i * DIST_RDMA_STRIDE);
    h->magic = DIST_RDMA_SEG_MAGIC;
    h->len = c->tx_pend_len;
    h->seq = c->tx_seq;
    h->reserved = 0;
    if (dist_rdma_trace())
        fprintf(stderr, "ds4 rdma: send[%d] seq=%u len=%u\n", i, h->seq, h->len);
    /* A provider that handed back a shallower send queue than requested refuses
     * the post once that queue is full instead of saying so up front. While
     * anything is still in flight, treat a refusal as backpressure - drain a
     * completion and retry - which self-tunes the usable depth the same way the
     * recv ring is tuned at wrap time. With nothing outstanding it is real. */
    unsigned idle = 0;
    while (dist_rdma_post_data_send(c, i) != 0) {
        int outstanding = 0;
        for (int k = 0; k < c->tx_nbuf; k++) if (c->tx_busy[k]) { outstanding = 1; break; }
        if (!outstanding) {
            fprintf(stderr, "ds4 rdma: ibv_post_send(data) failed: %s\n", strerror(errno));
            c->broken = 1;
            return -1;
        }
        int n = dist_rdma_tx_progress(c);
        if (n < 0) return -1;
        if (n == 0) {
            if ((++idle & 255u) == 0 && dist_rdma_liveness_gone(c)) {
                fprintf(stderr, "ds4 rdma: peer gone with the send queue full\n");
                c->broken = 1;
                return -1;
            }
            dist_rdma_idle_backoff(idle);
        } else {
            idle = 0;
        }
    }
    c->tx_busy[i] = 1;
    c->tx_seq++;
    c->tx_pend = -1;
    c->tx_pend_len = 0;
    return 0;
}

static int dist_rdma_flush(void *ctx) {
    dist_rdma_chan *c = ctx;
    pthread_mutex_lock(&c->tx_mu);
    const int rc = dist_rdma_tx_flush(c);
    pthread_mutex_unlock(&c->tx_mu);
    return rc;
}

static void dist_rdma_begin_message(void *ctx, size_t bytes) {
    dist_rdma_chan *c = ctx;
    pthread_mutex_lock(&c->tx_mu);
    c->tx_msg_left = (uint64_t)bytes;
    pthread_mutex_unlock(&c->tx_mu);
}

/* Coalescing write: append to the pending frame, posting whenever one fills, and
 * put the tail on the wire once the declared message is complete (or right away
 * if the caller never declared one). */
static int dist_rdma_write(void *ctx, const void *buf, size_t len) {
    dist_rdma_chan *c = ctx;
    const char *p = buf;
    int rc = 0;
    pthread_mutex_lock(&c->tx_mu);
    while (len > 0) {
        if (c->broken) { rc = -1; break; }
        if (dist_rdma_tx_acquire(c) < 0) { rc = -1; break; }
        char *sb = (char *)c->conn->tx.data_mem + (size_t)c->tx_pend * DIST_RDMA_STRIDE;
        size_t space = (size_t)DIST_RDMA_PAYLOAD - c->tx_pend_len;
        size_t chunk = len < space ? len : space;
        memcpy(sb + sizeof(dist_rdma_seg_hdr) + c->tx_pend_len, p, chunk);
        c->tx_pend_len += (uint32_t)chunk;
        p += chunk;
        len -= chunk;
        c->tx_msg_left = c->tx_msg_left > chunk ? c->tx_msg_left - chunk : 0;
        if (c->tx_pend_len == DIST_RDMA_PAYLOAD && dist_rdma_tx_flush(c) < 0) { rc = -1; break; }
    }
    if (rc == 0 && c->tx_msg_left == 0 && dist_rdma_tx_flush(c) < 0) rc = -1;
    pthread_mutex_unlock(&c->tx_mu);
    return rc;
}

/* ---- rx half (reader thread; rx_mu held) ---- */

/* Drain the rx CQ once, queueing arrived segments; the recv is re-posted only
 * after read() has consumed the buffer. */
static int dist_rdma_rx_progress(dist_rdma_chan *c) {
    struct ibv_wc wc[DIST_RDMA_NBUF];
    int n = ibv_poll_cq(c->conn->rx.cq, (int)(sizeof(wc) / sizeof(wc[0])), wc);
    if (n < 0) { fprintf(stderr, "ds4 rdma: ibv_poll_cq(rx) failed\n"); c->broken = 1; return -1; }
    for (int j = 0; j < n; j++) {
        if (wc[j].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "ds4 rdma: recv completion error: status=%d wr_id=0x%llx byte_len=%u\n",
                    (int)wc[j].status, (unsigned long long)wc[j].wr_id, wc[j].byte_len);
            c->broken = 1;
            return -1;
        }
        const int b = (int)(wc[j].wr_id & 0xffff);
        const dist_rdma_seg_hdr *h =
            (const dist_rdma_seg_hdr *)((char *)c->conn->rx.data_mem +
                                        (size_t)b * DIST_RDMA_STRIDE);
        if (h->magic != DIST_RDMA_SEG_MAGIC || h->len > DIST_RDMA_PAYLOAD) {
            fprintf(stderr, "ds4 rdma: bad segment on recv[%d]: magic=0x%08x len=%u byte_len=%u\n",
                    b, h->magic, h->len, wc[j].byte_len);
            c->broken = 1;
            return -1;
        }
        if (h->seq != c->rx_seq) {
            /* Completions are delivered in post order, so a mismatch means the
             * link dropped or reordered a frame and the stream is unrecoverable
             * (UC has no retransmit). Fail loudly instead of silently splicing. */
            fprintf(stderr, "ds4 rdma: segment sequence gap on recv[%d]: got %u expected %u\n",
                    b, h->seq, c->rx_seq);
            c->broken = 1;
            return -1;
        }
        c->rx_seq++;
        if (dist_rdma_trace())
            fprintf(stderr, "ds4 rdma: recv[%d] seq=%u len=%u byte_len=%u\n",
                    b, h->seq, h->len, wc[j].byte_len);
        const int slot = (c->inq_head + c->inq_count) % c->rx_nbuf;
        c->inq[slot].buf = b;
        c->inq[slot].off = 0;
        c->inq[slot].len = h->len;
        c->inq_count++;                     /* re-post deferred to read() */
    }
    return n;
}

static int dist_rdma_read(void *ctx, void *buf, size_t len) {
    dist_rdma_chan *c = ctx;
    char *p = buf;
    int rc = 1;
    unsigned idle = 0;
    pthread_mutex_lock(&c->rx_mu);
    while (len > 0) {
        if (c->inq_count == 0) {
            if (c->broken) { rc = -1; break; }
            int n = dist_rdma_rx_progress(c);
            if (n < 0) { rc = -1; break; }
            if (n == 0) {
                /* Periodically check whether the peer has disconnected so a
                 * one-shot coordinator exit doesn't leave the worker spinning
                 * forever. Return clean EOF (like a TCP recv of 0). */
                if ((++idle & 255u) == 0 && dist_rdma_liveness_gone(c)) { rc = 0; break; }
                dist_rdma_idle_backoff(idle);
            } else {
                idle = 0;
            }
            continue;                       /* poll for a data segment */
        }
        idle = 0;
        const int slot = c->inq_head;
        const int b = c->inq[slot].buf;
        const uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        const uint32_t take = (len < (size_t)avail) ? (uint32_t)len : avail;
        memcpy(p,
               (char *)c->conn->rx.data_mem + (size_t)b * DIST_RDMA_STRIDE +
                   sizeof(dist_rdma_seg_hdr) + c->inq[slot].off,
               take);
        p += take;
        len -= take;
        c->inq[slot].off += take;
        if (c->inq[slot].off == c->inq[slot].len) {
            /* Re-posting the recv is what returns hardware credit to the peer's
             * sender, so it happens the moment the buffer is drained. */
            if (dist_rdma_post_data_recv(c, b) != 0) { c->broken = 1; rc = -1; break; }
            c->inq_head = (c->inq_head + 1) % c->rx_nbuf;
            c->inq_count--;
        }
    }
    pthread_mutex_unlock(&c->rx_mu);
    return rc;
}

/* Flush outstanding work on both QPs to error so a poll in either thread
 * unblocks (UC has no FIN). Safe to call from a thread that owns neither half,
 * and safe to call more than once. */
static void dist_rdma_shutdown(void *ctx) {
    dist_rdma_chan *c = ctx;
    c->broken = 1;
    const ds4_rdma_verbs_api *V = dist_rdma_verbs();
    if (!V) return;
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_ERR;
    if (c->conn->tx.qp) V->modify_qp(c->conn->tx.qp, &attr, IBV_QP_STATE);
    if (c->conn->rx.qp) V->modify_qp(c->conn->rx.qp, &attr, IBV_QP_STATE);
}

/* Drain both queues before their MRs go away: ibv_dereg_mr clears the DART PTEs
 * synchronously, so a WQE the Thunderbolt controller is still reading/writing
 * would fault the IOMMU and panic the kernel. Forcing the QPs to ERR flushes
 * every WQE to a completion; counting them all (including IBV_WC_WR_FLUSH_ERR)
 * proves the ring is idle. Runs even when broken - shutdown()'s ERR is async. */
static void dist_rdma_teardown_drain(dist_rdma_chan *c) {
    const ds4_rdma_verbs_api *V = dist_rdma_verbs();
    if (!V) return;
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_ERR;
    if (c->conn->tx.qp) V->modify_qp(c->conn->tx.qp, &attr, IBV_QP_STATE);
    if (c->conn->rx.qp) V->modify_qp(c->conn->rx.qp, &attr, IBV_QP_STATE);

    int tx_left = 0;
    for (int k = 0; k < c->tx_nbuf; k++) if (c->tx_busy[k]) tx_left++;
    /* Recvs already reaped into inq have been observed; the rest are still
     * posted and will flush. */
    int rx_left = c->rx_nbuf - c->inq_count;
    if (rx_left < 0) rx_left = 0;

    struct ibv_wc wc[DIST_RDMA_NBUF];
    const int cap = (int)(sizeof(wc) / sizeof(wc[0]));
    for (long spin = 0; spin < 1000000L && (tx_left > 0 || rx_left > 0); spin++) {
        if (tx_left > 0 && c->conn->tx.cq) {
            int n = ibv_poll_cq(c->conn->tx.cq, cap, wc);
            if (n > 0) tx_left -= n;
        }
        if (rx_left > 0 && c->conn->rx.cq) {
            int n = ibv_poll_cq(c->conn->rx.cq, cap, wc);
            if (n > 0) rx_left -= n;
        }
    }
    if (tx_left > 0 || rx_left > 0) {
        fprintf(stderr,
                "ds4 rdma: teardown drain did not quiesce (tx=%d rx=%d work requests outstanding)\n",
                tx_left, rx_left);
    }
}

static void dist_rdma_close(void *ctx) {
    dist_rdma_chan *c = ctx;
    if (!c) return;
    /* Best-effort on a healthy channel: flush any pending write and let the
     * outstanding sends complete, so a final message (e.g. an ack) reaches the
     * peer before the QPs are torn down. Callers close only after their
     * reader/writer threads are joined. */
    if (!c->broken) {
        dist_rdma_tx_flush(c);
        for (int spin = 0; !c->broken && spin < 100000; spin++) {
            int busy = 0;
            for (int k = 0; k < c->tx_nbuf; k++) if (c->tx_busy[k]) { busy = 1; break; }
            if (!busy) break;
            if (dist_rdma_tx_progress(c) < 0) break;
        }
    }
    dist_rdma_teardown_drain(c);
    ds4_dist_rdma_conn_destroy(c->conn);
    c->conn = NULL;
    pthread_mutex_destroy(&c->tx_mu);
    pthread_mutex_destroy(&c->rx_mu);
    free(c);
}

static int dist_rdma_fd(void *ctx) {
    (void)ctx;
    return -1;   /* not socket-backed */
}

static const ds4_dist_dchan_ops dist_rdma_dchan_ops = {
    dist_rdma_write,
    dist_rdma_read,
    dist_rdma_shutdown,
    dist_rdma_close,
    dist_rdma_fd,
    dist_rdma_begin_message,
    dist_rdma_flush,
};

ds4_dist_dchan *ds4_dist_dchan_from_rdma_conn(ds4_dist_rdma_conn *conn, int liveness_fd) {
    if (!conn) return NULL;
    dist_rdma_chan *c = calloc(1, sizeof(*c));
    if (!c) {
        fprintf(stderr, "ds4 rdma: chan calloc failed\n");
        ds4_dist_rdma_conn_destroy(conn);
        return NULL;
    }
    c->conn = conn;
    c->tx_pend = -1;
    c->liveness_fd = liveness_fd;
    pthread_mutex_init(&c->tx_mu, NULL);
    pthread_mutex_init(&c->rx_mu, NULL);

    /* Fill the recv ring. Depth is self-tuned rather than assumed: if the
     * provider hands back a shallower queue than requested it refuses the extra
     * posts, and the ring is simply smaller. Nothing has to agree with the peer -
     * each side's ring is local, and the controller's own flow control matches
     * sends to whatever recvs the other end has posted. */
    int posted = 0;
    for (int i = 0; i < DIST_RDMA_NBUF; i++) {
        if (dist_rdma_post_data_recv(c, i) != 0) {
            if (posted < 2) {
                fprintf(stderr, "ds4 rdma: ibv_post_recv(data)[%d] failed: %s (only %d posted)\n",
                        i, strerror(errno), posted);
                goto fail;
            }
            break;   /* hit the provider's recv-depth cap; use what we posted */
        }
        posted++;
    }
    c->rx_nbuf = posted;
    c->tx_nbuf = DIST_RDMA_NBUF;   /* send ring is local: no peer agreement needed */
    if (posted < DIST_RDMA_NBUF) {
        fprintf(stderr, "ds4 rdma: recv ring depth limited to %d/%d buffers by the provider\n",
                posted, DIST_RDMA_NBUF);
    }

    ds4_dist_dchan *d = malloc(sizeof(*d));
    if (!d) { fprintf(stderr, "ds4 rdma: dchan malloc failed\n"); goto fail; }
    d->ops = &dist_rdma_dchan_ops;
    d->ctx = c;
    return d;

fail:
    /* Ownership of the connection arrives with the call, so a failed wrap must
     * destroy it here: callers null their handle unconditionally and would
     * otherwise strand kernel QP/PD rows that can never even be tombstoned. */
    ds4_dist_rdma_conn_destroy(c->conn);
    pthread_mutex_destroy(&c->tx_mu);
    pthread_mutex_destroy(&c->rx_mu);
    free(c);
    return NULL;
}

#else /* !DS4_RDMA_HAVE_VERBS: no <infiniband/verbs.h> (non-Apple) - TCP only */

#define DS4_RDMA_NO_VERBS_MSG "RDMA over Thunderbolt is Apple-only; this build has no verbs support"

int ds4_dist_rdma_available(void) { return 0; }

int ds4_dist_rdma_describe(char *buf, size_t buflen) {
    if (buflen) snprintf(buf, buflen, DS4_RDMA_NO_VERBS_MSG);
    return 0;
}

int ds4_dist_rdma_conn_create(ds4_dist_rdma_conn **out, const char *dev_name,
                              ds4_dist_rdma_endpoints *local_out, char *err, size_t errlen) {
    (void)dev_name;
    (void)local_out;
    if (out) *out = NULL;
    if (errlen) snprintf(err, errlen, DS4_RDMA_NO_VERBS_MSG);
    return -1;
}

void ds4_dist_rdma_conn_destroy(ds4_dist_rdma_conn *conn) { (void)conn; }

int ds4_dist_rdma_conn_connect(ds4_dist_rdma_conn *conn, const ds4_dist_rdma_endpoints *remote,
                               char *err, size_t errlen) {
    (void)conn;
    (void)remote;
    if (errlen) snprintf(err, errlen, DS4_RDMA_NO_VERBS_MSG);
    return -1;
}

ds4_dist_dchan *ds4_dist_dchan_from_rdma_conn(ds4_dist_rdma_conn *conn, int liveness_fd) {
    (void)conn;
    (void)liveness_fd;
    return NULL;
}

#endif /* DS4_RDMA_HAVE_VERBS */
