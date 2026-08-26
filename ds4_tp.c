/* Tensor-parallel transport and lockstep protocol.  See ds4_tp.h and
 * misc/METAL_TENSOR_PARALLELISM.md for the design.
 *
 * Wire notes: both ranks are identical Apple Silicon machines by
 * definition, so the wire format is host little-endian; the hello magic
 * doubles as a byte-order check.  The control socket is a plain blocking
 * TCP stream carrying framed commands.  Gate traffic goes over RDMA
 * (Thunderbolt UC queue pair, two-sided send/recv — see the driver quirks
 * note at ds4_tp_rdma) or over a dedicated full-duplex TCP socket at 16KB
 * per direction as the fallback. */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <stdarg.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "ds4_tp.h"
#include "ds4_gpu.h"

#if defined(__APPLE__) && defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#define DS4_TP_HAVE_VERBS 1
#endif
#endif

#define DS4_TP_MAGIC UINT32_C(0x44533454) /* "DS4T" */
#define DS4_TP_BATCH_MAGIC UINT32_C(0x44533442) /* "DS4B" */
#define DS4_TP_PROTOCOL_VERSION 10u


#define DS4_TP_DEFAULT_TIMEOUT_SEC 300
/* Once both ranks enter a Metal gate, a live exchange normally completes in
 * microseconds. Fail well before Metal's command-buffer watchdog if the peer
 * stalls while keeping its sockets open. */
#define DS4_TP_DEFAULT_GATE_TIMEOUT_MS 750
#define DS4_TP_MAX_TIMEOUT_SEC 86400u
#define DS4_TP_COMMAND_MAX_TOKENS 1048576u
#define DS4_TP_COMMAND_MAX_BATCH_ITEMS 4096u

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_tp_frame_header;

typedef struct {
    uint32_t magic;      /* also detects byte-order mismatch */
    uint32_t version;
    uint32_t role;
    uint32_t rdma_ok;    /* this side has a usable verbs device */
    uint32_t nhi_ok;     /* this side explicitly selected an NHI device */
    uint32_t nhi_ring_frames;
    uint32_t nhi_msg_frames;
    uint32_t transport;
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint32_t pad;
    uint64_t gate_slot_mask[DS4_TP_GATE_MASK_WORDS];
} ds4_tp_hello_fixed;

typedef struct {
    uint64_t slab_base;
    uint32_t rkey;
    uint32_t qpn;
    uint32_t psn;
    uint32_t mtu;
    uint16_t lid;
    uint8_t gid[16];
    uint8_t link_layer;
} ds4_tp_rdma_info;

/* TCP gate frames carry a small header so a desynchronized pair fails loudly
 * instead of silently mixing partials. */
typedef struct {
    uint32_t magic;
    uint16_t layer;
    uint16_t gate;
    uint64_t seq;
} ds4_tp_gate_header;

#ifdef DS4_TP_HAVE_VERBS
/* librdma is loaded at runtime so builds and machines without the RDMA
 * stack (or with it disabled) fall back to TCP with no link-time cost.
 * ibv_post_send()/ibv_poll_cq() are header inlines over context->ops, so
 * only the setup entry points need dlsym. */
typedef struct {
    void *handle;
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_device)(struct ibv_context *, struct ibv_device_attr *);
    int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *);
    int (*dealloc_pd)(struct ibv_pd *);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
    int (*dereg_mr)(struct ibv_mr *);
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
    int (*destroy_cq)(struct ibv_cq *);
    struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
    int (*destroy_qp)(struct ibv_qp *);
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
    int (*query_qp)(struct ibv_qp *, struct ibv_qp_attr *, int, struct ibv_qp_init_attr *);
} ds4_tp_verbs_api;

/* AppleThunderboltRDMA quirks (validated with scratchpad probes,
 * 2026-07-06): only UC queue pairs exist (RC/UD: ENOTSUP); RDMA WRITE work
 * requests are accepted but never execute, so the data plane is two-sided
 * SEND/RECV like Apple's own JACCL; messages above 16KB are not delivered;
 * RTR requires GRH addressing with the IPv4-mapped GID that appears only
 * once the Thunderbolt member interface has an IPv4 address of its own.
 * UC delivery is in-order and the gate sequence is globally deterministic
 * (a model-fixed number of gates per token). After any initial bulk prefill,
 * decode keeps a receive window posted by sequence number: recv for seq s
 * lands in the slab in-slot (s-1) % slots and its completion is the arrival
 * signal. */
#define DS4_TP_RDMA_MAX_MSG 16384
#define DS4_TP_RDMA_RECV_WINDOW 16
#define DS4_TP_RDMA_BULK_SLOTS 64
#define DS4_TP_RDMA_BULK_WR_TAG (UINT64_C(1) << 63)
#define DS4_TP_RDMA_BLOCK_WR_TAG (UINT64_C(1) << 61)

typedef struct {
    ds4_tp_verbs_api api;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    struct ibv_port_attr port;
    union ibv_gid gid;
    int gid_index;
    uint32_t max_inline;
    ds4_tp_rdma_info peer;
    uint32_t send_outstanding;  /* signaled sends not yet reaped */
    uint64_t recv_done;         /* highest gate seq whose recv completed */
    uint64_t last_gate_seq;     /* last real decode receive consumed */
    bool recv_window_active;    /* decode recvs are queued ahead */
    int warm_failed;            /* last warm-up round failed (dead direction) */
    uint32_t setup_attempt;     /* queue pair recreations so far */
    pthread_mutex_t post_lock;
    uint32_t recv_depth;        /* queue pair receive depth actually granted */
    uint32_t send_depth;        /* queue pair send depth actually granted */
    struct ibv_sge *win_sge;    /* window work request arrays (recv_depth) */
    struct ibv_recv_wr *win_rwr;
    struct ibv_send_wr *win_swr;
    /* Verify-block window (speculative decoding): one batch gate per layer,
     * receives posted a layer ahead of every send, no per-gate control
     * traffic (see ds4_tp_batch_block_begin). */
    bool block_active;
    uint32_t block_rows;
    uint32_t block_layers;
    uint32_t block_posted;      /* layers whose receives are posted */
    uint64_t block_recv_done;   /* row messages received in this block */
} ds4_tp_rdma;
#endif

struct ds4_tp {
    ds4_tp_options opt;
    int rank;                   /* 0 leader, 1 worker */
    int control_fd;
    int data_fd;                /* TCP fallback, headers, and verify gates */
    bool rdma_active;
    bool nhi_active;
    uint32_t peer_ctx;
    uint32_t local_ctx;
    uint32_t n_layer;
    uint32_t n_embd;
    uint64_t vec_bytes;
    uint32_t n_slots;
    /* Decode gate schedule (see ds4_tp_identity). */
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint64_t gate_slot_mask[DS4_TP_GATE_MASK_WORDS];
    uint8_t *slab;
    uint64_t slab_bytes;
    /* Slab regions, see ds4_tp.h layout comment. */
    uint64_t out_off;
    uint64_t in_off;
    uint64_t in_flags_off;
    uint64_t token_off;
    uint64_t out_flags_off;     /* local staging for RDMA flag writes */
    uint64_t gpu_flags_off;     /* GPU-written gate-ready flags (u32/slot) */
    uint64_t batch_out_off;     /* [layer][row] verify-block local partials */
    uint64_t batch_in_off;      /* [layer][row] verify-block peer partials */
    uint64_t timeout_sec;
    uint64_t gate_timeout_ms;
    atomic_bool failed;
#ifdef DS4_TP_HAVE_VERBS
    ds4_tp_rdma rdma;
#endif
};

/* ------------------------------------------------------------------------
 * Small socket helpers (same conventions as ds4_distributed.c).
 * --------------------------------------------------------------------- */

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void tp_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static int tp_write_full(int fd, const void *buf, size_t len) {
    const char *p = buf;
    while (len) {
#ifdef MSG_NOSIGNAL
        ssize_t w = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t w = send(fd, p, len, 0);
#endif
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (w == 0) return 0;
        p += w;
        len -= (size_t)w;
    }
    return 1;
}

static int tp_read_full(int fd, void *buf, size_t len) {
    char *p = buf;
    while (len) {
        ssize_t r = read(fd, p, len);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += r;
        len -= (size_t)r;
    }
    return 1;
}

static void tp_socket_tune(int fd, uint64_t timeout_sec) {
    int one = 1;
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    /* Gate exchanges are latency-critical 16KB messages; large socket
     * buffers only matter for the TCP fallback's pipelining. */
    int sz = 4 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
    struct timeval timeout = {
        .tv_sec = (time_t)timeout_sec,
        .tv_usec = 0,
    };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
}

static int tp_socket_set_gate_timeout(int fd, uint64_t timeout_ms) {
    struct timeval tv = {
        .tv_sec = (time_t)(timeout_ms / 1000u),
        .tv_usec = (suseconds_t)((timeout_ms % 1000u) * 1000u),
    };
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0)
        return 0;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0)
        return 0;
    return 1;
}

#ifdef DS4_TP_HAVE_VERBS
/* UC queue pairs do not report a dead remote reliably.  The control socket
 * does, so sample it while polling an RDMA completion and abort before the
 * Metal command-buffer watchdog fires. */
static int tp_peer_closed(const ds4_tp *tp) {
    char byte;
    const ssize_t n = recv(tp->control_fd, &byte, 1,
                           MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) return 1;
    if (n > 0) return 0;
    return errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
}
#endif

static int tp_listen(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    int rc = getaddrinfo(host && host[0] ? host : NULL, portbuf, &hints, &res);
    if (rc != 0) {
        tp_set_err(err, errlen, "tp listen resolve %s:%d: %s", host, port, gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 2) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) tp_set_err(err, errlen, "tp listen %s:%d: %s", host, port, strerror(errno));
    return fd;
}

static int tp_dial(const char *host, int port, double timeout_sec, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    double deadline = tp_now_sec() + timeout_sec;
    int last_errno = 0;
    uint32_t attempts = 0;
    do {
        struct addrinfo hints = {0}, *res = NULL;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        int gai = getaddrinfo(host, portbuf, &hints, &res);
        if (gai == 0) {
            for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
                int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0) continue;
                if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    return fd;
                }
                last_errno = errno;
                close(fd);
            }
            freeaddrinfo(res);
        }
        /* Retrying is normal while the peer loads its model; still say why
         * every ~10s so a wrong address or a policy block is visible. */
        if (attempts++ % 50 == 0) {
            fprintf(stderr, "ds4-tp: connecting to %s:%d ... (%s)\n", host, port,
                    gai != 0 ? gai_strerror(gai) :
                    last_errno ? strerror(last_errno) : "no address worked");
        }
        usleep(200 * 1000);
    } while (tp_now_sec() < deadline);
    tp_set_err(err, errlen, "tp connect %s:%d: %s", host, port,
               last_errno ? strerror(last_errno) : "unreachable");
    return -1;
}

static int tp_send_frame(int fd, uint32_t type, const void *payload, uint32_t bytes) {
    ds4_tp_frame_header h = { DS4_TP_MAGIC, type, bytes };
    if (!tp_write_full(fd, &h, sizeof(h))) return 0;
    if (bytes && !tp_write_full(fd, payload, bytes)) return 0;
    return 1;
}

static int tp_read_frame_header(int fd, uint32_t *type, uint32_t *bytes) {
    ds4_tp_frame_header h;
    if (!tp_read_full(fd, &h, sizeof(h))) return 0;
    if (h.magic != DS4_TP_MAGIC) return 0;
    *type = h.type;
    *bytes = h.bytes;
    return 1;
}

/* ------------------------------------------------------------------------
 * Options and CLI.
 * --------------------------------------------------------------------- */

bool ds4_tp_enabled(const ds4_tp_options *opt) {
    return opt && opt->role != DS4_TP_NONE;
}

void ds4_tp_usage(FILE *fp) {
    fprintf(fp,
        "Tensor parallelism (two identical machines):\n"
        "  --tensor-parallel           Use --role/--listen/--coordinator for a 50/50 TP pair.\n"
        "  --transport <auto|rdma|tcp|nhi>\n"
        "                              Gate transport (default auto; ROCm requires nhi).\n"
        "  --rdma-device <name>        Select a verbs device such as rdma_en1.\n"
        "  --nhi-device <path>         Select the local thunderbolt-stream device.\n"
        "  --nhi-ring-frames <n>       Imported NHI ring geometry (default 4096).\n"
        "  --rdma-gid-index <n>        Select the local verbs GID index.\n"
        "  --tensor-parallel-token-prefill\n"
        "                              GLM diagnostic: prefill one token at a time.\n"
        "  --debug-hash <n>            Cross-check hidden state every n tokens.\n");
}

int ds4_tp_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_tp_options *opt,
        char *err,
        size_t errlen)
{
    int i = *index;
    if (!strcmp(arg, "--tensor-parallel")) {
        opt->requested = true;
    } else if (!strcmp(arg, "--transport")) {
        if (i + 1 >= argc) goto missing;
        const char *v = argv[++i];
        if (!strcmp(v, "auto")) opt->transport = DS4_TP_TRANSPORT_AUTO;
        else if (!strcmp(v, "rdma")) opt->transport = DS4_TP_TRANSPORT_RDMA;
        else if (!strcmp(v, "tcp")) opt->transport = DS4_TP_TRANSPORT_TCP;
        else if (!strcmp(v, "nhi")) opt->transport = DS4_TP_TRANSPORT_NHI;
        else {
            tp_set_err(err, errlen, "invalid %s value: %s", arg, v);
            return DS4_TP_CLI_ERROR;
        }
    } else if (!strcmp(arg, "--rdma-device")) {
        if (i + 1 >= argc) goto missing;
        opt->rdma_device = argv[++i];
    } else if (!strcmp(arg, "--nhi-device")) {
        if (i + 1 >= argc) goto missing;
        opt->nhi_device = argv[++i];
        if (!opt->nhi_device[0]) {
            tp_set_err(err, errlen, "--nhi-device requires a nonempty path");
            return DS4_TP_CLI_ERROR;
        }
    } else if (!strcmp(arg, "--nhi-ring-frames")) {
        if (i + 1 >= argc) goto missing;
        char *end = NULL;
        errno = 0;
        unsigned long value = strtoul(argv[++i], &end, 10);
        if (errno != 0 || !end || *end != '\0' || value == 0 ||
            value > UINT32_MAX || (value % DS4_TP_NHI_MSG_FRAMES) != 0) {
            tp_set_err(err, errlen, "invalid --nhi-ring-frames %s", argv[i]);
            return DS4_TP_CLI_ERROR;
        }
        opt->nhi_ring_frames = (uint32_t)value;
    } else if (!strcmp(arg, "--rdma-gid-index")) {
        if (i + 1 >= argc) goto missing;
        char *end = NULL;
        errno = 0;
        long value = strtol(argv[++i], &end, 10);
        if (errno != 0 || !end || *end != '\0' || value < 0 || value > INT_MAX) {
            tp_set_err(err, errlen, "invalid --rdma-gid-index %s", argv[i]);
            return DS4_TP_CLI_ERROR;
        }
        opt->rdma_gid_index = (int)value;
        opt->rdma_gid_index_set = true;
    } else if (!strcmp(arg, "--tensor-parallel-token-prefill")) {
        opt->glm_token_prefill = true;
    } else if (!strcmp(arg, "--debug-hash")) {
        if (i + 1 >= argc) goto missing;
        opt->debug_hash = atoi(argv[++i]);
    } else {
        return DS4_TP_CLI_NOT_MATCHED;
    }
    *index = i;
    return DS4_TP_CLI_MATCHED;
missing:
    tp_set_err(err, errlen, "%s requires an argument", arg);
    return DS4_TP_CLI_ERROR;
}

int ds4_tp_adopt_distributed_options(
        ds4_tp_options *tp,
        ds4_distributed_options *dist,
        char *err,
        size_t errlen)
{
    if (!tp || !dist || !tp->requested) return 1;
    if (tp->role != DS4_TP_NONE) {
        tp_set_err(err, errlen,
                   "--tensor-parallel selects its role through --role");
        return 0;
    }
    if (dist->role == DS4_DISTRIBUTED_NONE) {
        tp_set_err(err, errlen,
                   "--tensor-parallel requires --role coordinator or --role worker");
        return 0;
    }
    if (dist->layers.set) {
        tp_set_err(err, errlen,
                   "tensor parallelism always uses one 50/50 worker; omit --layers");
        return 0;
    }
    if (dist->prefill_chunk || dist->prefill_window || dist->activation_bits ||
        dist->replay_check || dist->debug) {
        tp_set_err(err, errlen,
                   "--dist-* and distributed debug options cannot be used with --tensor-parallel");
        return 0;
    }

    if (dist->role == DS4_DISTRIBUTED_COORDINATOR) {
        if (!dist->listen_host || dist->listen_port <= 0) {
            tp_set_err(err, errlen,
                       "--role coordinator --tensor-parallel requires --listen HOST PORT");
            return 0;
        }
        if (dist->coordinator_host || dist->coordinator_port) {
            tp_set_err(err, errlen,
                       "--role coordinator must not use --coordinator");
            return 0;
        }
        tp->role = DS4_TP_LEADER;
        tp->listen_host = dist->listen_host;
        tp->listen_port = dist->listen_port;
    } else if (dist->role == DS4_DISTRIBUTED_WORKER) {
        if (!dist->coordinator_host || dist->coordinator_port <= 0) {
            tp_set_err(err, errlen,
                       "--role worker --tensor-parallel requires --coordinator HOST PORT");
            return 0;
        }
        if (dist->listen_host || dist->listen_port) {
            tp_set_err(err, errlen,
                       "--role worker --tensor-parallel must not use --listen");
            return 0;
        }
        tp->role = DS4_TP_WORKER;
        tp->leader_host = dist->coordinator_host;
        tp->leader_port = dist->coordinator_port;
    } else {
        tp_set_err(err, errlen, "invalid tensor-parallel role");
        return 0;
    }

    if (!tp->nhi_device && dist->nhi_device)
        tp->nhi_device = dist->nhi_device;
    memset(dist, 0, sizeof(*dist));
    return 1;
}

int ds4_tp_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen)
{
    if (!ds4_tp_enabled(&opt->tp)) {
        if (opt->tp.requested || opt->tp.transport != DS4_TP_TRANSPORT_AUTO ||
            opt->tp.rdma_device || opt->tp.nhi_device ||
            opt->tp.nhi_ring_frames != 0 || opt->tp.rdma_gid_index_set ||
            opt->tp.glm_token_prefill || opt->tp.debug_hash != 0) {
            tp_set_err(err, errlen,
                       "tensor-parallel options require --tensor-parallel and --role");
            return 0;
        }
        return 1;
    }
    bool backend_ok = opt->backend == DS4_BACKEND_METAL &&
                      opt->tp.transport != DS4_TP_TRANSPORT_NHI;
#ifdef DS4_ROCM_BUILD
    backend_ok = backend_ok ||
                 (opt->backend == DS4_BACKEND_CUDA &&
                  opt->tp.transport == DS4_TP_TRANSPORT_NHI);
#endif
    if (!backend_ok) {
        tp_set_err(err, errlen,
#ifdef DS4_ROCM_BUILD
                   "ROCm tensor parallelism requires --transport nhi"
#else
                   "tensor parallelism requires the Metal backend"
#endif
        );
        return 0;
    }
    if (opt->tp.transport == DS4_TP_TRANSPORT_NHI && !opt->tp.nhi_device) {
        tp_set_err(err, errlen,
                   "--transport nhi requires --nhi-device PATH");
        return 0;
    }
    if (opt->tp.transport != DS4_TP_TRANSPORT_NHI &&
        (opt->tp.nhi_device || opt->tp.nhi_ring_frames != 0)) {
        tp_set_err(err, errlen,
                   "--nhi-device/--nhi-ring-frames require --transport nhi");
        return 0;
    }
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE) {
        tp_set_err(err, errlen, "tensor parallelism and --role distributed modes are exclusive");
        return 0;
    }
    /* Speculative drafting (DSpark/MTP) is allowed on the leader: the
     * verify block is mirrored to the worker via DS4_TP_FRAME_VERIFY and
     * the legacy MTP path falls back to per-token decode under TP. */
    if (opt->load_slice) {
        tp_set_err(err, errlen, "tensor parallelism does not use distributed layer slices");
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Slab layout.
 * --------------------------------------------------------------------- */

uint64_t ds4_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd) {
    uint64_t vec = (uint64_t)n_embd * sizeof(float);
    uint64_t slots = (uint64_t)n_layer * DS4_TP_GATES_PER_LAYER;
    return slots * vec * 2 +    /* out + in vectors */
           slots * 8 * 2 +      /* in flags + out flag staging */
           16 +                 /* token slot */
           slots * 4 +          /* GPU-written gate-ready flags */
           (uint64_t)n_layer * DS4_TP_BATCH_MAX_ROWS * vec * 2; /* batch out+in */
}

static void tp_slab_layout(ds4_tp *tp) {
    uint64_t vec = tp->vec_bytes;
    uint64_t slots = tp->n_slots;
    tp->out_off = 0;
    tp->in_off = slots * vec;
    tp->in_flags_off = tp->in_off + slots * vec;
    tp->token_off = tp->in_flags_off + slots * 8;
    tp->out_flags_off = tp->token_off + 16;
    tp->gpu_flags_off = tp->out_flags_off + slots * 8;
    tp->batch_out_off = tp->gpu_flags_off + slots * 4;
    tp->batch_in_off = tp->batch_out_off +
                       (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * vec;
    tp->slab_bytes = tp->batch_in_off +
                     (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * vec;
}

uint64_t ds4_tp_slab_gpu_flags_offset(const ds4_tp *tp) {
    return tp->gpu_flags_off;
}

static uint32_t tp_slot(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    (void)tp;
    return layer * DS4_TP_GATES_PER_LAYER + gate;
}

uint64_t ds4_tp_slab_out_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    return tp->out_off + (uint64_t)tp_slot(tp, layer, gate) * tp->vec_bytes;
}

uint64_t ds4_tp_slab_in_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    return tp->in_off + (uint64_t)tp_slot(tp, layer, gate) * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_out_offset(const ds4_tp *tp, uint32_t layer) {
    return tp->batch_out_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_in_offset(const ds4_tp *tp, uint32_t layer) {
    return tp->batch_in_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

static uint32_t tp_gate_mask_count(
        const uint64_t mask[DS4_TP_GATE_MASK_WORDS]) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < DS4_TP_GATE_MASK_WORDS; i++)
        count += (uint32_t)__builtin_popcountll(mask[i]);
    return count;
}

static int tp_gate_mask_fits(
        const uint64_t mask[DS4_TP_GATE_MASK_WORDS],
        uint32_t n_slots) {
    for (uint32_t word = 0; word < DS4_TP_GATE_MASK_WORDS; word++) {
        uint64_t bits = mask[word];
        while (bits) {
            const uint32_t slot =
                word * 64u + (uint32_t)__builtin_ctzll(bits);
            if (slot >= n_slots) return 0;
            bits &= bits - 1u;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * RDMA path.
 * --------------------------------------------------------------------- */

#ifdef DS4_TP_HAVE_VERBS

static int tp_rdma_load_api(ds4_tp_verbs_api *api) {
    if (api->handle) return 1;
    void *h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
#define TP_SYM(field, name) \
    do { \
        api->field = (__typeof__(api->field))dlsym(h, name); \
        if (!api->field) { dlclose(h); return 0; } \
    } while (0)
    TP_SYM(get_device_list, "ibv_get_device_list");
    TP_SYM(free_device_list, "ibv_free_device_list");
    TP_SYM(get_device_name, "ibv_get_device_name");
    TP_SYM(open_device, "ibv_open_device");
    TP_SYM(close_device, "ibv_close_device");
    TP_SYM(query_device, "ibv_query_device");
    TP_SYM(query_port, "ibv_query_port");
    TP_SYM(query_gid, "ibv_query_gid");
    TP_SYM(alloc_pd, "ibv_alloc_pd");
    TP_SYM(dealloc_pd, "ibv_dealloc_pd");
    TP_SYM(reg_mr, "ibv_reg_mr");
    TP_SYM(dereg_mr, "ibv_dereg_mr");
    TP_SYM(create_cq, "ibv_create_cq");
    TP_SYM(destroy_cq, "ibv_destroy_cq");
    TP_SYM(create_qp, "ibv_create_qp");
    TP_SYM(destroy_qp, "ibv_destroy_qp");
    TP_SYM(modify_qp, "ibv_modify_qp");
    TP_SYM(query_qp, "ibv_query_qp");
#undef TP_SYM
    api->handle = h;
    return 1;
}

/* Probe only: does this machine expose a verbs device right now? */
static int tp_rdma_probe(ds4_tp_verbs_api *api) {
    if (!tp_rdma_load_api(api)) return 0;
    int num = 0;
    struct ibv_device **devs = api->get_device_list(&num);
    if (!devs) return 0;
    api->free_device_list(devs);
    return num > 0;
}

static int tp_rdma_open(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    int num = 0;
    struct ibv_device **devs = r->api.get_device_list(&num);
    if (!devs || num == 0) {
        tp_set_err(err, errlen, "tp rdma: no verbs devices");
        if (devs) r->api.free_device_list(devs);
        return 0;
    }
    /* One verbs device per Thunderbolt port (rdma_enN); pick the active one
     * unless the caller selected a device explicitly. */
    const char *want_name = tp->opt.rdma_device;
    char states[256] = "";
    for (int i = 0; i < num && !r->ctx; i++) {
        const char *name = r->api.get_device_name(devs[i]);
        if (want_name && strcmp(want_name, name) != 0) continue;
        struct ibv_context *ctx = r->api.open_device(devs[i]);
        if (!ctx) continue;
        struct ibv_port_attr pa;
        if (r->api.query_port(ctx, 1, &pa) == 0 &&
            (pa.state == IBV_PORT_ACTIVE || want_name)) {
            r->ctx = ctx;
            r->port = pa;
            fprintf(stderr, "ds4-tp: rdma device %s (port state %d)\n", name, (int)pa.state);
            break;
        }
        size_t off = strlen(states);
        snprintf(states + off, sizeof(states) - off, "%s%s=%d",
                 off ? ", " : "", name, (int)pa.state);
        r->api.close_device(ctx);
    }
    r->api.free_device_list(devs);
    if (!r->ctx) {
        tp_set_err(err, errlen,
                   "tp rdma: no device with an active port (%s); is the peer up "
                   "and rdma_ctl enabled on both machines?", states);
        return 0;
    }
    /* The driver only connects through the IPv4-mapped GID
     * (::ffff:a.b.c.d), which exists only when the Thunderbolt member
     * interface carries an IPv4 address (the bridge's address does not
     * count). */
    r->gid_index = -1;
    if (tp->opt.rdma_gid_index_set) {
        r->gid_index = tp->opt.rdma_gid_index;
        if (r->api.query_gid(r->ctx, 1, r->gid_index, &r->gid) != 0) {
            tp_set_err(err, errlen, "tp rdma: query_gid(%d): %s",
                       r->gid_index, strerror(errno));
            return 0;
        }
    } else {
        for (int i = 0; i < r->port.gid_tbl_len; i++) {
            union ibv_gid tmp;
            if (r->api.query_gid(r->ctx, 1, i, &tmp) != 0) continue;
            uint64_t hi;
            uint16_t mid, v4tag;
            memcpy(&hi, &tmp.raw[0], 8);
            memcpy(&mid, &tmp.raw[8], 2);
            memcpy(&v4tag, &tmp.raw[10], 2);
            if (hi == 0 && mid == 0 && v4tag == 0xffff) {
                r->gid = tmp;
                r->gid_index = i;
                break;
            }
        }
        if (r->gid_index < 0) {
            tp_set_err(err, errlen,
                       "tp rdma: no IPv4-mapped GID on the active port; give the "
                       "Thunderbolt interface its own IPv4 (e.g. sudo ifconfig en1 "
                       "inet 10.99.0.2/30 alias) on both machines");
            return 0;
        }
    }
    r->pd = r->api.alloc_pd(r->ctx);
    if (!r->pd) {
        tp_set_err(err, errlen, "tp rdma: alloc_pd failed");
        return 0;
    }
    {
        struct ibv_device_attr da;
        memset(&da, 0, sizeof(da));
        if (r->api.query_device && r->api.query_device(r->ctx, &da) == 0) {
            fprintf(stderr, "ds4-tp: rdma device limits: max_qp_wr %d, max_sge %d, max_cqe %d, max_mr_size %llu\n",
                    da.max_qp_wr, da.max_sge, da.max_cqe, (unsigned long long)da.max_mr_size);
        }
    }
    r->cq = r->api.create_cq(r->ctx, 512, NULL, NULL, 0);
    if (!r->cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq failed");
        return 0;
    }
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = r->cq;
    qia.recv_cq = r->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr = 1024;
    qia.cap.max_recv_wr = 1024;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp) {
        qia.cap.max_send_wr = 256;
        qia.cap.max_recv_wr = 64;
        r->qp = r->api.create_qp(r->pd, &qia);
    }
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC): %s", strerror(errno));
        return 0;
    }
    r->max_inline = qia.cap.max_inline_data;
    r->recv_depth = qia.cap.max_recv_wr ? qia.cap.max_recv_wr : 64u;
    r->send_depth = qia.cap.max_send_wr ? qia.cap.max_send_wr : 256u;
    if (r->recv_depth > 4096u) r->recv_depth = 4096u;
    if (r->send_depth > 4096u) r->send_depth = 4096u;

    pthread_mutex_init(&r->post_lock, NULL);
    return 1;
}

#define DS4_TP_RDMA_WARM_WR_TAG (UINT64_C(1) << 62)

static const char *tp_wc_status_str(int status);

/* Warm-up round after the ready barrier.  A fresh UC queue pair over
 * Thunderbolt RDMA sometimes cannot deliver in one direction at all, and UC
 * never reports a lost message: such runs timed out at the first bulk round
 * with no completions.  This round proves both directions before any real
 * traffic (the caller recreates the queue pair when it fails).  Each side posts exactly one full-size receive and resends
 * a tagged message until the peer confirms receipt over the control
 * channel.  A dropped UC message never arrives late, so once both sides
 * have received, no stray message can reach the receive queue. */
/* "Receives posted" barrier.  On Apple's Thunderbolt RDMA a UC send that
 * reaches a queue pair with no receive posted is not just lost: that
 * direction of the pair stops delivering for good.  Every sender therefore
 * waits for the peer's word that its receives are posted before posting
 * sends (one control-channel round trip). */
static int tp_rdma_posted_barrier(ds4_tp *tp, uint32_t tag) {
    uint32_t t = 0, b = 0, theirs = 0;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_POSTED, &tag, sizeof(tag))) return 0;
    if (!tp_read_frame_header(tp->control_fd, &t, &b) ||
        t != DS4_TP_FRAME_RDMA_POSTED || b != sizeof(theirs) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs)) || theirs != tag) {
        return 0;
    }
    return 1;
}

static int tp_rdma_warm_up(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    uint8_t *wsend = tp->slab + tp->batch_out_off;
    uint8_t *wrecv = tp->slab + tp->batch_in_off;
    /* Full-size message: 64 B warm messages pass on a fresh direction while
     * the first 16 KB chunks are still dropped. */
    const uint32_t bytes = DS4_TP_RDMA_MAX_MSG;
    memset(wsend, 0x5A, bytes);
    memset(wrecv, 0, bytes);
    struct ibv_sge rsge = {
        .addr = (uintptr_t)wrecv, .length = bytes, .lkey = r->mr->lkey,
    };
    struct ibv_recv_wr rwr;
    memset(&rwr, 0, sizeof(rwr));
    rwr.wr_id = DS4_TP_RDMA_WARM_WR_TAG;
    rwr.sg_list = &rsge;
    rwr.num_sge = 1;
    struct ibv_recv_wr *bad_r = NULL;
    if (ibv_post_recv(r->qp, &rwr, &bad_r) != 0) {
        tp_set_err(err, errlen, "tp rdma: warm-up post_recv: %s", strerror(errno));
        return 0;
    }
    if (!tp_rdma_posted_barrier(tp, 0xfeedu)) {
        tp_set_err(err, errlen, "tp rdma: warm-up posted barrier failed");
        return 0;
    }
    int got_recv = 0, peer_got = 0, attempts = 0;
    for (int attempt = 0; attempt < 20 && !(got_recv && peer_got); attempt++) {
        attempts = attempt + 1;
        int send_done = peer_got;
        if (!peer_got) {
            struct ibv_sge ssge = {
                .addr = (uintptr_t)wsend, .length = bytes, .lkey = r->mr->lkey,
            };
            struct ibv_send_wr swr;
            memset(&swr, 0, sizeof(swr));
            swr.wr_id = DS4_TP_RDMA_WARM_WR_TAG;
            swr.sg_list = &ssge;
            swr.num_sge = 1;
            swr.opcode = IBV_WR_SEND;
            swr.send_flags = IBV_SEND_SIGNALED;
            struct ibv_send_wr *bad_s = NULL;
            if (ibv_post_send(r->qp, &swr, &bad_s) != 0) {
                tp_set_err(err, errlen, "tp rdma: warm-up post_send: %s", strerror(errno));
                return 0;
            }
        }
        const double deadline = tp_now_sec() + 0.1;
        while (tp_now_sec() < deadline && !(send_done && got_recv)) {
            struct ibv_wc wc[8];
            int n = ibv_poll_cq(r->cq, 8, wc);
            if (n < 0) {
                tp_set_err(err, errlen, "tp rdma: warm-up poll_cq failed");
                return 0;
            }
            for (int i = 0; i < n; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    tp_set_err(err, errlen, "tp rdma: warm-up completion error: %s",
                               tp_wc_status_str(wc[i].status));
                    return 0;
                }
                if (wc[i].wr_id != DS4_TP_RDMA_WARM_WR_TAG) continue;
                if (wc[i].opcode & IBV_WC_RECV) got_recv = 1;
                else send_done = 1;
            }
        }
        uint32_t mine = (uint32_t)got_recv, theirs = 0, t = 0, b = 0;
        if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_WARM, &mine, sizeof(mine)) ||
            !tp_read_frame_header(tp->control_fd, &t, &b) ||
            t != DS4_TP_FRAME_RDMA_WARM || b != sizeof(theirs) ||
            !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
            tp_set_err(err, errlen, "tp rdma: warm-up status exchange failed");
            return 0;
        }
        peer_got = theirs != 0;
    }
    if (!(got_recv && peer_got)) {
        tp_set_err(err, errlen, "tp rdma: warm-up failed after %d attempts (recv %d, peer %d)",
                   attempts, got_recv, peer_got);
        r->warm_failed = 1;
        return 0;
    }
    fprintf(stderr, "ds4-tp: rdma warm-up ok (%d attempt%s), queue depth recv %u send %u\n",
            attempts, attempts == 1 ? "" : "s", r->recv_depth, r->send_depth);
    return 1;
}

static int tp_rdma_post_gate_recv(ds4_tp *tp, uint64_t seq);

/* A freshly created UC queue pair over Thunderbolt RDMA sometimes cannot
 * deliver in one direction at all (every send silently dropped, observed
 * right after a working run and after role swaps).  Recreating the pair
 * (new queue pair number) and re-exchanging the info fixes it; both sides
 * take this path in lockstep because the warm-up status exchange tells
 * them the same outcome. */
static int tp_rdma_recreate_qp(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    if (r->qp) { r->api.destroy_qp(r->qp); r->qp = NULL; }
    if (r->cq) { r->api.destroy_cq(r->cq); r->cq = NULL; }
    r->cq = r->api.create_cq(r->ctx, 512, NULL, NULL, 0);
    if (!r->cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq (retry) failed");
        return 0;
    }
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = r->cq;
    qia.recv_cq = r->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr = 1024;
    qia.cap.max_recv_wr = 1024;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp) {
        qia.cap.max_send_wr = 256;
        qia.cap.max_recv_wr = 64;
        r->qp = r->api.create_qp(r->pd, &qia);
    }
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp (retry): %s", strerror(errno));
        return 0;
    }
    r->max_inline = qia.cap.max_inline_data;
    r->recv_depth = qia.cap.max_recv_wr ? qia.cap.max_recv_wr : 64u;
    r->send_depth = qia.cap.max_send_wr ? qia.cap.max_send_wr : 256u;
    if (r->recv_depth > 4096u) r->recv_depth = 4096u;
    if (r->send_depth > 4096u) r->send_depth = 4096u;
    r->send_outstanding = 0;
    r->recv_done = 0;
    return 1;
}

static int tp_rdma_register_and_exchange(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!r->mr) {
        r->mr = r->api.reg_mr(r->pd, tp->slab, tp->slab_bytes,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_WRITE);
    }
    if (!r->mr) {
        tp_set_err(err, errlen, "tp rdma: reg_mr(%llu bytes): %s",
                   (unsigned long long)tp->slab_bytes, strerror(errno));
        return 0;
    }
    ds4_tp_rdma_info mine = {0};
    mine.slab_base = (uint64_t)(uintptr_t)tp->slab;
    mine.rkey = r->mr->rkey;
    mine.qpn = r->qp->qp_num;
    mine.psn = ((uint32_t)(getpid() ^ (uintptr_t)tp) + r->setup_attempt * 0x1000u) & 0xffffff;
    mine.mtu = (uint32_t)r->port.active_mtu;
    mine.lid = r->port.lid;
    memcpy(mine.gid, r->gid.raw, 16);
    mine.link_layer = r->port.link_layer;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_INFO, &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp rdma: info send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_RDMA_INFO || bytes != sizeof(r->peer) ||
        !tp_read_full(tp->control_fd, &r->peer, sizeof(r->peer))) {
        tp_set_err(err, errlen, "tp rdma: info recv failed");
        return 0;
    }

    /* INIT -> RTR -> RTS with the exact recipe the driver accepts (same as
     * JACCL): MTU 1024 and GRH via the IPv4-mapped GID. */
    struct ibv_qp_attr a = {0};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                        IBV_ACCESS_REMOTE_WRITE;
    if (r->api.modify_qp(r->qp, &a,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify INIT: %s", strerror(errno));
        return 0;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_1024;
    a.dest_qp_num = r->peer.qpn;
    a.rq_psn = r->peer.psn;
    a.ah_attr.dlid = (uint16_t)r->peer.lid;
    a.ah_attr.port_num = 1;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, r->peer.gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (r->api.modify_qp(r->qp, &a,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTR: %s", strerror(errno));
        return 0;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = mine.psn;
    if (r->api.modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTS: %s", strerror(errno));
        return 0;
    }
    if (tp->vec_bytes > 2ull * DS4_TP_RDMA_MAX_MSG) {
        tp_set_err(err, errlen,
                   "tp rdma: gate vector %llu bytes exceeds twice the driver's "
                   "%u message limit",
                   (unsigned long long)tp->vec_bytes, DS4_TP_RDMA_MAX_MSG);
        return 0;
    }
    if (tp->vec_bytes > DS4_TP_RDMA_MAX_MSG) {
        fprintf(stderr,
                "ds4-tp: rdma gate vectors ride as 2 chunked messages "
                "(%llu bytes > %u limit)\n",
                (unsigned long long)tp->vec_bytes, DS4_TP_RDMA_MAX_MSG);
    }
    /* Leave the receive queue empty for an initial bulk prefill.  The first
     * decode gate arms the normal lookahead window after prefill finishes. */
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_RDMA_READY, NULL, 0)) {
        tp_set_err(err, errlen, "tp rdma: ready send failed");
        return 0;
    }
    uint32_t rtype = 0, rbytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &rtype, &rbytes) ||
        rtype != DS4_TP_FRAME_RDMA_READY || rbytes != 0) {
        tp_set_err(err, errlen, "tp rdma: ready barrier failed");
        return 0;
    }
    if (getenv("DS4_TP_DISABLE_RDMA_WARMUP") == NULL &&
        !tp_rdma_warm_up(tp, err, errlen)) return 0;
    return 1;
}

/* ibv_wc_status_str lives in librdma; resolve lazily to keep the dlopen-only
 * linkage discipline. */
static const char *tp_wc_status_str(int status) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "wc status %d", status);
    return buf;
}

/* Slab slot a given gate seq lands in.  DS4 fires every slot in order
 * (identity mapping); GLM's schedule from the hello skips dense layers
 * and the ATTN slots. */
static uint32_t tp_gate_slot(const ds4_tp *tp, uint64_t seq) {
    if (tp->gate_slot_mask[0] || tp->gate_slot_mask[1] ||
        tp->gate_slot_mask[2]) {
        uint32_t ordinal = (uint32_t)((seq - 1) % tp->gates_per_token);
        for (uint32_t word = 0; word < DS4_TP_GATE_MASK_WORDS; word++) {
            uint64_t bits = tp->gate_slot_mask[word];
            const uint32_t count = (uint32_t)__builtin_popcountll(bits);
            if (ordinal >= count) {
                ordinal -= count;
                continue;
            }
            while (ordinal > 0) {
                bits &= bits - 1u;
                ordinal--;
            }
            return word * 64u + (uint32_t)__builtin_ctzll(bits);
        }
        return tp->n_slots;
    }
    if (tp->gates_per_token == 0)
        return (uint32_t)((seq - 1) % tp->n_slots);
    return tp->gate_slot_start +
           (uint32_t)((seq - 1) % tp->gates_per_token) * tp->gate_slot_step;
}

/* Reap completions: send CQEs free send-queue slots, recv CQEs advance the
 * arrival watermark (UC is in-order, so gate seq recv completions arrive
 * monotonically).  Returns 0 on any completion error. */
static int tp_rdma_drain_cq(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
    struct ibv_wc wc[16];
    int n = ibv_poll_cq(r->cq, 16, wc);
    if (n < 0) return 0;
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "ds4-tp: rdma completion error: %s (wr_id %llu)\n",
                    tp_wc_status_str(wc[i].status),
                    (unsigned long long)wc[i].wr_id);
            return 0;
        }
        if (wc[i].opcode & IBV_WC_RECV) {
            if (wc[i].wr_id & DS4_TP_RDMA_BLOCK_WR_TAG) {
                if (wc[i].byte_len == 0) {
                    fprintf(stderr, "ds4-tp: rdma verify-block receive of 0 bytes\n");
                    return 0;
                }
                r->block_recv_done++;
            } else if (wc[i].wr_id > r->recv_done) {
                r->recv_done = wc[i].wr_id;
            }
        } else if (r->send_outstanding > 0) {
            r->send_outstanding--;
        }
    }
    return 1;
}

/* Verify-block window helpers. */
static int tp_rdma_block_post_layer(ds4_tp *tp, uint32_t layer) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t rows = r->block_rows;
    const uint32_t chunks = (uint32_t)((tp->vec_bytes + DS4_TP_RDMA_MAX_MSG - 1u) /
                                       DS4_TP_RDMA_MAX_MSG);
    const uint32_t n = rows * chunks;
    if (n == 0 || n > r->recv_depth) return 0;
    struct ibv_sge *sge = r->win_sge;
    struct ibv_recv_wr *wr = r->win_rwr;
    memset(wr, 0, (size_t)n * sizeof(*wr));
    const uintptr_t base =
        (uintptr_t)(tp->slab + ds4_tp_slab_batch_in_offset(tp, layer));
    uint32_t wi = 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint64_t off = 0; off < tp->vec_bytes; ) {
            const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
                DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
            const int last = off + len == tp->vec_bytes;
            sge[wi].addr = base + (uintptr_t)row * tp->vec_bytes + off;
            sge[wi].length = (uint32_t)len;
            sge[wi].lkey = r->mr->lkey;
            wr[wi].wr_id = last ?
                (DS4_TP_RDMA_BLOCK_WR_TAG | ((uint64_t)layer * rows + row + 1u)) : 0;
            wr[wi].sg_list = &sge[wi];
            wr[wi].num_sge = 1;
            if (wi != 0) wr[wi - 1u].next = &wr[wi];
            wi++;
            off += len;
        }
    }
    struct ibv_recv_wr *bad = NULL;
    if (ibv_post_recv(r->qp, wr, &bad) != 0) {
        fprintf(stderr, "ds4-tp: rdma verify-block post_recv(layer %u): %s\n",
                layer, strerror(errno));
        return 0;
    }
    r->block_posted = layer + 1u;
    return 1;
}

static int tp_rdma_block_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!r->block_active || rows != r->block_rows || layer >= r->block_layers ||
        layer + 1u != r->block_posted) {
        fprintf(stderr, "ds4-tp: verify-block gate out of order (layer %u rows %u, posted %u/%u rows %u)\n",
                layer, rows, r->block_posted, r->block_layers, r->block_rows);
        return 0;
    }
    pthread_mutex_lock(&r->post_lock);
    int ok = 1;
    /* Post the next layer's receives BEFORE this layer's sends: the peer
     * can only send layer+1 after it has received this send, so its send
     * never reaches an unposted queue. */
    if (layer + 1u < r->block_layers) ok = tp_rdma_block_post_layer(tp, layer + 1u);
    const uint32_t chunks = (uint32_t)((tp->vec_bytes + DS4_TP_RDMA_MAX_MSG - 1u) /
                                       DS4_TP_RDMA_MAX_MSG);
    const uint32_t n = rows * chunks;
    if (ok) {
        struct ibv_sge *sge = r->win_sge + r->recv_depth / 2u;   /* separate half of the arrays */
        struct ibv_send_wr *wr = r->win_swr;
        if (n > r->recv_depth / 2u || n > r->send_depth) ok = 0;
        if (ok) {
            memset(wr, 0, (size_t)n * sizeof(*wr));
            const uintptr_t base =
                (uintptr_t)(tp->slab + ds4_tp_slab_batch_out_offset(tp, layer));
            uint32_t wi = 0;
            for (uint32_t row = 0; row < rows; row++) {
                for (uint64_t off = 0; off < tp->vec_bytes; ) {
                    const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
                        DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
                    sge[wi].addr = base + (uintptr_t)row * tp->vec_bytes + off;
                    sge[wi].length = (uint32_t)len;
                    sge[wi].lkey = r->mr->lkey;
                    wr[wi].wr_id = DS4_TP_RDMA_BLOCK_WR_TAG | 1u;
                    wr[wi].sg_list = &sge[wi];
                    wr[wi].num_sge = 1;
                    wr[wi].opcode = IBV_WR_SEND;
                    wr[wi].send_flags = wi + 1u == n ? IBV_SEND_SIGNALED : 0;
                    if (wi != 0) wr[wi - 1u].next = &wr[wi];
                    wi++;
                    off += len;
                }
            }
            struct ibv_send_wr *bad = NULL;
            if (ibv_post_send(r->qp, wr, &bad) != 0) {
                fprintf(stderr, "ds4-tp: rdma verify-block post_send(layer %u): %s\n",
                        layer, strerror(errno));
                ok = 0;
            } else {
                r->send_outstanding++;
            }
        }
    }
    pthread_mutex_unlock(&r->post_lock);
    const uint64_t want = (uint64_t)(layer + 1u) * rows;
    double deadline = 0.0;
    uint32_t peer_poll = 0;
    while (ok && r->block_recv_done < want) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && (++peer_poll & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr, "ds4-tp: peer disconnected during verify-block gate\n");
            ok = 0;
        }
        if (deadline == 0.0) deadline = tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0;
        else if (tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: timeout waiting verify-block gate layer %u (%llu/%llu rows)\n",
                    layer, (unsigned long long)r->block_recv_done, (unsigned long long)want);
            ok = 0;
        }
    }
    return ok;
}

/* Arm the receive for gate seq: UC delivery order pairs the peer's seq'th
 * send with our seq'th posted recv, landing it in the in-slot the combine
 * kernel reads. */
static int tp_rdma_post_gate_recv(ds4_tp *tp, uint64_t seq) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t slot = tp_gate_slot(tp, seq);
    const uintptr_t base =
        (uintptr_t)(tp->slab + tp->in_off + (uint64_t)slot * tp->vec_bytes);
    /* Vectors above the driver's 16KB message cap ride as two chunks
     * landing contiguously in the slot. UC delivery is in-order and both
     * sides post/send strictly in seq order, so the k'th send always
     * matches the k'th recv; only the FINAL chunk carries the seq as
     * wr_id, so the arrival watermark advances when the slot is whole. */
    struct ibv_sge sge[2];
    struct ibv_recv_wr wr[2];
    memset(wr, 0, sizeof(wr));
    uint32_t count = 0;
    for (uint64_t off = 0; off < tp->vec_bytes; count++) {
        const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
            DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        const int last = off + len == tp->vec_bytes;
        sge[count].addr = base + off;
        sge[count].length = (uint32_t)len;
        sge[count].lkey = r->mr->lkey;
        wr[count].wr_id = last ? seq : 0;
        wr[count].sg_list = &sge[count];
        wr[count].num_sge = 1;
        if (count != 0) wr[count - 1u].next = &wr[count];
        off += len;
    }
    struct ibv_recv_wr *bad = NULL;
    if (ibv_post_recv(r->qp, wr, &bad) != 0) {
        fprintf(stderr, "ds4-tp: rdma post_recv(seq %llu): %s\n",
                (unsigned long long)seq, strerror(errno));
        return 0;
    }
    return 1;
}

/* One decode gate: ensure the receive window is armed, send our partial,
 * wait for the peer's receive completion, and advance the window. */
/* Per-gate-kind RDMA timing (DS4_TP_GATE_PROFILE): post cost and the wait
 * for the peer's partial, printed every 860 gates. */
static double g_rdma_stat_post_us[2];
static double g_rdma_stat_wait_us[2];
static uint64_t g_rdma_stat_count[2];
static int g_rdma_stat_enabled = -1;

static int tp_rdma_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t slot = layer * DS4_TP_GATES_PER_LAYER + gate;
    if (g_rdma_stat_enabled < 0) g_rdma_stat_enabled = getenv("DS4_TP_GATE_PROFILE") != NULL;
    const double st0 = g_rdma_stat_enabled ? tp_now_sec() : 0.0;
    double st1 = 0.0;
    if (getenv("DS4_TP_GATE_TRACE")) {
        fprintf(stderr, "ds4-tp: gate trace l=%u g=%u seq=%llu want_slot=%u\n",
                layer, gate, (unsigned long long)seq, tp_gate_slot(tp, seq));
    }
    if (slot != tp_gate_slot(tp, seq)) {
        fprintf(stderr, "ds4-tp: gate order broke: layer %u gate %u vs seq %llu\n",
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    const uintptr_t send_base =
        (uintptr_t)(tp->slab + tp->out_off + (uint64_t)slot * tp->vec_bytes);
    pthread_mutex_lock(&r->post_lock);
    int ok = 1;
    if (!r->recv_window_active) {
        for (uint64_t s = seq; ok && s < seq + DS4_TP_RDMA_RECV_WINDOW; s++)
            ok = tp_rdma_post_gate_recv(tp, s);
        if (ok) r->recv_window_active = true;
    }
    struct ibv_sge send_sge[2];
    struct ibv_send_wr send_wr[2];
    memset(send_wr, 0, sizeof(send_wr));
    uint32_t send_count = 0;
    for (uint64_t off = 0; ok && off < tp->vec_bytes; send_count++) {
        const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
            DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        send_sge[send_count].addr = send_base + off;
        send_sge[send_count].length = (uint32_t)len;
        send_sge[send_count].lkey = r->mr->lkey;
        send_wr[send_count].wr_id = seq;
        send_wr[send_count].sg_list = &send_sge[send_count];
        send_wr[send_count].num_sge = 1;
        send_wr[send_count].opcode = IBV_WR_SEND;
        if (send_count != 0) send_wr[send_count - 1u].next = &send_wr[send_count];
        off += len;
    }
    if (ok) {
        send_wr[send_count - 1u].send_flags = IBV_SEND_SIGNALED;
        struct ibv_send_wr *bad = NULL;
        ok = ibv_post_send(r->qp, send_wr, &bad) == 0;
        if (!ok) {
            fprintf(stderr, "ds4-tp: rdma post_send: %s\n", strerror(errno));
        } else {
            r->send_outstanding++;
        }
    }
    if (g_rdma_stat_enabled) st1 = tp_now_sec();

    double deadline = 0.0;
    uint32_t peer_poll = 0;
    while (ok && r->recv_done < seq) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && (++peer_poll & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr, "ds4-tp: peer disconnected during RDMA gate\n");
            ok = 0;
        }
        if (deadline == 0.0) {
            deadline = tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0;
        }
        else if (tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: timeout waiting gate seq %llu (recv_done %llu)\n",
                    (unsigned long long)seq, (unsigned long long)r->recv_done);
            ok = 0;
        }
    }
    if (g_rdma_stat_enabled && ok && gate < 2u) {
        const double st2 = tp_now_sec();
        g_rdma_stat_post_us[gate] += (st1 - st0) * 1e6;
        g_rdma_stat_wait_us[gate] += (st2 - st1) * 1e6;
        if (++g_rdma_stat_count[gate] % 430 == 0) {
            fprintf(stderr, "ds4-tp: rdma gate %u: post %.1f us, peer wait %.1f us (n=%llu)\n",
                    gate, g_rdma_stat_post_us[gate] / (double)g_rdma_stat_count[gate],
                    g_rdma_stat_wait_us[gate] / (double)g_rdma_stat_count[gate],
                    (unsigned long long)g_rdma_stat_count[gate]);
        }
    }
    if (ok) ok = tp_rdma_post_gate_recv(tp, seq + DS4_TP_RDMA_RECV_WINDOW);
    if (ok) r->last_gate_seq = seq;
    pthread_mutex_unlock(&r->post_lock);
    return ok;
}

static int tp_rdma_big_gate_capable(const ds4_tp *tp) {
    const uint64_t stage_bytes =
        (uint64_t)DS4_TP_RDMA_BULK_SLOTS * DS4_TP_RDMA_MAX_MSG;
    const uint64_t batch_region_bytes =
        (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
    return tp->rdma.qp && tp->rdma.mr && batch_region_bytes >= stage_bytes;
}

/* Decode keeps a lookahead window of receives on the latency QP. Before a
 * later prompt can reuse that QP for bulk rows, consume those receives with
 * dummy sends on both ranks. The TCP big-gate header exchange is the barrier
 * that guarantees both sides have reached this transition. */
static int tp_rdma_drain_decode_window(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!r->recv_window_active) return 1;

    const uint32_t chunks_per_gate =
        (uint32_t)((tp->vec_bytes + DS4_TP_RDMA_MAX_MSG - 1u) /
                   DS4_TP_RDMA_MAX_MSG);
    const uint32_t nwr = DS4_TP_RDMA_RECV_WINDOW * chunks_per_gate;
    struct ibv_sge sge[DS4_TP_RDMA_RECV_WINDOW * 2u];
    struct ibv_send_wr wr[DS4_TP_RDMA_RECV_WINDOW * 2u];
    memset(wr, 0, sizeof(wr));
    uint8_t *scratch = tp->slab + tp->batch_out_off;
    uint32_t wi = 0;
    for (uint32_t gate = 0; gate < DS4_TP_RDMA_RECV_WINDOW; gate++) {
        for (uint64_t off = 0; off < tp->vec_bytes; ) {
            const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
                DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
            sge[wi] = (struct ibv_sge) {
                .addr = (uintptr_t)(scratch + off),
                .length = (uint32_t)len,
                .lkey = r->mr->lkey,
            };
            wr[wi].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)wi + 1u);
            wr[wi].sg_list = &sge[wi];
            wr[wi].num_sge = 1;
            wr[wi].opcode = IBV_WR_SEND;
            wr[wi].send_flags = wi + 1u == nwr ? IBV_SEND_SIGNALED : 0;
            if (wi > 0) wr[wi - 1u].next = &wr[wi];
            wi++;
            off += len;
        }
    }

    pthread_mutex_lock(&r->post_lock);
    struct ibv_send_wr *bad = NULL;
    if (ibv_post_send(r->qp, wr, &bad) != 0) {
        fprintf(stderr, "ds4-tp: rdma receive-window drain post failed: %s\n",
                strerror(errno));
        pthread_mutex_unlock(&r->post_lock);
        return 0;
    }

    uint32_t recv_done = 0;
    int send_done = 0;
    const double deadline =
        tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0;
    uint32_t peer_poll = 0;
    while (recv_done < nwr || !send_done) {
        struct ibv_wc wc[DS4_TP_RDMA_RECV_WINDOW * 2u + 1u];
        int n = ibv_poll_cq(r->cq,
                           (int)(DS4_TP_RDMA_RECV_WINDOW * 2u + 1u), wc);
        if (n < 0) {
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "ds4-tp: rdma receive-window drain: %s\n",
                        tp_wc_status_str(wc[i].status));
                pthread_mutex_unlock(&r->post_lock);
                return 0;
            }
            if (wc[i].opcode & IBV_WC_RECV) {
                recv_done++;
            } else if (wc[i].wr_id & DS4_TP_RDMA_BULK_WR_TAG) {
                send_done = 1;
            } else if (r->send_outstanding > 0) {
                r->send_outstanding--;
            }
        }
        if ((peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr,
                    "ds4-tp: peer disconnected while draining RDMA receives\n");
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
        if (tp_now_sec() > deadline) {
            fprintf(stderr,
                    "ds4-tp: timeout draining RDMA receive window (%u/%u)\n",
                    recv_done, nwr);
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
    }
    r->recv_done = r->last_gate_seq;
    r->recv_window_active = false;
    pthread_mutex_unlock(&r->post_lock);
    return 1;
}

/* Large prefill row swaps share the latency QP.  No future decode receives
 * are queued, so each round can post its 1 MiB receive window before sending
 * the matching 16 KiB messages.  Verify scratch provides already-registered
 * staging memory and is idle during normal prefill. */
/* One-byte barrier on the batch socket: both sides have their receives
 * posted for the coming window before either posts a send (a UC send with
 * no receive posted kills that direction on this driver). */
static int tp_rdma_window_barrier(ds4_tp *tp, uint8_t tag) {
    uint8_t peer = 0;
    if (!tp_write_full(tp->data_fd, &tag, 1) || !tp_read_full(tp->data_fd, &peer, 1)) return 0;
    return peer == tag;
}

/* Big-gate exchange (prefill partials): windows of up to recv_depth
 * messages; every window posts its receives, crosses the barrier, then
 * streams sends in send-depth sub-batches. Metal-owned payloads are staged
 * through the registered slab because Apple's verbs provider rejects direct
 * registration of these buffers. */
static int tp_rdma_big_gate_exchange(ds4_tp *tp,
                                     const void *out,
                                     void *in,
                                     uint64_t bytes) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!tp_rdma_big_gate_capable(tp) || r->recv_window_active) return 0;
    const uintptr_t slab_lo = (uintptr_t)tp->slab;
    const uintptr_t slab_hi = slab_lo + tp->slab_bytes;
    const uintptr_t out_lo = (uintptr_t)out;
    const uintptr_t in_lo = (uintptr_t)in;
    const bool in_slab =
        out_lo >= slab_lo && out_lo <= slab_hi && bytes <= slab_hi - out_lo &&
        in_lo >= slab_lo && in_lo <= slab_hi && bytes <= slab_hi - in_lo;
    struct ibv_mr *out_mr = r->mr;
    struct ibv_mr *in_mr = r->mr;
    const bool direct = in_slab;
    uint8_t *stage_send = tp->slab + tp->batch_out_off;
    uint8_t *stage_recv = tp->slab + tp->batch_in_off;
    const uint64_t stage_bytes = (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
    uint32_t depth = r->recv_depth ? r->recv_depth : 64u;
    if (depth > 256u) depth = 256u;
    if (!direct) {
        const uint32_t stage_slots = (uint32_t)(stage_bytes / DS4_TP_RDMA_MAX_MSG);
        if (depth > stage_slots) depth = stage_slots;
        if (depth == 0u) return 0;
        out_mr = in_mr = r->mr;
    }
    if (!r->win_sge) {
        const uint32_t cap = r->recv_depth > r->send_depth ? r->recv_depth : r->send_depth;
        r->win_sge = calloc(2u * cap, sizeof(*r->win_sge));
        r->win_rwr = calloc(cap, sizeof(*r->win_rwr));
        r->win_swr = calloc(cap, sizeof(*r->win_swr));
        if (!r->win_sge || !r->win_rwr || !r->win_swr) return 0;
    }
    const uint32_t send_depth = r->send_depth ? r->send_depth : 256u;
    uint64_t off = 0;
    uint8_t tag = 1;
    while (off < bytes) {
        const uint64_t remaining = bytes - off;
        uint32_t chunks = (uint32_t)((remaining + DS4_TP_RDMA_MAX_MSG - 1u) / DS4_TP_RDMA_MAX_MSG);
        if (chunks > depth) chunks = depth;
        uint64_t win_bytes = 0;
        for (uint32_t i = 0; i < chunks; i++) {
            const uint64_t left = remaining - win_bytes;
            const uint32_t len = (uint32_t)(left > DS4_TP_RDMA_MAX_MSG ? DS4_TP_RDMA_MAX_MSG : left);
            const uint64_t coff = win_bytes;
            if (!direct) {
                memcpy(stage_send + coff, (const uint8_t *)out + off + coff, len);
            }
            r->win_sge[i] = (struct ibv_sge) {
                .addr = direct ? in_lo + off + coff : (uintptr_t)(stage_recv + coff),
                .length = len,
                .lkey = in_mr->lkey,
            };
            memset(&r->win_rwr[i], 0, sizeof(r->win_rwr[i]));
            r->win_rwr[i].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            r->win_rwr[i].sg_list = &r->win_sge[i];
            r->win_rwr[i].num_sge = 1;
            r->win_rwr[i].next = i + 1u < chunks ? &r->win_rwr[i + 1u] : NULL;
            r->win_sge[depth + i] = (struct ibv_sge) {
                .addr = direct ? out_lo + off + coff : (uintptr_t)(stage_send + coff),
                .length = len,
                .lkey = out_mr->lkey,
            };
            win_bytes += len;
        }
        /* Post in chains of 64 (the chain length the driver is known to
         * accept); the work requests are already linked, so cut the links at
         * chain ends. */
        for (uint32_t c0 = 0; c0 < chunks; c0 += 64u) {
            const uint32_t c1 = c0 + 64u < chunks ? c0 + 64u : chunks;
            r->win_rwr[c1 - 1u].next = NULL;
            struct ibv_recv_wr *bad_recv = NULL;
            if (ibv_post_recv(r->qp, &r->win_rwr[c0], &bad_recv) != 0) {
                fprintf(stderr, "ds4-tp: big gate post_recv(%u of %u): %s\n", c0, chunks, strerror(errno));
                return 0;
            }
        }
        atomic_thread_fence(memory_order_release);
        if (!tp_rdma_window_barrier(tp, tag)) {
            fprintf(stderr, "ds4-tp: big gate window barrier failed\n");
            return 0;
        }
        tag++;
        /* Sends in sub-batches bounded by the send depth; the last of each
         * sub-batch is signaled. */
        uint32_t sent = 0, send_done = 0, recv_done = 0, signaled = 0;
        const double deadline = tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0 + 2.0;
        uint32_t peer_poll = 0;
        while (recv_done < chunks || send_done < signaled || sent < chunks) {
            if (sent < chunks && (signaled - send_done) < 4u) {
                uint32_t n = chunks - sent;
                if (n > 64u) n = 64u;
                if (n > send_depth / 4u && send_depth / 4u > 0u) n = send_depth / 4u;
                if (n == 0u) n = 1u;
                for (uint32_t i = 0; i < n; i++) {
                    struct ibv_send_wr *w = &r->win_swr[i];
                    memset(w, 0, sizeof(*w));
                    w->wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)(sent + i) + 1u);
                    w->sg_list = &r->win_sge[depth + sent + i];
                    w->num_sge = 1;
                    w->opcode = IBV_WR_SEND;
                    w->send_flags = i + 1u == n ? IBV_SEND_SIGNALED : 0;
                    w->next = i + 1u < n ? &r->win_swr[i + 1u] : NULL;
                }
                struct ibv_send_wr *bad_send = NULL;
                if (ibv_post_send(r->qp, r->win_swr, &bad_send) != 0) {
                    fprintf(stderr, "ds4-tp: big gate post_send(%u): %s\n", n, strerror(errno));
                    return 0;
                }
                sent += n;
                signaled++;
            }
            struct ibv_wc wc[64];
            int nwc = ibv_poll_cq(r->cq, 64, wc);
            if (nwc < 0) return 0;
            for (int i = 0; i < nwc; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "ds4-tp: big gate completion error: %s\n",
                            tp_wc_status_str(wc[i].status));
                    return 0;
                }
                if ((wc[i].wr_id & DS4_TP_RDMA_BULK_WR_TAG) == 0) {
                    if (wc[i].opcode & IBV_WC_RECV) {
                        if (wc[i].wr_id > r->recv_done) r->recv_done = wc[i].wr_id;
                    } else if (r->send_outstanding > 0) {
                        r->send_outstanding--;
                    }
                    continue;
                }
                if (wc[i].opcode & IBV_WC_RECV) {
                    const uint64_t idx = (wc[i].wr_id & ~DS4_TP_RDMA_BULK_WR_TAG) - 1u;
                    const uint32_t want = idx + 1u < chunks || win_bytes % DS4_TP_RDMA_MAX_MSG == 0u
                        ? (uint32_t)DS4_TP_RDMA_MAX_MSG : (uint32_t)(win_bytes % DS4_TP_RDMA_MAX_MSG);
                    if (idx >= chunks || wc[i].byte_len != want) {
                        fprintf(stderr, "ds4-tp: big gate chunk %llu received %u bytes, expected %u\n",
                                (unsigned long long)idx, wc[i].byte_len, want);
                        return 0;
                    }
                    recv_done++;
                } else {
                    send_done++;
                }
            }
            if (nwc == 0 && (peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
                fprintf(stderr, "ds4-tp: peer disconnected during big gate\n");
                return 0;
            }
            if (nwc == 0 && tp_now_sec() > deadline) {
                fprintf(stderr, "ds4-tp: timeout in big gate window (%u/%u recvs, %u/%u sends, %u sent)\n",
                        recv_done, chunks, send_done, signaled, sent);
                return 0;
            }
        }
        atomic_thread_fence(memory_order_acquire);
        if (!direct) {
            memcpy((uint8_t *)in + off, stage_recv, win_bytes);
        }
        off += win_bytes;
    }
    return 1;
}

static void tp_rdma_close(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
    free(r->win_sge); free(r->win_rwr); free(r->win_swr);
    r->win_sge = NULL; r->win_rwr = NULL; r->win_swr = NULL;
    if (r->qp) r->api.destroy_qp(r->qp);
    if (r->mr) r->api.dereg_mr(r->mr);
    if (r->cq) r->api.destroy_cq(r->cq);
    if (r->pd) r->api.dealloc_pd(r->pd);
    if (r->ctx) r->api.close_device(r->ctx);
    r->qp = NULL; r->mr = NULL; r->cq = NULL; r->pd = NULL; r->ctx = NULL;
}

#endif /* DS4_TP_HAVE_VERBS */

/* ------------------------------------------------------------------------
 * Bring-up.
 * --------------------------------------------------------------------- */

static int tp_hello_exchange(ds4_tp *tp, const ds4_tp_identity *id, int rdma_ok,
                             char *err, size_t errlen) {
    ds4_tp_hello_fixed mine = {
        .magic = DS4_TP_MAGIC,
        .version = DS4_TP_PROTOCOL_VERSION,
        .role = (uint32_t)tp->opt.role,
        .rdma_ok = (uint32_t)rdma_ok,
        .nhi_ok = (uint32_t)(tp->opt.transport == DS4_TP_TRANSPORT_NHI &&
                             tp->opt.nhi_device != NULL),
        .nhi_ring_frames = tp->opt.nhi_ring_frames,
        .nhi_msg_frames = DS4_TP_NHI_MSG_FRAMES,
        .transport = (uint32_t)tp->opt.transport,
        .gguf_bytes = id->gguf_bytes,
        .model_id = id->model_id,
        .n_layer = id->n_layer,
        .n_embd = id->n_embd,
        .n_vocab = id->n_vocab,
        .quant_bits = id->quant_bits,
        .ctx_size = id->ctx_size,
        .gate_slot_start = id->gate_slot_start,
        .gate_slot_step = id->gate_slot_step,
        .gates_per_token = id->gates_per_token,
    };
    memcpy(mine.gate_slot_mask, id->gate_slot_mask,
           sizeof(mine.gate_slot_mask));
    ds4_tp_hello_fixed theirs;
    if (!tp_write_full(tp->control_fd, &mine, sizeof(mine)) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp hello exchange failed");
        return 0;
    }
    if (theirs.magic != DS4_TP_MAGIC) {
        tp_set_err(err, errlen, "tp hello: bad magic (mixed byte order or wrong peer?)");
        return 0;
    }
    if (theirs.version != DS4_TP_PROTOCOL_VERSION) {
        tp_set_err(err, errlen, "tp hello: protocol version %u != %u",
                   theirs.version, DS4_TP_PROTOCOL_VERSION);
        return 0;
    }
    if (theirs.role == mine.role) {
        tp_set_err(err, errlen, "tp hello: both sides claim role %u", mine.role);
        return 0;
    }
    if (theirs.gguf_bytes != mine.gguf_bytes || theirs.model_id != mine.model_id ||
        theirs.n_layer != mine.n_layer || theirs.n_embd != mine.n_embd ||
        theirs.n_vocab != mine.n_vocab || theirs.quant_bits != mine.quant_bits ||
        theirs.gate_slot_start != mine.gate_slot_start ||
        theirs.gate_slot_step != mine.gate_slot_step ||
        theirs.gates_per_token != mine.gates_per_token ||
        memcmp(theirs.gate_slot_mask, mine.gate_slot_mask,
               sizeof(mine.gate_slot_mask)) != 0) {
        tp_set_err(err, errlen,
                   "tp hello: model mismatch (peer gguf=%llu id=%u layers=%u embd=%u "
                   "vocab=%u qbits=%u)",
                   (unsigned long long)theirs.gguf_bytes, theirs.model_id,
                   theirs.n_layer, theirs.n_embd, theirs.n_vocab, theirs.quant_bits);
        return 0;
    }
    const uint32_t mask_count = tp_gate_mask_count(mine.gate_slot_mask);
    const uint32_t n_slots = mine.n_layer * DS4_TP_GATES_PER_LAYER;
    if ((mask_count != 0 && mask_count != mine.gates_per_token) ||
        !tp_gate_mask_fits(mine.gate_slot_mask, n_slots)) {
        tp_set_err(err, errlen,
                   "tp hello: invalid gate schedule mask (%u bits, %u gates, %u slots)",
                   mask_count, mine.gates_per_token, n_slots);
        return 0;
    }
    tp->peer_ctx = theirs.ctx_size;
    tp->n_layer = id->n_layer;
    tp->n_embd = id->n_embd;
    tp->vec_bytes = (uint64_t)id->n_embd * sizeof(float);
    tp->n_slots = n_slots;
    tp->gate_slot_start = id->gate_slot_start;
    tp->gate_slot_step = id->gate_slot_step;
    tp->gates_per_token = id->gates_per_token;
    memcpy(tp->gate_slot_mask, id->gate_slot_mask,
           sizeof(tp->gate_slot_mask));
    tp_slab_layout(tp);
    if (theirs.transport != mine.transport) {
        tp_set_err(err, errlen,
                   "tp hello: transport mismatch (peer=%u local=%u)",
                   theirs.transport, mine.transport);
        return 0;
    }
    tp->nhi_active = mine.nhi_ok && theirs.nhi_ok;
    if (tp->opt.transport == DS4_TP_TRANSPORT_NHI) {
        if (!tp->nhi_active) {
            tp_set_err(err, errlen,
                       "tp: --transport nhi but %s side has no configured device",
                       mine.nhi_ok ? "the peer" : "this");
            return 0;
        }
        if (theirs.nhi_ring_frames != mine.nhi_ring_frames ||
            theirs.nhi_msg_frames != mine.nhi_msg_frames) {
            tp_set_err(err, errlen,
                       "tp hello: NHI geometry mismatch "
                       "(peer ring=%u msg=%u, local ring=%u msg=%u)",
                       theirs.nhi_ring_frames, theirs.nhi_msg_frames,
                       mine.nhi_ring_frames, mine.nhi_msg_frames);
            return 0;
        }
    }
    /* Legacy data plane: RDMA only when both AUTO/RDMA peers can. */
    const int want_rdma = tp->opt.transport == DS4_TP_TRANSPORT_AUTO ||
                          tp->opt.transport == DS4_TP_TRANSPORT_RDMA;
    tp->rdma_active = !tp->nhi_active && want_rdma && rdma_ok && theirs.rdma_ok;
    if (tp->opt.transport == DS4_TP_TRANSPORT_RDMA && !tp->rdma_active) {
        tp_set_err(err, errlen, "tp: --transport rdma but %s side has no active device",
                   rdma_ok ? "the peer" : "this");
        return 0;
    }
    return 1;
}

int ds4_tp_create(
        ds4_tp **out,
        const ds4_tp_options *opt,
        const ds4_tp_identity *id,
        char *err,
        size_t errlen)
{
    *out = NULL;
    ds4_tp *tp = calloc(1, sizeof(*tp));
    if (!tp) {
        tp_set_err(err, errlen, "tp: out of memory");
        return 0;
    }
    tp->opt = *opt;
    if (tp->opt.nhi_ring_frames == 0)
        tp->opt.nhi_ring_frames = DS4_TP_NHI_DEFAULT_RING_FRAMES;
    tp->rank = opt->role == DS4_TP_LEADER ? 0 : 1;
    tp->local_ctx = id->ctx_size;
    tp->control_fd = -1;
    tp->data_fd = -1;
    tp->timeout_sec = DS4_TP_DEFAULT_TIMEOUT_SEC;
    const char *tmo = getenv("DS4_TP_TIMEOUT_SEC");
    if (tmo) {
        char *end = NULL;
        errno = 0;
        unsigned long long value = strtoull(tmo, &end, 10);
        if (errno != 0 || !end || *end != '\0' || value == 0 ||
            value > DS4_TP_MAX_TIMEOUT_SEC) {
            tp_set_err(err, errlen, "tp: invalid DS4_TP_TIMEOUT_SEC %s", tmo);
            ds4_tp_free(tp);
            return 0;
        }
        tp->timeout_sec = (uint64_t)value;
    }
    tp->gate_timeout_ms = DS4_TP_DEFAULT_GATE_TIMEOUT_MS;
    const char *gate_tmo = getenv("DS4_TP_GATE_TIMEOUT_MS");
    if (gate_tmo) {
        const long value = strtol(gate_tmo, NULL, 10);
        if (value > 0 && value <= 60000) tp->gate_timeout_ms = (uint64_t)value;
    }

    int rdma_ok = 0;
#ifdef DS4_TP_HAVE_VERBS
    if ((opt->transport == DS4_TP_TRANSPORT_AUTO ||
         opt->transport == DS4_TP_TRANSPORT_RDMA) &&
        (uint64_t)id->n_embd * sizeof(float) <= 2ull * DS4_TP_RDMA_MAX_MSG)
        rdma_ok = tp_rdma_probe(&tp->rdma.api);
#endif

    int listener = -1;
    if (tp->rank == 0) {
        listener = tp_listen(opt->listen_host, opt->listen_port, err, errlen);
        if (listener < 0) goto fail;
        fprintf(stderr, "ds4-tp: waiting for worker on %s:%d ...\n",
                opt->listen_host ? opt->listen_host : "0.0.0.0", opt->listen_port);
        tp->control_fd = accept(listener, NULL, NULL);
        if (tp->control_fd < 0) {
            tp_set_err(err, errlen, "tp accept: %s", strerror(errno));
            goto fail;
        }
    } else {
        tp->control_fd = tp_dial(opt->leader_host, opt->leader_port,
                                 (double)tp->timeout_sec, err, errlen);
        if (tp->control_fd < 0) goto fail;
    }
    tp_socket_tune(tp->control_fd, tp->timeout_sec);

    if (!tp_hello_exchange(tp, id, rdma_ok, err, errlen)) goto fail;

#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) {
        if (!tp_rdma_open(tp, err, errlen)) goto fail;
    }
#endif
    {
        /* Second socket dedicated to gate traffic so control frames never
         * interleave with gate payloads.  Created under RDMA too for
         * headers, verify-block gates, and transport fallback. */
        if (tp->rank == 0) {
            tp->data_fd = accept(listener, NULL, NULL);
            if (tp->data_fd < 0) {
                tp_set_err(err, errlen, "tp data accept: %s", strerror(errno));
                goto fail;
            }
        } else {
            tp->data_fd = tp_dial(opt->leader_host, opt->leader_port,
                                  (double)tp->timeout_sec, err, errlen);
            if (tp->data_fd < 0) goto fail;
        }
        tp_socket_tune(tp->data_fd, tp->timeout_sec);
        if (!tp_socket_set_gate_timeout(tp->data_fd, tp->gate_timeout_ms)) {
            tp_set_err(err, errlen, "tp data socket timeout: %s", strerror(errno));
            goto fail;
        }
    }
    if (listener >= 0) close(listener);
    fprintf(stderr, "ds4-tp: %s connected, transport=%s gate-timeout=%llums\n",
            tp->rank == 0 ? "worker" : "leader",
            tp->nhi_active ? "nhi" : (tp->rdma_active ? "rdma" : "tcp"),
            (unsigned long long)tp->gate_timeout_ms);
    *out = tp;
    return 1;
fail:
    if (listener >= 0) close(listener);
    ds4_tp_free(tp);
    return 0;
}

int ds4_tp_attach_slab(ds4_tp *tp, void *base, char *err, size_t errlen) {
    tp->slab = base;
    memset(tp->slab + tp->in_flags_off, 0, (uint64_t)tp->n_slots * 8);
    memset(tp->slab + tp->token_off, 0, 16);
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) {
        for (;;) {
            tp->rdma.warm_failed = 0;
            if (tp_rdma_register_and_exchange(tp, err, errlen)) return 1;
            if (!tp->rdma.warm_failed || tp->rdma.setup_attempt >= 3) return 0;
            tp->rdma.setup_attempt++;
            fprintf(stderr, "ds4-tp: %s; recreating the queue pair (setup attempt %u)\n",
                    err, tp->rdma.setup_attempt + 1u);
            if (!tp_rdma_recreate_qp(tp, err, errlen)) return 0;
        }
    }
#endif
    (void)err; (void)errlen;
    return 1;
}

void ds4_tp_free(ds4_tp *tp) {
    if (!tp) return;
#ifdef DS4_TP_HAVE_VERBS
    tp_rdma_close(tp);
#endif
    if (tp->control_fd >= 0) close(tp->control_fd);
    if (tp->data_fd >= 0) close(tp->data_fd);
    free(tp);
}

int ds4_tp_rank(const ds4_tp *tp) { return tp->rank; }
bool ds4_tp_is_rdma(const ds4_tp *tp) { return tp->rdma_active; }
bool ds4_tp_is_nhi(const ds4_tp *tp) { return tp && tp->nhi_active; }
const char *ds4_tp_nhi_device(const ds4_tp *tp) {
    return tp && tp->nhi_active ? tp->opt.nhi_device : NULL;
}
uint32_t ds4_tp_nhi_ring_frames(const ds4_tp *tp) {
    return tp && tp->nhi_active ? tp->opt.nhi_ring_frames : 0;
}
uint32_t ds4_tp_peer_ctx(const ds4_tp *tp) { return tp->peer_ctx; }
bool ds4_tp_failed(const ds4_tp *tp) {
    return tp && atomic_load_explicit(&tp->failed, memory_order_acquire);
}
void ds4_tp_mark_failed(ds4_tp *tp) {
    if (tp) atomic_store_explicit(&tp->failed, true, memory_order_release);
}

int ds4_tp_nhi_ready_barrier(ds4_tp *tp, char *err, size_t errlen) {
    if (!tp || !tp->nhi_active) return tp != NULL;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_NHI_READY, NULL, 0)) {
        tp_set_err(err, errlen, "tp NHI ready send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_NHI_READY || bytes != 0) {
        tp_set_err(err, errlen, "tp NHI ready barrier failed");
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Gate exchange.
 * --------------------------------------------------------------------- */

int ds4_tp_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) return tp_rdma_gate_exchange(tp, layer, gate, seq);
#endif
    /* TCP: both sides write their partial then read the peer's.  16KB per
     * direction fits comfortably in the socket buffers, so the symmetric
     * write-then-read cannot deadlock.  Header and payload go out in one
     * writev so NODELAY does not split them into two segments. */
    ds4_tp_gate_header h = { DS4_TP_MAGIC, (uint16_t)layer, (uint16_t)gate, seq };
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + ds4_tp_slab_out_offset(tp, layer, gate), tp->vec_bytes },
    };
    size_t want = sizeof(h) + tp->vec_bytes;
    ssize_t w = writev(tp->data_fd, iov, 2);
    if (w < 0 || (size_t)w != want) {
        /* Short writev: finish with the plain path. */
        if (w < 0) return 0;
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(tp->data_fd, (char *)&h + done, sizeof(h) - done)) return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(tp->data_fd,
                           tp->slab + ds4_tp_slab_out_offset(tp, layer, gate) + payload_done,
                           tp->vec_bytes - payload_done))
            return 0;
    }
    ds4_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != DS4_TP_MAGIC || ph.layer != layer || ph.gate != gate || ph.seq != seq) {
        fprintf(stderr, "ds4-tp: gate desync: got l=%u g=%u seq=%llu, want l=%u g=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    if (!tp_read_full(tp->data_fd, tp->slab + ds4_tp_slab_in_offset(tp, layer, gate),
                      tp->vec_bytes))
        return 0;
    return 1;
}

/* Verify-block batch gate: one exchange per layer moving all block rows at
 * once. The payload lives in the registered slab, so RDMA sends it directly;
 * TCP remains the symmetric write-then-read fallback. */
/* Verify-block window: called on both ranks right before a speculative
 * verify block whose every layer ends in one batch gate of `rows` rows.
 * Drains the decode receive window, posts the first layer's receives,
 * crosses one control-byte barrier (so both sides are posted before any
 * send), and from then on each gate posts the next layer's receives and
 * sends its rows without any handshake.  Returns 1 also when RDMA is not
 * in use (the TCP fallback keeps its per-gate headers). */
int ds4_tp_batch_block_begin(ds4_tp *tp, uint32_t rows, uint32_t n_layers) {
#ifdef DS4_TP_HAVE_VERBS
    if (!tp->rdma_active || !tp_rdma_big_gate_capable(tp)) return 1;
    if (getenv("DS4_TP_DISABLE_VERIFY_WINDOW")) return 1;
    ds4_tp_rdma *r = &tp->rdma;
    if (rows == 0 || rows > DS4_TP_BATCH_MAX_ROWS || n_layers == 0 || r->block_active) return 0;
    if (!tp_rdma_drain_decode_window(tp)) return 0;
    pthread_mutex_lock(&r->post_lock);
    r->block_active = true;
    r->block_rows = rows;
    r->block_layers = n_layers;
    r->block_posted = 0;
    r->block_recv_done = 0;
    const int ok = tp_rdma_block_post_layer(tp, 0);
    pthread_mutex_unlock(&r->post_lock);
    if (!ok) { r->block_active = false; return 0; }
    if (!tp_rdma_window_barrier(tp, 0xB5u)) { r->block_active = false; return 0; }
    if (getenv("DS4_TP_BIG_GATE_DEBUG"))
        fprintf(stderr, "ds4-tp: verify window armed: %u rows x %u layers\n", rows, n_layers);
    return 1;
#else
    (void)tp; (void)rows; (void)n_layers;
    return 1;
#endif
}

/* End of the verify block: every gate must have been exchanged (all
 * posted receives consumed) and our signaled sends reaped. */
int ds4_tp_batch_block_end(ds4_tp *tp) {
#ifdef DS4_TP_HAVE_VERBS
    ds4_tp_rdma *r = &tp->rdma;
    if (!r->block_active) return 1;
    int ok = 1;
    const uint64_t want = (uint64_t)r->block_layers * r->block_rows;
    double deadline = tp_now_sec() + (double)tp->gate_timeout_ms / 1000.0;
    while (ok && (r->block_recv_done < want || r->send_outstanding > 0)) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: verify-block end: %llu/%llu rows received, %u sends pending\n",
                    (unsigned long long)r->block_recv_done, (unsigned long long)want,
                    r->send_outstanding);
            ok = 0;
        }
    }
    if (getenv("DS4_TP_BIG_GATE_DEBUG"))
        fprintf(stderr, "ds4-tp: verify window closed: %llu/%llu rows, ok=%d\n",
                (unsigned long long)r->block_recv_done, (unsigned long long)want, ok);
    r->block_active = false;
    return ok;
#else
    (void)tp;
    return 1;
#endif
}

int ds4_tp_batch_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows,
                               uint64_t seq) {
    if (tp->data_fd < 0 || rows == 0 || rows > DS4_TP_BATCH_MAX_ROWS) return 0;
    const uint64_t bytes = (uint64_t)rows * tp->vec_bytes;
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer,
                             (uint16_t)rows, seq };
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active && tp->rdma.block_active)
        return tp_rdma_block_gate_exchange(tp, layer, rows);
    if (tp->rdma_active && tp_rdma_big_gate_capable(tp)) {
        if (!tp_write_full(tp->data_fd, &h, sizeof(h))) return 0;
        ds4_tp_gate_header ph;
        if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
        if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
            ph.gate != rows || ph.seq != seq) {
            fprintf(stderr,
                    "ds4-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                    "want l=%u rows=%u seq=%llu\n",
                    ph.layer, ph.gate, (unsigned long long)ph.seq,
                    layer, rows, (unsigned long long)seq);
            return 0;
        }
        if (!tp_rdma_drain_decode_window(tp)) return 0;
        return tp_rdma_big_gate_exchange(
                tp,
                tp->slab + ds4_tp_slab_batch_out_offset(tp, layer),
                tp->slab + ds4_tp_slab_batch_in_offset(tp, layer),
                bytes);
    }
#endif
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + ds4_tp_slab_batch_out_offset(tp, layer), bytes },
    };
    size_t want = sizeof(h) + bytes;
    ssize_t w = writev(tp->data_fd, iov, 2);
    if (w < 0) return 0;
    if ((size_t)w != want) {
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(tp->data_fd, (char *)&h + done, sizeof(h) - done))
                return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(tp->data_fd,
                           tp->slab + ds4_tp_slab_batch_out_offset(tp, layer) +
                               payload_done,
                           bytes - payload_done))
            return 0;
    }
    ds4_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
    if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != rows || ph.seq != seq) {
        fprintf(stderr,
                "ds4-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                "want l=%u rows=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, rows, (unsigned long long)seq);
        return 0;
    }
    return tp_read_full(tp->data_fd,
                        tp->slab + ds4_tp_slab_batch_in_offset(tp, layer),
                        bytes);
}

/* Prefill batch gate: RDMA uses the pipelined registered-slab path above.
 * The fallback alternates 2MB TCP write/read rounds in the same order, so
 * neither side can fill its send buffer while the peer is also only writing
 * (the 4MB socket buffers absorb one round). */
#define DS4_TP_BIG_CHUNK (2ull * 1024ull * 1024ull)

int ds4_tp_big_gate_exchange(ds4_tp *tp, uint32_t layer, uint64_t seq,
                             const void *out, void *in, uint64_t bytes) {
    if (tp->data_fd < 0 || !out || !in || bytes == 0) return 0;
#ifdef DS4_TP_HAVE_VERBS
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("DS4_TP_BIG_GATE_DEBUG") != NULL;
    const double t_start = dbg ? tp_now_sec() : 0.0;
#endif
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer, 0xB16u, seq };
    if (!tp_write_full(tp->data_fd, &h, sizeof(h))) return 0;
    ds4_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
#ifdef DS4_TP_HAVE_VERBS
    const double t_hs = dbg ? tp_now_sec() : 0.0;
#endif
    if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != 0xB16u || ph.seq != seq) {
        fprintf(stderr,
                "ds4-tp: big gate desync: got l=%u tag=%x seq=%llu, want l=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, (unsigned long long)seq);
        return 0;
    }
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active && tp_rdma_big_gate_capable(tp)) {
        if (!tp_rdma_drain_decode_window(tp)) return 0;
        const int ok = tp_rdma_big_gate_exchange(tp, out, in, bytes);
        if (dbg) {
            static unsigned n;
            static double hs_acc, tx_acc, last_end;
            const double t_end = tp_now_sec();
            hs_acc += t_hs - t_start; tx_acc += t_end - t_hs;
            if (getenv("DS4_TP_BIG_GATE_DEBUG")[0] == '2')
                fprintf(stderr, "ds4-tp: big gate #%u layer %u: %llu bytes, since last release %.2f ms, handshake %.2f, transfer %.2f\n",
                        n + 1, layer, (unsigned long long)bytes,
                        last_end > 0.0 ? (t_start - last_end) * 1e3 : 0.0,
                        (t_hs - t_start) * 1e3, (t_end - t_hs) * 1e3);
            last_end = t_end;
            if ((++n % 32u) == 0u) {
                fprintf(stderr, "ds4-tp: big gate %u: %llu bytes, last: handshake wait %.2f ms, transfer %.2f ms; avg over 32: %.2f / %.2f ms\n",
                        n, (unsigned long long)bytes, (t_hs - t_start) * 1e3, (t_end - t_hs) * 1e3,
                        hs_acc / 32.0 * 1e3, tx_acc / 32.0 * 1e3);
                hs_acc = tx_acc = 0.0;
            }
        }
        return ok;
    }
#endif
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t n = bytes - off > DS4_TP_BIG_CHUNK ?
                           DS4_TP_BIG_CHUNK : bytes - off;
        if (!tp_write_full(tp->data_fd, (const char *)out + off, n)) return 0;
        if (!tp_read_full(tp->data_fd, (char *)in + off, n)) return 0;
        off += n;
    }
    if (getenv("DS4_GLM_TP_DEBUG")) {
        const float *o = (const float *)out, *i = (const float *)in;
        fprintf(stderr,
                "ds4-tp: big gate l=%u seq=%llu out[0..3]=%g %g %g %g in[0..3]=%g %g %g %g\n",
                layer, (unsigned long long)seq,
                o[0], o[1], o[2], o[3], i[0], i[1], i[2], i[3]);
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Lockstep control plane.
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t session_id;
    uint32_t count;
    uint32_t reserved;
} ds4_tp_token_command_header;

typedef struct {
    uint64_t session_id;
    uint32_t token_count;
    uint32_t image_count;
} ds4_tp_multimodal_command_header;

typedef struct {
    uint32_t token_start;
    uint32_t token_count;
    uint32_t data_width;
    uint32_t width;
    uint32_t height;
    uint32_t content_width;
    uint32_t content_height;
    uint8_t fingerprint[32];
} ds4_tp_vision_wire;

typedef struct {
    uint64_t session_id;
    int32_t value;
    uint32_t reserved;
} ds4_tp_value_command;

typedef struct {
    uint64_t session_id;
    uint64_t seq;
    int32_t token;
    uint32_t reserved;
} ds4_tp_eval_command;

typedef struct {
    uint32_t count;
    uint32_t reserved;
} ds4_tp_batch_command_header;

typedef struct {
    uint64_t prefill_session_id;
    uint32_t prompt_count;
    uint32_t item_count;
} ds4_tp_mixed_command_header;

typedef struct {
    uint64_t session_id;
    int32_t status;
    uint32_t reserved;
} ds4_tp_command_ack;

static int tp_send_token_command(ds4_tp *tp, uint32_t type,
                                 uint64_t session_id, const int *tokens,
                                 uint32_t count) {
    const uint64_t bytes64 = sizeof(ds4_tp_token_command_header) +
                             (uint64_t)count * sizeof(int32_t);
    if (!tp || (!tokens && count != 0) || bytes64 > UINT32_MAX ||
        (type == DS4_TP_FRAME_SYNC &&
         (count > tp->local_ctx || count > DS4_TP_COMMAND_MAX_TOKENS)) ||
        (type == DS4_TP_FRAME_VERIFY &&
         (count == 0 || count > DS4_TP_BATCH_MAX_ROWS))) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes ? bytes : 1u);
    if (!payload) return 0;
    ds4_tp_token_command_header h = { session_id, count, 0 };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = (int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < count; i++) wire_tokens[i] = (int32_t)tokens[i];
    const int ok = tp_send_frame(tp->control_fd, type, payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_session_create(ds4_tp *tp, uint64_t session_id, int ctx_size) {
    if (!tp || ctx_size <= 0 || (uint32_t)ctx_size > tp->local_ctx ||
        (uint32_t)ctx_size > DS4_TP_COMMAND_MAX_TOKENS) return 0;
    ds4_tp_value_command msg = { session_id, (int32_t)ctx_size, 0 };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_SESSION_CREATE,
                         &msg, sizeof(msg));
}

int ds4_tp_send_session_destroy(ds4_tp *tp, uint64_t session_id) {
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_SESSION_DESTROY,
                         &session_id, sizeof(session_id));
}

int ds4_tp_send_sync(ds4_tp *tp, uint64_t session_id,
                     const int *tokens, uint32_t n_tokens) {
    return tp_send_token_command(tp, DS4_TP_FRAME_SYNC, session_id,
                                 tokens, n_tokens);
}

int ds4_tp_send_sync_multimodal(ds4_tp *tp, uint64_t session_id,
                                const int *tokens, uint32_t n_tokens,
                                const ds4_vision_span *images,
                                uint32_t image_count) {
    uint64_t bytes64 = sizeof(ds4_tp_multimodal_command_header) +
                       (uint64_t)n_tokens * sizeof(int32_t);
    if (!tp || (!tokens && n_tokens) || (!images && image_count) ||
        tp->n_embd == 0 || bytes64 > UINT32_MAX)
        return 0;
    for (uint32_t i = 0; i < image_count; i++) {
        const uint64_t values = (uint64_t)images[i].embedding.token_count *
                                tp->n_embd;
        if (values > UINT64_MAX / sizeof(float)) return 0;
        const uint64_t data_bytes = values * sizeof(float);
        const uint64_t add_bytes = sizeof(ds4_tp_vision_wire) + data_bytes;
        if (!images[i].embedding.data || add_bytes > UINT32_MAX ||
            bytes64 > UINT32_MAX - add_bytes)
            return 0;
        bytes64 += add_bytes;
    }
    uint8_t *payload = malloc((size_t)bytes64);
    if (!payload) return 0;
    ds4_tp_multimodal_command_header h = {
        session_id, n_tokens, image_count
    };
    memcpy(payload, &h, sizeof(h));
    uint8_t *p = payload + sizeof(h);
    int32_t *wire_tokens = (int32_t *)p;
    for (uint32_t i = 0; i < n_tokens; i++) wire_tokens[i] = tokens[i];
    p += (uint64_t)n_tokens * sizeof(int32_t);
    for (uint32_t i = 0; i < image_count; i++) {
        const ds4_vision_embedding *embedding = &images[i].embedding;
        ds4_tp_vision_wire wire = {
            images[i].token_start,
            embedding->token_count,
            tp->n_embd,
            embedding->width,
            embedding->height,
            embedding->content_width,
            embedding->content_height,
            {0},
        };
        memcpy(wire.fingerprint, embedding->fingerprint,
               sizeof(wire.fingerprint));
        memcpy(p, &wire, sizeof(wire));
        p += sizeof(wire);
        uint64_t data_bytes = (uint64_t)embedding->token_count *
                              tp->n_embd * sizeof(float);
        memcpy(p, embedding->data, (size_t)data_bytes);
        p += data_bytes;
    }
    int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_SYNC_MULTIMODAL,
                           payload, (uint32_t)bytes64);
    free(payload);
    return ok;
}

int ds4_tp_send_eval(ds4_tp *tp, uint64_t session_id,
                     uint64_t seq, int token) {
    ds4_tp_eval_command msg = { session_id, seq, (int32_t)token, 0 };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_EVAL, &msg, sizeof(msg));
}

int ds4_tp_send_glm_mtp(ds4_tp *tp, uint64_t session_id,
                       uint64_t seq, int token, int limit) {
    if (limit < 1 || limit > 2) return 0;
    ds4_tp_eval_command msg = { session_id, seq, (int32_t)token, (uint32_t)limit };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_GLM_MTP, &msg, sizeof(msg));
}

int ds4_tp_send_rewind(ds4_tp *tp, uint64_t session_id, int pos) {
    ds4_tp_value_command msg = { session_id, (int32_t)pos, 0 };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_REWIND,
                         &msg, sizeof(msg));
}

int ds4_tp_send_invalidate(ds4_tp *tp, uint64_t session_id) {
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_INVALIDATE,
                         &session_id, sizeof(session_id));
}

int ds4_tp_send_eval_batch(ds4_tp *tp, const ds4_tp_batch_item *items,
                           uint32_t count) {
    const uint64_t bytes64 = sizeof(ds4_tp_batch_command_header) +
                             (uint64_t)count * sizeof(*items);
    if (!tp || !items || count == 0 ||
        count > DS4_TP_COMMAND_MAX_BATCH_ITEMS ||
        bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes);
    if (!payload) return 0;
    ds4_tp_batch_command_header h = { count, 0 };
    memcpy(payload, &h, sizeof(h));
    memcpy(payload + sizeof(h), items, (size_t)count * sizeof(*items));
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_EVAL_BATCH,
                                 payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_mixed_batch(ds4_tp *tp, uint64_t prefill_session_id,
                            const int *prompt, uint32_t prompt_count,
                            const ds4_tp_batch_item *items,
                            uint32_t count) {
    const uint64_t prompt_bytes = (uint64_t)prompt_count * sizeof(int32_t);
    const uint64_t item_bytes = (uint64_t)count * sizeof(*items);
    const uint64_t bytes64 = sizeof(ds4_tp_mixed_command_header) +
                             prompt_bytes + item_bytes;
    if (!tp || !prompt || prompt_count == 0 ||
        prompt_count > tp->local_ctx ||
        prompt_count > DS4_TP_COMMAND_MAX_TOKENS ||
        !items || count == 0 ||
        count > DS4_TP_COMMAND_MAX_BATCH_ITEMS ||
        bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes);
    if (!payload) return 0;
    ds4_tp_mixed_command_header h = {
        prefill_session_id, prompt_count, count
    };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = (int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < prompt_count; i++) {
        wire_tokens[i] = (int32_t)prompt[i];
    }
    memcpy(payload + sizeof(h) + prompt_bytes, items, (size_t)item_bytes);
    const int ok = tp_send_frame(tp->control_fd, DS4_TP_FRAME_MIXED_BATCH,
                                 payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_command_ack(ds4_tp *tp, uint64_t session_id, int status) {
    ds4_tp_command_ack ack = { session_id, (int32_t)status, 0 };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_COMMAND_ACK,
                         &ack, sizeof(ack));
}

int ds4_tp_wait_command_ack(ds4_tp *tp, uint64_t session_id,
                            const char *operation, char *err, size_t errlen) {
    uint32_t type = 0, bytes = 0;
    ds4_tp_command_ack ack;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_COMMAND_ACK || bytes != sizeof(ack) ||
        !tp_read_full(tp->control_fd, &ack, sizeof(ack))) {
        ds4_tp_mark_failed(tp);
        tp_set_err(err, errlen, "tp: worker failed during %s",
                   operation ? operation : "command");
        return 0;
    }
    if (ack.session_id != session_id || ack.status != 0) {
        tp_set_err(err, errlen,
                   "tp: worker %s failed (session %llu, status %d)",
                   operation ? operation : "command",
                   (unsigned long long)ack.session_id, (int)ack.status);
        return 0;
    }
    return 1;
}

int ds4_tp_send_stop(ds4_tp *tp) {
    if (!tp || !tp_send_frame(tp->control_fd, DS4_TP_FRAME_STOP, NULL, 0))
        return 0;
    if (!tp->nhi_active) return 1;
    uint32_t type = 0, bytes = 0;
    return tp_read_frame_header(tp->control_fd, &type, &bytes) &&
           type == DS4_TP_FRAME_NHI_CLOSED && bytes == 0;
}

void ds4_tp_command_free(ds4_tp_command *command) {
    if (!command) return;
    free(command->tokens);
    free(command->items);
    for (uint32_t i = 0; i < command->n_images; i++)
        ds4_vision_embedding_free(&command->images[i].embedding);
    free(command->images);
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
}

static int tp_command_frame_size_allowed(
        const ds4_tp *tp, uint32_t type, uint32_t bytes) {
    if (!tp) return 0;
    const uint64_t ctx_tokens = tp->peer_ctx < DS4_TP_COMMAND_MAX_TOKENS
        ? tp->peer_ctx : DS4_TP_COMMAND_MAX_TOKENS;
    switch (type) {
    case DS4_TP_FRAME_SYNC:
        return bytes >= sizeof(ds4_tp_token_command_header) &&
            bytes <= sizeof(ds4_tp_token_command_header) +
                     ctx_tokens * sizeof(int32_t);
    case DS4_TP_FRAME_VERIFY:
        return bytes >= sizeof(ds4_tp_token_command_header) + sizeof(int32_t) &&
            bytes <= sizeof(ds4_tp_token_command_header) +
                     (uint64_t)DS4_TP_BATCH_MAX_ROWS * sizeof(int32_t);
    case DS4_TP_FRAME_SESSION_CREATE:
    case DS4_TP_FRAME_REWIND:
        return bytes == sizeof(ds4_tp_value_command);
    case DS4_TP_FRAME_SESSION_DESTROY:
    case DS4_TP_FRAME_INVALIDATE:
        return bytes == sizeof(uint64_t);
    case DS4_TP_FRAME_EVAL:
        return bytes == sizeof(ds4_tp_eval_command);
    case DS4_TP_FRAME_EVAL_BATCH:
        return bytes >= sizeof(ds4_tp_batch_command_header) +
                        sizeof(ds4_tp_batch_item) &&
            bytes <= sizeof(ds4_tp_batch_command_header) +
                     (uint64_t)DS4_TP_COMMAND_MAX_BATCH_ITEMS *
                         sizeof(ds4_tp_batch_item);
    case DS4_TP_FRAME_MIXED_BATCH:
        return bytes >= sizeof(ds4_tp_mixed_command_header) + sizeof(int32_t) +
                        sizeof(ds4_tp_batch_item) &&
            bytes <= sizeof(ds4_tp_mixed_command_header) +
                     ctx_tokens * sizeof(int32_t) +
                     (uint64_t)DS4_TP_COMMAND_MAX_BATCH_ITEMS *
                         sizeof(ds4_tp_batch_item);
    case DS4_TP_FRAME_STOP:
        return bytes == 0;
    default:
        return 0;
    }
}

static int tp_command_decode_tokens(ds4_tp_command *command,
                                    const uint8_t *payload,
                                    uint32_t bytes,
                                    char *err, size_t errlen) {
    if (bytes < sizeof(ds4_tp_token_command_header)) return 0;
    ds4_tp_token_command_header h;
    memcpy(&h, payload, sizeof(h));
    const uint64_t want = sizeof(h) + (uint64_t)h.count * sizeof(int32_t);
    if (want != bytes) return 0;
    int *tokens = malloc(h.count ? (size_t)h.count * sizeof(*tokens) : 1u);
    if (!tokens) {
        tp_set_err(err, errlen, "tp: command token allocation failed");
        return -1;
    }
    const int32_t *wire_tokens = (const int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < h.count; i++) tokens[i] = wire_tokens[i];
    command->session_id = h.session_id;
    command->tokens = tokens;
    command->n_tokens = h.count;
    return 1;
}

static int tp_command_decode_multimodal(ds4_tp *tp,
                                        ds4_tp_command *command,
                                        const uint8_t *payload,
                                        uint32_t bytes,
                                        char *err, size_t errlen) {
    if (bytes < sizeof(ds4_tp_multimodal_command_header)) return 0;
    ds4_tp_multimodal_command_header h;
    memcpy(&h, payload, sizeof(h));
    uint64_t pos = sizeof(h);
    uint64_t token_bytes = (uint64_t)h.token_count * sizeof(int32_t);
    if (pos + token_bytes > bytes) return 0;
    if (h.image_count > ((uint64_t)bytes - pos - token_bytes) /
                        sizeof(ds4_tp_vision_wire))
        return 0;
    int *tokens = malloc(h.token_count ?
                         (size_t)h.token_count * sizeof(tokens[0]) : 1u);
    ds4_vision_span *images = h.image_count ?
        calloc(h.image_count, sizeof(images[0])) : NULL;
    if (!tokens || (h.image_count && !images)) {
        free(tokens);
        free(images);
        tp_set_err(err, errlen, "tp: multimodal command allocation failed");
        return -1;
    }
    const int32_t *wire_tokens = (const int32_t *)(payload + pos);
    for (uint32_t i = 0; i < h.token_count; i++) tokens[i] = wire_tokens[i];
    pos += token_bytes;
    for (uint32_t i = 0; i < h.image_count; i++) {
        if (pos + sizeof(ds4_tp_vision_wire) > bytes) goto malformed;
        ds4_tp_vision_wire wire;
        memcpy(&wire, payload + pos, sizeof(wire));
        pos += sizeof(wire);
        uint64_t values = (uint64_t)wire.token_count * wire.data_width;
        if (wire.token_count == 0 || wire.data_width != tp->n_embd ||
            wire.width == 0 ||
            values > SIZE_MAX / sizeof(float))
            goto malformed;
        uint64_t data_bytes = values * sizeof(float);
        if (data_bytes > (uint64_t)bytes - pos) goto malformed;
        images[i].token_start = wire.token_start;
        images[i].embedding.data = malloc((size_t)data_bytes);
        if (!images[i].embedding.data) {
            tp_set_err(err, errlen, "tp: image embedding allocation failed");
            goto allocation_failed;
        }
        images[i].embedding.token_count = wire.token_count;
        images[i].embedding.width = wire.width;
        images[i].embedding.height = wire.height;
        images[i].embedding.content_width = wire.content_width;
        images[i].embedding.content_height = wire.content_height;
        memcpy(images[i].embedding.fingerprint, wire.fingerprint,
               sizeof(wire.fingerprint));
        memcpy(images[i].embedding.data, payload + pos, (size_t)data_bytes);
        pos += data_bytes;
    }
    if (pos != bytes) goto malformed;
    command->session_id = h.session_id;
    command->tokens = tokens;
    command->n_tokens = h.token_count;
    command->images = images;
    command->n_images = h.image_count;
    return 1;

malformed:
    tp_set_err(err, errlen, "tp: malformed multimodal sync command");
allocation_failed:
    for (uint32_t i = 0; i < h.image_count; i++)
        ds4_vision_embedding_free(&images[i].embedding);
    free(images);
    free(tokens);
    return 0;
}

int ds4_tp_recv_command(ds4_tp *tp, ds4_tp_command *command,
                        char *err, size_t errlen) {
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
    uint32_t ftype = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &ftype, &bytes)) {
        tp_set_err(err, errlen, "tp: control channel closed");
        return 0;
    }
    if (!tp_command_frame_size_allowed(tp, ftype, bytes)) {
        ds4_tp_mark_failed(tp);
        shutdown(tp->control_fd, SHUT_RDWR);
        tp_set_err(err, errlen,
                   "tp: invalid or oversized command frame type %u (%u bytes)",
                   ftype, bytes);
        return 0;
    }
    uint8_t *payload = NULL;
    if (bytes != 0) {
        payload = malloc(bytes);
        if (!payload || !tp_read_full(tp->control_fd, payload, bytes)) {
            free(payload);
            tp_set_err(err, errlen, "tp: truncated command frame");
            return 0;
        }
    }
    int ok = 1;
    switch (ftype) {
    case DS4_TP_FRAME_SYNC:
    case DS4_TP_FRAME_VERIFY:
        ok = tp_command_decode_tokens(command, payload, bytes, err, errlen);
        break;
    case DS4_TP_FRAME_SYNC_MULTIMODAL:
        ok = tp_command_decode_multimodal(tp, command, payload, bytes,
                                          err, errlen);
        break;
    case DS4_TP_FRAME_SESSION_CREATE:
    case DS4_TP_FRAME_REWIND: {
        ds4_tp_value_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->value = msg.value;
        break;
    }
    case DS4_TP_FRAME_SESSION_DESTROY:
    case DS4_TP_FRAME_INVALIDATE:
        if (bytes != sizeof(command->session_id)) { ok = 0; break; }
        memcpy(&command->session_id, payload, sizeof(command->session_id));
        break;
    case DS4_TP_FRAME_EVAL:
    case DS4_TP_FRAME_GLM_MTP: {
        ds4_tp_eval_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->seq = msg.seq;
        command->value = msg.token;
        if (ftype == DS4_TP_FRAME_GLM_MTP) {
            if (msg.reserved < 1 || msg.reserved > 2) { ok = 0; break; }
            command->limit = (int)msg.reserved;
        } else if (msg.reserved != 0) {
            ok = 0;
        }
        break;
    }
    case DS4_TP_FRAME_EVAL_BATCH: {
        ds4_tp_batch_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t want = sizeof(h) +
                              (uint64_t)h.count * sizeof(ds4_tp_batch_item);
        if (h.count == 0 || h.count > DS4_TP_COMMAND_MAX_BATCH_ITEMS ||
            want != bytes) { ok = 0; break; }
        command->items = malloc((size_t)h.count * sizeof(*command->items));
        if (!command->items) { ok = -1; break; }
        memcpy(command->items, payload + sizeof(h),
               (size_t)h.count * sizeof(*command->items));
        command->n_items = h.count;
        break;
    }
    case DS4_TP_FRAME_MIXED_BATCH: {
        ds4_tp_mixed_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t token_bytes =
            (uint64_t)h.prompt_count * sizeof(int32_t);
        const uint64_t item_bytes =
            (uint64_t)h.item_count * sizeof(ds4_tp_batch_item);
        const uint64_t want = sizeof(h) + token_bytes + item_bytes;
        if (h.prompt_count == 0 || h.prompt_count > tp->peer_ctx ||
            h.item_count == 0 ||
            h.item_count > DS4_TP_COMMAND_MAX_BATCH_ITEMS || want != bytes) {
            ok = 0;
            break;
        }
        command->tokens = malloc((size_t)h.prompt_count *
                                 sizeof(*command->tokens));
        command->items = malloc((size_t)h.item_count *
                                sizeof(*command->items));
        if (!command->tokens || !command->items) { ok = -1; break; }
        const int32_t *wire_tokens =
            (const int32_t *)(payload + sizeof(h));
        for (uint32_t i = 0; i < h.prompt_count; i++) {
            command->tokens[i] = wire_tokens[i];
        }
        memcpy(command->items, payload + sizeof(h) + token_bytes,
               (size_t)item_bytes);
        command->session_id = h.prefill_session_id;
        command->n_tokens = h.prompt_count;
        command->n_items = h.item_count;
        break;
    }
    case DS4_TP_FRAME_STOP:
        if (bytes != 0) ok = 0;
        break;
    default:
        ok = 0;
        break;
    }
    free(payload);
    if (ok <= 0) {
        ds4_tp_command_free(command);
        if (ok == 0) {
            tp_set_err(err, errlen, "tp: invalid command frame type %u (%u bytes)",
                       ftype, bytes);
        } else if (!err || !err[0]) {
            tp_set_err(err, errlen, "tp: command allocation failed");
        }
        return 0;
    }
    command->type = (ds4_tp_frame_type)ftype;
    return 1;
}

int ds4_tp_send_logits_half(ds4_tp *tp, const float *half, uint32_t count) {
    ds4_tp_frame_header h = { DS4_TP_MAGIC, DS4_TP_FRAME_LOGITS,
                              count * (uint32_t)sizeof(float) };
    return tp_write_full(tp->control_fd, &h, sizeof(h)) &&
           tp_write_full(tp->control_fd, half, count * sizeof(float));
}

int ds4_tp_recv_logits_half(ds4_tp *tp, float *half, uint32_t count) {
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_LOGITS || bytes != count * sizeof(float)) {
        fprintf(stderr, "ds4-tp: bad logits frame (type %u bytes %u)\n", type, bytes);
        return 0;
    }
    return tp_read_full(tp->control_fd, half, count * sizeof(float));
}

int ds4_tp_send_verify(ds4_tp *tp, uint64_t session_id,
                       const int *drafts, uint32_t n) {
    return tp_send_token_command(tp, DS4_TP_FRAME_VERIFY, session_id,
                                 drafts, n);
}

int ds4_tp_send_verify_commit(ds4_tp *tp, int32_t mode, int32_t token_count) {
    struct { int32_t mode; int32_t count; } msg = { mode, token_count };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_VERIFY_COMMIT,
                         &msg, sizeof(msg));
}

int ds4_tp_recv_verify_commit(ds4_tp *tp, int32_t *mode, int32_t *token_count) {
    uint32_t type = 0, bytes = 0;
    struct { int32_t mode; int32_t count; } msg;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_VERIFY_COMMIT || bytes != sizeof(msg) ||
        !tp_read_full(tp->control_fd, &msg, sizeof(msg))) {
        fprintf(stderr, "ds4-tp: bad verify-commit frame (type %u bytes %u)\n",
                type, bytes);
        return 0;
    }
    *mode = msg.mode;
    *token_count = msg.count;
    return 1;
}

int ds4_tp_hash_check(ds4_tp *tp, uint64_t seq, uint64_t hash, char *err, size_t errlen) {
    struct { uint64_t seq; uint64_t hash; } mine = { seq, hash }, theirs;
    if (!tp_send_frame(tp->control_fd, DS4_TP_FRAME_HASH, &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp: hash send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_HASH || bytes != sizeof(theirs) ||
        !tp_read_full(tp->control_fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp: hash recv failed");
        return 0;
    }
    if (theirs.seq != seq || theirs.hash != hash) {
        tp_set_err(err, errlen,
                   "tp: LOCKSTEP DIVERGENCE at seq %llu: local %016llx peer %016llx",
                   (unsigned long long)seq,
                   (unsigned long long)hash, (unsigned long long)theirs.hash);
        return -1;
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Worker main loop.
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t id;
    ds4_session *session;
} ds4_tp_worker_session;

typedef struct {
    ds4_tp_worker_session *v;
    uint32_t len;
    uint32_t cap;
} ds4_tp_worker_sessions;

static int tp_worker_session_index(const ds4_tp_worker_sessions *sessions,
                                   uint64_t id) {
    if (!sessions || id == 0) return -1;
    for (uint32_t i = 0; i < sessions->len; i++) {
        if (sessions->v[i].id == id) return (int)i;
    }
    return -1;
}

static ds4_session *tp_worker_session_find(
        const ds4_tp_worker_sessions *sessions, uint64_t id) {
    const int index = tp_worker_session_index(sessions, id);
    return index >= 0 ? sessions->v[index].session : NULL;
}

static int tp_worker_session_add(ds4_tp_worker_sessions *sessions,
                                 uint64_t id, ds4_session *session) {
    if (!sessions || !session || id == 0 ||
        tp_worker_session_index(sessions, id) >= 0) return 0;
    if (sessions->len == sessions->cap) {
        uint32_t cap = sessions->cap ? sessions->cap * 2u : 8u;
        ds4_tp_worker_session *v =
            realloc(sessions->v, (size_t)cap * sizeof(*v));
        if (!v) return 0;
        sessions->v = v;
        sessions->cap = cap;
    }
    sessions->v[sessions->len++] = (ds4_tp_worker_session){ id, session };
    return 1;
}

static void tp_worker_session_remove(ds4_tp_worker_sessions *sessions,
                                     uint32_t index) {
    if (!sessions || index >= sessions->len) return;
    ds4_session_free(sessions->v[index].session);
    if (index + 1u < sessions->len) {
        memmove(&sessions->v[index], &sessions->v[index + 1u],
                (size_t)(sessions->len - index - 1u) * sizeof(sessions->v[0]));
    }
    sessions->len--;
}

static int tp_worker_send_logits(ds4_tp *tp, ds4_session *session,
                                 float *logits, int vocab) {
    if (!logits || vocab <= 0 || (vocab & 1) != 0) return 0;
    const uint32_t vhalf = (uint32_t)vocab / 2u;
    return ds4_session_copy_logits(session, logits, vocab) == vocab &&
           ds4_tp_send_logits_half(tp, logits + vhalf, vhalf);
}

int ds4_tp_worker_run(ds4_engine *engine, const ds4_tp_options *opt) {
    char err[256] = "";
    ds4_tp_identity id = {
        .gguf_bytes = ds4_engine_model_bytes(engine),
        .model_id = (uint32_t)ds4_engine_model_id(engine),
        .n_layer = (uint32_t)ds4_engine_layer_count(engine),
        .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
        .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
        .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
        .ctx_size = 0, /* adopt the leader's */
    };
    ds4_engine_tp_gate_schedule(engine,
                                &id.gate_slot_start,
                                &id.gate_slot_step,
                                &id.gates_per_token,
                                id.gate_slot_mask);

    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, opt, &id, err, sizeof(err))) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
        return 1;
    }
    if (!ds4_engine_tp_bind(engine, tp, err, sizeof(err))) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
        ds4_tp_free(tp);
        return 1;
    }
    ds4_tp_worker_sessions sessions = {0};
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = ds4_engine_tp_vocab_split(engine) ?
        malloc((size_t)vocab * sizeof(*logits)) : NULL;
    if (ds4_engine_tp_vocab_split(engine) && !logits) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: logits buffer allocation failed");
        ds4_engine_tp_unbind(engine);
        ds4_tp_free(tp);
        return 1;
    }
    ds4_log(stderr, DS4_LOG_OK, "tp worker ready for mirrored sessions");

    int rc = 0;
    bool normal_stop = false;
    ds4_tokens prompt = {0};
    while (1) {
        ds4_tp_command command;
        if (!ds4_tp_recv_command(tp, &command, err, sizeof(err))) {
            ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
            rc = 1;
            break;
        }
        if (command.type == DS4_TP_FRAME_STOP) {
            ds4_log(stderr, DS4_LOG_DEFAULT, "tp worker: leader finished");
            normal_stop = true;
            ds4_tp_command_free(&command);
            break;
        }

        if (command.type == DS4_TP_FRAME_SESSION_CREATE) {
            ds4_session *session = NULL;
            int status = 1;
            if (command.session_id != 0 && command.value > 0 &&
                tp_worker_session_index(&sessions, command.session_id) < 0 &&
                ds4_session_create(&session, engine, command.value) == 0 &&
                tp_worker_session_add(&sessions, command.session_id, session)) {
                /* Pay the first-submit cost before acknowledging creation so
                 * it cannot land in the leader's first timed prefill. */
                ds4_session_gpu_warmup(session);
                status = 0;
            } else if (session) {
                ds4_session_free(session);
            }
            if (!ds4_tp_send_command_ack(tp, command.session_id, status)) {
                rc = 1;
            }
            ds4_tp_command_free(&command);
            if (rc != 0) break;
            continue;
        }

        if (command.type == DS4_TP_FRAME_SESSION_DESTROY) {
            const int index = tp_worker_session_index(&sessions,
                                                      command.session_id);
            const int status = index >= 0 ? 0 : 1;
            if (index >= 0) tp_worker_session_remove(&sessions, (uint32_t)index);
            if (!ds4_tp_send_command_ack(tp, command.session_id, status)) rc = 1;
            ds4_tp_command_free(&command);
            if (rc != 0) break;
            continue;
        }

        ds4_session *session =
            tp_worker_session_find(&sessions, command.session_id);
        if (command.type != DS4_TP_FRAME_EVAL_BATCH &&
            command.type != DS4_TP_FRAME_MIXED_BATCH && !session) {
            ds4_log(stderr, DS4_LOG_ERROR,
                    "tp worker: unknown session %llu for frame %d",
                    (unsigned long long)command.session_id,
                    (int)command.type);
            ds4_tp_command_free(&command);
            rc = 1;
            break;
        }

        if (command.type == DS4_TP_FRAME_SYNC ||
            command.type == DS4_TP_FRAME_SYNC_MULTIMODAL) {
            prompt.len = 0;
            for (uint32_t i = 0; i < command.n_tokens; i++) {
                ds4_tokens_push(&prompt, command.tokens[i]);
            }
            int sync_rc = command.type == DS4_TP_FRAME_SYNC_MULTIMODAL ?
                ds4_session_sync_multimodal(session, &prompt,
                                            command.images, command.n_images,
                                            err, sizeof(err)) :
                ds4_session_sync(session, &prompt, err, sizeof(err));
            if (!ds4_tp_send_command_ack(tp, command.session_id, sync_rc)) {
                rc = 1;
            } else if (sync_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker sync: %s", err);
                rc = 1;
            } else if (ds4_engine_tp_vocab_split(engine) &&
                       !tp_worker_send_logits(tp, session, logits, vocab)) {
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_EVAL) {
            if (ds4_session_eval(session, command.value, err, sizeof(err)) != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker eval: %s", err);
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_GLM_MTP) {
            int spec_rc = ds4_session_glm_tp_spec_cycle(
                session, command.value, command.limit, err, sizeof(err));
            if (!ds4_tp_send_command_ack(tp, command.session_id,
                                         spec_rc < 0 ? 1 : 0) || spec_rc < 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker GLM MTP: %s", err);
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_VERIFY) {
            int spec_rc = ds4_session_tp_spec_cycle(session, command.tokens,
                                                    (int)command.n_tokens,
                                                    err, sizeof(err));
            if (spec_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker verify: %s", err);
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_REWIND) {
            ds4_session_rewind(session, command.value);
        } else if (command.type == DS4_TP_FRAME_INVALIDATE) {
            ds4_session_invalidate(session);
        } else if (command.type == DS4_TP_FRAME_EVAL_BATCH ||
                   command.type == DS4_TP_FRAME_MIXED_BATCH) {
            ds4_decode_item *items =
                calloc(command.n_items, sizeof(*items));
            bool mapped = items != NULL;
            for (uint32_t i = 0; mapped && i < command.n_items; i++) {
                items[i].session = tp_worker_session_find(
                    &sessions, command.items[i].session_id);
                items[i].token = command.items[i].token;
                mapped = items[i].session != NULL;
            }
            ds4_session *prefill = NULL;
            if (mapped && command.type == DS4_TP_FRAME_MIXED_BATCH) {
                prefill = tp_worker_session_find(&sessions,
                                                 command.session_id);
                mapped = prefill != NULL;
                prompt.len = 0;
                for (uint32_t i = 0; mapped && i < command.n_tokens; i++) {
                    ds4_tokens_push(&prompt, command.tokens[i]);
                }
            }
            int batch_rc = 1;
            if (mapped && command.type == DS4_TP_FRAME_EVAL_BATCH) {
                batch_rc = ds4_sessions_eval_batch(
                    items, (int)command.n_items, err, sizeof(err));
            } else if (mapped) {
                batch_rc = ds4_sessions_eval_batch_with_prefill(
                    items, (int)command.n_items, prefill, &prompt,
                    err, sizeof(err));
            }
            if (!ds4_tp_send_command_ack(tp, command.session_id, batch_rc)) {
                rc = 1;
            } else if (batch_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR,
                        "tp worker batch: %s", err[0] ? err : "failed");
                rc = 1;
            } else if (ds4_engine_tp_vocab_split(engine)) {
                if (prefill &&
                    !tp_worker_send_logits(tp, prefill, logits, vocab)) {
                    rc = 1;
                }
                for (uint32_t i = 0; rc == 0 && i < command.n_items; i++) {
                    if (!tp_worker_send_logits(tp, items[i].session,
                                               logits, vocab)) rc = 1;
                }
            }
            free(items);
        } else {
            ds4_log(stderr, DS4_LOG_ERROR, "tp worker: unexpected frame %d",
                    (int)command.type);
            rc = 1;
        }
        ds4_tp_command_free(&command);
        if (rc != 0) break;
    }
    ds4_tokens_free(&prompt);
    while (sessions.len != 0) {
        tp_worker_session_remove(&sessions, sessions.len - 1u);
    }
    free(sessions.v);
    free(logits);
    /* NHI teardown is receiver-last: worker quiesces and closes its GPU
     * data plane while the leader keeps its device open, then acknowledges
     * over the still-live TCP control channel. */
    ds4_engine_tp_unbind(engine);
    if (normal_stop && ds4_tp_is_nhi(tp) &&
        !tp_send_frame(tp->control_fd, DS4_TP_FRAME_NHI_CLOSED, NULL, 0))
        rc = 1;
    ds4_tp_free(tp);
    return rc;
}
