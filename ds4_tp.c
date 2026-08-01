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
#include <time.h>
#include <unistd.h>

#include "ds4_tp.h"

#if defined(__APPLE__) && defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#define DS4_TP_HAVE_VERBS 1
#endif
#endif

#define DS4_TP_MAGIC UINT32_C(0x44533454) /* "DS4T" */
#define DS4_TP_BATCH_MAGIC UINT32_C(0x44533442) /* "DS4B" */
#define DS4_TP_PROTOCOL_VERSION 7u

/* Default gate timeout is generous: the first gate after a sync waits for
 * the peer's whole (possibly cold page cache) prefill. */
#define DS4_TP_DEFAULT_TIMEOUT_SEC 300

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
} ds4_tp_verbs_api;

/* AppleThunderboltRDMA quirks (validated with scratchpad probes,
 * 2026-07-06): only UC queue pairs exist (RC/UD: ENOTSUP); RDMA WRITE work
 * requests are accepted but never execute, so the data plane is two-sided
 * SEND/RECV like Apple's own JACCL; messages above 16KB are not delivered;
 * RTR requires GRH addressing with the IPv4-mapped GID that appears only
 * once the Thunderbolt member interface has an IPv4 address of its own.
 * UC delivery is in-order and the gate sequence is globally deterministic
 * (86 gates per token, fixed order). After any initial bulk prefill, decode
 * keeps a receive window posted by sequence number: recv for seq s lands in
 * the slab in-slot (s-1) % slots and its completion IS the arrival signal. */
#define DS4_TP_RDMA_MAX_MSG 16384
#define DS4_TP_RDMA_RECV_WINDOW 16
#define DS4_TP_RDMA_BULK_SLOTS 64
#define DS4_TP_RDMA_BULK_WR_TAG (UINT64_C(1) << 63)

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
    pthread_mutex_t post_lock;
} ds4_tp_rdma;
#endif

struct ds4_tp {
    ds4_tp_options opt;
    int rank;                   /* 0 leader, 1 worker */
    int control_fd;
    int data_fd;                /* TCP fallback, headers, and verify gates */
    bool rdma_active;
    uint32_t peer_ctx;
    uint32_t n_layer;
    uint32_t n_embd;
    uint64_t vec_bytes;
    uint32_t n_slots;
    /* Decode gate schedule (see ds4_tp_identity). */
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
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

static void tp_socket_tune(int fd) {
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
        "  --transport <auto|rdma|tcp> Gate transport (default auto).\n"
        "  --rdma-device <name>        Select a verbs device such as rdma_en1.\n"
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
        else {
            tp_set_err(err, errlen, "invalid %s value: %s", arg, v);
            return DS4_TP_CLI_ERROR;
        }
    } else if (!strcmp(arg, "--rdma-device")) {
        if (i + 1 >= argc) goto missing;
        opt->rdma_device = argv[++i];
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
            opt->tp.rdma_device || opt->tp.rdma_gid_index_set ||
            opt->tp.glm_token_prefill || opt->tp.debug_hash != 0) {
            tp_set_err(err, errlen,
                       "tensor-parallel options require --tensor-parallel and --role");
            return 0;
        }
        return 1;
    }
    if (opt->backend != DS4_BACKEND_METAL) {
        tp_set_err(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }
    if (opt->ssd_streaming) {
        tp_set_err(err, errlen, "tensor parallelism requires resident weights (no --ssd-streaming)");
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
    r->cq = r->api.create_cq(r->ctx, 512, NULL, NULL, 0);
    if (!r->cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq failed");
        return 0;
    }
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = r->cq;
    qia.recv_cq = r->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr = 256;
    qia.cap.max_recv_wr = 64;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC): %s", strerror(errno));
        return 0;
    }
    r->max_inline = qia.cap.max_inline_data;

    pthread_mutex_init(&r->post_lock, NULL);
    return 1;
}

static int tp_rdma_post_gate_recv(ds4_tp *tp, uint64_t seq);

static int tp_rdma_register_and_exchange(ds4_tp *tp, char *err, size_t errlen) {
    ds4_tp_rdma *r = &tp->rdma;
    r->mr = r->api.reg_mr(r->pd, tp->slab, tp->slab_bytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE);
    if (!r->mr) {
        tp_set_err(err, errlen, "tp rdma: reg_mr(%llu bytes): %s",
                   (unsigned long long)tp->slab_bytes, strerror(errno));
        return 0;
    }
    ds4_tp_rdma_info mine = {0};
    mine.slab_base = (uint64_t)(uintptr_t)tp->slab;
    mine.rkey = r->mr->rkey;
    mine.qpn = r->qp->qp_num;
    mine.psn = (uint32_t)(getpid() ^ (uintptr_t)tp) & 0xffffff;
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
            if (wc[i].wr_id > r->recv_done) r->recv_done = wc[i].wr_id;
        } else if (r->send_outstanding > 0) {
            r->send_outstanding--;
        }
    }
    return 1;
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
    uint64_t off = 0;
    while (off < tp->vec_bytes) {
        const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
            DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        const int last = off + len == tp->vec_bytes;
        struct ibv_sge sge;
        struct ibv_recv_wr wr, *bad = NULL;
        memset(&wr, 0, sizeof(wr));
        sge.addr = base + off;
        sge.length = (uint32_t)len;
        sge.lkey = r->mr->lkey;
        wr.wr_id = last ? seq : 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        if (ibv_post_recv(r->qp, &wr, &bad) != 0) {
            fprintf(stderr, "ds4-tp: rdma post_recv(seq %llu off %llu): %s\n",
                    (unsigned long long)seq, (unsigned long long)off,
                    strerror(errno));
            return 0;
        }
        off += len;
    }
    return 1;
}

/* One decode gate: ensure the receive window is armed, send our partial,
 * wait for the peer's receive completion, and advance the window. */
static int tp_rdma_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    ds4_tp_rdma *r = &tp->rdma;
    const uint32_t slot = layer * DS4_TP_GATES_PER_LAYER + gate;
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
    for (uint64_t off = 0; ok && off < tp->vec_bytes; ) {
        const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
            DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        struct ibv_sge sge;
        struct ibv_send_wr wr, *bad = NULL;
        memset(&wr, 0, sizeof(wr));
        sge.addr = send_base + off;
        sge.length = (uint32_t)len;
        sge.lkey = r->mr->lkey;
        wr.wr_id = seq;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        ok = ibv_post_send(r->qp, &wr, &bad) == 0;
        if (!ok) {
            fprintf(stderr, "ds4-tp: rdma post_send: %s\n", strerror(errno));
        } else {
            r->send_outstanding++;
        }
        off += len;
    }

    double deadline = 0.0;
    uint32_t peer_poll = 0;
    while (ok && r->recv_done < seq) {
        ok = tp_rdma_drain_cq(tp);
        if (ok && (peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
            fprintf(stderr, "ds4-tp: peer disconnected during RDMA gate\n");
            ok = 0;
        }
        if (deadline == 0.0) deadline = tp_now_sec() + (double)tp->timeout_sec;
        else if (tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: timeout waiting gate seq %llu (recv_done %llu)\n",
                    (unsigned long long)seq, (unsigned long long)r->recv_done);
            ok = 0;
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
    const double deadline = tp_now_sec() + (double)tp->timeout_sec;
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
static int tp_rdma_big_gate_exchange(ds4_tp *tp,
                                     const void *out,
                                     void *in,
                                     uint64_t bytes) {
    ds4_tp_rdma *r = &tp->rdma;
    if (!tp_rdma_big_gate_capable(tp) || r->recv_window_active) return 0;

    /* Payloads already inside the registered slab (verify batches) can ride
     * directly. Ordinary prefill tensors use the idle verify regions as
     * registered staging because their standalone MTLBuffers are not in the
     * NIC memory region. */
    const uintptr_t slab_lo = (uintptr_t)tp->slab;
    const uintptr_t slab_hi = slab_lo + tp->slab_bytes;
    const uintptr_t out_lo = (uintptr_t)out;
    const uintptr_t in_lo = (uintptr_t)in;
    const bool direct =
        out_lo >= slab_lo && out_lo <= slab_hi && bytes <= slab_hi - out_lo &&
        in_lo >= slab_lo && in_lo <= slab_hi && bytes <= slab_hi - in_lo;
    uint8_t *stage_send = tp->slab + tp->batch_out_off;
    uint8_t *stage_recv = tp->slab + tp->batch_in_off;
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t remaining = bytes - off;
        uint32_t chunks = (uint32_t)((remaining + DS4_TP_RDMA_MAX_MSG - 1u) /
                                     DS4_TP_RDMA_MAX_MSG);
        if (chunks > DS4_TP_RDMA_BULK_SLOTS)
            chunks = DS4_TP_RDMA_BULK_SLOTS;

        uint32_t lens[DS4_TP_RDMA_BULK_SLOTS];
        uint64_t chunk_off[DS4_TP_RDMA_BULK_SLOTS];
        uint64_t round_bytes = 0;
        for (uint32_t i = 0; i < chunks; i++) {
            const uint64_t left = remaining - round_bytes;
            lens[i] = (uint32_t)(left > DS4_TP_RDMA_MAX_MSG ?
                                 DS4_TP_RDMA_MAX_MSG : left);
            chunk_off[i] = direct ? round_bytes :
                (uint64_t)i * DS4_TP_RDMA_MAX_MSG;
            if (!direct) {
                memcpy(stage_send + chunk_off[i],
                       (const uint8_t *)out + off + round_bytes, lens[i]);
            }
            round_bytes += lens[i];
        }

        struct ibv_sge recv_sge[DS4_TP_RDMA_BULK_SLOTS];
        struct ibv_recv_wr recv_wr[DS4_TP_RDMA_BULK_SLOTS];
        memset(recv_wr, 0, sizeof(recv_wr));
        for (uint32_t i = 0; i < chunks; i++) {
            recv_sge[i] = (struct ibv_sge) {
                .addr = direct ? in_lo + off + chunk_off[i] :
                                 (uintptr_t)(stage_recv + chunk_off[i]),
                .length = lens[i],
                .lkey = r->mr->lkey,
            };
            recv_wr[i].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            recv_wr[i].sg_list = &recv_sge[i];
            recv_wr[i].num_sge = 1;
            recv_wr[i].next = i + 1u < chunks ? &recv_wr[i + 1u] : NULL;
        }
        struct ibv_recv_wr *bad_recv = NULL;
        if (ibv_post_recv(r->qp, recv_wr, &bad_recv) != 0) {
            fprintf(stderr, "ds4-tp: bulk rdma post_recv: %s\n",
                    strerror(errno));
            return 0;
        }
        atomic_thread_fence(memory_order_release);
        struct ibv_sge send_sge[DS4_TP_RDMA_BULK_SLOTS];
        struct ibv_send_wr send_wr[DS4_TP_RDMA_BULK_SLOTS];
        memset(send_wr, 0, sizeof(send_wr));
        for (uint32_t i = 0; i < chunks; i++) {
            send_sge[i] = (struct ibv_sge) {
                .addr = direct ? out_lo + off + chunk_off[i] :
                                 (uintptr_t)(stage_send + chunk_off[i]),
                .length = lens[i],
                .lkey = r->mr->lkey,
            };
            send_wr[i].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            send_wr[i].sg_list = &send_sge[i];
            send_wr[i].num_sge = 1;
            send_wr[i].opcode = IBV_WR_SEND;
            send_wr[i].send_flags = i + 1u == chunks ? IBV_SEND_SIGNALED : 0;
            send_wr[i].next = i + 1u < chunks ? &send_wr[i + 1u] : NULL;
        }
        struct ibv_send_wr *bad_send = NULL;
        if (ibv_post_send(r->qp, send_wr, &bad_send) != 0) {
            fprintf(stderr, "ds4-tp: bulk rdma post_send: %s\n",
                    strerror(errno));
            return 0;
        }

        uint32_t recv_done = 0;
        int send_done = 0;
        const double deadline = tp_now_sec() + (double)tp->timeout_sec;
        uint32_t peer_poll = 0;
        while (recv_done < chunks || !send_done) {
            struct ibv_wc wc[DS4_TP_RDMA_BULK_SLOTS + 1u];
            int n = ibv_poll_cq(r->cq,
                               (int)(DS4_TP_RDMA_BULK_SLOTS + 1u), wc);
            if (n < 0) return 0;
            for (int i = 0; i < n; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr,
                            "ds4-tp: bulk rdma completion error: %s\n",
                            tp_wc_status_str(wc[i].status));
                    return 0;
                }
                if ((wc[i].wr_id & DS4_TP_RDMA_BULK_WR_TAG) == 0) {
                    /* A final latency-QP send completion can remain queued
                     * when a later prompt starts a bulk gate. */
                    if (wc[i].opcode & IBV_WC_RECV) {
                        if (wc[i].wr_id > r->recv_done)
                            r->recv_done = wc[i].wr_id;
                    } else if (r->send_outstanding > 0) {
                        r->send_outstanding--;
                    }
                    continue;
                }
                if (wc[i].opcode & IBV_WC_RECV) recv_done++;
                else send_done = 1;
            }
            if ((peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp)) {
                fprintf(stderr,
                        "ds4-tp: peer disconnected during bulk RDMA gate\n");
                return 0;
            }
            if (tp_now_sec() > deadline) {
                fprintf(stderr,
                        "ds4-tp: timeout waiting for bulk RDMA round "
                        "(%u/%u recvs, send=%d)\n",
                        recv_done, chunks, send_done);
                return 0;
            }
        }
        atomic_thread_fence(memory_order_acquire);
        if (!direct) {
            round_bytes = 0;
            for (uint32_t i = 0; i < chunks; i++) {
                memcpy((uint8_t *)in + off + round_bytes,
                       stage_recv + chunk_off[i], lens[i]);
                round_bytes += lens[i];
            }
        }
        off += round_bytes;
    }
    return 1;
}

static void tp_rdma_close(ds4_tp *tp) {
    ds4_tp_rdma *r = &tp->rdma;
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
        theirs.gates_per_token != mine.gates_per_token) {
        tp_set_err(err, errlen,
                   "tp hello: model mismatch (peer gguf=%llu id=%u layers=%u embd=%u "
                   "vocab=%u qbits=%u)",
                   (unsigned long long)theirs.gguf_bytes, theirs.model_id,
                   theirs.n_layer, theirs.n_embd, theirs.n_vocab, theirs.quant_bits);
        return 0;
    }
    tp->peer_ctx = theirs.ctx_size;
    tp->n_layer = id->n_layer;
    tp->n_embd = id->n_embd;
    tp->vec_bytes = (uint64_t)id->n_embd * sizeof(float);
    tp->n_slots = id->n_layer * DS4_TP_GATES_PER_LAYER;
    tp->gate_slot_start = id->gate_slot_start;
    tp->gate_slot_step = id->gate_slot_step;
    tp->gates_per_token = id->gates_per_token;
    tp_slab_layout(tp);
    /* Transport decision: RDMA only when both sides can. */
    int want_rdma = tp->opt.transport != DS4_TP_TRANSPORT_TCP;
    tp->rdma_active = want_rdma && rdma_ok && theirs.rdma_ok;
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
    tp->rank = opt->role == DS4_TP_LEADER ? 0 : 1;
    tp->control_fd = -1;
    tp->data_fd = -1;
    tp->timeout_sec = DS4_TP_DEFAULT_TIMEOUT_SEC;
    const char *tmo = getenv("DS4_TP_TIMEOUT_SEC");
    if (tmo) tp->timeout_sec = (uint64_t)atoi(tmo);

    int rdma_ok = 0;
#ifdef DS4_TP_HAVE_VERBS
    if (opt->transport != DS4_TP_TRANSPORT_TCP &&
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
    tp_socket_tune(tp->control_fd);

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
        tp_socket_tune(tp->data_fd);
    }
    if (listener >= 0) close(listener);
    fprintf(stderr, "ds4-tp: %s connected, transport=%s\n",
            tp->rank == 0 ? "worker" : "leader",
            tp->rdma_active ? "rdma" : "tcp");
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
    if (tp->rdma_active) return tp_rdma_register_and_exchange(tp, err, errlen);
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
uint32_t ds4_tp_peer_ctx(const ds4_tp *tp) { return tp->peer_ctx; }
bool ds4_tp_failed(const ds4_tp *tp) {
    return tp && atomic_load_explicit(&tp->failed, memory_order_acquire);
}
void ds4_tp_mark_failed(ds4_tp *tp) {
    if (tp) atomic_store_explicit(&tp->failed, true, memory_order_release);
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
int ds4_tp_batch_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows,
                               uint64_t seq) {
    if (tp->data_fd < 0 || rows == 0 || rows > DS4_TP_BATCH_MAX_ROWS) return 0;
    const uint64_t bytes = (uint64_t)rows * tp->vec_bytes;
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer,
                             (uint16_t)rows, seq };
#ifdef DS4_TP_HAVE_VERBS
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
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer, 0xB16u, seq };
    if (!tp_write_full(tp->data_fd, &h, sizeof(h))) return 0;
    ds4_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd, &ph, sizeof(ph))) return 0;
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
        return tp_rdma_big_gate_exchange(tp, out, in, bytes);
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
    if (!tp || (!tokens && count != 0) || bytes64 > UINT32_MAX) return 0;
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

int ds4_tp_send_eval(ds4_tp *tp, uint64_t session_id,
                     uint64_t seq, int token) {
    ds4_tp_eval_command msg = { session_id, seq, (int32_t)token, 0 };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_EVAL, &msg, sizeof(msg));
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
    if (!tp || !items || count == 0 || bytes64 > UINT32_MAX) return 0;
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
    if (!tp || !prompt || prompt_count == 0 || !items || count == 0 ||
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
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_STOP, NULL, 0);
}

void ds4_tp_command_free(ds4_tp_command *command) {
    if (!command) return;
    free(command->tokens);
    free(command->items);
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
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

int ds4_tp_recv_command(ds4_tp *tp, ds4_tp_command *command,
                        char *err, size_t errlen) {
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
    uint32_t ftype = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &ftype, &bytes)) {
        tp_set_err(err, errlen, "tp: control channel closed");
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
    case DS4_TP_FRAME_EVAL: {
        ds4_tp_eval_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->seq = msg.seq;
        command->value = msg.token;
        break;
    }
    case DS4_TP_FRAME_EVAL_BATCH: {
        ds4_tp_batch_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t want = sizeof(h) +
                              (uint64_t)h.count * sizeof(ds4_tp_batch_item);
        if (h.count == 0 || want != bytes) { ok = 0; break; }
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
        if (h.prompt_count == 0 || h.item_count == 0 || want != bytes) {
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
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_LOGITS,
                         half, count * sizeof(float));
}

int ds4_tp_recv_logits_half(ds4_tp *tp, float *half, uint32_t count) {
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_LOGITS || bytes != count * sizeof(float)) {
        fprintf(stderr, "ds4-tp: bad logits frame (type %u bytes %u)\n", type, bytes);
        return 0;
    }
    return tp_read_full(tp->control_fd, half, bytes);
}

int ds4_tp_send_verify(ds4_tp *tp, uint64_t session_id,
                       const int *drafts, uint32_t n) {
    return tp_send_token_command(tp, DS4_TP_FRAME_VERIFY, session_id,
                                 drafts, n);
}

int ds4_tp_send_verify_commit(ds4_tp *tp, int32_t full_accept, int32_t replay_n) {
    struct { int32_t full; int32_t replay; } msg = { full_accept, replay_n };
    return tp_send_frame(tp->control_fd, DS4_TP_FRAME_VERIFY_COMMIT,
                         &msg, sizeof(msg));
}

int ds4_tp_recv_verify_commit(ds4_tp *tp, int32_t *full_accept, int32_t *replay_n) {
    uint32_t type = 0, bytes = 0;
    struct { int32_t full; int32_t replay; } msg;
    if (!tp_read_frame_header(tp->control_fd, &type, &bytes) ||
        type != DS4_TP_FRAME_VERIFY_COMMIT || bytes != sizeof(msg) ||
        !tp_read_full(tp->control_fd, &msg, sizeof(msg))) {
        fprintf(stderr, "ds4-tp: bad verify-commit frame (type %u bytes %u)\n",
                type, bytes);
        return 0;
    }
    *full_accept = msg.full;
    *replay_n = msg.replay;
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
                                &id.gates_per_token);

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
        ds4_tp_free(tp);
        return 1;
    }
    ds4_log(stderr, DS4_LOG_OK, "tp worker ready for mirrored sessions");

    int rc = 0;
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

        if (command.type == DS4_TP_FRAME_SYNC) {
            prompt.len = 0;
            for (uint32_t i = 0; i < command.n_tokens; i++) {
                ds4_tokens_push(&prompt, command.tokens[i]);
            }
            int sync_rc = ds4_session_sync(session, &prompt, err, sizeof(err));
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
    ds4_tp_free(tp);
    return rc;
}
