/* Pipeline-parallel RPC transport for ds4.  See ds4_rpc.h for protocol
 * overview.  Plain blocking sockets, one connection per pair, framing is
 * length-prefixed.  Errors fail the whole connection rather than retrying;
 * the caller (the head session) is expected to surface a clear failure to
 * the user, who will restart the worker.  Reconnect/keepalive is a Phase 5
 * concern. */

#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#define _BSD_SOURCE
#define _DEFAULT_SOURCE

#include "ds4_rpc.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

struct ds4_rpc_handle {
    int fd;
    int listen_fd;   /* held until first accept, then closed; -1 otherwise */
};

static void rpc_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

/* Endianness helpers.  The wire is always little-endian.  On a likely-LE
 * Mac/x86 host these compile down to nothing; the explicit form lets a
 * future BE port work without surprises. */
static void put_u32_le(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)v;
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}
static void put_u64_le(uint8_t out[8], uint64_t v) {
    for (int i = 0; i < 8; i++) out[i] = (uint8_t)(v >> (i * 8));
}
static uint32_t get_u32_le(const uint8_t in[4]) {
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) | ((uint32_t)in[3] << 24);
}
static uint64_t get_u64_le(const uint8_t in[8]) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)in[i] << (i * 8);
    return v;
}

static int io_read_full(int fd, void *buf, size_t n, char *err, size_t errlen) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r > 0) { got += (size_t)r; continue; }
        if (r == 0) {
            rpc_set_err(err, errlen, "rpc: peer closed mid-frame after %zu/%zu bytes", got, n);
            return 1;
        }
        if (errno == EINTR) continue;
        rpc_set_err(err, errlen, "rpc: read: %s", strerror(errno));
        return 1;
    }
    return 0;
}

static int io_write_full(int fd, const void *buf, size_t n, char *err, size_t errlen) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, p + sent, n - sent);
        if (w > 0) { sent += (size_t)w; continue; }
        if (w < 0 && errno == EINTR) continue;
        rpc_set_err(err, errlen, "rpc: write: %s", strerror(errno));
        return 1;
    }
    return 0;
}

/* Frame layout: u32 length (excluding self) | u8 op | u8 reserved | u16 reserved
 * | payload bytes.  The opcode-payload pairing is fixed per opcode and known
 * to both sides, so we do not embed type tags. */
#define RPC_FRAME_HDR_BYTES 8u

static int frame_write(int fd, uint8_t op, const void *payload, uint32_t payload_bytes,
                       char *err, size_t errlen) {
    uint8_t hdr[RPC_FRAME_HDR_BYTES];
    put_u32_le(hdr, payload_bytes + (RPC_FRAME_HDR_BYTES - 4u));
    hdr[4] = op;
    hdr[5] = 0;
    hdr[6] = 0;
    hdr[7] = 0;
    if (io_write_full(fd, hdr, sizeof(hdr), err, errlen)) return 1;
    if (payload_bytes && io_write_full(fd, payload, payload_bytes, err, errlen)) return 1;
    return 0;
}

static int frame_read_header(int fd, uint8_t *out_op, uint32_t *out_payload_bytes,
                             char *err, size_t errlen) {
    uint8_t hdr[RPC_FRAME_HDR_BYTES];
    if (io_read_full(fd, hdr, sizeof(hdr), err, errlen)) return 1;
    const uint32_t frame_after_len = get_u32_le(hdr);
    if (frame_after_len < (RPC_FRAME_HDR_BYTES - 4u)) {
        rpc_set_err(err, errlen, "rpc: truncated frame header (%u bytes)", frame_after_len);
        return 1;
    }
    *out_op = hdr[4];
    *out_payload_bytes = frame_after_len - (RPC_FRAME_HDR_BYTES - 4u);
    return 0;
}

/* Connection setup. */

static int set_low_latency(int fd) {
    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    (void)setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    return 0;
}

int ds4_rpc_dial(const char *host, uint16_t port,
                 ds4_rpc_handle **out, char *err, size_t errlen) {
    if (!host || !out) {
        rpc_set_err(err, errlen, "rpc_dial: null arg");
        return 1;
    }
    *out = NULL;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);

    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, port_str, &hints, &res);
    if (gai != 0) {
        rpc_set_err(err, errlen, "rpc: getaddrinfo(%s:%s): %s", host, port_str, gai_strerror(gai));
        return 1;
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        rpc_set_err(err, errlen, "rpc: connect(%s:%u) failed", host, (unsigned)port);
        return 1;
    }
    set_low_latency(fd);

    ds4_rpc_handle *h = (ds4_rpc_handle *)calloc(1, sizeof(*h));
    if (!h) {
        close(fd);
        rpc_set_err(err, errlen, "rpc: out of memory");
        return 1;
    }
    h->fd = fd;
    h->listen_fd = -1;
    *out = h;
    return 0;
}

int ds4_rpc_listen_one(const char *bind_host, uint16_t port,
                       ds4_rpc_handle **out, char *err, size_t errlen) {
    if (!out) {
        rpc_set_err(err, errlen, "rpc_listen: null out");
        return 1;
    }
    *out = NULL;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%u", (unsigned)port);

    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    struct addrinfo *res = NULL;
    int gai = getaddrinfo(bind_host && bind_host[0] ? bind_host : NULL,
                          port_str, &hints, &res);
    if (gai != 0) {
        rpc_set_err(err, errlen, "rpc: getaddrinfo bind: %s", gai_strerror(gai));
        return 1;
    }

    int listen_fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        listen_fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (listen_fd < 0) continue;
        int one = 1;
        (void)setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(listen_fd, ai->ai_addr, ai->ai_addrlen) == 0 &&
            listen(listen_fd, 1) == 0) break;
        close(listen_fd);
        listen_fd = -1;
    }
    freeaddrinfo(res);
    if (listen_fd < 0) {
        rpc_set_err(err, errlen, "rpc: bind/listen on :%u failed: %s", (unsigned)port, strerror(errno));
        return 1;
    }

    fprintf(stderr, "ds4-rpc: listening on %s:%u, awaiting head\n",
            bind_host && bind_host[0] ? bind_host : "*", (unsigned)port);

    struct sockaddr_storage peer;
    socklen_t peerlen = sizeof(peer);
    int fd = accept(listen_fd, (struct sockaddr *)&peer, &peerlen);
    if (fd < 0) {
        rpc_set_err(err, errlen, "rpc: accept: %s", strerror(errno));
        close(listen_fd);
        return 1;
    }
    set_low_latency(fd);

    char peer_host[256] = "?";
    char peer_port[32] = "?";
    (void)getnameinfo((struct sockaddr *)&peer, peerlen,
                      peer_host, sizeof(peer_host),
                      peer_port, sizeof(peer_port),
                      NI_NUMERICHOST | NI_NUMERICSERV);
    fprintf(stderr, "ds4-rpc: accepted head from %s:%s\n", peer_host, peer_port);

    close(listen_fd);

    ds4_rpc_handle *h = (ds4_rpc_handle *)calloc(1, sizeof(*h));
    if (!h) {
        close(fd);
        rpc_set_err(err, errlen, "rpc: out of memory");
        return 1;
    }
    h->fd = fd;
    h->listen_fd = -1;
    *out = h;
    return 0;
}

void ds4_rpc_close(ds4_rpc_handle *h) {
    if (!h) return;
    if (h->fd >= 0) close(h->fd);
    if (h->listen_fd >= 0) close(h->listen_fd);
    free(h);
}

int ds4_rpc_fd(const ds4_rpc_handle *h) {
    return h ? h->fd : -1;
}

/* Config (de)serialization. */
#define RPC_CFG_BYTES (4u * 8u + 8u + 32u)

static void pack_config(uint8_t out[RPC_CFG_BYTES], const ds4_rpc_config *c) {
    uint8_t *p = out;
    put_u32_le(p, c->version);            p += 4;
    put_u32_le(p, c->n_layer_total);      p += 4;
    put_u32_le(p, c->n_embd);             p += 4;
    put_u32_le(p, c->n_hc);               p += 4;
    put_u32_le(p, c->n_vocab);            p += 4;
    put_u32_le(p, c->routed_quant_bits);  p += 4;
    put_u32_le(p, c->tail_layer_start);   p += 4;
    put_u32_le(p, c->tail_layer_end);     p += 4;
    put_u64_le(p, c->model_file_bytes);   p += 8;
    memcpy(p, c->model_sample, 32);       p += 32;
    (void)p;
}

static void unpack_config(ds4_rpc_config *c, const uint8_t in[RPC_CFG_BYTES]) {
    const uint8_t *p = in;
    c->version            = get_u32_le(p); p += 4;
    c->n_layer_total      = get_u32_le(p); p += 4;
    c->n_embd             = get_u32_le(p); p += 4;
    c->n_hc               = get_u32_le(p); p += 4;
    c->n_vocab            = get_u32_le(p); p += 4;
    c->routed_quant_bits  = get_u32_le(p); p += 4;
    c->tail_layer_start   = get_u32_le(p); p += 4;
    c->tail_layer_end     = get_u32_le(p); p += 4;
    c->model_file_bytes   = get_u64_le(p); p += 8;
    memcpy(c->model_sample, p, 32);
}

static bool configs_match(const ds4_rpc_config *a, const ds4_rpc_config *b) {
    if (a->version           != b->version)          return false;
    if (a->n_layer_total     != b->n_layer_total)    return false;
    if (a->n_embd            != b->n_embd)           return false;
    if (a->n_hc              != b->n_hc)             return false;
    if (a->n_vocab           != b->n_vocab)          return false;
    if (a->routed_quant_bits != b->routed_quant_bits) return false;
    if (a->tail_layer_start  != b->tail_layer_start) return false;
    if (a->tail_layer_end    != b->tail_layer_end)   return false;
    if (a->model_file_bytes  != b->model_file_bytes) return false;
    if (memcmp(a->model_sample, b->model_sample, 32) != 0) return false;
    return true;
}

/* Magic preamble: both sides write "DRPC" + version once at handshake.  The
 * version is duplicated into the config so an old client connecting to a new
 * server (or vice versa) gets a clear mismatch message rather than wedging on
 * unexpected payload sizes. */
static int write_magic(int fd, char *err, size_t errlen) {
    uint8_t buf[8];
    put_u32_le(buf,     DS4_RPC_MAGIC);
    put_u32_le(buf + 4, DS4_RPC_VERSION);
    return io_write_full(fd, buf, sizeof(buf), err, errlen);
}

static int read_magic(int fd, char *err, size_t errlen) {
    uint8_t buf[8];
    if (io_read_full(fd, buf, sizeof(buf), err, errlen)) return 1;
    const uint32_t magic = get_u32_le(buf);
    const uint32_t ver = get_u32_le(buf + 4);
    if (magic != DS4_RPC_MAGIC) {
        rpc_set_err(err, errlen, "rpc: bad magic %#x, expected %#x", magic, DS4_RPC_MAGIC);
        return 1;
    }
    if (ver != DS4_RPC_VERSION) {
        rpc_set_err(err, errlen, "rpc: protocol version mismatch %u vs %u", ver, DS4_RPC_VERSION);
        return 1;
    }
    return 0;
}

int ds4_rpc_handshake_client(ds4_rpc_handle *h, const ds4_rpc_config *cfg,
                             char *err, size_t errlen) {
    if (!h || !cfg) { rpc_set_err(err, errlen, "handshake: null arg"); return 1; }
    if (write_magic(h->fd, err, errlen)) return 1;

    uint8_t cbuf[RPC_CFG_BYTES];
    pack_config(cbuf, cfg);
    if (frame_write(h->fd, DS4_RPC_OP_HELLO_CLIENT, cbuf, sizeof(cbuf), err, errlen)) return 1;

    uint8_t op = 0;
    uint32_t payload_bytes = 0;
    if (frame_read_header(h->fd, &op, &payload_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_HELLO_SERVER) {
        rpc_set_err(err, errlen, "handshake: server replied op=%u, expected HELLO_SERVER", op);
        return 1;
    }
    if (payload_bytes < 4) {
        rpc_set_err(err, errlen, "handshake: server reply too short (%u bytes)", payload_bytes);
        return 1;
    }
    uint8_t status_buf[4];
    if (io_read_full(h->fd, status_buf, sizeof(status_buf), err, errlen)) return 1;
    const uint32_t status = get_u32_le(status_buf);

    if (status != 0) {
        char *msg = NULL;
        const uint32_t msg_bytes = payload_bytes - 4u;
        if (msg_bytes > 0 && msg_bytes < 4096) {
            msg = (char *)malloc((size_t)msg_bytes + 1u);
            if (msg) {
                if (io_read_full(h->fd, msg, msg_bytes, err, errlen) == 0) {
                    msg[msg_bytes] = '\0';
                    rpc_set_err(err, errlen, "rpc: server rejected handshake: %s", msg);
                }
                free(msg);
                return 1;
            }
        }
        /* Drain any leftover bytes silently so the caller's error is clean. */
        uint8_t tmp[256];
        uint32_t remaining = msg_bytes;
        while (remaining > 0) {
            uint32_t chunk = remaining < sizeof(tmp) ? remaining : (uint32_t)sizeof(tmp);
            if (io_read_full(h->fd, tmp, chunk, NULL, 0)) break;
            remaining -= chunk;
        }
        rpc_set_err(err, errlen, "rpc: server rejected handshake (status=%u)", status);
        return 1;
    }
    return 0;
}

int ds4_rpc_handshake_server(ds4_rpc_handle *h, const ds4_rpc_config *cfg,
                             ds4_rpc_config *peer, char *err, size_t errlen) {
    if (!h || !cfg) { rpc_set_err(err, errlen, "handshake: null arg"); return 1; }
    if (read_magic(h->fd, err, errlen)) return 1;

    uint8_t op = 0;
    uint32_t payload_bytes = 0;
    if (frame_read_header(h->fd, &op, &payload_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_HELLO_CLIENT) {
        rpc_set_err(err, errlen, "handshake: client opened with op=%u, expected HELLO_CLIENT", op);
        return 1;
    }
    if (payload_bytes != RPC_CFG_BYTES) {
        rpc_set_err(err, errlen, "handshake: client config payload %u bytes, expected %u",
                    payload_bytes, RPC_CFG_BYTES);
        return 1;
    }
    uint8_t cbuf[RPC_CFG_BYTES];
    if (io_read_full(h->fd, cbuf, sizeof(cbuf), err, errlen)) return 1;
    ds4_rpc_config got = {0};
    unpack_config(&got, cbuf);
    if (peer) *peer = got;

    /* Validate.  We require an exact config match: same model layout, same
     * quant, same split point.  Mismatch is reported back to the head with a
     * human-readable reason so the user sees the cause without a tcpdump. */
    char reject_msg[512] = {0};
    if (!configs_match(&got, cfg)) {
        if (got.version != cfg->version) {
            snprintf(reject_msg, sizeof(reject_msg),
                     "version %u != %u", got.version, cfg->version);
        } else if (got.n_layer_total != cfg->n_layer_total) {
            snprintf(reject_msg, sizeof(reject_msg),
                     "n_layer_total %u != %u", got.n_layer_total, cfg->n_layer_total);
        } else if (got.n_embd != cfg->n_embd ||
                   got.n_hc != cfg->n_hc ||
                   got.n_vocab != cfg->n_vocab) {
            snprintf(reject_msg, sizeof(reject_msg),
                     "model shape mismatch (n_embd=%u/%u n_hc=%u/%u n_vocab=%u/%u)",
                     got.n_embd, cfg->n_embd, got.n_hc, cfg->n_hc,
                     got.n_vocab, cfg->n_vocab);
        } else if (got.routed_quant_bits != cfg->routed_quant_bits) {
            snprintf(reject_msg, sizeof(reject_msg),
                     "routed quant %u != %u (don't mix q2 and q4)",
                     got.routed_quant_bits, cfg->routed_quant_bits);
        } else if (got.tail_layer_start != cfg->tail_layer_start ||
                   got.tail_layer_end != cfg->tail_layer_end) {
            snprintf(reject_msg, sizeof(reject_msg),
                     "split mismatch: head expects tail [%u, %u), worker owns [%u, %u)",
                     got.tail_layer_start, got.tail_layer_end,
                     cfg->tail_layer_start, cfg->tail_layer_end);
        } else if (got.model_file_bytes != cfg->model_file_bytes ||
                   memcmp(got.model_sample, cfg->model_sample, 32) != 0) {
            snprintf(reject_msg, sizeof(reject_msg),
                     "model fingerprint mismatch (file size or header bytes differ)");
        } else {
            snprintf(reject_msg, sizeof(reject_msg), "unknown handshake mismatch");
        }
    }

    const uint32_t status = reject_msg[0] ? 1u : 0u;
    const uint32_t msg_bytes = (uint32_t)strlen(reject_msg);
    const uint32_t reply_bytes = 4u + msg_bytes;
    uint8_t *reply = (uint8_t *)malloc(reply_bytes);
    if (!reply) {
        rpc_set_err(err, errlen, "handshake: out of memory");
        return 1;
    }
    put_u32_le(reply, status);
    if (msg_bytes) memcpy(reply + 4, reject_msg, msg_bytes);
    int rc = frame_write(h->fd, DS4_RPC_OP_HELLO_SERVER, reply, reply_bytes, err, errlen);
    free(reply);
    if (rc) return 1;

    if (status != 0) {
        rpc_set_err(err, errlen, "handshake rejected: %s", reject_msg);
        return 1;
    }
    return 0;
}

/* Decode request/reply. */

int ds4_rpc_decode_request(ds4_rpc_handle *h,
                           uint32_t token, uint32_t pos,
                           const float *residual_hc, uint64_t n_residual_floats,
                           float *out_logits, uint64_t n_logit_floats,
                           char *err, size_t errlen) {
    if (!h || !residual_hc || !out_logits) {
        rpc_set_err(err, errlen, "decode_request: null arg");
        return 1;
    }
    const uint64_t residual_bytes = n_residual_floats * sizeof(float);
    const uint64_t total = 4u + 4u + 8u + residual_bytes;
    if (total > UINT32_MAX) {
        rpc_set_err(err, errlen, "decode_request: residual too large");
        return 1;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)total);
    if (!buf) { rpc_set_err(err, errlen, "decode_request: oom"); return 1; }
    uint8_t *p = buf;
    put_u32_le(p, token);                  p += 4;
    put_u32_le(p, pos);                    p += 4;
    put_u64_le(p, n_residual_floats);      p += 8;
    memcpy(p, residual_hc, (size_t)residual_bytes);
    int rc = frame_write(h->fd, DS4_RPC_OP_DECODE_REQ, buf, (uint32_t)total, err, errlen);
    free(buf);
    if (rc) return 1;

    uint8_t op = 0;
    uint32_t reply_bytes = 0;
    if (frame_read_header(h->fd, &op, &reply_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_DECODE_REPLY) {
        rpc_set_err(err, errlen, "decode_request: expected DECODE_REPLY, got op=%u", op);
        return 1;
    }
    const uint64_t expect_bytes = 4u + 8u + n_logit_floats * sizeof(float);
    if (reply_bytes != expect_bytes) {
        rpc_set_err(err, errlen,
                    "decode_request: reply size mismatch (got %u, expected %llu)",
                    reply_bytes, (unsigned long long)expect_bytes);
        return 1;
    }
    uint8_t hdr[12];
    if (io_read_full(h->fd, hdr, sizeof(hdr), err, errlen)) return 1;
    const uint32_t status = get_u32_le(hdr);
    const uint64_t got_floats = get_u64_le(hdr + 4);
    if (status != 0) {
        rpc_set_err(err, errlen, "decode_request: tail returned error status %u", status);
        /* Drain logits region so the connection stays in sync. */
        uint8_t tmp[4096];
        uint64_t remaining = got_floats * sizeof(float);
        while (remaining > 0) {
            uint64_t chunk = remaining < sizeof(tmp) ? remaining : sizeof(tmp);
            if (io_read_full(h->fd, tmp, (size_t)chunk, NULL, 0)) break;
            remaining -= chunk;
        }
        return 1;
    }
    if (got_floats != n_logit_floats) {
        rpc_set_err(err, errlen, "decode_request: tail sent %llu floats, expected %llu",
                    (unsigned long long)got_floats, (unsigned long long)n_logit_floats);
        return 1;
    }
    return io_read_full(h->fd, out_logits, (size_t)(n_logit_floats * sizeof(float)), err, errlen);
}

int ds4_rpc_decode_recv(ds4_rpc_handle *h,
                        uint32_t *token, uint32_t *pos,
                        float *residual_hc, uint64_t n_residual_floats,
                        char *err, size_t errlen) {
    if (!h || !token || !pos || !residual_hc) {
        rpc_set_err(err, errlen, "decode_recv: null arg");
        return 1;
    }
    uint8_t op = 0;
    uint32_t payload_bytes = 0;
    if (frame_read_header(h->fd, &op, &payload_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_DECODE_REQ) {
        rpc_set_err(err, errlen, "decode_recv: expected DECODE_REQ, got op=%u", op);
        return 1;
    }
    const uint64_t expect = 4u + 4u + 8u + n_residual_floats * sizeof(float);
    if (payload_bytes != expect) {
        rpc_set_err(err, errlen,
                    "decode_recv: payload %u bytes, expected %llu",
                    payload_bytes, (unsigned long long)expect);
        return 1;
    }
    uint8_t hdr[16];
    if (io_read_full(h->fd, hdr, sizeof(hdr), err, errlen)) return 1;
    *token = get_u32_le(hdr);
    *pos   = get_u32_le(hdr + 4);
    const uint64_t got_floats = get_u64_le(hdr + 8);
    if (got_floats != n_residual_floats) {
        rpc_set_err(err, errlen,
                    "decode_recv: residual size mismatch (got %llu, expected %llu)",
                    (unsigned long long)got_floats, (unsigned long long)n_residual_floats);
        return 1;
    }
    return io_read_full(h->fd, residual_hc,
                        (size_t)(n_residual_floats * sizeof(float)), err, errlen);
}

int ds4_rpc_decode_reply(ds4_rpc_handle *h,
                         const float *logits, uint64_t n_logit_floats,
                         char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "decode_reply: null arg"); return 1; }
    const uint64_t logit_bytes = n_logit_floats * sizeof(float);
    const uint64_t total = 4u + 8u + logit_bytes;
    if (total > UINT32_MAX) {
        rpc_set_err(err, errlen, "decode_reply: logits too large");
        return 1;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)total);
    if (!buf) { rpc_set_err(err, errlen, "decode_reply: oom"); return 1; }
    uint8_t *p = buf;
    put_u32_le(p, logits ? 0u : 1u);   p += 4;
    put_u64_le(p, n_logit_floats);     p += 8;
    if (logits) memcpy(p, logits, (size_t)logit_bytes);
    else memset(p, 0, (size_t)logit_bytes);
    int rc = frame_write(h->fd, DS4_RPC_OP_DECODE_REPLY, buf, (uint32_t)total, err, errlen);
    free(buf);
    return rc;
}

int ds4_rpc_reset(ds4_rpc_handle *h, char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "reset: null"); return 1; }
    if (frame_write(h->fd, DS4_RPC_OP_RESET, NULL, 0, err, errlen)) return 1;
    uint8_t op = 0;
    uint32_t bytes = 0;
    if (frame_read_header(h->fd, &op, &bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_RESET_REPLY || bytes != 0) {
        rpc_set_err(err, errlen, "reset: unexpected reply op=%u bytes=%u", op, bytes);
        return 1;
    }
    return 0;
}

/* Non-destructive peek of the next frame's opcode.  Used by the tail worker
 * to dispatch DECODE_REQ vs RESET vs SHUTDOWN.  The op byte sits at offset 4
 * in the frame header; MSG_PEEK returns up to that without consuming bytes
 * from the socket buffer, so the matched *_recv helper can read the full
 * frame normally afterwards. */
int ds4_rpc_recv_op(ds4_rpc_handle *h, ds4_rpc_op *op,
                    char *err, size_t errlen) {
    if (!h || !op) { rpc_set_err(err, errlen, "recv_op: null"); return 1; }
    uint8_t buf[RPC_FRAME_HDR_BYTES];
    size_t got = 0;
    while (got < sizeof(buf)) {
        ssize_t r = recv(h->fd, buf + got, sizeof(buf) - got, MSG_PEEK);
        if (r > 0) { got = (size_t)r; continue; }
        if (r == 0) {
            rpc_set_err(err, errlen, "rpc: peer closed before next frame");
            return 1;
        }
        if (errno == EINTR) continue;
        rpc_set_err(err, errlen, "rpc: peek: %s", strerror(errno));
        return 1;
    }
    *op = (ds4_rpc_op)buf[4];
    return 0;
}

/* Consume a control frame whose payload is empty (RESET, SHUTDOWN). */
static int consume_empty_frame(int fd, uint8_t expected_op,
                               char *err, size_t errlen) {
    uint8_t op = 0;
    uint32_t bytes = 0;
    if (frame_read_header(fd, &op, &bytes, err, errlen)) return 1;
    if (op != expected_op || bytes != 0) {
        rpc_set_err(err, errlen,
                    "rpc: unexpected control frame op=%u bytes=%u (wanted op=%u, 0 bytes)",
                    op, bytes, expected_op);
        return 1;
    }
    return 0;
}

int ds4_rpc_reset_recv(ds4_rpc_handle *h, char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "reset_recv: null"); return 1; }
    return consume_empty_frame(h->fd, DS4_RPC_OP_RESET, err, errlen);
}

int ds4_rpc_shutdown_recv(ds4_rpc_handle *h, char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "shutdown_recv: null"); return 1; }
    return consume_empty_frame(h->fd, DS4_RPC_OP_SHUTDOWN, err, errlen);
}

int ds4_rpc_reset_reply(ds4_rpc_handle *h, char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "reset_reply: null"); return 1; }
    return frame_write(h->fd, DS4_RPC_OP_RESET_REPLY, NULL, 0, err, errlen);
}

int ds4_rpc_shutdown_send(ds4_rpc_handle *h) {
    if (!h) return 1;
    char err[64];
    return frame_write(h->fd, DS4_RPC_OP_SHUTDOWN, NULL, 0, err, sizeof(err));
}
