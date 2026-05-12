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
#include <sys/time.h>
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
    /* 5-minute read timeout.  Prefill of a 30k-token prompt can legitimately
     * take several minutes on TB but never hours, so a stuck peer surfaces
     * as EAGAIN/ETIMEDOUT instead of an indefinite hang.  Tunable via
     * DS4_RPC_RECV_TIMEOUT_SECS for debugging. */
    long recv_timeout = 300;
    const char *env = getenv("DS4_RPC_RECV_TIMEOUT_SECS");
    if (env && env[0]) {
        char *endp = NULL;
        long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) recv_timeout = v;
    }
    struct timeval tv = { .tv_sec = recv_timeout, .tv_usec = 0 };
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
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
#define RPC_CFG_BYTES (12u * 4u + 8u + 32u)

static void pack_config(uint8_t out[RPC_CFG_BYTES], const ds4_rpc_config *c) {
    uint8_t *p = out;
    put_u32_le(p, c->version);                  p += 4;
    put_u32_le(p, c->n_layer_total);            p += 4;
    put_u32_le(p, c->n_embd);                   p += 4;
    put_u32_le(p, c->n_hc);                     p += 4;
    put_u32_le(p, c->n_vocab);                  p += 4;
    put_u32_le(p, c->routed_quant_bits);        p += 4;
    put_u32_le(p, c->tail_layer_start);         p += 4;
    put_u32_le(p, c->tail_layer_end);           p += 4;
    put_u32_le(p, c->ctx_size);                 p += 4;
    put_u32_le(p, c->tail_has_mtp);             p += 4;
    put_u32_le(p, c->tail_mtp_draft_tokens);    p += 4;
    put_u32_le(p, c->reserved0);                p += 4;
    put_u64_le(p, c->model_file_bytes);         p += 8;
    memcpy(p, c->model_sample, 32);             p += 32;
    (void)p;
}

static void unpack_config(ds4_rpc_config *c, const uint8_t in[RPC_CFG_BYTES]) {
    const uint8_t *p = in;
    c->version                = get_u32_le(p); p += 4;
    c->n_layer_total          = get_u32_le(p); p += 4;
    c->n_embd                 = get_u32_le(p); p += 4;
    c->n_hc                   = get_u32_le(p); p += 4;
    c->n_vocab                = get_u32_le(p); p += 4;
    c->routed_quant_bits      = get_u32_le(p); p += 4;
    c->tail_layer_start       = get_u32_le(p); p += 4;
    c->tail_layer_end         = get_u32_le(p); p += 4;
    c->ctx_size               = get_u32_le(p); p += 4;
    c->tail_has_mtp           = get_u32_le(p); p += 4;
    c->tail_mtp_draft_tokens  = get_u32_le(p); p += 4;
    c->reserved0              = get_u32_le(p); p += 4;
    c->model_file_bytes       = get_u64_le(p); p += 8;
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
    if (a->ctx_size          != b->ctx_size)         return false;
    if (a->model_file_bytes  != b->model_file_bytes) return false;
    if (memcmp(a->model_sample, b->model_sample, 32) != 0) return false;
    /* tail_has_mtp and tail_mtp_draft_tokens are tail-side capabilities the
     * head learns from peer-returned config; not matched here. */
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

int ds4_rpc_handshake_client_peer(ds4_rpc_handle *h,
                                  const ds4_rpc_config *cfg,
                                  ds4_rpc_config *out_peer,
                                  char *err, size_t errlen);

int ds4_rpc_handshake_client(ds4_rpc_handle *h, const ds4_rpc_config *cfg,
                             char *err, size_t errlen) {
    return ds4_rpc_handshake_client_peer(h, cfg, NULL, err, errlen);
}

int ds4_rpc_handshake_client_peer(ds4_rpc_handle *h,
                                  const ds4_rpc_config *cfg,
                                  ds4_rpc_config *out_peer,
                                  char *err, size_t errlen) {
    if (!h || !cfg) { rpc_set_err(err, errlen, "handshake: null arg"); return 1; }
    if (write_magic(h->fd, err, errlen)) return 1;

    uint8_t cbuf[RPC_CFG_BYTES];
    pack_config(cbuf, cfg);
    if (frame_write(h->fd, DS4_RPC_OP_HELLO_CLIENT, cbuf, sizeof(cbuf), err, errlen)) return 1;

    uint8_t op = 0;
    uint32_t payload_bytes = 0;
    if (frame_read_header(h->fd,&op, &payload_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_HELLO_SERVER) {
        rpc_set_err(err, errlen, "handshake: server replied op=%u, expected HELLO_SERVER", op);
        return 1;
    }
    if (payload_bytes < 4u + RPC_CFG_BYTES) {
        rpc_set_err(err, errlen,
                    "handshake: server reply too short (%u bytes, need >= %u)",
                    payload_bytes, 4u + (uint32_t)RPC_CFG_BYTES);
        return 1;
    }
    uint8_t status_buf[4];
    if (io_read_full(h->fd,status_buf, sizeof(status_buf), err, errlen)) return 1;
    const uint32_t status = get_u32_le(status_buf);

    uint8_t peer_buf[RPC_CFG_BYTES];
    if (io_read_full(h->fd,peer_buf, sizeof(peer_buf), err, errlen)) return 1;
    if (out_peer) unpack_config(out_peer, peer_buf);

    const uint32_t msg_bytes = payload_bytes - 4u - (uint32_t)RPC_CFG_BYTES;

    if (status != 0) {
        char *msg = NULL;
        if (msg_bytes > 0 && msg_bytes < 4096) {
            msg = (char *)malloc((size_t)msg_bytes + 1u);
            if (msg) {
                if (io_read_full(h->fd,msg, msg_bytes, err, errlen) == 0) {
                    msg[msg_bytes] = '\0';
                    rpc_set_err(err, errlen, "rpc: server rejected handshake: %s", msg);
                }
                free(msg);
                return 1;
            }
        }
        uint8_t tmp[256];
        uint32_t remaining = msg_bytes;
        while (remaining > 0) {
            uint32_t chunk = remaining < sizeof(tmp) ? remaining : (uint32_t)sizeof(tmp);
            if (io_read_full(h->fd,tmp, chunk, NULL, 0)) break;
            remaining -= chunk;
        }
        rpc_set_err(err, errlen, "rpc: server rejected handshake (status=%u)", status);
        return 1;
    }
    /* status == 0: no error_msg present, drain any leftover defensively. */
    if (msg_bytes > 0) {
        uint8_t tmp[256];
        uint32_t remaining = msg_bytes;
        while (remaining > 0) {
            uint32_t chunk = remaining < sizeof(tmp) ? remaining : (uint32_t)sizeof(tmp);
            if (io_read_full(h->fd,tmp, chunk, NULL, 0)) break;
            remaining -= chunk;
        }
    }
    return 0;
}

int ds4_rpc_handshake_server(ds4_rpc_handle *h, const ds4_rpc_config *cfg,
                             ds4_rpc_config *peer, char *err, size_t errlen) {
    if (!h || !cfg) { rpc_set_err(err, errlen, "handshake: null arg"); return 1; }
    if (read_magic(h->fd, err, errlen)) return 1;

    uint8_t op = 0;
    uint32_t payload_bytes = 0;
    if (frame_read_header(h->fd,&op, &payload_bytes, err, errlen)) return 1;
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
    if (io_read_full(h->fd,cbuf, sizeof(cbuf), err, errlen)) return 1;
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
        } else if (got.ctx_size != cfg->ctx_size) {
            snprintf(reject_msg, sizeof(reject_msg),
                     "ctx mismatch: head ctx=%u, worker ctx=%u "
                     "(start the worker with --ctx %u to match)",
                     got.ctx_size, cfg->ctx_size, got.ctx_size);
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
    const uint32_t reply_bytes = 4u + (uint32_t)RPC_CFG_BYTES + msg_bytes;
    uint8_t *reply = (uint8_t *)malloc(reply_bytes);
    if (!reply) {
        rpc_set_err(err, errlen, "handshake: out of memory");
        return 1;
    }
    put_u32_le(reply, status);
    pack_config(reply + 4, cfg);
    if (msg_bytes) memcpy(reply + 4 + RPC_CFG_BYTES, reject_msg, msg_bytes);
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

/* Decode reply header layout (16 bytes):
 *   u32 status
 *   u32 n_drafts            (always present; 0 means no MTP attached)
 *   u64 n_logit_floats
 * Then n_drafts * sizeof(u32) of draft tokens, then n_logit_floats * sizeof(float).
 * Drafts come first because their length is a small fixed integer; logits come
 * after so the receiver can stream-read them straight into the caller's buffer.
 */
int ds4_rpc_decode_send(ds4_rpc_handle *h,
                        uint32_t token, uint32_t pos,
                        bool want_drafts,
                        const float *residual_hc, uint64_t n_residual_floats,
                        char *err, size_t errlen) {
    if (!h || !residual_hc) {
        rpc_set_err(err, errlen, "decode_send: null arg");
        return 1;
    }
    const uint64_t residual_bytes = n_residual_floats * sizeof(float);
    const uint64_t total = 4u + 4u + 4u + 4u + 8u + residual_bytes;
    if (total > UINT32_MAX) {
        rpc_set_err(err, errlen, "decode_send: residual too large");
        return 1;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)total);
    if (!buf) { rpc_set_err(err, errlen, "decode_send: oom"); return 1; }
    uint8_t *p = buf;
    put_u32_le(p, token);                  p += 4;
    put_u32_le(p, pos);                    p += 4;
    put_u32_le(p, want_drafts ? 1u : 0u);  p += 4;
    put_u32_le(p, 0u);                     p += 4; /* reserved */
    put_u64_le(p, n_residual_floats);      p += 8;
    memcpy(p, residual_hc, (size_t)residual_bytes);
    int rc = frame_write(h->fd, DS4_RPC_OP_DECODE_REQ, buf, (uint32_t)total, err, errlen);
    free(buf);
    return rc;
}

int ds4_rpc_decode_recv_reply(ds4_rpc_handle *h,
                              float *out_logits, uint64_t n_logit_floats,
                              uint32_t *out_drafts, uint32_t max_drafts,
                              uint32_t *out_n_drafts,
                              char *err, size_t errlen) {
    if (!h || !out_logits) {
        rpc_set_err(err, errlen, "decode_recv_reply: null arg");
        return 1;
    }
    if (out_n_drafts) *out_n_drafts = 0;

    uint8_t op = 0;
    uint32_t reply_bytes = 0;
    if (frame_read_header(h->fd,&op, &reply_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_DECODE_REPLY) {
        rpc_set_err(err, errlen, "decode_recv_reply: expected DECODE_REPLY, got op=%u", op);
        return 1;
    }
    if (reply_bytes < 16u) {
        rpc_set_err(err, errlen, "decode_recv_reply: reply header truncated (%u bytes)", reply_bytes);
        return 1;
    }
    uint8_t hdr[16];
    if (io_read_full(h->fd,hdr, sizeof(hdr), err, errlen)) return 1;
    const uint32_t status      = get_u32_le(hdr);
    const uint32_t n_drafts    = get_u32_le(hdr + 4);
    const uint64_t got_floats  = get_u64_le(hdr + 8);
    const uint64_t expect_payload =
        (uint64_t)n_drafts * sizeof(uint32_t) + got_floats * sizeof(float);
    if (reply_bytes - 16u != expect_payload) {
        rpc_set_err(err, errlen,
                    "decode_recv_reply: reply payload size %u, expected %llu "
                    "(n_drafts=%u, n_logits=%llu)",
                    reply_bytes - 16u, (unsigned long long)expect_payload,
                    n_drafts, (unsigned long long)got_floats);
        return 1;
    }
    if (status != 0) {
        uint8_t tmp[4096];
        uint64_t remaining = expect_payload;
        while (remaining > 0) {
            uint64_t chunk = remaining < sizeof(tmp) ? remaining : sizeof(tmp);
            if (io_read_full(h->fd,tmp, (size_t)chunk, NULL, 0)) break;
            remaining -= chunk;
        }
        rpc_set_err(err, errlen, "decode_recv_reply: tail returned error status %u", status);
        return 1;
    }
    if (got_floats != n_logit_floats) {
        rpc_set_err(err, errlen, "decode_recv_reply: tail sent %llu floats, expected %llu",
                    (unsigned long long)got_floats, (unsigned long long)n_logit_floats);
        return 1;
    }
    if (n_drafts > 0) {
        const uint64_t draft_bytes = (uint64_t)n_drafts * sizeof(uint32_t);
        if (out_drafts && max_drafts > 0) {
            const uint32_t accept = n_drafts < max_drafts ? n_drafts : max_drafts;
            uint8_t drafts_buf[DS4_RPC_MAX_DRAFTS * sizeof(uint32_t)];
            if (draft_bytes > sizeof(drafts_buf)) {
                rpc_set_err(err, errlen, "decode_recv_reply: too many drafts (%u)", n_drafts);
                return 1;
            }
            if (io_read_full(h->fd,drafts_buf, (size_t)draft_bytes, err, errlen)) return 1;
            for (uint32_t i = 0; i < accept; i++) {
                out_drafts[i] = get_u32_le(drafts_buf + i * 4);
            }
            if (out_n_drafts) *out_n_drafts = accept;
        } else {
            uint8_t tmp[DS4_RPC_MAX_DRAFTS * sizeof(uint32_t)];
            if (draft_bytes > sizeof(tmp)) {
                rpc_set_err(err, errlen, "decode_recv_reply: too many drafts (%u) to drain", n_drafts);
                return 1;
            }
            if (io_read_full(h->fd,tmp, (size_t)draft_bytes, err, errlen)) return 1;
        }
    }
    return io_read_full(h->fd,out_logits, (size_t)(n_logit_floats * sizeof(float)), err, errlen);
}

int ds4_rpc_decode_request(ds4_rpc_handle *h,
                           uint32_t token, uint32_t pos,
                           bool want_drafts,
                           const float *residual_hc, uint64_t n_residual_floats,
                           float *out_logits, uint64_t n_logit_floats,
                           uint32_t *out_drafts, uint32_t max_drafts,
                           uint32_t *out_n_drafts,
                           char *err, size_t errlen) {
    if (ds4_rpc_decode_send(h, token, pos, want_drafts,
                            residual_hc, n_residual_floats, err, errlen) != 0) {
        return 1;
    }
    return ds4_rpc_decode_recv_reply(h, out_logits, n_logit_floats,
                                     out_drafts, max_drafts, out_n_drafts,
                                     err, errlen);
}

int ds4_rpc_decode_recv(ds4_rpc_handle *h,
                        uint32_t *token, uint32_t *pos,
                        bool *want_drafts,
                        float *residual_hc, uint64_t n_residual_floats,
                        char *err, size_t errlen) {
    if (!h || !token || !pos || !want_drafts || !residual_hc) {
        rpc_set_err(err, errlen, "decode_recv: null arg");
        return 1;
    }
    uint8_t op = 0;
    uint32_t payload_bytes = 0;
    if (frame_read_header(h->fd,&op, &payload_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_DECODE_REQ) {
        rpc_set_err(err, errlen, "decode_recv: expected DECODE_REQ, got op=%u", op);
        return 1;
    }
    const uint64_t expect = 4u + 4u + 4u + 4u + 8u + n_residual_floats * sizeof(float);
    if (payload_bytes != expect) {
        rpc_set_err(err, errlen,
                    "decode_recv: payload %u bytes, expected %llu",
                    payload_bytes, (unsigned long long)expect);
        return 1;
    }
    uint8_t hdr[24];
    if (io_read_full(h->fd,hdr, sizeof(hdr), err, errlen)) return 1;
    *token       = get_u32_le(hdr);
    *pos         = get_u32_le(hdr + 4);
    *want_drafts = get_u32_le(hdr + 8) != 0;
    /* hdr+12..16 reserved */
    const uint64_t got_floats = get_u64_le(hdr + 16);
    if (got_floats != n_residual_floats) {
        rpc_set_err(err, errlen,
                    "decode_recv: residual size mismatch (got %llu, expected %llu)",
                    (unsigned long long)got_floats, (unsigned long long)n_residual_floats);
        return 1;
    }
    return io_read_full(h->fd,residual_hc,
                        (size_t)(n_residual_floats * sizeof(float)), err, errlen);
}

int ds4_rpc_decode_reply(ds4_rpc_handle *h,
                         const float *logits, uint64_t n_logit_floats,
                         const uint32_t *drafts, uint32_t n_drafts,
                         char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "decode_reply: null arg"); return 1; }
    if (n_drafts > DS4_RPC_MAX_DRAFTS) {
        rpc_set_err(err, errlen, "decode_reply: %u drafts exceeds max %u",
                    n_drafts, DS4_RPC_MAX_DRAFTS);
        return 1;
    }
    if (n_drafts > 0 && !drafts) {
        rpc_set_err(err, errlen, "decode_reply: n_drafts>0 but drafts is NULL");
        return 1;
    }
    const uint64_t logit_bytes = n_logit_floats * sizeof(float);
    const uint64_t draft_bytes = (uint64_t)n_drafts * sizeof(uint32_t);
    const uint64_t total = 4u + 4u + 8u + draft_bytes + logit_bytes;
    if (total > UINT32_MAX) {
        rpc_set_err(err, errlen, "decode_reply: payload too large");
        return 1;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)total);
    if (!buf) { rpc_set_err(err, errlen, "decode_reply: oom"); return 1; }
    uint8_t *p = buf;
    put_u32_le(p, logits ? 0u : 1u);                     p += 4; /* status */
    put_u32_le(p, n_drafts);                             p += 4;
    put_u64_le(p, n_logit_floats);                       p += 8;
    for (uint32_t i = 0; i < n_drafts; i++) {
        put_u32_le(p, drafts[i]);                        p += 4;
    }
    if (logits) memcpy(p, logits, (size_t)logit_bytes);
    else        memset(p, 0, (size_t)logit_bytes);
    int rc = frame_write(h->fd, DS4_RPC_OP_DECODE_REPLY, buf, (uint32_t)total, err, errlen);
    free(buf);
    return rc;
}

int ds4_rpc_mtp_trim(ds4_rpc_handle *h, uint32_t accepted_drafts,
                     char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "mtp_trim: null"); return 1; }
    uint8_t payload[4];
    put_u32_le(payload, accepted_drafts);
    if (frame_write(h->fd, DS4_RPC_OP_MTP_TRIM, payload, sizeof(payload),
                    err, errlen)) return 1;
    uint8_t op = 0;
    uint32_t bytes = 0;
    if (frame_read_header(h->fd,&op, &bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_MTP_TRIM_REPLY || bytes != 0) {
        rpc_set_err(err, errlen, "mtp_trim: unexpected reply op=%u bytes=%u", op, bytes);
        return 1;
    }
    return 0;
}

int ds4_rpc_mtp_trim_recv(ds4_rpc_handle *h, uint32_t *accepted_drafts,
                          char *err, size_t errlen) {
    if (!h || !accepted_drafts) {
        rpc_set_err(err, errlen, "mtp_trim_recv: null");
        return 1;
    }
    uint8_t op = 0;
    uint32_t bytes = 0;
    if (frame_read_header(h->fd,&op, &bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_MTP_TRIM || bytes != 4) {
        rpc_set_err(err, errlen, "mtp_trim_recv: unexpected op=%u bytes=%u", op, bytes);
        return 1;
    }
    uint8_t buf[4];
    if (io_read_full(h->fd,buf, sizeof(buf), err, errlen)) return 1;
    *accepted_drafts = get_u32_le(buf);
    return 0;
}

int ds4_rpc_mtp_trim_reply(ds4_rpc_handle *h, char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "mtp_trim_reply: null"); return 1; }
    return frame_write(h->fd, DS4_RPC_OP_MTP_TRIM_REPLY, NULL, 0, err, errlen);
}

/* Verify-batch wire format.  Request layout:
 *   u32 n_tokens
 *   u32 pos_start
 *   u32 n_expected   (= n_tokens - 1; the drafts head wants verified)
 *   u32 reserved
 *   u64 n_residual_floats
 *   float32[n_residual_floats]   batch_cur_hc rows
 *   u32[n_expected]              expected_next tokens
 *
 * Reply layout:
 *   u32 status
 *   u32 n_accepted   (0 = miss + KV reverted, n_tokens = full accept)
 *   u32 reserved
 *   u32 reserved
 *   u64 n_logit_floats
 *   float32[n_logit_floats]      logits (only meaningful when n_accepted > 0)
 */
int ds4_rpc_verify_batch_request(ds4_rpc_handle *h,
                                 uint32_t n_tokens, uint32_t pos_start,
                                 const float *batch_residual,
                                 uint64_t n_residual_floats,
                                 const uint32_t *expected_next,
                                 uint32_t n_expected,
                                 uint32_t *out_n_accepted,
                                 float *out_logits, uint64_t n_logit_floats,
                                 char *err, size_t errlen) {
    if (!h || !batch_residual || !out_n_accepted || !out_logits) {
        rpc_set_err(err, errlen, "verify_batch_request: null arg");
        return 1;
    }
    if (n_expected > 0 && !expected_next) {
        rpc_set_err(err, errlen, "verify_batch_request: n_expected>0 but null buffer");
        return 1;
    }
    *out_n_accepted = 0;

    const uint64_t residual_bytes = n_residual_floats * sizeof(float);
    const uint64_t expected_bytes = (uint64_t)n_expected * sizeof(uint32_t);
    const uint64_t total = 4u + 4u + 4u + 4u + 8u + residual_bytes + expected_bytes;
    if (total > UINT32_MAX) {
        rpc_set_err(err, errlen, "verify_batch_request: payload too large");
        return 1;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)total);
    if (!buf) { rpc_set_err(err, errlen, "verify_batch_request: oom"); return 1; }
    uint8_t *p = buf;
    put_u32_le(p, n_tokens);              p += 4;
    put_u32_le(p, pos_start);             p += 4;
    put_u32_le(p, n_expected);            p += 4;
    put_u32_le(p, 0u);                    p += 4;
    put_u64_le(p, n_residual_floats);     p += 8;
    memcpy(p, batch_residual, (size_t)residual_bytes); p += residual_bytes;
    for (uint32_t i = 0; i < n_expected; i++) {
        put_u32_le(p, expected_next[i]);  p += 4;
    }
    int rc = frame_write(h->fd, DS4_RPC_OP_VERIFY_BATCH, buf, (uint32_t)total,
                         err, errlen);
    free(buf);
    if (rc) return 1;

    uint8_t op = 0;
    uint32_t reply_bytes = 0;
    if (frame_read_header(h->fd,&op, &reply_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_VERIFY_BATCH_REPLY) {
        rpc_set_err(err, errlen, "verify_batch_request: expected reply op=%u, got %u",
                    DS4_RPC_OP_VERIFY_BATCH_REPLY, op);
        return 1;
    }
    const uint64_t reply_min = 4u + 4u + 4u + 4u + 8u;
    if (reply_bytes < reply_min) {
        rpc_set_err(err, errlen, "verify_batch_request: reply truncated (%u bytes)", reply_bytes);
        return 1;
    }
    uint8_t hdr[24];
    if (io_read_full(h->fd,hdr, sizeof(hdr), err, errlen)) return 1;
    const uint32_t status     = get_u32_le(hdr);
    const uint32_t n_accepted = get_u32_le(hdr + 4);
    const uint64_t got_floats = get_u64_le(hdr + 16);
    const uint64_t remaining  = reply_bytes - 24u;

    if (status != 0) {
        uint8_t tmp[4096];
        uint64_t left = remaining;
        while (left > 0) {
            uint64_t chunk = left < sizeof(tmp) ? left : sizeof(tmp);
            if (io_read_full(h->fd,tmp, (size_t)chunk, NULL, 0)) break;
            left -= chunk;
        }
        rpc_set_err(err, errlen, "verify_batch_request: tail status %u", status);
        return 1;
    }

    *out_n_accepted = n_accepted;
    if (remaining != got_floats * sizeof(float)) {
        rpc_set_err(err, errlen,
                    "verify_batch_request: reply payload %llu, expected %llu",
                    (unsigned long long)remaining,
                    (unsigned long long)(got_floats * sizeof(float)));
        return 1;
    }
    if (got_floats == 0) return 0;
    if (got_floats != n_logit_floats) {
        rpc_set_err(err, errlen,
                    "verify_batch_request: tail sent %llu floats, expected %llu",
                    (unsigned long long)got_floats,
                    (unsigned long long)n_logit_floats);
        return 1;
    }
    return io_read_full(h->fd,out_logits, (size_t)remaining, err, errlen);
}

int ds4_rpc_verify_batch_recv(ds4_rpc_handle *h,
                              uint32_t *n_tokens, uint32_t *pos_start,
                              float *batch_residual, uint64_t max_residual_floats,
                              uint64_t *out_n_residual_floats,
                              uint32_t *expected_next, uint32_t max_expected,
                              uint32_t *out_n_expected,
                              char *err, size_t errlen) {
    if (!h || !n_tokens || !pos_start || !batch_residual ||
        !out_n_residual_floats || !expected_next || !out_n_expected) {
        rpc_set_err(err, errlen, "verify_batch_recv: null arg");
        return 1;
    }
    uint8_t op = 0;
    uint32_t payload_bytes = 0;
    if (frame_read_header(h->fd,&op, &payload_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_VERIFY_BATCH) {
        rpc_set_err(err, errlen, "verify_batch_recv: expected VERIFY_BATCH, got op=%u", op);
        return 1;
    }
    if (payload_bytes < 24u) {
        rpc_set_err(err, errlen, "verify_batch_recv: payload truncated (%u bytes)", payload_bytes);
        return 1;
    }
    uint8_t hdr[24];
    if (io_read_full(h->fd,hdr, sizeof(hdr), err, errlen)) return 1;
    *n_tokens   = get_u32_le(hdr);
    *pos_start  = get_u32_le(hdr + 4);
    const uint32_t n_expected_in = get_u32_le(hdr + 8);
    const uint64_t n_floats      = get_u64_le(hdr + 16);
    const uint64_t expected_bytes = (uint64_t)n_expected_in * sizeof(uint32_t);
    const uint64_t want_bytes     = n_floats * sizeof(float);
    if (payload_bytes - 24u != want_bytes + expected_bytes) {
        rpc_set_err(err, errlen,
                    "verify_batch_recv: payload size mismatch (got %u, expected %llu + %llu)",
                    payload_bytes - 24u,
                    (unsigned long long)want_bytes,
                    (unsigned long long)expected_bytes);
        return 1;
    }
    if (n_floats > max_residual_floats) {
        rpc_set_err(err, errlen,
                    "verify_batch_recv: residual %llu floats exceeds buffer %llu",
                    (unsigned long long)n_floats,
                    (unsigned long long)max_residual_floats);
        return 1;
    }
    if (n_expected_in > max_expected) {
        rpc_set_err(err, errlen,
                    "verify_batch_recv: %u expected exceeds buffer %u",
                    n_expected_in, max_expected);
        return 1;
    }
    *out_n_residual_floats = n_floats;
    *out_n_expected = n_expected_in;
    if (io_read_full(h->fd,batch_residual, (size_t)want_bytes, err, errlen)) return 1;
    if (n_expected_in > 0) {
        uint8_t tmp[16 * 4];
        if (expected_bytes > sizeof(tmp)) {
            rpc_set_err(err, errlen, "verify_batch_recv: %u expected too many",
                        n_expected_in);
            return 1;
        }
        if (io_read_full(h->fd,tmp, (size_t)expected_bytes, err, errlen)) return 1;
        for (uint32_t i = 0; i < n_expected_in; i++) {
            expected_next[i] = get_u32_le(tmp + i * 4);
        }
    }
    return 0;
}

int ds4_rpc_verify_batch_reply(ds4_rpc_handle *h,
                               uint32_t n_accepted,
                               const float *logits, uint64_t n_logit_floats,
                               char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "verify_batch_reply: null"); return 1; }
    const bool has_logits = (logits != NULL && n_logit_floats > 0);
    const uint64_t logit_bytes = has_logits ? n_logit_floats * sizeof(float) : 0u;
    const uint64_t total = 4u + 4u + 4u + 4u + 8u + logit_bytes;
    if (total > UINT32_MAX) {
        rpc_set_err(err, errlen, "verify_batch_reply: payload too large");
        return 1;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)total);
    if (!buf) { rpc_set_err(err, errlen, "verify_batch_reply: oom"); return 1; }
    uint8_t *p = buf;
    put_u32_le(p, 0u);                              p += 4; /* status */
    put_u32_le(p, n_accepted);                      p += 4;
    put_u32_le(p, 0u);                              p += 4;
    put_u32_le(p, 0u);                              p += 4;
    put_u64_le(p, has_logits ? n_logit_floats : 0u); p += 8;
    if (has_logits) memcpy(p, logits, (size_t)logit_bytes);
    int rc = frame_write(h->fd, DS4_RPC_OP_VERIFY_BATCH_REPLY, buf, (uint32_t)total,
                         err, errlen);
    free(buf);
    return rc;
}

/* Prefill request: one chunk of the prompt's batch_cur_hc, sized
 * (n_tokens * DS4_N_HC * DS4_N_EMBD) floats.  Reply is empty if
 * !want_logits, otherwise carries one DS4_N_VOCAB-sized logits vector.
 * Frame layout for the request:
 *   u32 n_tokens
 *   u32 pos_start
 *   u32 want_logits  (0 or 1)
 *   u32 reserved
 *   u64 n_residual_floats
 *   float32[n_residual_floats] batch residual data
 * Frame layout for the reply:
 *   u32 status (0 = ok, !=0 = error)
 *   u32 has_logits (0 or 1; on error always 0)
 *   u64 n_logit_floats (matches has_logits)
 *   float32[n_logit_floats] logits */
int ds4_rpc_prefill_request(ds4_rpc_handle *h,
                            uint32_t n_tokens, uint32_t pos_start,
                            bool want_logits,
                            const float *batch_residual_hc,
                            uint64_t n_residual_floats,
                            float *out_logits, uint64_t n_logit_floats,
                            char *err, size_t errlen) {
    if (!h || !batch_residual_hc) {
        rpc_set_err(err, errlen, "prefill_request: null arg");
        return 1;
    }
    if (want_logits && (!out_logits || n_logit_floats == 0)) {
        rpc_set_err(err, errlen, "prefill_request: want_logits set but no output buffer");
        return 1;
    }
    const uint64_t residual_bytes = n_residual_floats * sizeof(float);
    const uint64_t payload_bytes = 4u + 4u + 4u + 4u + 8u + residual_bytes;
    if (payload_bytes > UINT32_MAX) {
        rpc_set_err(err, errlen, "prefill_request: chunk too large to frame");
        return 1;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)payload_bytes);
    if (!buf) { rpc_set_err(err, errlen, "prefill_request: oom"); return 1; }
    uint8_t *p = buf;
    put_u32_le(p, n_tokens);                   p += 4;
    put_u32_le(p, pos_start);                  p += 4;
    put_u32_le(p, want_logits ? 1u : 0u);      p += 4;
    put_u32_le(p, 0u);                         p += 4;
    put_u64_le(p, n_residual_floats);          p += 8;
    memcpy(p, batch_residual_hc, (size_t)residual_bytes);
    int rc = frame_write(h->fd, DS4_RPC_OP_PREFILL_REQ, buf, (uint32_t)payload_bytes,
                         err, errlen);
    free(buf);
    if (rc) return 1;

    uint8_t op = 0;
    uint32_t reply_bytes = 0;
    if (frame_read_header(h->fd,&op, &reply_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_PREFILL_REPLY) {
        rpc_set_err(err, errlen, "prefill_request: expected PREFILL_REPLY, got op=%u", op);
        return 1;
    }
    if (reply_bytes < 4u + 4u + 8u) {
        rpc_set_err(err, errlen, "prefill_request: reply header truncated (%u bytes)", reply_bytes);
        return 1;
    }
    uint8_t hdr[16];
    if (io_read_full(h->fd,hdr, sizeof(hdr), err, errlen)) return 1;
    const uint32_t status     = get_u32_le(hdr);
    const uint32_t has_logits = get_u32_le(hdr + 4);
    const uint64_t got_floats = get_u64_le(hdr + 8);
    const uint64_t remaining  = reply_bytes - 16u;

    if (status != 0) {
        /* Drain any payload bytes so the connection stays in sync. */
        uint8_t tmp[4096];
        uint64_t left = remaining;
        while (left > 0) {
            uint64_t chunk = left < sizeof(tmp) ? left : sizeof(tmp);
            if (io_read_full(h->fd,tmp, (size_t)chunk, NULL, 0)) break;
            left -= chunk;
        }
        rpc_set_err(err, errlen, "prefill_request: tail returned error status %u", status);
        return 1;
    }
    if (!want_logits) {
        if (has_logits || got_floats != 0 || remaining != 0) {
            rpc_set_err(err, errlen, "prefill_request: tail returned logits we did not request");
            return 1;
        }
        return 0;
    }
    if (!has_logits || got_floats != n_logit_floats) {
        rpc_set_err(err, errlen,
                    "prefill_request: tail returned %llu floats (has=%u), expected %llu (has=1)",
                    (unsigned long long)got_floats, has_logits,
                    (unsigned long long)n_logit_floats);
        return 1;
    }
    if (remaining != n_logit_floats * sizeof(float)) {
        rpc_set_err(err, errlen,
                    "prefill_request: reply payload size mismatch (%llu vs %llu)",
                    (unsigned long long)remaining,
                    (unsigned long long)(n_logit_floats * sizeof(float)));
        return 1;
    }
    return io_read_full(h->fd,out_logits, (size_t)remaining, err, errlen);
}

int ds4_rpc_prefill_recv(ds4_rpc_handle *h,
                         uint32_t *n_tokens, uint32_t *pos_start,
                         bool *want_logits,
                         float *batch_residual_hc, uint64_t max_residual_floats,
                         uint64_t *out_n_residual_floats,
                         char *err, size_t errlen) {
    if (!h || !n_tokens || !pos_start || !want_logits || !batch_residual_hc ||
        !out_n_residual_floats) {
        rpc_set_err(err, errlen, "prefill_recv: null arg");
        return 1;
    }
    uint8_t op = 0;
    uint32_t payload_bytes = 0;
    if (frame_read_header(h->fd,&op, &payload_bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_PREFILL_REQ) {
        rpc_set_err(err, errlen, "prefill_recv: expected PREFILL_REQ, got op=%u", op);
        return 1;
    }
    if (payload_bytes < 4u + 4u + 4u + 4u + 8u) {
        rpc_set_err(err, errlen, "prefill_recv: payload truncated (%u bytes)", payload_bytes);
        return 1;
    }
    uint8_t hdr[24];
    if (io_read_full(h->fd,hdr, sizeof(hdr), err, errlen)) return 1;
    *n_tokens                  = get_u32_le(hdr);
    *pos_start                 = get_u32_le(hdr + 4);
    *want_logits               = get_u32_le(hdr + 8) != 0;
    /* hdr+12..16 reserved */
    const uint64_t n_floats    = get_u64_le(hdr + 16);
    const uint64_t want_bytes  = n_floats * sizeof(float);
    if (payload_bytes - 24u != want_bytes) {
        rpc_set_err(err, errlen,
                    "prefill_recv: residual size mismatch (header says %llu floats = %llu bytes, "
                    "frame has %llu bytes of payload after header)",
                    (unsigned long long)n_floats,
                    (unsigned long long)want_bytes,
                    (unsigned long long)(payload_bytes - 24u));
        return 1;
    }
    if (n_floats > max_residual_floats) {
        rpc_set_err(err, errlen,
                    "prefill_recv: chunk needs %llu residual floats but caller buffer holds only %llu",
                    (unsigned long long)n_floats,
                    (unsigned long long)max_residual_floats);
        return 1;
    }
    *out_n_residual_floats = n_floats;
    return io_read_full(h->fd,batch_residual_hc, (size_t)want_bytes, err, errlen);
}

int ds4_rpc_prefill_reply(ds4_rpc_handle *h,
                          bool has_logits,
                          const float *logits, uint64_t n_logit_floats,
                          char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "prefill_reply: null arg"); return 1; }
    if (has_logits && (!logits || n_logit_floats == 0)) {
        rpc_set_err(err, errlen, "prefill_reply: has_logits set but no logits provided");
        return 1;
    }
    const uint64_t logit_bytes = has_logits ? n_logit_floats * sizeof(float) : 0u;
    const uint64_t payload_bytes = 4u + 4u + 8u + logit_bytes;
    if (payload_bytes > UINT32_MAX) {
        rpc_set_err(err, errlen, "prefill_reply: logits too large to frame");
        return 1;
    }
    uint8_t *buf = (uint8_t *)malloc((size_t)payload_bytes);
    if (!buf) { rpc_set_err(err, errlen, "prefill_reply: oom"); return 1; }
    uint8_t *p = buf;
    put_u32_le(p, 0u);                                   p += 4; /* status = ok */
    put_u32_le(p, has_logits ? 1u : 0u);                 p += 4;
    put_u64_le(p, has_logits ? n_logit_floats : 0u);     p += 8;
    if (has_logits) memcpy(p, logits, (size_t)logit_bytes);
    int rc = frame_write(h->fd, DS4_RPC_OP_PREFILL_REPLY, buf, (uint32_t)payload_bytes,
                         err, errlen);
    free(buf);
    return rc;
}

int ds4_rpc_reset(ds4_rpc_handle *h, char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "reset: null"); return 1; }
    if (frame_write(h->fd, DS4_RPC_OP_RESET, NULL, 0, err, errlen)) return 1;
    uint8_t op = 0;
    uint32_t bytes = 0;
    if (frame_read_header(h->fd,&op, &bytes, err, errlen)) return 1;
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

int ds4_rpc_rewind(ds4_rpc_handle *h, uint32_t target_pos,
                   char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "rewind: null"); return 1; }
    uint8_t payload[4];
    put_u32_le(payload, target_pos);
    if (frame_write(h->fd, DS4_RPC_OP_REWIND, payload, sizeof(payload),
                    err, errlen)) return 1;
    uint8_t op = 0;
    uint32_t bytes = 0;
    if (frame_read_header(h->fd,&op, &bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_REWIND_REPLY || bytes != 0) {
        rpc_set_err(err, errlen, "rewind: unexpected reply op=%u bytes=%u", op, bytes);
        return 1;
    }
    return 0;
}

int ds4_rpc_rewind_recv(ds4_rpc_handle *h, uint32_t *target_pos,
                        char *err, size_t errlen) {
    if (!h || !target_pos) { rpc_set_err(err, errlen, "rewind_recv: null"); return 1; }
    uint8_t op = 0;
    uint32_t bytes = 0;
    if (frame_read_header(h->fd,&op, &bytes, err, errlen)) return 1;
    if (op != DS4_RPC_OP_REWIND || bytes != 4) {
        rpc_set_err(err, errlen, "rewind_recv: unexpected op=%u bytes=%u", op, bytes);
        return 1;
    }
    uint8_t buf[4];
    if (io_read_full(h->fd,buf, sizeof(buf), err, errlen)) return 1;
    *target_pos = get_u32_le(buf);
    return 0;
}

int ds4_rpc_rewind_reply(ds4_rpc_handle *h, char *err, size_t errlen) {
    if (!h) { rpc_set_err(err, errlen, "rewind_reply: null"); return 1; }
    return frame_write(h->fd, DS4_RPC_OP_REWIND_REPLY, NULL, 0, err, errlen);
}
