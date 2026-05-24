#include "ds4_web.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DS4_WEB_DEFAULT_PORT 9333
#define DS4_WEB_CONNECT_TIMEOUT_MS 3000
#define DS4_WEB_CDP_TIMEOUT_MS 20000
#define DS4_WEB_MAX_RESULT_BYTES (1024*1024)

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} web_buf;

struct ds4_web {
    char home[PATH_MAX];
    char profile_dir[PATH_MAX];
    int port;
    pid_t chrome_pid;
    bool browser_allowed;
    ds4_web_confirm_fn confirm;
    void *confirm_privdata;
    ds4_web_log_fn log;
    void *log_privdata;
    int next_cdp_id;
};

typedef struct {
    int fd;
    int next_id;
} cdp_ws;

typedef struct {
    char *id;
    char *ws_url;
} web_tab;

static void *web_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("ds4_web: malloc");
        exit(1);
    }
    return p;
}

static char *web_xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = web_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void web_buf_append(web_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        char *p = realloc(b->ptr, cap);
        if (!p) {
            perror("ds4_web: realloc");
            exit(1);
        }
        b->ptr = p;
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void web_buf_puts(web_buf *b, const char *s) {
    web_buf_append(b, s, strlen(s));
}

static char *web_buf_take(web_buf *b) {
    if (!b->ptr) return web_xstrdup("");
    char *p = b->ptr;
    b->ptr = NULL;
    b->len = b->cap = 0;
    return p;
}

static void web_set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (!err || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static void web_log(ds4_web *web, const char *msg) {
    if (web && web->log) web->log(web->log_privdata, msg);
}

static bool web_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) return false;
        *p = '/';
    }
    return mkdir(tmp, 0700) == 0 || errno == EEXIST;
}

static int web_tcp_connect(const char *host, int port, int timeout_ms,
                           char *err, size_t err_len) {
    char service[32];
    snprintf(service, sizeof(service), "%d", port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, service, &hints, &res);
    if (gai != 0) {
        web_set_err(err, err_len, "getaddrinfo %s: %s", host, gai_strerror(gai));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc == 0) {
            if (flags >= 0) fcntl(fd, F_SETFL, flags);
            break;
        }
        if (errno == EINPROGRESS) {
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            rc = poll(&pfd, 1, timeout_ms);
            if (rc > 0) {
                int soerr = 0;
                socklen_t slen = sizeof(soerr);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &slen);
                if (soerr == 0) {
                    if (flags >= 0) fcntl(fd, F_SETFL, flags);
                    break;
                }
                errno = soerr;
            }
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) web_set_err(err, err_len, "connect %s:%d failed: %s",
                            host, port, strerror(errno));
    return fd;
}

static int web_write_all(int fd, const void *buf, size_t len) {
    const char *p = buf;
    while (len) {
#ifdef MSG_NOSIGNAL
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t n = write(fd, p, len);
#endif
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static ssize_t web_read_some(int fd, char *buf, size_t len, int timeout_ms) {
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    int rc = poll(&pfd, 1, timeout_ms);
    if (rc <= 0) return rc == 0 ? 0 : -1;
    for (;;) {
        ssize_t n = read(fd, buf, len);
        if (n < 0 && errno == EINTR) continue;
        return n;
    }
}

static char *web_http_request(const char *method, int port, const char *path,
                              char *err, size_t err_len) {
    int fd = web_tcp_connect("127.0.0.1", port, DS4_WEB_CONNECT_TIMEOUT_MS,
                             err, err_len);
    if (fd < 0) return NULL;
    web_buf req = {0};
    char line[512];
    snprintf(line, sizeof(line),
             "%s %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nConnection: close\r\n\r\n",
             method, path, port);
    web_buf_puts(&req, line);
    if (web_write_all(fd, req.ptr, req.len) != 0) {
        web_set_err(err, err_len, "write HTTP request failed: %s", strerror(errno));
        close(fd);
        free(req.ptr);
        return NULL;
    }
    free(req.ptr);

    web_buf resp = {0};
    char tmp[4096];
    for (;;) {
        ssize_t n = web_read_some(fd, tmp, sizeof(tmp), DS4_WEB_CONNECT_TIMEOUT_MS);
        if (n < 0) {
            web_set_err(err, err_len, "read HTTP response failed: %s", strerror(errno));
            close(fd);
            free(resp.ptr);
            return NULL;
        }
        if (n == 0) break;
        web_buf_append(&resp, tmp, (size_t)n);
    }
    close(fd);
    if (!resp.ptr) {
        web_set_err(err, err_len, "empty HTTP response");
        return NULL;
    }
    char *body = strstr(resp.ptr, "\r\n\r\n");
    if (!body) {
        web_set_err(err, err_len, "malformed HTTP response");
        free(resp.ptr);
        return NULL;
    }
    body += 4;
    char *out = web_xstrdup(body);
    free(resp.ptr);
    return out;
}

static bool web_cdp_alive(ds4_web *web) {
    char err[160] = {0};
    char *body = web_http_request("GET", web->port, "/json/version", err, sizeof(err));
    if (!body) return false;
    bool ok = strstr(body, "webSocketDebuggerUrl") != NULL;
    free(body);
    return ok;
}

static char *web_json_get_string(const char *json, const char *key);

static char *web_url_encode(const char *s) {
    static const char hex[] = "0123456789ABCDEF";
    web_buf b = {0};
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        unsigned char c = *p;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            web_buf_append(&b, (const char *)&c, 1);
        } else {
            char e[3] = {'%', hex[c >> 4], hex[c & 15]};
            web_buf_append(&b, e, 3);
        }
    }
    return web_buf_take(&b);
}

static void web_random_bytes(unsigned char *buf, size_t len) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        size_t off = 0;
        while (off < len) {
            ssize_t n = read(fd, buf + off, len - off);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            off += (size_t)n;
        }
        close(fd);
        if (off == len) return;
    }
    uint64_t x = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);
    for (size_t i = 0; i < len; i++) {
        x = x * 6364136223846793005ULL + 1;
        buf[i] = (unsigned char)(x >> 32);
    }
}

static char *web_base64(const unsigned char *data, size_t len) {
    static const char tab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t outlen = ((len + 2) / 3) * 4;
    char *out = web_xmalloc(outlen + 1);
    size_t j = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= data[i + 2];
        out[j++] = tab[(v >> 18) & 63];
        out[j++] = tab[(v >> 12) & 63];
        out[j++] = (i + 1 < len) ? tab[(v >> 6) & 63] : '=';
        out[j++] = (i + 2 < len) ? tab[v & 63] : '=';
    }
    out[j] = '\0';
    return out;
}

static char *web_json_quote(const char *s) {
    web_buf b = {0};
    web_buf_puts(&b, "\"");
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        unsigned char c = *p;
        switch (c) {
        case '\\': web_buf_puts(&b, "\\\\"); break;
        case '"': web_buf_puts(&b, "\\\""); break;
        case '\n': web_buf_puts(&b, "\\n"); break;
        case '\r': web_buf_puts(&b, "\\r"); break;
        case '\t': web_buf_puts(&b, "\\t"); break;
        default:
            if (c < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                web_buf_puts(&b, tmp);
            } else {
                web_buf_append(&b, (const char *)&c, 1);
            }
            break;
        }
    }
    web_buf_puts(&b, "\"");
    return web_buf_take(&b);
}

static int web_ws_connect(const char *ws_url, cdp_ws *ws,
                          char *err, size_t err_len) {
    const char *p = ws_url;
    if (strncmp(p, "ws://", 5) != 0) {
        web_set_err(err, err_len, "unsupported websocket URL: %s", ws_url);
        return -1;
    }
    p += 5;
    const char *slash = strchr(p, '/');
    if (!slash) {
        web_set_err(err, err_len, "malformed websocket URL");
        return -1;
    }
    char hostport[256];
    size_t hp_len = (size_t)(slash - p);
    if (hp_len >= sizeof(hostport)) hp_len = sizeof(hostport) - 1;
    memcpy(hostport, p, hp_len);
    hostport[hp_len] = '\0';
    char *colon = strrchr(hostport, ':');
    int port = 80;
    if (colon) {
        *colon = '\0';
        port = atoi(colon + 1);
    }
    const char *host = hostport;
    int fd = web_tcp_connect(host, port, DS4_WEB_CONNECT_TIMEOUT_MS, err, err_len);
    if (fd < 0) return -1;

    unsigned char rnd[16];
    web_random_bytes(rnd, sizeof(rnd));
    char *key = web_base64(rnd, sizeof(rnd));
    web_buf req = {0};
    char line[512];
    snprintf(line, sizeof(line),
             "GET %s HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Upgrade: websocket\r\n"
             "Connection: Upgrade\r\n"
             "Sec-WebSocket-Key: %s\r\n"
             "Sec-WebSocket-Version: 13\r\n\r\n",
             slash, host, port, key);
    web_buf_puts(&req, line);
    free(key);
    if (web_write_all(fd, req.ptr, req.len) != 0) {
        web_set_err(err, err_len, "websocket handshake write failed");
        close(fd);
        free(req.ptr);
        return -1;
    }
    free(req.ptr);

    web_buf resp = {0};
    char tmp[1024];
    while (!strstr(resp.ptr ? resp.ptr : "", "\r\n\r\n")) {
        ssize_t n = web_read_some(fd, tmp, sizeof(tmp), DS4_WEB_CONNECT_TIMEOUT_MS);
        if (n <= 0) {
            web_set_err(err, err_len, "websocket handshake read failed");
            close(fd);
            free(resp.ptr);
            return -1;
        }
        web_buf_append(&resp, tmp, (size_t)n);
        if (resp.len > 8192) break;
    }
    bool ok = resp.ptr && strstr(resp.ptr, " 101 ") != NULL;
    free(resp.ptr);
    if (!ok) {
        web_set_err(err, err_len, "websocket handshake rejected");
        close(fd);
        return -1;
    }
    ws->fd = fd;
    ws->next_id = 1;
    return 0;
}

static void web_ws_close(cdp_ws *ws) {
    if (ws && ws->fd >= 0) {
        close(ws->fd);
        ws->fd = -1;
    }
}

static int web_read_exact(int fd, unsigned char *buf, size_t len, int timeout_ms) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = web_read_some(fd, (char *)buf + off, len - off, timeout_ms);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return 0;
}

static int web_ws_send_text(cdp_ws *ws, const char *text,
                            char *err, size_t err_len) {
    size_t len = strlen(text);
    web_buf frame = {0};
    unsigned char hdr[14];
    size_t h = 0;
    hdr[h++] = 0x81;
    if (len < 126) {
        hdr[h++] = 0x80 | (unsigned char)len;
    } else if (len <= 0xffff) {
        hdr[h++] = 0x80 | 126;
        hdr[h++] = (unsigned char)(len >> 8);
        hdr[h++] = (unsigned char)len;
    } else {
        hdr[h++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--) hdr[h++] = (unsigned char)((uint64_t)len >> (i * 8));
    }
    unsigned char mask[4];
    web_random_bytes(mask, sizeof(mask));
    for (int i = 0; i < 4; i++) hdr[h++] = mask[i];
    web_buf_append(&frame, (const char *)hdr, h);
    for (size_t i = 0; i < len; i++) {
        char c = text[i] ^ mask[i & 3];
        web_buf_append(&frame, &c, 1);
    }
    int rc = web_write_all(ws->fd, frame.ptr, frame.len);
    free(frame.ptr);
    if (rc != 0) {
        web_set_err(err, err_len, "websocket write failed: %s", strerror(errno));
        return -1;
    }
    return 0;
}

static int web_ws_send_pong(cdp_ws *ws, const unsigned char *payload, size_t len) {
    if (len > 125) len = 125;
    unsigned char hdr[2 + 4 + 125];
    hdr[0] = 0x8a;
    hdr[1] = 0x80 | (unsigned char)len;
    unsigned char mask[4];
    web_random_bytes(mask, sizeof(mask));
    memcpy(hdr + 2, mask, 4);
    for (size_t i = 0; i < len; i++) hdr[6 + i] = payload[i] ^ mask[i & 3];
    return web_write_all(ws->fd, hdr, 6 + len);
}

static char *web_ws_read_message(cdp_ws *ws, char *err, size_t err_len) {
    web_buf msg = {0};
    for (;;) {
        unsigned char h[2];
        if (web_read_exact(ws->fd, h, 2, DS4_WEB_CDP_TIMEOUT_MS) != 0) {
            web_set_err(err, err_len, "websocket read timeout");
            free(msg.ptr);
            return NULL;
        }
        bool fin = (h[0] & 0x80) != 0;
        int opcode = h[0] & 0x0f;
        bool masked = (h[1] & 0x80) != 0;
        uint64_t len = h[1] & 0x7f;
        if (len == 126) {
            unsigned char x[2];
            if (web_read_exact(ws->fd, x, 2, DS4_WEB_CDP_TIMEOUT_MS) != 0) goto fail;
            len = ((uint64_t)x[0] << 8) | x[1];
        } else if (len == 127) {
            unsigned char x[8];
            if (web_read_exact(ws->fd, x, 8, DS4_WEB_CDP_TIMEOUT_MS) != 0) goto fail;
            len = 0;
            for (int i = 0; i < 8; i++) len = (len << 8) | x[i];
        }
        unsigned char mask[4] = {0};
        if (masked && web_read_exact(ws->fd, mask, 4, DS4_WEB_CDP_TIMEOUT_MS) != 0)
            goto fail;
        if (len > DS4_WEB_MAX_RESULT_BYTES * 4ULL) {
            web_set_err(err, err_len, "websocket message too large");
            free(msg.ptr);
            return NULL;
        }
        unsigned char *payload = web_xmalloc((size_t)len + 1);
        if (len && web_read_exact(ws->fd, payload, (size_t)len,
                                  DS4_WEB_CDP_TIMEOUT_MS) != 0) {
            free(payload);
            goto fail;
        }
        for (uint64_t i = 0; masked && i < len; i++) payload[i] ^= mask[i & 3];
        payload[len] = '\0';
        if (opcode == 0x8) {
            free(payload);
            web_set_err(err, err_len, "websocket closed");
            free(msg.ptr);
            return NULL;
        } else if (opcode == 0x9) {
            web_ws_send_pong(ws, payload, (size_t)len);
            free(payload);
            continue;
        } else if (opcode == 0x1 || opcode == 0x0) {
            web_buf_append(&msg, (const char *)payload, (size_t)len);
            free(payload);
            if (fin) return web_buf_take(&msg);
        } else {
            free(payload);
        }
    }
fail:
    web_set_err(err, err_len, "websocket frame read failed");
    free(msg.ptr);
    return NULL;
}

static bool web_json_id_matches(const char *json, int id) {
    const char *p = strstr(json, "\"id\"");
    if (!p) return false;
    p = strchr(p, ':');
    if (!p) return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p) == id;
}

static char *web_cdp_call(cdp_ws *ws, const char *method, const char *params,
                          char *err, size_t err_len) {
    int id = ws->next_id++;
    web_buf req = {0};
    char head[256];
    snprintf(head, sizeof(head), "{\"id\":%d,\"method\":", id);
    web_buf_puts(&req, head);
    char *qmethod = web_json_quote(method);
    web_buf_puts(&req, qmethod);
    free(qmethod);
    if (params && params[0]) {
        web_buf_puts(&req, ",\"params\":");
        web_buf_puts(&req, params);
    }
    web_buf_puts(&req, "}");
    char *wire = web_buf_take(&req);
    if (web_ws_send_text(ws, wire, err, err_len) != 0) {
        free(wire);
        return NULL;
    }
    free(wire);
    for (;;) {
        char *msg = web_ws_read_message(ws, err, err_len);
        if (!msg) return NULL;
        if (web_json_id_matches(msg, id)) return msg;
        free(msg);
    }
}

static void web_cdp_call_optional(cdp_ws *ws, const char *method, const char *params) {
    char err[160] = {0};
    char *resp = web_cdp_call(ws, method, params, err, sizeof(err));
    free(resp);
}

static int web_hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        int x;
        if (c >= '0' && c <= '9') x = c - '0';
        else if (c >= 'a' && c <= 'f') x = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') x = c - 'A' + 10;
        else return -1;
        v = (v << 4) | x;
    }
    return v;
}

static void web_utf8_append(web_buf *b, unsigned code) {
    char out[4];
    if (code <= 0x7f) {
        out[0] = (char)code;
        web_buf_append(b, out, 1);
    } else if (code <= 0x7ff) {
        out[0] = (char)(0xc0 | (code >> 6));
        out[1] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 2);
    } else if (code <= 0xffff) {
        out[0] = (char)(0xe0 | (code >> 12));
        out[1] = (char)(0x80 | ((code >> 6) & 0x3f));
        out[2] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 3);
    } else {
        out[0] = (char)(0xf0 | (code >> 18));
        out[1] = (char)(0x80 | ((code >> 12) & 0x3f));
        out[2] = (char)(0x80 | ((code >> 6) & 0x3f));
        out[3] = (char)(0x80 | (code & 0x3f));
        web_buf_append(b, out, 4);
    }
}

static char *web_json_parse_string_at(const char *q, const char **endp) {
    if (*q != '"') return NULL;
    q++;
    web_buf b = {0};
    while (*q && *q != '"') {
        if (*q != '\\') {
            web_buf_append(&b, q++, 1);
            continue;
        }
        q++;
        switch (*q) {
        case '"': web_buf_append(&b, "\"", 1); q++; break;
        case '\\': web_buf_append(&b, "\\", 1); q++; break;
        case '/': web_buf_append(&b, "/", 1); q++; break;
        case 'b': web_buf_append(&b, "\b", 1); q++; break;
        case 'f': web_buf_append(&b, "\f", 1); q++; break;
        case 'n': web_buf_append(&b, "\n", 1); q++; break;
        case 'r': web_buf_append(&b, "\r", 1); q++; break;
        case 't': web_buf_append(&b, "\t", 1); q++; break;
        case 'u': {
            int v = web_hex4(q + 1);
            if (v < 0) { free(b.ptr); return NULL; }
            q += 5;
            if (v >= 0xd800 && v <= 0xdbff && q[0] == '\\' && q[1] == 'u') {
                int lo = web_hex4(q + 2);
                if (lo >= 0xdc00 && lo <= 0xdfff) {
                    unsigned code = 0x10000 + (((unsigned)v - 0xd800) << 10) +
                                    ((unsigned)lo - 0xdc00);
                    web_utf8_append(&b, code);
                    q += 6;
                    break;
                }
            }
            web_utf8_append(&b, (unsigned)v);
            break;
        }
        default:
            if (*q) web_buf_append(&b, q++, 1);
            break;
        }
    }
    if (*q != '"') {
        free(b.ptr);
        return NULL;
    }
    if (endp) *endp = q + 1;
    return web_buf_take(&b);
}

static char *web_json_get_string(const char *json, const char *key) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = json;
    while ((p = strstr(p, pat)) != NULL) {
        p += strlen(pat);
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p++ != ':') continue;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '"') return web_json_parse_string_at(p, NULL);
    }
    return NULL;
}

static char *web_cdp_eval_string(cdp_ws *ws, const char *expr,
                                 char *err, size_t err_len) {
    char *qexpr = web_json_quote(expr);
    web_buf params = {0};
    web_buf_puts(&params, "{\"expression\":");
    web_buf_puts(&params, qexpr);
    web_buf_puts(&params, ",\"returnByValue\":true,\"awaitPromise\":true,\"includeCommandLineAPI\":true}");
    free(qexpr);
    char *params_s = web_buf_take(&params);
    char *resp = web_cdp_call(ws, "Runtime.evaluate", params_s, err, err_len);
    free(params_s);
    if (!resp) return NULL;
    if (strstr(resp, "\"exceptionDetails\"")) {
        web_set_err(err, err_len, "JavaScript evaluation failed");
        free(resp);
        return NULL;
    }
    char *val = web_json_get_string(resp, "value");
    free(resp);
    if (!val) web_set_err(err, err_len, "Runtime.evaluate did not return a string");
    return val;
}

static bool web_wait_ready(cdp_ws *ws, char *err, size_t err_len) {
    const char *expr = "document.readyState";
    for (int i = 0; i < 80; i++) {
        char *state = web_cdp_eval_string(ws, expr, err, err_len);
        if (state && (!strcmp(state, "complete") || !strcmp(state, "interactive"))) {
            free(state);
            usleep(800000);
            return true;
        }
        free(state);
        usleep(250000);
    }
    return true;
}

static bool web_cdp_navigate(cdp_ws *ws, const char *url,
                             char *err, size_t err_len) {
    char *qurl = web_json_quote(url);
    web_buf params = {0};
    web_buf_puts(&params, "{\"url\":");
    web_buf_puts(&params, qurl);
    web_buf_puts(&params, "}");
    free(qurl);
    char *params_s = web_buf_take(&params);
    char *resp = web_cdp_call(ws, "Page.navigate", params_s, err, err_len);
    free(params_s);
    if (!resp) return false;
    free(resp);
    return true;
}

static bool web_page_probe(cdp_ws *ws, char **href_out, char **ready_out,
                           long *text_len_out, char *err, size_t err_len) {
    const char *expr =
        "location.href+'\\n'+document.readyState+'\\n'+"
        "((document.body&&document.body.innerText)||'').length";
    char *probe = web_cdp_eval_string(ws, expr, err, err_len);
    if (!probe) return false;

    char *nl1 = strchr(probe, '\n');
    char *nl2 = nl1 ? strchr(nl1 + 1, '\n') : NULL;
    if (!nl1 || !nl2) {
        free(probe);
        web_set_err(err, err_len, "page readiness probe returned malformed data");
        return false;
    }
    *nl1 = '\0';
    *nl2 = '\0';
    if (href_out) *href_out = web_xstrdup(probe);
    if (ready_out) *ready_out = web_xstrdup(nl1 + 1);
    if (text_len_out) *text_len_out = strtol(nl2 + 1, NULL, 10);
    free(probe);
    return true;
}

static bool web_wait_navigated_ready(cdp_ws *ws, const char *url,
                                     char *err, size_t err_len) {
    (void)url;
    long last_len = -1;
    int stable = 0;
    bool saw_real_url = false;

    for (int i = 0; i < 100; i++) {
        char *href = NULL;
        char *ready = NULL;
        long text_len = 0;
        bool ok = web_page_probe(ws, &href, &ready, &text_len, err, err_len);
        if (!ok) {
            free(href);
            free(ready);
            usleep(250000);
            continue;
        }

        bool real_url = href && href[0] &&
                        strcmp(href, "about:blank") &&
                        strncmp(href, "chrome://", 9);
        bool ready_state = ready &&
            (!strcmp(ready, "complete") || !strcmp(ready, "interactive"));
        if (real_url) saw_real_url = true;
        if (text_len > 0 && text_len == last_len) stable++;
        else stable = 0;
        last_len = text_len;

        free(href);
        free(ready);

        if (saw_real_url && ready_state && text_len > 0 && stable >= 2) {
            usleep(500000);
            return true;
        }
        if (saw_real_url && ready_state && i >= 24) return true;
        usleep(250000);
    }
    return true;
}

static bool web_cdp_prepare_page(cdp_ws *ws, char *err, size_t err_len) {
    char *resp = web_cdp_call(ws, "Page.enable", "{}", err, err_len);
    if (!resp) return false;
    free(resp);
    resp = web_cdp_call(ws, "Runtime.enable", "{}", err, err_len);
    if (!resp) return false;
    free(resp);
    web_cdp_call_optional(ws, "Emulation.setFocusEmulationEnabled",
                          "{\"enabled\":true}");
    web_cdp_call_optional(ws, "Emulation.setDeviceMetricsOverride",
                          "{\"width\":1365,\"height\":900,\"deviceScaleFactor\":1,\"mobile\":false}");
    return web_wait_ready(ws, err, err_len);
}

static void web_scroll_dynamic_page(cdp_ws *ws) {
    const char *expr =
        "(() => new Promise(resolve => {"
        "const root=()=>document.scrollingElement||document.documentElement||document.body;"
        "const blockSel='h1,h2,h3,h4,h5,h6,p,li,pre,blockquote,td,th,[id=\"content-text\"],[class*=\"comment-body\"],[class*=\"comment-content\"],[data-testid*=\"comment-text\"]';"
        "const lazySel='[onscroll],[loading=\"lazy\"],[data-src],[data-lazy],[class*=\"lazy\"],[class*=\"infinite\"],[class*=\"virtual\"],[role=\"feed\"],[id*=\"comment\"],[class*=\"comment\"],[data-testid*=\"comment\"]';"
        "const hookCount=()=>{let n=0;try{if(window.onscroll)n++;if(document.onscroll)n++;if(document.body&&document.body.onscroll)n++;}catch(e){}"
        "try{if(typeof getEventListeners==='function'){for(const o of [window,document,document.body]){if(!o)continue;const ev=getEventListeners(o);if(ev&&ev.scroll)n+=ev.scroll.length;}}}catch(e){}"
        "try{n+=document.querySelectorAll(lazySel).length;}catch(e){}return n;};"
        "const metrics=()=>{const r=root();return {"
        "height:r?r.scrollHeight:0,"
        "view:innerHeight||900,"
        "y:scrollY||(r&&r.scrollTop)||0,"
        "text:((document.body&&document.body.innerText)||'').length,"
        "links:document.links?document.links.length:0,"
        "blocks:document.body?document.body.querySelectorAll(blockSel).length:0,"
        "hooks:hookCount()};};"
        "const sig=m=>[m.height,m.text,m.links,m.blocks].join('|');"
        "const grew=(a,b)=>b.height>a.height+20||b.text>a.text+200||b.links>a.links+2||b.blocks>a.blocks+2;"
        "const scrollOnce=()=>{const r=root();if(!r)return;"
        "const h=Math.max(700,Math.floor((innerHeight||900)*0.85));"
        "window.scrollTo(0,Math.min(r.scrollHeight,(scrollY||r.scrollTop||0)+h));};"
        "let last=metrics(),lastSig=sig(last),same=0,steps=0;"
        "const scrollable=last.height>last.view*1.35;"
        "if(!scrollable||last.hooks===0){resolve('scroll skipped hooks='+last.hooks+' text='+last.text);return;}"
        "const tick=()=>{"
        "if(steps>=28){resolve('scrolled '+steps+' text='+last.text);return;}"
        "const before=last;"
        "scrollOnce();steps++;"
        "setTimeout(()=>{const now=metrics(),nowSig=sig(now);"
        "if(nowSig===lastSig)same++;else same=0;"
        "const loaded=grew(before,now);"
        "last=now;lastSig=nowSig;"
        "if(steps===1&&!loaded){resolve('scroll probe unchanged text='+now.text);return;}"
        "const atBottom=now.y+now.view+20>=now.height;"
        "if(same>=4||(atBottom&&same>=1)){resolve('scrolled '+steps+' text='+now.text);return;}"
        "tick();},900);"
        "};tick();"
        "}))()";
    char err[160] = {0};
    char *res = web_cdp_eval_string(ws, expr, err, sizeof(err));
    free(res);
}

static char *web_chrome_executable(void) {
    const char *env = getenv("DS4_CHROME");
    if (env && env[0]) return web_xstrdup(env);
#ifdef __APPLE__
    if (access("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", X_OK) == 0)
        return web_xstrdup("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
    if (access("/Applications/Chromium.app/Contents/MacOS/Chromium", X_OK) == 0)
        return web_xstrdup("/Applications/Chromium.app/Contents/MacOS/Chromium");
#endif
    const char *paths[] = {
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
        "/opt/google/chrome/chrome",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        if (access(paths[i], X_OK) == 0) return web_xstrdup(paths[i]);
    }

    const char *names[] = {
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        NULL
    };
    const char *pathenv = getenv("PATH");
    if (pathenv) {
        char *path = web_xstrdup(pathenv);
        char *save = NULL;
        for (char *dir = strtok_r(path, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
            for (int i = 0; names[i]; i++) {
                char candidate[PATH_MAX];
                snprintf(candidate, sizeof(candidate), "%s/%s", dir[0] ? dir : ".", names[i]);
                if (access(candidate, X_OK) == 0) {
                    char *res = web_xstrdup(candidate);
                    free(path);
                    return res;
                }
            }
        }
        free(path);
    }
    return web_xstrdup("google-chrome");
}

#ifdef __APPLE__
static const char *web_macos_chrome_app_name(void) {
    if (getenv("DS4_CHROME")) return NULL;
    if (access("/Applications/Google Chrome.app", F_OK) == 0)
        return "Google Chrome";
    if (access("/Applications/Chromium.app", F_OK) == 0)
        return "Chromium";
    return NULL;
}
#endif

static bool web_spawn_chrome(ds4_web *web, char *err, size_t err_len) {
    if (!web_mkdir_p(web->profile_dir)) {
        web_set_err(err, err_len, "failed to create Chrome profile dir %s: %s",
                    web->profile_dir, strerror(errno));
        return false;
    }
    char *exe = web_chrome_executable();
#ifdef __APPLE__
    const char *mac_app_name = web_macos_chrome_app_name();
    bool launched_via_open = mac_app_name != NULL && access("/usr/bin/open", X_OK) == 0;
#else
    bool launched_via_open = false;
#endif
    char port_arg[64], profile_arg[PATH_MAX + 64];
    snprintf(port_arg, sizeof(port_arg), "--remote-debugging-port=%d", web->port);
    snprintf(profile_arg, sizeof(profile_arg), "--user-data-dir=%s", web->profile_dir);
    pid_t pid = fork();
    if (pid < 0) {
        web_set_err(err, err_len, "failed to fork Chrome: %s", strerror(errno));
        free(exe);
        return false;
    }
    if (pid == 0) {
        int nullfd = open("/dev/null", O_RDWR);
        if (nullfd >= 0) {
            dup2(nullfd, STDOUT_FILENO);
            dup2(nullfd, STDERR_FILENO);
            if (nullfd > 2) close(nullfd);
        }
#ifdef __APPLE__
        if (launched_via_open) {
            execlp("/usr/bin/open", "open", "-g", "-na", mac_app_name,
                   "--args", port_arg, "--remote-allow-origins=*",
                   profile_arg, "--no-first-run", "--no-default-browser-check",
                   "--disable-sync", "--use-mock-keychain", "--password-store=basic",
                   "--mute-audio", "about:blank", (char *)NULL);
        } else {
            execlp(exe, exe, port_arg, "--remote-allow-origins=*",
                   profile_arg, "--no-first-run", "--no-default-browser-check",
                   "--disable-sync", "--use-mock-keychain", "--password-store=basic",
                   "--mute-audio", "about:blank", (char *)NULL);
        }
#else
        if (geteuid() == 0) {
            execlp(exe, exe, port_arg, "--remote-allow-origins=*",
                   profile_arg, "--no-first-run", "--no-default-browser-check",
                   "--disable-sync", "--password-store=basic", "--no-sandbox",
                   "--mute-audio", "about:blank", (char *)NULL);
        } else {
            execlp(exe, exe, port_arg, "--remote-allow-origins=*",
                   profile_arg, "--no-first-run", "--no-default-browser-check",
                   "--disable-sync", "--password-store=basic",
                   "--mute-audio", "about:blank", (char *)NULL);
        }
#endif
        _exit(127);
    }
    free(exe);
    web->chrome_pid = pid;
    for (int i = 0; i < 80; i++) {
        if (web_cdp_alive(web)) {
            web_log(web, "Chrome browser session is ready");
            return true;
        }
        int status = 0;
        pid_t rc = waitpid(pid, &status, WNOHANG);
        if (rc == pid) {
            web->chrome_pid = 0;
            if (launched_via_open) continue;
            web_set_err(err, err_len, "Chrome exited before CDP became ready");
            return false;
        }
        usleep(250000);
    }
    web_set_err(err, err_len, "Chrome did not expose CDP on port %d", web->port);
    return false;
}

static bool web_ensure_browser(ds4_web *web, char *err, size_t err_len) {
    if (web_cdp_alive(web)) return true;
    if (web->chrome_pid > 0) {
        int status = 0;
        waitpid(web->chrome_pid, &status, WNOHANG);
        web->chrome_pid = 0;
    }
    if (!web->browser_allowed) {
        if (!web->confirm) {
            web_set_err(err, err_len,
                        "starting a visible Chrome browser requires interactive approval");
            return false;
        }
        if (!web->confirm(web->confirm_privdata,
                          "The web tool wants to start a visible Chrome browser. Allow? (y/n) ",
                          err, err_len))
        {
            if (err && !err[0]) web_set_err(err, err_len, "user denied Chrome browser start");
            return false;
        }
        web->browser_allowed = true;
    }
    return web_spawn_chrome(web, err, err_len);
}

static void web_tab_free(web_tab *tab) {
    if (!tab) return;
    free(tab->id);
    free(tab->ws_url);
    tab->id = NULL;
    tab->ws_url = NULL;
}

static char *web_browser_ws_url(ds4_web *web, char *err, size_t err_len) {
    char *body = web_http_request("GET", web->port, "/json/version", err, err_len);
    if (!body) return NULL;
    char *ws = web_json_get_string(body, "webSocketDebuggerUrl");
    free(body);
    if (!ws) web_set_err(err, err_len, "Chrome did not return a browser WebSocket URL");
    return ws;
}

static bool web_open_tab(ds4_web *web, const char *url, web_tab *tab,
                         char *err, size_t err_len) {
    memset(tab, 0, sizeof(*tab));

    char *browser_url = web_browser_ws_url(web, err, err_len);
    if (!browser_url) return false;
    cdp_ws browser = {.fd = -1};
    if (web_ws_connect(browser_url, &browser, err, err_len) != 0) {
        free(browser_url);
        return false;
    }
    free(browser_url);

    char *qurl = web_json_quote(url);
    web_buf params = {0};
    web_buf_puts(&params, "{\"url\":");
    web_buf_puts(&params, qurl);
    web_buf_puts(&params, ",\"background\":true,\"newWindow\":false}");
    free(qurl);
    char *params_s = web_buf_take(&params);
    char *resp = web_cdp_call(&browser, "Target.createTarget",
                              params_s, err, err_len);
    free(params_s);
    web_ws_close(&browser);
    if (!resp) return false;

    tab->id = web_json_get_string(resp, "targetId");
    free(resp);
    if (!tab->id) {
        web_tab_free(tab);
        web_set_err(err, err_len, "Chrome did not return a page target id");
        return false;
    }

    char ws_url[PATH_MAX + 128];
    snprintf(ws_url, sizeof(ws_url), "ws://127.0.0.1:%d/devtools/page/%s",
             web->port, tab->id);
    tab->ws_url = web_xstrdup(ws_url);
    return true;
}

static void web_close_tab(ds4_web *web, const web_tab *tab) {
    if (!web || !tab || !tab->id || !tab->id[0]) return;
    char *enc = web_url_encode(tab->id);
    web_buf path = {0};
    web_buf_puts(&path, "/json/close/");
    web_buf_puts(&path, enc);
    free(enc);

    char err[160] = {0};
    char *path_s = web_buf_take(&path);
    char *body = web_http_request("GET", web->port, path_s, err, sizeof(err));
    free(path_s);
    if (body) {
        free(body);
    } else if (err[0]) {
        web_log(web, err);
    }
}

static const char *web_click_google_consent_js =
"(() => {"
"const clean=s=>(s||'').replace(/\\s+/g,' ').trim();"
"const pats=[/accept all/i,/i agree/i,/agree/i,/accetta tutto/i,/tout accepter/i,/aceptar todo/i,/alle akzeptieren/i];"
"const els=[...document.querySelectorAll('button,[role=button],input[type=submit],a')];"
"for (const el of els){const t=clean(el.innerText||el.value||el.textContent);"
"if(!t)continue; if(pats.some(p=>p.test(t))){el.click(); return 'clicked '+t;}}"
"return '';"
"})()";

static const char *web_extract_search_js =
"(() => {"
"const clean=s=>(s||'').replace(/\\s+/g,' ').trim();"
"const esc=s=>clean(s).replace(/\\\\/g,'\\\\\\\\').replace(/\\[/g,'\\\\[').replace(/\\]/g,'\\\\]').replace(/\\n/g,' ');"
"const visible=el=>{const r=el.getBoundingClientRect();const st=getComputedStyle(el);return r.width>0&&r.height>0&&st.display!=='none'&&st.visibility!=='hidden'&&st.opacity!=='0';};"
"const bad=h=>(/(^|\\.)google\\./.test(h)||/(^|\\.)gstatic\\./.test(h)||/(^|\\.)googleusercontent\\./.test(h));"
"const lines=['# Google search results','',`URL: ${location.href}`,'','## Visible links'];"
"const seen=new Set();"
"for(const a of document.querySelectorAll('a[href]')){if(!visible(a))continue;let href=a.href||'';"
"try{const u=new URL(href);if(u.pathname==='/url'&&u.searchParams.get('q'))href=u.searchParams.get('q');}catch{}"
"let u;try{u=new URL(href);}catch{continue;}if(!/^https?:$/.test(u.protocol))continue;if(bad(u.hostname))continue;"
"const text=esc(a.innerText||a.textContent);if(text.length<3)continue;if(seen.has(u.href))continue;seen.add(u.href);"
"lines.push(`- [${text.slice(0,180)}](${u.href})`);if(seen.size>=20)break;}"
"lines.push('','## Text snapshot',clean(document.body.innerText).slice(0,1200));"
"return lines.join('\\n');"
"})()";

static const char *web_extract_page_js =
"(() => {"
"const clean=s=>(s||'').replace(/\\s+/g,' ').trim();"
"const esc=s=>clean(s).replace(/\\\\/g,'\\\\\\\\').replace(/\\[/g,'\\\\[').replace(/\\]/g,'\\\\]').replace(/\\n/g,' ');"
"const visible=el=>{const r=el.getBoundingClientRect();const st=getComputedStyle(el);return r.width>0&&r.height>0&&st.display!=='none'&&st.visibility!=='hidden'&&st.opacity!=='0';};"
"const inline=n=>{if(!n)return'';if(n.nodeType===3)return n.nodeValue;if(n.nodeType!==1)return'';const el=n;"
"if(el.tagName==='SCRIPT'||el.tagName==='STYLE'||el.tagName==='NOSCRIPT')return'';"
"if(el.tagName==='A'){const t=esc(el.innerText||el.textContent);const h=el.href||'';return t&&h?`[${t}](${h})`:t;}"
"if(el.tagName==='CODE')return '`'+clean(el.innerText||el.textContent).replace(/`/g,'\\\\`')+'`';"
"return [...el.childNodes].map(inline).join('');};"
"const lines=[`# ${clean(document.title)||location.href}`,'',`URL: ${location.href}`,'','## Content'];"
"const blocks=[...document.body.querySelectorAll('h1,h2,h3,h4,h5,h6,p,li,pre,blockquote,td,th,[id=\"content-text\"],[class*=\"comment-body\"],[class*=\"comment-content\"],[data-testid*=\"comment-text\"]')];"
"const seen=new Set();"
"for(const el of blocks){if(!visible(el))continue;let s='';const tag=el.tagName;"
"if(/^H[1-6]$/.test(tag)){s='#'.repeat(Number(tag[1]))+' '+inline(el);}"
"else if(tag==='LI'){s='- '+inline(el);}"
"else if(tag==='PRE'){s='```\\n'+(el.innerText||el.textContent||'').trimEnd()+'\\n```';}"
"else if(tag==='BLOCKQUOTE'){s='> '+clean(el.innerText||el.textContent);}"
"else{s=inline(el);}s=s.trim();if(!s||seen.has(s))continue;seen.add(s);lines.push('',s);"
"if(lines.join('\\n').length>900000){lines.push('','[Content truncated by browser extractor.]');break;}}"
"lines.push('','## Visible links');let n=0;const linkSeen=new Set();"
"for(const a of document.querySelectorAll('a[href]')){if(!visible(a))continue;const t=esc(a.innerText||a.textContent);if(t.length<3)continue;"
"let u;try{u=new URL(a.href);}catch{continue;}if(!/^https?:$/.test(u.protocol)||linkSeen.has(u.href))continue;linkSeen.add(u.href);"
"lines.push(`- [${t.slice(0,160)}](${u.href})`);if(++n>=80)break;}"
"return lines.join('\\n');"
"})()";

static char *web_run_page_js(ds4_web *web, const char *url, const char *js,
                             bool dynamic_scroll,
                             char *err, size_t err_len) {
    if (!web_ensure_browser(web, err, err_len)) return NULL;
    web_tab tab = {0};
    if (!web_open_tab(web, "about:blank", &tab, err, err_len)) return NULL;
    cdp_ws ws = {.fd = -1};
    if (web_ws_connect(tab.ws_url, &ws, err, err_len) != 0) {
        web_close_tab(web, &tab);
        web_tab_free(&tab);
        return NULL;
    }
    if (!web_cdp_prepare_page(&ws, err, err_len)) {
        web_ws_close(&ws);
        web_close_tab(web, &tab);
        web_tab_free(&tab);
        return NULL;
    }
    if (!web_cdp_navigate(&ws, url, err, err_len) ||
        !web_wait_navigated_ready(&ws, url, err, err_len))
    {
        web_ws_close(&ws);
        web_close_tab(web, &tab);
        web_tab_free(&tab);
        return NULL;
    }
    char *clicked = web_cdp_eval_string(&ws, web_click_google_consent_js, err, err_len);
    if (clicked && clicked[0]) {
        web_log(web, clicked);
        usleep(1500000);
        (void)web_wait_navigated_ready(&ws, url, err, err_len);
    }
    free(clicked);
    if (dynamic_scroll) web_scroll_dynamic_page(&ws);
    char *out = web_cdp_eval_string(&ws, js, err, err_len);
    web_ws_close(&ws);
    web_close_tab(web, &tab);
    web_tab_free(&tab);
    return out;
}

ds4_web *ds4_web_create(const ds4_web_config *cfg) {
    ds4_web *web = web_xmalloc(sizeof(*web));
    memset(web, 0, sizeof(*web));
    const char *home = cfg && cfg->home_dir && cfg->home_dir[0] ?
        cfg->home_dir : getenv("HOME");
    if (!home || !home[0]) home = ".";
    snprintf(web->home, sizeof(web->home), "%s", home);
    snprintf(web->profile_dir, sizeof(web->profile_dir), "%s/.ds4/browser", home);
    web->port = cfg && cfg->port > 0 ? cfg->port : DS4_WEB_DEFAULT_PORT;
    web->chrome_pid = 0;
    web->next_cdp_id = 1;
    if (cfg) {
        web->confirm = cfg->confirm;
        web->confirm_privdata = cfg->confirm_privdata;
        web->log = cfg->log;
        web->log_privdata = cfg->log_privdata;
    }
    return web;
}

void ds4_web_free(ds4_web *web) {
    if (!web) return;
    /* Do not kill Chrome.  The browser profile is user-visible state and keeping
     * it alive makes repeated web tool calls cheaper and less suspicious. */
    free(web);
}

char *ds4_web_google_search(ds4_web *web, const char *query,
                            char *err, size_t err_len) {
    if (!web) {
        web_set_err(err, err_len, "web subsystem is not initialized");
        return NULL;
    }
    if (!query || !query[0]) {
        web_set_err(err, err_len, "google_search requires query");
        return NULL;
    }
    char *q = web_url_encode(query);
    web_buf url = {0};
    web_buf_puts(&url, "https://www.google.com/search?q=");
    web_buf_puts(&url, q);
    free(q);
    char *url_s = web_buf_take(&url);
    char *out = web_run_page_js(web, url_s, web_extract_search_js, false, err, err_len);
    free(url_s);
    return out;
}

char *ds4_web_visit_page(ds4_web *web, const char *url,
                         char *err, size_t err_len) {
    if (!web) {
        web_set_err(err, err_len, "web subsystem is not initialized");
        return NULL;
    }
    if (!url || !url[0]) {
        web_set_err(err, err_len, "visit_page requires url");
        return NULL;
    }
    return web_run_page_js(web, url, web_extract_page_js, true, err, err_len);
}
