#include "ds4.h"
#include "rax.h"

/* OpenAI/Anthropic compatible local server.
 *
 * HTTP is intentionally simple: each client connection is handled by a small
 * blocking thread that parses one request, then queues a job to the single
 * Metal worker.  The worker owns the ds4_session and therefore owns all live KV
 * cache state.  That keeps session reuse, disk checkpointing, and future
 * batching decisions in one place instead of spreading graph mutations across
 * client threads. */

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_listen_fd = -1;

#define DS4_SERVER_IO_TIMEOUT_SEC 10
#define DS4_SERVER_SEND_STALL_TIMEOUT_MS 2000

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

static void die(const char *msg) {
    fprintf(stderr, "ds4-server: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static bool random_bytes(void *dst, size_t len) {
    unsigned char *p = dst;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return false;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            close(fd);
            return false;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    close(fd);
    return true;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static void buf_reserve(buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) cap *= 2;
    b->ptr = xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void buf_append(buf *b, const void *p, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void buf_putc(buf *b, char c) {
    buf_append(b, &c, 1);
}

static void buf_puts(buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}

static void buf_printf(buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) die("vsnprintf failed");
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *buf_take(buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void utf8_put(buf *b, uint32_t cp) {
    if (cp <= 0x7f) {
        buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static bool json_string(const char **p, char **out) {
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            buf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff && json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
            }
            utf8_put(&b, cp);
            break;
        }
        default:
            goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

static bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    if (v < 0) v = 0;
    if (v > INT_MAX) v = INT_MAX;
    *out = (int)v;
    return true;
}

static bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) {
        *out = true;
        return true;
    }
    if (json_lit(p, "false")) {
        *out = false;
        return true;
    }
    return false;
}

/* The request parser only understands the API fields we use and skips the
 * rest.  Skipping is recursive because JSON values nest, so keep an explicit
 * ceiling: without it, a useless ignored field like {"x":[[[...]]]} can spend
 * the whole C stack before the request is rejected. */
#define JSON_MAX_NESTING 256

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') {
        (*p)++;
        return true;
    }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

static bool json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

static bool json_raw_value(const char **p, char **out) {
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    size_t n = (size_t)(*p - start);
    char *s = xmalloc(n + 1);
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}

static char *json_minify_raw_value(const char *json) {
    const char *p = json ? json : "null";
    json_ws(&p);
    const char *start = p;
    if (!json_skip_value(&p)) return xstrdup(json ? json : "null");
    const char *end = p;

    buf b = {0};
    bool in_string = false;
    bool escape = false;
    for (const char *s = start; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (in_string) {
            buf_putc(&b, (char)c);
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
        } else if (c == '"') {
            in_string = true;
            buf_putc(&b, (char)c);
        } else if (!isspace(c)) {
            buf_putc(&b, (char)c);
        }
    }
    return buf_take(&b);
}

static bool json_content(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }

    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) goto fail;
            buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            json_ws(p);
            while (**p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) goto fail;
                json_ws(p);
                if (**p != ':') {
                    free(key);
                    goto fail;
                }
                (*p)++;
                if (!strcmp(key, "text")) {
                    char *s = NULL;
                    if (!json_string(p, &s)) {
                        free(key);
                        goto fail;
                    }
                    buf_puts(&b, s);
                    free(s);
                } else if (!json_skip_value(p)) {
                    free(key);
                    goto fail;
                }
                free(key);
                json_ws(p);
                if (**p == ',') (*p)++;
                json_ws(p);
            }
            if (**p != '}') goto fail;
            (*p)++;
        } else if (!json_skip_value(p)) {
            goto fail;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

typedef enum {
    REQ_CHAT,
    REQ_COMPLETION,
} req_kind;

typedef enum {
    API_OPENAI,
    API_ANTHROPIC,
    API_RESPONSES,
} api_style;

static void random_tool_id(char *dst, size_t dstlen, api_style api) {
    static uint64_t fallback_ctr;
    unsigned char bytes[16];
    const char *prefix = api == API_ANTHROPIC ? "toolu_" : "call_";
    size_t pos = snprintf(dst, dstlen, "%s", prefix);
    if (pos >= dstlen) return;

    if (!random_bytes(bytes, sizeof(bytes))) {
        uint64_t a = ((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid();
        uint64_t b = ++fallback_ctr ^ (uint64_t)(uintptr_t)dst;
        memcpy(bytes, &a, sizeof(a));
        memcpy(bytes + sizeof(a), &b, sizeof(b));
    }

    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(bytes) && pos + 2 < dstlen; i++) {
        dst[pos++] = hex[bytes[i] >> 4];
        dst[pos++] = hex[bytes[i] & 15];
    }
    dst[pos] = '\0';
}

typedef struct server server;

typedef struct {
    char *id;
    char *name;
    char *arguments;
} tool_call;

typedef struct {
    tool_call *v;
    int len;
    int cap;
    char *raw_dsml;
} tool_calls;

typedef struct {
    int mem;
    int disk;
    int canonical;
    int missing_ids;
} tool_replay_stats;

typedef struct {
    char *name;
    char *wire_name;
    char *namespace;
    /* Distinguish the Responses hosted tool from a normal function that
     * happens to be named "tool_search". */
    bool responses_tool_search;
    char **prop;
    int len;
    int cap;
} tool_schema_order;

typedef struct {
    tool_schema_order *v;
    int len;
    int cap;
} tool_schema_orders;

typedef struct {
    char *role;
    char *content;
    char *reasoning;
    char *tool_call_id;
    char **tool_call_ids;
    int tool_call_ids_len;
    int tool_call_ids_cap;
    tool_calls calls;
} chat_msg;

typedef struct {
    chat_msg *v;
    int len;
    int cap;
} chat_msgs;

static void tool_memory_attach_to_messages(server *s, chat_msgs *msgs,
                                           tool_replay_stats *stats);
static bool tool_memory_has_id(server *s, const char *id);
static void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs);

typedef struct {
    char **v;
    int len;
    int cap;
    size_t max_len;
} stop_list;

static void stop_list_clear(stop_list *stops);
static bool id_list_contains(const stop_list *ids, const char *id);
static void id_list_push_unique(stop_list *ids, const char *id);
static void id_list_free(stop_list *ids);
static bool responses_live_has_call_id(server *s, const char *id);
static bool anthropic_live_has_call_id(server *s, const char *id);

typedef struct {
    req_kind kind;
    api_style api;
    ds4_tokens prompt;
    char *model;
    stop_list stops;
    char *raw_body;
    char *prompt_text;
    tool_schema_orders tool_orders;
    int max_tokens;
    int top_k;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool stream;
    bool stream_include_usage;
    int cache_read_tokens;
    int cache_write_tokens;
    ds4_think_mode think_mode;
    bool has_tools;
    bool prompt_preserves_reasoning;
    /* For /v1/responses: emit reasoning_summary_* events / fields only when the
     * client opted in via reasoning.summary. Other APIs leave this false; the
     * field is ignored on those code paths. */
    bool reasoning_summary_emit;
    /* Responses continuation contract:
     *
     * A live Responses tool loop is not a normal "new prompt with a long
     * prefix" request.  The protocol gives tool outputs a call_id that binds
     * them to a prior assistant tool call.  If that call_id is still known in
     * memory, the live KV is the authoritative prefix, including any hidden
     * thinking that the client did not replay.  These fields carry the parsed
     * evidence needed by generate_job() to append only the new suffix.
     *
     * A tool-output-only request has no stateless prefix to match.  If the live
     * call_id binding is gone by the time the worker executes it, DS4 must ask
     * for a full replay rather than cold-prefilling a prompt that starts with a
     * naked tool result.  Similarly, if live state is gone, a reasoning-mode
     * tool replay must contain the prior reasoning item (or an equivalent
     * opaque reasoning state from a future implementation). */
    bool responses_requires_live_tool_state;
    bool responses_requires_live_reasoning;
    stop_list responses_live_call_ids;
    char *responses_live_suffix_text;
    bool anthropic_requires_live_tool_state;
    stop_list anthropic_live_call_ids;
    char *anthropic_live_suffix_text;
    tool_replay_stats tool_replay;
} request;

static void tool_call_free(tool_call *tc) {
    free(tc->id);
    free(tc->name);
    free(tc->arguments);
    memset(tc, 0, sizeof(*tc));
}

static void tool_calls_free(tool_calls *calls) {
    for (int i = 0; i < calls->len; i++) tool_call_free(&calls->v[i]);
    free(calls->raw_dsml);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}

static void tool_calls_push(tool_calls *calls, tool_call tc) {
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 4;
        calls->v = xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = tc;
}

static void chat_msg_add_tool_call_id(chat_msg *m, const char *id) {
    if (!m || !id || !id[0]) return;
    if (!m->tool_call_id) m->tool_call_id = xstrdup(id);
    for (int i = 0; i < m->tool_call_ids_len; i++) {
        if (m->tool_call_ids[i] && !strcmp(m->tool_call_ids[i], id)) return;
    }
    if (m->tool_call_ids_len == m->tool_call_ids_cap) {
        m->tool_call_ids_cap = m->tool_call_ids_cap ? m->tool_call_ids_cap * 2 : 2;
        m->tool_call_ids = xrealloc(m->tool_call_ids,
            (size_t)m->tool_call_ids_cap * sizeof(m->tool_call_ids[0]));
    }
    m->tool_call_ids[m->tool_call_ids_len++] = xstrdup(id);
}

static void chat_msg_free(chat_msg *m) {
    free(m->role);
    free(m->content);
    free(m->reasoning);
    free(m->tool_call_id);
    for (int i = 0; i < m->tool_call_ids_len; i++) free(m->tool_call_ids[i]);
    free(m->tool_call_ids);
    tool_calls_free(&m->calls);
    memset(m, 0, sizeof(*m));
}

static void chat_msgs_free(chat_msgs *msgs) {
    for (int i = 0; i < msgs->len; i++) chat_msg_free(&msgs->v[i]);
    free(msgs->v);
    memset(msgs, 0, sizeof(*msgs));
}

static void chat_msgs_push(chat_msgs *msgs, chat_msg msg) {
    if (msgs->len == msgs->cap) {
        msgs->cap = msgs->cap ? msgs->cap * 2 : 8;
        msgs->v = xrealloc(msgs->v, (size_t)msgs->cap * sizeof(msgs->v[0]));
    }
    msgs->v[msgs->len++] = msg;
}

static void tool_schema_order_free(tool_schema_order *o) {
    free(o->name);
    free(o->wire_name);
    free(o->namespace);
    for (int i = 0; i < o->len; i++) free(o->prop[i]);
    free(o->prop);
    memset(o, 0, sizeof(*o));
}

static void tool_schema_orders_free(tool_schema_orders *orders) {
    for (int i = 0; i < orders->len; i++) tool_schema_order_free(&orders->v[i]);
    free(orders->v);
    memset(orders, 0, sizeof(*orders));
}

static void tool_schema_order_prop_push(tool_schema_order *o, char *prop) {
    if (o->len == o->cap) {
        o->cap = o->cap ? o->cap * 2 : 8;
        o->prop = xrealloc(o->prop, (size_t)o->cap * sizeof(o->prop[0]));
    }
    o->prop[o->len++] = prop;
}

static int tool_schema_orders_find_index(const tool_schema_orders *orders, const char *name) {
    if (!orders || !name) return -1;
    for (int i = 0; i < orders->len; i++) {
        if (orders->v[i].name && !strcmp(orders->v[i].name, name)) return i;
    }
    return -1;
}

static void tool_schema_orders_push(tool_schema_orders *orders, tool_schema_order order) {
    int idx = tool_schema_orders_find_index(orders, order.name);
    if (idx >= 0) {
        tool_schema_order_free(&orders->v[idx]);
        orders->v[idx] = order;
        return;
    }
    if (orders->len == orders->cap) {
        orders->cap = orders->cap ? orders->cap * 2 : 8;
        orders->v = xrealloc(orders->v, (size_t)orders->cap * sizeof(orders->v[0]));
    }
    orders->v[orders->len++] = order;
}

static const tool_schema_order *tool_schema_orders_find(const tool_schema_orders *orders, const char *name) {
    int idx = tool_schema_orders_find_index(orders, name);
    return idx >= 0 ? &orders->v[idx] : NULL;
}

static void request_init(request *r, req_kind kind, int max_tokens) {
    memset(r, 0, sizeof(*r));
    r->kind = kind;
    r->api = API_OPENAI;
    r->model = xstrdup("deepseek-v4-flash");
    r->max_tokens = max_tokens;
    r->top_k = 0;
    r->temperature = DS4_DEFAULT_TEMPERATURE;
    r->top_p = DS4_DEFAULT_TOP_P;
    r->min_p = DS4_DEFAULT_MIN_P;
    r->think_mode = DS4_THINK_HIGH;
}

static void request_free(request *r) {
    ds4_tokens_free(&r->prompt);
    free(r->model);
    for (int i = 0; i < r->stops.len; i++) free(r->stops.v[i]);
    free(r->stops.v);
    free(r->raw_body);
    free(r->prompt_text);
    stop_list_clear(&r->responses_live_call_ids);
    free(r->responses_live_call_ids.v);
    free(r->responses_live_suffix_text);
    stop_list_clear(&r->anthropic_live_call_ids);
    free(r->anthropic_live_call_ids.v);
    free(r->anthropic_live_suffix_text);
    tool_schema_orders_free(&r->tool_orders);
    memset(r, 0, sizeof(*r));
}

static ds4_think_mode think_mode_from_enabled(bool enabled, ds4_think_mode effort) {
    if (!enabled || effort == DS4_THINK_NONE) return DS4_THINK_NONE;
    return effort == DS4_THINK_MAX ? DS4_THINK_MAX : DS4_THINK_HIGH;
}

static bool parse_reasoning_effort_name(const char *s, ds4_think_mode *out) {
    if (!s) return false;
    if (!strcmp(s, "max")) {
        *out = DS4_THINK_MAX;
        return true;
    }
    if (!strcmp(s, "xhigh") || !strcmp(s, "high") ||
        !strcmp(s, "medium") || !strcmp(s, "low") ||
        !strcmp(s, "minimal"))
    {
        /* DS4 only exposes HIGH and MAX above zero, so "minimal" collapses to
         * the smallest non-zero level (HIGH). Callers that need *no* reasoning
         * must use "none" instead. */
        *out = DS4_THINK_HIGH;
        return true;
    }
    if (!strcmp(s, "none")) {
        *out = DS4_THINK_NONE;
        return true;
    }
    return false;
}

static bool parse_reasoning_effort_value(const char **p, ds4_think_mode *out) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    char *effort = NULL;
    if (!json_string(p, &effort)) return false;
    bool ok = parse_reasoning_effort_name(effort, out);
    free(effort);
    return ok;
}

static bool parse_thinking_control_value(const char **p, bool *thinking_enabled) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p == 't' || **p == 'f') return json_bool(p, thinking_enabled);
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "type")) {
            char *type = NULL;
            if (!json_string(p, &type)) {
                free(key);
                return false;
            }
            if (!strcmp(type, "enabled")) *thinking_enabled = true;
            else if (!strcmp(type, "disabled")) *thinking_enabled = false;
            free(type);
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_output_config_effort(const char **p, ds4_think_mode *effort) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "effort")) {
            if (!parse_reasoning_effort_value(p, effort)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool model_alias_disables_thinking(const char *model) {
    return model && !strcmp(model, "deepseek-chat");
}

static bool model_alias_enables_thinking(const char *model) {
    return model && !strcmp(model, "deepseek-reasoner");
}

static void stop_list_clear(stop_list *stops) {
    for (int i = 0; i < stops->len; i++) free(stops->v[i]);
    stops->len = 0;
    stops->max_len = 0;
}

static void stop_list_push(stop_list *stops, char *s) {
    if (!s || !s[0]) {
        free(s);
        return;
    }
    if (stops->len == stops->cap) {
        stops->cap = stops->cap ? stops->cap * 2 : 4;
        stops->v = xrealloc(stops->v, (size_t)stops->cap * sizeof(stops->v[0]));
    }
    size_t n = strlen(s);
    if (n > stops->max_len) stops->max_len = n;
    stops->v[stops->len++] = s;
}

static bool parse_stop(const char **p, stop_list *out) {
    json_ws(p);
    stop_list_clear(out);
    if (**p == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        stop_list_push(out, s);
        return true;
    }
    if (**p != '[') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) return false;
            stop_list_push(out, s);
        } else if (!json_skip_value(p)) {
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool stop_list_find_from(const stop_list *stops, const char *text,
                                size_t from, size_t *pos, size_t *len) {
    if (!stops->len || !text) return false;
    bool found = false;
    size_t best_pos = 0, best_len = 0;
    for (int i = 0; i < stops->len; i++) {
        char *p = strstr(text + from, stops->v[i]);
        if (!p) continue;
        size_t ppos = (size_t)(p - text);
        size_t plen = strlen(stops->v[i]);
        if (!found || ppos < best_pos) {
            found = true;
            best_pos = ppos;
            best_len = plen;
        }
    }
    if (!found) return false;
    *pos = best_pos;
    *len = best_len;
    return true;
}

static size_t stop_list_stream_safe_len(const stop_list *stops, size_t text_len) {
    /* Streaming cannot emit the last max_stop_len-1 bytes yet: a stop sequence
     * may start there and finish in the next token.  The final flush releases
     * this small tail once generation ends without a stop hit. */
    if (!stops->len || stops->max_len <= 1) return text_len;
    const size_t hold = stops->max_len - 1;
    return text_len > hold ? text_len - hold : 0;
}

static int utf8_expected_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}

/* Tokenizers can split a multi-byte UTF-8 character across two tokens.  If an
 * SSE delta ends at that boundary, some clients replace the incomplete byte
 * sequence with U+FFFD and later send the corrupted text back, destroying KV
 * cache prefix matches.  Hold only the trailing incomplete character; the next
 * generated token will complete it. */
static size_t utf8_stream_safe_len(const char *s, size_t start,
                                   size_t limit, bool final) {
    if (final || !s || limit <= start) return limit;

    size_t p = limit;
    int cont = 0;
    while (p > start && cont < 4 &&
           (((unsigned char)s[p - 1] & 0xc0) == 0x80))
    {
        p--;
        cont++;
    }

    if (p == limit) {
        return utf8_expected_len((unsigned char)s[limit - 1]) > 1 ?
               limit - 1 : limit;
    }
    if (p == start && (((unsigned char)s[p] & 0xc0) == 0x80)) return start;

    size_t lead = p - 1;
    int need = utf8_expected_len((unsigned char)s[lead]);
    return (limit - lead) < (size_t)need ? lead : limit;
}

static bool parse_stream_options(const char **p, bool *include_usage) {
    json_ws(p);
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "include_usage")) {
            if (!json_bool(p, include_usage)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_function_call(const char **p, tool_call *tc) {
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') {
            free(key);
            goto bad;
        }
        (*p)++;
        if (!strcmp(key, "name")) {
            free(tc->name);
            if (!json_string(p, &tc->name)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "arguments")) {
            free(tc->arguments);
            json_ws(p);
            if (**p == '"') {
                if (!json_string(p, &tc->arguments)) {
                    free(key);
                    goto bad;
                }
            } else if (!json_raw_value(p, &tc->arguments)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;
    return true;
bad:
    return false;
}

static bool parse_tool_calls_value(const char **p, tool_calls *calls) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        tool_call tc = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto bad;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto bad;
            }
            (*p)++;
            if (!strcmp(key, "id")) {
                free(tc.id);
                if (!json_string(p, &tc.id)) {
                    free(key);
                    goto bad;
                }
            } else if (!strcmp(key, "function")) {
                if (!parse_function_call(p, &tc)) {
                    free(key);
                    goto bad;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto bad;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto bad;
        (*p)++;
        if (tc.name && tc.arguments) {
            tool_calls_push(calls, tc);
            memset(&tc, 0, sizeof(tc));
        }
        tool_call_free(&tc);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
bad:
        tool_call_free(&tc);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static void append_raw_json_line(buf *b, const char *json) {
    if (!json || !json[0]) return;
    if (b->len) buf_putc(b, '\n');
    buf_puts(b, json);
}

static void json_escape(buf *b, const char *s);

static char *openai_function_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) return NULL;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return NULL;
        }
        p++;
        if (!strcmp(key, "function")) {
            free(key);
            if (!json_raw_value(&p, &value)) return NULL;
            return value;
        }
        free(key);
        if (!json_skip_value(&p)) return NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return NULL;
}

static char *responses_special_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;

    char *type = NULL;
    char *description = NULL;
    char *parameters = NULL;
    char *out = NULL;

    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(&p, &type)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "description")) {
            free(description);
            if (!json_string(&p, &description)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "parameters")) {
            free(parameters);
            if (!json_raw_value(&p, &parameters)) {
                free(key);
                goto done;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }

    if (type && !strcmp(type, "tool_search")) {
        buf b = {0};
        buf_puts(&b, "{\"name\":\"tool_search\",\"description\":");
        json_escape(&b, description ? description : "Search available tools.");
        buf_puts(&b, ",\"parameters\":");
        buf_puts(&b, parameters ? parameters :
                 "{\"type\":\"object\",\"properties\":{}}");
        buf_putc(&b, '}');
        out = buf_take(&b);
    }

done:
    free(type);
    free(description);
    free(parameters);
    return out;
}

static char *responses_namespace_function_schema_from_tool(const char *raw,
                                                           const char *namespace,
                                                           char **wire_name) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;

    char *type = NULL;
    char *name = NULL;
    char *description = NULL;
    char *parameters = NULL;
    char *out = NULL;

    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(&p, &type)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(&p, &name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "description")) {
            free(description);
            if (!json_string(&p, &description)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "parameters") || !strcmp(key, "input_schema")) {
            free(parameters);
            if (!json_raw_value(&p, &parameters)) {
                free(key);
                goto done;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }

    if ((!type || !strcmp(type, "function")) && namespace && name && name[0]) {
        buf prompt_name = {0};
        buf_puts(&prompt_name, namespace);
        buf_puts(&prompt_name, name);

        buf b = {0};
        buf_puts(&b, "{\"name\":");
        json_escape(&b, prompt_name.ptr ? prompt_name.ptr : name);
        buf_puts(&b, ",\"description\":");
        json_escape(&b, description ? description : "");
        buf_puts(&b, ",\"parameters\":");
        buf_puts(&b, parameters ? parameters :
                 "{\"type\":\"object\",\"properties\":{}}");
        buf_putc(&b, '}');
        out = buf_take(&b);
        if (wire_name) *wire_name = xstrdup(name);
        buf_free(&prompt_name);
    }

done:
    free(type);
    free(name);
    free(description);
    free(parameters);
    return out;
}

static bool parse_schema_properties(const char *json, tool_schema_order *order) {
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return false;
        }
        p++;
        if (!strcmp(key, "properties")) {
            free(key);
            json_ws(&p);
            if (*p != '{') return false;
            p++;
            json_ws(&p);
            while (*p && *p != '}') {
                char *prop = NULL;
                if (!json_string(&p, &prop)) return false;
                json_ws(&p);
                if (*p != ':') {
                    free(prop);
                    return false;
                }
                p++;
                tool_schema_order_prop_push(order, prop);
                if (!json_skip_value(&p)) return false;
                json_ws(&p);
                if (*p == ',') p++;
                json_ws(&p);
            }
            if (*p != '}') return false;
            p++;
        } else {
            free(key);
            if (!json_skip_value(&p)) return false;
        }
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return *p == '}';
}

static void tool_schema_orders_add_json_wire(tool_schema_orders *orders,
                                             const char *json,
                                             const char *namespace,
                                             const char *wire_name,
                                             bool responses_tool_search) {
    if (!orders || !json) return;
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return;
    p++;
    tool_schema_order order = {0};
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "name")) {
            free(order.name);
            if (!json_string(&p, &order.name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "input_schema") || !strcmp(key, "parameters")) {
            char *schema = NULL;
            if (!json_raw_value(&p, &schema)) {
                free(key);
                goto done;
            }
            parse_schema_properties(schema, &order);
            free(schema);
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (order.name) {
        if (namespace && namespace[0]) order.namespace = xstrdup(namespace);
        if (wire_name && wire_name[0]) order.wire_name = xstrdup(wire_name);
        order.responses_tool_search = responses_tool_search;
        tool_schema_orders_push(orders, order);
        memset(&order, 0, sizeof(order));
    }
done:
    tool_schema_order_free(&order);
}

static void tool_schema_orders_add_json(tool_schema_orders *orders, const char *json) {
    tool_schema_orders_add_json_wire(orders, json, NULL, NULL, false);
}

static bool append_responses_namespace_tool_schemas(buf *schemas,
                                                    tool_schema_orders *orders,
                                                    const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return false;
    p++;

    char *type = NULL;
    char *name = NULL;
    char *tools = NULL;
    bool appended = false;

    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(&p, &type)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(&p, &name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "tools")) {
            free(tools);
            if (!json_raw_value(&p, &tools)) {
                free(key);
                goto done;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }

    if (!type || strcmp(type, "namespace") || !name || !tools) goto done;

    const char *tp = tools;
    json_ws(&tp);
    if (*tp != '[') goto done;
    tp++;
    json_ws(&tp);
    while (*tp && *tp != ']') {
        char *tool_raw = NULL;
        if (!json_raw_value(&tp, &tool_raw)) goto done;
        char *wire_name = NULL;
        char *schema =
            responses_namespace_function_schema_from_tool(tool_raw, name, &wire_name);
        if (schema) {
            append_raw_json_line(schemas, schema);
            tool_schema_orders_add_json_wire(orders, schema, name, wire_name, false);
            appended = true;
        }
        free(schema);
        free(wire_name);
        free(tool_raw);
        json_ws(&tp);
        if (*tp == ',') tp++;
        json_ws(&tp);
    }

done:
    free(type);
    free(name);
    free(tools);
    return appended;
}

/* OpenAI wraps tools as {"type":"function","function":{...}}. Anthropic sends
 * the function schema directly as {"name":...,"input_schema":...}. The DS4
 * prompt wants one raw function schema per line, so unwrap OpenAI tools and keep
 * already-direct schemas unchanged. Responses can additionally group tools in a
 * namespace item; those are flattened for DSML prompt rendering while preserving
 * their client-facing name and namespace for response output. */
static bool parse_tools_value(const char **p, char **out, tool_schema_orders *orders) {
    json_ws(p);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') return false;
    (*p)++;
    buf schemas = {0};

    json_ws(p);
    while (**p && **p != ']') {
        char *raw = NULL;
        if (!json_raw_value(p, &raw)) goto bad;
        char *function = openai_function_schema_from_tool(raw);
        if (function) {
            append_raw_json_line(&schemas, function);
            tool_schema_orders_add_json(orders, function);
        } else if (!append_responses_namespace_tool_schemas(&schemas, orders, raw)) {
            char *special = responses_special_schema_from_tool(raw);
            if (special) {
                append_raw_json_line(&schemas, special);
                tool_schema_orders_add_json_wire(orders, special,
                                                 NULL, NULL, true);
            } else {
                append_raw_json_line(&schemas, raw);
                tool_schema_orders_add_json(orders, raw);
            }
            free(special);
        }
        free(function);
        free(raw);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto bad;
    (*p)++;
    *out = buf_take(&schemas);
    return true;
bad:
    buf_free(&schemas);
    return false;
}

static bool parse_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto fail;
            }
            (*p)++;
            if (!strcmp(key, "role")) {
                free(msg.role);
                if (!json_string(p, &msg.role)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "content")) {
                free(msg.content);
                if (!json_content(p, &msg.content)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "reasoning_content")) {
                free(msg.reasoning);
                if (!json_content(p, &msg.reasoning)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "tool_call_id")) {
                char *id = NULL;
                if (!json_string(p, &id)) {
                    free(key);
                    goto fail;
                }
                chat_msg_add_tool_call_id(&msg, id);
                free(id);
            } else if (!strcmp(key, "tool_calls")) {
                tool_calls_free(&msg.calls);
                if (!parse_tool_calls_value(p, &msg.calls)) {
                    free(key);
                    goto fail;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto fail;
        (*p)++;
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
        chat_msgs_push(msgs, msg);
        memset(&msg, 0, sizeof(msg));
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
fail:
        chat_msg_free(&msg);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static void append_tool_result_text(buf *b, const char *s);

static bool append_anthropic_block_content(buf *dst, const char *text) {
    if (!text || !text[0]) return true;
    buf_puts(dst, text);
    return true;
}

/* Anthropic content is block-structured, while the engine consumes one compact
 * chat_msg per role.  Parsing collapses text/thinking into strings, converts
 * assistant tool_use blocks to tool_calls, and keeps tool_result blocks as
 * escaped text because DS4 sees tool results in its chat template. */
static bool parse_anthropic_content_block(const char **p, const char *role, chat_msg *msg) {
    (void)role;
    if (**p != '{') return false;
    (*p)++;
    char *type = NULL;
    char *text = NULL;
    char *thinking = NULL;
    char *id = NULL;
    char *name = NULL;
    char *input = NULL;
    char *tool_result = NULL;

    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') {
            free(key);
            goto bad;
        }
        (*p)++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(p, &type)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "text")) {
            free(text);
            if (!json_content(p, &text)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            free(thinking);
            if (!json_content(p, &thinking)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "id") || !strcmp(key, "tool_use_id")) {
            free(id);
            if (!json_string(p, &id)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(p, &name)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "input")) {
            free(input);
            if (!json_raw_value(p, &input)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "content")) {
            free(tool_result);
            if (!json_content(p, &tool_result)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;

    /* JSON object member order is not meaningful.  Some Anthropic-compatible
     * clients serialize a message as {"content": ..., "role": ...}, so the
     * caller may not know the enclosing role yet while parsing content blocks.
     * Classify protocol blocks by their own "type" field; later rendering and
     * validation use the final message role. */
    if (type && !strcmp(type, "tool_use")) {
        tool_call tc = {0};
        tc.id = id ? xstrdup(id) : NULL;
        tc.name = name ? xstrdup(name) : xstrdup("");
        tc.arguments = input ? xstrdup(input) : xstrdup("{}");
        tool_calls_push(&msg->calls, tc);
    } else if (type && !strcmp(type, "tool_result")) {
        chat_msg_add_tool_call_id(msg, id);
        buf b = {0};
        buf_puts(&b, msg->content ? msg->content : "");
        buf_puts(&b, "<tool_result>");
        append_tool_result_text(&b, tool_result);
        buf_puts(&b, "</tool_result>");
        free(msg->content);
        msg->content = buf_take(&b);
    } else {
        if (text) {
            buf b = {0};
            buf_puts(&b, msg->content ? msg->content : "");
            append_anthropic_block_content(&b, text);
            free(msg->content);
            msg->content = buf_take(&b);
        }
        if (thinking) {
            buf b = {0};
            buf_puts(&b, msg->reasoning ? msg->reasoning : "");
            append_anthropic_block_content(&b, thinking);
            free(msg->reasoning);
            msg->reasoning = buf_take(&b);
        }
    }

    free(type);
    free(text);
    free(thinking);
    free(id);
    free(name);
    free(input);
    free(tool_result);
    return true;
bad:
    free(type);
    free(text);
    free(thinking);
    free(id);
    free(name);
    free(input);
    free(tool_result);
    return false;
}

static bool parse_anthropic_content(const char **p, chat_msg *msg) {
    json_ws(p);
    if (**p == '"') return json_string(p, &msg->content);
    if (json_lit(p, "null")) {
        msg->content = xstrdup("");
        return true;
    }
    if (**p != '[') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) return false;
            buf b = {0};
            buf_puts(&b, msg->content ? msg->content : "");
            buf_puts(&b, s);
            free(msg->content);
            msg->content = buf_take(&b);
            free(s);
        } else if (**p == '{') {
            if (!parse_anthropic_content_block(p, msg->role ? msg->role : "", msg)) return false;
        } else if (!json_skip_value(p)) {
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    if (!msg->content) msg->content = xstrdup("");
    return true;
}

static bool parse_anthropic_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto fail;
            }
            (*p)++;
            if (!strcmp(key, "role")) {
                free(msg.role);
                if (!json_string(p, &msg.role)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "content")) {
                free(msg.content);
                msg.content = NULL;
                if (!parse_anthropic_content(p, &msg)) {
                    free(key);
                    goto fail;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto fail;
        (*p)++;
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
        chat_msgs_push(msgs, msg);
        memset(&msg, 0, sizeof(msg));
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
fail:
        chat_msg_free(&msg);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool anthropic_system_part_is_private(const char *s) {
    return s && !strncmp(s, "x-anthropic-", 12);
}

static void append_anthropic_system_part(buf *b, const char *s) {
    if (!s || !s[0] || anthropic_system_part_is_private(s)) return;
    if (b->len && b->ptr[b->len - 1] != '\n') buf_putc(b, '\n');
    buf_puts(b, s);
}

static bool parse_anthropic_system_object(const char **p, buf *out) {
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "text")) {
            char *text = NULL;
            if (!json_string(p, &text)) {
                free(key);
                return false;
            }
            append_anthropic_system_part(out, text);
            free(text);
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_anthropic_system(const char **p, char **out) {
    json_ws(p);
    buf b = {0};
    if (**p == '"') {
        char *text = NULL;
        if (!json_string(p, &text)) return false;
        append_anthropic_system_part(&b, text);
        free(text);
        *out = buf_take(&b);
        return true;
    }
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *text = NULL;
            if (!json_string(p, &text)) goto bad;
            append_anthropic_system_part(&b, text);
            free(text);
        } else if (**p == '{') {
            if (!parse_anthropic_system_object(p, &b)) goto bad;
        } else if (!json_skip_value(p)) {
            goto bad;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto bad;
    (*p)++;
    *out = buf_take(&b);
    return true;
bad:
    buf_free(&b);
    return false;
}

static void append_tools_prompt_text(buf *b, const char *tool_schemas) {
    if (!tool_schemas || !tool_schemas[0]) return;
    buf_puts(b,
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user question. "
        "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
        "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n\n"
        "String parameters should be specified as raw text and set `string=\"true\"`. "
        "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. "
        "Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. "
        "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n");
    buf_puts(b, tool_schemas);
    buf_puts(b, "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. "
                "Use the exact parameter names from the schemas.");
}

static void json_escape(buf *b, const char *s);

typedef struct {
    char *key;
    char *value;
    bool is_string;
    bool used;
} json_arg;

typedef struct {
    json_arg *v;
    int len;
    int cap;
} json_args;

static void json_args_free(json_args *args) {
    for (int i = 0; i < args->len; i++) {
        free(args->v[i].key);
        free(args->v[i].value);
    }
    free(args->v);
    memset(args, 0, sizeof(*args));
}

static void json_args_push(json_args *args, json_arg arg) {
    if (args->len == args->cap) {
        args->cap = args->cap ? args->cap * 2 : 8;
        args->v = xrealloc(args->v, (size_t)args->cap * sizeof(args->v[0]));
    }
    args->v[args->len++] = arg;
}

static int json_args_find_unused(json_args *args, const char *key) {
    if (!key) return -1;
    for (int i = 0; i < args->len; i++) {
        if (!args->v[i].used && args->v[i].key && !strcmp(args->v[i].key, key)) return i;
    }
    return -1;
}

static bool json_args_parse(const char *json, json_args *args) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        bool is_string = false;
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') goto bad;
        p++;
        json_ws(&p);
        if (*p == '"') {
            is_string = true;
            if (!json_string(&p, &value)) goto bad;
        } else {
            char *raw = NULL;
            if (!json_raw_value(&p, &raw)) goto bad;
            value = json_minify_raw_value(raw);
            free(raw);
        }

        json_arg arg = {.key = key, .value = value, .is_string = is_string};
        json_args_push(args, arg);
        key = value = NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
        continue;
bad:
        free(key);
        free(value);
        json_args_free(args);
        return false;
    }
    if (*p != '}') {
        json_args_free(args);
        return false;
    }
    return true;
}

static void append_dsml_attr_escaped(buf *b, const char *s) {
    for (s = s ? s : ""; *s; s++) {
        if (*s == '&') buf_puts(b, "&amp;");
        else if (*s == '<') buf_puts(b, "&lt;");
        else if (*s == '>') buf_puts(b, "&gt;");
        else if (*s == '"') buf_puts(b, "&quot;");
        else buf_putc(b, *s);
    }
}

static void append_dsml_parameter_text(buf *b, const char *s) {
    const char *end = "</｜DSML｜parameter>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "&lt;");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

static void append_tool_result_text(buf *b, const char *s) {
    /* Tool output is data.  DeepSeek's renderer keeps it as ordinary text inside
     * <tool_result>...</tool_result>, so preserving literal '<', '>' and '&' is
     * important for read-file tools and shell output.  The only delimiter we must
     * protect is the wrapper's own closing tag; otherwise a file containing that
     * exact sentinel would terminate the result early. */
    const char *end = "</tool_result>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "&lt;");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

static void append_dsml_json_literal(buf *b, const char *s) {
    const char *end = "</｜DSML｜parameter>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "\\u003c");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

static void append_dsml_arg(buf *b, const json_arg *arg) {
    buf_puts(b, "<｜DSML｜parameter name=\"");
    append_dsml_attr_escaped(b, arg->key);
    buf_puts(b, "\" string=\"");
    buf_puts(b, arg->is_string ? "true" : "false");
    buf_puts(b, "\">");
    if (arg->is_string) append_dsml_parameter_text(b, arg->value);
    else append_dsml_json_literal(b, arg->value);
    buf_puts(b, "</｜DSML｜parameter>\n");
}

static bool append_dsml_arguments_from_json(buf *b, const char *json, const tool_schema_order *order) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) return false;
    if (order) {
        for (int i = 0; i < order->len; i++) {
            int idx = json_args_find_unused(&args, order->prop[i]);
            if (idx < 0) continue;
            append_dsml_arg(b, &args.v[idx]);
            args.v[idx].used = true;
        }
    }
    for (int i = 0; i < args.len; i++) {
        if (args.v[i].used) continue;
        append_dsml_arg(b, &args.v[i]);
    }
    json_args_free(&args);
    return true;
}

static void append_json_arg_pair(buf *b, const json_arg *arg) {
    json_escape(b, arg->key);
    buf_puts(b, ":");
    if (arg->is_string) json_escape(b, arg->value);
    else buf_puts(b, arg->value);
}

static void append_json_object_or_empty(buf *b, const char *json) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) {
        buf_puts(b, "{}");
        return;
    }
    buf_putc(b, '{');
    bool wrote = false;
    for (int i = 0; i < args.len; i++) {
        if (wrote) buf_putc(b, ',');
        append_json_arg_pair(b, &args.v[i]);
        wrote = true;
    }
    buf_putc(b, '}');
    json_args_free(&args);
}

static void append_dsml_tool_calls_text(buf *b, const tool_calls *calls) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_dsml && calls->raw_dsml[0]) {
        buf_puts(b, calls->raw_dsml);
        return;
    }
    buf_puts(b, "\n\n<｜DSML｜tool_calls>\n");
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        buf_puts(b, "<｜DSML｜invoke name=\"");
        append_dsml_attr_escaped(b, tc->name);
        buf_puts(b, "\">\n");
        if (!append_dsml_arguments_from_json(b, tc->arguments, NULL)) {
            buf_puts(b, "<｜DSML｜parameter name=\"arguments\" string=\"true\">");
            append_dsml_parameter_text(b, tc->arguments);
            buf_puts(b, "</｜DSML｜parameter>\n");
        }
        buf_puts(b, "</｜DSML｜invoke>\n");
    }
    buf_puts(b, "</｜DSML｜tool_calls>");
}

static bool role_is_system(const char *role) {
    return !strcmp(role, "system") || !strcmp(role, "developer");
}

static bool role_is_user_like(const char *role) {
    return !strcmp(role, "user") || !strcmp(role, "tool") || !strcmp(role, "function");
}

static bool chat_history_uses_tool_context(const chat_msgs *msgs,
                                           const char *tool_schemas) {
    if (tool_schemas && tool_schemas[0]) return true;
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if ((!strcmp(m->role, "assistant") && m->calls.len > 0) ||
            !strcmp(m->role, "tool") || !strcmp(m->role, "function"))
        {
            return true;
        }
    }
    return false;
}

static char *render_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                     const tool_schema_orders *tool_orders,
                                     ds4_think_mode think_mode) {
    (void)tool_orders;
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool tool_context = chat_history_uses_tool_context(msgs, tool_schemas);
    int last_user_idx = -1;
    buf system = {0};
    /* Render tool schemas before the client system content so
     * --kv-cache-boundary-trim-tokens chops a dynamic tail from the client
     * message instead of the much larger tool-schema region. */
    if (tool_schemas && tool_schemas[0]) {
        append_tools_prompt_text(&system, tool_schemas);
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!role_is_system(m->role)) continue;
        if (system.len) buf_puts(&system, "\n\n");
        buf_puts(&system, m->content ? m->content : "");
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_user_like(m->role)) last_user_idx = i;
    }

    buf out = {0};
    buf_puts(&out, "<｜begin▁of▁sentence｜>");
    if (think_mode == DS4_THINK_MAX) buf_puts(&out, ds4_think_max_prefix());
    buf_puts(&out, system.ptr ? system.ptr : "");

    bool pending_assistant = false;
    bool pending_tool_result = false;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            continue;
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<｜User｜>");
            buf_puts(&out, m->content ? m->content : "");
            pending_assistant = true;
            pending_tool_result = false;
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            if (!pending_tool_result) buf_puts(&out, "<｜User｜>");
            buf_puts(&out, "<tool_result>");
            append_tool_result_text(&out, m->content);
            buf_puts(&out, "</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if (!strcmp(m->role, "assistant")) {
            if (pending_assistant) {
                buf_puts(&out, "<｜Assistant｜>");
                if (think) {
                    if (tool_context || i > last_user_idx) {
                        buf_puts(&out, "<think>");
                        buf_puts(&out, m->reasoning ? m->reasoning : "");
                        buf_puts(&out, "</think>");
                    } else {
                        buf_puts(&out, "</think>");
                    }
                } else {
                    buf_puts(&out, "</think>");
                }
            }
            buf_puts(&out, m->content ? m->content : "");
            append_dsml_tool_calls_text(&out, &m->calls);
            buf_puts(&out, "<｜end▁of▁sentence｜>");
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if (pending_assistant) {
        buf_puts(&out, "<｜Assistant｜>");
        buf_puts(&out, think ? "<think>" : "</think>");
    }

    buf_free(&system);
    return buf_take(&out);
}

/* Render only the semantic tail that must be appended to the live KV for a
 * tool-result continuation.
 *
 * In the common agent tool path, the previous assistant tool-call turn is
 * already in the model session, including hidden thinking and exact sampled
 * DSML.  The next request provides only the tool results, either as OpenAI
 * Responses tool-output items or Anthropic user content blocks.  Re-rendering
 * the assistant call here would duplicate it and destroy cache alignment, so
 * this function starts at the first new item and emits only:
 *
 *   previous EOS, tool results, and the next assistant prefix.
 *
 * This is intentionally independent from req.prompt's already-tokenized suffix:
 * suffix tokenization happens later after the cache decision, using the live
 * token prefix as the boundary.  That avoids BPE merges across the visible
 * replay/live-KV boundary. */
static char *render_live_tool_tail(const chat_msgs *msgs, int start,
                                   ds4_think_mode think_mode) {
    const bool think = ds4_think_mode_enabled(think_mode);
    buf out = {0};
    buf_puts(&out, "<｜end▁of▁sentence｜>");

    bool pending_assistant = false;
    bool pending_tool_result = false;
    for (int i = start; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            continue;
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<｜User｜>");
            buf_puts(&out, m->content ? m->content : "");
            pending_assistant = true;
            pending_tool_result = false;
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            if (!pending_tool_result) buf_puts(&out, "<｜User｜>");
            buf_puts(&out, "<tool_result>");
            append_tool_result_text(&out, m->content);
            buf_puts(&out, "</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if (!strcmp(m->role, "assistant")) {
            if (pending_assistant) {
                buf_puts(&out, "<｜Assistant｜>");
                if (think) {
                    buf_puts(&out, "<think>");
                    buf_puts(&out, m->reasoning ? m->reasoning : "");
                    buf_puts(&out, "</think>");
                } else {
                    buf_puts(&out, "</think>");
                }
            }
            buf_puts(&out, m->content ? m->content : "");
            append_dsml_tool_calls_text(&out, &m->calls);
            buf_puts(&out, "<｜end▁of▁sentence｜>");
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if (pending_assistant) {
        buf_puts(&out, "<｜Assistant｜>");
        buf_puts(&out, think ? "<think>" : "</think>");
    }
    return buf_take(&out);
}

static bool chat_msg_has_call_id(const chat_msg *m, const char *id) {
    if (!m || !id || !id[0] || strcmp(m->role, "assistant")) return false;
    for (int i = 0; i < m->calls.len; i++) {
        if (m->calls.v[i].id && !strcmp(m->calls.v[i].id, id)) return true;
    }
    return false;
}

static void chat_msg_collect_tool_call_ids(const chat_msg *m, stop_list *ids) {
    if (!m || !ids) return;
    id_list_push_unique(ids, m->tool_call_id);
    for (int i = 0; i < m->tool_call_ids_len; i++) {
        id_list_push_unique(ids, m->tool_call_ids[i]);
    }
}

static const chat_msg *responses_find_prior_call_msg(const chat_msgs *msgs,
                                                     int before,
                                                     const char *id) {
    if (!msgs || !id || !id[0]) return NULL;
    if (before > msgs->len) before = msgs->len;
    for (int i = before - 1; i >= 0; i--) {
        if (chat_msg_has_call_id(&msgs->v[i], id)) return &msgs->v[i];
    }
    return NULL;
}

/* Validate Responses tool outputs before rendering.
 *
 * A tool output with a call_id is meaningful only if either:
 *   1. DS4 still has the matching live assistant call in memory, or
 *   2. the same request replays the prior assistant call item.
 *
 * Case 1 is the fast, protocol-native continuation path: keep the live KV and
 * append only the tool result.  Case 2 is stateless replay after restart or
 * branching.  In thinking mode, case 2 is less faithful if the replay omits
 * reasoning state for the assistant call.  Official Responses clients can
 * carry that state with reasoning items / encrypted reasoning content; when
 * they do not, the request is still renderable as visible history.  Mark that
 * condition so generate_job() can prefer live / visible checkpoints and emit a
 * warning if it must fall back to visible replay instead of aborting the
 * session. */
static bool responses_validate_tool_outputs(server *s, const chat_msgs *msgs,
                                            ds4_think_mode think_mode,
                                            bool *requires_live_tool_state,
                                            bool *requires_live_reasoning,
                                            char *err, size_t errlen) {
    if (!msgs) return true;
    if (requires_live_tool_state) *requires_live_tool_state = false;
    if (requires_live_reasoning) *requires_live_reasoning = false;
    const bool needs_reasoning = ds4_think_mode_enabled(think_mode);
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (strcmp(m->role, "tool") && strcmp(m->role, "function")) continue;

        stop_list ids = {0};
        chat_msg_collect_tool_call_ids(m, &ids);
        for (int j = 0; j < ids.len; j++) {
            const char *id = ids.v[j];
            const bool live_known = responses_live_has_call_id(s, id);
            const chat_msg *prior = responses_find_prior_call_msg(msgs, i, id);
            if (!live_known && !prior) {
                snprintf(err, errlen,
                         "Responses continuation state is not available for call_id %s; retry by replaying the full input history",
                         id);
                id_list_free(&ids);
                return false;
            }
            if (!prior) {
                if (requires_live_tool_state) *requires_live_tool_state = true;
                continue;
            }
            if (needs_reasoning &&
                (!prior->reasoning || !prior->reasoning[0]))
            {
                if (requires_live_reasoning) *requires_live_reasoning = true;
            }
        }
        id_list_free(&ids);
    }
    return true;
}

/* Record the call ids and suffix candidate for a live Responses continuation.
 *
 * This only prepares evidence.  generate_job() later checks that the live
 * server state is still exactly at the remembered token frontier before using
 * it.  If another request already replaced the session, normal token/text/disk
 * prefix matching handles the request instead. */
static void responses_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs) {
    if (!r || r->api != API_RESPONSES || !msgs || msgs->len == 0) return;

    int tail_start = msgs->len;
    while (tail_start > 0) {
        const chat_msg *m = &msgs->v[tail_start - 1];
        if (strcmp(m->role, "tool") && strcmp(m->role, "function")) break;
        tail_start--;
    }
    if (tail_start == msgs->len) return;

    stop_list_clear(&r->responses_live_call_ids);
    if (tail_start > 0) {
        const int anchor = tail_start - 1;
        const chat_msg *assistant = &msgs->v[anchor];
        if (strcmp(assistant->role, "assistant") || assistant->calls.len == 0) return;
        for (int i = 0; i < assistant->calls.len; i++) {
            id_list_push_unique(&r->responses_live_call_ids, assistant->calls.v[i].id);
        }
    } else {
        for (int i = tail_start; i < msgs->len; i++) {
            chat_msg_collect_tool_call_ids(&msgs->v[i], &r->responses_live_call_ids);
        }
    }
    if (r->responses_live_call_ids.len == 0) return;

    free(r->responses_live_suffix_text);
    r->responses_live_suffix_text =
        render_live_tool_tail(msgs, tail_start, r->think_mode);
}

static bool anthropic_msg_is_tool_result_tail(const chat_msg *m) {
    return m && !strcmp(m->role, "user") &&
           ((m->tool_call_id && m->tool_call_id[0]) ||
            m->tool_call_ids_len > 0);
}

/* Validate Anthropic tool results before rendering.
 *
 * A tool_result.tool_use_id is valid if it is either still bound to the live
 * Anthropic assistant tool-call frontier or the same request replays the prior
 * assistant tool_use block.  The first case is the fast path: keep the sampled
 * KV and append only the tool-result suffix.  The second case is a normal
 * stateless replay, where exact DSML tool memory can restore the sampled tool
 * bytes before prefix matching.  A tool-result-only request with an unknown
 * live id has no safe prefix to reconstruct, so report a clear client error. */
static bool anthropic_validate_tool_results(server *s, const chat_msgs *msgs,
                                            bool *requires_live_tool_state,
                                            char *err, size_t errlen) {
    if (requires_live_tool_state) *requires_live_tool_state = false;
    if (!msgs) return true;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!anthropic_msg_is_tool_result_tail(m)) continue;

        stop_list ids = {0};
        chat_msg_collect_tool_call_ids(m, &ids);
        for (int j = 0; j < ids.len; j++) {
            const char *id = ids.v[j];
            const bool live_known = anthropic_live_has_call_id(s, id);
            const chat_msg *prior = responses_find_prior_call_msg(msgs, i, id);
            if (!live_known && !prior) {
                snprintf(err, errlen,
                         "Anthropic continuation state is not available for tool_use_id %s; retry by replaying the full messages history",
                         id);
                id_list_free(&ids);
                return false;
            }
            if (!prior && requires_live_tool_state) {
                *requires_live_tool_state = true;
            }
        }
        id_list_free(&ids);
    }
    return true;
}

/* Prepare the Anthropic live-tool fast path.
 *
 * Anthropic's visible replay normally includes the assistant tool_use JSON and
 * the user tool_result.  That replay is still only a description of what the
 * model sampled.  If the incoming tool_result IDs match the live sampled
 * frontier, generate_job() can skip replay matching entirely and append just
 * EOS + tool_result + next assistant prefix to the real KV. */
static void anthropic_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs) {
    if (!r || r->api != API_ANTHROPIC || !msgs || msgs->len == 0) return;

    int tail_end = msgs->len;
    while (tail_end > 0 && role_is_system(msgs->v[tail_end - 1].role)) tail_end--;
    int tail_start = tail_end;
    while (tail_start > 0 &&
           anthropic_msg_is_tool_result_tail(&msgs->v[tail_start - 1]))
    {
        tail_start--;
    }
    if (tail_start == tail_end) return;

    stop_list_clear(&r->anthropic_live_call_ids);
    for (int i = tail_start; i < msgs->len; i++) {
        chat_msg_collect_tool_call_ids(&msgs->v[i], &r->anthropic_live_call_ids);
    }
    if (r->anthropic_live_call_ids.len == 0) return;

    free(r->anthropic_live_suffix_text);
    r->anthropic_live_suffix_text =
        render_live_tool_tail(msgs, tail_start, r->think_mode);
}

/* The API parsers are intentionally selective JSON parsers: they keep only
 * fields that affect model semantics, rendering, streaming, or cache keys, and
 * skip extension fields.  The output is always a rendered DS4 chat/completion
 * prompt plus the small amount of protocol state needed to translate the reply. */
static bool parse_chat_request(ds4_engine *e, server *s, const char *body, int def_tokens,
                               int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    const char *p = body;
    bool got_messages = false;
    bool tool_choice_none = false;
    bool got_thinking = false;
    bool thinking_enabled = true;
    ds4_think_mode reasoning_effort = DS4_THINK_HIGH;
    chat_msgs msgs = {0};
    char *tool_schemas = NULL;

    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        if (!strcmp(key, "messages")) {
            chat_msgs_free(&msgs);
            if (!parse_messages(&p, &msgs)) {
                free(key);
                goto bad;
            }
            got_messages = true;
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            if (!parse_tools_value(&p, &tool_schemas, &r->tool_orders)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tool_choice")) {
            json_ws(&p);
            if (*p == '"') {
                char *choice = NULL;
                if (!json_string(&p, &choice)) {
                    free(key);
                    goto bad;
                }
                tool_choice_none = !strcmp(choice, "none");
                free(choice);
            } else if (!json_skip_value(&p)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            free(r->model);
            if (!json_string(&p, &r->model)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "max_tokens") || !strcmp(key, "max_completion_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "temperature")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->temperature = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
        } else if (!strcmp(key, "min_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->min_p = (float)v;
        } else if (!strcmp(key, "top_k")) {
            if (!json_int(&p, &r->top_k)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "seed")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->seed = v > 0.0 ? (uint64_t)v : 0;
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream_options")) {
            if (!parse_stream_options(&p, &r->stream_include_usage)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            if (!parse_thinking_control_value(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "think")) {
            if (!json_bool(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "stop")) {
            if (!parse_stop(&p, &r->stops)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_messages) {
        snprintf(err, errlen, "missing messages");
        chat_msgs_free(&msgs);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    r->has_tools = tool_schemas && tool_schemas[0] && !tool_choice_none;
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    r->think_mode = ds4_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort), ctx_size);
    kv_cache_restore_tool_memory_for_messages(s, &msgs);
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    const char *active_tool_schemas = r->has_tools ? tool_schemas : NULL;
    r->prompt_preserves_reasoning =
        chat_history_uses_tool_context(&msgs, active_tool_schemas);
    r->prompt_text = render_chat_prompt_text(&msgs, active_tool_schemas,
                                             &r->tool_orders, r->think_mode);
    ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
    chat_msgs_free(&msgs);
    free(tool_schemas);
    return true;
bad:
    chat_msgs_free(&msgs);
    free(tool_schemas);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

static bool parse_anthropic_request(ds4_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    r->api = API_ANTHROPIC;
    const char *p = body;
    bool got_messages = false;
    bool tool_choice_none = false;
    bool got_thinking = false;
    bool thinking_enabled = true;
    ds4_think_mode reasoning_effort = DS4_THINK_HIGH;
    chat_msgs msgs = {0};
    char *system = NULL;
    char *tool_schemas = NULL;

    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        if (!strcmp(key, "messages")) {
            chat_msgs_free(&msgs);
            if (!parse_anthropic_messages(&p, &msgs)) {
                free(key);
                goto bad;
            }
            got_messages = true;
        } else if (!strcmp(key, "system")) {
            free(system);
            if (!parse_anthropic_system(&p, &system)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            if (!parse_tools_value(&p, &tool_schemas, &r->tool_orders)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tool_choice")) {
            json_ws(&p);
            if (*p == '{') {
                p++;
                json_ws(&p);
                while (*p && *p != '}') {
                    char *ckey = NULL;
                    if (!json_string(&p, &ckey)) {
                        free(key);
                        goto bad;
                    }
                    json_ws(&p);
                    if (*p != ':') {
                        free(ckey);
                        free(key);
                        goto bad;
                    }
                    p++;
                    if (!strcmp(ckey, "type")) {
                        char *choice = NULL;
                        if (!json_string(&p, &choice)) {
                            free(ckey);
                            free(key);
                            goto bad;
                        }
                        tool_choice_none = !strcmp(choice, "none");
                        free(choice);
                    } else if (!json_skip_value(&p)) {
                        free(ckey);
                        free(key);
                        goto bad;
                    }
                    free(ckey);
                    json_ws(&p);
                    if (*p == ',') p++;
                    json_ws(&p);
                }
                if (*p != '}') {
                    free(key);
                    goto bad;
                }
                p++;
            } else if (!json_skip_value(&p)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            free(r->model);
            if (!json_string(&p, &r->model)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "max_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "temperature")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->temperature = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
        } else if (!strcmp(key, "top_k")) {
            if (!json_int(&p, &r->top_k)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stop_sequences")) {
            if (!parse_stop(&p, &r->stops)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            if (!parse_thinking_control_value(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "output_config")) {
            if (!parse_output_config_effort(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_messages) {
        snprintf(err, errlen, "missing messages");
        chat_msgs_free(&msgs);
        free(system);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    if (system && system[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = system;
        system = NULL;
        chat_msgs_push(&msgs, msg);
    }
    r->has_tools = tool_schemas && tool_schemas[0] && !tool_choice_none;
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    r->think_mode = ds4_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort), ctx_size);
    if (!anthropic_validate_tool_results(s, &msgs,
                                         &r->anthropic_requires_live_tool_state,
                                         err, errlen))
    {
        chat_msgs_free(&msgs);
        free(system);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    kv_cache_restore_tool_memory_for_messages(s, &msgs);
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    anthropic_prepare_live_continuation(r, &msgs);
    const char *active_tool_schemas = r->has_tools ? tool_schemas : NULL;
    r->prompt_preserves_reasoning =
        chat_history_uses_tool_context(&msgs, active_tool_schemas);
    r->prompt_text = render_chat_prompt_text(&msgs, active_tool_schemas,
                                             &r->tool_orders, r->think_mode);
    ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
    chat_msgs_free(&msgs);
    free(system);
    free(tool_schemas);
    return true;
bad:
    chat_msgs_free(&msgs);
    free(system);
    free(tool_schemas);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

/* Responses API: convert a content-array item (input_text/output_text/text) into a
 * concatenated string. Strict shape check: bare string, null, or an array of
 * recognized text blocks. Numbers / objects / arrays-of-primitives at the top
 * level all reject so the client sees a 400 instead of an answer built on
 * silently dropped context. */
static bool parse_responses_content_array(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        return false;
    }
    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) goto fail;
            buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            char *type = NULL;
            char *text = NULL;
            json_ws(p);
            while (**p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) {
                    free(type);
                    free(text);
                    goto fail;
                }
                json_ws(p);
                if (**p != ':') {
                    free(key);
                    free(type);
                    free(text);
                    goto fail;
                }
                (*p)++;
                if (!strcmp(key, "type")) {
                    free(type);
                    if (!json_string(p, &type)) {
                        free(key);
                        free(text);
                        goto fail;
                    }
                } else if (!strcmp(key, "text")) {
                    free(text);
                    /* The text field of a typed content block is a plain JSON
                     * string. Accept null as the empty string for parity with
                     * upstream serializers that emit null for empty blocks. */
                    json_ws(p);
                    if (json_lit(p, "null")) {
                        text = xstrdup("");
                    } else if (!json_string(p, &text)) {
                        free(key);
                        free(type);
                        goto fail;
                    }
                } else if (!json_skip_value(p)) {
                    free(key);
                    free(type);
                    free(text);
                    goto fail;
                }
                free(key);
                json_ws(p);
                if (**p == ',') (*p)++;
                json_ws(p);
            }
            if (**p != '}') {
                free(type);
                free(text);
                goto fail;
            }
            (*p)++;
            /* Fail closed: a content object must carry a known text-like type
             * AND a text field. Anything else — missing type, missing text,
             * image/file/audio types, future schema-drift — is rejected so the
             * client gets a 400 instead of an answer built on context the
             * server discarded silently. */
            bool is_text_block = type && (
                !strcmp(type, "input_text") ||
                !strcmp(type, "output_text") ||
                !strcmp(type, "text") ||
                !strcmp(type, "summary_text") ||
                !strcmp(type, "reasoning_text"));
            if (!is_text_block || !text) {
                free(type);
                free(text);
                goto fail;
            }
            buf_puts(&b, text);
            free(type);
            free(text);
        } else {
            /* Reject primitives, arrays-of-arrays, nulls: a content array
             * element must be either a string or a typed text object. */
            goto fail;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

/* Codex /v1/responses input items have a `type` discriminator (message,
 * function_call, function_call_output, reasoning, custom_tool_call,
 * custom_tool_call_output, ...). We collapse them into chat_msgs the same way
 * the chat completion / Anthropic parsers do, so the rest of the engine sees a
 * single conversation history shape.
 *
 * Protocol contract for stateless replay:
 *   - The client must replay response.output items before tool outputs.
 *   - For reasoning models, the replay must also include reasoning state.  DS4
 *     can render plain reasoning summaries/content, but it cannot decrypt
 *     reasoning.encrypted_content.  If live state is unavailable and the replay
 *     only contains visible messages/tool calls, later validation marks it as a
 *     lower-fidelity replay; generate_job() logs that and continues from the
 *     visible transcript rather than killing a recoverable agent session.
 *
 * Reasoning items are merged into the next assistant message so
 * render_chat_prompt_text can wrap them in <think>. */
static bool parse_responses_input(const char **p, chat_msgs *msgs,
                                  buf *loaded_tool_schemas,
                                  tool_schema_orders *orders) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    buf pending_reasoning = {0};

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') goto fail;
        (*p)++;
        char *type = NULL;
        char *role = NULL;
        char *content = NULL;
        char *name = NULL;
        char *namespace = NULL;
        char *call_id = NULL;
        char *item_id = NULL;
        char *arguments = NULL;
        char *output = NULL;
        char *input_str = NULL;
        char *summary = NULL;
        char *action = NULL;
        char *result = NULL;
        char *tools_json = NULL;
        char *status_str = NULL;
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto item_fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto item_fail;
            }
            (*p)++;
            if (!strcmp(key, "type")) {
                free(type);
                if (!json_string(p, &type)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "role")) {
                free(role);
                if (!json_string(p, &role)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "content")) {
                free(content);
                if (!parse_responses_content_array(p, &content)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "name")) {
                free(name);
                if (!json_string(p, &name)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "namespace")) {
                free(namespace);
                if (!json_string(p, &namespace)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "call_id")) {
                free(call_id);
                if (!json_string(p, &call_id)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "id")) {
                free(item_id);
                if (!json_string(p, &item_id)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "arguments")) {
                free(arguments);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &arguments)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &arguments)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "output")) {
                free(output);
                json_ws(p);
                if (**p == '[') {
                    if (!parse_responses_content_array(p, &output)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (**p == '"') {
                    if (!json_string(p, &output)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &output)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "input")) {
                free(input_str);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &input_str)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &input_str)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "summary")) {
                free(summary);
                if (!parse_responses_content_array(p, &summary)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "action")) {
                free(action);
                if (!json_raw_value(p, &action)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "result")) {
                free(result);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &result)) {
                        free(key);
                        goto item_fail;
                    }
                } else if (!json_raw_value(p, &result)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "status")) {
                free(status_str);
                if (!json_string(p, &status_str)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!strcmp(key, "tools")) {
                /* tool_search_output items carry their discovered tool list
                 * here instead of in `output` / `result`. Keep it separate
                 * from the human-visible result body so malformed tool lists
                 * never get mistaken for normal tool output. */
                free(tools_json);
                if (!json_raw_value(p, &tools_json)) {
                    free(key);
                    goto item_fail;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto item_fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
            continue;
item_fail:
            free(type);
            free(role);
            free(content);
            free(name);
            free(namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            buf_free(&pending_reasoning);
            return false;
        }
        if (**p != '}') {
            free(type);
            free(role);
            free(content);
            free(name);
            free(namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            goto fail;
        }
        (*p)++;

        const char *t = type ? type : "message";
        /* Replayed items must be in a terminal "completed" state. in_progress,
         * incomplete, and failed all represent partial model state the client
         * never confirmed — feeding them back as history would let DS4 continue
         * from a tool action that never finished. Reject explicitly. */
        if (status_str && status_str[0] &&
            strcmp(status_str, "completed") != 0)
        {
            free(type);
            free(role);
            free(content);
            free(name);
            free(namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            buf_free(&pending_reasoning);
            return false;
        }
        /* Three classes of items:
         *   1. consumes_reasoning: assistant message / function_call / hosted-tool
         *      call. Attaches pending reasoning to its own assistant message.
         *   2. is_bookkeeping: compaction / context_compaction etc. Semantically
         *      transparent — passes through without touching pending_reasoning.
         *   3. everything else (user message, tool output): forces pending
         *      reasoning to flush in-position as an empty assistant message so it
         *      stays before this item in the rendered history. */
        bool consumes_reasoning =
            (!strcmp(t, "message") && role && !strcmp(role, "assistant")) ||
            !strcmp(t, "function_call") || !strcmp(t, "custom_tool_call") ||
            !strcmp(t, "local_shell_call") || !strcmp(t, "web_search_call") ||
            !strcmp(t, "tool_search_call") || !strcmp(t, "image_generation_call");
        bool is_bookkeeping =
            !strcmp(t, "compaction") || !strcmp(t, "context_compaction");
        if (!consumes_reasoning && !is_bookkeeping && pending_reasoning.len) {
            chat_msg flush_msg = {0};
            flush_msg.role = xstrdup("assistant");
            flush_msg.content = xstrdup("");
            flush_msg.reasoning = buf_take(&pending_reasoning);
            chat_msgs_push(msgs, flush_msg);
        }
        if (!strcmp(t, "message")) {
            chat_msg msg = {0};
            msg.role = xstrdup(role ? role : "user");
            msg.content = content ? content : xstrdup("");
            content = NULL;
            if (!strcmp(msg.role, "assistant") && pending_reasoning.len) {
                msg.reasoning = buf_take(&pending_reasoning);
            }
            chat_msgs_push(msgs, msg);
        } else if (!strcmp(t, "function_call") || !strcmp(t, "custom_tool_call")) {
            tool_call tc = {0};
            tc.id = xstrdup(call_id ? call_id : item_id ? item_id : "");
            /* function_call uses `arguments` (JSON string); custom_tool_call uses
             * `input` (free text). Treat both as the same on-wire argument blob —
             * append_dsml_arguments_from_json will fall back to a single text param
             * if the value isn't a JSON object. */
            const char *args_src = arguments ? arguments :
                                   input_str ? input_str : "{}";
            tc.arguments = xstrdup(args_src);
            if (strcmp(t, "custom_tool_call") && namespace && namespace[0] &&
                name && name[0])
            {
                buf qualified = {0};
                buf_puts(&qualified, namespace);
                buf_puts(&qualified, name);
                tc.name = buf_take(&qualified);
            } else {
                tc.name = xstrdup(name ? name : "");
            }
            /* A Responses turn that has both message text and tool calls splits
             * them across separate output items; the chat template renders the
             * second assistant record without an `<|Assistant|>` prefix, leaving
             * the tool call bare. Merge into the previous assistant message
             * when nothing user-like / tool-output-like came between them. */
            chat_msg *last = msgs->len ? &msgs->v[msgs->len - 1] : NULL;
            if (last && !strcmp(last->role, "assistant")) {
                if (pending_reasoning.len && (!last->reasoning || !last->reasoning[0])) {
                    free(last->reasoning);
                    last->reasoning = buf_take(&pending_reasoning);
                }
                tool_calls_push(&last->calls, tc);
            } else {
                chat_msg msg = {0};
                msg.role = xstrdup("assistant");
                msg.content = xstrdup("");
                if (pending_reasoning.len) msg.reasoning = buf_take(&pending_reasoning);
                tool_calls_push(&msg.calls, tc);
                chat_msgs_push(msgs, msg);
            }
        } else if (!strcmp(t, "function_call_output") || !strcmp(t, "custom_tool_call_output")) {
            chat_msg msg = {0};
            msg.role = xstrdup("tool");
            msg.content = output ? output : xstrdup("");
            output = NULL;
            if (call_id || item_id) {
                chat_msg_add_tool_call_id(&msg, call_id ? call_id : item_id);
            }
            chat_msgs_push(msgs, msg);
        } else if (!strcmp(t, "reasoning")) {
            /* Stash so it merges into the next assistant message. summary is the
             * short-form list, content is the verbose chain. Either can be empty. */
            if (summary && summary[0]) {
                if (pending_reasoning.len) buf_putc(&pending_reasoning, '\n');
                buf_puts(&pending_reasoning, summary);
            }
            if (content && content[0]) {
                if (pending_reasoning.len) buf_putc(&pending_reasoning, '\n');
                buf_puts(&pending_reasoning, content);
            }
        } else if (!strcmp(t, "local_shell_call") || !strcmp(t, "web_search_call") ||
                   !strcmp(t, "tool_search_call") || !strcmp(t, "image_generation_call"))
        {
            /* Hosted-tool history isn't natively supported (DS4 doesn't register
             * these tools), but a Codex client may still replay them when the
             * model used them in a prior turn. Surface them as function_call
             * shaped history so the next prompt retains the action that ran. */
            tool_call tc = {0};
            tc.id = xstrdup(call_id ? call_id : item_id ? item_id : "");
            if (!strcmp(t, "tool_search_call")) {
                tc.name = xstrdup("tool_search");
            } else if (!strcmp(t, "local_shell_call")) {
                tc.name = xstrdup("local_shell");
            } else {
                tc.name = xstrdup(t);
            }
            const char *args_src = action ? action :
                                   arguments ? arguments :
                                   input_str ? input_str : "{}";
            tc.arguments = xstrdup(args_src);
            chat_msg *last = msgs->len ? &msgs->v[msgs->len - 1] : NULL;
            if (last && !strcmp(last->role, "assistant")) {
                if (pending_reasoning.len && (!last->reasoning || !last->reasoning[0])) {
                    free(last->reasoning);
                    last->reasoning = buf_take(&pending_reasoning);
                }
                tool_calls_push(&last->calls, tc);
            } else {
                chat_msg msg = {0};
                msg.role = xstrdup("assistant");
                msg.content = xstrdup("");
                if (pending_reasoning.len) msg.reasoning = buf_take(&pending_reasoning);
                tool_calls_push(&msg.calls, tc);
                chat_msgs_push(msgs, msg);
            }
        } else if (!strcmp(t, "local_shell_call_output") ||
                   !strcmp(t, "web_search_call_output") ||
                   !strcmp(t, "tool_search_output") ||
                   !strcmp(t, "tool_search_call_output") ||
                   !strcmp(t, "image_generation_call_output"))
        {
            if (!strcmp(t, "tool_search_output") && tools_json &&
                loaded_tool_schemas && orders)
            {
                const char *tools_p = tools_json;
                char *schemas = NULL;
                if (!parse_tools_value(&tools_p, &schemas, orders)) {
                    free(schemas);
                    free(type);
                    free(role);
                    free(content);
                    free(name);
                    free(namespace);
                    free(call_id);
                    free(item_id);
                    free(arguments);
                    free(output);
                    free(input_str);
                    free(summary);
                    free(action);
                    free(result);
                    free(tools_json);
                    free(status_str);
                    buf_free(&pending_reasoning);
                    return false;
                }
                if (schemas && schemas[0]) {
                    if (loaded_tool_schemas->len) buf_putc(loaded_tool_schemas, '\n');
                    buf_puts(loaded_tool_schemas, schemas);
                }
                free(schemas);
            }
            chat_msg msg = {0};
            msg.role = xstrdup("tool");
            const char *body = output ? output :
                               result ? result :
                               tools_json ? tools_json : "";
            msg.content = xstrdup(body);
            if (call_id || item_id) {
                chat_msg_add_tool_call_id(&msg, call_id ? call_id : item_id);
            }
            chat_msgs_push(msgs, msg);
        } else if (!is_bookkeeping) {
            /* Anything we don't have an explicit branch for would silently
             * drop replay context. Fail the parse instead so the client sees
             * the limitation rather than ending up with stale generation
             * built on an incomplete history. Only compaction/context_compaction
             * (true Codex bookkeeping) are allowed to pass through silently. */
            free(type);
            free(role);
            free(content);
            free(name);
            free(namespace);
            free(call_id);
            free(item_id);
            free(arguments);
            free(output);
            free(input_str);
            free(summary);
            free(action);
            free(result);
            free(tools_json);
            free(status_str);
            buf_free(&pending_reasoning);
            return false;
        }

        free(type);
        free(role);
        free(content);
        free(name);
        free(namespace);
        free(call_id);
        free(item_id);
        free(arguments);
        free(output);
        free(input_str);
        free(summary);
        free(action);
        free(result);
        free(tools_json);
        free(status_str);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto fail;
    (*p)++;
    /* Trailing reasoning with no following message/tool item: attach it to an
     * empty assistant message so the next turn still renders a <think>...</think>
     * block. Dropping it loses model state when a previous response ended with
     * a reasoning-only incomplete turn and the client replays the history. */
    if (pending_reasoning.len) {
        chat_msg msg = {0};
        msg.role = xstrdup("assistant");
        msg.content = xstrdup("");
        msg.reasoning = buf_take(&pending_reasoning);
        chat_msgs_push(msgs, msg);
    }
    buf_free(&pending_reasoning);
    return true;
fail:
    buf_free(&pending_reasoning);
    return false;
}

/* Responses API has `reasoning: {"effort": "...", "summary": "..."}`. effort
 * controls thinking depth; summary mode (auto/concise/detailed) controls
 * whether the wire emits summary deltas at all — per the spec, no reasoning
 * summary is surfaced unless the client opts in. */
static bool parse_responses_reasoning(const char **p, ds4_think_mode *effort,
                                      bool *summary_opted_in,
                                      bool *effort_seen) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "effort")) {
            json_ws(p);
            /* A `null` effort doesn't change thinking_enabled — it's the same
             * as omitting the field. Only treat the field as a control if it
             * carried an actual value. */
            if (json_lit(p, "null")) {
                /* nothing */
            } else {
                if (!parse_reasoning_effort_value(p, effort)) {
                    free(key);
                    return false;
                }
                if (effort_seen) *effort_seen = true;
            }
        } else if (!strcmp(key, "summary")) {
            json_ws(p);
            if (json_lit(p, "null")) {
                /* explicit null disables summary */
            } else if (**p == '"') {
                char *mode = NULL;
                if (!json_string(p, &mode)) {
                    free(key);
                    return false;
                }
                if (summary_opted_in &&
                    (!strcmp(mode, "auto") ||
                     !strcmp(mode, "concise") ||
                     !strcmp(mode, "detailed")))
                {
                    *summary_opted_in = true;
                }
                free(mode);
            } else if (!json_skip_value(p)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_responses_request(ds4_engine *e, server *s, const char *body, int def_tokens,
                                    int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    r->api = API_RESPONSES;
    const char *p = body;
    bool got_input = false;
    bool tool_choice_none = false;
    bool got_thinking = false;
    bool thinking_enabled = true;
    ds4_think_mode reasoning_effort = DS4_THINK_HIGH;
    chat_msgs msgs = {0};
    buf loaded_tool_schemas = {0};
    char *instructions = NULL;
    char *tool_schemas = NULL;

    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        if (!strcmp(key, "input")) {
            chat_msgs_free(&msgs);
            json_ws(&p);
            /* Codex CLI always sends `input` as an array; tolerate bare strings
             * for parity with other Responses-API callers. */
            if (*p == '"') {
                char *plain = NULL;
                if (!json_string(&p, &plain)) {
                    free(key);
                    goto bad;
                }
                chat_msg msg = {0};
                msg.role = xstrdup("user");
                msg.content = plain;
                chat_msgs_push(&msgs, msg);
            } else if (!parse_responses_input(&p, &msgs, &loaded_tool_schemas,
                                              &r->tool_orders)) {
                free(key);
                goto bad;
            }
            got_input = true;
        } else if (!strcmp(key, "instructions")) {
            free(instructions);
            instructions = NULL;
            json_ws(&p);
            if (json_lit(&p, "null")) {
                instructions = xstrdup("");
            } else if (!json_string(&p, &instructions)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            if (!parse_tools_value(&p, &tool_schemas, &r->tool_orders)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "tool_choice")) {
            json_ws(&p);
            if (*p == '"') {
                char *choice = NULL;
                if (!json_string(&p, &choice)) {
                    free(key);
                    goto bad;
                }
                /* DS4 honours "none" (disable tools) and "auto" (model decides).
                 * "required" and explicit function targets need constrained
                 * decoding we don't implement — reject so clients see the
                 * limitation instead of silently downgrading to auto. */
                if (!strcmp(choice, "none")) {
                    tool_choice_none = true;
                } else if (strcmp(choice, "auto") != 0) {
                    snprintf(err, errlen, "tool_choice=%s not supported", choice);
                    free(choice);
                    free(key);
                    chat_msgs_free(&msgs);
                    buf_free(&loaded_tool_schemas);
                    free(instructions);
                    free(tool_schemas);
                    request_free(r);
                    return false;
                }
                free(choice);
            } else if (*p == '{') {
                snprintf(err, errlen, "forced tool_choice not supported");
                free(key);
                chat_msgs_free(&msgs);
                buf_free(&loaded_tool_schemas);
                free(instructions);
                free(tool_schemas);
                request_free(r);
                return false;
            } else if (!json_skip_value(&p)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            free(r->model);
            if (!json_string(&p, &r->model)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "max_output_tokens") || !strcmp(key, "max_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "temperature")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->temperature = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "reasoning")) {
            bool effort_seen = false;
            if (!parse_responses_reasoning(&p, &reasoning_effort,
                                           &r->reasoning_summary_emit,
                                           &effort_seen)) {
                free(key);
                goto bad;
            }
            /* Only an explicit effort value counts as the client opting into
             * thinking control. summary alone, or `reasoning: null`, leaves the
             * default behaviour (and the model_alias_* fallbacks below) intact. */
            if (effort_seen) {
                got_thinking = true;
                /* Responses-API effort of "minimal" / "none" maps to disabled
                 * thinking. Other effort values choose between HIGH and MAX. */
                if (reasoning_effort == DS4_THINK_NONE) thinking_enabled = false;
            }
        } else if (!strcmp(key, "previous_response_id") ||
                   !strcmp(key, "conversation"))
        {
            /* Official Responses state can be durable:
             *   previous_response_id chains to a stored prior response, and
             *   conversation points at a persistent Conversations object.
             *
             * DS4 does not yet implement that durable store.  The supported
             * modes are either (a) a live in-memory continuation checked by
             * visible transcript / tool call ids, or (b) stateless replay of
             * the full input items.  Accepting a non-null durable reference
             * without loading the referenced items would silently truncate the
             * prompt, so reject it explicitly. */
            json_ws(&p);
            if (!json_lit(&p, "null")) {
                snprintf(err, errlen,
                         "%s is not supported; replay full input instead",
                         key);
                free(key);
                chat_msgs_free(&msgs);
                buf_free(&loaded_tool_schemas);
                free(instructions);
                free(tool_schemas);
                request_free(r);
                return false;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_input) {
        snprintf(err, errlen, "missing input");
        chat_msgs_free(&msgs);
        buf_free(&loaded_tool_schemas);
        free(instructions);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    /* instructions in the Responses API replaces any system message — for Codex
     * it carries the full agent system prompt. Prepend it so render produces a
     * standard system+chat layout. */
    if (instructions && instructions[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = instructions;
        instructions = NULL;
        /* Insert at the head so it precedes the conversation. */
        chat_msgs_push(&msgs, msg);
        if (msgs.len > 1) {
            chat_msg tmp = msgs.v[msgs.len - 1];
            for (int i = msgs.len - 1; i > 0; i--) msgs.v[i] = msgs.v[i - 1];
            msgs.v[0] = tmp;
        }
    }
    buf combined_tool_schemas = {0};
    if (tool_schemas && tool_schemas[0]) buf_puts(&combined_tool_schemas, tool_schemas);
    if (loaded_tool_schemas.len) {
        if (combined_tool_schemas.len) buf_putc(&combined_tool_schemas, '\n');
        buf_append(&combined_tool_schemas, loaded_tool_schemas.ptr,
                   loaded_tool_schemas.len);
    }
    const char *active_tool_schemas =
        (!tool_choice_none && combined_tool_schemas.len) ?
        combined_tool_schemas.ptr : NULL;
    r->has_tools = active_tool_schemas && active_tool_schemas[0];
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    r->think_mode = ds4_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort), ctx_size);
    if (!responses_validate_tool_outputs(s, &msgs, r->think_mode,
                                         &r->responses_requires_live_tool_state,
                                         &r->responses_requires_live_reasoning,
                                         err, errlen)) {
        chat_msgs_free(&msgs);
        buf_free(&combined_tool_schemas);
        buf_free(&loaded_tool_schemas);
        free(instructions);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    kv_cache_restore_tool_memory_for_messages(s, &msgs);
    tool_memory_attach_to_messages(s, &msgs, &r->tool_replay);
    r->prompt_preserves_reasoning =
        chat_history_uses_tool_context(&msgs, active_tool_schemas);
    responses_prepare_live_continuation(r, &msgs);
    r->prompt_text = render_chat_prompt_text(&msgs, active_tool_schemas,
                                             &r->tool_orders, r->think_mode);
    ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
    chat_msgs_free(&msgs);
    buf_free(&combined_tool_schemas);
    buf_free(&loaded_tool_schemas);
    free(instructions);
    free(tool_schemas);
    return true;
bad:
    chat_msgs_free(&msgs);
    buf_free(&loaded_tool_schemas);
    free(instructions);
    free(tool_schemas);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

static bool parse_prompt(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    if (**p == '"') {
        if (!json_string(p, out)) return false;
    } else {
        *out = xstrdup("");
        if (**p && **p != ']' && !json_skip_value(p)) return false;
    }
    while (**p && **p != ']') {
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            if (!json_skip_value(p)) return false;
        } else {
            break;
        }
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool parse_completion_request(ds4_engine *e, const char *body, int def_tokens,
                                     int ctx_size, request *r, char *err, size_t errlen) {
    request_init(r, REQ_COMPLETION, def_tokens);
    const char *p = body;
    char *prompt = NULL;
    bool got_thinking = false;
    bool thinking_enabled = true;
    ds4_think_mode reasoning_effort = DS4_THINK_HIGH;

    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto bad;
        }
        p++;
        if (!strcmp(key, "prompt")) {
            free(prompt);
            if (!parse_prompt(&p, &prompt)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "model")) {
            free(r->model);
            if (!json_string(&p, &r->model)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "max_tokens")) {
            if (!json_int(&p, &r->max_tokens)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "temperature")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->temperature = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->top_p = (float)v;
        } else if (!strcmp(key, "min_p")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->min_p = (float)v;
        } else if (!strcmp(key, "top_k")) {
            if (!json_int(&p, &r->top_k)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "seed")) {
            double v = 0.0;
            if (!json_number(&p, &v)) {
                free(key);
                goto bad;
            }
            r->seed = v > 0.0 ? (uint64_t)v : 0;
        } else if (!strcmp(key, "stream")) {
            if (!json_bool(&p, &r->stream)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "stream_options")) {
            if (!parse_stream_options(&p, &r->stream_include_usage)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            if (!parse_thinking_control_value(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "reasoning_effort")) {
            if (!parse_reasoning_effort_value(&p, &reasoning_effort)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "think")) {
            if (!json_bool(&p, &thinking_enabled)) {
                free(key);
                goto bad;
            }
            got_thinking = true;
        } else if (!strcmp(key, "stop")) {
            if (!parse_stop(&p, &r->stops)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!prompt) {
        snprintf(err, errlen, "missing prompt");
        request_free(r);
        return false;
    }
    if (!got_thinking && model_alias_disables_thinking(r->model)) thinking_enabled = false;
    if (!got_thinking && model_alias_enables_thinking(r->model)) thinking_enabled = true;
    r->think_mode = ds4_think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort), ctx_size);
    buf rendered = {0};
    buf_puts(&rendered, "<｜begin▁of▁sentence｜>");
    if (r->think_mode == DS4_THINK_MAX) buf_puts(&rendered, ds4_think_max_prefix());
    buf_puts(&rendered, "You are a helpful assistant<｜User｜>");
    buf_puts(&rendered, prompt);
    buf_puts(&rendered, "<｜Assistant｜>");
    buf_puts(&rendered, ds4_think_mode_enabled(r->think_mode) ? "<think>" : "</think>");
    r->prompt_text = buf_take(&rendered);
    ds4_tokenize_rendered_chat(e, r->prompt_text, &r->prompt);
    free(prompt);
    return true;
bad:
    free(prompt);
    snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

static long long wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static bool send_all(int fd, const void *p, size_t n) {
    const char *s = p;
    long long deadline = wall_ms() + DS4_SERVER_SEND_STALL_TIMEOUT_MS;
    while (n) {
        if (g_stop_requested) return false;
        ssize_t w = send(fd, s, n, 0);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            long long remaining = deadline - wall_ms();
            if (remaining <= 0) return false;
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            int timeout = remaining > 50 ? 50 : (int)remaining;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout);
            } while (rc < 0 && errno == EINTR);
            if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) return false;
            continue;
        }
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
        deadline = wall_ms() + DS4_SERVER_SEND_STALL_TIMEOUT_MS;
    }
    return true;
}

static void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

static void json_escape_n(buf *b, const char *s, size_t n) {
    char *tmp = xstrndup(s ? s : "", n);
    json_escape(b, tmp);
    free(tmp);
}

static void json_escape_fragment_n(buf *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
}

#define DS4_DSML "｜DSML｜"
#define DS4_DSML_SHORT "DSML｜"
#define DS4_TOOL_CALLS_START "<" DS4_DSML "tool_calls>"
#define DS4_TOOL_CALLS_END "</" DS4_DSML "tool_calls>"
#define DS4_INVOKE_START "<" DS4_DSML "invoke"
#define DS4_INVOKE_END "</" DS4_DSML "invoke>"
#define DS4_PARAM_START "<" DS4_DSML "parameter"
#define DS4_PARAM_END "</" DS4_DSML "parameter>"
#define DS4_TOOL_CALLS_START_SHORT "<" DS4_DSML_SHORT "tool_calls>"
#define DS4_TOOL_CALLS_END_SHORT "</" DS4_DSML_SHORT "tool_calls>"
#define DS4_INVOKE_START_SHORT "<" DS4_DSML_SHORT "invoke"
#define DS4_INVOKE_END_SHORT "</" DS4_DSML_SHORT "invoke>"
#define DS4_PARAM_START_SHORT "<" DS4_DSML_SHORT "parameter"
#define DS4_PARAM_END_SHORT "</" DS4_DSML_SHORT "parameter>"

static const char *find_any_tool_start(const char *s) {
    const char *best = NULL;
    const char *candidates[] = {
        strstr(s, DS4_TOOL_CALLS_START),
        strstr(s, DS4_TOOL_CALLS_START_SHORT),
        strstr(s, "<tool_calls>"),
    };
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        if (candidates[i] && (!best || candidates[i] < best)) best = candidates[i];
    }
    return best;
}

static const char *find_any_tool_end(const char *s) {
    const char *best = NULL;
    const char *candidates[] = {
        strstr(s, DS4_TOOL_CALLS_END),
        strstr(s, DS4_TOOL_CALLS_END_SHORT),
        strstr(s, "</tool_calls>"),
    };
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); i++) {
        if (candidates[i] && (!best || candidates[i] < best)) best = candidates[i];
    }
    return best;
}

static void observe_tool_markers(const char *scan, bool *saw_start,
                                 bool *saw_end, bool *orphan_end) {
    if (!scan) return;
    bool had_start = *saw_start;
    const char *start = find_any_tool_start(scan);
    if (start) *saw_start = true;

    const char *end_scan = had_start ? scan : (start ? start : NULL);
    const char *end = end_scan ? find_any_tool_end(end_scan) : NULL;
    if (end) {
        *saw_end = true;
    } else if (!had_start && !start && find_any_tool_end(scan)) {
        if (orphan_end) *orphan_end = true;
    }
}

static size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit) {
    while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
    return limit;
}

static const char *skip_ascii_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static const char *find_last_substr(const char *s, const char *needle) {
    if (!s || !needle || !needle[0]) return NULL;
    const char *last = NULL;
    const char *p = s;
    while ((p = strstr(p, needle)) != NULL) {
        last = p;
        p++;
    }
    return last;
}

/* The prompt renderer escapes DSML text so a tool argument can safely contain
 * shell operators or closing tags.  The generated-DSML parser must undo exactly
 * those entities before it turns parameters back into JSON; otherwise
 * parse->render is not a stable cache key. */
static char *dsml_unescape_text(const char *s) {
    buf b = {0};
    for (s = s ? s : ""; *s; s++) {
        if (*s != '&') {
            buf_putc(&b, *s);
        } else if (!strncmp(s, "&amp;", 5)) {
            buf_putc(&b, '&');
            s += 4;
        } else if (!strncmp(s, "&lt;", 4)) {
            buf_putc(&b, '<');
            s += 3;
        } else if (!strncmp(s, "&gt;", 4)) {
            buf_putc(&b, '>');
            s += 3;
        } else if (!strncmp(s, "&quot;", 6)) {
            buf_putc(&b, '"');
            s += 5;
        } else if (!strncmp(s, "&apos;", 6)) {
            buf_putc(&b, '\'');
            s += 5;
        } else {
            buf_putc(&b, '&');
        }
    }
    return buf_take(&b);
}

static char *dsml_attr(const char *tag, const char *name) {
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat);
    if (!p) return NULL;
    p += strlen(pat);
    const char *q = strchr(p, '"');
    if (!q) return NULL;
    char *raw = xstrndup(p, (size_t)(q - p));
    char *decoded = dsml_unescape_text(raw);
    free(raw);
    return decoded;
}

static void tool_call_json_args_add(buf *args, const char *name, const char *value, const char *is_string) {
    if (args->len) buf_puts(args, ", ");
    json_escape(args, name ? name : "");
    buf_puts(args, ": ");
    if (is_string && !strcmp(is_string, "true")) {
        json_escape(args, value ? value : "");
    } else {
        char *min = json_minify_raw_value(value ? value : "null");
        buf_puts(args, min && min[0] ? min : "null");
        free(min);
    }
}

/* DSML produced by the model is usually a flat list of typed parameters:
 *
 *   <parameter name="path" string="true">/tmp/x</parameter>
 *   <parameter name="timeout" string="false">10</parameter>
 *
 * Long generations sometimes drift into a looser XML-ish shape, omitting the
 * outer string attribute and putting child parameters inside it.  The server
 * does not know client tool schemas, so it cannot make that semantically
 * perfect.  Still, returning a structured JSON value lets the client/tool layer
 * reject or repair the call, which is much better than aborting the assistant
 * turn and losing the whole sampled continuation.
 */
static bool dsml_parse_leaf_param_json(const char **p_in, const char *param_start,
                                       const char *param_end, buf *out) {
    const char *p = *p_in;
    if (strncmp(p, param_start, strlen(param_start)) != 0) return false;
    const char *tag_end = strchr(p, '>');
    if (!tag_end) return false;

    char *tag = xstrndup(p, (size_t)(tag_end - p + 1));
    char *name = dsml_attr(tag, "name");
    char *is_string = dsml_attr(tag, "string");
    free(tag);
    if (!name) {
        free(is_string);
        return false;
    }

    const char *value_start = tag_end + 1;
    const char *value_end = strstr(value_start, param_end);
    if (!value_end) {
        free(name);
        free(is_string);
        return false;
    }

    char *raw_value = xstrndup(value_start, (size_t)(value_end - value_start));
    const char *type = is_string ? is_string : "true";
    char *value = !strcmp(type, "true") ?
        dsml_unescape_text(raw_value) : xstrdup(raw_value);
    tool_call_json_args_add(out, name, value, type);

    free(name);
    free(is_string);
    free(raw_value);
    free(value);
    *p_in = value_end + strlen(param_end);
    return true;
}

static bool dsml_parse_nested_params_object(const char **p_in,
                                            const char *param_start,
                                            const char *param_end,
                                            buf *out) {
    const char *p = *p_in;
    buf members = {0};
    bool any = false;

    for (;;) {
        p = skip_ascii_ws(p);
        if (strncmp(p, param_start, strlen(param_start)) != 0) break;
        if (!dsml_parse_leaf_param_json(&p, param_start, param_end, &members)) {
            buf_free(&members);
            return false;
        }
        any = true;
    }

    if (!any) {
        buf_free(&members);
        return false;
    }
    buf_putc(out, '{');
    buf_puts(out, members.ptr ? members.ptr : "");
    buf_putc(out, '}');
    buf_free(&members);
    *p_in = p;
    return true;
}

static void split_reasoning_content(const char *text, size_t n, char **content_out, char **reasoning_out) {
    char *s = xstrndup(text ? text : "", n);
    char *body = s;
    if (!strncmp(body, "<think>", 7)) body += 7;

    char *think_end = strstr(body, "</think>");
    if (think_end) {
        *think_end = '\0';
        *reasoning_out = xstrdup(body);
        *content_out = xstrdup(think_end + 8);
    } else {
        *reasoning_out = NULL;
        *content_out = xstrdup(s);
    }
    free(s);
}

static bool parse_generated_message_ex(const char *text, bool require_thinking_closed,
                                       char **content_out, char **reasoning_out,
                                       tool_calls *calls) {
    text = text ? text : "";
    const char *tool_search = text;

    /* When thinking mode is enabled the model is expected to close
     * </think> before it enters the executable assistant surface.  DSML inside
     * reasoning is just model text: it may be a mistaken attempt, a quotation,
     * or an explanation of the protocol.  Treating it as a real tool call
     * duplicates it into both reasoning and structured tool_calls, and can make
     * clients execute something the assistant had not actually emitted as its
     * post-thinking action. */
    if (require_thinking_closed) {
        const char *think_end = find_last_substr(text, "</think>");
        if (!think_end) {
            split_reasoning_content(text, strlen(text), content_out, reasoning_out);
            return true;
        }
        tool_search = think_end + 8;
    }

    const char *start = strstr(tool_search, "\n\n" DS4_TOOL_CALLS_START);
    int style = 0; /* 0: DSML, 1: plain XML, 2: DSML with the first vertical bar omitted. */
    if (!start) start = strstr(tool_search, DS4_TOOL_CALLS_START);
    if (!start) {
        start = strstr(tool_search, "\n\n" DS4_TOOL_CALLS_START_SHORT);
        style = start ? 2 : style;
    }
    if (!start) {
        start = strstr(tool_search, DS4_TOOL_CALLS_START_SHORT);
        style = start ? 2 : style;
    }
    if (!start) {
        start = strstr(tool_search, "\n\n<tool_calls>");
        style = start ? 1 : style;
    }
    if (!start) {
        start = strstr(tool_search, "<tool_calls>");
        style = start ? 1 : style;
    }
    if (!start) {
        split_reasoning_content(text, strlen(text), content_out, reasoning_out);
        return true;
    }

    size_t content_len = trim_tool_separator_ws(text, 0, (size_t)(start - text));
    const char *raw_block_start = start;
    const char *tool_calls_start = DS4_TOOL_CALLS_START;
    const char *tool_calls_end = DS4_TOOL_CALLS_END;
    const char *invoke_start = DS4_INVOKE_START;
    const char *invoke_end = DS4_INVOKE_END;
    const char *param_start = DS4_PARAM_START;
    const char *param_end = DS4_PARAM_END;
    if (style == 1) {
        tool_calls_start = "<tool_calls>";
        tool_calls_end = "</tool_calls>";
        invoke_start = "<invoke";
        invoke_end = "</invoke>";
        param_start = "<parameter";
        param_end = "</parameter>";
    } else if (style == 2) {
        tool_calls_start = DS4_TOOL_CALLS_START_SHORT;
        tool_calls_end = DS4_TOOL_CALLS_END_SHORT;
        invoke_start = DS4_INVOKE_START_SHORT;
        invoke_end = DS4_INVOKE_END_SHORT;
        param_start = DS4_PARAM_START_SHORT;
        param_end = DS4_PARAM_END_SHORT;
    }

    const char *p = strstr(start, tool_calls_start);
    if (!p) return false;
    p += strlen(tool_calls_start);

    for (;;) {
        p = skip_ascii_ws(p);
        if (!strncmp(p, tool_calls_end, strlen(tool_calls_end))) {
            const char *raw_block_end = p + strlen(tool_calls_end);
            free(calls->raw_dsml);
            calls->raw_dsml = xstrndup(raw_block_start, (size_t)(raw_block_end - raw_block_start));
            split_reasoning_content(text, content_len, content_out, reasoning_out);
            return true;
        }
        if (strncmp(p, invoke_start, strlen(invoke_start)) != 0) return false;
        const char *tag_end = strchr(p, '>');
        if (!tag_end) return false;
        char *tag = xstrndup(p, (size_t)(tag_end - p + 1));
        char *name = dsml_attr(tag, "name");
        free(tag);
        if (!name) return false;
        p = tag_end + 1;

        buf args = {0};
        while (true) {
            p = skip_ascii_ws(p);
            if (!strncmp(p, invoke_end, strlen(invoke_end))) {
                p += strlen(invoke_end);
                break;
            }
            if (strncmp(p, param_start, strlen(param_start)) != 0) {
                free(name);
                buf_free(&args);
                return false;
            }
            tag_end = strchr(p, '>');
            if (!tag_end) {
                free(name);
                buf_free(&args);
                return false;
            }
            tag = xstrndup(p, (size_t)(tag_end - p + 1));
            char *param_name = dsml_attr(tag, "name");
            char *param_is_string = dsml_attr(tag, "string");
            free(tag);
            if (!param_name) {
                free(name);
                free(param_name);
                free(param_is_string);
                buf_free(&args);
                return false;
            }
            const char *value_start = tag_end + 1;
            if (!param_is_string &&
                !strncmp(skip_ascii_ws(value_start), param_start, strlen(param_start)))
            {
                buf nested = {0};
                const char *nested_p = value_start;
                if (!dsml_parse_nested_params_object(&nested_p, param_start,
                                                     param_end, &nested)) {
                    free(name);
                    free(param_name);
                    buf_free(&nested);
                    buf_free(&args);
                    return false;
                }
                tool_call_json_args_add(&args, param_name,
                                        nested.ptr ? nested.ptr : "{}",
                                        "false");
                buf_free(&nested);
                p = skip_ascii_ws(nested_p);
                if (!strncmp(p, param_end, strlen(param_end))) {
                    p += strlen(param_end);
                }
                free(param_name);
                continue;
            }
            const char *value_end = strstr(value_start, param_end);
            if (!value_end) {
                free(name);
                free(param_name);
                free(param_is_string);
                buf_free(&args);
                return false;
            }
            char *raw_value = xstrndup(value_start, (size_t)(value_end - value_start));
            const char *type = param_is_string ? param_is_string : "true";
            char *value = !strcmp(type, "true") ?
                dsml_unescape_text(raw_value) : xstrdup(raw_value);
            tool_call_json_args_add(&args, param_name, value, type);
            free(param_name);
            free(param_is_string);
            free(raw_value);
            free(value);
            p = value_end + strlen(param_end);
        }

        tool_call tc = {0};
        tc.name = name;
        buf wrapped = {0};
        buf_putc(&wrapped, '{');
        buf_puts(&wrapped, args.ptr ? args.ptr : "");
        buf_putc(&wrapped, '}');
        tc.arguments = buf_take(&wrapped);
        tool_calls_push(calls, tc);
        buf_free(&args);
    }
}

static const char *tool_parse_failure_recovery_finish(const char *finish) {
    /* Once DSML failed to parse there is no executable tool call to report.
     * Preserve a true length stop, because callers can distinguish truncation
     * from a completed turn.  Every other non-error tool-parse failure becomes
     * a normal assistant stop with the raw model text returned as content. */
    if (finish && !strcmp(finish, "length")) return "length";
    return "stop";
}

static bool parse_generated_message_for_response(const char *text,
                                                 bool has_tools,
                                                 bool saw_tool_start,
                                                 bool require_thinking_closed,
                                                 const char **finish_io,
                                                 char *err,
                                                 size_t errlen,
                                                 char **content_out,
                                                 char **reasoning_out,
                                                 tool_calls *calls,
                                                 bool *recovered_out) {
    if (recovered_out) *recovered_out = false;

    bool parsed_ok = parse_generated_message_ex(text ? text : "",
                                                require_thinking_closed,
                                                content_out, reasoning_out,
                                                calls);
    if (parsed_ok) return true;

    free(*content_out);
    free(*reasoning_out);
    *content_out = xstrdup(text ? text : "");
    *reasoning_out = NULL;
    tool_calls_free(calls);

    /* A malformed tool block is model output, not a server failure.  Returning
     * the sampled text as a regular assistant message keeps the API response
     * valid and gives the caller enough transcript context to retry or recover
     * on a later turn, instead of terminating the whole session on
     * finish_reason="error". */
    const char *finish = finish_io && *finish_io ? *finish_io : "stop";
    if (has_tools && saw_tool_start && strcmp(finish, "error") != 0) {
        if (finish_io) *finish_io = tool_parse_failure_recovery_finish(finish);
        if (err && errlen) snprintf(err, errlen, "invalid tool call");
        if (recovered_out) *recovered_out = true;
    }
    return false;
}

static void append_json_object_string(buf *b, const char *json) {
    buf tmp = {0};
    append_json_object_or_empty(&tmp, json);
    json_escape(b, tmp.ptr ? tmp.ptr : "{}");
    buf_free(&tmp);
}

static void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                   const tool_schema_orders *orders) {
    (void)orders;
    buf_putc(b, '[');
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i) buf_putc(b, ',');
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "%s_tool_%d", id_prefix, i);
        buf_puts(b, "{\"id\":");
        json_escape(b, tc->id ? tc->id : idbuf);
        buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ",\"arguments\":");
        append_json_object_string(b, tc->arguments);
        buf_puts(b, "}}");
    }
    buf_putc(b, ']');
}

static void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix,
                                         const tool_schema_orders *orders) {
    (void)orders;
    buf_putc(b, '[');
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        if (i) buf_putc(b, ',');
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "%s_tool_%d", id_prefix, i);
        buf_puts(b, "{\"index\":");
        buf_printf(b, "%d", i);
        buf_puts(b, ",\"id\":");
        json_escape(b, tc->id ? tc->id : idbuf);
        buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(b, tc->name ? tc->name : "");
        buf_puts(b, ",\"arguments\":");
        append_json_object_string(b, tc->arguments);
        buf_puts(b, "}}");
    }
    buf_putc(b, ']');
}

static void append_cors_headers(buf *h) {
    buf_puts(h,
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n");
}

static bool http_response(int fd, bool enable_cors, int code, const char *type, const char *body) {
    const char *reason = code == 200 ? "OK" :
                         code == 204 ? "No Content" :
                         code == 400 ? "Bad Request" :
                         code == 404 ? "Not Found" :
                         code == 409 ? "Conflict" :
                         code == 500 ? "Internal Server Error" : "Error";
    const size_t body_len = body ? strlen(body) : 0;
    buf h = {0};
    buf_printf(&h,
        "HTTP/1.1 %d %s\r\n"
        "Content-Length: %zu\r\n",
        code, reason, body_len);
    if (type && type[0]) {
        buf_puts(&h, "Content-Type: ");
        buf_puts(&h, type);
        buf_puts(&h, "\r\n");
    }
    if (enable_cors) append_cors_headers(&h);
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    if (ok && body_len) ok = send_all(fd, body, body_len);
    buf_free(&h);
    return ok;
}

static bool http_error(int fd, bool enable_cors, int code, const char *msg) {
    buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":");
    json_escape(&b, msg);
    buf_puts(&b, ",\"type\":\"invalid_request_error\"}}\n");
    bool ok = http_response(fd, enable_cors, code, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static const char *context_length_error_param(const request *r) {
    if (!r) return "prompt";
    if (r->api == API_RESPONSES) return "input";
    return r->kind == REQ_COMPLETION ? "prompt" : "messages";
}

static bool request_exceeds_context(const request *r, int ctx_size) {
    /* ds4_session_sync() rejects prompt->len >= ctx_size because generation
     * needs at least one free context slot.  Catch the same boundary here so
     * clients get a normal protocol error instead of a later backend failure. */
    return r && r->prompt.len >= ctx_size;
}

static bool http_error_context_length_exceeded(int fd, bool enable_cors,
                                               const request *r,
                                               int n_prompt_tokens,
                                               int ctx_size) {
    buf b = {0};
    char msg[160];
    snprintf(msg, sizeof(msg),
             "Prompt has %d tokens, but the configured context size is %d tokens",
             n_prompt_tokens, ctx_size);

    if (r && r->api == API_ANTHROPIC) {
        buf_puts(&b, "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":");
        json_escape(&b, msg);
        buf_puts(&b, ",\"n_prompt_tokens\":");
        buf_printf(&b, "%d", n_prompt_tokens);
        buf_puts(&b, ",\"n_ctx\":");
        buf_printf(&b, "%d", ctx_size);
        buf_puts(&b, "}}\n");
    } else {
        buf_puts(&b, "{\"error\":{\"message\":");
        json_escape(&b, msg);
        buf_puts(&b, ",\"type\":\"invalid_request_error\",\"param\":");
        json_escape(&b, context_length_error_param(r));
        buf_puts(&b, ",\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":");
        buf_printf(&b, "%d", n_prompt_tokens);
        buf_puts(&b, ",\"n_ctx\":");
        buf_printf(&b, "%d", ctx_size);
        buf_puts(&b, "}}\n");
    }
    bool ok = http_response(fd, enable_cors, 400, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

/* Streaming is a translation state machine over the raw DS4 text.  The model
 * may produce <think> and DSML tool blocks; clients should receive those as
 * protocol-native reasoning/tool deltas, never as visible assistant text. */
static bool sse_headers(int fd, bool enable_cors) {
    buf h = {0};
    buf_puts(&h,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n");
    if (enable_cors) append_cors_headers(&h);
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    buf_free(&h);
    return ok;
}

static bool sse_chunk(int fd, const request *r, const char *id, const char *text, const char *finish) {
    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":");
        if (text) {
            buf_puts(&b, "{\"content\":");
            json_escape(&b, text);
            buf_putc(&b, '}');
        } else {
            buf_puts(&b, finish ? "{}" : "{\"role\":\"assistant\"}");
        }
        buf_puts(&b, ",\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text ? text : "");
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    }
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static int clamp_usage_tokens(int value, int max) {
    if (value < 0) return 0;
    if (max >= 0 && value > max) return max;
    return value;
}

static void append_openai_usage_json(buf *b, const request *r,
                                     int prompt_tokens, int completion_tokens) {
    int cached_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    cached_tokens = clamp_usage_tokens(cached_tokens, prompt_tokens);
    cache_write_tokens = clamp_usage_tokens(cache_write_tokens, prompt_tokens - cached_tokens);
    /* OpenAI defines cached_tokens as prompt tokens retrieved from cache.
     * Newly-prefilled tokens are useful to expose, but they are a DS4 extension
     * and must stay separate so OpenAI-compatible clients do not over-count
     * cache hits. */
    buf_printf(b,
               "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
               "\"prompt_tokens_details\":{\"cached_tokens\":%d,\"cache_write_tokens\":%d}}",
               prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
               cached_tokens, cache_write_tokens);
}

static bool sse_usage_chunk(int fd, const request *r, const char *id,
                            int prompt_tokens, int completion_tokens) {
    if (!r->stream_include_usage) return true;

    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
    }
    append_openai_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}\n\n");

    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static bool sse_done(int fd, const request *r, const char *id,
                     int prompt_tokens, int completion_tokens) {
    return sse_usage_chunk(fd, r, id, prompt_tokens, completion_tokens) &&
           send_all(fd, "data: [DONE]\n\n", 14);
}

static bool sse_chat_finish(int fd, const request *r, const char *id, const char *content,
                            const char *reasoning, const tool_calls *calls, const char *finish,
                            int prompt_tokens, int completion_tokens) {
    if (!sse_chunk(fd, r, id, NULL, NULL)) return false;

    buf b = {0};
    long now = (long)time(NULL);
    if (reasoning && reasoning[0]) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":");
        json_escape(&b, reasoning);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    if (content && content[0]) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
        json_escape(&b, content);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    if (calls && calls->len) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        append_tool_call_deltas_json(&b, calls, id, &r->tool_orders);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    json_escape(&b, finish);
    buf_puts(&b, "}]}\n\n");

    bool ok = send_all(fd, b.ptr, b.len) &&
              sse_done(fd, r, id, prompt_tokens, completion_tokens);
    buf_free(&b);
    return ok;
}

typedef enum {
    OPENAI_STREAM_THINKING,
    OPENAI_STREAM_TEXT,
    OPENAI_STREAM_TOOL,
    OPENAI_STREAM_SUPPRESS,
} openai_stream_mode;

typedef enum {
    DSML_TOOL_BETWEEN_INVOKES,
    DSML_TOOL_BETWEEN_PARAMS,
    DSML_TOOL_PARAM_VALUE,
    DSML_TOOL_DONE,
    DSML_TOOL_ERROR,
} dsml_tool_stream_state;

/* Shared states for protocol-specific DSML stream projections.  The model
 * still samples DSML; these states only translate already-sampled bytes into
 * OpenAI / Anthropic wire events while final parsing remains authoritative. */
typedef struct {
    dsml_tool_stream_state state;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} openai_tool_stream;

typedef struct {
    openai_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool sent_reasoning;
    bool sent_content;
    openai_tool_stream tool;
} openai_stream;

static void openai_stream_start(const request *r, openai_stream *st) {
    memset(st, 0, sizeof(*st));
    st->active = true;
    st->mode = ds4_think_mode_enabled(r->think_mode) ? OPENAI_STREAM_THINKING : OPENAI_STREAM_TEXT;
}

static void openai_tool_stream_free(openai_tool_stream *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->ids_cap; i++) free(ts->ids[i]);
    free(ts->ids);
    ts->ids = NULL;
    ts->ids_cap = 0;
}

static void openai_stream_free(openai_stream *st) {
    if (!st) return;
    openai_tool_stream_free(&st->tool);
}

static bool openai_tool_stream_has_id(const openai_tool_stream *ts,
                                      const char *id, int upto) {
    if (!ts || !id || !id[0]) return false;
    if (upto > ts->ids_cap) upto = ts->ids_cap;
    for (int i = 0; i < upto; i++) {
        if (ts->ids[i] && !strcmp(ts->ids[i], id)) return true;
    }
    return false;
}

static const char *openai_tool_stream_id(server *s, openai_tool_stream *ts,
                                         int index) {
    if (!ts || index < 0) return "";
    if (index >= ts->ids_cap) {
        int old = ts->ids_cap;
        int cap = old ? old : 4;
        while (cap <= index) cap *= 2;
        ts->ids = xrealloc(ts->ids, (size_t)cap * sizeof(ts->ids[0]));
        memset(ts->ids + old, 0, (size_t)(cap - old) * sizeof(ts->ids[0]));
        ts->ids_cap = cap;
    }
    if (!ts->ids[index]) {
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), API_OPENAI);
            if (!openai_tool_stream_has_id(ts, id, index) &&
                !tool_memory_has_id(s, id)) break;
        }
        ts->ids[index] = xstrdup(id);
    }
    return ts->ids[index];
}

static size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final);

static bool sse_chat_delta_n(int fd, const request *r, const char *id,
                             const char *field, const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{");
    json_escape(&b, field);
    buf_putc(&b, ':');
    json_escape_n(&b, text, len);
    buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

/* OpenAI clients can consume function.arguments as a stream of JSON text
 * fragments.  DS4 generates XML-ish DSML instead, so this parser switches to a
 * hidden tool mode at <...tool_calls>, emits the tool header once the invoke tag
 * is complete, then translates each parameter body into argument deltas while
 * holding only tiny tails for partial closing tags, UTF-8, and DSML entities. */
static bool sse_chat_tool_call_start_delta(int fd, const request *r, const char *id,
                                           int index, const char *tool_id,
                                           const char *name) {
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    buf_printf(&b, "%d", index);
    buf_puts(&b, ",\"id\":");
    json_escape(&b, tool_id ? tool_id : "");
    buf_puts(&b, ",\"type\":\"function\",\"function\":{\"name\":");
    json_escape(&b, name ? name : "");
    buf_puts(&b, ",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static bool sse_chat_tool_call_args_delta_n(int fd, const request *r, const char *id,
                                            int index, const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    long now = (long)time(NULL);
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    buf_printf(&b, "%d", index);
    buf_puts(&b, ",\"function\":{\"arguments\":");
    json_escape_n(&b, text, len);
    buf_puts(&b, "}}]},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    return pos <= raw_len && raw_len - pos >= n && !memcmp(raw + pos, lit, n);
}

static bool raw_partial_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos > raw_len || raw_len - pos >= n) return false;
    return !memcmp(raw + pos, lit, raw_len - pos);
}

static bool raw_partial_any(const char *raw, size_t raw_len, size_t pos,
                            const char *a, const char *b) {
    return raw_partial_lit(raw, raw_len, pos, a) || raw_partial_lit(raw, raw_len, pos, b);
}

static const char *find_lit_bounded(const char *s, size_t n, const char *lit) {
    size_t m = strlen(lit);
    if (m == 0) return s;
    if (n < m) return NULL;
    for (size_t i = 0; i <= n - m; i++) {
        if (!memcmp(s + i, lit, m)) return s + i;
    }
    return NULL;
}

typedef enum {
    DSML_DECODE_OUTSIDE,
    DSML_DECODE_STRUCTURAL,
    DSML_DECODE_STRING_BODY,
    DSML_DECODE_JSON_STRUCTURAL,
    DSML_DECODE_JSON_STRING,
} dsml_decode_state;

typedef enum {
    DSML_TRACK_SEARCH,
    DSML_TRACK_STRUCTURAL,
    DSML_TRACK_STRING_BODY,
    DSML_TRACK_JSON_PARAM,
    DSML_TRACK_DONE,
} dsml_track_mode;

typedef struct {
    const char *tool_calls_start;
    const char *tool_calls_end;
    const char *invoke_start;
    const char *invoke_end;
    const char *param_start;
    const char *param_end;
} dsml_syntax;

static const dsml_syntax dsml_syntaxes[] = {
    {
        DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END,
        DS4_INVOKE_START, DS4_INVOKE_END,
        DS4_PARAM_START, DS4_PARAM_END,
    },
    {
        DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT,
        DS4_INVOKE_START_SHORT, DS4_INVOKE_END_SHORT,
        DS4_PARAM_START_SHORT, DS4_PARAM_END_SHORT,
    },
    {
        "<tool_calls>", "</tool_calls>",
        "<invoke", "</invoke>",
        "<parameter", "</parameter>",
    },
};

typedef struct {
    dsml_track_mode mode;
    dsml_decode_state decode;
    const dsml_syntax *syn;
    size_t pos;
    bool json_in_string;
    bool json_escaped;
} dsml_decode_tracker;

static bool raw_partial_lit_min(const char *raw, size_t raw_len, size_t pos,
                                const char *lit, size_t min_len) {
    size_t lit_len = strlen(lit);
    if (!raw || pos > raw_len || raw_len - pos >= lit_len) return false;
    size_t avail = raw_len - pos;
    return avail >= min_len && !memcmp(raw + pos, lit, avail);
}

static size_t dsml_max_tool_start_len(void) {
    size_t max = 0;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        size_t n = strlen(dsml_syntaxes[i].tool_calls_start);
        if (n > max) max = n;
    }
    return max;
}

static bool dsml_find_tool_start(const char *raw, size_t raw_len,
                                 size_t *pos_out,
                                 const dsml_syntax **syn_out) {
    const char *best = NULL;
    const dsml_syntax *best_syn = NULL;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        const char *p = find_lit_bounded(raw, raw_len, dsml_syntaxes[i].tool_calls_start);
        if (p && (!best || p < best)) {
            best = p;
            best_syn = &dsml_syntaxes[i];
        }
    }
    if (!best) return false;
    *pos_out = (size_t)(best - raw) + strlen(best_syn->tool_calls_start);
    *syn_out = best_syn;
    return true;
}

static bool dsml_find_tool_start_from(const char *raw, size_t raw_len,
                                      size_t start,
                                      size_t *pos_out,
                                      const dsml_syntax **syn_out) {
    if (start > raw_len) return false;
    size_t rel = 0;
    if (!dsml_find_tool_start(raw + start, raw_len - start, &rel, syn_out)) {
        return false;
    }
    *pos_out = start + rel;
    return true;
}

static bool dsml_attr_is_string_true(const char *raw, size_t raw_len,
                                     size_t tag_start, size_t tag_end) {
    if (tag_end <= tag_start || tag_end > raw_len) return false;
    char *tag = xstrndup(raw + tag_start, tag_end - tag_start);
    char *is_string = dsml_attr(tag, "string");
    bool result = is_string && !strcmp(is_string, "true");
    free(is_string);
    free(tag);
    return result;
}

#ifdef DS4_SERVER_TEST
static bool raw_suffix_partial_lit(const char *raw, size_t raw_len,
                                   const char *lit, size_t min_len) {
    size_t lit_len = strlen(lit);
    if (!raw || raw_len == 0 || lit_len == 0) return false;
    size_t max = raw_len < lit_len ? raw_len : lit_len - 1;
    for (size_t n = min_len; n <= max; n++) {
        if (!memcmp(raw + raw_len - n, lit, n)) return true;
    }
    return false;
}

static dsml_decode_state dsml_decode_scan_json_param(const char *raw,
                                                     size_t raw_len,
                                                     size_t pos,
                                                     const dsml_syntax *syn) {
    bool in_string = false;
    bool escaped = false;
    while (pos < raw_len) {
        if (!in_string && raw_full_lit(raw, raw_len, pos, syn->param_end)) {
            return DSML_DECODE_STRUCTURAL;
        }
        unsigned char c = (unsigned char)raw[pos++];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else if (c == '"') {
            in_string = true;
        }
    }
    if (!in_string && raw_suffix_partial_lit(raw, raw_len, syn->param_end, 2)) {
        return DSML_DECODE_STRUCTURAL;
    }
    return in_string ? DSML_DECODE_JSON_STRING : DSML_DECODE_JSON_STRUCTURAL;
}

/* Slow reference recognizer used by tests. */
static dsml_decode_state dsml_decode_state_for_text(const char *raw, size_t raw_len) {
    if (!raw || raw_len == 0) return DSML_DECODE_OUTSIDE;

    size_t pos = 0;
    const dsml_syntax *syn = NULL;
    if (!dsml_find_tool_start(raw, raw_len, &pos, &syn)) {
        return DSML_DECODE_OUTSIDE;
    }

    for (;;) {
        while (pos < raw_len && isspace((unsigned char)raw[pos])) pos++;
        if (pos >= raw_len) return DSML_DECODE_STRUCTURAL;

        if (raw_full_lit(raw, raw_len, pos, syn->tool_calls_end)) {
            return DSML_DECODE_OUTSIDE;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_end)) {
            pos += strlen(syn->invoke_end);
            continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->invoke_start)) {
            const char *tag_end = memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end) return DSML_DECODE_STRUCTURAL;
            pos = (size_t)(tag_end - raw) + 1;
            continue;
        }
        if (raw_full_lit(raw, raw_len, pos, syn->param_start)) {
            size_t tag_start = pos;
            const char *tag_end_ptr = memchr(raw + pos, '>', raw_len - pos);
            if (!tag_end_ptr) return DSML_DECODE_STRUCTURAL;
            size_t tag_end = (size_t)(tag_end_ptr - raw) + 1;
            bool string_value = dsml_attr_is_string_true(raw, raw_len, tag_start, tag_end);
            pos = tag_end;

            if (string_value) {
                const char *end = find_lit_bounded(raw + pos, raw_len - pos, syn->param_end);
                if (!end) {
                    if (raw_suffix_partial_lit(raw, raw_len, syn->param_end, 2)) {
                        return DSML_DECODE_STRUCTURAL;
                    }
                    return DSML_DECODE_STRING_BODY;
                }
                pos = (size_t)(end - raw) + strlen(syn->param_end);
                continue;
            }

            dsml_decode_state json_state =
                dsml_decode_scan_json_param(raw, raw_len, pos, syn);
            if (json_state == DSML_DECODE_STRUCTURAL) {
                const char *end = find_lit_bounded(raw + pos, raw_len - pos, syn->param_end);
                if (!end) return DSML_DECODE_STRUCTURAL;
                pos = (size_t)(end - raw) + strlen(syn->param_end);
                continue;
            }
            return json_state;
        }

        for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
            if (raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].tool_calls_end) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].invoke_start) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].invoke_end) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].param_start) ||
                raw_partial_lit(raw, raw_len, pos, dsml_syntaxes[i].param_end))
            {
                return DSML_DECODE_STRUCTURAL;
            }
        }
        return DSML_DECODE_STRUCTURAL;
    }
}
#endif

static bool dsml_decode_state_is_tool(dsml_decode_state state) {
    return state != DSML_DECODE_OUTSIDE;
}

static bool dsml_decode_state_uses_payload_sampling(dsml_decode_state state) {
    return state == DSML_DECODE_STRING_BODY || state == DSML_DECODE_JSON_STRING;
}

static void dsml_decode_tracker_init(dsml_decode_tracker *dt) {
    memset(dt, 0, sizeof(*dt));
    dt->mode = DSML_TRACK_SEARCH;
    dt->decode = DSML_DECODE_OUTSIDE;
}

/* Track where generation is inside a DSML tool call.  This is intentionally a
 * forgiving recognizer, not a validator: malformed DSML still gets parsed later
 * by the normal tool-call parser.  Here we only need enough state to decide
 * whether the next token belongs to protocol syntax or arbitrary payload. */
static void dsml_decode_tracker_update(dsml_decode_tracker *dt,
                                       const char *raw, size_t raw_len) {
    if (!dt || !raw) return;

    for (;;) {
        if (dt->mode == DSML_TRACK_DONE) {
            dt->decode = DSML_DECODE_OUTSIDE;
            return;
        }

        if (dt->mode == DSML_TRACK_SEARCH) {
            size_t pos = 0;
            const dsml_syntax *syn = NULL;
            if (!dsml_find_tool_start_from(raw, raw_len, dt->pos, &pos, &syn)) {
                size_t hold = dsml_max_tool_start_len();
                dt->pos = raw_len > hold ? raw_len - hold : 0;
                dt->decode = DSML_DECODE_OUTSIDE;
                return;
            }
            dt->syn = syn;
            dt->pos = pos;
            dt->mode = DSML_TRACK_STRUCTURAL;
            dt->decode = DSML_DECODE_STRUCTURAL;
        }

        if (dt->mode == DSML_TRACK_STRING_BODY) {
            while (dt->pos < raw_len) {
                if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                    dt->pos += strlen(dt->syn->param_end);
                    dt->mode = DSML_TRACK_STRUCTURAL;
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    goto structural;
                }
                if (raw_partial_lit_min(raw, raw_len, dt->pos, dt->syn->param_end, 2)) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                dt->pos++;
            }
            dt->decode = DSML_DECODE_STRING_BODY;
            return;
        }

        if (dt->mode == DSML_TRACK_JSON_PARAM) {
            while (dt->pos < raw_len) {
                if (!dt->json_in_string) {
                    if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_end)) {
                        dt->pos += strlen(dt->syn->param_end);
                        dt->mode = DSML_TRACK_STRUCTURAL;
                        dt->decode = DSML_DECODE_STRUCTURAL;
                        goto structural;
                    }
                    if (raw_partial_lit_min(raw, raw_len, dt->pos, dt->syn->param_end, 2)) {
                        dt->decode = DSML_DECODE_STRUCTURAL;
                        return;
                    }
                }

                unsigned char c = (unsigned char)raw[dt->pos++];
                if (dt->json_in_string) {
                    if (dt->json_escaped) {
                        dt->json_escaped = false;
                    } else if (c == '\\') {
                        dt->json_escaped = true;
                    } else if (c == '"') {
                        dt->json_in_string = false;
                    }
                } else if (c == '"') {
                    dt->json_in_string = true;
                }
            }
            dt->decode = dt->json_in_string ?
                DSML_DECODE_JSON_STRING : DSML_DECODE_JSON_STRUCTURAL;
            return;
        }

structural:
        while (dt->mode == DSML_TRACK_STRUCTURAL) {
            while (dt->pos < raw_len && isspace((unsigned char)raw[dt->pos])) dt->pos++;
            if (dt->pos >= raw_len) {
                dt->decode = DSML_DECODE_STRUCTURAL;
                return;
            }

            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end)) {
                dt->mode = DSML_TRACK_DONE;
                dt->pos += strlen(dt->syn->tool_calls_end);
                dt->decode = DSML_DECODE_OUTSIDE;
                return;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_end)) {
                dt->pos += strlen(dt->syn->invoke_end);
                continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->invoke_start)) {
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                dt->pos = (size_t)(tag_end - raw) + 1;
                continue;
            }
            if (raw_full_lit(raw, raw_len, dt->pos, dt->syn->param_start)) {
                size_t tag_start = dt->pos;
                const char *tag_end = memchr(raw + dt->pos, '>', raw_len - dt->pos);
                if (!tag_end) {
                    dt->decode = DSML_DECODE_STRUCTURAL;
                    return;
                }
                size_t tag_after = (size_t)(tag_end - raw) + 1;
                bool string_value = dsml_attr_is_string_true(raw, raw_len, tag_start, tag_after);
                dt->pos = tag_after;
                if (string_value) {
                    dt->mode = DSML_TRACK_STRING_BODY;
                    dt->decode = DSML_DECODE_STRING_BODY;
                } else {
                    dt->mode = DSML_TRACK_JSON_PARAM;
                    dt->json_in_string = false;
                    dt->json_escaped = false;
                    dt->decode = DSML_DECODE_JSON_STRUCTURAL;
                }
                break;
            }

            if (raw_partial_lit(raw, raw_len, dt->pos, dt->syn->tool_calls_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_start) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->invoke_end) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_start) ||
                raw_partial_lit(raw, raw_len, dt->pos, dt->syn->param_end))
            {
                dt->decode = DSML_DECODE_STRUCTURAL;
                return;
            }

            dt->decode = DSML_DECODE_STRUCTURAL;
            return;
        }
    }
}

static size_t dsml_entity_stream_safe_len(const char *raw, size_t start, size_t limit) {
    static const char *ents[] = {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;"};
    const size_t max_ent = 6;
    size_t scan = limit > start + max_ent ? limit - max_ent : start;
    for (size_t i = limit; i > scan; i--) {
        if (raw[i - 1] != '&') continue;
        size_t amp = i - 1;
        size_t tail = limit - amp;
        for (size_t ei = 0; ei < sizeof(ents) / sizeof(ents[0]); ei++) {
            size_t elen = strlen(ents[ei]);
            if (tail < elen && !memcmp(raw + amp, ents[ei], tail)) return amp;
        }
        break;
    }
    return limit;
}

static size_t tool_param_value_stream_safe_len(const char *raw, size_t start,
                                               size_t raw_len, const char *param_end,
                                               bool is_string) {
    size_t limit = raw_len;
    size_t end_len = strlen(param_end);
    size_t scan = raw_len > start + end_len ? raw_len - end_len : start;
    for (size_t i = raw_len; i > scan; i--) {
        if (raw[i - 1] != '<') continue;
        size_t marker = i - 1;
        size_t tail = raw_len - marker;
        if (tail < end_len && !memcmp(raw + marker, param_end, tail)) limit = marker;
        break;
    }
    if (is_string) limit = dsml_entity_stream_safe_len(raw, start, limit);
    return utf8_stream_safe_len(raw, start, limit, false);
}

static bool openai_tool_emit_args_fragment(int fd, const request *r, const char *id,
                                           openai_tool_stream *ts,
                                           const char *text, size_t len) {
    return sse_chat_tool_call_args_delta_n(fd, r, id, ts->index, text, len);
}

static bool openai_tool_emit_string_value(int fd, const request *r, const char *id,
                                          openai_tool_stream *ts,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    char *raw = xstrndup(text, len);
    char *unescaped = dsml_unescape_text(raw);
    buf frag = {0};
    json_escape_fragment_n(&frag, unescaped, strlen(unescaped));
    bool ok = openai_tool_emit_args_fragment(fd, r, id, ts, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    free(unescaped);
    free(raw);
    return ok;
}

static bool openai_tool_emit_param_prefix(int fd, const request *r, const char *id,
                                          openai_tool_stream *ts,
                                          const char *name, bool is_string) {
    buf frag = {0};
    if (ts->first_param) ts->first_param = false;
    else buf_putc(&frag, ',');
    json_escape(&frag, name ? name : "");
    buf_putc(&frag, ':');
    if (is_string) buf_putc(&frag, '"');
    bool ok = openai_tool_emit_args_fragment(fd, r, id, ts, frag.ptr ? frag.ptr : "", frag.len);
    buf_free(&frag);
    return ok;
}

static bool openai_tool_stream_init(openai_tool_stream *ts, const char *raw,
                                    size_t raw_len, size_t pos) {
    openai_tool_stream_free(ts);
    memset(ts, 0, sizeof(*ts));
    ts->active = true;
    ts->state = DSML_TOOL_BETWEEN_INVOKES;
    ts->parse_pos = pos;
    if (raw_full_lit(raw, raw_len, pos, DS4_TOOL_CALLS_START)) {
        ts->parse_pos += strlen(DS4_TOOL_CALLS_START);
        ts->tool_calls_end = DS4_TOOL_CALLS_END;
        ts->invoke_start = DS4_INVOKE_START;
        ts->invoke_end = DS4_INVOKE_END;
        ts->param_start = DS4_PARAM_START;
        ts->param_end = DS4_PARAM_END;
    } else if (raw_full_lit(raw, raw_len, pos, DS4_TOOL_CALLS_START_SHORT)) {
        ts->parse_pos += strlen(DS4_TOOL_CALLS_START_SHORT);
        ts->tool_calls_end = DS4_TOOL_CALLS_END_SHORT;
        ts->invoke_start = DS4_INVOKE_START_SHORT;
        ts->invoke_end = DS4_INVOKE_END_SHORT;
        ts->param_start = DS4_PARAM_START_SHORT;
        ts->param_end = DS4_PARAM_END_SHORT;
    } else if (raw_full_lit(raw, raw_len, pos, "<tool_calls>")) {
        ts->parse_pos += strlen("<tool_calls>");
        ts->tool_calls_end = "</tool_calls>";
        ts->invoke_start = "<invoke";
        ts->invoke_end = "</invoke>";
        ts->param_start = "<parameter";
        ts->param_end = "</parameter>";
    } else {
        ts->active = false;
        ts->state = DSML_TOOL_ERROR;
        return false;
    }
    return true;
}

static bool openai_tool_stream_fail(openai_tool_stream *ts) {
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return true;
}

static bool openai_tool_start_invoke(int fd, server *s, const request *r, const char *id,
                                     openai_tool_stream *ts,
                                     const char *raw, size_t raw_len) {
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    free(tag);
    if (!name) return openai_tool_stream_fail(ts);

    const char *tool_id = openai_tool_stream_id(s, ts, ts->index);
    bool ok = sse_chat_tool_call_start_delta(fd, r, id, ts->index, tool_id, name) &&
              openai_tool_emit_args_fragment(fd, r, id, ts, "{", 1);
    free(name);
    if (!ok) return false;

    ts->emitted_any = true;
    ts->args_open = true;
    ts->first_param = true;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}

static bool openai_tool_start_param(int fd, const request *r, const char *id,
                                    openai_tool_stream *ts,
                                    const char *raw, size_t raw_len) {
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    char *is_string = dsml_attr(tag, "string");
    free(tag);
    if (!name || !is_string) {
        free(name);
        free(is_string);
        return openai_tool_stream_fail(ts);
    }
    bool string_value = !strcmp(is_string, "true");
    bool ok = openai_tool_emit_param_prefix(fd, r, id, ts, name, string_value);
    free(name);
    free(is_string);
    if (!ok) return false;

    ts->param_is_string = string_value;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_PARAM_VALUE;
    return true;
}

static bool openai_tool_finish_param(int fd, const request *r, const char *id,
                                     openai_tool_stream *ts,
                                     const char *raw, size_t value_end) {
    if (value_end > ts->parse_pos) {
        bool ok = ts->param_is_string ?
            openai_tool_emit_string_value(fd, r, id, ts, raw + ts->parse_pos,
                                          value_end - ts->parse_pos) :
            openai_tool_emit_args_fragment(fd, r, id, ts, raw + ts->parse_pos,
                                           value_end - ts->parse_pos);
        if (!ok) return false;
    }
    if (ts->param_is_string &&
        !openai_tool_emit_args_fragment(fd, r, id, ts, "\"", 1)) return false;
    ts->parse_pos = value_end + strlen(ts->param_end);
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}

static bool openai_tool_stream_update(int fd, server *s, const request *r, const char *id,
                                      openai_tool_stream *ts,
                                      const char *raw, size_t raw_len) {
    while (ts->active && ts->parse_pos < raw_len) {
        if (ts->state == DSML_TOOL_BETWEEN_INVOKES) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->tool_calls_end)) {
                ts->parse_pos += strlen(ts->tool_calls_end);
                ts->active = false;
                ts->state = DSML_TOOL_DONE;
                return true;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, ts->tool_calls_end, ts->invoke_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->invoke_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!openai_tool_start_invoke(fd, s, r, id, ts, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return openai_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_BETWEEN_PARAMS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->invoke_end)) {
                if (ts->args_open &&
                    !openai_tool_emit_args_fragment(fd, r, id, ts, "}", 1)) return false;
                ts->args_open = false;
                ts->parse_pos += strlen(ts->invoke_end);
                ts->index++;
                ts->state = DSML_TOOL_BETWEEN_INVOKES;
                continue;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos, ts->invoke_end, ts->param_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->param_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!openai_tool_start_param(fd, r, id, ts, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return openai_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_PARAM_VALUE) {
            const char *end = find_lit_bounded(raw + ts->parse_pos,
                                               raw_len - ts->parse_pos,
                                               ts->param_end);
            if (end) {
                if (!openai_tool_finish_param(fd, r, id, ts, raw,
                                              (size_t)(end - raw))) return false;
                continue;
            }
            size_t limit = tool_param_value_stream_safe_len(raw, ts->parse_pos,
                                                            raw_len, ts->param_end,
                                                            ts->param_is_string);
            if (limit > ts->parse_pos) {
                bool ok = ts->param_is_string ?
                    openai_tool_emit_string_value(fd, r, id, ts, raw + ts->parse_pos,
                                                  limit - ts->parse_pos) :
                    openai_tool_emit_args_fragment(fd, r, id, ts, raw + ts->parse_pos,
                                                   limit - ts->parse_pos);
                if (!ok) return false;
                ts->parse_pos = limit;
            }
            return true;
        }

        return true;
    }
    return true;
}

static bool openai_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                     openai_stream *st,
                                     const char *raw, size_t raw_len,
                                     bool final) {
    if (!st->active || !raw) return true;

    if (st->mode == OPENAI_STREAM_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = "<think>";
            const size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) {
                return true;
            }
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) {
                st->emit_pos = open_len;
            }
            st->checked_think_prefix = true;
        }

        const char *close = strstr(raw + st->emit_pos, "</think>");
        size_t limit;
        if (close) {
            limit = (size_t)(close - raw);
        } else if (final) {
            limit = raw_len;
        } else {
            const size_t hold = strlen("</think>") - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }

        if (limit > st->emit_pos) {
            if (!sse_chat_delta_n(fd, r, id, "reasoning_content",
                                  raw + st->emit_pos,
                                  limit - st->emit_pos)) return false;
            st->sent_reasoning = true;
            st->emit_pos = limit;
        }

        if (close) {
            st->emit_pos = (size_t)(close - raw) + strlen("</think>");
            st->mode = OPENAI_STREAM_TEXT;
        } else if (final) {
            st->mode = OPENAI_STREAM_SUPPRESS;
            return true;
        } else {
            return true;
        }
    }

    if (st->mode == OPENAI_STREAM_TEXT) {
        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!sse_chat_delta_n(fd, r, id, "content",
                                  raw + st->emit_pos,
                                  limit - st->emit_pos)) return false;
            st->sent_content = true;
            st->emit_pos = limit;
        }

        if (tool) {
            st->emit_pos = (size_t)(tool - raw);
            if (openai_tool_stream_init(&st->tool, raw, raw_len, st->emit_pos)) {
                st->mode = OPENAI_STREAM_TOOL;
            } else {
                st->mode = OPENAI_STREAM_SUPPRESS;
            }
        } else if (final) {
            st->mode = OPENAI_STREAM_SUPPRESS;
        }
    }

    if (st->mode == OPENAI_STREAM_TOOL) {
        if (!openai_tool_stream_update(fd, s, r, id, &st->tool, raw, raw_len)) return false;
        if (!st->tool.active) st->mode = OPENAI_STREAM_SUPPRESS;
    }
    return true;
}

static bool openai_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                   openai_stream *st, const char *raw,
                                   size_t raw_len, const tool_calls *calls,
                                   const char *finish, int prompt_tokens,
                                   int completion_tokens) {
    if (!openai_sse_stream_update(fd, s, r, id, st, raw, raw_len, true)) return false;

    buf b = {0};
    long now = (long)time(NULL);
    if (calls && calls->len && !st->tool.emitted_any) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
        append_tool_call_deltas_json(&b, calls, id, &r->tool_orders);
        buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    }
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, now);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    json_escape(&b, finish);
    buf_puts(&b, "}]}\n\n");

    bool ok = send_all(fd, b.ptr, b.len) &&
              sse_done(fd, r, id, prompt_tokens, completion_tokens);
    buf_free(&b);
    return ok;
}

static bool request_uses_openai_live_stream(const request *r) {
    return r->stream && r->api == API_OPENAI && r->kind == REQ_CHAT;
}

static bool request_uses_responses_live_stream(const request *r) {
    return r->stream && r->api == API_RESPONSES && r->kind == REQ_CHAT;
}

static bool request_uses_structured_stream(const request *r) {
    return r->stream && (r->api == API_ANTHROPIC ||
                         r->api == API_RESPONSES ||
                         request_uses_openai_live_stream(r));
}

/* Codex' Responses API uses 24-hex suffixes for response/item ids. Prefix
 * controls the variant (resp_, rs_, msg_, fc_) so each event references a
 * stable identifier across output_item.added / .done. */
static void responses_random_id(char *dst, size_t dstlen, const char *prefix) {
    unsigned char bytes[12];
    size_t pos = snprintf(dst, dstlen, "%s", prefix);
    if (pos >= dstlen) return;
    static uint64_t fallback_ctr;
    if (!random_bytes(bytes, sizeof(bytes))) {
        uint64_t a = ((uint64_t)time(NULL) << 32) ^ (uint64_t)getpid();
        uint64_t b = ++fallback_ctr ^ (uint64_t)(uintptr_t)dst;
        memcpy(bytes, &a, sizeof(a));
        memcpy(bytes + sizeof(a), &b, sizeof(uint32_t));
    }
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(bytes) && pos + 2 < dstlen; i++) {
        dst[pos++] = hex[bytes[i] >> 4];
        dst[pos++] = hex[bytes[i] & 15];
    }
    dst[pos] = '\0';
}

typedef enum {
    RESP_STREAM_THINKING,
    RESP_STREAM_TEXT,
    RESP_STREAM_SUPPRESS,
} responses_stream_mode;

typedef struct {
    responses_stream_mode mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool reasoning_item_opened;
    bool reasoning_item_closed;
    bool reasoning_summary_started;
    bool reasoning_closed_naturally;
    bool message_item_opened;
    bool message_text_part_open;
    bool message_item_closed;
    bool reasoning_emitted_any;
    bool message_emitted_any;
    buf reasoning_text;
    buf message_text;
    char response_id[40];
    char reasoning_id[40];
    char message_id[40];
    int reasoning_index;   /* output_index of the reasoning item (0 if present) */
    int message_index;     /* output_index of the assistant message item */
    int next_output_index; /* monotonic counter for upcoming output items */
    int sequence;          /* monotonic per-event sequence_number Codex consumes */
} responses_stream;

static void responses_stream_init(const request *r, responses_stream *st) {
    memset(st, 0, sizeof(*st));
    st->mode = ds4_think_mode_enabled(r->think_mode) ? RESP_STREAM_THINKING : RESP_STREAM_TEXT;
    responses_random_id(st->response_id, sizeof(st->response_id), "resp_");
    responses_random_id(st->reasoning_id, sizeof(st->reasoning_id), "rs_");
    responses_random_id(st->message_id, sizeof(st->message_id), "msg_");
    st->reasoning_index = -1;
    st->message_index = -1;
}

static void responses_stream_free(responses_stream *st) {
    if (!st) return;
    buf_free(&st->reasoning_text);
    buf_free(&st->message_text);
}

/* Codex parses an explicit sequence_number on every Responses event for
 * ordering and reconnect resilience. We inject it after the `{"type":...` head
 * so emitters can stay readable while still producing the wire shape Codex
 * expects. */
static bool responses_sse_emit_event(int fd, responses_stream *st, const char *body) {
    buf b = {0};
    buf_puts(&b, "data: ");
    /* body always starts with `{"type":"..."`. We splice in sequence_number
     * after the closing quote of that string so every event has it as the
     * second field. */
    const char *type_close = NULL;
    if (body[0] == '{') {
        const char *p = body + 1;
        /* Skip the literal `"type":` then the value string. */
        if (!strncmp(p, "\"type\":\"", 8)) {
            const char *q = p + 8;
            while (*q && *q != '"') {
                if (*q == '\\' && q[1]) q += 2;
                else q++;
            }
            if (*q == '"') type_close = q + 1;
        }
    }
    if (type_close) {
        size_t head_len = (size_t)(type_close - body);
        buf_append(&b, body, head_len);
        buf_printf(&b, ",\"sequence_number\":%d", st->sequence++);
        buf_puts(&b, type_close);
    } else {
        buf_puts(&b, body);
    }
    buf_puts(&b, "\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

static bool responses_sse_created(int fd, const request *r, responses_stream *st,
                                  long created_at) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.created\",\"response\":{\"id\":\"%s\","
        "\"object\":\"response\",\"created_at\":%ld,\"status\":\"in_progress\","
        "\"model\":", st->response_id, created_at);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"output\":[]}}");
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_reasoning_added(int fd, responses_stream *st) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"in_progress\","
        "\"summary\":[]}}",
        st->reasoning_index, st->reasoning_id);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_reasoning_summary_part_added(int fd, responses_stream *st) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_part.added\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,"
        "\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}",
        st->reasoning_id, st->reasoning_index);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_reasoning_delta(int fd, responses_stream *st,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_text.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"delta\":",
        st->reasoning_id, st->reasoning_index);
    json_escape_n(&b, text, len);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

static const char *responses_item_status_for_finish(const char *finish) {
    if (finish && (!strcmp(finish, "length") || !strcmp(finish, "error"))) return "incomplete";
    return "completed";
}

static bool responses_sse_reasoning_done(int fd, responses_stream *st,
                                         const char *finish) {
    /* If the stream terminates before `</think>` was actually observed the
     * reasoning item is partial — regardless of why generation stopped (EOS,
     * stop sequence, tool_calls, length, error). Force the item to incomplete
     * so a client replay rejects it instead of feeding unfinished hidden state
     * back as completed history. */
    (void)finish;
    const char *item_status =
        st->reasoning_closed_naturally ? "completed" : "incomplete";
    /* Mirror the message-item close sequence: emit summary_text.done +
     * summary_part.done before the output_item.done so clients that key off
     * part lifecycle don't see a dangling open summary part. */
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.reasoning_summary_text.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"text\":",
        st->reasoning_id, st->reasoning_index);
    json_escape_n(&b, st->reasoning_text.ptr ? st->reasoning_text.ptr : "",
                  st->reasoning_text.len);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        buf_free(&b);
        return false;
    }

    if (st->reasoning_summary_started) {
        buf_free(&b);
        buf_printf(&b,
            "{\"type\":\"response.reasoning_summary_part.done\","
            "\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,"
            "\"part\":{\"type\":\"summary_text\",\"text\":",
            st->reasoning_id, st->reasoning_index);
        json_escape_n(&b, st->reasoning_text.ptr ? st->reasoning_text.ptr : "",
                      st->reasoning_text.len);
        buf_puts(&b, "}}");
        ok = responses_sse_emit_event(fd, st, b.ptr);
        if (!ok) {
            buf_free(&b);
            return false;
        }
    }

    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\",\"summary\":[",
        st->reasoning_index, st->reasoning_id, item_status);
    if (st->reasoning_text.len) {
        buf_puts(&b, "{\"type\":\"summary_text\",\"text\":");
        json_escape_n(&b, st->reasoning_text.ptr, st->reasoning_text.len);
        buf_putc(&b, '}');
    }
    buf_puts(&b, "]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_message_added(int fd, responses_stream *st) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_item.added\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"in_progress\","
        "\"role\":\"assistant\",\"content\":[]}}",
        st->message_index, st->message_id);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_message_text_part_added(int fd, responses_stream *st) {
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.content_part.added\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}",
        st->message_id, st->message_index);
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_output_text_delta(int fd, responses_stream *st,
                                            const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_text.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"delta\":",
        st->message_id, st->message_index);
    json_escape_n(&b, text, len);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

static bool responses_sse_message_done(int fd, responses_stream *st,
                                       const char *finish) {
    const char *item_status = responses_item_status_for_finish(finish);
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_text.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"text\":",
        st->message_id, st->message_index);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        buf_free(&b);
        return false;
    }

    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"response.content_part.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,"
        "\"part\":{\"type\":\"output_text\",\"text\":",
        st->message_id, st->message_index);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    buf_puts(&b, ",\"annotations\":[]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        buf_free(&b);
        return false;
    }

    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"response.output_item.done\",\"output_index\":%d,"
        "\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
        "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
        st->message_index, st->message_id, item_status);
    json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                  st->message_text.len);
    buf_puts(&b, ",\"annotations\":[]}]}}");
    ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

/* Item identity per tool call must be stable across added/done/completed. */
typedef struct {
    char fc_id[40];
    char call_id[64];
    bool is_custom;
    int output_index;
} responses_tool_item;

static bool responses_tool_call_is_tool_search(const tool_call *tc,
                                               const tool_schema_order *order) {
    return tc && tc->name && !strcmp(tc->name, "tool_search") &&
           (!order || order->responses_tool_search);
}

/* The internal tool_call doesn't track whether it came from a function_call or
 * a custom_tool_call (or what tool kind is registered). For round-trip
 * correctness with the rare custom_tool_call clients, we preserve any provided
 * call_id verbatim and pre-assign a stable fc_id; the discriminator currently
 * defaults to function_call because Codex CLI registers all its tools as
 * function tools. */
static void responses_tool_items_build(responses_tool_item **out,
                                       const tool_calls *calls,
                                       int starting_output_index) {
    *out = NULL;
    if (!calls || calls->len == 0) return;
    responses_tool_item *items = xmalloc((size_t)calls->len * sizeof(*items));
    for (int i = 0; i < calls->len; i++) {
        memset(&items[i], 0, sizeof(items[i]));
        responses_random_id(items[i].fc_id, sizeof(items[i].fc_id), "fc_");
        if (calls->v[i].id && calls->v[i].id[0]) {
            snprintf(items[i].call_id, sizeof(items[i].call_id), "%s", calls->v[i].id);
        } else {
            responses_random_id(items[i].call_id, sizeof(items[i].call_id), "call_");
        }
        items[i].is_custom = false;
        items[i].output_index = starting_output_index + i;
    }
    *out = items;
}

static void responses_append_function_call_item(buf *b, const tool_call *tc,
                                                const responses_tool_item *item,
                                                const char *item_status,
                                                bool with_args,
                                                const tool_schema_orders *orders) {
    const tool_schema_order *order = tool_schema_orders_find(orders, tc->name);
    if (responses_tool_call_is_tool_search(tc, order)) {
        buf_printf(b,
            "{\"id\":\"%s\",\"type\":\"tool_search_call\",\"status\":\"%s\","
            "\"call_id\":\"%s\",\"execution\":\"client\",\"arguments\":",
            item->fc_id, item_status, item->call_id);
        if (with_args) append_json_object_or_empty(b, tc->arguments);
        else buf_puts(b, "{}");
        buf_putc(b, '}');
        return;
    }

    const char *item_type = item->is_custom ? "custom_tool_call" : "function_call";
    const char *body_field = item->is_custom ? "input" : "arguments";
    buf_printf(b,
        "{\"id\":\"%s\",\"type\":\"%s\",\"status\":\"%s\",\"name\":",
        item->fc_id, item_type, item_status);
    json_escape(b, order && order->wire_name ? order->wire_name :
                   (tc->name ? tc->name : ""));
    if (order && order->namespace) {
        buf_puts(b, ",\"namespace\":");
        json_escape(b, order->namespace);
    }
    buf_puts(b, ",\"call_id\":");
    json_escape(b, item->call_id);
    buf_printf(b, ",\"%s\":", body_field);
    if (!with_args) {
        buf_puts(b, "\"\"");
    } else if (item->is_custom) {
        json_escape(b, tc->arguments ? tc->arguments : "");
    } else {
        append_json_object_string(b, tc->arguments);
    }
    buf_putc(b, '}');
}

static bool responses_sse_function_call_event(int fd, responses_stream *st,
                                              const tool_call *tc,
                                              const responses_tool_item *item,
                                              const tool_schema_orders *orders,
                                              const char *finish,
                                              bool done) {
    /* The added event marks a tool call as in_progress per the Responses
     * lifecycle; only output_item.done (and the terminal response output)
     * carry the final completed / incomplete status. The added item ships with
     * an empty arguments string so clients that accumulate via
     * function_call_arguments.delta + .done don't end up with doubled JSON. */
    const char *item_status = done ? responses_item_status_for_finish(finish) : "in_progress";
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.output_item.%s\",\"output_index\":%d,\"item\":",
        done ? "done" : "added", item->output_index);
    responses_append_function_call_item(&b, tc, item, item_status, done, orders);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

/* Stream function-call arguments as a single delta + done, since DS4 generates
 * the whole DSML invoke as one unit before the worker decides which tool was
 * called. Clients that follow the OpenAI Responses lifecycle expect both
 * events between output_item.added (in_progress) and output_item.done. */
static bool responses_sse_function_call_arguments_done(int fd, responses_stream *st,
                                                       const tool_call *tc,
                                                       const responses_tool_item *item,
                                                       const tool_schema_orders *orders) {
    const tool_schema_order *order = tool_schema_orders_find(orders, tc->name);
    if (item->is_custom || responses_tool_call_is_tool_search(tc, order)) return true;
    buf args = {0};
    append_json_object_string(&args, tc->arguments);
    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"response.function_call_arguments.delta\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"delta\":",
        item->fc_id, item->output_index);
    buf_append(&b, args.ptr ? args.ptr : "\"\"", args.ptr ? args.len : 2);
    buf_putc(&b, '}');
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    if (!ok) {
        buf_free(&b);
        buf_free(&args);
        return false;
    }

    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"response.function_call_arguments.done\","
        "\"item_id\":\"%s\",\"output_index\":%d,\"name\":",
        item->fc_id, item->output_index);
    json_escape(&b, order && order->wire_name ? order->wire_name :
                    (tc->name ? tc->name : ""));
    if (order && order->namespace) {
        buf_puts(&b, ",\"namespace\":");
        json_escape(&b, order->namespace);
    }
    buf_puts(&b, ",\"arguments\":");
    buf_append(&b, args.ptr ? args.ptr : "\"\"", args.ptr ? args.len : 2);
    buf_putc(&b, '}');
    ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    buf_free(&args);
    return ok;
}

static const char *responses_status_for_finish(const char *finish) {
    if (finish && !strcmp(finish, "length")) return "incomplete";
    if (finish && !strcmp(finish, "error")) return "failed";
    return "completed";
}

static void append_responses_usage_json(buf *b, const request *r,
                                        int input_tokens, int output_tokens) {
    int cached_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    cached_tokens = clamp_usage_tokens(cached_tokens, input_tokens);
    cache_write_tokens = clamp_usage_tokens(cache_write_tokens, input_tokens - cached_tokens);
    buf_printf(b,
        "{\"input_tokens\":%d,\"input_tokens_details\":{\"cached_tokens\":%d,\"cache_write_tokens\":%d},"
        "\"output_tokens\":%d,\"output_tokens_details\":{\"reasoning_tokens\":0},"
        "\"total_tokens\":%d}",
        input_tokens, cached_tokens, cache_write_tokens,
        output_tokens, input_tokens + output_tokens);
}

static bool responses_sse_completed(int fd, const request *r,
                                    responses_stream *st,
                                    const tool_calls *calls,
                                    const responses_tool_item *tool_items,
                                    const char *finish,
                                    int prompt_tokens, int completion_tokens,
                                    long created_at) {
    /* Codex routes terminal behaviour off the event type, not response.status.
     * Decide here so clients see response.failed / response.incomplete instead
     * of a "completed" wrapper marked failed in a sub-field. */
    const char *event_type = "response.completed";
    if (finish && !strcmp(finish, "error")) event_type = "response.failed";
    else if (finish && !strcmp(finish, "length")) event_type = "response.incomplete";
    const char *status = responses_status_for_finish(finish);

    buf b = {0};
    buf_printf(&b,
        "{\"type\":\"%s\",\"response\":{\"id\":\"%s\","
        "\"object\":\"response\",\"created_at\":%ld,\"status\":\"%s\",\"model\":",
        event_type, st->response_id, created_at, status);
    json_escape(&b, r->model);
    if (!strcmp(event_type, "response.failed")) {
        buf_puts(&b, ",\"error\":{\"code\":\"server_error\","
                     "\"message\":\"generation failed\"}");
    } else if (!strcmp(event_type, "response.incomplete")) {
        buf_puts(&b, ",\"incomplete_details\":{\"reason\":\"max_tokens\"}");
    }
    const char *item_status = responses_item_status_for_finish(finish);
    buf_puts(&b, ",\"output\":[");
    bool wrote = false;
    if (st->reasoning_emitted_any) {
        /* Match responses_sse_reasoning_done: if the stream stopped before
         * </think>, the reasoning item is partial regardless of the
         * response-level finish status, so replay must reject it. */
        const char *reasoning_status =
            st->reasoning_closed_naturally ? "completed" : "incomplete";
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\",\"summary\":[",
            st->reasoning_id, reasoning_status);
        if (st->reasoning_text.len) {
            buf_puts(&b, "{\"type\":\"summary_text\",\"text\":");
            json_escape_n(&b, st->reasoning_text.ptr, st->reasoning_text.len);
            buf_putc(&b, '}');
        }
        buf_puts(&b, "]}");
        wrote = true;
    }
    if (st->message_emitted_any) {
        if (wrote) buf_putc(&b, ',');
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
            "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
            st->message_id, item_status);
        json_escape_n(&b, st->message_text.ptr ? st->message_text.ptr : "",
                      st->message_text.len);
        buf_puts(&b, ",\"annotations\":[]}]}");
        wrote = true;
    }
    if (calls && tool_items) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(&b, ',');
            responses_append_function_call_item(&b, &calls->v[i], &tool_items[i],
                                                item_status, true,
                                                &r->tool_orders);
            wrote = true;
        }
    }
    buf_putc(&b, ']');
    buf_puts(&b, ",\"usage\":");
    append_responses_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}}");
    bool ok = responses_sse_emit_event(fd, st, b.ptr);
    buf_free(&b);
    return ok;
}

/* Responses streaming consumes the same raw token text the OpenAI live stream
 * consumes: <think>...</think> is reasoning, anything before the tool-call
 * marker is output text. Tool-call argument deltas are not surfaced because
 * Codex' SSE parser only ingests function_call items via output_item.done. */
static bool responses_sse_stream_update(int fd, const request *r,
                                        responses_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final) {
    if (!st->active || !raw) return true;

    /* The client only sees reasoning if it explicitly opted in via
     * reasoning.summary. Otherwise we still need to walk past <think>...</think>
     * to find the user-visible text, but we suppress the per-chunk emission. */
    const bool emit_reasoning = r->reasoning_summary_emit;

    if (st->mode == RESP_STREAM_THINKING) {
        if (!st->checked_think_prefix) {
            /* The chat template ends the prompt with the literal `<think>` (or
             * `</think>` when thinking is off), so generation usually starts
             * mid-reasoning rather than with the open tag. If the model does
             * happen to repeat `<think>` we skip it; otherwise start from
             * position 0. The earlier "no-think-prefix => switch to TEXT"
             * shortcut here was incorrect: it leaked reasoning to clients as
             * regular output_text because the model was already inside the
             * think block when it produced its first token. The actual
             * mode change to TEXT happens only when `</think>` is observed. */
            const char *open = "<think>";
            const size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) {
                return true;
            }
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) {
                st->emit_pos = open_len;
            }
            st->checked_think_prefix = true;
        }

        const char *close = strstr(raw + st->emit_pos, "</think>");
        size_t limit;
        if (close) {
            limit = (size_t)(close - raw);
        } else if (final) {
            limit = raw_len;
        } else {
            const size_t hold = strlen("</think>") - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }

        if (limit > st->emit_pos) {
            if (emit_reasoning) {
                if (!st->reasoning_item_opened) {
                    st->reasoning_index = st->next_output_index++;
                    if (!responses_sse_reasoning_added(fd, st)) return false;
                    st->reasoning_item_opened = true;
                }
                if (!st->reasoning_summary_started) {
                    if (!responses_sse_reasoning_summary_part_added(fd, st)) return false;
                    st->reasoning_summary_started = true;
                }
                if (!responses_sse_reasoning_delta(fd, st,
                                                   raw + st->emit_pos,
                                                   limit - st->emit_pos)) return false;
                buf_append(&st->reasoning_text, raw + st->emit_pos, limit - st->emit_pos);
                st->reasoning_emitted_any = true;
            }
            st->emit_pos = limit;
        }

        if (close) {
            st->emit_pos = (size_t)(close - raw) + strlen("</think>");
            st->mode = RESP_STREAM_TEXT;
            st->reasoning_closed_naturally = true;
        } else if (final) {
            st->mode = RESP_STREAM_SUPPRESS;
            return true;
        } else {
            return true;
        }
    }

    if (st->mode == RESP_STREAM_TEXT) {
        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!st->message_item_opened) {
                st->message_index = st->next_output_index++;
                if (!responses_sse_message_added(fd, st)) return false;
                st->message_item_opened = true;
            }
            if (!st->message_text_part_open) {
                if (!responses_sse_message_text_part_added(fd, st)) return false;
                st->message_text_part_open = true;
            }
            if (!responses_sse_output_text_delta(fd, st,
                                                 raw + st->emit_pos,
                                                 limit - st->emit_pos)) return false;
            buf_append(&st->message_text, raw + st->emit_pos, limit - st->emit_pos);
            st->message_emitted_any = true;
            st->emit_pos = limit;
        }

        if (tool) {
            st->emit_pos = (size_t)(tool - raw);
            st->mode = RESP_STREAM_SUPPRESS;
        } else if (final) {
            st->mode = RESP_STREAM_SUPPRESS;
        }
    }
    return true;
}

static bool responses_sse_finish_live(int fd, const request *r,
                                      responses_stream *st,
                                      const char *raw, size_t raw_len,
                                      const char *recovered_content,
                                      const tool_calls *calls,
                                      const char *finish,
                                      int prompt_tokens, int completion_tokens,
                                      long created_at) {
    if (!responses_sse_stream_update(fd, r, st, raw, raw_len, true)) return false;

    /* Close any half-open reasoning summary so the TUI knows the part ended
     * before we slot in any tool calls or completion. */
    if (st->reasoning_item_opened && !st->reasoning_item_closed) {
        if (!responses_sse_reasoning_done(fd, st, finish)) return false;
        st->reasoning_item_closed = true;
    }
    /* Recovery path: when DSML tool parsing fails the worker promotes the entire
     * generation to assistant text. Streaming had already entered suppress mode
     * at the tool marker, so anything in raw[st->emit_pos..raw_len] never made
     * it to the client. Emit those bytes as additional output_text deltas so
     * what the client accumulates matches output_item.done and the terminal
     * response. We use the stream cursor instead of comparing against
     * recovered_content because the raw text can begin with `<think>...</think>`
     * which the streaming side consumed as reasoning, not message text. */
    if (recovered_content && raw && st->emit_pos < raw_len) {
        const char *tail = raw + st->emit_pos;
        size_t tail_len = raw_len - st->emit_pos;
        if (!st->message_item_opened) {
            st->message_index = st->next_output_index++;
            if (!responses_sse_message_added(fd, st)) return false;
            st->message_item_opened = true;
        }
        if (!st->message_text_part_open) {
            if (!responses_sse_message_text_part_added(fd, st)) return false;
            st->message_text_part_open = true;
        }
        if (!responses_sse_output_text_delta(fd, st, tail, tail_len)) return false;
        buf_append(&st->message_text, tail, tail_len);
        st->message_emitted_any = true;
        st->emit_pos = raw_len;
    }
    if (st->message_item_opened && !st->message_item_closed) {
        if (!responses_sse_message_done(fd, st, finish)) return false;
        st->message_item_closed = true;
    }
    responses_tool_item *items = NULL;
    responses_tool_items_build(&items, calls, st->next_output_index);
    if (items && calls) st->next_output_index += calls->len;
    bool ok = true;
    if (items && calls) {
        for (int i = 0; i < calls->len && ok; i++) {
            ok = responses_sse_function_call_event(fd, st, &calls->v[i], &items[i],
                                                   &r->tool_orders, finish, false);
            if (ok) ok = responses_sse_function_call_arguments_done(fd, st, &calls->v[i],
                                                                    &items[i],
                                                                    &r->tool_orders);
            if (ok) ok = responses_sse_function_call_event(fd, st, &calls->v[i], &items[i],
                                                           &r->tool_orders, finish, true);
        }
    }
    if (ok) ok = responses_sse_completed(fd, r, st, calls, items, finish,
                                         prompt_tokens, completion_tokens, created_at);
    free(items);
    return ok;
}

static bool responses_final_response(int fd, bool enable_cors,
                                     const request *r, const char *id,
                                     const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens) {
    (void)id;
    char response_id[40], reasoning_id[40], message_id[40];
    responses_random_id(response_id, sizeof(response_id), "resp_");
    responses_random_id(reasoning_id, sizeof(reasoning_id), "rs_");
    responses_random_id(message_id, sizeof(message_id), "msg_");

    responses_tool_item *items = NULL;
    responses_tool_items_build(&items, calls, 0);

    long now = (long)time(NULL);
    const char *status = responses_status_for_finish(finish);
    const char *item_status = responses_item_status_for_finish(finish);
    buf b = {0};
    buf_printf(&b,
        "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,\"status\":\"%s\","
        "\"model\":",
        response_id, now, status);
    json_escape(&b, r->model);
    if (finish && !strcmp(finish, "error")) {
        buf_puts(&b, ",\"error\":{\"code\":\"server_error\","
                     "\"message\":\"generation failed\"}");
    } else if (finish && !strcmp(finish, "length")) {
        buf_puts(&b, ",\"incomplete_details\":{\"reason\":\"max_tokens\"}");
    }
    buf_puts(&b, ",\"output\":[");
    bool wrote = false;
    if (reasoning && reasoning[0] && r->reasoning_summary_emit) {
        /* Non-streaming path runs after the worker has post-processed the
         * generation, so any reasoning here came from a parsed assistant turn
         * where </think> was observed (otherwise the reasoning text would be
         * empty). Tag it with the response-level item_status which still flips
         * to incomplete/failed when finish is length/error. */
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\","
            "\"summary\":[{\"type\":\"summary_text\",\"text\":",
            reasoning_id, item_status);
        json_escape(&b, reasoning);
        buf_puts(&b, "}]}");
        wrote = true;
    }
    if (text && text[0]) {
        if (wrote) buf_putc(&b, ',');
        buf_printf(&b,
            "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\","
            "\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
            message_id, item_status);
        json_escape(&b, text);
        buf_puts(&b, ",\"annotations\":[]}]}");
        wrote = true;
    }
    if (calls && items) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(&b, ',');
            responses_append_function_call_item(&b, &calls->v[i], &items[i],
                                                item_status, true,
                                                &r->tool_orders);
            wrote = true;
        }
    }
    buf_putc(&b, ']');
    buf_puts(&b, ",\"usage\":");
    append_responses_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_putc(&b, '}');
    bool ok = http_response(fd, enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    free(items);
    return ok;
}

static bool final_response(int fd, bool enable_cors,
                           const request *r, const char *id, const char *text,
                           const char *reasoning, const tool_calls *calls, const char *finish,
                           int prompt_tokens, int completion_tokens) {
    buf b = {0};
    long now = (long)time(NULL);
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        json_escape(&b, text ? text : "");
        if (reasoning && reasoning[0]) {
            buf_puts(&b, ",\"reasoning_content\":");
            json_escape(&b, reasoning);
        }
        if (calls && calls->len) {
            buf_puts(&b, ",\"tool_calls\":");
            append_tool_calls_json(&b, calls, id, &r->tool_orders);
        }
        buf_puts(&b, "},\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    } else {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, now);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text);
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    }
    append_openai_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}\n");
    bool ok = http_response(fd, enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static const char *anthropic_stop_reason(const char *finish) {
    if (finish && !strcmp(finish, "tool_calls")) return "tool_use";
    if (finish && !strcmp(finish, "length")) return "max_tokens";
    return "end_turn";
}

static void append_anthropic_tool_use(buf *b, const tool_call *tc, const char *id_prefix, int i,
                                      const tool_schema_orders *orders) {
    (void)orders;
    char idbuf[128];
    snprintf(idbuf, sizeof(idbuf), "toolu_%s_%d", id_prefix, i);
    buf_puts(b, "{\"type\":\"tool_use\",\"id\":");
    json_escape(b, tc->id && tc->id[0] ? tc->id : idbuf);
    buf_puts(b, ",\"name\":");
    json_escape(b, tc->name ? tc->name : "");
    buf_puts(b, ",\"input\":");
    append_json_object_or_empty(b, tc->arguments);
    buf_putc(b, '}');
}

static void append_anthropic_thinking(buf *b, const char *reasoning, const char *signature) {
    buf_puts(b, "{\"type\":\"thinking\",\"thinking\":");
    json_escape(b, reasoning ? reasoning : "");
    buf_puts(b, ",\"signature\":");
    json_escape(b, signature ? signature : "");
    buf_putc(b, '}');
}

static void append_anthropic_content(buf *b, const char *text, const char *reasoning,
                                     const tool_calls *calls, const char *id_prefix,
                                     const tool_schema_orders *orders) {
    buf_putc(b, '[');
    bool wrote = false;
    bool wrote_after_thinking = false;
    if (reasoning && reasoning[0]) {
        append_anthropic_thinking(b, reasoning, id_prefix);
        wrote = true;
    }
    if (text && text[0]) {
        if (wrote) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"text\",\"text\":");
        json_escape(b, text);
        buf_putc(b, '}');
        wrote = true;
        wrote_after_thinking = true;
    }
    if (calls) {
        for (int i = 0; i < calls->len; i++) {
            if (wrote) buf_putc(b, ',');
            append_anthropic_tool_use(b, &calls->v[i], id_prefix, i, orders);
            wrote = true;
            wrote_after_thinking = true;
        }
    }
    if (!wrote || ((reasoning && reasoning[0]) && !wrote_after_thinking)) {
        if (wrote) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"text\",\"text\":\"\"}");
    }
    buf_putc(b, ']');
}

static void append_anthropic_usage_json(buf *b, const request *r,
                                        int prompt_tokens, int completion_tokens) {
    int cache_read_tokens = r ? r->cache_read_tokens : 0;
    int cache_write_tokens = r ? r->cache_write_tokens : 0;
    cache_read_tokens = clamp_usage_tokens(cache_read_tokens, prompt_tokens);
    cache_write_tokens = clamp_usage_tokens(cache_write_tokens, prompt_tokens - cache_read_tokens);
    int input_tokens = prompt_tokens - cache_read_tokens - cache_write_tokens;
    if (input_tokens < 0) input_tokens = 0;
    buf_printf(b,
               "{\"input_tokens\":%d,\"output_tokens\":%d,"
               "\"cache_read_input_tokens\":%d,\"cache_creation_input_tokens\":%d}",
               input_tokens, completion_tokens, cache_read_tokens, cache_write_tokens);
}

static bool anthropic_final_response(int fd, bool enable_cors,
                                     const request *r, const char *id, const char *text,
                                     const char *reasoning, const tool_calls *calls, const char *finish,
                                     int prompt_tokens, int completion_tokens) {
    buf b = {0};
    buf_printf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", id);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"content\":");
    append_anthropic_content(&b, text, reasoning, calls, id, &r->tool_orders);
    buf_puts(&b, ",\"stop_reason\":");
    json_escape(&b, anthropic_stop_reason(finish));
    buf_puts(&b, ",\"stop_sequence\":null,\"usage\":");
    append_anthropic_usage_json(&b, r, prompt_tokens, completion_tokens);
    buf_puts(&b, "}\n");
    bool ok = http_response(fd, enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool sse_event(int fd, const char *event, const char *data) {
    buf b = {0};
    buf_puts(&b, "event: ");
    buf_puts(&b, event);
    buf_puts(&b, "\ndata: ");
    buf_puts(&b, data);
    buf_puts(&b, "\n\n");
    bool ok = send_all(fd, b.ptr, b.len);
    buf_free(&b);
    return ok;
}

typedef enum {
    ANTH_STREAM_THINKING,
    ANTH_STREAM_TEXT,
    ANTH_STREAM_TOOL,
    ANTH_STREAM_SUPPRESS,
} anthropic_stream_mode;

typedef enum {
    ANTH_BLOCK_NONE,
    ANTH_BLOCK_THINKING,
    ANTH_BLOCK_TEXT,
    ANTH_BLOCK_TOOL,
} anthropic_block_type;

typedef struct {
    dsml_tool_stream_state state;
    const dsml_syntax *syn;
    size_t parse_pos;
    int index;
    bool active;
    bool emitted_any;
    bool args_open;
    bool first_param;
    bool param_is_string;
    char **ids;
    int ids_cap;
} anthropic_tool_stream;

/* Anthropic streaming uses the same sampled DSML bytes that will later be
 * parsed and remembered for exact continuation.  This state is only a wire
 * projection: it turns an in-progress DSML block into content_block/tool_use
 * SSE events, and never rewrites the model-visible transcript or cache key. */
typedef struct {
    anthropic_stream_mode mode;
    anthropic_block_type open_block;
    int next_index;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
    bool sent_thinking;
    bool sent_text;
    anthropic_tool_stream tool;
} anthropic_stream;

static bool anthropic_sse_start_live(int fd, const request *r, const char *id,
                                     int prompt_tokens, anthropic_stream *st) {
    buf b = {0};
    json_escape(&b, r->model);
    char *model_json = buf_take(&b);

    buf_printf(&b,
        "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\","
        "\"role\":\"assistant\",\"model\":%s,\"content\":[],\"stop_reason\":null,"
        "\"stop_sequence\":null,\"usage\":",
        id, model_json);
    append_anthropic_usage_json(&b, r, prompt_tokens, 0);
    buf_puts(&b, "}}");
    bool ok = sse_event(fd, "message_start", b.ptr);
    buf_free(&b);
    free(model_json);

    memset(st, 0, sizeof(*st));
    st->active = ok;
    st->mode = ds4_think_mode_enabled(r->think_mode) ? ANTH_STREAM_THINKING : ANTH_STREAM_TEXT;
    return ok;
}

static void anthropic_tool_stream_free(anthropic_tool_stream *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->ids_cap; i++) free(ts->ids[i]);
    free(ts->ids);
    ts->ids = NULL;
    ts->ids_cap = 0;
}

static void anthropic_stream_free(anthropic_stream *st) {
    if (!st) return;
    anthropic_tool_stream_free(&st->tool);
}

static bool anthropic_tool_stream_has_id(const anthropic_tool_stream *ts,
                                         const char *id, int upto) {
    if (!ts || !id || !id[0]) return false;
    if (upto > ts->ids_cap) upto = ts->ids_cap;
    for (int i = 0; i < upto; i++) {
        if (ts->ids[i] && !strcmp(ts->ids[i], id)) return true;
    }
    return false;
}

static const char *anthropic_tool_stream_id(server *s, anthropic_tool_stream *ts,
                                            int index) {
    if (!ts || index < 0) return "";
    if (index >= ts->ids_cap) {
        int old = ts->ids_cap;
        int cap = old ? old : 4;
        while (cap <= index) cap *= 2;
        ts->ids = xrealloc(ts->ids, (size_t)cap * sizeof(ts->ids[0]));
        memset(ts->ids + old, 0, (size_t)(cap - old) * sizeof(ts->ids[0]));
        ts->ids_cap = cap;
    }
    if (!ts->ids[index]) {
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), API_ANTHROPIC);
            if (!anthropic_tool_stream_has_id(ts, id, index) &&
                !tool_memory_has_id(s, id)) break;
        }
        ts->ids[index] = xstrdup(id);
    }
    return ts->ids[index];
}

/* Text and thinking blocks have fixed JSON shapes.  Tool blocks are opened by
 * name later, after the DSML invoke tag is complete, so they use a dedicated
 * opener instead of this helper. */
static bool anthropic_sse_open_block(int fd, anthropic_stream *st,
                                     anthropic_block_type type) {
    if (st->open_block == type) return true;
    if (st->open_block != ANTH_BLOCK_NONE) return false;

    buf b = {0};
    if (type == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\","
                   "\"signature\":\"\"}}",
                   st->next_index);
    } else {
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
                   st->next_index);
    }
    bool ok = sse_event(fd, "content_block_start", b.ptr);
    buf_free(&b);
    if (ok) st->open_block = type;
    return ok;
}

static bool anthropic_sse_open_tool_block(int fd, anthropic_stream *st,
                                          const char *tool_id,
                                          const char *name) {
    if (st->open_block == ANTH_BLOCK_TOOL) return true;
    if (st->open_block != ANTH_BLOCK_NONE) return false;

    buf b = {0};
    buf_printf(&b,
               "{\"type\":\"content_block_start\",\"index\":%d,"
               "\"content_block\":{\"type\":\"tool_use\",\"id\":",
               st->next_index);
    json_escape(&b, tool_id ? tool_id : "");
    buf_puts(&b, ",\"name\":");
    json_escape(&b, name ? name : "");
    buf_puts(&b, ",\"input\":{}}}");
    bool ok = sse_event(fd, "content_block_start", b.ptr);
    buf_free(&b);
    if (ok) st->open_block = ANTH_BLOCK_TOOL;
    return ok;
}

static bool anthropic_sse_delta_live(int fd, const anthropic_stream *st,
                                     anthropic_block_type type,
                                     const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    if (type == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":",
                   st->next_index);
        json_escape_n(&b, text, len);
        buf_puts(&b, "}}");
    } else {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"text_delta\",\"text\":",
                   st->next_index);
        json_escape_n(&b, text, len);
        buf_puts(&b, "}}");
    }
    bool ok = sse_event(fd, "content_block_delta", b.ptr);
    buf_free(&b);
    return ok;
}

/* Anthropic's input_json_delta carries a fragment of a JSON object, encoded as
 * a JSON string.  We stream exactly the same object that the final DSML parser
 * will build: an opening "{", quoted keys, raw JSON values or escaped string
 * contents, and the closing "}". */
static bool anthropic_sse_tool_delta_live(int fd, const anthropic_stream *st,
                                          const char *text, size_t len) {
    if (len == 0) return true;
    buf b = {0};
    buf_printf(&b,
               "{\"type\":\"content_block_delta\",\"index\":%d,"
               "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
               st->next_index);
    json_escape_n(&b, text, len);
    buf_puts(&b, "}}");
    bool ok = sse_event(fd, "content_block_delta", b.ptr);
    buf_free(&b);
    return ok;
}

static bool anthropic_sse_close_block_live(int fd, const char *id,
                                           anthropic_stream *st) {
    if (st->open_block == ANTH_BLOCK_NONE) return true;

    buf b = {0};
    bool ok = true;
    if (st->open_block == ANTH_BLOCK_THINKING) {
        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"signature_delta\",\"signature\":",
                   st->next_index);
        json_escape(&b, id);
        buf_puts(&b, "}}");
        ok = sse_event(fd, "content_block_delta", b.ptr);
        buf_free(&b);
    }
    if (ok) {
        buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}",
                   st->next_index);
        ok = sse_event(fd, "content_block_stop", b.ptr);
        buf_free(&b);
    }
    if (ok) {
        st->open_block = ANTH_BLOCK_NONE;
        st->next_index++;
    }
    return ok;
}

static bool anthropic_tool_emit_args_fragment(int fd, anthropic_stream *st,
                                              const char *text, size_t len) {
    return anthropic_sse_tool_delta_live(fd, st, text, len);
}

static bool anthropic_tool_emit_string_value(int fd, anthropic_stream *st,
                                             const char *text, size_t len) {
    if (len == 0) return true;
    char *raw = xstrndup(text, len);
    char *unescaped = dsml_unescape_text(raw);
    buf frag = {0};
    json_escape_fragment_n(&frag, unescaped, strlen(unescaped));
    bool ok = anthropic_tool_emit_args_fragment(fd, st,
                                                frag.ptr ? frag.ptr : "",
                                                frag.len);
    buf_free(&frag);
    free(unescaped);
    free(raw);
    return ok;
}

static bool anthropic_tool_emit_param_prefix(int fd, anthropic_stream *st,
                                             const char *name, bool is_string) {
    anthropic_tool_stream *ts = &st->tool;
    buf frag = {0};
    if (ts->first_param) ts->first_param = false;
    else buf_putc(&frag, ',');
    json_escape(&frag, name ? name : "");
    buf_putc(&frag, ':');
    if (is_string) buf_putc(&frag, '"');
    bool ok = anthropic_tool_emit_args_fragment(fd, st,
                                                frag.ptr ? frag.ptr : "",
                                                frag.len);
    buf_free(&frag);
    return ok;
}

/* The parser below mirrors the OpenAI tool-delta parser but keeps Anthropic's
 * content-block lifecycle local.  A callback abstraction would save lines, but
 * it would hide the different block/stop semantics that make this code easy to
 * audit when a client reports a streaming regression. */
static bool anthropic_tool_stream_init(anthropic_tool_stream *ts,
                                       const char *raw, size_t raw_len,
                                       size_t pos) {
    anthropic_tool_stream_free(ts);
    memset(ts, 0, sizeof(*ts));
    ts->active = true;
    ts->state = DSML_TOOL_BETWEEN_INVOKES;
    for (size_t i = 0; i < sizeof(dsml_syntaxes) / sizeof(dsml_syntaxes[0]); i++) {
        const dsml_syntax *syn = &dsml_syntaxes[i];
        if (raw_full_lit(raw, raw_len, pos, syn->tool_calls_start)) {
            ts->syn = syn;
            ts->parse_pos = pos + strlen(syn->tool_calls_start);
            return true;
        }
    }
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return false;
}

static bool anthropic_tool_stream_fail(anthropic_tool_stream *ts) {
    ts->active = false;
    ts->state = DSML_TOOL_ERROR;
    return true;
}

static bool anthropic_tool_start_invoke(int fd, server *s, anthropic_stream *st,
                                        const char *raw, size_t raw_len) {
    anthropic_tool_stream *ts = &st->tool;
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos,
                         (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    free(tag);
    if (!name) return anthropic_tool_stream_fail(ts);

    /* This id is already visible to the client.  After final parsing,
     * apply_anthropic_stream_tool_ids() copies it into the parsed tool_call
     * before tool_memory_remember(), so the next tool_result can continue from
     * the live KV state instead of re-rendering canonical JSON. */
    const char *tool_id = anthropic_tool_stream_id(s, ts, ts->index);
    bool ok = anthropic_sse_open_tool_block(fd, st, tool_id, name) &&
              anthropic_tool_emit_args_fragment(fd, st, "{", 1);
    free(name);
    if (!ok) return false;

    ts->emitted_any = true;
    ts->args_open = true;
    ts->first_param = true;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}

static bool anthropic_tool_start_param(int fd, anthropic_stream *st,
                                       const char *raw, size_t raw_len) {
    anthropic_tool_stream *ts = &st->tool;
    const char *tag_end = memchr(raw + ts->parse_pos, '>', raw_len - ts->parse_pos);
    if (!tag_end) return true;
    char *tag = xstrndup(raw + ts->parse_pos,
                         (size_t)(tag_end - (raw + ts->parse_pos) + 1));
    char *name = dsml_attr(tag, "name");
    char *is_string = dsml_attr(tag, "string");
    free(tag);
    if (!name || !is_string) {
        free(name);
        free(is_string);
        return anthropic_tool_stream_fail(ts);
    }
    bool string_value = !strcmp(is_string, "true");
    bool ok = anthropic_tool_emit_param_prefix(fd, st, name, string_value);
    free(name);
    free(is_string);
    if (!ok) return false;

    ts->param_is_string = string_value;
    ts->parse_pos = (size_t)(tag_end - raw) + 1;
    ts->state = DSML_TOOL_PARAM_VALUE;
    return true;
}

static bool anthropic_tool_finish_param(int fd, anthropic_stream *st,
                                        const char *raw, size_t value_end) {
    anthropic_tool_stream *ts = &st->tool;
    if (value_end > ts->parse_pos) {
        bool ok = ts->param_is_string ?
            anthropic_tool_emit_string_value(fd, st, raw + ts->parse_pos,
                                             value_end - ts->parse_pos) :
            anthropic_tool_emit_args_fragment(fd, st, raw + ts->parse_pos,
                                              value_end - ts->parse_pos);
        if (!ok) return false;
    }
    if (ts->param_is_string &&
        !anthropic_tool_emit_args_fragment(fd, st, "\"", 1)) return false;
    ts->parse_pos = value_end + strlen(ts->syn->param_end);
    ts->state = DSML_TOOL_BETWEEN_PARAMS;
    return true;
}

static bool anthropic_tool_stream_update(int fd, server *s, const char *id,
                                         anthropic_stream *st,
                                         const char *raw, size_t raw_len) {
    anthropic_tool_stream *ts = &st->tool;
    while (ts->active && ts->parse_pos < raw_len) {
        if (ts->state == DSML_TOOL_BETWEEN_INVOKES) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->syn->tool_calls_end)) {
                ts->parse_pos += strlen(ts->syn->tool_calls_end);
                ts->active = false;
                ts->state = DSML_TOOL_DONE;
                return true;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos,
                                ts->syn->tool_calls_end, ts->syn->invoke_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->syn->invoke_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!anthropic_tool_start_invoke(fd, s, st, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return anthropic_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_BETWEEN_PARAMS) {
            while (ts->parse_pos < raw_len && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= raw_len) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->syn->invoke_end)) {
                if (ts->args_open &&
                    !anthropic_tool_emit_args_fragment(fd, st, "}", 1)) return false;
                ts->args_open = false;
                if (!anthropic_sse_close_block_live(fd, id, st)) return false;
                ts->parse_pos += strlen(ts->syn->invoke_end);
                ts->index++;
                ts->state = DSML_TOOL_BETWEEN_INVOKES;
                continue;
            }
            if (raw_partial_any(raw, raw_len, ts->parse_pos,
                                ts->syn->invoke_end, ts->syn->param_start)) return true;
            if (raw_full_lit(raw, raw_len, ts->parse_pos, ts->syn->param_start)) {
                size_t before_pos = ts->parse_pos;
                dsml_tool_stream_state before_state = ts->state;
                if (!anthropic_tool_start_param(fd, st, raw, raw_len)) return false;
                if (ts->parse_pos == before_pos && ts->state == before_state) return true;
                continue;
            }
            return anthropic_tool_stream_fail(ts);
        }

        if (ts->state == DSML_TOOL_PARAM_VALUE) {
            const char *end = find_lit_bounded(raw + ts->parse_pos,
                                               raw_len - ts->parse_pos,
                                               ts->syn->param_end);
            if (end) {
                if (!anthropic_tool_finish_param(fd, st, raw,
                                                 (size_t)(end - raw))) return false;
                continue;
            }
            size_t limit = tool_param_value_stream_safe_len(raw, ts->parse_pos,
                                                            raw_len,
                                                            ts->syn->param_end,
                                                            ts->param_is_string);
            if (limit > ts->parse_pos) {
                bool ok = ts->param_is_string ?
                    anthropic_tool_emit_string_value(fd, st, raw + ts->parse_pos,
                                                     limit - ts->parse_pos) :
                    anthropic_tool_emit_args_fragment(fd, st, raw + ts->parse_pos,
                                                      limit - ts->parse_pos);
                if (!ok) return false;
                ts->parse_pos = limit;
            }
            return true;
        }

        return true;
    }
    return true;
}

static size_t text_stream_safe_limit(const char *raw, size_t start,
                                     size_t raw_len, bool has_tools,
                                     bool final) {
    if (raw_len <= start) return raw_len;

    size_t limit = raw_len;
    if (has_tools) {
        const char *tool = find_any_tool_start(raw + start);
        if (tool) {
            limit = trim_tool_separator_ws(raw, start, (size_t)(tool - raw));
            return utf8_stream_safe_len(raw, start, limit, true);
        }

        if (!final) {
            /* Tool calls are hidden from the API client and returned as
             * structured tool_use/tool_calls blocks.  The whitespace just
             * before the DSML marker is syntax too: if we stream it as
             * assistant text, the next client request sends it back and our
             * renderer adds the canonical "\n\n" separator again.  Hold
             * trailing whitespace until a following non-whitespace byte proves
             * it is ordinary text, or until a tool marker proves it should be
             * dropped. */
            while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;

            /* Also hold a partial '<...tool_calls...' marker that may be split
             * across generated tokens. */
            const size_t max_marker = 80;
            size_t scan = raw_len - start > max_marker ? raw_len - max_marker : start;
            for (size_t i = raw_len; i > scan; i--) {
                if (raw[i - 1] == '<') {
                    size_t marker = i - 1;
                    if (marker < limit) limit = marker;
                    break;
                }
            }
            limit = trim_tool_separator_ws(raw, start, limit);
        }
    }
    return utf8_stream_safe_len(raw, start, limit, final);
}

static bool anthropic_sse_stream_update(int fd, server *s, const request *r, const char *id,
                                        anthropic_stream *st,
                                        const char *raw, size_t raw_len,
                                        bool final) {
    if (!st->active || !raw) return true;

    if (st->mode == ANTH_STREAM_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = "<think>";
            const size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) {
                return true;
            }
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) {
                st->emit_pos = open_len;
            }
            st->checked_think_prefix = true;
        }

        const char *close = strstr(raw + st->emit_pos, "</think>");
        size_t limit;
        if (close) {
            limit = (size_t)(close - raw);
        } else if (final) {
            limit = raw_len;
        } else {
            const size_t hold = strlen("</think>") - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }

        if (limit > st->emit_pos) {
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_THINKING)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_THINKING,
                                          raw + st->emit_pos,
                                          limit - st->emit_pos)) return false;
            st->sent_thinking = true;
            st->emit_pos = limit;
        }

        if (close || final) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            if (close) {
                st->emit_pos = (size_t)(close - raw) + strlen("</think>");
                st->mode = ANTH_STREAM_TEXT;
            } else {
                st->mode = ANTH_STREAM_SUPPRESS;
                return true;
            }
        } else {
            return true;
        }
    }

    if (st->mode == ANTH_STREAM_TEXT) {
        const char *tool = r->has_tools ? find_any_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len,
                                              r->has_tools, final);

        if (limit > st->emit_pos) {
            if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
            if (!anthropic_sse_delta_live(fd, st, ANTH_BLOCK_TEXT,
                                          raw + st->emit_pos,
                                          limit - st->emit_pos)) return false;
            st->sent_text = true;
            st->emit_pos = limit;
        }

        if (tool) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->emit_pos = (size_t)(tool - raw);
            /* On normal token-by-token updates, switch from hidden text to a
             * live tool_use projection as soon as the DSML block starts.  On
             * final catch-up from plain text, leave the block for the existing
             * final emitter so old non-incremental behavior stays unchanged. */
            if (!final &&
                anthropic_tool_stream_init(&st->tool, raw, raw_len, st->emit_pos)) {
                st->mode = ANTH_STREAM_TOOL;
            } else {
                st->mode = ANTH_STREAM_SUPPRESS;
            }
        } else if (final) {
            if (!anthropic_sse_close_block_live(fd, id, st)) return false;
            st->mode = ANTH_STREAM_SUPPRESS;
        }
    }

    if (st->mode == ANTH_STREAM_TOOL) {
        if (!anthropic_tool_stream_update(fd, s, id, st, raw, raw_len)) return false;
        if (!st->tool.active) st->mode = ANTH_STREAM_SUPPRESS;
    }
    return true;
}

static bool anthropic_sse_tool_blocks_live(int fd, const request *r, const char *id,
                                           anthropic_stream *st,
                                           const tool_calls *calls) {
    (void)r;
    if (!calls) return true;

    buf b = {0};
    /* Tool calls completed by anthropic_tool_stream_update() have already
     * produced start/delta/stop events.  Only emit the tail calls that were not
     * seen by the live projection, for example if the first DSML bytes only
     * become available during final flush. */
    int already_streamed = st->tool.emitted_any ? st->tool.index : 0;
    if (already_streamed > calls->len) already_streamed = calls->len;
    for (int i = already_streamed; i < calls->len; i++, st->next_index++) {
        const tool_call *tc = &calls->v[i];
        char idbuf[128];
        snprintf(idbuf, sizeof(idbuf), "toolu_%s_%d", id, i);
        buf_printf(&b,
                   "{\"type\":\"content_block_start\",\"index\":%d,"
                   "\"content_block\":{\"type\":\"tool_use\",\"id\":",
                   st->next_index);
        json_escape(&b, tc->id && tc->id[0] ? tc->id : idbuf);
        buf_puts(&b, ",\"name\":");
        json_escape(&b, tc->name ? tc->name : "");
        buf_puts(&b, ",\"input\":{}}}");
        bool ok = sse_event(fd, "content_block_start", b.ptr);
        buf_free(&b);
        if (!ok) return false;

        buf_printf(&b,
                   "{\"type\":\"content_block_delta\",\"index\":%d,"
                   "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
                   st->next_index);
        append_json_object_string(&b, tc->arguments);
        buf_puts(&b, "}}");
        ok = sse_event(fd, "content_block_delta", b.ptr);
        buf_free(&b);
        if (!ok) return false;

        buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}",
                   st->next_index);
        ok = sse_event(fd, "content_block_stop", b.ptr);
        buf_free(&b);
        if (!ok) return false;
    }
    return true;
}

static bool anthropic_sse_stop_live(int fd, const char *finish,
                                    int completion_tokens) {
    buf b = {0};
    buf_puts(&b, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
    json_escape(&b, anthropic_stop_reason(finish));
    buf_puts(&b, ",\"stop_sequence\":null},\"usage\":{\"output_tokens\":");
    buf_printf(&b, "%d}}", completion_tokens);
    bool ok = sse_event(fd, "message_delta", b.ptr);
    buf_free(&b);
    if (ok) ok = sse_event(fd, "message_stop", "{\"type\":\"message_stop\"}");
    return ok;
}

static bool anthropic_sse_finish_live(int fd, server *s, const request *r, const char *id,
                                      anthropic_stream *st, const char *raw,
                                      size_t raw_len, const tool_calls *calls,
                                      const char *finish, int completion_tokens) {
    if (!anthropic_sse_stream_update(fd, s, r, id, st, raw, raw_len, true)) return false;

    if (st->sent_thinking && !st->sent_text && (!calls || calls->len == 0)) {
        if (!anthropic_sse_open_block(fd, st, ANTH_BLOCK_TEXT)) return false;
        if (!anthropic_sse_close_block_live(fd, id, st)) return false;
    }

    if (!anthropic_sse_tool_blocks_live(fd, r, id, st, calls)) return false;
    return anthropic_sse_stop_live(fd, finish, completion_tokens);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void server_log(ds4_log_type type, const char *fmt, ...) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char ts[16];
    strftime(ts, sizeof(ts), "%m%d %H:%M:%S", &tm);

    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);

    fprintf(stderr, "%s ", ts);
    if (n < 0) {
        ds4_log(stderr, type, "%s", fmt);
    } else {
        char *line = xmalloc((size_t)n + 1);
        vsnprintf(line, (size_t)n + 1, fmt, ap);
        ds4_log(stderr, type, "%s", line);
        free(line);
    }
    va_end(ap);
    fputc('\n', stderr);
}

typedef struct job job;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence.  The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
} kv_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} kv_cache_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    kv_cache_options opt;
    int continued_last_store_tokens;
    kv_entry *entry;
    int len;
    int cap;
} kv_disk_cache;

typedef enum {
    TOOL_MEMORY_RAM = 0,
    TOOL_MEMORY_DISK = 1,
} tool_memory_source;

typedef struct tool_memory_entry tool_memory_entry;

typedef struct {
    char *dsml;
    size_t len;
    size_t bytes;
    int refs;
    uint64_t seen;
    tool_memory_entry *entries;
} tool_memory_block;

struct tool_memory_entry {
    char *id;
    tool_memory_block *block;
    size_t bytes;
    uint64_t stamp;
    tool_memory_source source;
    tool_memory_entry *prev;
    tool_memory_entry *next;
    tool_memory_entry *block_next;
};

typedef struct {
    rax *by_id;
    rax *by_block;
    tool_memory_entry *head;
    tool_memory_entry *tail;
    int entries;
    int max_entries;
    size_t bytes;
    size_t max_bytes;
    uint64_t clock;
    uint64_t scan_clock;
} tool_memory;

typedef struct {
    bool valid;
    /* Token frontier of a live assistant tool-call turn. Continuing from this
     * point preserves hidden thinking and sampled DSML bytes that are not
     * necessarily present in the client-visible replay. */
    int live_tokens;
    /* Optional rendered conversation text that the client is expected to replay.
     * Responses uses this because visible replay can omit hidden reasoning.
     * Anthropic currently uses only the call-id side of the state. */
    char *visible_text;
    size_t visible_len;
    /* Tool-call ids generated at the same live frontier. A following tool
     * result for these ids is a direct protocol continuation and should not
     * trigger prompt-prefix matching or checkpoint canonicalization. */
    stop_list call_ids;
} live_tool_state;

typedef struct {
    bool valid;
    /* Token frontier of the live sampled session.  The visible text below is
     * what clients will replay, but the payload at this frontier may also
     * contain hidden thinking tokens that are intentionally absent from that
     * visible replay. */
    int live_tokens;
    char *visible_text;
    size_t visible_len;
} visible_live_state;

static bool id_list_contains(const stop_list *ids, const char *id);
static void id_list_push_unique(stop_list *ids, const char *id);

struct server {
    ds4_engine *engine;
    ds4_session *session;
    int default_tokens;
    kv_disk_cache kv;
    tool_memory tool_mem;
    live_tool_state responses_live;
    live_tool_state anthropic_live;
    visible_live_state thinking_live;
    bool disable_exact_dsml_tool_replay;
    bool enable_cors;
    pthread_mutex_t tool_mu;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t clients_cv;
    job *head;
    job *tail;
    bool stopping;
    int clients;
    uint64_t seq;
    FILE *trace;
    pthread_mutex_t trace_mu;
    uint64_t trace_seq;
};

/* Jobs are stack-owned by the client thread.  The worker signals completion
 * after the response has been written, so request data and the socket remain
 * valid without heap-allocating per-request job objects. */
struct job {
    int fd;
    request req;
    bool done;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *next;
};

/* =========================================================================
 * Tool Call Text Memory.
 * =========================================================================
 *
 * The model speaks DSML, while OpenAI and Anthropic clients round-trip tool
 * calls as JSON.  Re-rendering that JSON is not always the same byte sequence:
 * clients may preserve, sort, or rebuild object keys differently.  Tool call
 * ids are the bridge between both worlds.  For every generated tool call we
 * remember the exact DSML block sampled by the model under a random id.  When
 * the client later sends the same id back in conversation history, we replay
 * the sampled DSML verbatim and keep the KV cache aligned with the live model
 * state.
 */

#define DS4_TOOL_MEMORY_DEFAULT_MAX_IDS 100000
#define DS4_TOOL_MEMORY_MAX_BYTES (512u * 1024u * 1024u)

static int tool_memory_max_entries(const tool_memory *m) {
    return m && m->max_entries > 0 ? m->max_entries : DS4_TOOL_MEMORY_DEFAULT_MAX_IDS;
}

static size_t tool_memory_max_bytes(const tool_memory *m) {
    return m && m->max_bytes > 0 ? m->max_bytes : DS4_TOOL_MEMORY_MAX_BYTES;
}

static void tool_memory_init_locked(tool_memory *m) {
    if (m->by_id && m->by_block) return;
    m->by_id = raxNew();
    m->by_block = raxNew();
    if (!m->by_id || !m->by_block) die("out of memory");
}

static void tool_memory_link_head(tool_memory *m, tool_memory_entry *e) {
    e->prev = NULL;
    e->next = m->head;
    if (m->head) m->head->prev = e;
    else m->tail = e;
    m->head = e;
}

static void tool_memory_unlink(tool_memory *m, tool_memory_entry *e) {
    if (e->prev) e->prev->next = e->next;
    else m->head = e->next;
    if (e->next) e->next->prev = e->prev;
    else m->tail = e->prev;
    e->prev = e->next = NULL;
}

static void tool_memory_touch(tool_memory *m, tool_memory_entry *e) {
    e->stamp = ++m->clock;
    if (m->head == e) return;
    tool_memory_unlink(m, e);
    tool_memory_link_head(m, e);
}

static void tool_block_unlink_entry(tool_memory_block *b, tool_memory_entry *e) {
    tool_memory_entry **p = &b->entries;
    while (*p) {
        if (*p == e) {
            *p = e->block_next;
            e->block_next = NULL;
            return;
        }
        p = &(*p)->block_next;
    }
}

static tool_memory_block *tool_memory_find_block_locked(tool_memory *m,
                                                        const char *dsml,
                                                        size_t len) {
    if (!m->by_block || !dsml || len == 0) return NULL;
    void *v = raxFind(m->by_block, (unsigned char *)dsml, len);
    return v == raxNotFound ? NULL : v;
}

static tool_memory_block *tool_memory_get_block_locked(tool_memory *m,
                                                       const char *dsml,
                                                       size_t len) {
    tool_memory_block *b = tool_memory_find_block_locked(m, dsml, len);
    if (b) return b;

    b = xmalloc(sizeof(*b));
    memset(b, 0, sizeof(*b));
    b->dsml = xstrndup(dsml, len);
    b->len = len;
    b->bytes = len + 1 + sizeof(*b);
    if (!raxInsert(m->by_block, (unsigned char *)b->dsml, b->len, b, NULL)) {
        free(b->dsml);
        free(b);
        die("out of memory");
    }
    m->bytes += b->bytes;
    return b;
}

static void tool_memory_release_block_locked(tool_memory *m, tool_memory_block *b) {
    if (!b) return;
    if (--b->refs > 0) return;
    if (m->by_block) {
        void *old = NULL;
        (void)raxRemove(m->by_block, (unsigned char *)b->dsml, b->len, &old);
    }
    if (m->bytes >= b->bytes) m->bytes -= b->bytes;
    else m->bytes = 0;
    free(b->dsml);
    free(b);
}

static void tool_memory_remove_entry_locked(tool_memory *m, tool_memory_entry *e) {
    if (!e) return;
    if (m->by_id && e->id) {
        void *old = NULL;
        (void)raxRemove(m->by_id, (unsigned char *)e->id, strlen(e->id), &old);
    }
    tool_memory_unlink(m, e);
    if (e->block) tool_block_unlink_entry(e->block, e);
    if (m->bytes >= e->bytes) m->bytes -= e->bytes;
    else m->bytes = 0;
    if (m->entries > 0) m->entries--;
    free(e->id);
    tool_memory_release_block_locked(m, e->block);
    free(e);
}

static void tool_memory_prune_locked(tool_memory *m) {
    while ((m->entries > tool_memory_max_entries(m) ||
            m->bytes > tool_memory_max_bytes(m)) && m->tail)
    {
        tool_memory_remove_entry_locked(m, m->tail);
    }
}

static tool_memory_entry *tool_memory_find_entry_locked(tool_memory *m,
                                                        const char *id) {
    if (!m->by_id || !id || !id[0]) return NULL;
    void *v = raxFind(m->by_id, (unsigned char *)id, strlen(id));
    return v == raxNotFound ? NULL : v;
}

static void tool_memory_put_locked(tool_memory *m, const char *id,
                                   const char *dsml, tool_memory_source source) {
    if (!id || !id[0] || !dsml || !dsml[0]) return;
    tool_memory_init_locked(m);

    size_t dsml_len = strlen(dsml);
    tool_memory_entry *old = tool_memory_find_entry_locked(m, id);
    if (old && old->block && old->block->len == dsml_len &&
        !memcmp(old->block->dsml, dsml, dsml_len))
    {
        if (source == TOOL_MEMORY_RAM) old->source = TOOL_MEMORY_RAM;
        tool_memory_touch(m, old);
        tool_memory_prune_locked(m);
        return;
    }
    if (old) tool_memory_remove_entry_locked(m, old);

    tool_memory_block *b = tool_memory_get_block_locked(m, dsml, dsml_len);
    tool_memory_entry *e = xmalloc(sizeof(*e));
    memset(e, 0, sizeof(*e));
    e->id = xstrdup(id);
    e->block = b;
    e->bytes = strlen(id) + 1 + sizeof(*e);
    e->stamp = ++m->clock;
    e->source = source;
    e->block_next = b->entries;
    b->entries = e;
    b->refs++;

    if (!raxInsert(m->by_id, (unsigned char *)e->id, strlen(e->id), e, NULL)) {
        tool_block_unlink_entry(b, e);
        free(e->id);
        free(e);
        tool_memory_release_block_locked(m, b);
        die("out of memory");
    }
    tool_memory_link_head(m, e);
    m->entries++;
    m->bytes += e->bytes;
    tool_memory_prune_locked(m);
}

static void tool_memory_free(tool_memory *m) {
    while (m->tail) tool_memory_remove_entry_locked(m, m->tail);
    if (m->by_id) raxFree(m->by_id);
    if (m->by_block) raxFree(m->by_block);
    memset(m, 0, sizeof(*m));
}

/* Single live protocol-tool state.
 *
 * This is not an implementation of durable remote conversation storage.  It is
 * only an in-memory binding from protocol tool-call IDs to the current sampled
 * KV frontier.  If it does not match, DS4 falls back to the same prefix and
 * disk-cache machinery used by chat/completions, or returns a clear error for
 * tool-result-only requests that have no replayable prefix. */
static void live_tool_state_clear_locked(live_tool_state *st) {
    if (!st) return;
    stop_list_clear(&st->call_ids);
    free(st->visible_text);
    st->visible_text = NULL;
    st->visible_len = 0;
    st->valid = false;
    st->live_tokens = 0;
}

static void live_tool_state_free(live_tool_state *st) {
    if (!st) return;
    live_tool_state_clear_locked(st);
    free(st->call_ids.v);
    memset(st, 0, sizeof(*st));
}

static void visible_live_clear_locked(visible_live_state *st) {
    if (!st) return;
    free(st->visible_text);
    st->visible_text = NULL;
    st->visible_len = 0;
    st->live_tokens = 0;
    st->valid = false;
}

static void visible_live_free(visible_live_state *st) {
    if (!st) return;
    visible_live_clear_locked(st);
    memset(st, 0, sizeof(*st));
}

static void thinking_live_clear(server *s) {
    if (!s) return;
    pthread_mutex_lock(&s->tool_mu);
    visible_live_clear_locked(&s->thinking_live);
    pthread_mutex_unlock(&s->tool_mu);
}

static void thinking_live_remember(server *s, const char *visible_text) {
    if (!s || !visible_text || !visible_text[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    visible_live_clear_locked(&s->thinking_live);
    s->thinking_live.visible_text = xstrdup(visible_text);
    s->thinking_live.visible_len = strlen(visible_text);
    s->thinking_live.live_tokens = ds4_session_pos(s->session);
    s->thinking_live.valid = true;
    pthread_mutex_unlock(&s->tool_mu);
}

static void responses_live_remember(server *s, const char *visible_text,
                                    const tool_calls *calls) {
    if (!s || !visible_text || !visible_text[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&s->responses_live);
    s->responses_live.visible_text = xstrdup(visible_text);
    s->responses_live.visible_len = strlen(visible_text);
    if (calls) {
        for (int i = 0; i < calls->len; i++) {
            id_list_push_unique(&s->responses_live.call_ids, calls->v[i].id);
        }
    }
    s->responses_live.live_tokens = ds4_session_pos(s->session);
    s->responses_live.valid = true;
    pthread_mutex_unlock(&s->tool_mu);
}

static void anthropic_live_remember(server *s, const tool_calls *calls) {
    if (!s || !calls || calls->len == 0) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&s->anthropic_live);
    for (int i = 0; i < calls->len; i++) {
        id_list_push_unique(&s->anthropic_live.call_ids, calls->v[i].id);
    }
    s->anthropic_live.live_tokens = ds4_session_pos(s->session);
    s->anthropic_live.valid = s->anthropic_live.call_ids.len > 0;
    pthread_mutex_unlock(&s->tool_mu);
}

static void responses_live_clear(server *s) {
    if (!s) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&s->responses_live);
    pthread_mutex_unlock(&s->tool_mu);
}

static void anthropic_live_clear(server *s) {
    if (!s) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&s->anthropic_live);
    pthread_mutex_unlock(&s->tool_mu);
}

static bool responses_live_has_call_id(server *s, const char *id) {
    if (!s || !id || !id[0]) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool found = s->responses_live.valid &&
                 id_list_contains(&s->responses_live.call_ids, id);
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}

static bool anthropic_live_has_call_id(server *s, const char *id) {
    if (!s || !id || !id[0]) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool found = s->anthropic_live.valid &&
                 id_list_contains(&s->anthropic_live.call_ids, id);
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}

static bool responses_live_matches_request(server *s, const stop_list *ids,
                                           int live_tokens) {
    if (!s || !ids || ids->len == 0) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->responses_live.valid &&
              s->responses_live.live_tokens == live_tokens &&
              s->responses_live.call_ids.len == ids->len;
    for (int i = 0; ok && i < ids->len; i++) {
        ok = id_list_contains(&s->responses_live.call_ids, ids->v[i]);
    }
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}

static bool anthropic_live_matches_request(server *s, const stop_list *ids,
                                           int live_tokens) {
    if (!s || !ids || ids->len == 0) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->anthropic_live.valid &&
              s->anthropic_live.live_tokens == live_tokens &&
              s->anthropic_live.call_ids.len == ids->len;
    for (int i = 0; ok && i < ids->len; i++) {
        ok = id_list_contains(&s->anthropic_live.call_ids, ids->v[i]);
    }
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}

static bool tool_memory_has_id(server *s, const char *id) {
    if (!s || s->disable_exact_dsml_tool_replay || !id || !id[0]) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool found = tool_memory_find_entry_locked(&s->tool_mem, id) != NULL;
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}

static const char *tool_memory_lookup_locked(tool_memory *m, const char *id,
                                             tool_memory_source *source,
                                             tool_memory_block **block) {
    tool_memory_entry *e = tool_memory_find_entry_locked(m, id);
    if (!e || !e->block) return NULL;
    tool_memory_touch(m, e);
    if (source) *source = e->source;
    if (block) *block = e->block;
    return e->block->dsml;
}

static void tool_memory_remember(server *s, const tool_calls *calls) {
    if (!s || s->disable_exact_dsml_tool_replay ||
        !calls || !calls->raw_dsml || !calls->raw_dsml[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    for (int i = 0; i < calls->len; i++) {
        tool_memory_put_locked(&s->tool_mem, calls->v[i].id, calls->raw_dsml,
                               TOOL_MEMORY_RAM);
    }
    pthread_mutex_unlock(&s->tool_mu);
}

static void tool_memory_put_source(server *s, const char *id, const char *dsml,
                                   tool_memory_source source) {
    if (!s || s->disable_exact_dsml_tool_replay ||
        !id || !id[0] || !dsml || !dsml[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    tool_memory_put_locked(&s->tool_mem, id, dsml, source);
    pthread_mutex_unlock(&s->tool_mu);
}

#ifdef DS4_SERVER_TEST
static void tool_memory_put(server *s, const char *id, const char *dsml) {
    tool_memory_put_source(s, id, dsml, TOOL_MEMORY_RAM);
}
#endif

static void tool_memory_attach_to_messages(server *s, chat_msgs *msgs,
                                           tool_replay_stats *stats) {
    if (!msgs) return;
    if (!s || s->disable_exact_dsml_tool_replay) {
        if (stats) {
            for (int i = 0; i < msgs->len; i++) {
                tool_calls *calls = &msgs->v[i].calls;
                if (calls->len == 0 || calls->raw_dsml) continue;
                stats->canonical++;
                stats->missing_ids += calls->len;
            }
        }
        return;
    }
    pthread_mutex_lock(&s->tool_mu);
    for (int i = 0; i < msgs->len; i++) {
        tool_calls *calls = &msgs->v[i].calls;
        if (calls->len == 0 || calls->raw_dsml) continue;
        tool_memory_block *matched = NULL;
        tool_memory_source matched_source = TOOL_MEMORY_DISK;
        bool exact = true;
        int missing = 0;
        for (int j = 0; j < calls->len; j++) {
            tool_memory_source source = TOOL_MEMORY_DISK;
            tool_memory_block *block = NULL;
            const char *dsml =
                tool_memory_lookup_locked(&s->tool_mem, calls->v[j].id,
                                          &source, &block);
            if (!dsml) {
                exact = false;
                missing++;
                continue;
            }
            if (!matched) {
                matched = block;
                matched_source = source;
            } else if (matched != block) {
                exact = false;
            }
            if (source == TOOL_MEMORY_RAM) matched_source = TOOL_MEMORY_RAM;
        }
        if (exact && matched) {
            calls->raw_dsml = xstrdup(matched->dsml);
            if (stats) {
                if (matched_source == TOOL_MEMORY_RAM) stats->mem++;
                else stats->disk++;
            }
        } else if (stats) {
            stats->canonical++;
            stats->missing_ids += missing;
        }
    }
    pthread_mutex_unlock(&s->tool_mu);
}

static bool tool_calls_contains_id(const tool_calls *calls, const char *id, int upto) {
    if (!calls || !id || !id[0]) return false;
    if (upto > calls->len) upto = calls->len;
    for (int i = 0; i < upto; i++) {
        if (calls->v[i].id && !strcmp(calls->v[i].id, id)) return true;
    }
    return false;
}

static void assign_tool_call_ids(server *s, tool_calls *calls, api_style api) {
    if (!calls) return;
    for (int i = 0; i < calls->len; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), api);
            if (!tool_calls_contains_id(calls, id, i) && !tool_memory_has_id(s, id)) break;
        }
        calls->v[i].id = xstrdup(id);
    }
}

static void apply_openai_stream_tool_ids(tool_calls *calls,
                                         const openai_stream *st) {
    if (!calls || !st) return;
    int n = calls->len < st->tool.ids_cap ? calls->len : st->tool.ids_cap;
    for (int i = 0; i < n; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        if (st->tool.ids[i] && st->tool.ids[i][0]) calls->v[i].id = xstrdup(st->tool.ids[i]);
    }
}

static void apply_anthropic_stream_tool_ids(tool_calls *calls,
                                            const anthropic_stream *st) {
    if (!calls || !st) return;
    /* The SSE stream may have exposed tool ids before final DSML parsing.  The
     * parsed calls must inherit those ids before assign_tool_call_ids() and
     * tool_memory_remember(), otherwise the client returns a tool_result for an
     * id that the continuation fast path does not know. */
    int n = calls->len < st->tool.ids_cap ? calls->len : st->tool.ids_cap;
    for (int i = 0; i < n; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        if (st->tool.ids[i] && st->tool.ids[i][0]) calls->v[i].id = xstrdup(st->tool.ids[i]);
    }
}

/* =========================================================================
 * KV Cache.
 * =========================================================================
 *
 * The server has one live Metal session.  We persist reusable DS4 session
 * snapshots when a cold prompt reaches a useful prefix, when a long continued
 * conversation has grown far enough, and when a request evicts the live session.
 * The cache key is the SHA1 of the rendered byte prefix.  The payload still
 * stores exact token IDs and graph state; the filename only selects a checkpoint
 * whose decoded transcript bytes are a prefix of the next rendered request.
 *
 * Files are loaded with plain read/write I/O into the existing graph tensors;
 * mmap is deliberately avoided here so cache restore cannot add more VM
 * mappings to a process that already maps a very large GGUF.
 *
 * Stores are created only when the live graph is already at the checkpoint we
 * want to persist.  For long cold prompts this means prefill reaches the stable
 * boundary first, writes that prefix, and then continues with the suffix.  We
 * never roll the session backward just to build a disk cache entry: that would
 * turn cache population into a second hidden prefill.
 *
 * File layout:
 *
 *   "KVC" version
 *   quant bits, save reason, token count, hit count, context size
 *   creation time, last-used time, payload byte count
 *   rendered text byte count + rendered text for human inspection
 *   DS4 engine payload written by ds4_session_save_payload()
 *   optional tool-id map section
 *
 * The filename is SHA1(cache text bytes), not SHA1(token ids).  For ordinary
 * checkpoints the cache text is the rendered token prefix.  For live hidden
 * state it can instead be the client-visible transcript: the payload still
 * contains sampled reasoning KV, but the lookup key must be what the client can
 * replay after a process restart or session switch.
 *
 * The optional tool-id map is not part of model state, but it is needed to
 * render future client JSON back to the exact DSML sampled by the model.  We
 * persist only mappings whose DSML block appears in the saved cache text.
 */

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
#define KV_CACHE_VERSION 1u
#define KV_CACHE_FIXED_HEADER 48u
#define KV_CACHE_DEFAULT_MIN_TOKENS 512
#define KV_CACHE_DEFAULT_COLD_MAX_TOKENS 30000
/* Tokenizers may merge text across the prompt boundary.  Trimming a small tail
 * still improves the cheap token-prefix path, while text-prefix lookup handles
 * the cases where canonical prompt tokenization spells the same bytes
 * differently.  The 2048 alignment also matches the Metal prefill chunk
 * schedule, which keeps compressor row finalization identical to a cold full
 * prompt. */
#define KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS 32
#define KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS 2048
#define KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS 10000
#define KV_CACHE_DEFAULT_MB 4096
/* Disk-hit counts are evidence that a checkpoint was useful, but only while
 * the workload still resembles the one that produced those hits.  Prompt or
 * tool-schema changes can make a once-hot checkpoint impossible to match, so
 * eviction decays only the hit bonus with inactivity.  The baseline token/byte
 * value remains intact: old files are not punished for age, they just stop
 * carrying stale popularity forever. */
#define KV_CACHE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)
#define KV_CACHE_MIN_EFFECTIVE_HITS 0.01
#define KV_EXT_TOOL_MAP (1u << 0)
#define KV_EXT_RESPONSES_VISIBLE (1u << 1)
#define KV_EXT_THINKING_VISIBLE (1u << 2)
#define KV_TOOL_MAP_MAGIC0 'K'
#define KV_TOOL_MAP_MAGIC1 'T'
#define KV_TOOL_MAP_MAGIC2 'M'
#define KV_TOOL_MAP_VERSION 1u
#define KV_TOOL_MAP_HEADER 8u

typedef enum {
    KV_REASON_UNKNOWN   = 0,
    KV_REASON_COLD      = 1,
    KV_REASON_CONTINUED = 2,
    KV_REASON_EVICT     = 3,
    KV_REASON_SHUTDOWN  = 4,
} kv_cache_reason;

static uint8_t kv_reason_code(const char *reason) {
    if (!reason) return KV_REASON_UNKNOWN;
    if (!strcmp(reason, "cold")) return KV_REASON_COLD;
    if (!strcmp(reason, "continued")) return KV_REASON_CONTINUED;
    if (!strcmp(reason, "evict")) return KV_REASON_EVICT;
    if (!strcmp(reason, "shutdown")) return KV_REASON_SHUTDOWN;
    return KV_REASON_UNKNOWN;
}

static kv_cache_options kv_cache_default_options(void) {
    return (kv_cache_options){
        .min_tokens = KV_CACHE_DEFAULT_MIN_TOKENS,
        .cold_max_tokens = KV_CACHE_DEFAULT_COLD_MAX_TOKENS,
        .continued_interval_tokens = KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS,
        .boundary_trim_tokens = KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS,
        .boundary_align_tokens = KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS,
    };
}

static void le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint32_t le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

typedef struct {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} sha1_ctx;

static uint32_t rol32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void sha1_transform(sha1_ctx *c, const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++) w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = c->h[0], b = c->h[1], d = c->h[3], e = c->h[4];
    uint32_t cc = c->h[2];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ cc ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = rol32(b, 30);
        b = a;
        a = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += cc;
    c->h[3] += d;
    c->h[4] += e;
}

static void sha1_init(sha1_ctx *c) {
    c->h[0] = 0x67452301u;
    c->h[1] = 0xefcdab89u;
    c->h[2] = 0x98badcfeu;
    c->h[3] = 0x10325476u;
    c->h[4] = 0xc3d2e1f0u;
    c->bytes = 0;
    c->used = 0;
}

static void sha1_update(sha1_ctx *c, const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    c->bytes += len;
    while (len != 0) {
        size_t n = 64 - c->used;
        if (n > len) n = len;
        memcpy(c->block + c->used, p, n);
        c->used += n;
        p += n;
        len -= n;
        if (c->used == 64) {
            sha1_transform(c, c->block);
            c->used = 0;
        }
    }
}

static void sha1_final(sha1_ctx *c, uint8_t out[20]) {
    uint64_t bits = c->bytes * 8;
    uint8_t one = 0x80;
    uint8_t zero = 0;
    sha1_update(c, &one, 1);
    while (c->used != 56) sha1_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
    sha1_update(c, len, sizeof(len));
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c->h[i];
    }
}

static void hex20(const uint8_t in[20], char out[41]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        out[i * 2] = hex[in[i] >> 4];
        out[i * 2 + 1] = hex[in[i] & 15];
    }
    out[40] = '\0';
}

static void sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    sha1_ctx c;
    sha1_init(&c);
    sha1_update(&c, ptr, len);
    uint8_t digest[20];
    sha1_final(&c, digest);
    hex20(digest, out);
}

static bool id_list_contains(const stop_list *ids, const char *id) {
    if (!ids || !id || !id[0]) return false;
    for (int i = 0; i < ids->len; i++) {
        if (ids->v[i] && !strcmp(ids->v[i], id)) return true;
    }
    return false;
}

static void id_list_push_unique(stop_list *ids, const char *id) {
    if (!ids || !id || !id[0] || id_list_contains(ids, id)) return;
    stop_list_push(ids, xstrdup(id));
}

static void id_list_free(stop_list *ids) {
    stop_list_clear(ids);
    free(ids->v);
    memset(ids, 0, sizeof(*ids));
}

static void collect_tool_call_ids(const chat_msgs *msgs, stop_list *ids) {
    if (!msgs || !ids) return;
    for (int i = 0; i < msgs->len; i++) {
        id_list_push_unique(ids, msgs->v[i].tool_call_id);
        for (int j = 0; j < msgs->v[i].tool_call_ids_len; j++) {
            id_list_push_unique(ids, msgs->v[i].tool_call_ids[j]);
        }
        const tool_calls *calls = &msgs->v[i].calls;
        for (int j = 0; j < calls->len; j++) {
            id_list_push_unique(ids, calls->v[j].id);
        }
    }
}

static bool sha_hex_name(const char *name, char sha[41]) {
    if (strlen(name) != 43 || strcmp(name + 40, ".kv")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        sha[i] = (char)tolower((unsigned char)name[i]);
    }
    sha[40] = '\0';
    return true;
}

static char *path_join(const char *dir, const char *name) {
    buf b = {0};
    buf_puts(&b, dir);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') buf_putc(&b, '/');
    buf_puts(&b, name);
    return buf_take(&b);
}

static char *kv_path_for_sha(kv_disk_cache *kc, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return path_join(kc->dir, name);
}

static bool mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

static void kv_entry_free(kv_entry *e) {
    free(e->path);
    memset(e, 0, sizeof(*e));
}

static void kv_cache_clear(kv_disk_cache *kc) {
    for (int i = 0; i < kc->len; i++) kv_entry_free(&kc->entry[i]);
    free(kc->entry);
    kc->entry = NULL;
    kc->len = 0;
    kc->cap = 0;
}

static void kv_cache_push(kv_disk_cache *kc, kv_entry e) {
    if (kc->len == kc->cap) {
        kc->cap = kc->cap ? kc->cap * 2 : 16;
        kc->entry = xrealloc(kc->entry, (size_t)kc->cap * sizeof(kc->entry[0]));
    }
    kc->entry[kc->len++] = e;
}

static const char *find_next_dsml_tool_block(const char *p, const char **end_out) {
    struct block_form {
        const char *start;
        const char *end;
    } forms[] = {
        {"\n\n" DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END},
        {DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END},
        {"\n\n" DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT},
        {DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT},
        {"\n\n<tool_calls>", "</tool_calls>"},
        {"<tool_calls>", "</tool_calls>"},
    };

    const char *best = NULL;
    const char *best_end = NULL;
    for (size_t i = 0; i < sizeof(forms) / sizeof(forms[0]); i++) {
        const char *s = strstr(p, forms[i].start);
        if (!s || (best && s >= best)) continue;
        const char *e = strstr(s, forms[i].end);
        if (!e) continue;
        best = s;
        best_end = e + strlen(forms[i].end);
    }
    if (end_out) *end_out = best_end;
    return best;
}

static int tool_memory_count_dsml_in_text(server *s, const char *text) {
    if (!s || s->disable_exact_dsml_tool_replay || !text || !text[0]) return 0;
    int count = 0;
    pthread_mutex_lock(&s->tool_mu);
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_dsml_tool_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b) count += b->refs;
        p = end;
    }
    pthread_mutex_unlock(&s->tool_mu);
    return count;
}

static bool kv_tool_map_measure_locked(server *s, const char *text,
                                       uint32_t *count_out,
                                       uint64_t *bytes_out) {
    uint32_t count = 0;
    uint64_t bytes = KV_TOOL_MAP_HEADER;
    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_dsml_tool_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t dsml_len = b->len;
                if (id_len > UINT32_MAX || dsml_len > UINT32_MAX) continue;
                if (count == UINT32_MAX) return false;
                if (UINT64_MAX - bytes < 8u ||
                    UINT64_MAX - bytes - 8u < (uint64_t)id_len ||
                    UINT64_MAX - bytes - 8u - (uint64_t)id_len < (uint64_t)dsml_len)
                    return false;
                count++;
                bytes += 8u + (uint64_t)id_len + (uint64_t)dsml_len;
            }
        }
        p = end;
    }
    if (count == 0) bytes = 0;
    if (count_out) *count_out = count;
    if (bytes_out) *bytes_out = bytes;
    return true;
}

static bool kv_tool_map_serialized_size(server *s, const char *text,
                                        uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!s || s->disable_exact_dsml_tool_replay || !text || !text[0]) return true;

    pthread_mutex_lock(&s->tool_mu);
    bool ok = kv_tool_map_measure_locked(s, text, NULL, bytes_out);
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}

static bool kv_tool_map_write(server *s, FILE *fp, const char *text,
                              uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!s || s->disable_exact_dsml_tool_replay || !fp || !text || !text[0]) return true;

    pthread_mutex_lock(&s->tool_mu);
    uint32_t count = 0;
    uint64_t bytes = 0;
    bool ok = kv_tool_map_measure_locked(s, text, &count, &bytes);
    if (!ok) {
        pthread_mutex_unlock(&s->tool_mu);
        return false;
    }
    if (count == 0) {
        pthread_mutex_unlock(&s->tool_mu);
        return true;
    }

    uint8_t h[KV_TOOL_MAP_HEADER];
    h[0] = KV_TOOL_MAP_MAGIC0;
    h[1] = KV_TOOL_MAP_MAGIC1;
    h[2] = KV_TOOL_MAP_MAGIC2;
    h[3] = KV_TOOL_MAP_VERSION;
    le_put32(h + 4, count);
    ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h);

    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_dsml_tool_block(p, &end);
        if (!start || !end || !ok) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; ok && e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t dsml_len = b->len;
                if (id_len > UINT32_MAX || dsml_len > UINT32_MAX) continue;
                uint8_t lens[8];
                le_put32(lens, (uint32_t)id_len);
                le_put32(lens + 4, (uint32_t)dsml_len);
                ok = fwrite(lens, 1, sizeof(lens), fp) == sizeof(lens) &&
                     fwrite(e->id, 1, id_len, fp) == id_len &&
                     fwrite(b->dsml, 1, dsml_len, fp) == dsml_len;
            }
        }
        p = end;
    }
    pthread_mutex_unlock(&s->tool_mu);

    if (ok && written_bytes) *written_bytes = bytes;
    return ok;
}

static int kv_tool_map_load_from_pos(server *s, FILE *fp, const stop_list *wanted) {
    if (!s || s->disable_exact_dsml_tool_replay || !fp) return 0;
    uint8_t h[KV_TOOL_MAP_HEADER];
    size_t n = fread(h, 1, sizeof(h), fp);
    if (n == 0 && feof(fp)) return 0;
    if (n != sizeof(h)) return 0;
    if (h[0] != KV_TOOL_MAP_MAGIC0 || h[1] != KV_TOOL_MAP_MAGIC1 ||
        h[2] != KV_TOOL_MAP_MAGIC2 || h[3] != KV_TOOL_MAP_VERSION) return 0;

    uint32_t count = le_get32(h + 4);
    if ((uint64_t)count > (uint64_t)tool_memory_max_entries(&s->tool_mem) * 4u) return 0;
    int loaded = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint8_t lens[8];
        if (fread(lens, 1, sizeof(lens), fp) != sizeof(lens)) return loaded;
        uint32_t id_len = le_get32(lens);
        uint32_t dsml_len = le_get32(lens + 4);
        if (id_len == 0 || id_len > 256 || dsml_len == 0 ||
            dsml_len > DS4_TOOL_MEMORY_MAX_BYTES) return loaded;
        char *id = xmalloc((size_t)id_len + 1);
        char *dsml = xmalloc((size_t)dsml_len + 1);
        bool ok = fread(id, 1, id_len, fp) == id_len &&
                  fread(dsml, 1, dsml_len, fp) == dsml_len;
        id[id_len] = '\0';
        dsml[dsml_len] = '\0';
        if (ok && (!wanted || id_list_contains(wanted, id))) {
            tool_memory_put_source(s, id, dsml, TOOL_MEMORY_DISK);
            loaded++;
        }
        free(id);
        free(dsml);
        if (!ok) return loaded;
    }
    return loaded;
}

static void kv_fill_header(uint8_t h[KV_CACHE_FIXED_HEADER], uint8_t quant_bits,
                           uint8_t reason, uint8_t ext_flags,
                           uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                           uint64_t created_at, uint64_t last_used,
                           uint64_t payload_bytes) {
    memset(h, 0, KV_CACHE_FIXED_HEADER);
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    le_put32(h + 8, tokens);
    le_put32(h + 12, hits);
    le_put32(h + 16, ctx_size);
    le_put64(h + 24, created_at);
    le_put64(h + 32, last_used);
    le_put64(h + 40, payload_bytes);
}

static bool kv_read_header(FILE *fp, kv_entry *e, uint32_t *text_bytes) {
    uint8_t h[KV_CACHE_FIXED_HEADER];
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2 || h[3] != KV_CACHE_VERSION) return false;
    e->quant_bits = h[4];
    e->reason = h[5] <= KV_REASON_SHUTDOWN ? h[5] : KV_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->tokens = le_get32(h + 8);
    e->hits = le_get32(h + 12);
    e->ctx_size = le_get32(h + 16);
    e->created_at = le_get64(h + 24);
    e->last_used = le_get64(h + 32);
    e->payload_bytes = le_get64(h + 40);
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = le_get32(tb);
    e->text_bytes = *text_bytes;
    return e->tokens != 0 && (e->quant_bits == 2 || e->quant_bits == 4);
}

static bool kv_read_entry_file(const char *path, const char sha[41], kv_entry *out) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < (off_t)(KV_CACHE_FIXED_HEADER + 4)) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    kv_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &e, &text_bytes);
    fclose(fp);
    if (!ok) return false;
    const uint64_t fixed = KV_CACHE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < (uint64_t)text_bytes ||
        UINT64_MAX - fixed - (uint64_t)text_bytes < e.payload_bytes) return false;
    const uint64_t expected = fixed + (uint64_t)text_bytes + e.payload_bytes;
    if ((uint64_t)st.st_size < expected) return false;
    memcpy(e.sha, sha, 41);
    e.path = xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

static void kv_cache_refresh(kv_disk_cache *kc) {
    if (!kc->enabled) return;
    kv_cache_clear(kc);
    DIR *d = opendir(kc->dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!sha_hex_name(de->d_name, sha)) continue;
        char *path = path_join(kc->dir, de->d_name);
        kv_entry e = {0};
        if (kv_read_entry_file(path, sha, &e)) kv_cache_push(kc, e);
        free(path);
    }
    closedir(d);
}

static bool kv_cache_touch_file(const char *path, uint32_t hits) {
    FILE *fp = fopen(path, "r+b");
    if (!fp) return false;
    kv_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &e, &text_bytes);
    if (ok) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        uint64_t now = (uint64_t)time(NULL);
        kv_fill_header(h, e.quant_bits, e.reason, e.ext_flags, e.tokens, hits, e.ctx_size,
                       e.created_at, now, e.payload_bytes);
        ok = fseek(fp, 0, SEEK_SET) == 0 &&
             fwrite(h, 1, sizeof(h), fp) == sizeof(h);
    }
    fclose(fp);
    return ok;
}

static void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs) {
    if (!s || s->disable_exact_dsml_tool_replay || !s->kv.enabled || !msgs) return;
    stop_list wanted = {0};
    collect_tool_call_ids(msgs, &wanted);
    if (wanted.len == 0) return;

    DIR *d = opendir(s->kv.dir);
    if (!d) {
        id_list_free(&wanted);
        return;
    }
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!sha_hex_name(de->d_name, sha)) continue;
        (void)sha;
        char *path = path_join(s->kv.dir, de->d_name);
        FILE *fp = fopen(path, "rb");
        free(path);
        if (!fp) continue;

        kv_entry hdr = {0};
        uint32_t text_bytes = 0;
        bool ok = kv_read_header(fp, &hdr, &text_bytes);
        uint64_t skip = (uint64_t)text_bytes + hdr.payload_bytes;
        if (ok && (hdr.ext_flags & KV_EXT_TOOL_MAP) &&
            skip <= (uint64_t)INT64_MAX &&
            fseeko(fp, (off_t)skip, SEEK_CUR) == 0)
        {
            kv_tool_map_load_from_pos(s, fp, &wanted);
        }
        fclose(fp);
    }
    closedir(d);
    id_list_free(&wanted);
}

static double kv_entry_eviction_score(const kv_entry *e, const ds4_tokens *live,
                                      const char *protected_sha,
                                      uint64_t now) {
    if (!e || e->file_size == 0) return 0.0;
    (void)live;
    if (protected_sha && !strcmp(e->sha, protected_sha)) {
        /* The store path calls eviction immediately after renaming the new
         * checkpoint into place.  Without a protected score, a full cache can
         * choose that brand-new zero-hit file as the cheapest single deletion,
         * turning the save into pure I/O waste.  Treat it as maximally valuable
         * for this eviction pass.  If it is the only remaining entry and the
         * budget is still exceeded, the loop will still delete it below. */
        return DBL_MAX;
    }
    /*
     * Hits count successful disk reuses, but a fresh snapshot is still useful:
     * it may be the only copy of the session that is about to be evicted from
     * RAM.  Continued checkpoints are deliberately aligned restart frontiers,
     * so do not demote them just because they are a prefix of the live session.
     * Use hits+1 for eviction value so a just-written checkpoint does not get
     * deleted immediately just because its persisted hit counter is still 0.
     *
     * Fold time-since-last_used into the hit bonus so a workload change does
     * not let once-popular files outrank live prefixes forever.  Effective
     * hits halve every KV_CACHE_HIT_HALF_LIFE_SECONDS of inactivity.  A fresh
     * hit rewrites last_used, so matching prompts re-boost the entry.  Once
     * the bonus is below one hundredth of a hit, treat it as zero; this lets
     * the last_used tie-breaker evict stale files against equally dense fresh
     * checkpoints instead of preserving a microscopic floating-point bonus.
     */
    double effective_hits = (double)e->hits;
    uint64_t used_at = e->last_used ? e->last_used : e->created_at;
    if (used_at == 0) {
        effective_hits = 0.0;
    } else if (now > used_at) {
        double elapsed = (double)(now - used_at);
        effective_hits *= exp2(-elapsed / (double)KV_CACHE_HIT_HALF_LIFE_SECONDS);
        if (effective_hits < KV_CACHE_MIN_EFFECTIVE_HITS) effective_hits = 0.0;
    }
    return (effective_hits + 1.0) * (double)e->tokens / (double)e->file_size;
}

static void kv_cache_evict(kv_disk_cache *kc, const ds4_tokens *live,
                           const char *protected_sha) {
    if (!kc->enabled || kc->budget_bytes == 0) return;
    kv_cache_refresh(kc);
    const uint64_t now = (uint64_t)time(NULL);
    uint64_t total = 0;
    for (int i = 0; i < kc->len; i++) total += kc->entry[i].file_size;
    if (protected_sha) {
        uint64_t protected_size = 0;
        for (int i = 0; i < kc->len; i++) {
            if (!strcmp(kc->entry[i].sha, protected_sha)) {
                protected_size = kc->entry[i].file_size;
                break;
            }
        }
        /* Do not preserve an entry that cannot fit in the budget by itself:
         * protecting it would first delete every older checkpoint, then delete
         * the new one anyway. */
        if (protected_size > kc->budget_bytes) protected_sha = NULL;
    }
    while (total > kc->budget_bytes && kc->len > 0) {
        int victim = 0;
        double victim_score = kv_entry_eviction_score(&kc->entry[0], live, protected_sha, now);
        for (int i = 1; i < kc->len; i++) {
            double score = kv_entry_eviction_score(&kc->entry[i], live, protected_sha, now);
            if (score < victim_score ||
                (score == victim_score && kc->entry[i].last_used < kc->entry[victim].last_used))
            {
                victim = i;
                victim_score = score;
            }
        }
        kv_entry e = kc->entry[victim];
        if (unlink(e.path) == 0) {
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache evicted reason=disk-cache-full tokens=%u hits=%u size=%.2f MiB file=%s",
                       e.tokens,
                       e.hits,
                       (double)e.file_size / (1024.0 * 1024.0),
                       e.path ? e.path : "?");
            if (total >= e.file_size) total -= e.file_size;
            else total = 0;
        } else {
            total = 0;
        }
        kv_entry_free(&e);
        memmove(kc->entry + victim, kc->entry + victim + 1,
                (size_t)(kc->len - victim - 1) * sizeof(kc->entry[0]));
        kc->len--;
    }
}

static bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb,
                          bool reject_different_quant, kv_cache_options opt) {
    memset(kc, 0, sizeof(*kc));
    if (!dir) return false;
    if (!mkdir_p(dir)) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to create KV cache directory %s: %s", dir, strerror(errno));
        return false;
    }
    kc->enabled = true;
    kc->dir = xstrdup(dir);
    if (budget_mb == 0) budget_mb = KV_CACHE_DEFAULT_MB;
    kc->budget_bytes = budget_mb * 1024ull * 1024ull;
    kc->reject_different_quant = reject_different_quant;
    kc->opt = opt;
    kv_cache_evict(kc, NULL, NULL);
    server_log(DS4_LOG_KVCACHE,
               "ds4-server: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d, hit_half_life=%llus)",
               kc->dir,
               (unsigned long long)(kc->budget_bytes / (1024ull * 1024ull)),
               reject_different_quant ? "reject" : "accept",
               kc->opt.min_tokens,
               kc->opt.cold_max_tokens,
               kc->opt.continued_interval_tokens,
               kc->opt.boundary_trim_tokens,
               kc->opt.boundary_align_tokens,
               (unsigned long long)KV_CACHE_HIT_HALF_LIFE_SECONDS);
    return true;
}

static void kv_cache_close(kv_disk_cache *kc) {
    kv_cache_clear(kc);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

static char *render_tokens_text(ds4_engine *engine, const ds4_tokens *tokens, size_t *out_len) {
    buf b = {0};
    for (int i = 0; i < tokens->len; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, tokens->v[i], &len);
        buf_append(&b, piece, len);
        free(piece);
    }
    if (out_len) *out_len = b.len;
    return buf_take(&b);
}

static bool byte_prefix_match(const char *text, size_t text_len,
                              const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

static const char *kv_cache_key_kind(uint8_t ext_flags) {
    if (ext_flags & KV_EXT_RESPONSES_VISIBLE) return "responses-visible";
    if (ext_flags & KV_EXT_THINKING_VISIBLE) return "thinking-visible";
    return "token-text";
}

static void tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n) {
    dst->len = 0;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, src->v[i]);
}

static void tokens_append(ds4_tokens *dst, const ds4_tokens *src) {
    if (!dst || !src) return;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

static void build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out)
{
    ds4_tokens_copy(out, exact_prefix);

    ds4_tokens suffix = {0};
    /* The suffix may start with DS4 chat markers such as <｜User｜> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE. */
    ds4_tokenize_rendered_chat(engine, suffix_text ? suffix_text : "", &suffix);
    tokens_append(out, &suffix);
    ds4_tokens_free(&suffix);
}

static int kv_cache_store_len(const kv_disk_cache *kc, int tokens) {
    const int trim = kc->opt.boundary_trim_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (tokens > kc->opt.min_tokens + trim) {
        int stable = tokens - trim;
        if (align > 0) stable -= stable % align;
        if (stable >= kc->opt.min_tokens) return stable;
    }
    return tokens;
}

static int kv_cache_chat_anchor_pos(const kv_disk_cache *kc,
                                    const ds4_tokens *prompt,
                                    int user_token_id,
                                    int assistant_token_id) {
    if (!prompt || user_token_id < 0 || assistant_token_id < 0) return -1;

    /* Cold checkpoints are meant to maximize reuse across independent agent
     * sessions.  The stable part of a rendered chat is everything before the
     * user message that asks the model to do this specific task.  Some clients
     * put stable user-role scaffolding first, for example Codex renders:
     *
     *   <｜User｜><environment_context>...</environment_context><｜User｜>task
     *
     * So use the last user marker before the first assistant marker, not the
     * first one.  Stop at the first assistant marker so multi-turn histories do
     * not turn old conversation content into low-value cold cache files; live
     * and continued checkpoints cover same-session continuation instead.
     *
     * The returned value is a token prefix length and deliberately excludes the
     * selected <｜User｜> token.  This is an exact special-token boundary, so the
     * generic trim/align heuristic is not needed to avoid tokenizer merges. */
    int last_user = -1;
    for (int i = 0; i < prompt->len; i++) {
        const int token = prompt->v[i];
        if (token == assistant_token_id) break;
        if (token == user_token_id) last_user = i;
    }
    return last_user >= kc->opt.min_tokens ? last_user : -1;
}

static int kv_cache_continued_step(const kv_disk_cache *kc) {
    if (!kc->enabled || kc->opt.continued_interval_tokens <= 0) return 0;
    int step = kc->opt.continued_interval_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (align > 0) {
        step = ((step + align - 1) / align) * align;
        if (step <= 0) step = align;
    }
    return step;
}

static int kv_cache_continued_store_target(const kv_disk_cache *kc, int live_tokens) {
    const int step = kv_cache_continued_step(kc);
    if (step <= 0) return 0;
    if (live_tokens < kc->opt.min_tokens) return 0;

    /* Continued checkpoints cannot roll the live graph backward.  Save only
     * at absolute aligned frontiers, not relative to the last cold/evict file.
     * Otherwise an early cold checkpoint can shift the whole schedule and leave
     * long generations with no recent durable restart point. */
    if (live_tokens % step != 0) return 0;
    if (live_tokens <= kc->continued_last_store_tokens) return 0;
    return live_tokens;
}

/* A same-text-prefix file can be reused by a larger context, but not by a
 * smaller one: the payload was validated against the context capacity recorded
 * in the file.  If the existing file cannot be used by this server, replace it
 * so this context can still populate its own cache. */
static bool kv_cache_file_text_matches(const char *path, const char sha[41],
                                       const char *text, size_t text_len) {
    if (text_len > UINT32_MAX) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    kv_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &hdr, &text_bytes) &&
              text_bytes == (uint32_t)text_len;
    char *stored = NULL;
    if (ok) {
        stored = xmalloc((size_t)text_bytes + 1);
        ok = fread(stored, 1, text_bytes, fp) == text_bytes;
    }
    fclose(fp);
    if (!ok) {
        free(stored);
        return false;
    }

    char stored_sha[41];
    sha1_bytes_hex(stored, text_bytes, stored_sha);
    ok = !strcmp(stored_sha, sha) &&
         (text_len == 0 || memcmp(stored, text, text_len) == 0);
    free(stored);
    return ok;
}

static bool kv_cache_file_size_bytes(uint64_t text_bytes,
                                     uint64_t payload_bytes,
                                     uint64_t tool_map_bytes,
                                     uint64_t *file_bytes) {
    const uint64_t fixed = KV_CACHE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < text_bytes ||
        UINT64_MAX - fixed - text_bytes < payload_bytes ||
        UINT64_MAX - fixed - text_bytes - payload_bytes < tool_map_bytes)
        return false;
    if (file_bytes) *file_bytes = fixed + text_bytes + payload_bytes + tool_map_bytes;
    return true;
}

static bool kv_cache_budget_required(uint64_t file_bytes,
                                     uint64_t *required_bytes) {
    /* The serialized size is deterministic for one snapshot, including the
     * optional tool map.  We still reserve 1% headroom so filesystem/accounting
     * surprises or a concurrently-added tool mapping cannot produce a file that
     * is immediately removed by the cache budget pass. */
    uint64_t slack = file_bytes / 100u;
    if (file_bytes % 100u) slack++;
    if (UINT64_MAX - file_bytes < slack) return false;
    if (required_bytes) *required_bytes = file_bytes + slack;
    return true;
}

static bool kv_cache_file_size_fits(const kv_disk_cache *kc,
                                    uint64_t text_bytes,
                                    uint64_t payload_bytes,
                                    uint64_t tool_map_bytes,
                                    uint64_t *file_bytes_out,
                                    uint64_t *required_bytes_out) {
    uint64_t file_bytes = 0;
    if (!kv_cache_file_size_bytes(text_bytes, payload_bytes, tool_map_bytes,
                                  &file_bytes))
        return false;
    if (file_bytes_out) *file_bytes_out = file_bytes;
    if (!kc || kc->budget_bytes == 0) return true;
    uint64_t required = 0;
    if (!kv_cache_budget_required(file_bytes, &required)) return false;
    if (required_bytes_out) *required_bytes_out = required;
    return required <= kc->budget_bytes;
}

static bool kv_cache_existing_compatible(kv_disk_cache *kc, const char *path,
                                         const char sha[41],
                                         const char *text, size_t text_len,
                                         int quant_bits, int ctx_size) {
    if (access(path, F_OK) != 0) return false;
    kv_entry e = {0};
    if (!kv_read_entry_file(path, sha, &e)) return false;
    bool compatible = (!kc->reject_different_quant || e.quant_bits == (uint8_t)quant_bits) &&
                      e.ctx_size <= (uint32_t)ctx_size &&
                      kv_cache_file_text_matches(path, sha, text, text_len);
    kv_entry_free(&e);
    if (!compatible) {
        if (unlink(path) == 0) {
            server_log(DS4_LOG_KVCACHE, "ds4-server: kv cache replaced incompatible file %s", path);
        }
        return false;
    }
    return true;
}

static void kv_cache_rewrite_tool_map(server *s, const char *path, const char *text) {
    if (!s || !path || !text || tool_memory_count_dsml_in_text(s, text) == 0) return;
    FILE *fp = fopen(path, "r+b");
    if (!fp) return;
    kv_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &hdr, &text_bytes);
    uint64_t end = KV_CACHE_FIXED_HEADER + 4ull + (uint64_t)text_bytes + hdr.payload_bytes;
    if (ok && end <= (uint64_t)INT64_MAX &&
        fseeko(fp, (off_t)end, SEEK_SET) == 0 &&
        ftruncate(fileno(fp), (off_t)end) == 0)
    {
        uint64_t ignored = 0;
        ok = kv_tool_map_write(s, fp, text, &ignored) && fflush(fp) == 0;
        if (ok && ignored > 0) {
            uint8_t h[KV_CACHE_FIXED_HEADER];
            uint64_t now = (uint64_t)time(NULL);
            kv_fill_header(h, hdr.quant_bits, hdr.reason,
                           (uint8_t)(hdr.ext_flags | KV_EXT_TOOL_MAP),
                           hdr.tokens, hdr.hits, hdr.ctx_size,
                           hdr.created_at, now, hdr.payload_bytes);
            ok = fseeko(fp, 0, SEEK_SET) == 0 &&
                 fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
                 fflush(fp) == 0;
        }
    }
    fclose(fp);
    (void)ok;
}

static bool kv_cache_store_live_prefix_text(server *s, const ds4_tokens *tokens,
                                            int store_len, const char *reason,
                                            const char *cache_text_override,
                                            uint8_t cache_text_ext,
                                            const char *cache_text_key) {
    kv_disk_cache *kc = &s->kv;
    if (!kc->enabled) return false;
    if (!tokens || store_len < kc->opt.min_tokens) return false;
    const int original_len = tokens->len;

    ds4_tokens store_tokens = {0};
    tokens_copy_prefix(&store_tokens, tokens, store_len);

    const int quant_bits = ds4_engine_routed_quant_bits(s->engine);
    if (quant_bits != 2 && quant_bits != 4) {
        ds4_tokens_free(&store_tokens);
        return false;
    }
    char err[160] = {0};
    /* Disk cache persistence must observe the graph exactly as-is.  If callers
     * want a shorter prefix, they first prefill to that prefix and only then call
     * this function.  This keeps cache population from doing hidden inference. */
    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens ||
        live_tokens->len != store_tokens.len ||
        !ds4_tokens_starts_with(live_tokens, &store_tokens))
    {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                   store_tokens.len,
                   reason,
                   live_tokens ? live_tokens->len : -1);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    uint64_t payload_bytes = ds4_session_payload_bytes(s->session);
    if (payload_bytes == 0) {
        ds4_tokens_free(&store_tokens);
        return false;
    }

    size_t text_len = 0;
    char *text = NULL;
    const bool text_override = cache_text_override && cache_text_override[0];
    if (text_override) {
        text = xstrdup(cache_text_override);
        text_len = strlen(text);
    } else {
        text = render_tokens_text(s->engine, &store_tokens, &text_len);
    }
    if (text_len > UINT32_MAX) {
        server_log(DS4_LOG_KVCACHE, "ds4-server: kv cache skipped tokens=%d because rendered text is too large", store_tokens.len);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    uint64_t tool_map_est_bytes = 0;
    if (!kv_tool_map_serialized_size(s, text, &tool_map_est_bytes)) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache skipped tokens=%d reason=%s because tool map size overflowed",
                   store_tokens.len, reason);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    uint64_t est_file_bytes = 0, est_required_bytes = 0;
    if (!kv_cache_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                 tool_map_est_bytes,
                                 &est_file_bytes, &est_required_bytes)) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache skipped tokens=%d reason=%s because estimated file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB",
                   store_tokens.len,
                   reason,
                   (double)est_file_bytes / (1024.0 * 1024.0),
                   (double)est_required_bytes / (1024.0 * 1024.0),
                   (double)kc->budget_bytes / (1024.0 * 1024.0));
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    char sha[41];
    sha1_bytes_hex(text, text_len, sha);
    char *path = kv_path_for_sha(kc, sha);

    if (kv_cache_existing_compatible(kc, path, sha, text, text_len,
                                     quant_bits, ds4_session_ctx(s->session))) {
        kv_cache_rewrite_tool_map(s, path, text);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return true;
    }

    buf tmpb = {0};
    buf_printf(&tmpb, "%s.tmp.%ld", path, (long)getpid());
    char *tmp = buf_take(&tmpb);
    const double save_t0 = now_sec();
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        server_log(DS4_LOG_KVCACHE, "ds4-server: kv cache failed to create %s: %s save=%.1f ms",
                   tmp, strerror(errno), (now_sec() - save_t0) * 1000.0);
        free(tmp);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    const uint64_t now = (uint64_t)time(NULL);
    uint8_t h[KV_CACHE_FIXED_HEADER];
    uint8_t ext_flags = tool_map_est_bytes > 0 ? KV_EXT_TOOL_MAP : 0;
    if (text_override) ext_flags |= cache_text_ext;
    kv_fill_header(h, (uint8_t)quant_bits, kv_reason_code(reason), ext_flags,
                   (uint32_t)store_tokens.len, 0,
                   (uint32_t)ds4_session_ctx(s->session), now, now, payload_bytes);
    uint8_t tb[4];
    le_put32(tb, (uint32_t)text_len);
    uint64_t tool_map_bytes = 0;
    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              ds4_session_save_payload(s->session, fp, err, sizeof(err)) == 0 &&
              kv_tool_map_write(s, fp, text, &tool_map_bytes) &&
              fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    uint64_t final_file_bytes = 0, final_required_bytes = 0;
    bool final_size_over_budget = false;
    if (ok && !kv_cache_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                       tool_map_bytes,
                                       &final_file_bytes, &final_required_bytes))
    {
        final_size_over_budget = true;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    const double save_ms = (now_sec() - save_t0) * 1000.0;
    if (!ok) {
        if (final_size_over_budget) {
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache skipped tokens=%d reason=%s because final file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB save=%.1f ms",
                       store_tokens.len,
                       reason,
                       (double)final_file_bytes / (1024.0 * 1024.0),
                       (double)final_required_bytes / (1024.0 * 1024.0),
                       (double)kc->budget_bytes / (1024.0 * 1024.0),
                       save_ms);
        } else {
            server_log(DS4_LOG_KVCACHE, "ds4-server: kv cache store failed (%s): %s save=%.1f ms",
                       reason,
                       saved_errno ? strerror(saved_errno) : (err[0] ? err : "unknown error"),
                       save_ms);
        }
        unlink(tmp);
    } else {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache stored tokens=%d trimmed=%d reason=%s key=%s size=%.2f MiB save=%.1f ms",
                   store_tokens.len,
                   original_len - store_tokens.len,
                   reason,
                   text_override ? (cache_text_key ? cache_text_key : "visible-transcript") : "token-text",
                   (double)(KV_CACHE_FIXED_HEADER + 4ull + text_len + payload_bytes + tool_map_bytes) / (1024.0 * 1024.0),
                   save_ms);
        kv_cache_evict(kc, live_tokens, sha);
    }
    free(tmp);
    free(text);
    free(path);
    ds4_tokens_free(&store_tokens);
    return ok;
}

static bool kv_cache_store_live_prefix(server *s, const ds4_tokens *tokens,
                                       int store_len, const char *reason) {
    return kv_cache_store_live_prefix_text(s, tokens, store_len, reason,
                                           NULL, 0, NULL);
}

static void kv_cache_store_current(server *s, const char *reason) {
    const ds4_tokens *tokens = ds4_session_tokens(s->session);
    if (!tokens) return;

    char *visible_text = NULL;
    uint8_t visible_ext = 0;
    const char *visible_key = NULL;
    pthread_mutex_lock(&s->tool_mu);
    if (s->responses_live.valid &&
        s->responses_live.live_tokens == tokens->len &&
        s->responses_live.visible_text &&
        s->responses_live.visible_text[0])
    {
        visible_text = xstrdup(s->responses_live.visible_text);
        visible_ext = KV_EXT_RESPONSES_VISIBLE;
        visible_key = "responses-visible";
    } else if (s->thinking_live.valid &&
               s->thinking_live.live_tokens == tokens->len &&
               s->thinking_live.visible_text &&
               s->thinking_live.visible_text[0])
    {
        visible_text = xstrdup(s->thinking_live.visible_text);
        visible_ext = KV_EXT_THINKING_VISIBLE;
        visible_key = "thinking-visible";
    }
    pthread_mutex_unlock(&s->tool_mu);

    /* A visible live checkpoint can contain hidden reasoning that the client
     * intentionally does not replay.  For disk recovery after a session switch,
     * key that payload by the visible protocol transcript, not by rendering the
     * hidden sampled tokens.  On load, DS4 restores the hidden KV payload and
     * tokenizes only the visible suffix that follows this key. */
    if (visible_text) {
        kv_cache_store_live_prefix_text(s, tokens, tokens->len, reason,
                                        visible_text, visible_ext, visible_key);
        free(visible_text);
    } else {
        kv_cache_store_live_prefix(s, tokens, tokens->len, reason);
    }
}

static void kv_cache_note_store(kv_disk_cache *kc, int tokens) {
    if (tokens > kc->continued_last_store_tokens) {
        kc->continued_last_store_tokens = tokens;
    }
}

static int kv_cache_suppress_continued_store(kv_disk_cache *kc, int tokens) {
    if (kv_cache_continued_store_target(kc, tokens) != tokens) return -1;
    int old = kc->continued_last_store_tokens;
    kv_cache_note_store(kc, tokens);
    return old;
}

static void kv_cache_restore_suppressed_continued(kv_disk_cache *kc,
                                                  int old_tokens,
                                                  int suppressed_tokens) {
    if (old_tokens >= 0 && kc->continued_last_store_tokens == suppressed_tokens) {
        kc->continued_last_store_tokens = old_tokens;
    }
}

static void kv_cache_maybe_store_continued(server *s) {
    kv_disk_cache *kc = &s->kv;
    const ds4_tokens *tokens = ds4_session_tokens(s->session);
    if (!tokens) return;
    const int target = kv_cache_continued_store_target(kc, tokens->len);
    if (target == 0) return;
    if (kv_cache_store_live_prefix(s, tokens, target, "continued")) {
        kv_cache_note_store(kc, target);
    }
}

static int kv_cache_find_text_prefix(kv_disk_cache *kc, const char *prompt_text,
                                     int quant_bits, int ctx_size) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    kv_cache_refresh(kc);
    int best = -1;
    for (int i = 0; i < kc->len; i++) {
        kv_entry *e = &kc->entry[i];
        if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
        if ((int)e->tokens < kc->opt.min_tokens) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        if (best >= 0) {
            kv_entry *b = &kc->entry[best];
            if (e->text_bytes < b->text_bytes) continue;
            if (e->text_bytes == b->text_bytes && e->tokens <= b->tokens) continue;
        }
        char sha[41];
        sha1_bytes_hex(prompt_text, (size_t)e->text_bytes, sha);
        if (!strcmp(sha, e->sha)) best = i;
    }
    return best;
}

static int kv_cache_try_load_text(server *s, const char *prompt_text,
                                  ds4_tokens *effective_prompt,
                                  char **loaded_path_out,
                                  uint8_t *loaded_ext_flags_out,
                                  bool responses_protocol) {
    if (loaded_path_out) *loaded_path_out = NULL;
    if (loaded_ext_flags_out) *loaded_ext_flags_out = 0;
    if (effective_prompt) effective_prompt->len = 0;
    kv_disk_cache *kc = &s->kv;
    if (!kc->enabled || !prompt_text) return 0;
    const int quant_bits = ds4_engine_routed_quant_bits(s->engine);
    if (quant_bits != 2 && quant_bits != 4) return 0;
    const size_t prompt_bytes = strlen(prompt_text);
    int idx = kv_cache_find_text_prefix(kc, prompt_text, quant_bits,
                                        ds4_session_ctx(s->session));
    if (idx < 0) return 0;

    kv_entry e = kc->entry[idx];
    char *path = xstrdup(e.path);
    const double load_t0 = now_sec();
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return 0;
    }
    uint32_t text_bytes = 0;
    kv_entry hdr = {0};
    const char *fail_reason = "invalid header";
    bool header_ok = kv_read_header(fp, &hdr, &text_bytes);
    char *cached_text = NULL;
    if (header_ok) {
        if ((uint64_t)text_bytes > prompt_bytes) {
            header_ok = false;
            fail_reason = "cached text is longer than prompt";
        } else {
            cached_text = xmalloc((size_t)text_bytes + 1);
            if (fread(cached_text, 1, text_bytes, fp) != text_bytes) {
                header_ok = false;
                fail_reason = "truncated cached text";
            } else {
                cached_text[text_bytes] = '\0';
                char text_sha[41];
                sha1_bytes_hex(cached_text, text_bytes, text_sha);
                if (strcmp(text_sha, e.sha)) {
                    header_ok = false;
                    fail_reason = "cached text hash mismatch";
                } else if (!byte_prefix_match(prompt_text, prompt_bytes,
                                              cached_text, text_bytes)) {
                    header_ok = false;
                    fail_reason = "cached text prefix mismatch";
                }
            }
        }
    }
    char err[160] = {0};
    int loaded = 0;
    if (header_ok && ds4_session_load_payload(s->session, fp, hdr.payload_bytes, err, sizeof(err)) == 0) {
        const ds4_tokens *loaded_tokens = ds4_session_tokens(s->session);
        if (loaded_tokens && loaded_tokens->len == (int)hdr.tokens) {
            loaded = (int)hdr.tokens;
            if (effective_prompt) {
                /* The cache lookup was by bytes, but the graph state is still
                 * the exact token history stored in the payload.  Build the
                 * prompt we give ds4_session_sync() from that exact history and
                 * tokenize only the text suffix after the byte prefix. */
                build_prompt_from_exact_prefix_and_text_suffix(
                    s->engine, loaded_tokens, prompt_text + text_bytes,
                    effective_prompt);
            }
            if (hdr.ext_flags & KV_EXT_TOOL_MAP) kv_tool_map_load_from_pos(s, fp, NULL);
        } else {
            ds4_session_invalidate(s->session);
            unlink(path);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache discarded corrupt text-prefix payload%s%s %s",
                       responses_protocol ? " " : "",
                       responses_protocol ? "RESPPROTO" : "",
                       path);
        }
    } else {
        if (header_ok) ds4_session_invalidate(s->session);
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache load failed%s%s %s: %s load=%.1f ms",
                   responses_protocol ? " " : "",
                   responses_protocol ? "RESPPROTO" : "",
                   path,
                   header_ok ? err : fail_reason,
                   (now_sec() - load_t0) * 1000.0);
    }
    fclose(fp);

    if (loaded > 0) {
        const double load_ms = (now_sec() - load_t0) * 1000.0;
        if (loaded_path_out) *loaded_path_out = xstrdup(path);
        if (loaded_ext_flags_out) *loaded_ext_flags_out = hdr.ext_flags;
        kc->continued_last_store_tokens = loaded;
        const char *key_kind = kv_cache_key_kind(hdr.ext_flags);
        if (kc->opt.cold_max_tokens > 0 && loaded > kc->opt.cold_max_tokens) {
            unlink(path);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms consumed file=%s",
                       responses_protocol ? " " : "",
                       responses_protocol ? "RESPPROTO" : "",
                       loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        } else {
            kv_cache_touch_file(path, hdr.hits + 1);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s",
                       responses_protocol ? " " : "",
                       responses_protocol ? "RESPPROTO" : "",
                       loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        }
    }
    free(cached_text);
    free(path);
    return loaded;
}

static int kv_cache_try_load(server *s, const request *req,
                             ds4_tokens *effective_prompt,
                             char **loaded_path_out,
                             uint8_t *loaded_ext_flags_out) {
    return kv_cache_try_load_text(s, req ? req->prompt_text : NULL,
                                  effective_prompt,
                                  loaded_path_out,
                                  loaded_ext_flags_out,
                                  req && req->api == API_RESPONSES);
}

static int live_text_prefix_prompt(server *s, const request *req,
                                   ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len <= 0) return 0;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, live_tokens, &live_text_len);
    const size_t prompt_text_len = strlen(req->prompt_text);
    if (!byte_prefix_match(req->prompt_text, prompt_text_len,
                           live_text, live_text_len))
    {
        free(live_text);
        return 0;
    }

    /* This is the core text-prefix case.  The live graph is authoritative, so
     * keep its sampled tokenization and tokenize only the request bytes that
     * come after it.  Reusing req->prompt's token suffix would be wrong: full
     * prompt BPE may have merged across this byte boundary. */
    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + live_text_len,
        effective_prompt);
    free(live_text);
    return live_tokens->len;
}

/* Tool-output-only Responses continuation.
 *
 * Some clients send just the new tool outputs after a tool call.  There is no
 * long visible prefix to match in that shape; the call_id itself is the
 * protocol binding to the previous live assistant output.  Use it only when the
 * remembered live frontier and call-id set match exactly. */
static int responses_live_continuation_prompt(server *s, const request *req,
                                              int live_pos,
                                              ds4_tokens *effective_prompt,
                                              int *matched_ids) {
    if (!s || !req || !effective_prompt) return 0;
    if (req->api != API_RESPONSES || !req->responses_live_suffix_text) return 0;
    if (req->responses_live_call_ids.len == 0) return 0;
    if (!responses_live_matches_request(s, &req->responses_live_call_ids,
                                        live_pos)) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->responses_live_suffix_text,
        effective_prompt);
    if (matched_ids) *matched_ids = req->responses_live_call_ids.len;
    return live_tokens->len;
}

/* Tool-result Anthropic continuation.
 *
 * /v1/messages has no server-side response object like the OpenAI Responses
 * API, but its tool_use_id is still a precise continuation handle inside a live
 * local agent loop.  When the IDs and live token frontier match, continue from
 * the sampled DSML state and append only the user tool_result suffix. */
static int anthropic_live_continuation_prompt(server *s, const request *req,
                                              int live_pos,
                                              ds4_tokens *effective_prompt,
                                              int *matched_ids) {
    if (!s || !req || !effective_prompt) return 0;
    if (req->api != API_ANTHROPIC || !req->anthropic_live_suffix_text) return 0;
    if (req->anthropic_live_call_ids.len == 0) return 0;
    if (!anthropic_live_matches_request(s, &req->anthropic_live_call_ids,
                                        live_pos)) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->anthropic_live_suffix_text,
        effective_prompt);
    if (matched_ids) *matched_ids = req->anthropic_live_call_ids.len;
    return live_tokens->len;
}

/* Visible-replay Responses continuation.
 *
 * Other clients send the full visible transcript on every turn even though the
 * API semantics still make the request a continuation.  For Responses, exact
 * token-prefix matching is the wrong first question: hidden reasoning may be
 * live in KV but absent from the replay by design.  Instead, verify that the
 * request's rendered text begins with the visible transcript remembered at the
 * live frontier.  If it does, continue from the live token prefix and tokenize
 * only the bytes after that visible boundary.
 *
 * If this check fails, DS4 has no special Responses state to trust.  The caller
 * then uses normal token/text/disk matching, which is the correct fallback for
 * cold starts, edits, restarts, or cross-client replays. */
static int responses_live_visible_prefix_prompt(server *s, const request *req,
                                                int live_pos,
                                                ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->api != API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->responses_live.valid &&
              s->responses_live.live_tokens == live_pos &&
              s->responses_live.visible_text &&
              s->responses_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                s->responses_live.visible_text,
                                s->responses_live.visible_len);
    if (ok) visible_len = s->responses_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        effective_prompt);
    return live_tokens->len;
}

/* Tool-less thinking continuation.
 *
 * Chat/completions and Anthropic do not have a previous_response_id object that
 * binds a later request to the last sampled turn.  Still, after a normal
 * tool-less thinking answer, the next prompt renderer intentionally omits that
 * hidden reasoning.  The live KV state is richer than the visible transcript.
 *
 * Remembering the visible transcript as a key lets us keep the sampled hidden
 * KV when the next request clearly extends that same visible history.  This is
 * the same byte-prefix idea used by the disk cache: the client-visible text
 * selects the checkpoint, while the payload stays the exact sampled token
 * frontier.  If the visible key does not match, callers fall back to ordinary
 * token/text/disk matching. */
static int thinking_live_visible_prefix_prompt(server *s, const request *req,
                                               int live_pos,
                                               ds4_tokens *effective_prompt) {
    if (!s || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->kind != REQ_CHAT || req->api == API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = s->thinking_live.valid &&
              s->thinking_live.live_tokens == live_pos &&
              s->thinking_live.visible_text &&
              s->thinking_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                s->thinking_live.visible_text,
                                s->thinking_live.visible_len);
    if (ok) visible_len = s->thinking_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(s->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        effective_prompt);
    return live_tokens->len;
}

/* =========================================================================
 * Trace Diagnostics.
 * =========================================================================
 *
 * The human transcript is not enough to debug prompt-cache misses.  The model
 * may generate text that is semantically accepted as a tool call, while the
 * next OpenAI request re-renders a slightly different canonical DSML block.
 * That creates a token mismatch even if the conversation "looks" continuous.
 *
 * When --trace is enabled we therefore record the exact cache decision and a
 * small token window around the first mismatch between the live KV checkpoint
 * and the incoming prompt.  Normal server logs stay compact; trace files get
 * enough data to diagnose tokenizer-boundary and canonicalization problems.
 */

#define TRACE_CACHE_BEFORE 8
#define TRACE_CACHE_AFTER  8
#define TRACE_CACHE_WINDOW (TRACE_CACHE_BEFORE + 1 + TRACE_CACHE_AFTER)

typedef struct {
    bool valid;
    int old_pos;
    int prompt_len;
    int common;
    int start;
    int count;
    int live_id[TRACE_CACHE_WINDOW];
    int prompt_id[TRACE_CACHE_WINDOW];
} trace_cache_diag;

static void trace_cache_capture(
        trace_cache_diag *d,
        const ds4_tokens *live,
        const ds4_tokens *prompt,
        int old_pos,
        int common)
{
    memset(d, 0, sizeof(*d));
    d->valid = true;
    d->old_pos = old_pos;
    d->prompt_len = prompt ? prompt->len : 0;
    d->common = common;

    const int live_len = live ? live->len : 0;
    const int prompt_len = prompt ? prompt->len : 0;
    int max_len = live_len > prompt_len ? live_len : prompt_len;
    int start = common - TRACE_CACHE_BEFORE;
    if (start < 0) start = 0;
    int end = common + TRACE_CACHE_AFTER + 1;
    if (end > max_len) end = max_len;
    if (end < start) end = start;

    d->start = start;
    d->count = end - start;
    if (d->count > TRACE_CACHE_WINDOW) d->count = TRACE_CACHE_WINDOW;
    for (int i = 0; i < d->count; i++) {
        int pos = start + i;
        d->live_id[i] = live && pos < live->len ? live->v[pos] : -1;
        d->prompt_id[i] = prompt && pos < prompt->len ? prompt->v[pos] : -1;
    }
}

static const char *trace_cache_miss_reason(const trace_cache_diag *d) {
    if (!d || !d->valid) return "unknown";
    if (d->old_pos == 0) return "no-live-checkpoint";
    if (d->common != d->old_pos) return "token-mismatch";
    if (d->prompt_len < d->old_pos) return "incoming-prompt-shorter-than-live-checkpoint";
    return "live-prefix-match";
}

static void trace_write_escaped_bytes(FILE *fp, const char *p, size_t len) {
    static const char hex[] = "0123456789abcdef";
    fputc('"', fp);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)p[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20 || c == 0x7f) {
            fputs("\\x", fp);
            fputc(hex[c >> 4], fp);
            fputc(hex[c & 15], fp);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void trace_write_token(FILE *fp, ds4_engine *engine, int token) {
    if (token < 0) {
        fputs("- <none>", fp);
        return;
    }
    size_t len = 0;
    char *piece = ds4_token_text(engine, token, &len);
    fprintf(fp, "%d ", token);
    trace_write_escaped_bytes(fp, piece, len);
    free(piece);
}

static void trace_write_cache_diag(
        server *s,
        const trace_cache_diag *d,
        const tool_replay_stats *tool_replay,
        int cached,
        const char *cache_source,
        int disk_cached,
        const char *disk_path)
{
    fprintf(s->trace,
            "\n--- cache decision ---\n"
            "live_tokens_before: %d\n"
            "prompt_tokens: %d\n"
            "live_prompt_common: %d\n"
            "memory_token_reusable: %d\n"
            "memory_miss_reason: %s\n"
            "tool_replay: mem=%d disk=%d canonical=%d missing_ids=%d\n"
            "cache_source: %s\n"
            "cached_tokens: %d\n"
            "disk_cached_tokens: %d\n",
            d && d->valid ? d->old_pos : 0,
            d && d->valid ? d->prompt_len : 0,
            d && d->valid ? d->common : 0,
            d && d->valid && d->old_pos > 0 &&
                d->common == d->old_pos && d->prompt_len >= d->old_pos ? 1 : 0,
            trace_cache_miss_reason(d),
            tool_replay ? tool_replay->mem : 0,
            tool_replay ? tool_replay->disk : 0,
            tool_replay ? tool_replay->canonical : 0,
            tool_replay ? tool_replay->missing_ids : 0,
            cache_source ? cache_source : "none",
            cached,
            disk_cached);
    if (disk_path && disk_path[0]) fprintf(s->trace, "disk_cache_file: %s\n", disk_path);

    if (!d || !d->valid || d->old_pos == 0 ||
        (d->common == d->old_pos && d->prompt_len >= d->old_pos))
    {
        return;
    }

    fprintf(s->trace,
            "\nfirst_mismatch_token: %d\n"
            "token_window: [%d..%d)\n",
            d->common,
            d->start,
            d->start + d->count);
    for (int i = 0; i < d->count; i++) {
        int pos = d->start + i;
        int live = d->live_id[i];
        int prompt = d->prompt_id[i];
        const char *mark;
        if (live < 0) mark = "prompt-only";
        else if (prompt < 0) mark = "live-only";
        else mark = live == prompt ? "==" : "!=";

        fprintf(s->trace, "%7d %-11s live ", pos, mark);
        trace_write_token(s->trace, s->engine, live);
        fputs(" | prompt ", s->trace);
        trace_write_token(s->trace, s->engine, prompt);
        fputc('\n', s->trace);
    }
}

static void trace_time(FILE *fp) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    fputs(buf, fp);
}

static uint64_t trace_begin(
        server *s,
        const job *j,
        int cached,
        int effective_prompt_tokens,
        const trace_cache_diag *cache_diag,
        const char *cache_source,
        int disk_cached,
        const char *disk_path) {
    if (!s->trace) return 0;

    pthread_mutex_lock(&s->trace_mu);
    uint64_t id = ++s->trace_seq;
    fprintf(s->trace, "\n===== request %llu ", (unsigned long long)id);
    trace_time(s->trace);
    fprintf(s->trace,
            " =====\nkind: %s\nmodel: %s\nstream: %d\ntools: %d\nthink_mode: %s\nprompt_tokens: %d\neffective_prompt_tokens: %d\ncached_tokens: %d\nmax_tokens: %d\ntemperature: %.3f\ntop_k: %d\ntop_p: %.3f\nmin_p: %.3f\nseed: %llu\n",
            j->req.kind == REQ_CHAT ? "chat" : "completion",
            j->req.model ? j->req.model : "",
            j->req.stream ? 1 : 0,
            j->req.has_tools ? 1 : 0,
            ds4_think_mode_name(j->req.think_mode),
            j->req.prompt.len,
            effective_prompt_tokens,
            cached,
            j->req.max_tokens,
            j->req.temperature,
            j->req.top_k,
            j->req.top_p,
            j->req.min_p,
            (unsigned long long)j->req.seed);
    fprintf(s->trace, "stream_include_usage: %d\n",
            j->req.stream_include_usage ? 1 : 0);
    trace_write_cache_diag(s, cache_diag, &j->req.tool_replay, cached,
                           cache_source, disk_cached, disk_path);
    if (j->req.raw_body) {
        fputs("\n--- raw request json ---\n", s->trace);
        fputs(j->req.raw_body, s->trace);
        if (!j->req.raw_body[0] || j->req.raw_body[strlen(j->req.raw_body) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    if (j->req.prompt_text) {
        fputs("\n--- rendered prompt ---\n", s->trace);
        fputs(j->req.prompt_text, s->trace);
        if (!j->req.prompt_text[0] || j->req.prompt_text[strlen(j->req.prompt_text) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    fputs("\n--- generated text ---\n", s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
    return id;
}

static void trace_piece(server *s, uint64_t id, const char *piece, size_t len) {
    if (!s->trace || !id || !piece || !len) return;
    pthread_mutex_lock(&s->trace_mu);
    fwrite(piece, 1, len, s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void trace_event(server *s, uint64_t id, const char *fmt, ...) {
    if (!s->trace || !id) return;
    pthread_mutex_lock(&s->trace_mu);
    fputs("\n\n--- trace: ", s->trace);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(s->trace, fmt, ap);
    va_end(ap);
    fputs(" ---\n\n", s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void trace_finish(
        server *s,
        uint64_t id,
        const request *r,
        const char *final_finish,
        int completion,
        bool saw_tool_start,
        bool saw_tool_end,
        const char *parsed_content,
        const char *parsed_reasoning,
        const tool_calls *parsed_calls,
        double elapsed) {
    if (!s->trace || !id) return;

    pthread_mutex_lock(&s->trace_mu);
    fprintf(s->trace,
            "\n\n--- parsed message ---\nfinish: %s\ngenerated_tokens: %d\ndsml_start: %d\ndsml_end: %d\nelapsed_sec: %.3f\n",
            final_finish,
            completion,
            saw_tool_start ? 1 : 0,
            saw_tool_end ? 1 : 0,
            elapsed);
    if (r->kind == REQ_CHAT) {
        if (parsed_reasoning && parsed_reasoning[0]) {
            fputs("\nreasoning:\n", s->trace);
            fputs(parsed_reasoning, s->trace);
            fputc('\n', s->trace);
        }
        if (parsed_content && parsed_content[0]) {
            fputs("\ncontent:\n", s->trace);
            fputs(parsed_content, s->trace);
            fputc('\n', s->trace);
        }
        for (int i = 0; i < parsed_calls->len; i++) {
            const tool_call *tc = &parsed_calls->v[i];
            fprintf(s->trace, "\ntool_call[%d]:\nid: %s\nname: %s\narguments:\n%s\n",
                    i,
                    tc->id ? tc->id : "",
                    tc->name ? tc->name : "",
                    tc->arguments ? tc->arguments : "");
        }
    }
    fprintf(s->trace, "\n===== end request %llu =====\n", (unsigned long long)id);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

typedef struct {
    server *srv;
    req_kind kind;
    int prompt_tokens;
    int cached_tokens;
    char ctx[48];
    const char *phase;
    bool has_tools;
    bool responses_protocol;
    double t0;
    double last_t;
    int last_current;
    bool seen;
} server_prefill_progress;

static void request_ctx_span(char *buf, size_t len, int cached, int prompt) {
    int suffix = prompt - cached;
    if (suffix < 0) suffix = 0;
    snprintf(buf, len, "%d..%d:%d", cached, prompt, suffix);
}

static void log_flags(char *buf, size_t len, bool responses_protocol,
                      bool tools, bool thinking,
                      bool dsml_start, bool dsml_end) {
    size_t used = 0;
    buf[0] = '\0';
#define ADD_FLAG(name) do { \
    int n = snprintf(buf + used, used < len ? len - used : 0, "%s%s", used ? " " : "", name); \
    if (n > 0) used += (size_t)n; \
} while (0)
    if (responses_protocol) ADD_FLAG("RESPPROTO");
    if (tools) ADD_FLAG("TOOLS");
    if (thinking) ADD_FLAG("THINKING");
    if (dsml_start) ADD_FLAG("DSML_START");
    if (dsml_end) ADD_FLAG("DSML_END");
#undef ADD_FLAG
}

static void log_decode_progress(req_kind kind, int prompt_tokens, int completion,
                                bool responses_protocol,
                                bool tools, bool thinking,
                                bool dsml_start, bool dsml_end,
                                double decode_t0,
                                double *last_t, int *last_completion) {
    const double now = now_sec();
    const double elapsed = now - decode_t0;
    const double interval_s = now - *last_t;
    const int interval_tokens = completion - *last_completion;
    const double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    const double avg_tps = elapsed > 0.0 ? (double)completion / elapsed : 0.0;
    char ctx[48];
    request_ctx_span(ctx, sizeof(ctx),
                     prompt_tokens + *last_completion,
                     prompt_tokens + completion);
    char flags[80];
    log_flags(flags, sizeof(flags), responses_protocol,
              tools, thinking, dsml_start, dsml_end);
    server_log(DS4_LOG_GENERATION,
               "ds4-server: %s ctx=%s gen=%d%s%s decoding chunk=%.2f t/s avg=%.2f t/s %.3fs",
               kind == REQ_CHAT ? "chat" : "completion",
               ctx,
               completion,
               flags[0] ? " " : "",
               flags,
               chunk_tps,
               avg_tps,
               elapsed);
    *last_t = now;
    *last_completion = completion;
}

typedef struct {
    bool inside;
    char tail[8]; /* Long enough for "</think>". */
    int tail_len;
} thinking_state;

static bool thinking_tail_ends_with(const thinking_state *st, const char *s) {
    int n = (int)strlen(s);
    return st->tail_len >= n && !memcmp(st->tail + st->tail_len - n, s, (size_t)n);
}

static void thinking_state_feed(thinking_state *st, const char *p, size_t len) {
    if (!st || !p) return;
    for (size_t i = 0; i < len; i++) {
        if (st->tail_len == (int)sizeof(st->tail)) {
            memmove(st->tail, st->tail + 1, sizeof(st->tail) - 1);
            st->tail_len--;
        }
        st->tail[st->tail_len++] = p[i];
        if (thinking_tail_ends_with(st, "<think>")) st->inside = true;
        else if (thinking_tail_ends_with(st, "</think>")) st->inside = false;
    }
}

static thinking_state thinking_state_from_prompt(const request *r) {
    thinking_state st = {0};
    if (r && r->prompt_text) {
        thinking_state_feed(&st, r->prompt_text, strlen(r->prompt_text));
    } else if (r && ds4_think_mode_enabled(r->think_mode)) {
        st.inside = true;
    }
    return st;
}

static bool should_remember_thinking_checkpoint(const request *r,
                                                const thinking_state *thinking,
                                                const char *finish) {
    if (!r || r->kind != REQ_CHAT || r->has_tools) return false;
    if (r->prompt_preserves_reasoning) return false;
    if (!ds4_think_mode_enabled(r->think_mode)) return false;
    if (finish && (!strcmp(finish, "error") || !strcmp(finish, "length"))) return false;
    if (thinking && thinking->inside) return false;
    return true;
}

static void log_tool_calls_summary(const char *ctx, const tool_calls *calls,
                                   bool responses_protocol) {
    if (!calls || calls->len == 0) return;
    buf names = {0};
    buf ids = {0};
    for (int i = 0; i < calls->len; i++) {
        if (i) buf_putc(&names, ',');
        if (i) buf_putc(&ids, ',');
        buf_puts(&names, calls->v[i].name ? calls->v[i].name : "?");
        buf_puts(&ids, calls->v[i].id ? calls->v[i].id : "?");
    }
    char flags[32];
    log_flags(flags, sizeof(flags), responses_protocol, false, false, false, false);
    server_log(DS4_LOG_TOOL,
               "ds4-server: tool calls ctx=%s%s%s n=%d raw_dsml=%d ids=[%s] names=[%s]",
               ctx,
               flags[0] ? " " : "",
               flags,
               calls->len,
               calls->raw_dsml && calls->raw_dsml[0] ? 1 : 0,
               ids.ptr ? ids.ptr : "",
               names.ptr ? names.ptr : "");
    buf_free(&ids);
    buf_free(&names);
}

static void server_progress_cb(void *ud, const char *event, int current, int total) {
    server_prefill_progress *p = ud;
    if (!p || !event || strcmp(event, "prefill_chunk")) return;

    double now = now_sec();
    double elapsed = now - p->t0;
    if (p->seen && current == p->last_current) {
        if (p->srv && current > p->cached_tokens) {
            kv_cache_maybe_store_continued(p->srv);
        }
        return;
    }
    int display_start = p->cached_tokens;
    if (display_start < 0 || display_start > p->prompt_tokens) display_start = 0;
    int display_total = p->prompt_tokens - display_start;
    if (display_total <= 0) {
        display_start = 0;
        display_total = p->prompt_tokens > total ? p->prompt_tokens : total;
    }
    int display_current = current - display_start;
    if (display_current < 0) display_current = 0;
    if (display_current > display_total) display_current = display_total;
    double pct = display_total > 0 ? 100.0 * (double)display_current / (double)display_total : 100.0;
    double avg_tps = elapsed > 0.0 ? (double)display_current / elapsed : 0.0;
    int interval_tokens = p->seen ? current - p->last_current : 0;
    if (interval_tokens < 0) interval_tokens = 0;
    double interval_s = p->seen ? now - p->last_t : 0.0;
    double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    p->last_current = current;
    p->last_t = now;
    p->seen = true;
    char flags[64];
    log_flags(flags, sizeof(flags), p->responses_protocol,
              p->has_tools, false, false, false);
    const char *phase = p->phase ? p->phase : "prefill";
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s %s chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs",
               p->kind == REQ_CHAT ? "chat" : "completion",
               p->ctx,
               flags[0] ? " " : "",
               flags,
               phase,
               display_current,
               display_total,
               pct,
               chunk_tps,
               avg_tps,
               elapsed);
    if (p->srv && current > p->cached_tokens) {
        kv_cache_maybe_store_continued(p->srv);
    }
}

static char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls) {
    buf suffix = {0};
    if (ds4_think_mode_enabled(r->think_mode)) {
        buf_puts(&suffix, reasoning ? reasoning : "");
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_dsml_tool_calls_text(&suffix, calls);
    buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    return buf_take(&suffix);
}

static char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls) {
    buf suffix = {0};
    /* This suffix mirrors what a Responses client can replay, not necessarily
     * every token in KV.  Hidden reasoning stays live in the session unless the
     * next client replay is expected to include it.  In practice, pi replays
     * reasoning summaries for tool-call turns, but not for final assistant
     * answers; Codex currently requests no summaries at all.  So only include
     * reasoning in the remembered visible prefix when this assistant turn ended
     * in tool calls.  A client that does replay final-answer reasoning will not
     * match this visible shortcut and can still use exact token-prefix replay. */
    if (ds4_think_mode_enabled(r->think_mode)) {
        if (r->reasoning_summary_emit && calls && calls->len > 0) {
            buf_puts(&suffix, reasoning ? reasoning : "");
        }
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_dsml_tool_calls_text(&suffix, calls);
    buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    return buf_take(&suffix);
}

/* In thinking mode without tools, old assistant reasoning is intentionally not
 * rendered back into later prompts.  The sampled live graph still contains the
 * reasoning bytes, so the next request would miss the session cache even though
 * the visible conversation prefix is logically the same.
 *
 *   prompt-without-final-<think> + </think> + visible-content + eos
 *
 * is exactly the visible prefix that render_chat_prompt_text() will produce on
 * the next turn.  Do not rebuild the KV cache to erase hidden reasoning here:
 * that caused long post-answer pauses and threw away useful sampled state.
 * Instead, remember the visible bytes as a key for the current sampled frontier.
 * The next request can then continue from live KV while tokenizing only the new
 * visible suffix. */
static char *build_toolless_thinking_visible_text(const request *r,
                                                  const char *content) {
    if (!r || !r->prompt_text) return NULL;
    if (!ds4_think_mode_enabled(r->think_mode)) return NULL;

    size_t pt_len = strlen(r->prompt_text);
    const char *think_tag = "<think>";
    size_t tag_len = strlen(think_tag);
    if (pt_len < tag_len ||
        memcmp(r->prompt_text + pt_len - tag_len, think_tag, tag_len) != 0) {
        return NULL;
    }

    buf visible = {0};
    buf_append(&visible, r->prompt_text, pt_len - tag_len);
    buf_puts(&visible, "</think>");
    buf_puts(&visible, content ? content : "");
    buf_puts(&visible, "<｜end▁of▁sentence｜>");
    return buf_take(&visible);
}

static void remember_thinking_checkpoint(server *s, const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content) {
    char *visible = build_toolless_thinking_visible_text(&j->req, content);
    if (!visible) return;

    thinking_live_remember(s, visible);
    server_log(DS4_LOG_KVCACHE,
               "ds4-server: thinking live checkpoint remembered ctx=%s live=%d visible=%zu",
               ctx, ds4_session_pos(s->session), strlen(visible));
    trace_event(s, trace_id,
                "thinking live checkpoint remembered: live=%d visible=%zu",
                ds4_session_pos(s->session), strlen(visible));
    free(visible);
}

/* After a successful tool-call finish, make the live checkpoint match what the
 * next request will render.  Usually that is just the exact DSML remembered by
 * tool id.  If a client sends a tool call without an id we know, the fallback
 * renderer still builds valid DSML from JSON, and this function either rewrites
 * the short suffix in place or reloads an older disk checkpoint before replay. */
static void canonicalize_tool_checkpoint(server *s, const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content,
                                         const char *reasoning, const tool_calls *calls) {
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;

    char *suffix_text = build_tool_checkpoint_suffix(&j->req, content, reasoning, calls);

    buf rendered = {0};
    buf_puts(&rendered, j->req.prompt_text);
    buf_puts(&rendered, suffix_text);

    ds4_tokens canonical = {0};
    ds4_tokenize_rendered_chat(s->engine, rendered.ptr ? rendered.ptr : "", &canonical);
    const int live_len = ds4_session_pos(s->session);
    const int common = ds4_session_common_prefix(s->session, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, ds4_session_tokens(s->session), &live_text_len);
    if (live_text_len == rendered.len &&
        (live_text_len == 0 || memcmp(live_text, rendered.ptr, live_text_len) == 0))
    {
        /* The graph already represents the bytes the next request will render.
         * Token-level canonicalization would only replace a valid sampled
         * history with a different BPE spelling of the same transcript. */
        free(live_text);
        goto done;
    }
    free(live_text);

    if (common < j->req.prompt.len) {
        trace_event(s, trace_id,
                    "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d",
                    common, j->req.prompt.len, live_len, canonical.len);
        goto done;
    }

    char err[160] = {0};
    ds4_session_rewrite_result rr =
        ds4_session_rewrite_from_common(s->session, &canonical, common,
                                        err, sizeof(err));
    if (rr == DS4_SESSION_REWRITE_OK) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d",
                   ctx, common, live_len, canonical.len);
        trace_event(s, trace_id,
                    "tool checkpoint canonicalized: common=%d live=%d canonical=%d",
                    common, live_len, canonical.len);
    } else if (rr == DS4_SESSION_REWRITE_REBUILD_NEEDED) {
        /* The generated DSML suffix and the canonical prompt share a prefix,
         * but the generated tail is too large to overwrite safely inside the
         * live raw-window ring.  Prefer an older disk checkpoint over replaying
         * a very long conversation from token zero. */
        char *path = NULL;
        ds4_tokens effective = {0};
        int loaded = kv_cache_try_load_text(s, rendered.ptr ? rendered.ptr : "",
                                            &effective, &path, NULL, false);
        if (loaded == 0) ds4_session_invalidate(s->session);

        char sync_err[160] = {0};
        const ds4_tokens *sync_prompt = loaded > 0 ? &effective : &canonical;
        char rebuild_ctx[48];
        request_ctx_span(rebuild_ctx, sizeof(rebuild_ctx), loaded, sync_prompt->len);
        int replay_tokens = sync_prompt->len - loaded;
        if (replay_tokens < 0) replay_tokens = sync_prompt->len;
        int canonical_tail_tokens = canonical.len - common;
        if (canonical_tail_tokens < 0) canonical_tail_tokens = canonical.len;
        int discarded_live_tokens = live_len - common;
        if (discarded_live_tokens < 0) discarded_live_tokens = 0;
        const char *source = loaded > 0 ? "disk" : "full";
        const double rebuild_t0 = now_sec();
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalization needs %d tokens rebuild ctx=%s request_ctx=%s reason=canonical-tail-rewrite tail=%d discard=%d common=%d live=%d target=%d cached=%d source=%s%s%s",
                   replay_tokens,
                   rebuild_ctx,
                   ctx,
                   canonical_tail_tokens,
                   discarded_live_tokens,
                   common,
                   live_len,
                   canonical.len,
                   loaded,
                   source,
                   path ? " file=" : "",
                   path ? path : "");
        server_prefill_progress rebuild_progress = {
            .srv = s,
            .kind = j->req.kind,
            .prompt_tokens = sync_prompt->len,
            .cached_tokens = loaded,
            .phase = "tool checkpoint rebuild",
            .has_tools = j->req.has_tools,
            .t0 = rebuild_t0,
        };
        snprintf(rebuild_progress.ctx, sizeof(rebuild_progress.ctx), "%s", rebuild_ctx);
        ds4_session_set_progress(s->session, server_progress_cb, &rebuild_progress);
        if (ds4_session_sync(s->session, sync_prompt, sync_err, sizeof(sync_err)) == 0) {
            ds4_session_set_progress(s->session, NULL, NULL);
            const double rebuild_sec = now_sec() - rebuild_t0;
            if (loaded > 0) {
                server_log(DS4_LOG_KVCACHE,
                           "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=disk cached=%d replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, loaded, replay_tokens, canonical.len, rebuild_sec);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s",
                            common, live_len, canonical.len, loaded, path ? path : "");
            } else {
                server_log(DS4_LOG_KVCACHE,
                           "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=full cached=0 replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, replay_tokens, canonical.len, rebuild_sec);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s",
                            common, live_len, canonical.len, err);
            }
        } else {
            ds4_session_set_progress(s->session, NULL, NULL);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: tool checkpoint rebuild failed ctx=%s request_ctx=%s source=%s cached=%d replay=%d target=%d error=\"%s\"",
                       rebuild_ctx, ctx, source, loaded, replay_tokens,
                       canonical.len, sync_err);
            trace_event(s, trace_id, "tool checkpoint canonicalization failed after rebuild request: %s", sync_err);
        }
        ds4_tokens_free(&effective);
        free(path);
    } else {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        trace_event(s, trace_id, "tool checkpoint canonicalization failed: %s", err);
    }

done:
    ds4_tokens_free(&canonical);
    buf_free(&rendered);
    free(suffix_text);
}

static bool should_canonicalize_tool_checkpoint(const server *s, const tool_calls *calls) {
    if (!calls || calls->len == 0) return false;
    if (s && !s->disable_exact_dsml_tool_replay &&
        calls->raw_dsml && calls->raw_dsml[0])
    {
        return false;
    }
    return true;
}

/* Execute one request on the worker-owned session.
 *
 * Clients resend full prompts as text.  The worker first tries the old exact
 * token-prefix hit, then a rendered-text prefix hit for the live checkpoint,
 * then disk text-prefix restart snapshots, then a cold prefill.  On text-prefix
 * hits we build a fresh effective prompt from the checkpoint's exact token
 * history plus a newly tokenized string suffix; the canonical full-prompt
 * tokens are not sliced because BPE may merge across the byte boundary.  Cold
 * prompt caching is handled before generation: if the stable checkpoint is
 * shorter than the full prompt, we prefill to that boundary, store it, and
 * immediately continue to the real prompt.  The live graph therefore always
 * moves forward. */
static void generate_job(server *s, job *j) {
    char err[160];
    err[0] = '\0';
    const int old_pos = ds4_session_pos(s->session);
    const int common = ds4_session_common_prefix(s->session, &j->req.prompt);
    trace_cache_diag cache_diag = {0};
    trace_cache_capture(&cache_diag, ds4_session_tokens(s->session),
                        &j->req.prompt, old_pos, common);
    ds4_tokens effective_prompt = {0};
    const ds4_tokens *prompt_for_sync = &j->req.prompt;
    const bool responses_protocol = j->req.api == API_RESPONSES;
    bool responses_live_continuation = false;
    bool anthropic_live_continuation = false;
    bool thinking_live_continuation = false;
    const char *responses_live_match = NULL;
    int responses_live_match_ids = 0;
    int anthropic_live_match_ids = 0;
    /* Responses gets the first chance to continue from live state.  This is
     * the whole point of the API shape: a request that is bound to prior live
     * output by visible transcript or tool call ids does not need to prove an
     * exact token-prefix match.  Exact token/text/disk matching remains the
     * fallback when the live state is absent or no longer describes the
     * request. */
    int cached = responses_live_visible_prefix_prompt(s, &j->req, old_pos,
                                                      &effective_prompt);
    const char *cache_source = cached > 0 ? "responses-visible" : "none";
    if (cached > 0) {
        responses_live_match = "visible-prefix";
        if (responses_live_matches_request(s, &j->req.responses_live_call_ids,
                                           old_pos))
        {
            responses_live_match_ids = j->req.responses_live_call_ids.len;
        }
    }
    if (cached == 0) {
        cached = responses_live_continuation_prompt(s, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &responses_live_match_ids);
        cache_source = cached > 0 ? "responses-tool-output" : "none";
        if (cached > 0) responses_live_match = "tool-output-ids";
    }
    if (cached > 0) {
        responses_live_continuation = true;
        prompt_for_sync = &effective_prompt;
    } else {
        cached = anthropic_live_continuation_prompt(s, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &anthropic_live_match_ids);
        if (cached > 0) {
            anthropic_live_continuation = true;
            cache_source = "anthropic-tool-output";
            prompt_for_sync = &effective_prompt;
        }
    }
    if (cached == 0 && responses_protocol &&
        j->req.responses_requires_live_tool_state)
    {
        /* The parser saw a valid live call_id, but by worker execution time the
         * live frontier no longer matches.  Since the request did not replay
         * the prior assistant call, there is no stateless prefix to match and
         * no disk key to search by. */
        ds4_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Responses continuation state is not available; retry by replaying the full input history");
        return;
    } else if (cached == 0 && j->req.api == API_ANTHROPIC &&
               j->req.anthropic_requires_live_tool_state)
    {
        ds4_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Anthropic continuation state is not available; retry by replaying the full messages history");
        return;
    } else if (cached == 0) {
        cached = common == old_pos && j->req.prompt.len >= old_pos ? common : 0;
        cache_source = cached > 0 ? "memory-token" : "none";
    }
    if (cached == 0) {
        int thinking_cached =
            thinking_live_visible_prefix_prompt(s, &j->req, old_pos,
                                                &effective_prompt);
        if (thinking_cached > 0) {
            cached = thinking_cached;
            cache_source = "thinking-visible";
            thinking_live_continuation = true;
            prompt_for_sync = &effective_prompt;
        }
    }
    int disk_cached = 0;
    char *disk_cache_path = NULL;
    uint8_t disk_cache_ext_flags = 0;
    if (cached == 0) {
        int text_cached = live_text_prefix_prompt(s, &j->req, &effective_prompt);
        if (text_cached > 0) {
            cached = text_cached;
            cache_source = "memory-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    if (cached == 0 && old_pos > 0) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: live kv cache miss%s live=%d prompt=%d common=%d reason=%s",
                   responses_protocol ? " RESPPROTO" : "",
                   old_pos, j->req.prompt.len, common,
                   trace_cache_miss_reason(&cache_diag));
    }
    if (cached == 0) s->kv.continued_last_store_tokens = 0;
    if (s->kv.enabled && cached == 0 && old_pos >= s->kv.opt.min_tokens) {
        /* Loading a disk snapshot replaces the live Metal session.  Persist the
         * current checkpoint first, otherwise a cache hit for an older prefix
         * would silently discard the newer conversation state. */
        kv_cache_store_current(s, "evict");
    }
    if (cached == 0) {
        disk_cached = kv_cache_try_load(s, &j->req, &effective_prompt,
                                        &disk_cache_path,
                                        &disk_cache_ext_flags);
        if (disk_cached > 0) {
            cached = disk_cached;
            cache_source = "disk-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    const bool responses_reasoning_state_preserved =
        cached > 0 &&
        ((!strcmp(cache_source, "responses-visible") ||
          !strcmp(cache_source, "responses-tool-output")) ||
         (!strcmp(cache_source, "disk-text") &&
          (disk_cache_ext_flags & KV_EXT_RESPONSES_VISIBLE)));
    const bool responses_visible_replay_without_reasoning =
        responses_protocol &&
        j->req.responses_requires_live_reasoning &&
        !responses_reasoning_state_preserved;
    const int prompt_tokens = prompt_for_sync->len;
    /* OpenAI usage details: the reusable prefix is a cache read, while the
     * effective prompt suffix evaluated by ds4_session_sync() is written into
     * the live KV cache and can be reused by the next request. */
    j->req.cache_read_tokens = cached;
    j->req.cache_write_tokens = prompt_tokens > cached ? prompt_tokens - cached : 0;

    const double t0 = now_sec();
    uint64_t trace_id = trace_begin(s, j, cached, prompt_tokens, &cache_diag,
                                    cache_source, disk_cached, disk_cache_path);
    free(disk_cache_path);
    char ctx_span[48];
    request_ctx_span(ctx_span, sizeof(ctx_span), cached, prompt_tokens);
    server_prefill_progress progress = {
        .srv = s,
        .kind = j->req.kind,
        .prompt_tokens = prompt_tokens,
        .cached_tokens = cached,
        .has_tools = j->req.has_tools,
        .responses_protocol = responses_protocol,
        .t0 = t0,
    };
    snprintf(progress.ctx, sizeof(progress.ctx), "%s", ctx_span);
    char req_flags[64];
    log_flags(req_flags, sizeof(req_flags), responses_protocol,
              j->req.has_tools, false, false, false);
    if (responses_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: responses live continuation RESPPROTO match=%s ids=%d cached=%d prompt=%d",
                   responses_live_match ? responses_live_match : "unknown",
                   responses_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (anthropic_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: anthropic live continuation match=tool-output-ids ids=%d cached=%d prompt=%d",
                   anthropic_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (thinking_live_continuation) {
        server_log(DS4_LOG_PREFILL,
                   "ds4-server: thinking live continuation match=visible-prefix cached=%d prompt=%d",
                   cached,
                   prompt_tokens);
    }
    if (responses_visible_replay_without_reasoning) {
        /* The request replays a prior tool-call turn but omits the hidden
         * reasoning that originally led to it.  A live Responses checkpoint, or
         * a responses-visible disk checkpoint, would preserve that hidden KV.
         * If neither is available, continue from the visible transcript instead
         * of surfacing a hard error to the user.  This is lower fidelity, but it
         * lets old / restarted agent sessions recover and is exactly what the
         * client asked us to prefill. */
        server_log(DS4_LOG_WARNING,
                   "ds4-server: responses replay RESPPROTO missing reasoning state; continuing from visible history source=%s cached=%d prompt=%d",
                   cache_source,
                   cached,
                   prompt_tokens);
        trace_event(s, trace_id,
                    "responses replay missing reasoning state; continuing from visible history source=%s cached=%d",
                    cache_source, cached);
    }
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s prompt start",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               ctx_span,
               req_flags[0] ? " " : "",
               req_flags);
    ds4_session_set_progress(s->session, server_progress_cb, &progress);

    int cold_store_len = 0;
    if (cached == 0 &&
        s->kv.enabled &&
        prompt_for_sync->len >= s->kv.opt.min_tokens &&
        s->kv.opt.cold_max_tokens > 0 &&
        prompt_for_sync->len <= s->kv.opt.cold_max_tokens)
    {
        const int anchor = kv_cache_chat_anchor_pos(&s->kv, prompt_for_sync,
                                                    ds4_token_user(s->engine),
                                                    ds4_token_assistant(s->engine));
        cold_store_len = anchor >= s->kv.opt.min_tokens ?
                         anchor : kv_cache_store_len(&s->kv, prompt_for_sync->len);
    }
    int suppressed_continued_last = -1;
    if (cold_store_len >= s->kv.opt.min_tokens) {
        /* A cold checkpoint can land exactly on the continued-checkpoint
         * frontier.  The prefill progress callback would then write the same
         * prefix as "continued" while we are intentionally stopping there to
         * write it as "cold".  Mark the frontier as already handled before the
         * sync reaches it; if the cold write fails, restore the old schedule so
         * a later continued write can still try. */
        suppressed_continued_last =
            kv_cache_suppress_continued_store(&s->kv, cold_store_len);
    }

    if (s->kv.enabled &&
        cold_store_len >= s->kv.opt.min_tokens &&
        cold_store_len < prompt_for_sync->len)
    {
        ds4_tokens prefix = {0};
        tokens_copy_prefix(&prefix, prompt_for_sync, cold_store_len);
        if (ds4_session_sync(s->session, &prefix, err, sizeof(err)) != 0) {
            ds4_tokens_free(&prefix);
            ds4_tokens_free(&effective_prompt);
            ds4_session_set_progress(s->session, NULL, NULL);
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
            trace_event(s, trace_id, "prefill failed: %s", err);
            http_error(j->fd, s->enable_cors, 500, err);
            return;
        }
        if (kv_cache_store_live_prefix(s, prompt_for_sync, cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, cold_store_len);
            suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
            suppressed_continued_last = -1;
        }
        ds4_tokens_free(&prefix);
    }

    if (ds4_session_sync(s->session, prompt_for_sync, err, sizeof(err)) != 0) {
        ds4_tokens_free(&effective_prompt);
        ds4_session_set_progress(s->session, NULL, NULL);
        kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                              cold_store_len);
        trace_event(s, trace_id, "prefill failed: %s", err);
        http_error(j->fd, s->enable_cors, 500, err);
        return;
    }
    /* Once a non-live request wins, old protocol live bindings are stale. Keep
     * a binding only when this request explicitly continued from it. */
    if (!responses_live_continuation) responses_live_clear(s);
    if (!anthropic_live_continuation) anthropic_live_clear(s);
    if (!thinking_live_continuation) thinking_live_clear(s);
    ds4_session_set_progress(s->session, NULL, NULL);
    kv_cache_maybe_store_continued(s);
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s prompt done %.3fs",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               ctx_span,
               req_flags[0] ? " " : "",
               req_flags,
               now_sec() - t0);
    if (cold_store_len == prompt_for_sync->len) {
        if (kv_cache_store_live_prefix(s, prompt_for_sync, cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, cold_store_len);
            suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, suppressed_continued_last,
                                                  cold_store_len);
        }
    }
    char id[96];
    snprintf(id, sizeof(id), "%s-%llu",
             j->req.kind == REQ_CHAT ? "chatcmpl" : "cmpl",
             (unsigned long long)++s->seq);

    bool structured_stream = request_uses_structured_stream(&j->req);
    anthropic_stream anthropic_live = {0};
    openai_stream openai_live = {0};
    responses_stream responses_live = {0};
    const bool openai_live_chat = request_uses_openai_live_stream(&j->req);
    const bool responses_live_chat = request_uses_responses_live_stream(&j->req);
    long responses_created_at = (long)time(NULL);
    if (j->req.stream) {
        if (!sse_headers(j->fd, s->enable_cors)) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s sse headers failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        if (j->req.api == API_ANTHROPIC &&
            !anthropic_sse_start_live(j->fd, &j->req, id,
                                      prompt_tokens, &anthropic_live)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s anthropic stream start failed", ctx_span);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        if (j->req.api == API_OPENAI && j->req.kind == REQ_CHAT &&
            !sse_chunk(j->fd, &j->req, id, NULL, NULL)) {
            server_log(DS4_LOG_GENERATION, "ds4-server: chat ctx=%s openai role chunk failed", ctx_span);
            ds4_tokens_free(&effective_prompt);
            return;
        }
        if (openai_live_chat) openai_stream_start(&j->req, &openai_live);
        if (responses_live_chat) {
            responses_stream_init(&j->req, &responses_live);
            responses_live.active = true;
            if (!responses_sse_created(j->fd, &j->req, &responses_live, responses_created_at)) {
                server_log(DS4_LOG_GENERATION,
                           "ds4-server: chat ctx=%s%s%s responses created event failed",
                           ctx_span,
                           req_flags[0] ? " " : "",
                           req_flags);
                responses_stream_free(&responses_live);
                ds4_tokens_free(&effective_prompt);
                return;
            }
        }
    }

    buf text = {0};
    size_t plain_stream_pos = 0;
    size_t stop_scan_from = 0;
    const char *finish = "length";
    int completion = 0;
    int max_tokens = j->req.max_tokens;
    int room = ds4_session_ctx(s->session) - ds4_session_pos(s->session);
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    bool saw_orphan_tool_end = false;
    size_t tool_scan_from = 0;
    int next_tool_progress = 128;
    int next_decode_log = 50;
    uint64_t rng = j->req.seed ? j->req.seed :
        (((uint64_t)time(NULL) << 32) ^ ((uint64_t)s->seq << 1) ^ (uint64_t)(uintptr_t)j);
    if (max_tokens < 0) max_tokens = 0;
    if (max_tokens > room) max_tokens = room;
    trace_event(s, trace_id, "prefill done; decode_max=%d ctx_room=%d", max_tokens, room);
    const double decode_t0 = now_sec();
    double last_decode_log_t = decode_t0;
    int last_decode_log_completion = 0;
    thinking_state thinking = thinking_state_from_prompt(&j->req);
    const bool thinking_gates_tool_markers = ds4_think_mode_enabled(j->req.think_mode);
    bool tool_scan_waiting_for_think_close =
        thinking_gates_tool_markers && thinking.inside;
    dsml_decode_tracker dsml_tracker;
    dsml_decode_tracker_init(&dsml_tracker);

    while (!g_stop_requested && completion < max_tokens &&
           ds4_session_pos(s->session) < ds4_session_ctx(s->session)) {
        dsml_decode_state dsml_state = j->req.kind == REQ_CHAT && j->req.has_tools ?
            dsml_tracker.decode : DSML_DECODE_OUTSIDE;
        const bool in_tool_call = dsml_decode_state_is_tool(dsml_state);
        if (!(j->req.kind == REQ_CHAT && j->req.has_tools && (saw_tool_start || in_tool_call))) {
            kv_cache_maybe_store_continued(s);
        }
        float temperature = j->req.temperature;
        int top_k = j->req.top_k;
        float top_p = j->req.top_p;
        float min_p = j->req.min_p;
        if (ds4_think_mode_enabled(j->req.think_mode)) {
            temperature = DS4_DEFAULT_TEMPERATURE;
            top_k = 0;
            top_p = DS4_DEFAULT_TOP_P;
            min_p = DS4_DEFAULT_MIN_P;
        }
        if (in_tool_call && !dsml_decode_state_uses_payload_sampling(dsml_state)) {
            temperature = 0.0f;
        }
        int token = ds4_session_sample(s->session, temperature, top_k, top_p, min_p, &rng);
        if (token == ds4_token_eos(s->engine)) {
            finish = "stop";
            break;
        }

        int toks[17];
        int ntok = 0;
        if (temperature <= 0.0f &&
            ds4_engine_mtp_draft_tokens(s->engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL)
        {
            ntok = ds4_session_eval_speculative_argmax(s->session,
                                                       token,
                                                       max_tokens - completion,
                                                       ds4_token_eos(s->engine),
                                                       toks,
                                                       (int)(sizeof(toks) / sizeof(toks[0])),
                                                       err,
                                                       sizeof(err));
            if (ntok < 0) {
                finish = "error";
                break;
            }
        } else {
            if (ds4_session_eval(s->session, token, err, sizeof(err)) != 0) {
                finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        }

        bool stop_decode = false;
        for (int ti = 0; ti < ntok && completion < max_tokens; ti++) {
            token = toks[ti];
            if (token == ds4_token_eos(s->engine)) {
                finish = "stop";
                stop_decode = true;
                break;
            }

            size_t piece_len = 0;
            char *piece = ds4_token_text(s->engine, token, &piece_len);
            completion++;

            trace_piece(s, trace_id, piece, piece_len);
            buf_append(&text, piece, piece_len);
            thinking_state_feed(&thinking, piece, piece_len);
            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                dsml_decode_tracker_update(&dsml_tracker, text.ptr, text.len);
            }

            size_t stop_pos = 0, stop_len = 0;
            bool hit_stop = stop_list_find_from(&j->req.stops, text.ptr,
                                                stop_scan_from,
                                                &stop_pos, &stop_len);
            size_t stream_len = hit_stop ?
                stop_pos : stop_list_stream_safe_len(&j->req.stops, text.len);
            if (stream_len > text.len) stream_len = text.len;
            stream_len = utf8_stream_safe_len(text.ptr, plain_stream_pos,
                                              stream_len, hit_stop);
            if (!hit_stop && j->req.stops.max_len > 1) {
                const size_t hold = j->req.stops.max_len - 1;
                stop_scan_from = text.len > hold ? text.len - hold : 0;
            }

            if (j->req.stream && !structured_stream && stream_len > plain_stream_pos) {
                char *delta = xstrndup(text.ptr + plain_stream_pos, stream_len - plain_stream_pos);
                bool ok = sse_chunk(j->fd, &j->req, id, delta, NULL);
                free(delta);
                if (!ok) {
                    finish = "error";
                    snprintf(err, sizeof(err), "client stream write failed");
                    free(piece);
                    stop_decode = true;
                    break;
                }
                plain_stream_pos = stream_len;
            }
            if (j->req.stream && j->req.api == API_ANTHROPIC &&
                !anthropic_sse_stream_update(j->fd, s, &j->req, id,
                                             &anthropic_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (openai_live_chat &&
                !openai_sse_stream_update(j->fd, s, &j->req, id,
                                          &openai_live, text.ptr, stream_len,
                                          false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            if (responses_live_chat &&
                !responses_sse_stream_update(j->fd, &j->req,
                                             &responses_live, text.ptr, stream_len,
                                             false)) {
                finish = "error";
                snprintf(err, sizeof(err), "client stream write failed");
                free(piece);
                stop_decode = true;
                break;
            }
            free(piece);

            if (j->req.kind == REQ_CHAT && j->req.has_tools) {
                if (thinking_gates_tool_markers && thinking.inside) {
                    /* A DSML block inside reasoning is not executable.  This is
                     * the live guard: do not let a quoted or mistaken marker in
                     * <think> stop decoding as a real tool call. */
                    tool_scan_waiting_for_think_close = true;
                    tool_scan_from = text.len;
                } else {
                    if (tool_scan_waiting_for_think_close) {
                        const char *think_end = find_last_substr(text.ptr, "</think>");
                        tool_scan_from = think_end ? (size_t)((think_end + 8) - text.ptr) : text.len;
                        if (tool_scan_from > text.len) tool_scan_from = text.len;
                        tool_scan_waiting_for_think_close = false;
                    }
                    if (tool_scan_from > text.len) tool_scan_from = text.len;
                    const char *tool_scan = text.ptr ? text.ptr + tool_scan_from : "";
                    bool orphan_end = false;
                    bool old_start = saw_tool_start;
                    bool old_end = saw_tool_end;
                    observe_tool_markers(tool_scan, &saw_tool_start, &saw_tool_end, &orphan_end);
                    if (orphan_end && !saw_orphan_tool_end) {
                        saw_orphan_tool_end = true;
                        server_log(DS4_LOG_WARNING,
                                   "ds4-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens",
                                   ctx_span,
                                   req_flags[0] ? " " : "",
                                   req_flags,
                                   completion);
                        trace_event(s, trace_id,
                                    "ignored orphan tool-call end marker after %d generated tokens",
                                    completion);
                    }
                    if (saw_tool_start && !old_start) {
                        trace_event(s, trace_id, "entered tool-call block after %d generated tokens", completion);
                    }
                    if (saw_tool_end && !old_end) {
                        trace_event(s, trace_id, "closed tool-call block after %d generated tokens", completion);
                    }
                    const size_t marker_hold = 80;
                    size_t hold_from = text.len > marker_hold ? text.len - marker_hold : 0;
                    if (hold_from > tool_scan_from) tool_scan_from = hold_from;
                    if (s->trace && completion >= next_tool_progress) {
                        trace_event(s, trace_id,
                                    "progress gen=%d dsml_start=%d dsml_end=%d",
                                    completion, saw_tool_start ? 1 : 0, saw_tool_end ? 1 : 0);
                        next_tool_progress += 128;
                    }
                }
            }

            if (completion >= next_decode_log) {
                log_decode_progress(j->req.kind, prompt_tokens, completion,
                                    responses_protocol,
                                    j->req.has_tools,
                                    thinking.inside,
                                    saw_tool_start,
                                    saw_tool_end,
                                    decode_t0,
                                    &last_decode_log_t,
                                    &last_decode_log_completion);
                next_decode_log += 50;
            }

            if (hit_stop) {
                (void)stop_len;
                finish = "stop";
                text.len = stop_pos;
                text.ptr[text.len] = '\0';
                ds4_session_invalidate(s->session);
                stop_decode = true;
                break;
            }

            if (j->req.kind == REQ_CHAT && j->req.has_tools && saw_tool_end) {
                finish = "tool_calls";
                stop_decode = true;
                break;
            }
        }
        if (stop_decode) break;
    }

    if (g_stop_requested && strcmp(finish, "error") != 0) {
        finish = "error";
        snprintf(err, sizeof(err), "shutdown requested");
    }

    if (j->req.kind == REQ_CHAT && j->req.has_tools &&
        saw_tool_start && !saw_tool_end && strcmp(finish, "error") != 0)
    {
        /* A partial streamed tool call cannot be retracted.  If the model ends
         * before closing the DSML block, fail the turn instead of letting clients
         * execute an incomplete `{}` or partially parsed argument object. */
        finish = "error";
        snprintf(err, sizeof(err), "unterminated tool call");
    }

    if (completion > last_decode_log_completion) {
        log_decode_progress(j->req.kind, prompt_tokens, completion,
                            responses_protocol,
                            j->req.has_tools,
                            thinking.inside,
                            saw_tool_start,
                            saw_tool_end,
                            decode_t0,
                            &last_decode_log_t,
                            &last_decode_log_completion);
    }

    if (j->req.stream && !structured_stream && text.len > plain_stream_pos) {
        char *tail = xstrndup(text.ptr + plain_stream_pos, text.len - plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, id, tail, NULL)) finish = "error";
        free(tail);
    }

    tool_calls parsed_calls = {0};
    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    const char *final_finish = finish;
    bool recovered_tool_parse_failure = false;
    if (j->req.kind == REQ_CHAT) {
        bool parsed_ok = parse_generated_message_for_response(
            text.ptr ? text.ptr : "",
            j->req.has_tools,
            saw_tool_start,
            ds4_think_mode_enabled(j->req.think_mode),
            &final_finish,
            err,
            sizeof(err),
            &parsed_content,
            &parsed_reasoning,
            &parsed_calls,
            &recovered_tool_parse_failure);
        if (!parsed_ok && recovered_tool_parse_failure) {
            server_log(DS4_LOG_WARNING,
                       "ds4-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags,
                       final_finish);
            trace_event(s, trace_id,
                        "invalid tool call returned as assistant text finish=%s",
                        final_finish);
        }
        if (parsed_calls.len) {
            if (openai_live_chat) apply_openai_stream_tool_ids(&parsed_calls, &openai_live);
            if (j->req.api == API_ANTHROPIC && j->req.stream)
                apply_anthropic_stream_tool_ids(&parsed_calls, &anthropic_live);
            assign_tool_call_ids(s, &parsed_calls, j->req.api);
            tool_memory_remember(s, &parsed_calls);
            final_finish = "tool_calls";
        } else if (j->req.api == API_RESPONSES) {
            responses_live_clear(s);
        }
    }
    log_tool_calls_summary(ctx_span, &parsed_calls,
                           responses_protocol);

    trace_finish(s, trace_id, &j->req, final_finish, completion,
                 saw_tool_start, saw_tool_end,
                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                 parsed_reasoning, &parsed_calls, now_sec() - t0);

    if (j->req.api == API_RESPONSES) {
        if (strcmp(final_finish, "error") && strcmp(final_finish, "length")) {
            /* Store the post-turn visible transcript plus the live token
             * frontier.  The next Responses request may replay only this
             * visible surface, while the real session also contains hidden
             * reasoning and exact sampled tool-call bytes. */
            char *visible_suffix =
                build_responses_visible_assistant_suffix(&j->req,
                    parsed_content ? parsed_content : "",
                    parsed_reasoning,
                    &parsed_calls);
            buf visible = {0};
            buf_puts(&visible, j->req.prompt_text ? j->req.prompt_text : "");
            buf_puts(&visible, visible_suffix ? visible_suffix : "");
            responses_live_remember(s, visible.ptr ? visible.ptr : "",
                                    parsed_calls.len ? &parsed_calls : NULL);
            buf_free(&visible);
            free(visible_suffix);
        } else {
            responses_live_clear(s);
        }
    }
    if (j->req.api == API_ANTHROPIC) {
        if (parsed_calls.len && strcmp(final_finish, "error") &&
            strcmp(final_finish, "length"))
        {
            anthropic_live_remember(s, &parsed_calls);
        } else {
            anthropic_live_clear(s);
        }
    }

    if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        should_canonicalize_tool_checkpoint(s, &parsed_calls))
    {
        /* Chat/completions has no protocol object that binds the next request
         * to this live KV state.  Canonicalize only the fallback tool-call
         * path where we lack exact sampled DSML replay; when raw DSML is known,
         * replaying those bytes keeps future prompts aligned without rebuilding
         * hidden reasoning.  Responses deliberately skips this path because its
         * previous_response_id contract binds the next turn to live state. */
        canonicalize_tool_checkpoint(s, j, ctx_span, trace_id,
                                     parsed_content ? parsed_content : "",
                                     parsed_reasoning, &parsed_calls);
        thinking_live_clear(s);
    } else if (parsed_calls.len) {
        thinking_live_clear(s);
    } else if (!parsed_calls.len &&
               should_remember_thinking_checkpoint(&j->req, &thinking, final_finish)) {
        remember_thinking_checkpoint(s, j, ctx_span, trace_id,
                                     parsed_content ? parsed_content : "");
    } else if (!parsed_calls.len) {
        thinking_live_clear(s);
    }

    if (j->req.stream) {
        bool response_ok = true;
        if (j->req.api == API_ANTHROPIC) {
            response_ok = anthropic_sse_finish_live(j->fd, s, &j->req, id, &anthropic_live,
                                                    text.ptr ? text.ptr : "", text.len,
                                                    &parsed_calls, final_finish, completion);
        } else if (openai_live_chat) {
            response_ok = openai_sse_finish_live(j->fd, s, &j->req, id, &openai_live,
                                                 text.ptr ? text.ptr : "", text.len,
                                                 &parsed_calls, final_finish,
                                                 prompt_tokens, completion);
        } else if (responses_live_chat) {
            /* If parse recovered a malformed tool call back to plain text,
             * pass parsed_content so the streaming tail can be flushed; in
             * the normal path parsed_content is the assistant text we already
             * streamed and the diff is empty. */
            const char *recover =
                recovered_tool_parse_failure ? parsed_content : NULL;
            response_ok = responses_sse_finish_live(j->fd, &j->req, &responses_live,
                                                    text.ptr ? text.ptr : "", text.len,
                                                    recover,
                                                    &parsed_calls, final_finish,
                                                    prompt_tokens, completion,
                                                    responses_created_at);
        } else if (structured_stream) {
            response_ok = sse_chat_finish(j->fd, &j->req, id,
                                          parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                          parsed_reasoning,
                                          &parsed_calls, final_finish,
                                          prompt_tokens, completion);
        } else {
            response_ok = sse_chunk(j->fd, &j->req, id, NULL, final_finish) &&
                          sse_done(j->fd, &j->req, id, prompt_tokens, completion);
        }
        if (!response_ok) {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: %s ctx=%s%s%s final stream failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       req_flags[0] ? " " : "",
                       req_flags);
        }
    } else if (j->req.api == API_ANTHROPIC) {
        anthropic_final_response(j->fd, s->enable_cors, &j->req, id,
                                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 prompt_tokens, completion);
    } else if (j->req.api == API_RESPONSES) {
        responses_final_response(j->fd, s->enable_cors, &j->req, id,
                                 parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 prompt_tokens, completion);
    } else {
        final_response(j->fd, s->enable_cors, &j->req, id,
                       parsed_content ? parsed_content : (text.ptr ? text.ptr : ""),
                       parsed_reasoning,
                       &parsed_calls, final_finish,
                       prompt_tokens, completion);
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  responses_protocol,
                  true,
                  thinking.inside,
                  saw_tool_start,
                  saw_tool_end);
        if (!strcmp(final_finish, "error") && err[0]) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: chat ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       err,
                       now_sec() - t0);
        } else {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: chat ctx=%s gen=%d%s%s finish=%s %.3fs",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       now_sec() - t0);
        }
    } else {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  responses_protocol,
                  j->req.has_tools,
                  thinking.inside,
                  false,
                  false);
        if (!strcmp(final_finish, "error") && err[0]) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       err,
                       now_sec() - t0);
        } else {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       ctx_span,
                       completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       now_sec() - t0);
        }
    }
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&parsed_calls);
    anthropic_stream_free(&anthropic_live);
    openai_stream_free(&openai_live);
    responses_stream_free(&responses_live);
    buf_free(&text);
    ds4_tokens_free(&effective_prompt);
}

static bool enqueue(server *s, job *j) {
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return true;
}

static job *dequeue(server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->head && !s->stopping) pthread_cond_wait(&s->cv, &s->mu);
    if (!s->head) {
        pthread_mutex_unlock(&s->mu);
        return NULL;
    }
    job *j = s->head;
    s->head = j->next;
    if (!s->head) s->tail = NULL;
    pthread_mutex_unlock(&s->mu);
    j->next = NULL;
    return j;
}

static void *worker_main(void *arg) {
    server *s = arg;
    for (;;) {
        job *j = dequeue(s);
        if (!j) break;
        generate_job(s, j);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
    return NULL;
}

typedef struct {
    char method[8];
    char path[256];
    char *body;
    size_t body_len;
} http_request;

static void http_request_free(http_request *r) {
    free(r->body);
    memset(r, 0, sizeof(*r));
}

static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            return strtol(v, NULL, 10);
        }
        if (p < end) p++;
    }
    return 0;
}

static bool read_http_request(int fd, http_request *r) {
    buf b = {0};
    ssize_t hend = -1;
    const size_t max_header = 64 * 1024;
    const size_t max_body = 64 * 1024 * 1024;

    while (hend < 0 && b.len < max_header) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
        hend = header_end(b.ptr, b.len);
    }
    if (hend < 0) goto fail;

    char line[512];
    size_t i = 0;
    while (i < b.len && b.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", r->method, r->path) != 2) goto fail;
    char *q = strchr(r->path, '?');
    if (q) *q = '\0';

    long clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0 || (size_t)clen > max_body) goto fail;
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    buf_free(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

typedef struct {
    server *srv;
    int fd;
} client_arg;

static void append_model_json_values(buf *b, int ctx, int default_tokens) {
    const int max_completion = default_tokens < ctx ? default_tokens : ctx;
    buf_printf(b,
        "{\"id\":\"deepseek-v4-flash\","
        "\"object\":\"model\","
        "\"created\":1767225600,"
        "\"owned_by\":\"ds4.c\","
        "\"name\":\"DeepSeek V4 Flash\","
        "\"context_length\":%d,"
        "\"top_provider\":{"
            "\"context_length\":%d,"
            "\"max_completion_tokens\":%d,"
            "\"is_moderated\":false},"
        "\"supported_parameters\":["
            "\"tools\","
            "\"tool_choice\","
            "\"max_tokens\","
            "\"temperature\","
            "\"top_p\","
            "\"top_k\","
            "\"min_p\","
            "\"stop\","
            "\"seed\","
            "\"stream\","
            "\"reasoning_effort\"]}",
        ctx,
        ctx,
        max_completion);
}

static void append_model_json(buf *b, const server *s) {
    append_model_json_values(b, ds4_session_ctx(s->session), s->default_tokens);
}

static bool send_model(server *s, int fd) {
    buf b = {0};
    append_model_json(&b, s);
    buf_putc(&b, '\n');
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool send_models(server *s, int fd) {
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    append_model_json(&b, s);
    buf_puts(&b, "]}\n");
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static void client_done(server *s) {
    pthread_mutex_lock(&s->mu);
    if (s->clients > 0) s->clients--;
    pthread_cond_broadcast(&s->clients_cv);
    pthread_mutex_unlock(&s->mu);
}

static void set_client_socket_nonblocking(int fd);

static void *client_main(void *arg) {
    client_arg *ca = arg;
    server *s = ca->srv;
    int fd = ca->fd;
    free(ca);

    http_request hr = {0};
    if (!read_http_request(fd, &hr)) {
        http_error(fd, s->enable_cors, 400, "bad HTTP request");
        goto done;
    }

    if (!strcmp(hr.method, "OPTIONS")) {
        http_response(fd, s->enable_cors, 204, NULL, "");
        http_request_free(&hr);
        goto done;
    }

    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models")) {
        send_models(s, fd);
        http_request_free(&hr);
        goto done;
    }
    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models/deepseek-v4-flash")) {
        send_model(s, fd);
        http_request_free(&hr);
        goto done;
    }

    request req;
    char err[160];
    bool ok = false;
    const int ctx_size = ds4_session_ctx(s->session);
    if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/messages")) {
        ok = parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/chat/completions")) {
        ok = parse_chat_request(s->engine, s, hr.body, s->default_tokens,
                                ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/responses")) {
        ok = parse_responses_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/completions")) {
        ok = parse_completion_request(s->engine, hr.body, s->default_tokens,
                                      ctx_size, &req, err, sizeof(err));
    } else {
        http_error(fd, s->enable_cors, 404, "unknown endpoint");
        http_request_free(&hr);
        goto done;
    }
    if (ok) req.raw_body = xstrndup(hr.body, hr.body_len);
    http_request_free(&hr);
    if (!ok) {
        http_error(fd, s->enable_cors, 400, err);
        goto done;
    }
    if (request_exceeds_context(&req, ctx_size)) {
        http_error_context_length_exceeded(fd, s->enable_cors, &req, req.prompt.len, ctx_size);
        request_free(&req);
        goto done;
    }

    set_client_socket_nonblocking(fd);
    job j;
    memset(&j, 0, sizeof(j));
    j.fd = fd;
    j.req = req;
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);

    pthread_mutex_lock(&j.mu);
    if (!enqueue(s, &j)) {
        pthread_mutex_unlock(&j.mu);
        http_error(fd, s->enable_cors, 503, "server shutting down");
        pthread_cond_destroy(&j.cv);
        pthread_mutex_destroy(&j.mu);
        request_free(&j.req);
        goto done;
    }
    while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
    pthread_mutex_unlock(&j.mu);

    pthread_cond_destroy(&j.cv);
    pthread_mutex_destroy(&j.mu);
    request_free(&j.req);
done:
    close(fd);
    client_done(s);
    return NULL;
}

static int listen_on(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = DS4_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static void set_client_socket_nonblocking(int fd) {
    /* The inference worker writes streaming responses itself.  Once a request is
     * queued, a blocked socket would block every other request too, so slow
     * clients are failed instead of back-pressuring the model session. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

typedef struct {
    ds4_engine_options engine;
    const char *host;
    int port;
    int ctx_size;
    int default_tokens;
    const char *chdir_path;
    const char *trace_path;
    const char *kv_disk_dir;
    uint64_t kv_disk_space_mb;
    kv_cache_options kv_cache;
    bool kv_cache_reject_different_quant;
    bool disable_exact_dsml_tool_replay;
    int tool_memory_max_ids;
    bool enable_cors;
} server_config;

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > INT_MAX) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonneg_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < 0 || v > INT_MAX) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

static float parse_float_arg(const char *s, const char *opt, float minv, float maxv) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || v < minv || v > maxv) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: missing value for %s", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static void log_context_memory(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = ds4_context_memory_estimate(backend, ctx_size);
    server_log(DS4_LOG_DEFAULT,
               "ds4-server: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)",
               (double)m.total_bytes / (1024.0 * 1024.0),
               ctx_size,
               ds4_backend_name(backend),
               m.prefill_cap,
               m.raw_cap,
               m.comp_cap);
}

static void server_close_resources(server *s) {
    if (s->trace) {
        fclose(s->trace);
        s->trace = NULL;
    }
    kv_cache_close(&s->kv);
    tool_memory_free(&s->tool_mem);
    live_tool_state_free(&s->responses_live);
    live_tool_state_free(&s->anthropic_live);
    visible_live_free(&s->thinking_live);
    pthread_mutex_destroy(&s->tool_mu);
    pthread_mutex_destroy(&s->trace_mu);
    pthread_cond_destroy(&s->clients_cv);
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);
    ds4_session_free(s->session);
    ds4_engine_close(s->engine);
    memset(s, 0, sizeof(*s));
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-server [options]\n"
        "\n"
        "Model and runtime:\n"
        "  -m, --model FILE\n"
        "      GGUF model path. Default: ds4flash.gguf\n"
        "  --mtp FILE\n"
        "      Optional MTP support GGUF used for draft-token probes.\n"
        "  --mtp-draft N\n"
        "      Maximum autoregressive MTP draft tokens per speculative step. Default: 1\n"
        "  --mtp-margin F\n"
        "      Minimum recursive-draft confidence for the fast N=2 verifier. Default: 3\n"
        "  -c, --ctx N\n"
        "      Context size allocated at startup. Default: 32768\n"
        "  -n, --tokens N\n"
        "      Default max output tokens when the client omits a limit. Default: 393216 (384K)\n"
        "  -t, --threads N\n"
        "      CPU helper threads for lightweight host-side work.\n"
        "  --chdir DIR\n"
        "      Change working directory before loading the model or runtime assets.\n"
        "  --quality\n"
        "      Prefer exact kernels where faster approximate paths exist; MTP uses strict verification.\n"
        "  --dir-steering-file FILE\n"
        "      Load one f32 direction vector per layer for directional steering.\n"
        "  --dir-steering-ffn F\n"
        "      Apply steering after FFN outputs: y -= F*v*dot(v,y). Default with file: 1\n"
        "  --dir-steering-attn F\n"
        "      Apply steering after attention outputs. Default: 0\n"
        "  --warm-weights\n"
        "      Touch mapped tensor pages before serving. Slower startup, fewer first-use stalls.\n"
        "  --metal | --cuda | --cpu | --backend NAME\n"
        "      Select backend explicitly. Defaults to Metal on macOS and CUDA on CUDA builds.\n"
        "\n"
        "HTTP API:\n"
        "  --host HOST\n"
        "      Bind address. Default: 127.0.0.1\n"
        "  --port N\n"
        "      Bind port. Default: 8000\n"
        "  --cors\n"
        "      Add Access-Control-Allow-* headers for browser JS clients. Does not change --host.\n"
        "  --trace FILE\n"
        "      Write a human-readable session trace: prompts, cache decisions, output, tool calls.\n"
        "\n"
        "Thinking and sampling:\n"
        "  DeepSeek-compatible chat requests default to thinking mode with high effort.\n"
        "  Only reasoning_effort=max or output_config.effort=max requests Think Max.\n"
        "  Think Max is applied only when --ctx is at least 393216 tokens; smaller contexts use high.\n"
        "  thinking={type:disabled}, think=false, or model=deepseek-chat selects non-thinking mode.\n"
        "  API defaults are temperature=1, top_p=1, min_p=0.05, and no top-k cap.\n"
        "  In thinking mode, client sampling knobs are ignored like the official API.\n"
        "\n"
        "Disk KV cache:\n"
        "  --kv-disk-dir DIR\n"
        "      Enable disk KV checkpoints in DIR. The directory is created if needed.\n"
        "  --kv-disk-space-mb N\n"
        "      Disk budget for checkpoint files. Default when enabled: 4096\n"
        "  --kv-cache-min-tokens N\n"
        "      Do not save or load checkpoints shorter than N tokens. Default: 512\n"
        "  --kv-cache-cold-max-tokens N\n"
        "      Cold first prompts in [min,N] are saved automatically. 0 disables cold saves. Default: 30000\n"
        "  --kv-cache-continued-interval-tokens N\n"
        "      Save at absolute aligned frontiers spaced about N tokens apart. 0 disables. Default: 10000\n"
        "  --kv-cache-boundary-trim-tokens N\n"
        "      Trim this many tail tokens before cold boundary saves to avoid tokenizer boundary merges. Default: 32\n"
        "  --kv-cache-boundary-align-tokens N\n"
        "      Align cold boundary saves down to this token multiple. 0 disables alignment. Default: 2048\n"
        "  --kv-cache-reject-different-quant\n"
        "      Refuse checkpoints written by the same model with a different routed-expert quantization.\n"
        "  --disable-exact-dsml-tool-replay\n"
        "      Disable the tool-id -> exact sampled DSML map. Tool history falls back to canonical JSON rendering.\n"
        "  --tool-memory-max-ids N\n"
        "      Maximum exact tool-call IDs kept in RAM for replay. Default: 100000\n"
        "\n"
        "  Cache triggers:\n"
        "      cold       save a stable prefix of a long first prompt before generation starts\n"
        "      continued  save absolute aligned restart frontiers during long prefill or generation\n"
        "      evict      save the live conversation before another request replaces it\n"
        "      shutdown   save the live conversation when the server exits cleanly\n"
        "\n"
        "Normal server command:\n"
        "  ./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192\n"
        "\n"
        "Notes:\n"
        "  Use /v1/chat/completions, /v1/responses, /v1/completions, or /v1/messages.\n"
        "  Larger --ctx values allocate more KV memory at startup; the startup log prints the estimate.\n"
        "  Disk KV caching is best for agents that resend long prompts with stable prefixes.\n"
        "\n"
        "  -h, --help\n"
        "      Show this help.\n");
}

static ds4_backend parse_backend_arg(const char *s, const char *arg) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    server_log(DS4_LOG_DEFAULT, "ds4-server: invalid %s value: %s", arg, s);
    server_log(DS4_LOG_DEFAULT, "ds4-server: valid server backends are: metal, cuda, cpu");
    exit(2);
}

static ds4_backend default_server_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static server_config parse_options(int argc, char **argv) {
    server_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_server_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .host = "127.0.0.1",
        .port = 8000,
        .ctx_size = 32768,
        .default_tokens = 393216,
        .tool_memory_max_ids = DS4_TOOL_MEMORY_DEFAULT_MAX_IDS,
    };
    c.kv_cache = kv_cache_default_options();

    bool directional_steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.default_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cors")) {
            c.enable_cors = true;
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-dir")) {
            c.kv_disk_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-space-mb")) {
            c.kv_disk_space_mb = (uint64_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-min-tokens")) {
            c.kv_cache.min_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-cold-max-tokens")) {
            c.kv_cache.cold_max_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-continued-interval-tokens")) {
            c.kv_cache.continued_interval_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-trim-tokens")) {
            c.kv_cache.boundary_trim_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-align-tokens")) {
            c.kv_cache.boundary_align_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-reject-different-quant")) {
            c.kv_cache_reject_different_quant = true;
        } else if (!strcmp(arg, "--disable-exact-dsml-tool-replay")) {
            c.disable_exact_dsml_tool_replay = true;
        } else if (!strcmp(arg, "--tool-memory-max-ids")) {
            c.tool_memory_max_ids = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else {
            server_log(DS4_LOG_DEFAULT, "ds4-server: unknown option: %s", arg);
            usage(stderr);
            exit(2);
        }
    }
    if (c.kv_cache.cold_max_tokens > 0 &&
        c.kv_cache.cold_max_tokens < c.kv_cache.min_tokens)
    {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens");
        exit(2);
    }
    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    return c;
}

#ifndef DS4_SERVER_TEST
int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    server_config cfg = parse_options(argc, argv);
    if (cfg.chdir_path && chdir(cfg.chdir_path) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to chdir to %s: %s",
                   cfg.chdir_path, strerror(errno));
        return 1;
    }

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;

    log_context_memory(cfg.engine.backend, cfg.ctx_size);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to create %s session",
                   ds4_backend_name(cfg.engine.backend));
        ds4_engine_close(engine);
        return 1;
    }

    server s;
    memset(&s, 0, sizeof(s));
    s.engine = engine;
    s.session = session;
    s.default_tokens = cfg.default_tokens;
    s.disable_exact_dsml_tool_replay = cfg.disable_exact_dsml_tool_replay;
    s.tool_mem.max_entries = cfg.tool_memory_max_ids;
    s.enable_cors = cfg.enable_cors;
    if (cfg.kv_disk_dir) {
        kv_cache_open(&s.kv, cfg.kv_disk_dir, cfg.kv_disk_space_mb,
                      cfg.kv_cache_reject_different_quant, cfg.kv_cache);
    }
    if (s.disable_exact_dsml_tool_replay) {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: exact DSML tool replay disabled; tool history uses canonical JSON rendering");
    }
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);
    pthread_cond_init(&s.clients_cv, NULL);
    pthread_mutex_init(&s.tool_mu, NULL);
    pthread_mutex_init(&s.trace_mu, NULL);
    if (cfg.trace_path) {
        s.trace = fopen(cfg.trace_path, "w");
        if (!s.trace) {
            server_log(DS4_LOG_DEFAULT, "ds4-server: failed to open trace file %s: %s",
                       cfg.trace_path, strerror(errno));
            server_close_resources(&s);
            return 1;
        }
        setvbuf(s.trace, NULL, _IONBF, 0);
        server_log(DS4_LOG_DEFAULT, "ds4-server: tracing session to %s", cfg.trace_path);
    }

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_main, &s) != 0) die("failed to start worker");

    int lfd = listen_on(cfg.host, cfg.port);
    if (lfd < 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to listen on %s:%d: %s", cfg.host, cfg.port, strerror(errno));
        pthread_mutex_lock(&s.mu);
        s.stopping = true;
        pthread_cond_broadcast(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(worker, NULL);
        server_close_resources(&s);
        return 1;
    }
    g_listen_fd = lfd;
    server_log(DS4_LOG_DEFAULT, "ds4-server: listening on http://%s:%d", cfg.host, cfg.port);

    while (!g_stop_requested) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop_requested) break;
            if (errno == EINTR) continue;
            server_log(DS4_LOG_DEFAULT, "ds4-server: accept failed: %s", strerror(errno));
            continue;
        }
        if (g_stop_requested) {
            close(fd);
            break;
        }

        configure_client_socket(fd);
        client_arg *ca = xmalloc(sizeof(*ca));
        ca->srv = &s;
        ca->fd = fd;
        pthread_mutex_lock(&s.mu);
        s.clients++;
        pthread_mutex_unlock(&s.mu);
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, ca) != 0) {
            pthread_mutex_lock(&s.mu);
            s.clients--;
            pthread_cond_broadcast(&s.clients_cv);
            pthread_mutex_unlock(&s.mu);
            free(ca);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) {
        close(lfd);
        g_listen_fd = -1;
    }

    server_log(DS4_LOG_DEFAULT, "ds4-server: shutdown requested, draining requests");
    pthread_mutex_lock(&s.mu);
    s.stopping = true;
    pthread_cond_broadcast(&s.cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(worker, NULL);
    pthread_mutex_lock(&s.mu);
    while (s.clients > 0) pthread_cond_wait(&s.clients_cv, &s.mu);
    pthread_mutex_unlock(&s.mu);

    const ds4_tokens *tokens = ds4_session_tokens(s.session);
    if (s.kv.enabled && tokens && tokens->len >= s.kv.opt.min_tokens) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: persisting current KV cache before shutdown tokens=%d",
                   tokens->len);
        kv_cache_store_current(&s, "shutdown");
    }
    server_close_resources(&s);
    return 0;
}
#else

static int test_failures = 0;

static void test_assert(bool cond, const char *file, int line, const char *expr) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    test_failures++;
}

#define TEST_ASSERT(expr) test_assert((expr), __FILE__, __LINE__, #expr)

static void test_tool_schema_order_from_anthropic_schema(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"},"
        "\"description\":{\"type\":\"string\"}}}}");
    const tool_schema_order *order = tool_schema_orders_find(&orders, "bash");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->len == 2);
    TEST_ASSERT(order && !strcmp(order->prop[0], "command"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "description"));
    tool_schema_orders_free(&orders);
}

static void test_tool_schema_order_from_openai_tools(void) {
    const char *json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"parameters\":{"
        "\"type\":\"object\",\"properties\":{"
        "\"filePath\":{\"type\":\"string\"},"
        "\"oldString\":{\"type\":\"string\"},"
        "\"newString\":{\"type\":\"string\"}}}}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"edit\""));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "edit");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->len == 3);
    TEST_ASSERT(order && !strcmp(order->prop[0], "filePath"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "oldString"));
    TEST_ASSERT(order && !strcmp(order->prop[2], "newString"));
    free(schemas);
    tool_schema_orders_free(&orders);
}

static void test_tool_schema_order_from_responses_tool_search(void) {
    const char *json =
        "[{\"type\":\"tool_search\",\"execution\":\"client\","
        "\"description\":\"Search deferred tools\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},"
        "\"limit\":{\"type\":\"number\"}},\"required\":[\"query\"]}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"tool_search\""));
    TEST_ASSERT(schemas && strstr(schemas, "\"description\":\"Search deferred tools\""));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "tool_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->responses_tool_search);
    TEST_ASSERT(order && order->len == 2);
    TEST_ASSERT(order && !strcmp(order->prop[0], "query"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "limit"));
    free(schemas);
    tool_schema_orders_free(&orders);
}

static void test_responses_function_named_tool_search_stays_function_call(void) {
    const char *json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"tool_search\","
        "\"description\":\"A normal user function that happens to use a reserved name\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "tool_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && !order->responses_tool_search);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_user_tool_search");
    tc.name = xstrdup("tool_search");
    tc.arguments = xstrdup("{\"query\":\"plain function\"}");
    tool_calls_push(&calls, tc);
    responses_tool_item item = {
        .fc_id = "fc_user_tool_search",
        .call_id = "call_user_tool_search",
        .is_custom = false,
        .output_index = 0,
    };

    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"function_call\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"tool_search_call\"") == NULL);

    buf_free(&out);
    tool_calls_free(&calls);
    free(schemas);
    tool_schema_orders_free(&orders);
}

static void test_responses_namespace_tool_schemas_restore_wire_namespace(void) {
    const char *json =
        "[{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"description\":\"Perplexity tools\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"description\":\"Search the web\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},"
        "\"recency\":{\"type\":\"number\"}}}}]}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"mcp__perplexity__perplexity_search\""));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"perplexity_search\"") == NULL);

    const tool_schema_order *order =
        tool_schema_orders_find(&orders, "mcp__perplexity__perplexity_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->namespace && !strcmp(order->namespace, "mcp__perplexity__"));
    TEST_ASSERT(order && order->wire_name && !strcmp(order->wire_name, "perplexity_search"));
    TEST_ASSERT(order && order->len == 2);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_ns");
    tc.name = xstrdup("mcp__perplexity__perplexity_search");
    tc.arguments = xstrdup("{\"query\":\"deepseek\",\"recency\":7}");
    tool_calls_push(&calls, tc);
    responses_tool_item item = {
        .fc_id = "fc_ns",
        .call_id = "call_ns",
        .is_custom = false,
        .output_index = 0,
    };
    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"name\":\"perplexity_search\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"namespace\":\"mcp__perplexity__\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "mcp__perplexity__perplexity_search") == NULL);

    buf_free(&out);
    tool_calls_free(&calls);
    free(schemas);
    tool_schema_orders_free(&orders);
}

static void test_responses_input_tool_search_output_loads_tools(void) {
    const char *json =
        "["
        "{\"type\":\"tool_search_call\",\"call_id\":\"call_search\","
        "\"execution\":\"client\",\"arguments\":{\"query\":\"perplexity\"}},"
        "{\"type\":\"tool_search_output\",\"call_id\":\"call_search\","
        "\"status\":\"completed\",\"execution\":\"client\",\"tools\":["
        "{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"description\":\"Perplexity tools\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"description\":\"Search with Perplexity\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}]}]}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    buf loaded = {0};
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_responses_input(&p, &msgs, &loaded, &orders));
    TEST_ASSERT(loaded.ptr && strstr(loaded.ptr, "\"name\":\"mcp__perplexity__perplexity_search\""));
    const tool_schema_order *order =
        tool_schema_orders_find(&orders, "mcp__perplexity__perplexity_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->namespace && !strcmp(order->namespace, "mcp__perplexity__"));
    TEST_ASSERT(order && order->wire_name && !strcmp(order->wire_name, "perplexity_search"));
    TEST_ASSERT(msgs.len == 2);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(!strcmp(msgs.v[0].calls.v[0].name, "tool_search"));
    TEST_ASSERT(strstr(msgs.v[1].content, "mcp__perplexity__") != NULL);

    buf_free(&loaded);
    tool_schema_orders_free(&orders);
    chat_msgs_free(&msgs);
}

static void test_responses_input_tool_search_output_rejects_bad_tools(void) {
    const char *json =
        "[{\"type\":\"tool_search_output\",\"call_id\":\"call_search\","
        "\"status\":\"completed\",\"tools\":{\"not\":\"a tool array\"}}]";
    const char *p = json;
    chat_msgs msgs = {0};
    buf loaded = {0};
    tool_schema_orders orders = {0};
    TEST_ASSERT(!parse_responses_input(&p, &msgs, &loaded, &orders));
    buf_free(&loaded);
    tool_schema_orders_free(&orders);
    chat_msgs_free(&msgs);
}

static void test_responses_input_function_call_namespace_round_trips_to_dsml(void) {
    const char *tools_json =
        "[{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}]}]";
    const char *tools_p = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tools_p, &schemas, &orders));

    const char *input_json =
        "[{\"type\":\"function_call\",\"call_id\":\"call_ns\","
        "\"name\":\"perplexity_search\",\"namespace\":\"mcp__perplexity__\","
        "\"arguments\":{\"query\":\"deepseek\"}}]";
    const char *input_p = input_json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_responses_input(&input_p, &msgs, NULL, NULL));
    TEST_ASSERT(msgs.len == 1);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(!strcmp(msgs.v[0].calls.v[0].name,
                        "mcp__perplexity__perplexity_search"));

    char *prompt = render_chat_prompt_text(&msgs, schemas, &orders, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt,
        "<｜DSML｜invoke name=\"mcp__perplexity__perplexity_search\">") != NULL);
    TEST_ASSERT(strstr(prompt, "<｜DSML｜invoke name=\"perplexity_search\">") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
    free(schemas);
    tool_schema_orders_free(&orders);
}

static void test_responses_output_sends_tool_search_call_item(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_search");
    tc.name = xstrdup("tool_search");
    tc.arguments = xstrdup("{\"limit\":3,\"query\":\"perplexity\"}");
    tool_calls_push(&calls, tc);
    const char *tools_json =
        "[{\"type\":\"tool_search\",\"execution\":\"client\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},\"limit\":{\"type\":\"number\"}}}}]";
    const char *tools_p = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tools_p, &schemas, &orders));
    responses_tool_item item = {
        .fc_id = "fc_search",
        .call_id = "call_search",
        .is_custom = false,
        .output_index = 0,
    };

    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"tool_search_call\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"execution\":\"client\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"status\":\"completed\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"arguments\":{\"limit\":3,\"query\":\"perplexity\"}") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"function_call\"") == NULL);

    buf_free(&out);
    free(schemas);
    tool_schema_orders_free(&orders);
    tool_calls_free(&calls);
}

static tool_calls make_swapped_bash_call(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&calls, tc);
    return calls;
}

static tool_schema_orders make_bash_order(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"},"
        "\"description\":{\"type\":\"string\"}}}}");
    return orders;
}

static char *read_socket_text(int fd) {
    buf b = {0};
    char tmp[1024];
    ssize_t n;
    while ((n = read(fd, tmp, sizeof(tmp))) > 0) {
        buf_append(&b, tmp, (size_t)n);
    }
    return buf_take(&b);
}

static void test_context_length_error_uses_protocol_standard_shape(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.prompt.len = 16;
    TEST_ASSERT(request_exceeds_context(&r, 16));
    TEST_ASSERT(!request_exceeds_context(&r, 17));

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], false, &r, 16, 16));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 400") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"code\":\"context_length_exceeded\"") != NULL);
        TEST_ASSERT(strstr(out, "\"param\":\"messages\"") != NULL);
        TEST_ASSERT(strstr(out, "\"n_prompt_tokens\":16") != NULL);
        TEST_ASSERT(strstr(out, "\"n_ctx\":16") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
    request_free(&r);

    request a;
    request_init(&a, REQ_CHAT, 128);
    a.api = API_ANTHROPIC;

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], false, &a, 20, 20));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "{\"type\":\"error\",\"error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"n_prompt_tokens\":20") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
    request_free(&a);
}

static void test_cors_headers_are_opt_in(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_response(sv[0], false, 200, "application/json", "{}"));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin") == NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_response(sv[0], true, 200, "application/json", "{}"));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin: *") != NULL);
        TEST_ASSERT(strstr(out, "Access-Control-Allow-Methods: GET, POST, OPTIONS") != NULL);
        TEST_ASSERT(strstr(out, "Access-Control-Allow-Headers: *") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
}

static void test_cors_preflight_response_is_no_content(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    TEST_ASSERT(http_response(sv[0], true, 204, NULL, ""));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "HTTP/1.1 204 No Content") != NULL);
    TEST_ASSERT(strstr(out, "Content-Length: 0") != NULL);
    TEST_ASSERT(strstr(out, "Content-Type:") == NULL);
    TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin: *") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);
}

static void test_cors_sse_headers(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    TEST_ASSERT(sse_headers(sv[0], true));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(out, "Content-Type: text/event-stream") != NULL);
    TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin: *") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);
}

static void test_anthropic_live_stream_sends_incremental_blocks(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.stream = true;
    r.think_mode = DS4_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_test", 10, &st));
    const char *raw1 = "need a tool</think>Hello.\n\n";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_test", &st,
                                            raw1, strlen(raw1), false));

    const char *raw =
        "need a tool</think>Hello.\n\n"
        DS4_TOOL_CALLS_START "\n";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_test", &st,
                                            raw, strlen(raw), false));

    tool_calls calls = make_swapped_bash_call();
    TEST_ASSERT(anthropic_sse_finish_live(sv[0], NULL, &r, "msg_test", &st,
                                          raw, strlen(raw), &calls,
                                          "tool_calls", 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *msg_start = strstr(out, "event: message_start");
    const char *thinking = strstr(out, "\"thinking\":\"need a tool\"");
    const char *signature = strstr(out, "\"type\":\"signature_delta\"");
    const char *text = strstr(out, "\"text\":\"Hello.\"");
    const char *tool = strstr(out, "\"type\":\"tool_use\"");
    const char *stop = strstr(out, "event: message_stop");
    TEST_ASSERT(msg_start != NULL);
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(signature != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(stop != NULL);
    TEST_ASSERT(msg_start < thinking);
    TEST_ASSERT(thinking < signature);
    TEST_ASSERT(signature < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < stop);
    TEST_ASSERT(strstr(out, DS4_TOOL_CALLS_START) == NULL);

    free(out);
    tool_calls_free(&calls);
    anthropic_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_anthropic_tool_stream_sends_live_tool_use(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_tool", 7, &st));

    const char *raw =
        "Before.\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">echo partial";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_tool", &st,
                                            raw, strlen(raw), false));

    const char *raw_complete =
        "Before.\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">echo partial done" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_tool", &st,
                                            raw_complete, strlen(raw_complete), false));

    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(raw_complete, false, &parsed_content,
                                           &parsed_reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    apply_anthropic_stream_tool_ids(&calls, &st);
    TEST_ASSERT(calls.v[0].id != NULL);
    TEST_ASSERT(!strncmp(calls.v[0].id, "toolu_", 6));
    TEST_ASSERT(anthropic_sse_finish_live(sv[0], NULL, &r, "msg_tool", &st,
                                          raw_complete, strlen(raw_complete),
                                          &calls, "tool_calls", 5));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *text = strstr(out, "\"text\":\"Before.\"");
    const char *tool = strstr(out, "\"type\":\"tool_use\"");
    const char *key = strstr(out, "\\\"command\\\":\\\"");
    const char *partial = strstr(out, "\"partial_json\":\"echo partial\"");
    const char *rest = strstr(out, "\"partial_json\":\" done\"");
    const char *stop = strstr(out, "event: message_stop");
    int tool_use_count = 0;
    for (const char *p = out; (p = strstr(p, "\"type\":\"tool_use\"")) != NULL; p++) {
        tool_use_count++;
    }
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(key != NULL);
    TEST_ASSERT(partial != NULL);
    TEST_ASSERT(rest != NULL);
    TEST_ASSERT(stop != NULL);
    TEST_ASSERT(strstr(out, calls.v[0].id) != NULL);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < key);
    TEST_ASSERT(key < partial);
    TEST_ASSERT(partial < rest);
    TEST_ASSERT(rest < stop);
    TEST_ASSERT(tool_use_count == 1);
    TEST_ASSERT(strstr(out, DS4_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, DS4_PARAM_START) == NULL);

    free(out);
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&calls);
    anthropic_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_anthropic_usage_reports_cache_details(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    TEST_ASSERT(anthropic_final_response(sv[0], false, &r, "msg_usage", "OK", NULL, NULL, "stop", 10, 2));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_read_input_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_creation_input_tokens\":3") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_usage_stream", 10, &st));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "event: message_start") != NULL);
    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_read_input_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_creation_input_tokens\":3") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);
    request_free(&r);
}

static void test_openai_tool_stream_sends_incremental_text(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_test", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "<think>need a tool</think>Hello.\n\n";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw1, strlen(raw1), false));

    const char *raw =
        "<think>need a tool</think>Hello.\n\n"
        DS4_TOOL_CALLS_START "\n";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw, strlen(raw), false));

    tool_calls calls = make_swapped_bash_call();
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_test", &st,
                                       raw, strlen(raw), &calls,
                                       "tool_calls", 10, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *role = strstr(out, "\"role\":\"assistant\"");
    const char *thinking = strstr(out, "\"reasoning_content\":\"need a tool\"");
    const char *text = strstr(out, "\"content\":\"Hello.\"");
    const char *tool = strstr(out, "\"tool_calls\"");
    const char *done = strstr(out, "data: [DONE]");
    TEST_ASSERT(role != NULL);
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(done != NULL);
    TEST_ASSERT(role < thinking);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < done);
    TEST_ASSERT(strstr(out, DS4_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, "<think>") == NULL);

    free(out);
    tool_calls_free(&calls);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_stream_usage_reports_cache_details(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.stream_include_usage = true;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    TEST_ASSERT(sse_done(sv[0], &r, "chatcmpl_usage", 10, 2));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"prompt_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"completion_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);
    TEST_ASSERT(strstr(out, "\"prompt_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "data: [DONE]") != NULL);

    free(out);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_responses_usage_reports_cache_details(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    TEST_ASSERT(responses_final_response(sv[0], false, &r, "resp_usage", "OK", NULL, NULL,
                                         "stop", 10, 2));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"input_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    responses_stream st;
    responses_stream_init(&r, &st);
    TEST_ASSERT(responses_sse_completed(sv[0], &r, &st, NULL, NULL,
                                        "stop", 10, 2, 1234));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"type\":\"response.completed\"") != NULL);
    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"input_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);

    free(out);
    responses_stream_free(&st);
    close(sv[0]);
    close(sv[1]);
    request_free(&r);
}

static void test_openai_chat_stream_splits_reasoning_without_tools(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_HIGH;
    r.has_tools = false;

    TEST_ASSERT(request_uses_structured_stream(&r));
    TEST_ASSERT(request_uses_openai_live_stream(&r));
    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_title", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "We need to generate a title";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_title", &st,
                                         raw1, strlen(raw1), false));

    const char *raw2 =
        "We need to generate a title</think>Free disk space check";
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_title", &st,
                                       raw2, strlen(raw2), NULL,
                                       "stop", 12, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *role = strstr(out, "\"role\":\"assistant\"");
    const char *reasoning1 = strstr(out, "\"reasoning_content\":\"We need to generate \"");
    const char *reasoning2 = strstr(out, "\"reasoning_content\":\"a title\"");
    const char *content = strstr(out, "\"content\":\"Free disk space check\"");
    const char *done = strstr(out, "data: [DONE]");
    TEST_ASSERT(role != NULL);
    TEST_ASSERT(reasoning1 != NULL);
    TEST_ASSERT(reasoning2 != NULL);
    TEST_ASSERT(content != NULL);
    TEST_ASSERT(done != NULL);
    TEST_ASSERT(role < reasoning1);
    TEST_ASSERT(reasoning1 < reasoning2);
    TEST_ASSERT(reasoning2 < content);
    TEST_ASSERT(content < done);
    TEST_ASSERT(strstr(out, "\"content\":\"We need to generate a title") == NULL);
    TEST_ASSERT(strstr(out, "</think>") == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_sends_partial_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_partial_tool", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        "Before.\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">echo partial";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                         raw, strlen(raw), false));

    const char *raw_complete =
        "Before.\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">echo partial done" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                         raw_complete, strlen(raw_complete), false));

    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(raw_complete, false, &parsed_content, &parsed_reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    apply_openai_stream_tool_ids(&calls, &st);
    TEST_ASSERT(calls.v[0].id != NULL);
    TEST_ASSERT(!strncmp(calls.v[0].id, "call_", 5));
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                       raw_complete, strlen(raw_complete), &calls,
                                       "tool_calls", 10, 4));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *text = strstr(out, "\"content\":\"Before.\"");
    const char *tool = strstr(out, "\"tool_calls\"");
    const char *key = strstr(out, "\\\"command\\\":\\\"");
    const char *partial = strstr(out, "\"arguments\":\"echo partial\"");
    const char *rest = strstr(out, "\"arguments\":\" done\"");
    int tool_id_count = 0;
    for (const char *p = out; (p = strstr(p, "\"id\":\"call_")) != NULL; p++) tool_id_count++;
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(key != NULL);
    TEST_ASSERT(partial != NULL);
    TEST_ASSERT(rest != NULL);
    TEST_ASSERT(strstr(out, calls.v[0].id) != NULL);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < partial);
    TEST_ASSERT(partial < rest);
    TEST_ASSERT(tool_id_count == 1);
    TEST_ASSERT(strstr(out, DS4_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, DS4_PARAM_START) == NULL);

    free(out);
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&calls);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_waits_for_incomplete_tool_tags(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw_invoke = DS4_TOOL_CALLS_START "\n" DS4_INVOKE_START;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_incomplete_tool", &st,
                                         raw_invoke, strlen(raw_invoke), false));
    TEST_ASSERT(st.mode == OPENAI_STREAM_TOOL);
    TEST_ASSERT(st.tool.state == DSML_TOOL_BETWEEN_INVOKES);

    const char *raw_param =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_incomplete_tool", &st,
                                         raw_param, strlen(raw_param), false));
    TEST_ASSERT(st.mode == OPENAI_STREAM_TOOL);
    TEST_ASSERT(st.tool.state == DSML_TOOL_BETWEEN_PARAMS);

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, DS4_PARAM_START) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_sends_partial_raw_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"edits\" string=\"false\">[1,2,3";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_raw_tool", &st,
                                         raw, strlen(raw), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"name\":\"edit\"") != NULL);
    TEST_ASSERT(strstr(out, "\\\"edits\\\":") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\"[1,2,3\"") != NULL);
    TEST_ASSERT(strstr(out, DS4_TOOL_CALLS_START) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_holds_partial_dsml_entities(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw_partial =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">echo &amp";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_entity_tool", &st,
                                         raw_partial, strlen(raw_partial), false));

    const char *raw_complete =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">echo &amp; done" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_entity_tool", &st,
                                         raw_complete, strlen(raw_complete), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"arguments\":\"echo \"") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\"& done\"") != NULL);
    TEST_ASSERT(strstr(out, "&amp") == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_holds_partial_utf8_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char prefix[] =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"write\">\n"
        DS4_PARAM_START " name=\"content\" string=\"true\">flag ";
    const char suffix[] =
        " done" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;
    const char flag_utf8[] = {(char)0xf0, (char)0x9f, (char)0x9a, (char)0xa9, 0};
    const char replacement[] = {(char)0xef, (char)0xbf, (char)0xbd, 0};

    buf partial = {0};
    buf_append(&partial, prefix, strlen(prefix));
    buf_putc(&partial, (char)0xf0);
    buf_putc(&partial, (char)0x9f);
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8_tool", &st,
                                         partial.ptr, partial.len, false));

    buf complete = {0};
    buf_append(&complete, prefix, strlen(prefix));
    buf_append(&complete, flag_utf8, 4);
    buf_append(&complete, suffix, strlen(suffix));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8_tool", &st,
                                         complete.ptr, complete.len, false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"arguments\":\"flag \"") != NULL);
    TEST_ASSERT(strstr(out, flag_utf8) != NULL);
    TEST_ASSERT(strstr(out, replacement) == NULL);

    free(out);
    buf_free(&partial);
    buf_free(&complete);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_openai_tool_stream_handles_multiple_calls(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"read\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">a.c" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">wc -l a.c" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_multi_tool", &st,
                                         raw, strlen(raw), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    int tool_id_count = 0;
    for (const char *p = out; (p = strstr(p, "\"id\":\"call_")) != NULL; p++) tool_id_count++;
    TEST_ASSERT(tool_id_count == 2);
    TEST_ASSERT(strstr(out, "\"name\":\"read\"") != NULL);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, "\\\"path\\\":") != NULL);
    TEST_ASSERT(strstr(out, "\\\"command\\\":") != NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_streaming_holds_partial_utf8(void) {
    const char partial[] = {'A', ' ', (char)0xf0, (char)0x9f, 0};
    const char complete[] = {'A', ' ', (char)0xf0, (char)0x9f,
                             (char)0x9a, (char)0xa9, ' ', 'd', 'o', 'n', 'e', 0};
    const char flag_done[] = {(char)0xf0, (char)0x9f,
                              (char)0x9a, (char)0xa9, ' ', 'd', 'o', 'n', 'e', 0};
    const char replacement[] = {(char)0xef, (char)0xbf, (char)0xbd, 0};

    TEST_ASSERT(utf8_stream_safe_len(partial, 0, strlen(partial), false) == 2);
    TEST_ASSERT(utf8_stream_safe_len(complete, 0, strlen(complete), false) == strlen(complete));

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = DS4_THINK_NONE;

    openai_stream st;
    openai_stream_start(&r, &st);
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8", &st,
                                         partial, strlen(partial), false));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8", &st,
                                         complete, strlen(complete), false));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"content\":\"A \"") != NULL);
    TEST_ASSERT(strstr(out, flag_done) != NULL);
    TEST_ASSERT(strstr(out, replacement) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}

static void test_request_defaults_use_min_p_filtering(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    TEST_ASSERT(r.think_mode == DS4_THINK_HIGH);
    TEST_ASSERT(r.temperature == DS4_DEFAULT_TEMPERATURE);
    TEST_ASSERT(r.top_p == DS4_DEFAULT_TOP_P);
    TEST_ASSERT(r.top_k == 0);
    TEST_ASSERT(r.min_p == DS4_DEFAULT_MIN_P);
    request_free(&r);
}

static void test_reasoning_effort_mapping(void) {
    ds4_think_mode mode = DS4_THINK_NONE;
    TEST_ASSERT(parse_reasoning_effort_name("low", &mode) && mode == DS4_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("medium", &mode) && mode == DS4_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("high", &mode) && mode == DS4_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("xhigh", &mode) && mode == DS4_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("max", &mode) && mode == DS4_THINK_MAX);
    TEST_ASSERT(!parse_reasoning_effort_name("banana", &mode));
    TEST_ASSERT(ds4_think_mode_for_context(DS4_THINK_MAX, 32768) == DS4_THINK_HIGH);
    TEST_ASSERT(ds4_think_mode_for_context(DS4_THINK_MAX,
                                           (int)ds4_think_max_min_context()) == DS4_THINK_MAX);
}

static void test_api_thinking_controls_parse(void) {
    bool enabled = true;
    const char *thinking = "{\"type\":\"disabled\",\"budget_tokens\":1024}";
    TEST_ASSERT(parse_thinking_control_value(&thinking, &enabled));
    TEST_ASSERT(!enabled);
    thinking = "true";
    TEST_ASSERT(parse_thinking_control_value(&thinking, &enabled));
    TEST_ASSERT(enabled);

    ds4_think_mode mode = DS4_THINK_HIGH;
    const char *anth_effort = "{\"effort\":\"max\",\"other\":true}";
    TEST_ASSERT(parse_output_config_effort(&anth_effort, &mode));
    TEST_ASSERT(mode == DS4_THINK_MAX);

    const char *openai_effort = "\"xhigh\"";
    mode = DS4_THINK_HIGH;
    TEST_ASSERT(parse_reasoning_effort_value(&openai_effort, &mode));
    TEST_ASSERT(mode == DS4_THINK_HIGH);
}

static void test_render_think_max_prompt_prefix(void) {
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_MAX);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(!strncmp(prompt, "<｜begin▁of▁sentence｜>", strlen("<｜begin▁of▁sentence｜>")));
    TEST_ASSERT(strstr(prompt, ds4_think_max_prefix()) != NULL);
    TEST_ASSERT(strstr(prompt, "You are terse.<｜User｜>Hello<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(prompt, "</think>") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_non_thinking_prompt_closes_think(void) {
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_NONE);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, ds4_think_max_prefix()) == NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>Hello<｜Assistant｜></think>") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_drops_old_reasoning_without_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("old hidden reasoning");
    assistant.content = xstrdup("first answer");
    chat_msgs_push(&msgs, assistant);
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("second");
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "old hidden reasoning") == NULL);
    TEST_ASSERT(strstr(prompt, "<｜Assistant｜></think>first answer") != NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>second<｜Assistant｜><think>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_preserves_reasoning_with_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("tool reasoning");
    assistant.content = xstrdup("");
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>/tmp</tool_result>") != NULL);
    free(prompt);

    prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>/tmp</tool_result>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_chat_prompt_text_renders_tools_before_system(void) {
    /* The tool-schema block must sit at the head of the system region so the
     * client's system content stays at the tail, right before <｜User｜>.
     * That keeps a per-request dynamic tail (e.g. a timestamp) out of the
     * cached prefix without losing the tool schemas to the trim. */
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("CLIENT_SYSTEM_MARKER");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, "TOOL_SCHEMA_MARKER", NULL,
                                           DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    const char *tools  = strstr(prompt, "## Tools");
    const char *client = strstr(prompt, "CLIENT_SYSTEM_MARKER");
    const char *user_m = strstr(prompt, "<｜User｜>");
    TEST_ASSERT(tools && client && user_m);
    TEST_ASSERT(tools  < client);
    TEST_ASSERT(client < user_m);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_dsml_tool_args_preserve_call_order(void) {
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_dsml_tool_calls_text(&b, &calls);
    const char *command = strstr(b.ptr, "name=\"command\"");
    const char *description = strstr(b.ptr, "name=\"description\"");
    const char *timeout = strstr(b.ptr, "name=\"timeout\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
}

static void test_openai_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_tool_calls_json(&b, &calls, "test", &r.tool_orders);
    const char *command = strstr(b.ptr, "\\\"command\\\"");
    const char *description = strstr(b.ptr, "\\\"description\\\"");
    const char *timeout = strstr(b.ptr, "\\\"timeout\\\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_anthropic_thinking_and_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_anthropic_content(&b, "done", "thinking text", &calls, "msg_1", &r.tool_orders);
    const char *thinking = strstr(b.ptr, "\"type\":\"thinking\"");
    const char *text = strstr(b.ptr, "\"type\":\"text\"");
    const char *tool = strstr(b.ptr, "\"type\":\"tool_use\"");
    const char *command = strstr(b.ptr, "\"command\"");
    const char *description = strstr(b.ptr, "\"description\"");
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_parse_short_dsml_and_canonical_suffix(void) {
    const char *generated =
        "<think>need a tool</think>"
        "<DSML｜tool_calls>\n"
        "<DSML｜invoke name=\"bash\">\n"
        "<DSML｜parameter name=\"description\" string=\"true\">list files</DSML｜parameter>\n"
        "<DSML｜parameter name=\"command\" string=\"true\">ls -la</DSML｜parameter>\n"
        "</DSML｜invoke>\n"
        "</DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.tool_orders = make_bash_order();
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    const char *command = strstr(suffix, "name=\"command\"");
    const char *description = strstr(suffix, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(strstr(suffix, "</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜end▁of▁sentence｜>") != NULL);

    free(suffix);
    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_dsml_parser_recovers_loose_nested_parameters(void) {
    const char *generated =
        "review done\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"path\">/private/tmp/tetris.c" DS4_PARAM_END "\n"
        DS4_PARAM_START " name=\"edits\">\n"
        DS4_PARAM_START " name=\"oldText\" string=\"true\">old &lt;text&gt;" DS4_PARAM_END "\n"
        DS4_PARAM_START " name=\"newText\" string=\"true\">new text" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(content && !strcmp(content, "review done"));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "edit"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/private/tmp/tetris.c\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"edits\": {") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"oldText\":\"old <text>\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"newText\":\"new text\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_tool_parse_failure_returns_recoverable_finish(void) {
    const char *generated =
        "trying a tool\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START ">\n"
        DS4_TOOL_CALLS_END;

    char err[128] = {0};
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    const char *finish = "tool_calls";
    bool recovered = false;

    TEST_ASSERT(!parse_generated_message_for_response(generated,
                                                       true,
                                                       true,
                                                       false,
                                                       &finish,
                                                       err,
                                                       sizeof(err),
                                                       &content,
                                                       &reasoning,
                                                       &calls,
                                                       &recovered));
    TEST_ASSERT(recovered);
    TEST_ASSERT(!strcmp(finish, "stop"));
    TEST_ASSERT(!strcmp(err, "invalid tool call"));
    TEST_ASSERT(content && strstr(content, DS4_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(reasoning == NULL);
    TEST_ASSERT(calls.len == 0);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_thinking_dsml_is_not_executable_before_think_close(void) {
    const char *generated =
        "<think>I might mention a malformed or tentative tool call here:\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">true" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END
        "\nBut it is still reasoning, not an assistant action.</think>Final answer.";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 0);
    TEST_ASSERT(reasoning && strstr(reasoning, DS4_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(content && !strcmp(content, "Final answer."));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_thinking_dsml_after_think_close_is_executable(void) {
    const char *generated =
        "<think>need a shell check</think>\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">pwd" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a shell check"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"pwd\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_tool_checkpoint_suffix_is_future_prompt_canonical(void) {
    tool_schema_orders orders = make_bash_order();
    const char *tool_schemas =
        "{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{},\"description\":{},\"timeout\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("inspect");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, DS4_THINK_HIGH);

    const char *generated =
        "need a tool</think>\n\n"
        DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">cd /tmp && git diff 2>/dev/null</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(strstr(calls.v[0].arguments, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "&amp;&amp;") == NULL);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    TEST_ASSERT(strstr(suffix, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(suffix, "&amp;&amp;") == NULL);
    TEST_ASSERT(strstr(suffix, "2&gt;/dev/null") == NULL);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("inspect");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, DS4_THINK_HIGH);

    TEST_ASSERT(!strcmp(canonical.ptr, future_prompt));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}

static void test_tool_checkpoint_minifies_json_parameters(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}");
    const char *tool_schemas =
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("edit");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, DS4_THINK_HIGH);

    const char *generated =
        "need edit</think>\n\n"
        DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"edit\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/file</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"edits\" string=\"false\">"
        "[{\"oldText\": \"status=created\", \"newText\": \"status=created\\nstatus2=resumed\"}]"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("edit");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, DS4_THINK_HIGH);

    TEST_ASSERT(!strcmp(canonical.ptr, future_prompt));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}

static void test_tool_memory_replays_sampled_dsml(void) {
    const char *generated =
        "<think>need shell</think>\n\n"
        DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls sampled = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &sampled));
    TEST_ASSERT(sampled.len == 1);

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    assign_tool_call_ids(&s, &sampled, API_OPENAI);
    TEST_ASSERT(sampled.v[0].id != NULL);
    TEST_ASSERT(!strncmp(sampled.v[0].id, "call_", 5));
    tool_memory_remember(&s, &sampled);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    tool_call tc = {0};
    tc.id = xstrdup(sampled.v[0].id);
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.disk == 0);
    TEST_ASSERT(stats.canonical == 0);
    TEST_ASSERT(stats.missing_ids == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    const char *command = strstr(prompt, "name=\"command\"");
    const char *timeout = strstr(prompt, "name=\"timeout\"");
    const char *description = strstr(prompt, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < timeout);
    TEST_ASSERT(timeout < description);

    free(prompt);
    chat_msgs_free(&msgs);
    free(content);
    free(reasoning);
    tool_calls_free(&sampled);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_anthropic_tool_memory_replays_sampled_dsml(void) {
    const char *sampled_dsml =
        "\n\n" DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"Bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        DS4_TOOL_CALLS_END;

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    tool_memory_put(&s, "toolu_exact", sampled_dsml);

    const char *json =
        "["
        "{\"role\":\"assistant\",\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"toolu_exact\",\"name\":\"Bash\","
        "\"input\":{\"description\":\"list files\",\"command\":\"ls -la\"}}"
        "]},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_exact\",\"content\":\"ok\"}"
        "]}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 2);
    TEST_ASSERT(msgs.v[1].tool_call_id && !strcmp(msgs.v[1].tool_call_id, "toolu_exact"));

    stop_list ids = {0};
    collect_tool_call_ids(&msgs, &ids);
    TEST_ASSERT(id_list_contains(&ids, "toolu_exact"));
    id_list_free(&ids);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.canonical == 0);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    const char *command = strstr(prompt, "name=\"command\"");
    const char *description = strstr(prompt, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < description);

    free(prompt);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_anthropic_live_tail_renders_tool_results_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.think_mode = DS4_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_live");
    tc.name = xstrdup("Bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>/tmp</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_live");
    chat_msgs_push(&msgs, user);

    /* Anthropic system text is parsed separately and appended to chat_msgs for
     * rendering.  The live-tail finder must ignore it when locating the final
     * tool_result run. */
    chat_msg system = {0};
    system.role = xstrdup("system");
    system.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, system);

    anthropic_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.anthropic_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.anthropic_live_call_ids.v[0], "toolu_live"));
    TEST_ASSERT(r.anthropic_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.anthropic_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "Bash") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}

static void test_anthropic_tool_result_id_validation(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>out</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_missing");
    chat_msgs_push(&msgs, user);

    char err[160] = {0};
    TEST_ASSERT(!anthropic_validate_tool_results(&s, &msgs, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Anthropic continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    s.anthropic_live.valid = true;
    s.anthropic_live.live_tokens = 10;
    id_list_push_unique(&s.anthropic_live.call_ids, "toolu_missing");
    pthread_mutex_unlock(&s.tool_mu);
    bool needs_live_tool_state = false;
    err[0] = '\0';
    TEST_ASSERT(anthropic_validate_tool_results(&s, &msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.anthropic_live);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_anthropic_full_replay_allows_unknown_live_id(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_replay");
    tc.name = xstrdup("Bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>/tmp</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_replay");
    chat_msgs_push(&msgs, user);

    bool needs_live_tool_state = false;
    char err[160] = {0};
    TEST_ASSERT(anthropic_validate_tool_results(&s, &msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);

    chat_msgs_free(&msgs);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_anthropic_tool_use_parses_before_role(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    /* GitHub #127 regression: Crush can replay full Anthropic history with
     * message objects serialized as {"content": ..., "role": ...}.  The parser
     * must still remember prior assistant tool_use ids, otherwise old
     * tool_result blocks are mistaken for live-only continuations and rejected
     * once the live frontier has moved on to newer tool calls. */
    pthread_mutex_lock(&s.tool_mu);
    s.anthropic_live.valid = true;
    s.anthropic_live.live_tokens = 100;
    id_list_push_unique(&s.anthropic_live.call_ids, "toolu_current");
    pthread_mutex_unlock(&s.tool_mu);

    const char *json =
        "["
        "{\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"toolu_old\",\"name\":\"Bash\","
        "\"input\":{\"command\":\"ls\"}}"
        "],\"role\":\"assistant\"},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_old\",\"content\":\"ok\"}"
        "]},"
        "{\"role\":\"user\",\"content\":\"continue\"}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 3);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(msgs.v[0].calls.v[0].id &&
                !strcmp(msgs.v[0].calls.v[0].id, "toolu_old"));

    bool needs_live_tool_state = false;
    char err[160] = {0};
    TEST_ASSERT(anthropic_validate_tool_results(&s, &msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.anthropic_live);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_tool_checkpoint_canonicalization_gate_exact_replay(void) {
    server s;
    memset(&s, 0, sizeof(s));

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_exact");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{}");
    tool_calls_push(&calls, tc);
    calls.raw_dsml = xstrdup(
        "\n\n" DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "</｜DSML｜invoke>\n"
        DS4_TOOL_CALLS_END);

    TEST_ASSERT(!should_canonicalize_tool_checkpoint(&s, &calls));

    s.disable_exact_dsml_tool_replay = true;
    TEST_ASSERT(should_canonicalize_tool_checkpoint(&s, &calls));

    s.disable_exact_dsml_tool_replay = false;
    free(calls.raw_dsml);
    calls.raw_dsml = NULL;
    TEST_ASSERT(should_canonicalize_tool_checkpoint(&s, &calls));

    tool_calls_free(&calls);
}

static void test_responses_live_tail_renders_tool_outputs_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = DS4_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_live");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    responses_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.responses_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.responses_live_call_ids.v[0], "call_live"));
    TEST_ASSERT(r.responses_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.responses_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "exec_command") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}

static void test_responses_tool_output_id_validation(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_missing");
    tool.content = xstrdup("out");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    TEST_ASSERT(!responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH, NULL, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Responses continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    s.responses_live.valid = true;
    s.responses_live.live_tokens = 10;
    id_list_push_unique(&s.responses_live.call_ids, "call_missing");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    bool needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state, NULL,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_responses_stateless_tool_replay_requires_reasoning(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_replay");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_replay");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    bool needs_live_reasoning = false;
    bool needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    pthread_mutex_lock(&s.tool_mu);
    s.responses_live.valid = true;
    s.responses_live.live_tokens = 123;
    id_list_push_unique(&s.responses_live.call_ids, "call_replay");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = xstrdup("replayed hidden reasoning");
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = NULL;
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_NONE,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_responses_visible_suffix_matches_client_replay(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = DS4_THINK_HIGH;
    r.reasoning_summary_emit = true;

    char *suffix = build_responses_visible_assistant_suffix(&r, "5",
                                                            "hidden summary",
                                                            NULL);
    TEST_ASSERT(strstr(suffix, "hidden summary") == NULL);
    TEST_ASSERT(strstr(suffix, "</think>5") != NULL);
    free(suffix);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&calls, tc);

    suffix = build_responses_visible_assistant_suffix(&r, "",
                                                      "tool summary",
                                                      &calls);
    TEST_ASSERT(strstr(suffix, "tool summary</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜DSML｜tool_calls>") != NULL);
    free(suffix);

    tool_calls_free(&calls);
    request_free(&r);
}

static void test_exact_dsml_tool_replay_can_be_disabled(void) {
    const char *dsml =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    tool_memory_put(&s, "call_disabled", dsml);
    s.disable_exact_dsml_tool_replay = true;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disabled");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"canonical\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t bytes = 123;
    TEST_ASSERT(kv_tool_map_write(&s, fp, dsml, &bytes));
    TEST_ASSERT(bytes == 0);

    if (fp) fclose(fp);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_dsml_decode_state_separates_structure_and_payload(void) {
    dsml_decode_tracker tracker;
    dsml_decode_tracker_init(&tracker);

    const char *prefix =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n";
    TEST_ASSERT(dsml_decode_state_for_text(prefix, strlen(prefix)) ==
                DSML_DECODE_STRUCTURAL);
    dsml_decode_tracker_update(&tracker, prefix, strlen(prefix));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRUCTURAL);

    const char *path_param =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">/tmp/a.py";
    TEST_ASSERT(dsml_decode_state_for_text(path_param, strlen(path_param)) ==
                DSML_DECODE_STRING_BODY);
    dsml_decode_tracker_update(&tracker, path_param, strlen(path_param));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRING_BODY);

    const char *path_closing =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">/tmp/a.py</";
    TEST_ASSERT(dsml_decode_state_for_text(path_closing, strlen(path_closing)) ==
                DSML_DECODE_STRUCTURAL);
    dsml_decode_tracker_update(&tracker, path_closing, strlen(path_closing));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRUCTURAL);

    const char *json_struct =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"edits\" string=\"false\">[{";
    TEST_ASSERT(dsml_decode_state_for_text(json_struct, strlen(json_struct)) ==
                DSML_DECODE_JSON_STRUCTURAL);
    dsml_decode_tracker_init(&tracker);
    dsml_decode_tracker_update(&tracker, json_struct, strlen(json_struct));
    TEST_ASSERT(tracker.decode == DSML_DECODE_JSON_STRUCTURAL);

    const char *json_string =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"edits\" string=\"false\">[{\"newText\":\"for i in";
    TEST_ASSERT(dsml_decode_state_for_text(json_string, strlen(json_string)) ==
                DSML_DECODE_JSON_STRING);
    dsml_decode_tracker_update(&tracker, json_string, strlen(json_string));
    TEST_ASSERT(tracker.decode == DSML_DECODE_JSON_STRING);

    const char *done =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"edits\" string=\"false\">[]"
        DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;
    TEST_ASSERT(dsml_decode_state_for_text(done, strlen(done)) ==
                DSML_DECODE_OUTSIDE);
    dsml_decode_tracker_init(&tracker);
    dsml_decode_tracker_update(&tracker, done, strlen(done));
    TEST_ASSERT(tracker.decode == DSML_DECODE_OUTSIDE);
}

static void test_tool_memory_max_ids_prunes_oldest(void) {
    const char *a_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">a</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const char *b_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">b</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const char *c_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">c</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    s.tool_mem.max_entries = 2;
    tool_memory_put(&s, "call_a", a_dsml);
    tool_memory_put(&s, "call_b", b_dsml);
    tool_memory_put(&s, "call_c", c_dsml);

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call tc = {.id = xstrdup("call_a"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&a.calls, tc);
    chat_msgs_push(&msgs, a);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_tool_separator_whitespace_is_not_content(void) {
    const char *generated =
        "<think>need a tool</think>"
        "I will inspect the files.\n\n\n\n"
        DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && !strcmp(content, "I will inspect the files."));
    TEST_ASSERT(calls.len == 1);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_dsml_prompt_escapes_tool_supplied_text(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo 2>&1 && echo </｜DSML｜tool_calls>\",\"count\":1}");
    tool_calls_push(&calls, tc);

    buf b = {0};
    append_dsml_tool_calls_text(&b, &calls);
    TEST_ASSERT(strstr(b.ptr, "echo 2>&1 && echo </｜DSML｜tool_calls>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "2&gt;&amp;1") == NULL);
    TEST_ASSERT(strstr(b.ptr, "&amp;&amp;") == NULL);
    buf_free(&b);
    tool_calls_free(&calls);

    memset(&calls, 0, sizeof(calls));
    memset(&tc, 0, sizeof(tc));
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo </｜DSML｜parameter>\",\"count\":1}");
    tool_calls_push(&calls, tc);

    append_dsml_tool_calls_text(&b, &calls);
    TEST_ASSERT(strstr(b.ptr, "echo &lt;/｜DSML｜parameter>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "echo </｜DSML｜parameter>") == NULL);
    buf_free(&b);
    tool_calls_free(&calls);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("console.log('<<< < > >>>');\n</tool_result>\n<｜DSML｜tool_calls>not a real tool call");
    chat_msgs_push(&msgs, tool);
    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "console.log('<<< < > >>>');") != NULL);
    TEST_ASSERT(strstr(prompt, "console.log('&lt;") == NULL);
    TEST_ASSERT(strstr(prompt, "&lt;/tool_result>\n<｜DSML｜tool_calls>not a real tool call") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>console.log('<<< < > >>>');\n</tool_result>\n") == NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_stop_list_parses_all_sequences(void) {
    stop_list stops = {0};
    const char *json = "[\"END\",\"STOP\"]";
    TEST_ASSERT(parse_stop(&json, &stops));
    TEST_ASSERT(stops.len == 2);
    TEST_ASSERT(stops.max_len == 4);

    size_t pos = 0, len = 0;
    TEST_ASSERT(stop_list_find_from(&stops, "hello STOP tail END", 0, &pos, &len));
    TEST_ASSERT(pos == strlen("hello "));
    TEST_ASSERT(len == strlen("STOP"));
    TEST_ASSERT(stop_list_stream_safe_len(&stops, strlen("abcdef")) == 3);
    stop_list_clear(&stops);
    free(stops.v);
}

static void test_stop_list_streaming_holds_and_trims_stop_text(void) {
    stop_list stops = {0};
    const char *json = "[\"</END>\",\"STOP\"]";
    TEST_ASSERT(parse_stop(&json, &stops));

    size_t safe = stop_list_stream_safe_len(&stops, strlen("hello </"));
    TEST_ASSERT(safe == strlen("hel"));

    size_t pos = 0, len = 0;
    TEST_ASSERT(stop_list_find_from(&stops, "answer STOP hidden", 0, &pos, &len));
    TEST_ASSERT(pos == strlen("answer "));
    TEST_ASSERT(len == strlen("STOP"));

    stop_list_clear(&stops);
    free(stops.v);
}

static char *test_nested_json_array(int depth) {
    buf b = {0};
    for (int i = 0; i < depth; i++) buf_putc(&b, '[');
    buf_putc(&b, '0');
    for (int i = 0; i < depth; i++) buf_putc(&b, ']');
    return buf_take(&b);
}

static void test_json_skip_has_nesting_limit(void) {
    char *ok = test_nested_json_array(JSON_MAX_NESTING);
    const char *p = ok;
    TEST_ASSERT(json_skip_value(&p));
    TEST_ASSERT(*p == '\0');
    free(ok);

    char *bad = test_nested_json_array(JSON_MAX_NESTING + 1);
    p = bad;
    TEST_ASSERT(!json_skip_value(&p));
    free(bad);
}

static void test_model_metadata_clamps_completion_to_context(void) {
    buf b = {0};
    append_model_json_values(&b, 32768, 393216);
    TEST_ASSERT(strstr(b.ptr, "\"context_length\":32768") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"max_completion_tokens\":32768") != NULL);
    buf_free(&b);

    append_model_json_values(&b, 100000, 4096);
    TEST_ASSERT(strstr(b.ptr, "\"context_length\":100000") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"max_completion_tokens\":4096") != NULL);
    buf_free(&b);
}

static void test_client_socket_nonblocking_flag(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    set_client_socket_nonblocking(sv[0]);
    int flags = fcntl(sv[0], F_GETFL, 0);
    TEST_ASSERT(flags >= 0);
    TEST_ASSERT((flags & O_NONBLOCK) != 0);
    close(sv[0]);
    close(sv[1]);
}

static void test_thinking_state_tracks_prompt_and_generated_tags(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.prompt_text = xstrdup("<｜Assistant｜><think>");
    thinking_state st = thinking_state_from_prompt(&r);
    TEST_ASSERT(st.inside == true);
    thinking_state_feed(&st, "reasoning body", strlen("reasoning body"));
    TEST_ASSERT(st.inside == true);
    thinking_state_feed(&st, "</thi", strlen("</thi"));
    TEST_ASSERT(st.inside == true);
    thinking_state_feed(&st, "nk>answer", strlen("nk>answer"));
    TEST_ASSERT(st.inside == false);
    thinking_state_feed(&st, "<thi", strlen("<thi"));
    TEST_ASSERT(st.inside == false);
    thinking_state_feed(&st, "nk>more", strlen("nk>more"));
    TEST_ASSERT(st.inside == true);
    request_free(&r);

    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_NONE;
    r.prompt_text = xstrdup("<｜Assistant｜></think>");
    st = thinking_state_from_prompt(&r);
    TEST_ASSERT(st.inside == false);
    request_free(&r);
}

static void test_thinking_checkpoint_remember_gate(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    thinking_state st = {.inside = true};

    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "length"));
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));

    st.inside = false;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "length"));
    TEST_ASSERT(should_remember_thinking_checkpoint(&r, &st, "stop"));

    r.prompt_preserves_reasoning = true;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));
    r.prompt_preserves_reasoning = false;
    r.has_tools = true;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));
    r.has_tools = false;
    r.think_mode = DS4_THINK_NONE;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));

    request_free(&r);
}

static void test_tool_marker_state_ignores_orphan_end(void) {
    bool saw_start = false;
    bool saw_end = false;
    bool orphan_end = false;

    observe_tool_markers("reasoning\n" DS4_PARAM_END "\n" DS4_INVOKE_END "\n" DS4_TOOL_CALLS_END,
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(!saw_start);
    TEST_ASSERT(!saw_end);
    TEST_ASSERT(orphan_end);

    orphan_end = false;
    observe_tool_markers(DS4_TOOL_CALLS_START "\n" DS4_INVOKE_START " name=\"bash\">",
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(saw_start);
    TEST_ASSERT(!saw_end);
    TEST_ASSERT(!orphan_end);

    observe_tool_markers(DS4_INVOKE_END "\n" DS4_TOOL_CALLS_END,
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(saw_start);
    TEST_ASSERT(saw_end);
}

static void test_canonical_rewrite_rebuilds_when_live_tail_changes(void) {
    /* Regression for the first canonical-KV rewrite attempt: replacing a small
     * live suffix looks tempting because the raw SWA ring may still contain the
     * needed rows, but compressed KV counters and compressor/indexer frontiers
     * are already past the shared prefix.  Until those graph frontiers can be
     * restored exactly, every rewrite behind the live end must rebuild or load a
     * disk checkpoint. */
    TEST_ASSERT(ds4_session_rewrite_requires_rebuild(19296, 19290, 19081));
    TEST_ASSERT(ds4_session_rewrite_requires_rebuild(1024, 1030, 1000));
    TEST_ASSERT(ds4_session_rewrite_requires_rebuild(1024, 900, 900));

    TEST_ASSERT(!ds4_session_rewrite_requires_rebuild(1024, 1024, 1024));
    TEST_ASSERT(!ds4_session_rewrite_requires_rebuild(1024, 1100, 1024));
}

static void test_kv_cache_store_len_uses_configured_boundary(void) {
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    TEST_ASSERT(kv_cache_store_len(&kc, 11011) == 10240);
    TEST_ASSERT(kv_cache_store_len(&kc, 1695) == 1695);

    kc.opt.boundary_trim_tokens = 0;
    kc.opt.boundary_align_tokens = 1000;
    TEST_ASSERT(kv_cache_store_len(&kc, 3500) == 3000);

    kc.opt.boundary_align_tokens = 0;
    TEST_ASSERT(kv_cache_store_len(&kc, 3500) == 3500);
}

static void test_kv_cache_chat_anchor_uses_last_user_before_assistant(void) {
    const int user = 9001;
    const int assistant = 9002;
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    kc.opt.min_tokens = 4;

    ds4_tokens codex = {0};
    ds4_tokens_push(&codex, 1);     /* BOS / system */
    ds4_tokens_push(&codex, 2);
    ds4_tokens_push(&codex, user);  /* environment_context item */
    ds4_tokens_push(&codex, 3);
    ds4_tokens_push(&codex, 4);
    ds4_tokens_push(&codex, user);  /* actual task starts here */
    ds4_tokens_push(&codex, 5);
    ds4_tokens_push(&codex, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &codex, user, assistant) == 5);

    ds4_tokens claude = {0};
    ds4_tokens_push(&claude, 1);
    ds4_tokens_push(&claude, 2);
    ds4_tokens_push(&claude, 3);
    ds4_tokens_push(&claude, 4);
    ds4_tokens_push(&claude, user); /* system reminder and task share a turn */
    ds4_tokens_push(&claude, 5);
    ds4_tokens_push(&claude, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &claude, user, assistant) == 4);

    ds4_tokens_free(&codex);
    ds4_tokens_free(&claude);
}

static void test_kv_cache_chat_anchor_ignores_multiturn_tail(void) {
    const int user = 9001;
    const int assistant = 9002;
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    kc.opt.min_tokens = 2;

    ds4_tokens prompt = {0};
    ds4_tokens_push(&prompt, 1);
    ds4_tokens_push(&prompt, 2);
    ds4_tokens_push(&prompt, user);      /* first task */
    ds4_tokens_push(&prompt, 3);
    ds4_tokens_push(&prompt, assistant); /* stop scanning here */
    ds4_tokens_push(&prompt, 4);
    ds4_tokens_push(&prompt, user);      /* later turn: not a cold anchor */
    ds4_tokens_push(&prompt, 5);
    ds4_tokens_push(&prompt, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, assistant) == 2);

    kc.opt.min_tokens = 3;
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, assistant) == -1);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, -1, assistant) == -1);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, -1) == -1);

    ds4_tokens_free(&prompt);
}

static void test_kv_cache_continued_uses_aligned_frontiers(void) {
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.opt = kv_cache_default_options();

    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10239) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);

    kc.continued_last_store_tokens = 4096;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);

    kc.continued_last_store_tokens = 24576;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30720) == 30720);

    kc.continued_last_store_tokens = 10240;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 18432) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 20480) == 20480);

    kc.opt.boundary_align_tokens = 0;
    kc.continued_last_store_tokens = 20480;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 29999) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30000) == 30000);
}

static void test_kv_cache_cold_store_suppresses_duplicate_continued_boundary(void) {
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.opt = kv_cache_default_options();

    int old = kv_cache_suppress_continued_store(&kc, 10240);
    TEST_ASSERT(old == 0);
    TEST_ASSERT(kc.continued_last_store_tokens == 10240);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 0);

    kv_cache_restore_suppressed_continued(&kc, old, 10240);
    TEST_ASSERT(kc.continued_last_store_tokens == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);
}

static void test_kv_cache_file_size_must_fit_budget(void) {
    kv_disk_cache kc = {0};
    kc.budget_bytes = 1100;

    TEST_ASSERT(kv_cache_file_size_fits(&kc, 100, 930, 0, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, 100, 938, 0, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, 100, 900, 40, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, UINT64_MAX, 1, 0, NULL, NULL));

    kc.budget_bytes = 0;
    TEST_ASSERT(kv_cache_file_size_fits(&kc, 100, 900, 40, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, UINT64_MAX, 1, 0, NULL, NULL));
}

static void test_sha1_bytes_hex_matches_known_vector(void) {
    char sha[41];
    sha1_bytes_hex("abc", 3, sha);
    TEST_ASSERT(!strcmp(sha, "a9993e364706816aba3e25717850c26c9cd0d89d"));
}

static void test_kv_stub_file(const char *dir, const char *sha,
                              uint8_t reason, uint32_t tokens, uint32_t hits,
                              uint64_t last_used, uint64_t payload_bytes) {
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) {
        free(path);
        return;
    }

    uint8_t h[KV_CACHE_FIXED_HEADER];
    kv_fill_header(h, 2, reason, 0, tokens, hits, 32768, 100, last_used, payload_bytes);
    uint8_t text_len[4] = {0};
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    for (uint64_t i = 0; i < payload_bytes; i++) {
        TEST_ASSERT(fputc(0, fp) != EOF);
    }
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}

static void test_kv_text_stub_file(const char *dir, const char *text,
                                   uint32_t tokens, uint64_t payload_bytes) {
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) {
        free(path);
        return;
    }

    uint8_t h[KV_CACHE_FIXED_HEADER];
    kv_fill_header(h, 2, KV_REASON_COLD, 0, tokens, 0, 32768, 100, 100, payload_bytes);
    uint8_t text_len[4];
    le_put32(text_len, (uint32_t)strlen(text));
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
    for (uint64_t i = 0; i < payload_bytes; i++) {
        TEST_ASSERT(fputc(0, fp) != EOF);
    }
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}

static void test_kv_cache_lookup_uses_longest_text_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-text-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *short_text = "transcript prefix";
    const char *long_text = "transcript prefix with sampled token bytes";
    test_kv_text_stub_file(dir, short_text, 512, 0);
    test_kv_text_stub_file(dir, long_text, 768, 0);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    int idx = kv_cache_find_text_prefix(&kc,
        "transcript prefix with sampled token bytes and suffix",
        2, 32768);
    TEST_ASSERT(idx >= 0);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].tokens == 768);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].text_bytes == strlen(long_text));
    TEST_ASSERT(kv_cache_find_text_prefix(&kc, "transcript prefiX", 2, 32768) < 0);

    kv_cache_close(&kc);
    char short_sha[41], long_sha[41];
    sha1_bytes_hex(short_text, strlen(short_text), short_sha);
    sha1_bytes_hex(long_text, strlen(long_text), long_sha);
    char short_name[44], long_name[44];
    snprintf(short_name, sizeof(short_name), "%.40s.kv", short_sha);
    snprintf(long_name, sizeof(long_name), "%.40s.kv", long_sha);
    char *short_path = path_join(dir, short_name);
    char *long_path = path_join(dir, long_name);
    unlink(short_path);
    unlink(long_path);
    free(short_path);
    free(long_path);
    rmdir(dir);
}

static void test_kv_tool_map_filters_by_dsml_text(void) {
    const char *dsml_keep =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    const char *dsml_drop =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">zzzz</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    server src = {0}, dst = {0};
    pthread_mutex_init(&src.tool_mu, NULL);
    pthread_mutex_init(&dst.tool_mu, NULL);
    tool_memory_put(&src, "call_keep", dsml_keep);
    tool_memory_put(&src, "call_drop", dsml_drop);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t estimated_bytes = 0;
    TEST_ASSERT(kv_tool_map_serialized_size(&src, dsml_keep, &estimated_bytes));
    uint64_t bytes = 0;
    TEST_ASSERT(kv_tool_map_write(&src, fp, dsml_keep, &bytes));
    TEST_ASSERT(bytes > 0);
    TEST_ASSERT(estimated_bytes == bytes);
    rewind(fp);
    TEST_ASSERT(kv_tool_map_load_from_pos(&dst, fp, NULL) == 1);

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call keep = {.id = xstrdup("call_keep"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&a.calls, keep);
    chat_msgs_push(&msgs, a);
    chat_msg b = {0};
    b.role = xstrdup("assistant");
    tool_call drop = {.id = xstrdup("call_drop"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&b.calls, drop);
    chat_msgs_push(&msgs, b);
    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&dst, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(msgs.v[1].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.disk == 1);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);
    TEST_ASSERT(strstr(msgs.v[0].calls.raw_dsml, "pwd") != NULL);
    TEST_ASSERT(strstr(msgs.v[0].calls.raw_dsml, "zzzz") == NULL);

    chat_msgs_free(&msgs);
    if (fp) fclose(fp);
    tool_memory_free(&src.tool_mem);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&src.tool_mu);
    pthread_mutex_destroy(&dst.tool_mu);
}

static void test_kv_tool_map_restores_before_prompt_render(void) {
    char tmpl[] = "/tmp/ds4-kv-tool-map-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *sha = "3333333333333333333333333333333333333333";
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    const char *dsml =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">echo exact</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    const char *text = dsml;

    server src = {0};
    pthread_mutex_init(&src.tool_mu, NULL);
    tool_memory_put(&src, "call_disk", dsml);

    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (fp) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        kv_fill_header(h, 2, KV_REASON_CONTINUED, KV_EXT_TOOL_MAP, 512, 0, 32768, 100, 100, 0);
        uint8_t text_len[4];
        le_put32(text_len, (uint32_t)strlen(text));
        TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
        TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
        TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
        uint64_t ignored = 0;
        TEST_ASSERT(kv_tool_map_write(&src, fp, dsml, &ignored));
        TEST_ASSERT(fclose(fp) == 0);
    }

    server dst = {0};
    pthread_mutex_init(&dst.tool_mu, NULL);
    dst.kv.enabled = true;
    dst.kv.dir = xstrdup(dir);
    dst.kv.opt = kv_cache_default_options();

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disk");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo canonical\"}");
    tool_calls_push(&a.calls, tc);
    chat_msgs_push(&msgs, a);

    kv_cache_restore_tool_memory_for_messages(&dst, &msgs);
    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&dst, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.disk == 1);
    TEST_ASSERT(stats.canonical == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(strstr(prompt, "echo exact") != NULL);
    TEST_ASSERT(strstr(prompt, "echo canonical") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
    kv_cache_close(&dst.kv);
    tool_memory_free(&src.tool_mem);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&src.tool_mu);
    pthread_mutex_destroy(&dst.tool_mu);
    unlink(path);
    free(path);
    rmdir(dir);
}

static void test_kv_cache_eviction_values_fresh_snapshots(void) {
    char tmpl[] = "/tmp/ds4-kv-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_UNKNOWN, 512, 0, now, 4096);
    test_kv_stub_file(dir, new_sha, KV_REASON_UNKNOWN, 2048, 0, now, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}

static void test_kv_cache_eviction_protects_current_store(void) {
    char tmpl[] = "/tmp/ds4-kv-current-store-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 4096, 0, now, 2048);
    test_kv_stub_file(dir, new_sha, KV_REASON_CONTINUED, 2048, 0, now, 4096);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 4096u) + 16u;
    kv_cache_evict(&kc, NULL, new_sha);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}

static void test_kv_cache_eviction_does_not_protect_oversize_current_store(void) {
    char tmpl[] = "/tmp/ds4-kv-oversize-store-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 4096, 0, now, 1024);
    test_kv_stub_file(dir, new_sha, KV_REASON_CONTINUED, 4096, 0, now, 4096);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 1024u) + 16u;
    kv_cache_evict(&kc, NULL, new_sha);

    TEST_ASSERT(access(old_path, F_OK) == 0);
    TEST_ASSERT(access(new_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}

static void test_kv_cache_eviction_score_decays_stale_hits(void) {
    /* stale: lower tokens-per-byte (e.g. tool-heavy prompt) but boosted by
     * 10 hits well in the past.  fresh: higher tokens-per-byte and zero hits,
     * just stored.  The stale hit bonus decays by inactivity, so fresh wins on
     * its better baseline even though stale once had more successful hits. */
    const uint64_t now = 1000u + 14u * KV_CACHE_HIT_HALF_LIFE_SECONDS;
    kv_entry stale = {.tokens = 1024, .hits = 10, .file_size = 4096, .last_used = 1000};
    kv_entry fresh = {.tokens = 2048, .hits = 0,  .file_size = 4096, .last_used = now};

    double s_on = kv_entry_eviction_score(&stale, NULL, NULL, now);
    double f_on = kv_entry_eviction_score(&fresh, NULL, NULL, now);
    TEST_ASSERT(s_on < f_on);

    /* A fresh entry's score never decays below its (0+1) * tokens/size floor,
     * regardless of how old another entry's hit history is. */
    TEST_ASSERT(f_on == 1.0 * (double)fresh.tokens / (double)fresh.file_size);
}

static void test_kv_cache_eviction_decayed_hits_tie_break_by_age(void) {
    char tmpl[] = "/tmp/ds4-kv-stale-hit-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    uint64_t stale = now > KV_CACHE_HIT_HALF_LIFE_SECONDS * 14ull
        ? now - KV_CACHE_HIT_HALF_LIFE_SECONDS * 14ull
        : 1;
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 2048, 15, stale, 2048);
    test_kv_stub_file(dir, new_sha, KV_REASON_COLD, 2048, 0, now, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}

static void test_kv_cache_eviction_keeps_aligned_continued_frontiers(void) {
    char tmpl[] = "/tmp/ds4-kv-live-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *cold_sha = "1111111111111111111111111111111111111111";
    const char *continued_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, cold_sha, KV_REASON_COLD, 512, 0, now, 2048);
    test_kv_stub_file(dir, continued_sha, KV_REASON_CONTINUED, 2048, 0, now, 2048);

    char cold_name[44], continued_name[44];
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    char *cold_path = path_join(dir, cold_name);
    char *continued_path = path_join(dir, continued_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, NULL);

    TEST_ASSERT(access(cold_path, F_OK) != 0);
    TEST_ASSERT(access(continued_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(cold_path);
    unlink(continued_path);
    free(cold_path);
    free(continued_path);
    rmdir(dir);
}

static void test_thinking_checkpoint_canonical_matches_future_prompt(void) {
    /* Simulate: user sends a single message, thinking mode on, no tools.
     * Model generates reasoning + content.  The next request will drop the
     * reasoning from this turn.  Verify that:
     *   prompt_text[:-len("<think>")] + "</think>" + content + "<|eos|>"
     * equals what render_chat_prompt_text produces for the history. */

    chat_msgs prefix_msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&prefix_msgs, user1);

    /* This is what prompt_text looks like for the first generation */
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, NULL, NULL, DS4_THINK_HIGH);
    /* prompt_text should end with <think> */
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(pt_len >= 7);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* The model generates: reasoning + </think> + content */
    const char *reasoning = "Let me think... 2+2 = 4";
    const char *content = "The answer is 4.";

    /* Build the canonical checkpoint text (what we'd produce after canonicalization) */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);  /* strip <think> */
    buf_puts(&canonical, "</think>");
    buf_puts(&canonical, content);
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.prompt_text = xstrdup(prompt_text);
    char *visible = build_toolless_thinking_visible_text(&r, content);
    TEST_ASSERT(visible != NULL);
    TEST_ASSERT(!strcmp(visible, canonical.ptr));
    free(visible);
    request_free(&r);

    /* Now build what the NEXT request would render: history includes this
     * assistant message, plus a new user message.  Extract just the prefix
     * up to and including the eos of the assistant turn. */
    chat_msgs history_msgs = {0};
    chat_msg h_user1 = {0};
    h_user1.role = xstrdup("user");
    h_user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&history_msgs, h_user1);
    chat_msg h_asst = {0};
    h_asst.role = xstrdup("assistant");
    h_asst.reasoning = xstrdup(reasoning);
    h_asst.content = xstrdup(content);
    chat_msgs_push(&history_msgs, h_asst);
    chat_msg h_user2 = {0};
    h_user2.role = xstrdup("user");
    h_user2.content = xstrdup("Thanks!");
    chat_msgs_push(&history_msgs, h_user2);

    char *future_prompt = render_chat_prompt_text(&history_msgs, NULL, NULL, DS4_THINK_HIGH);

    /* The future prompt should START with our canonical text */
    size_t clen = canonical.len;
    TEST_ASSERT(strlen(future_prompt) > clen);
    TEST_ASSERT(!memcmp(future_prompt, canonical.ptr, clen));

    /* And what comes after is the new user turn + assistant prefix */
    const char *rest = future_prompt + clen;
    TEST_ASSERT(strstr(rest, "Thanks!") != NULL);
    TEST_ASSERT(strstr(rest, "<think>") != NULL);  /* new turn starts thinking */

    /* Verify reasoning is NOT in the future prompt for this turn */
    const char *asst_turn = strstr(future_prompt, "<" "\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c" ">");
    TEST_ASSERT(asst_turn != NULL);
    TEST_ASSERT(strstr(future_prompt, reasoning) == NULL);  /* reasoning dropped */

    free(future_prompt);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&prefix_msgs);
    chat_msgs_free(&history_msgs);
}

static void test_thinking_canonical_empty_content(void) {
    /* Edge case: model thinks but produces empty content (e.g. tool-less
     * thinking where answer is entirely in reasoning).  Canonical should
     * still be valid: prompt_text[:-7] + "</think><|eos|>" */
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Think about life");
    chat_msgs_push(&msgs, user);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);

    /* Build canonical with empty content */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);
    buf_puts(&canonical, "</think>");
    /* empty content */
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    /* Future prompt with empty content assistant message */
    chat_msgs history = {0};
    chat_msg h_u = {0};
    h_u.role = xstrdup("user");
    h_u.content = xstrdup("Think about life");
    chat_msgs_push(&history, h_u);
    chat_msg h_a = {0};
    h_a.role = xstrdup("assistant");
    h_a.reasoning = xstrdup("Deep thoughts about existence...");
    h_a.content = xstrdup("");
    chat_msgs_push(&history, h_a);
    chat_msg h_u2 = {0};
    h_u2.role = xstrdup("user");
    h_u2.content = xstrdup("Continue");
    chat_msgs_push(&history, h_u2);

    char *future = render_chat_prompt_text(&history, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));
    /* reasoning dropped */
    TEST_ASSERT(strstr(future, "Deep thoughts") == NULL);

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}

static void test_thinking_canonical_multi_turn(void) {
    /* Multi-turn: 3 user messages, 2 assistant responses with reasoning.
     * Both prior assistant turns should have reasoning dropped.
     * The canonical after the SECOND generation should produce text that
     * matches the start of a 3rd-turn future prompt. */
    chat_msgs turn2_prefix = {0};
    chat_msg u1 = {0};
    u1.role = xstrdup("user");
    u1.content = xstrdup("Hello");
    chat_msgs_push(&turn2_prefix, u1);
    chat_msg a1 = {0};
    a1.role = xstrdup("assistant");
    a1.reasoning = xstrdup("first reasoning");
    a1.content = xstrdup("Hi there");
    chat_msgs_push(&turn2_prefix, a1);
    chat_msg u2 = {0};
    u2.role = xstrdup("user");
    u2.content = xstrdup("How are you?");
    chat_msgs_push(&turn2_prefix, u2);

    /* prompt_text for the 2nd generation (includes 1st assistant turn) */
    char *prompt_text = render_chat_prompt_text(&turn2_prefix, NULL, NULL, DS4_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* 1st turn reasoning is already dropped in this prompt_text */
    TEST_ASSERT(strstr(prompt_text, "first reasoning") == NULL);
    TEST_ASSERT(strstr(prompt_text, "Hi there") != NULL);

    /* After 2nd generation: canonical drops 2nd reasoning too */
    const char *content2 = "I'm doing well";
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);
    buf_puts(&canonical, "</think>");
    buf_puts(&canonical, content2);
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    /* Future: 3rd user message arrives */
    chat_msgs future_msgs = {0};
    chat_msg fu1 = {0}; fu1.role = xstrdup("user"); fu1.content = xstrdup("Hello");
    chat_msgs_push(&future_msgs, fu1);
    chat_msg fa1 = {0}; fa1.role = xstrdup("assistant");
    fa1.reasoning = xstrdup("first reasoning");
    fa1.content = xstrdup("Hi there");
    chat_msgs_push(&future_msgs, fa1);
    chat_msg fu2 = {0}; fu2.role = xstrdup("user"); fu2.content = xstrdup("How are you?");
    chat_msgs_push(&future_msgs, fu2);
    chat_msg fa2 = {0}; fa2.role = xstrdup("assistant");
    fa2.reasoning = xstrdup("second reasoning");
    fa2.content = xstrdup(content2);
    chat_msgs_push(&future_msgs, fa2);
    chat_msg fu3 = {0}; fu3.role = xstrdup("user"); fu3.content = xstrdup("Great");
    chat_msgs_push(&future_msgs, fu3);

    char *future = render_chat_prompt_text(&future_msgs, NULL, NULL, DS4_THINK_HIGH);
    /* Both reasonings dropped */
    TEST_ASSERT(strstr(future, "first reasoning") == NULL);
    TEST_ASSERT(strstr(future, "second reasoning") == NULL);
    /* Canonical is a prefix of future */
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&turn2_prefix);
    chat_msgs_free(&future_msgs);
}

static void test_thinking_canonical_with_tools_preserves_reasoning(void) {
    /* When tools ARE present, reasoning is preserved in re-render.
     * The toolless thinking live binding should NOT fire (has_tools gate),
     * and the tool-call replay path handles it.  Verify the template
     * preserves reasoning when tool_context is true. */
    const char *tool_schemas = "{\"name\":\"bash\"}";

    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("run ls");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, tool_schemas, NULL, DS4_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* With tools, next render KEEPS reasoning */
    chat_msgs history = {0};
    chat_msg hu = {0}; hu.role = xstrdup("user"); hu.content = xstrdup("run ls");
    chat_msgs_push(&history, hu);
    chat_msg ha = {0}; ha.role = xstrdup("assistant");
    ha.reasoning = xstrdup("I should run bash");
    ha.content = xstrdup("Here you go");
    chat_msgs_push(&history, ha);
    chat_msg hu2 = {0}; hu2.role = xstrdup("user"); hu2.content = xstrdup("thanks");
    chat_msgs_push(&history, hu2);

    char *future = render_chat_prompt_text(&history, tool_schemas, NULL, DS4_THINK_HIGH);
    /* Reasoning IS preserved when tools present */
    TEST_ASSERT(strstr(future, "I should run bash") != NULL);
    TEST_ASSERT(strstr(future, "<think>I should run bash</think>") != NULL);

    free(future);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}

static void test_thinking_canonical_non_thinking_mode_noop(void) {
    /* When thinking is disabled (deepseek-chat), prompt_text ends with
     * </think> not <think>.  The toolless thinking live binding is a no-op
     * (early return on memcmp check). */
    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("Hello");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_NONE);
    size_t pt_len = strlen(prompt_text);
    /* Should end with </think>, not <think> */
    TEST_ASSERT(pt_len >= 8);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 8, "</think>", 8));
    /* Does NOT end with <think> */
    TEST_ASSERT(memcmp(prompt_text + pt_len - 7, "<think>", 7) != 0);

    free(prompt_text);
    chat_msgs_free(&msgs);
}

static void ds4_server_unit_tests_run(void) {
    test_request_defaults_use_min_p_filtering();
    test_reasoning_effort_mapping();
    test_api_thinking_controls_parse();
    test_render_think_max_prompt_prefix();
    test_render_non_thinking_prompt_closes_think();
    test_render_drops_old_reasoning_without_tools();
    test_render_preserves_reasoning_with_tools();
    test_render_chat_prompt_text_renders_tools_before_system();
    test_tool_schema_order_from_anthropic_schema();
    test_tool_schema_order_from_openai_tools();
    test_tool_schema_order_from_responses_tool_search();
    test_responses_function_named_tool_search_stays_function_call();
    test_responses_namespace_tool_schemas_restore_wire_namespace();
    test_responses_input_tool_search_output_loads_tools();
    test_responses_input_tool_search_output_rejects_bad_tools();
    test_responses_input_function_call_namespace_round_trips_to_dsml();
    test_responses_output_sends_tool_search_call_item();
    test_dsml_tool_args_preserve_call_order();
    test_openai_tool_args_preserve_call_order();
    test_anthropic_thinking_and_tool_args_preserve_call_order();
    test_context_length_error_uses_protocol_standard_shape();
    test_cors_headers_are_opt_in();
    test_cors_preflight_response_is_no_content();
    test_cors_sse_headers();
    test_anthropic_live_stream_sends_incremental_blocks();
    test_anthropic_usage_reports_cache_details();
    test_anthropic_tool_stream_sends_live_tool_use();
    test_openai_tool_stream_sends_incremental_text();
    test_openai_stream_usage_reports_cache_details();
    test_responses_usage_reports_cache_details();
    test_openai_chat_stream_splits_reasoning_without_tools();
    test_openai_tool_stream_sends_partial_arguments();
    test_openai_tool_stream_waits_for_incomplete_tool_tags();
    test_openai_tool_stream_sends_partial_raw_arguments();
    test_openai_tool_stream_holds_partial_dsml_entities();
    test_openai_tool_stream_holds_partial_utf8_arguments();
    test_openai_tool_stream_handles_multiple_calls();
    test_streaming_holds_partial_utf8();
    test_parse_short_dsml_and_canonical_suffix();
    test_dsml_parser_recovers_loose_nested_parameters();
    test_tool_parse_failure_returns_recoverable_finish();
    test_thinking_dsml_is_not_executable_before_think_close();
    test_thinking_dsml_after_think_close_is_executable();
    test_tool_checkpoint_suffix_is_future_prompt_canonical();
    test_tool_checkpoint_minifies_json_parameters();
    test_tool_memory_replays_sampled_dsml();
    test_anthropic_tool_memory_replays_sampled_dsml();
    test_anthropic_live_tail_renders_tool_results_only();
    test_anthropic_tool_result_id_validation();
    test_anthropic_full_replay_allows_unknown_live_id();
    test_anthropic_tool_use_parses_before_role();
    test_tool_checkpoint_canonicalization_gate_exact_replay();
    test_responses_live_tail_renders_tool_outputs_only();
    test_responses_tool_output_id_validation();
    test_responses_stateless_tool_replay_requires_reasoning();
    test_responses_visible_suffix_matches_client_replay();
    test_exact_dsml_tool_replay_can_be_disabled();
    test_dsml_decode_state_separates_structure_and_payload();
    test_tool_memory_max_ids_prunes_oldest();
    test_kv_tool_map_filters_by_dsml_text();
    test_kv_tool_map_restores_before_prompt_render();
    test_thinking_checkpoint_canonical_matches_future_prompt();
    test_thinking_canonical_empty_content();
    test_thinking_canonical_multi_turn();
    test_thinking_canonical_with_tools_preserves_reasoning();
    test_thinking_canonical_non_thinking_mode_noop();
    test_tool_separator_whitespace_is_not_content();
    test_dsml_prompt_escapes_tool_supplied_text();
    test_stop_list_parses_all_sequences();
    test_stop_list_streaming_holds_and_trims_stop_text();
    test_json_skip_has_nesting_limit();
    test_model_metadata_clamps_completion_to_context();
    test_client_socket_nonblocking_flag();
    test_thinking_state_tracks_prompt_and_generated_tags();
    test_thinking_checkpoint_remember_gate();
    test_tool_marker_state_ignores_orphan_end();
    test_canonical_rewrite_rebuilds_when_live_tail_changes();
    test_kv_cache_store_len_uses_configured_boundary();
    test_kv_cache_chat_anchor_uses_last_user_before_assistant();
    test_kv_cache_chat_anchor_ignores_multiturn_tail();
    test_kv_cache_continued_uses_aligned_frontiers();
    test_kv_cache_cold_store_suppresses_duplicate_continued_boundary();
    test_kv_cache_file_size_must_fit_budget();
    test_sha1_bytes_hex_matches_known_vector();
    test_kv_cache_lookup_uses_longest_text_prefix();
    test_kv_cache_eviction_values_fresh_snapshots();
    test_kv_cache_eviction_protects_current_store();
    test_kv_cache_eviction_does_not_protect_oversize_current_store();
    test_kv_cache_eviction_score_decays_stale_hits();
    test_kv_cache_eviction_decayed_hits_tie_break_by_age();
    test_kv_cache_eviction_keeps_aligned_continued_frontiers();
}

#ifndef DS4_SERVER_TEST_NO_MAIN
int main(void) {
    ds4_server_unit_tests_run();
    if (test_failures) {
        fprintf(stderr, "ds4-server tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4-server tests: ok");
    return 0;
}
#endif

#endif
