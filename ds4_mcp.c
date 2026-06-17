#include "ds4_mcp.h"
#include "ds4_acp.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* MCP servers are client-provided subprocesses.  A silent server should fail the
 * ACP session setup or tool call instead of leaving the protocol loop stuck. */
#define DS4_MCP_CONNECT_TIMEOUT_MS 15000
#define DS4_MCP_CALL_TIMEOUT_MS 300000

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} mcp_buf;

static void mcp_oom(const char *what) {
    fprintf(stderr, "%s\n", what);
    exit(1);
}

static void *mcp_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) mcp_oom("ds4-mcp: malloc");
    return p;
}

static void *mcp_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) mcp_oom("ds4-mcp: realloc");
    return p;
}

static char *mcp_xstrdup(const char *s) {
    size_t n = strlen(s ? s : "");
    char *p = mcp_xmalloc(n + 1);
    memcpy(p, s ? s : "", n + 1);
    return p;
}

static void mcp_buf_append(mcp_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 1024;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = mcp_xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void mcp_buf_puts(mcp_buf *b, const char *s) {
    mcp_buf_append(b, s, strlen(s));
}

static char *mcp_buf_take(mcp_buf *b) {
    if (!b->ptr) return mcp_xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void mcp_set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (!err || !err_len) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static ssize_t mcp_write_pipe(int fd, const char *s, size_t n) {
    sigset_t set, oldset, pending;
    bool blocked = false, had_sigpipe = false;

    sigemptyset(&set);
    sigaddset(&set, SIGPIPE);
    if (pthread_sigmask(SIG_BLOCK, &set, &oldset) == 0) {
        blocked = true;
        if (sigpending(&pending) == 0 &&
            sigismember(&pending, SIGPIPE) == 1)
            had_sigpipe = true;
    }

    ssize_t wr = write(fd, s, n);
    int saved_errno = errno;
    if (blocked && wr < 0 && saved_errno == EPIPE && !had_sigpipe) {
        if (sigpending(&pending) == 0 &&
            sigismember(&pending, SIGPIPE) == 1)
        {
            int sig;
            (void)sigwait(&set, &sig);
        }
    }
    if (blocked) pthread_sigmask(SIG_SETMASK, &oldset, NULL);
    errno = saved_errno;
    return wr;
}

static bool mcp_write_all(int fd, const char *s, size_t n) {
    while (n) {
        ssize_t wr = mcp_write_pipe(fd, s, n);
        if (wr > 0) {
            s += wr;
            n -= (size_t)wr;
            continue;
        }
        if (wr < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

static bool mcp_cancelled(ds4_mcp_cancel_fn cancel, void *privdata) {
    return cancel && cancel(privdata);
}

static int64_t mcp_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int mcp_timeout_default(int timeout_ms, int default_ms) {
    return timeout_ms > 0 ? timeout_ms : default_ms;
}

static bool mcp_read_line(ds4_mcp_server *s, char **line_out,
                          ds4_mcp_cancel_fn cancel, void *cancel_privdata,
                          int64_t deadline_ms, char *err, size_t err_len) {
    mcp_buf b = {0};
    for (;;) {
        if (mcp_cancelled(cancel, cancel_privdata)) {
            free(b.ptr);
            mcp_set_err(err, err_len, "interrupted");
            return false;
        }
        int poll_ms = 100;
        if (deadline_ms > 0) {
            int64_t rem = deadline_ms - mcp_now_ms();
            if (rem <= 0) {
                free(b.ptr);
                mcp_set_err(err, err_len, "MCP server %s timed out", s->name);
                return false;
            }
            if (rem < poll_ms) poll_ms = (int)rem;
        }
        struct pollfd pfd = {.fd = s->out_fd, .events = POLLIN};
        int pr = poll(&pfd, 1, poll_ms);
        if (pr < 0) {
            if (errno == EINTR) continue;
            free(b.ptr);
            mcp_set_err(err, err_len, "poll failed: %s", strerror(errno));
            return false;
        }
        if (pr == 0) continue;
        if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) {
            free(b.ptr);
            mcp_set_err(err, err_len, "MCP server %s closed stdout", s->name);
            return false;
        }
        if (!(pfd.revents & POLLIN)) continue;
        char c;
        ssize_t n = read(s->out_fd, &c, 1);
        if (n > 0) {
            if (c == '\n') {
                *line_out = mcp_buf_take(&b);
                return true;
            }
            mcp_buf_append(&b, &c, 1);
            if (b.len > 4 * 1024 * 1024) {
                free(b.ptr);
                mcp_set_err(err, err_len, "MCP server %s line too large", s->name);
                return false;
            }
            continue;
        }
        if (n == 0) {
            free(b.ptr);
            mcp_set_err(err, err_len, "MCP server %s closed stdout", s->name);
            return false;
        }
        if (errno == EINTR) continue;
        free(b.ptr);
        mcp_set_err(err, err_len, "read failed: %s", strerror(errno));
        return false;
    }
}

static void mcp_server_free(ds4_mcp_server *s) {
    if (!s) return;
    if (s->in_fd >= 0) close(s->in_fd);
    if (s->out_fd >= 0) close(s->out_fd);
    if (s->pid > 0) {
        kill(s->pid, SIGTERM);
        for (int i = 0; i < 20; i++) {
            if (waitpid(s->pid, NULL, WNOHANG) == s->pid) break;
            usleep(50000);
        }
        if (waitpid(s->pid, NULL, WNOHANG) == 0) {
            kill(s->pid, SIGKILL);
            while (waitpid(s->pid, NULL, 0) < 0 && errno == EINTR) {}
        }
    }
    free(s->name);
    free(s->command);
    for (int i = 0; i < s->argc; i++) free(s->args[i]);
    free(s->args);
    for (int i = 0; i < s->envc; i++) {
        free(s->env_names[i]);
        free(s->env_values[i]);
    }
    free(s->env_names);
    free(s->env_values);
    memset(s, 0, sizeof(*s));
    s->in_fd = -1;
    s->out_fd = -1;
}

static void mcp_tool_free(ds4_mcp_tool *t) {
    if (!t) return;
    free(t->name);
    free(t->title);
    free(t->description);
    free(t->input_schema);
    free(t->dsml_name);
    memset(t, 0, sizeof(*t));
}

void ds4_mcp_init(ds4_mcp *mcp) {
    memset(mcp, 0, sizeof(*mcp));
    mcp->connect_timeout_ms = DS4_MCP_CONNECT_TIMEOUT_MS;
    mcp->call_timeout_ms = DS4_MCP_CALL_TIMEOUT_MS;
}

void ds4_mcp_close(ds4_mcp *mcp) {
    if (!mcp) return;
    for (int i = 0; i < mcp->servers_len; i++) mcp_server_free(&mcp->servers[i]);
    for (int i = 0; i < mcp->tools_len; i++) mcp_tool_free(&mcp->tools[i]);
    free(mcp->servers);
    free(mcp->tools);
    memset(mcp, 0, sizeof(*mcp));
}

static bool mcp_json_array_next(const char **p, char **raw_out,
                                char *err, size_t err_len) {
    ds4_acp_json_ws(p);
    if (**p == ']') {
        (*p)++;
        *raw_out = NULL;
        return true;
    }
    if (!ds4_acp_json_raw_value(p, raw_out)) {
        mcp_set_err(err, err_len, "invalid JSON array value");
        return false;
    }
    ds4_acp_json_ws(p);
    if (**p == ',') {
        (*p)++;
        return true;
    }
    if (**p == ']') {
        return true;
    }
    mcp_set_err(err, err_len, "invalid JSON array");
    free(*raw_out);
    *raw_out = NULL;
    return false;
}

static bool mcp_json_array_done(const char *p) {
    ds4_acp_json_ws(&p);
    return *p == '\0';
}

static bool mcp_parse_string_array(const char *json, char ***out, int *len_out,
                                   char *err, size_t err_len) {
    const char *p = json;
    ds4_acp_json_ws(&p);
    if (*p != '[') {
        mcp_set_err(err, err_len, "expected string array");
        return false;
    }
    p++;
    char **v = NULL;
    int len = 0, cap = 0;
    for (;;) {
        char *raw = NULL;
        if (!mcp_json_array_next(&p, &raw, err, err_len)) goto fail;
        if (!raw) break;
        const char *q = raw;
        char *s = NULL;
        bool ok = ds4_acp_json_string(&q, &s);
        ds4_acp_json_ws(&q);
        free(raw);
        if (!ok || *q) {
            free(s);
            mcp_set_err(err, err_len, "array item must be a string");
            goto fail;
        }
        if (len == cap) {
            cap = cap ? cap * 2 : 4;
            v = mcp_xrealloc(v, (size_t)cap * sizeof(v[0]));
        }
        v[len++] = s;
    }
    if (!mcp_json_array_done(p)) {
        mcp_set_err(err, err_len, "trailing data after array");
        goto fail;
    }
    *out = v;
    *len_out = len;
    return true;

fail:
    for (int i = 0; i < len; i++) free(v[i]);
    free(v);
    return false;
}

static bool mcp_parse_env_array(const char *json, ds4_mcp_server *s,
                                char *err, size_t err_len) {
    const char *p = json;
    ds4_acp_json_ws(&p);
    if (*p != '[') {
        mcp_set_err(err, err_len, "env must be an array");
        return false;
    }
    p++;
    for (;;) {
        char *raw = NULL;
        if (!mcp_json_array_next(&p, &raw, err, err_len)) return false;
        if (!raw) break;
        char *name = NULL;
        char *value = NULL;
        bool ok = ds4_acp_object_get_string(raw, "name", &name) &&
                  ds4_acp_object_get_string(raw, "value", &value);
        free(raw);
        if (!ok) {
            free(name);
            free(value);
            mcp_set_err(err, err_len, "env entries require name and value");
            return false;
        }
        s->env_names = mcp_xrealloc(s->env_names,
            (size_t)(s->envc + 1) * sizeof(s->env_names[0]));
        s->env_values = mcp_xrealloc(s->env_values,
            (size_t)(s->envc + 1) * sizeof(s->env_values[0]));
        s->env_names[s->envc] = name;
        s->env_values[s->envc] = value;
        s->envc++;
    }
    if (!mcp_json_array_done(p)) {
        mcp_set_err(err, err_len, "trailing data after env");
        return false;
    }
    return true;
}

static void mcp_servers_push(ds4_mcp *mcp, ds4_mcp_server *s) {
    if (mcp->servers_len == mcp->servers_cap) {
        mcp->servers_cap = mcp->servers_cap ? mcp->servers_cap * 2 : 2;
        mcp->servers = mcp_xrealloc(mcp->servers,
            (size_t)mcp->servers_cap * sizeof(mcp->servers[0]));
    }
    mcp->servers[mcp->servers_len++] = *s;
    memset(s, 0, sizeof(*s));
    s->in_fd = -1;
    s->out_fd = -1;
}

static bool mcp_parse_server(const char *json, ds4_mcp_server *s,
                             char *err, size_t err_len) {
    memset(s, 0, sizeof(*s));
    s->in_fd = -1;
    s->out_fd = -1;
    char *type = NULL;
    if (ds4_acp_object_get_string(json, "type", &type) && strcmp(type, "stdio")) {
        mcp_set_err(err, err_len, "unsupported MCP transport: %s", type);
        free(type);
        return false;
    }
    free(type);
    if (!ds4_acp_object_get_string(json, "name", &s->name) ||
        !ds4_acp_object_get_string(json, "command", &s->command))
    {
        mcp_set_err(err, err_len, "MCP stdio server requires name and command");
        return false;
    }
    char *args = NULL;
    if (!ds4_acp_object_get_raw(json, "args", &args)) {
        mcp_set_err(err, err_len, "MCP stdio server requires args");
        return false;
    }
    if (!mcp_parse_string_array(args, &s->args, &s->argc, err, err_len)) {
        free(args);
        return false;
    }
    free(args);
    char *env = NULL;
    if (ds4_acp_object_get_raw(json, "env", &env)) {
        bool ok = mcp_parse_env_array(env, s, err, err_len);
        free(env);
        if (!ok) return false;
    }
    return true;
}

static bool mcp_start_server(ds4_mcp_server *s, char *err, size_t err_len) {
    int to_child[2];
    int from_child[2];
    if (pipe(to_child) != 0) {
        mcp_set_err(err, err_len, "pipe failed: %s", strerror(errno));
        return false;
    }
    if (pipe(from_child) != 0) {
        mcp_set_err(err, err_len, "pipe failed: %s", strerror(errno));
        close(to_child[0]);
        close(to_child[1]);
        return false;
    }
    pid_t pid = fork();
    if (pid < 0) {
        mcp_set_err(err, err_len, "fork failed: %s", strerror(errno));
        close(to_child[0]);
        close(to_child[1]);
        close(from_child[0]);
        close(from_child[1]);
        return false;
    }
    if (pid == 0) {
        dup2(to_child[0], STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        close(to_child[0]);
        close(to_child[1]);
        close(from_child[0]);
        close(from_child[1]);
        for (int i = 0; i < s->envc; i++)
            setenv(s->env_names[i], s->env_values[i], 1);
        char **argv = mcp_xmalloc((size_t)(s->argc + 2) * sizeof(argv[0]));
        argv[0] = s->command;
        for (int i = 0; i < s->argc; i++) argv[i + 1] = s->args[i];
        argv[s->argc + 1] = NULL;
        execvp(s->command, argv);
        _exit(127);
    }
    close(to_child[0]);
    close(from_child[1]);
    s->pid = pid;
    s->in_fd = to_child[1];
    s->out_fd = from_child[0];
    s->next_id = 1;
    return true;
}

static bool mcp_send_raw(ds4_mcp_server *s, const char *json,
                         char *err, size_t err_len) {
    if (!mcp_write_all(s->in_fd, json, strlen(json)) ||
        !mcp_write_all(s->in_fd, "\n", 1))
    {
        mcp_set_err(err, err_len, "write to MCP server %s failed: %s",
                    s->name, strerror(errno));
        return false;
    }
    return true;
}

static bool mcp_request(ds4_mcp_server *s, const char *method,
                        const char *params_json, char **result_out,
                        ds4_mcp_cancel_fn cancel, void *cancel_privdata,
                        int timeout_ms, char *err, size_t err_len) {
    int id = s->next_id++;
    int64_t deadline_ms = timeout_ms > 0 ? mcp_now_ms() + timeout_ms : 0;
    char idbuf[32];
    snprintf(idbuf, sizeof(idbuf), "%d", id);
    char *qmethod = ds4_acp_json_escape(method, strlen(method));
    mcp_buf req = {0};
    mcp_buf_puts(&req, "{\"jsonrpc\":\"2.0\",\"id\":");
    mcp_buf_puts(&req, idbuf);
    mcp_buf_puts(&req, ",\"method\":");
    mcp_buf_puts(&req, qmethod);
    if (params_json) {
        mcp_buf_puts(&req, ",\"params\":");
        mcp_buf_puts(&req, params_json);
    }
    mcp_buf_puts(&req, "}");
    char *wire = mcp_buf_take(&req);
    bool ok = mcp_send_raw(s, wire, err, err_len);
    free(wire);
    free(qmethod);
    if (!ok) return false;

    for (;;) {
        char *line = NULL;
        if (!mcp_read_line(s, &line, cancel, cancel_privdata, deadline_ms,
                           err, err_len))
            return false;
        ds4_acp_message msg;
        char parse_err[160] = {0};
        ds4_acp_parse_result parsed =
            ds4_acp_parse_message(line, &msg, parse_err, sizeof(parse_err));
        free(line);
        if (parsed != DS4_ACP_PARSE_OK) continue;
        if (msg.has_method) {
            if (msg.has_id) {
                mcp_buf resp = {0};
                mcp_buf_puts(&resp, "{\"jsonrpc\":\"2.0\",\"id\":");
                mcp_buf_puts(&resp, msg.id_json);
                mcp_buf_puts(&resp,
                    ",\"error\":{\"code\":-32601,\"message\":\"Method not found\"}}");
                char *r = mcp_buf_take(&resp);
                (void)mcp_send_raw(s, r, NULL, 0);
                free(r);
            }
            ds4_acp_message_free(&msg);
            continue;
        }
        if (!msg.has_id || strcmp(msg.id_json, idbuf)) {
            ds4_acp_message_free(&msg);
            continue;
        }
        if (msg.has_error) {
            mcp_set_err(err, err_len, "MCP server %s returned error: %s",
                        s->name, msg.error_json ? msg.error_json : "unknown");
            ds4_acp_message_free(&msg);
            return false;
        }
        if (!msg.has_result) {
            mcp_set_err(err, err_len, "MCP server %s returned no result", s->name);
            ds4_acp_message_free(&msg);
            return false;
        }
        *result_out = msg.result_json;
        msg.result_json = NULL;
        ds4_acp_message_free(&msg);
        return true;
    }
}

static bool mcp_notify_initialized(ds4_mcp_server *s, char *err, size_t err_len) {
    return mcp_send_raw(s, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
                        err, err_len);
}

static void mcp_sanitize_name(mcp_buf *b, const char *s) {
    bool last_us = false;
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (isalnum(c)) {
            mcp_buf_append(b, (const char *)&c, 1);
            last_us = false;
        } else if (!last_us) {
            mcp_buf_puts(b, "_");
            last_us = true;
        }
    }
    if (b->len && b->ptr[b->len - 1] == '_') b->ptr[--b->len] = '\0';
}

static char *mcp_make_dsml_name(ds4_mcp *mcp, const char *server, const char *tool) {
    mcp_buf b = {0};
    mcp_buf_puts(&b, "mcp__");
    mcp_sanitize_name(&b, server);
    mcp_buf_puts(&b, "__");
    mcp_sanitize_name(&b, tool);
    char *base = mcp_buf_take(&b);
    if (!base[0] || !strcmp(base, "mcp____")) {
        free(base);
        base = mcp_xstrdup("mcp__tool");
    }
    for (int suffix = 0;; suffix++) {
        bool used = false;
        char candidate[512];
        if (suffix == 0) snprintf(candidate, sizeof(candidate), "%s", base);
        else snprintf(candidate, sizeof(candidate), "%s_%d", base, suffix + 1);
        for (int i = 0; i < mcp->tools_len; i++) {
            if (!strcmp(mcp->tools[i].dsml_name, candidate)) {
                used = true;
                break;
            }
        }
        if (!used) {
            free(base);
            return mcp_xstrdup(candidate);
        }
    }
}

static void mcp_tools_push(ds4_mcp *mcp, ds4_mcp_tool *tool) {
    if (mcp->tools_len == mcp->tools_cap) {
        mcp->tools_cap = mcp->tools_cap ? mcp->tools_cap * 2 : 8;
        mcp->tools = mcp_xrealloc(mcp->tools,
            (size_t)mcp->tools_cap * sizeof(mcp->tools[0]));
    }
    mcp->tools[mcp->tools_len++] = *tool;
    memset(tool, 0, sizeof(*tool));
}

static bool mcp_parse_tools(ds4_mcp *mcp, int server_index,
                            const char *result_json, char **next_cursor,
                            char *err, size_t err_len) {
    *next_cursor = NULL;
    char *tools = NULL;
    if (!ds4_acp_object_get_raw(result_json, "tools", &tools)) {
        mcp_set_err(err, err_len, "tools/list result missing tools");
        return false;
    }
    const char *p = tools;
    ds4_acp_json_ws(&p);
    if (*p != '[') {
        free(tools);
        mcp_set_err(err, err_len, "tools must be an array");
        return false;
    }
    p++;
    for (;;) {
        char *raw = NULL;
        if (!mcp_json_array_next(&p, &raw, err, err_len)) {
            free(tools);
            return false;
        }
        if (!raw) break;
        ds4_mcp_tool t = {0};
        t.server_index = server_index;
        bool ok = ds4_acp_object_get_string(raw, "name", &t.name);
        (void)ds4_acp_object_get_string(raw, "title", &t.title);
        (void)ds4_acp_object_get_string(raw, "description", &t.description);
        if (!ds4_acp_object_get_raw(raw, "inputSchema", &t.input_schema))
            t.input_schema = mcp_xstrdup("{\"type\":\"object\",\"properties\":{}}");
        if (ok) {
            t.dsml_name = mcp_make_dsml_name(mcp, mcp->servers[server_index].name,
                                             t.name);
            mcp_tools_push(mcp, &t);
        } else {
            mcp_tool_free(&t);
        }
        free(raw);
    }
    free(tools);
    (void)ds4_acp_object_get_string(result_json, "nextCursor", next_cursor);
    return true;
}

static bool mcp_initialize_server(ds4_mcp *mcp, int index,
                                  ds4_mcp_cancel_fn cancel,
                                  void *cancel_privdata,
                                  char *err, size_t err_len) {
    ds4_mcp_server *s = &mcp->servers[index];
    char *result = NULL;
    bool ok = mcp_request(s, "initialize",
        "{\"protocolVersion\":\"2025-06-18\","
        "\"capabilities\":{},"
        "\"clientInfo\":{\"name\":\"ds4-agent\",\"version\":\"0\"}}",
        &result, cancel, cancel_privdata, mcp->connect_timeout_ms,
        err, err_len);
    free(result);
    if (!ok) return false;
    if (!mcp_notify_initialized(s, err, err_len)) return false;
    char *cursor = NULL;
    do {
        mcp_buf params = {0};
        if (cursor) {
            char *qcursor = ds4_acp_json_escape(cursor, strlen(cursor));
            mcp_buf_puts(&params, "{\"cursor\":");
            mcp_buf_puts(&params, qcursor);
            mcp_buf_puts(&params, "}");
            free(qcursor);
        } else {
            mcp_buf_puts(&params, "{}");
        }
        char *params_json = mcp_buf_take(&params);
        result = NULL;
        ok = mcp_request(s, "tools/list", params_json, &result, cancel,
                         cancel_privdata, mcp->connect_timeout_ms,
                         err, err_len);
        free(params_json);
        if (!ok) {
            free(cursor);
            return false;
        }
        free(cursor);
        cursor = NULL;
        ok = mcp_parse_tools(mcp, index, result, &cursor, err, err_len);
        free(result);
        if (!ok) {
            free(cursor);
            return false;
        }
    } while (cursor);
    free(cursor);
    return ok;
}

bool ds4_mcp_connect_json(ds4_mcp *mcp, const char *servers_json,
                          ds4_mcp_cancel_fn cancel, void *cancel_privdata,
                          char *err, size_t err_len) {
    int connect_timeout_ms =
        mcp_timeout_default(mcp->connect_timeout_ms, DS4_MCP_CONNECT_TIMEOUT_MS);
    int call_timeout_ms =
        mcp_timeout_default(mcp->call_timeout_ms, DS4_MCP_CALL_TIMEOUT_MS);
    ds4_mcp_close(mcp);
    ds4_mcp_init(mcp);
    mcp->connect_timeout_ms = connect_timeout_ms;
    mcp->call_timeout_ms = call_timeout_ms;
    if (!servers_json) return true;
    const char *p = servers_json;
    ds4_acp_json_ws(&p);
    if (*p != '[') {
        mcp_set_err(err, err_len, "mcpServers must be an array");
        return false;
    }
    p++;
    for (;;) {
        char *raw = NULL;
        if (!mcp_json_array_next(&p, &raw, err, err_len)) return false;
        if (!raw) break;
        ds4_mcp_server s;
        bool ok = mcp_parse_server(raw, &s, err, err_len);
        free(raw);
        if (!ok) {
            mcp_server_free(&s);
            return false;
        }
        if (!mcp_start_server(&s, err, err_len)) {
            mcp_server_free(&s);
            return false;
        }
        mcp_servers_push(mcp, &s);
    }
    if (!mcp_json_array_done(p)) {
        mcp_set_err(err, err_len, "trailing data after mcpServers");
        return false;
    }
    for (int i = 0; i < mcp->servers_len; i++) {
        if (!mcp_initialize_server(mcp, i, cancel, cancel_privdata,
                                   err, err_len))
            return false;
    }
    return true;
}

char *ds4_mcp_tools_prompt(const ds4_mcp *mcp) {
    if (!mcp || !mcp->tools_len) return mcp_xstrdup("");
    mcp_buf b = {0};
    mcp_buf_puts(&b,
        "\n\n### MCP Tools\n\n"
        "The following tools are provided by MCP stdio servers. Invoke them by "
        "using their DSML name exactly as listed.\n\n");
    for (int i = 0; i < mcp->tools_len; i++) {
        const ds4_mcp_tool *t = &mcp->tools[i];
        const ds4_mcp_server *s = &mcp->servers[t->server_index];
        char *qname = ds4_acp_json_escape(t->dsml_name, strlen(t->dsml_name));
        char *qdesc = ds4_acp_json_escape(t->description ? t->description : "",
                                          strlen(t->description ? t->description : ""));
        mcp_buf_puts(&b, "{\n  \"type\": \"function\",\n  \"function\": {\n");
        mcp_buf_puts(&b, "    \"name\": ");
        mcp_buf_puts(&b, qname);
        mcp_buf_puts(&b, ",\n    \"description\": ");
        if (t->description && t->description[0]) {
            mcp_buf_puts(&b, qdesc);
        } else {
            char desc[512];
            snprintf(desc, sizeof(desc), "MCP tool %s from server %s.",
                     t->name, s->name);
            char *qd = ds4_acp_json_escape(desc, strlen(desc));
            mcp_buf_puts(&b, qd);
            free(qd);
        }
        mcp_buf_puts(&b, ",\n    \"parameters\": ");
        mcp_buf_puts(&b, t->input_schema ? t->input_schema :
                     "{\"type\":\"object\",\"properties\":{}}");
        mcp_buf_puts(&b, "\n  }\n}\n\n");
        free(qname);
        free(qdesc);
    }
    return mcp_buf_take(&b);
}

const ds4_mcp_tool *ds4_mcp_find_tool(const ds4_mcp *mcp, const char *dsml_name) {
    if (!mcp || !dsml_name) return NULL;
    for (int i = 0; i < mcp->tools_len; i++) {
        if (!strcmp(mcp->tools[i].dsml_name, dsml_name)) return &mcp->tools[i];
    }
    return NULL;
}

static bool mcp_json_bool(const char *json, bool *out) {
    const char *p = json;
    ds4_acp_json_ws(&p);
    if (!strncmp(p, "true", 4)) {
        p += 4;
        ds4_acp_json_ws(&p);
        if (*p) return false;
        *out = true;
        return true;
    }
    if (!strncmp(p, "false", 5)) {
        p += 5;
        ds4_acp_json_ws(&p);
        if (*p) return false;
        *out = false;
        return true;
    }
    return false;
}

static void mcp_append_content_text(mcp_buf *b, const char *block) {
    char *type = NULL;
    if (!ds4_acp_object_get_string(block, "type", &type)) return;
    if (!strcmp(type, "text")) {
        char *text = NULL;
        if (ds4_acp_object_get_string(block, "text", &text)) {
            if (b->len) mcp_buf_puts(b, "\n");
            mcp_buf_puts(b, text);
        }
        free(text);
    } else if (!strcmp(type, "resource_link")) {
        char *uri = NULL;
        char *name = NULL;
        (void)ds4_acp_object_get_string(block, "uri", &uri);
        (void)ds4_acp_object_get_string(block, "name", &name);
        if (uri) {
            if (b->len) mcp_buf_puts(b, "\n");
            mcp_buf_puts(b, "[resource_link");
            if (name && name[0]) {
                mcp_buf_puts(b, " ");
                mcp_buf_puts(b, name);
            }
            mcp_buf_puts(b, ": ");
            mcp_buf_puts(b, uri);
            mcp_buf_puts(b, "]");
        }
        free(uri);
        free(name);
    } else if (!strcmp(type, "resource")) {
        char *res = NULL;
        if (ds4_acp_object_get_raw(block, "resource", &res)) {
            char *uri = NULL;
            char *text = NULL;
            (void)ds4_acp_object_get_string(res, "uri", &uri);
            if (ds4_acp_object_get_string(res, "text", &text)) {
                if (b->len) mcp_buf_puts(b, "\n");
                mcp_buf_puts(b, "[resource");
                if (uri && uri[0]) {
                    mcp_buf_puts(b, " ");
                    mcp_buf_puts(b, uri);
                }
                mcp_buf_puts(b, "]\n");
                mcp_buf_puts(b, text);
            }
            free(uri);
            free(text);
        }
        free(res);
    } else if (!strcmp(type, "image") || !strcmp(type, "audio")) {
        if (b->len) mcp_buf_puts(b, "\n");
        mcp_buf_puts(b, "[unsupported MCP ");
        mcp_buf_puts(b, type);
        mcp_buf_puts(b, " content]");
    }
    free(type);
}

static char *mcp_result_text(const char *result_json, bool *is_error) {
    *is_error = false;
    char *raw = NULL;
    if (ds4_acp_object_get_raw(result_json, "isError", &raw)) {
        (void)mcp_json_bool(raw, is_error);
        free(raw);
    }
    mcp_buf b = {0};
    char *content = NULL;
    if (ds4_acp_object_get_raw(result_json, "content", &content)) {
        const char *p = content;
        ds4_acp_json_ws(&p);
        if (*p == '[') {
            p++;
            for (;;) {
                char *block = NULL;
                char err[80];
                if (!mcp_json_array_next(&p, &block, err, sizeof(err))) break;
                if (!block) break;
                mcp_append_content_text(&b, block);
                free(block);
            }
        }
        free(content);
    }
    if (!b.len && ds4_acp_object_get_raw(result_json, "structuredContent", &raw)) {
        mcp_buf_puts(&b, raw);
        free(raw);
    }
    if (!b.len) mcp_buf_puts(&b, "(no MCP tool content)");
    if (!b.ptr || b.ptr[b.len - 1] != '\n') mcp_buf_puts(&b, "\n");
    return mcp_buf_take(&b);
}

char *ds4_mcp_call_tool(ds4_mcp *mcp, const char *dsml_name,
                        const char *arguments_json,
                        ds4_mcp_cancel_fn cancel, void *cancel_privdata,
                        char *err, size_t err_len) {
    const ds4_mcp_tool *t = ds4_mcp_find_tool(mcp, dsml_name);
    if (!t) {
        mcp_set_err(err, err_len, "unknown MCP tool: %s", dsml_name);
        return NULL;
    }
    ds4_mcp_server *s = &mcp->servers[t->server_index];
    char *qname = ds4_acp_json_escape(t->name, strlen(t->name));
    mcp_buf params = {0};
    mcp_buf_puts(&params, "{\"name\":");
    mcp_buf_puts(&params, qname);
    mcp_buf_puts(&params, ",\"arguments\":");
    mcp_buf_puts(&params, arguments_json ? arguments_json : "{}");
    mcp_buf_puts(&params, "}");
    char *params_json = mcp_buf_take(&params);
    char *result = NULL;
    bool ok = mcp_request(s, "tools/call", params_json, &result, cancel,
                          cancel_privdata,
                          mcp_timeout_default(mcp->call_timeout_ms,
                                              DS4_MCP_CALL_TIMEOUT_MS),
                          err, err_len);
    free(params_json);
    free(qname);
    if (!ok) return NULL;
    bool is_error = false;
    char *text = mcp_result_text(result, &is_error);
    free(result);
    if (is_error) {
        mcp_buf b = {0};
        mcp_buf_puts(&b, "Tool error: MCP tool ");
        mcp_buf_puts(&b, dsml_name);
        mcp_buf_puts(&b, " returned an error\n");
        mcp_buf_puts(&b, text);
        free(text);
        return mcp_buf_take(&b);
    }
    return text;
}
