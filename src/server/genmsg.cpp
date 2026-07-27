#include "ds4_server_internal.h"



static long long wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}



/* Thread-local: only the GPU worker installs a writer (around a job), so
 * client threads always see NULL here and keep the classic bounded-blocking
 * send_all behavior with zero contention. */
static __thread slot_writer *g_slot_writer;



void slot_writer_init(slot_writer *w, int fd) {
    memset(w, 0, sizeof(*w));
    w->fd = fd;
}



void slot_writer_install(slot_writer *w) {
    g_slot_writer = w;
}



void slot_writer_free(slot_writer *w) {
    if (!w) return;
    if (g_slot_writer == w) g_slot_writer = NULL;
    buf_free(&w->pending);
    w->off = 0;
    w->stall_deadline_ms = 0;
}



/* Push queued bytes without ever sleeping. Returns false once the writer has
 * failed (hard socket error, stall timeout with bytes still queued, or pending
 * overflow); bytes may legitimately remain queued on a true return. */
bool slot_writer_flush(slot_writer *w) {
    if (w->failed) return false;
    while (w->off < w->pending.len) {
        ssize_t r = send(w->fd, w->pending.ptr + w->off, w->pending.len - w->off, 0);
        if (r > 0) {
            w->off += (size_t)r;
            w->stall_deadline_ms = wall_ms() + DS4_SERVER_SEND_STALL_TIMEOUT_MS;
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        w->failed = true;
        return false;
    }
    if (w->off == w->pending.len) {
        w->pending.len = 0;
        w->off = 0;
        w->stall_deadline_ms = 0;
        return true;
    }
    if (w->off >= 65536) {
        /* Compact the consumed prefix so a long slow stream cannot grow the
         * queue by its full transferred size. */
        memmove(w->pending.ptr, w->pending.ptr + w->off, w->pending.len - w->off);
        w->pending.len -= w->off;
        w->off = 0;
    }
    if (wall_ms() >= w->stall_deadline_ms) {
        w->failed = true;
        return false;
    }
    return true;
}



static bool slot_writer_send(slot_writer *w, const void *p, size_t n) {
    if (w->failed) return false;
    if (g_stop_requested) {
        /* Match blocking send_all: any write during shutdown fails. */
        w->failed = true;
        return false;
    }
    const char *s = (const char *)p;
    if (w->off == w->pending.len) {
        /* Nothing queued: try the wire directly so hard errors (EPIPE from a
         * closed peer) surface on this call, exactly like the blocking path. */
        while (n) {
            ssize_t r = send(w->fd, s, n, 0);
            if (r > 0) {
                s += r;
                n -= (size_t)r;
                continue;
            }
            if (r < 0 && errno == EINTR) continue;
            if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
            w->failed = true;
            return false;
        }
        if (!n) return true;
        w->pending.len = 0;
        w->off = 0;
    }
    buf_append(&w->pending, s, n);
    if (w->pending.len - w->off > DS4_SERVER_WRITER_MAX_PENDING_BYTES) {
        w->failed = true;
        return false;
    }
    if (!w->stall_deadline_ms) {
        w->stall_deadline_ms = wall_ms() + DS4_SERVER_SEND_STALL_TIMEOUT_MS;
    }
    return slot_writer_flush(w);
}



/* Blocking completion of all queued bytes, used once per job after the final
 * response: same poll cadence and stall timeout as blocking send_all. */
bool slot_writer_drain(slot_writer *w) {
    while (!w->failed && w->off < w->pending.len) {
        if (g_stop_requested) {
            w->failed = true;
            break;
        }
        if (!slot_writer_flush(w)) break;
        if (w->off >= w->pending.len) break;
        long long remaining = w->stall_deadline_ms - wall_ms();
        if (remaining <= 0) {
            w->failed = true;
            break;
        }
        struct pollfd pfd = {.fd = w->fd, .events = POLLOUT};
        int timeout = remaining > 50 ? 50 : (int)remaining;
        int rc;
        do {
            rc = poll(&pfd, 1, timeout);
        } while (rc < 0 && errno == EINTR);
        if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
            w->failed = true;
            break;
        }
    }
    return !w->failed;
}



bool send_all(int fd, const void *p, size_t n) {
    if (g_slot_writer && g_slot_writer->fd == fd) {
        return slot_writer_send(g_slot_writer, p, n);
    }
    const char *s = (const char *)p;
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



void json_escape(buf *b, const char *s) {
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



void json_escape_n(buf *b, const char *s, size_t n) {
    char *tmp = xstrndup(s ? s : "", n);
    json_escape(b, tmp);
    free(tmp);
}



void json_escape_fragment_n(buf *b, const char *s, size_t n) {
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


const char *find_any_tool_start(const char *s) {
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



void observe_tool_markers(const char *scan, bool *saw_start,
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



size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit) {
    while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
    return limit;
}



static const char *skip_ascii_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}



const char *find_last_substr(const char *s, const char *needle) {
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
char *dsml_unescape_text(const char *s) {
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



char *dsml_attr(const char *tag, const char *name) {
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
    const bool opened_think = !strncmp(body, "<think>", 7);
    if (opened_think) body += 7;

    char *think_end = strstr(body, "</think>");
    if (think_end) {
        *think_end = '\0';
        *reasoning_out = xstrdup(body);
        *content_out = xstrdup(think_end + 8);
    } else if (opened_think) {
        /* Generation ended inside the think block (token cap / stop). The
         * partial reasoning is still reasoning: surfacing it as content
         * hands raw chain-of-thought to clients that score or display the
         * answer channel (the streaming path already promises reasoning is
         * never visible assistant text). */
        *reasoning_out = xstrdup(body);
        *content_out = xstrdup("");
    } else {
        *reasoning_out = NULL;
        *content_out = xstrdup(s);
    }
    free(s);
}



bool parse_generated_message_ex(const char *text, bool require_thinking_closed,
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
            /* Model did not close thinking (truncation or stop inside the
             * think block) — the whole text is reasoning, even when the
             * <think> opener lives in the prompt rather than the generation.
             * Ignore any DSML in it, and keep it off the content channel. */
            server_log(DS4_LOG_TOOL, "thinking not closed, ignoring DSML in reasoning");
            const char *body = !strncmp(text, "<think>", 7) ? text + 7 : text;
            *reasoning_out = xstrdup(body);
            *content_out = xstrdup("");
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



/* Try to repair a truncated DSML block.
 *
 * DSML nesting order is: tool_calls > invoke > parameter.
 * Single-pass scan: count opens vs closes, then append missing closing tags.
 *
 * Returns true if repair was applied, false if the text had no recognizable DSML
 * or was already balanced.  This deliberately does not rewrite malformed but
 * balanced DSML into assistant text; semantic recovery belongs to the model. */
bool try_repair_dsml(const char *s, size_t len, buf *out) {
    if (!s || !len) return false;

    /* Only scan DSML tags after the last </think>.  DSML mentioned inside
     * reasoning is not executable — it inflates tag counts and causes false
     * positive repairs.  If no </think> is found, scan from the start
     * (thinking mode is not active or thinking was never opened). */
    const char *think_end = find_last_substr(s, "</think>");
    const char *scan_start = think_end ? (think_end + 8) : s;
    size_t scan_len = (size_t)((s + len) - scan_start);

    /* Detect style from first <tool_calls> tag */
    const char *ts, *te, *is, *ie, *ps, *pe;
    if (strstr(scan_start, DS4_TOOL_CALLS_START)) {
        ts = DS4_TOOL_CALLS_START;  te = DS4_TOOL_CALLS_END;
        is = DS4_INVOKE_START;      ie = DS4_INVOKE_END;
        ps = DS4_PARAM_START;       pe = DS4_PARAM_END;
    } else if (strstr(scan_start, DS4_TOOL_CALLS_START_SHORT)) {
        ts = DS4_TOOL_CALLS_START_SHORT;  te = DS4_TOOL_CALLS_END_SHORT;
        is = DS4_INVOKE_START_SHORT;      ie = DS4_INVOKE_END_SHORT;
        ps = DS4_PARAM_START_SHORT;       pe = DS4_PARAM_END_SHORT;
    } else if (strstr(scan_start, "<tool_calls>")) {
        ts = "<tool_calls>";   te = "</tool_calls>";
        is = "<invoke";        ie = "</invoke>";
        ps = "<parameter";     pe = "</parameter>";
    } else {
        return false; /* No recognizable DSML start tag */
    }

    /* Single-pass: count all 6 tag types in one scan */
    size_t tos = 0, toe = 0, ios = 0, ioe = 0, pos = 0, poe = 0;
    const char *e = scan_start + scan_len;
    for (const char *p = scan_start; p < e; ) {
        size_t d;
        if ((d = strlen(ts)) && !strncmp(p, ts, d)) { tos++; p += d; }
        else if ((d = strlen(te)) && !strncmp(p, te, d)) { toe++; p += d; }
        else if ((d = strlen(is)) && !strncmp(p, is, d)) { ios++; p += d; }
        else if ((d = strlen(ie)) && !strncmp(p, ie, d)) { ioe++; p += d; }
        else if ((d = strlen(ps)) && !strncmp(p, ps, d)) { pos++; p += d; }
        else if ((d = strlen(pe)) && !strncmp(p, pe, d)) { poe++; p += d; }
        else p++;
    }
    if (tos == toe && ios == ioe && pos == poe) return false;
    if (toe > tos || ioe > ios || poe > pos) {
        /* Extra closing tags are not a truncation pattern.  Refuse repair so the
         * unsigned differences below cannot wrap and append a huge suffix. */
        return false;
    }
    /* Repair: copy original text and append missing closing tags in reverse order */
    buf_puts(out, s);
    for (size_t i = 0; i < pos - poe; i++) buf_puts(out, pe);
    for (size_t i = 0; i < ios - ioe; i++) buf_puts(out, ie);
    for (size_t i = 0; i < tos - toe; i++) buf_puts(out, te);
    return true;
}



static const char *tool_parse_failure_recovery_finish(const char *finish) {
    /* Once DSML failed to parse there is no executable tool call to report.
     * Preserve a true length stop, because callers can distinguish truncation
     * from a completed turn.  Every other non-error tool-parse failure becomes
     * a normal assistant stop with the raw model text returned as content. */
    if (finish && !strcmp(finish, "length")) return "length";
    return "stop";
}



bool parse_generated_message_for_response(const char *text,
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

    /* A malformed tool block is model output, not a server failure.  The
     * generation worker may hide this turn from the client, append a tool error
     * plus protocol reminder to the live session, and let the model try again.
     * If that continuation is unavailable, parsed_content keeps the raw text as
     * a last-resort assistant fallback instead of crashing the request. */
    const char *finish = finish_io && *finish_io ? *finish_io : "stop";
    if (has_tools && saw_tool_start && strcmp(finish, "error") != 0) {
        if (finish_io) *finish_io = tool_parse_failure_recovery_finish(finish);
        if (err && errlen) snprintf(err, errlen, "invalid tool call");
        if (recovered_out) *recovered_out = true;
    }
    return false;
}



void append_json_object_string(buf *b, const char *json) {
    buf tmp = {0};
    append_json_object_or_empty(&tmp, json);
    json_escape(b, tmp.ptr ? tmp.ptr : "{}");
    buf_free(&tmp);
}



void append_tool_calls_json(buf *b, const tool_calls *calls, const char *id_prefix,
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



void append_tool_call_deltas_json(buf *b, const tool_calls *calls, const char *id_prefix,
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

