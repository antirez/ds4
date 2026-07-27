#include "ds4_agent_internal.h"



/* ============================================================================
 * Tool Argument Parsing And File Tool Helpers
 * ============================================================================
 */

int agent_parse_timeout(const char *s) {
    if (!s || !s[0]) return 3600;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || v <= 0.0 || !isfinite(v)) return 3600;
    if (v < 1.0) v = 1.0;
    if (v > 24.0 * 3600.0) v = 24.0 * 3600.0;
    return (int)v;
}



int agent_parse_int_default(const char *s, int def, int min, int max) {
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) return def;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return def;
    if (v < min) v = min;
    if (v > max) v = max;
    return (int)v;
}



bool agent_parse_bool_default(const char *s, bool def) {
    if (!s || !s[0]) return def;
    if (!strcasecmp(s, "true") || !strcasecmp(s, "yes") || !strcmp(s, "1"))
        return true;
    if (!strcasecmp(s, "false") || !strcasecmp(s, "no") || !strcmp(s, "0"))
        return false;
    return def;
}



void agent_line_spans_free(agent_line_spans *spans) {
    free(spans->v);
    memset(spans, 0, sizeof(*spans));
}



static void agent_line_spans_push(agent_line_spans *spans, agent_line_span span) {
    if (spans->len == spans->cap) {
        spans->cap = spans->cap ? spans->cap * 2 : 128;
        spans->v = (agent_line_span *)agent_xrealloc(spans->v, (size_t)spans->cap * sizeof(spans->v[0]));
    }
    spans->v[spans->len++] = span;
}



/* Split a text buffer into line spans.  content_end excludes CR/LF so callers
 * can print or compare line content without newline spelling differences. */
void agent_split_lines(const char *data, size_t len, agent_line_spans *spans) {
    size_t pos = 0;
    while (pos < len) {
        size_t start = pos;
        while (pos < len && data[pos] != '\n' && data[pos] != '\r') pos++;
        size_t content_end = pos;
        if (pos < len) {
            if (data[pos] == '\r' && pos + 1 < len && data[pos + 1] == '\n')
                pos += 2;
            else
                pos++;
        }
        agent_line_spans_push(spans, (agent_line_span){
            .start = start,
            .content_end = content_end,
            .end = pos,
        });
    }
}



int agent_read_file_bytes(const char *path, char **data, size_t *len,
                                 char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    char *buf = NULL;
    size_t used = 0, cap = 0;
    char tmp[8192];
    while (true) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        if (n) {
            if (used + n > AGENT_FILE_MAX_BYTES) {
                fclose(fp);
                free(buf);
                snprintf(err, errlen, "file too large: %s exceeds %d bytes",
                         path, AGENT_FILE_MAX_BYTES);
                return -1;
            }
            if (used + n + 1 > cap) {
                cap = cap ? cap * 2 : 8192;
                while (cap < used + n + 1) cap *= 2;
                buf = (char *)agent_xrealloc(buf, cap);
            }
            memcpy(buf + used, tmp, n);
            used += n;
            buf[used] = '\0';
        }
        if (n < sizeof(tmp)) {
            if (ferror(fp)) {
                snprintf(err, errlen, "read %s: %s", path, strerror(errno));
                fclose(fp);
                free(buf);
                return -1;
            }
            break;
        }
    }
    fclose(fp);
    if (!buf) buf = xstrdup("");
    *data = buf;
    *len = used;
    return 0;
}



static int agent_line_for_offset(const agent_line_spans *spans, size_t offset) {
    if (!spans || spans->len <= 0) return 1;
    for (int i = 0; i < spans->len; i++) {
        if (offset < spans->v[i].end) return i + 1;
    }
    return spans->len;
}



bool agent_old_new_line_effect(const char *old_data, size_t old_len,
                                      const char *new_data, size_t new_len,
                                      size_t edit_offset, size_t replaced_len,
                                      int *start_line, int *end_line,
                                      int *delta) {
    agent_line_spans old_spans = {0};
    agent_line_spans new_spans = {0};
    agent_split_lines(old_data, old_len, &old_spans);
    agent_split_lines(new_data, new_len, &new_spans);
    bool ok = old_spans.len > 0;
    if (ok) {
        size_t old_last = edit_offset;
        if (replaced_len > 0) old_last = edit_offset + replaced_len - 1;
        if (old_last >= old_len) old_last = old_len ? old_len - 1 : 0;
        if (start_line) *start_line = agent_line_for_offset(&old_spans, edit_offset);
        if (end_line) *end_line = agent_line_for_offset(&old_spans, old_last);
        if (delta) *delta = new_spans.len - old_spans.len;
    }
    agent_line_spans_free(&old_spans);
    agent_line_spans_free(&new_spans);
    return ok;
}



void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end);



char *agent_edit_result(const char *path,
                                       int start_line, int end_line, int delta,
                                       const char *new_data, size_t new_len,
                                       const char *kind) {
    agent_buf b = {0};
    char msg[PATH_MAX + 180];
    snprintf(msg, sizeof(msg), "Edited %s using %s\n", path, kind);
    agent_buf_puts(&b, msg);
    if (start_line > 0 && end_line >= start_line) {
        snprintf(msg, sizeof(msg),
                 "Touched old lines %d-%d; current post-edit context follows.\n",
                 start_line, end_line);
        agent_buf_puts(&b, msg);
        if (delta != 0) {
            snprintf(msg, sizeof(msg),
                     "Line shift: old lines after %d moved by %+d (old line %d is now line %d). Re-read before relying on old line numbers there.\n",
                     end_line, delta, end_line + 1, end_line + 1 + delta);
            agent_buf_puts(&b, msg);
        }
    }
    if (start_line > 0 && end_line >= start_line) {
        int new_anchor_end = end_line + delta;
        if (new_anchor_end < start_line) new_anchor_end = start_line;
        agent_edit_result_append_context(&b, path, new_data, new_len,
                                         start_line, new_anchor_end);
    }
    return agent_buf_take(&b);
}



static void agent_worker_set_more(agent_worker *w, const char *path,
                                  int next_line, bool bare) {
    snprintf(w->more_path, sizeof(w->more_path), "%s", path ? path : "");
    w->more_next_line = next_line;
    w->more_bare = bare;
    w->more_valid = path && path[0] && next_line > 0;
}



bool agent_tool_result_fits_context(agent_worker *w, const char *result,
                                           int reserve_tokens,
                                           int *tokens_out) {
    ds4_tokens tmp = {0};
    ds4_tokens_copy(&tmp, &w->transcript);
    ds4_chat_append_message(w->engine, &tmp, "tool", result ? result : "");
    int tokens = tmp.len;
    ds4_tokens_free(&tmp);
    if (tokens_out) *tokens_out = tokens;
    return tokens + reserve_tokens < w->cfg->gen.ctx_size;
}



/* Read file text for the model.  Normal mode shows plain line numbers.  Raw
 * mode is reserved for cases where line decoration would corrupt the payload
 * being inspected. */
static char *agent_read_range(agent_worker *w, const char *path, int start_line,
                              int max_lines, bool whole_file, bool bare,
                              bool set_more) {
    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (!path || !path[0]) return xstrdup("Tool error: read requires path\n");
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (start_line < 1) start_line = 1;
    int start_idx = start_line - 1;
    if (start_idx > spans.len) start_idx = spans.len;
    if (whole_file) {
        max_lines = spans.len - start_idx;
    } else {
        if (max_lines <= 0) max_lines = AGENT_READ_DEFAULT_LINES;
    }
    int end_idx = start_idx + max_lines;
    if (end_idx > spans.len) end_idx = spans.len;

    agent_buf out = {0};
    if (bare) {
        size_t start = start_idx < spans.len ? spans.v[start_idx].start : len;
        size_t end = end_idx > start_idx ? spans.v[end_idx - 1].end : start;
        agent_buf_append(&out, data + start, end - start);
        if (end > start && out.ptr[out.len - 1] != '\n') agent_buf_puts(&out, "\n");
        if (end_idx < spans.len) {
            char note[160];
            snprintf(note, sizeof(note),
                     "[Read truncated at line %d of %d. continue_offset=%d. "
                     "Call more with count=%d to read the next chunk.]\n",
                     end_idx, spans.len, end_idx + 1,
                     max_lines > 0 ? max_lines : AGENT_READ_DEFAULT_LINES);
            agent_buf_puts(&out, note);
        }
    } else {
        char hdr[PATH_MAX + 160];
        if (end_idx < spans.len) {
            snprintf(hdr, sizeof(hdr),
                     "%s: lines %d-%d of %d; continue_offset=%d; "
                     "call more with count=%d to read the next chunk\n",
                     path, spans.len ? start_idx + 1 : 0, end_idx, spans.len,
                     end_idx + 1, max_lines > 0 ? max_lines : AGENT_READ_DEFAULT_LINES);
        } else {
            snprintf(hdr, sizeof(hdr), "%s: lines %d-%d of %d\n",
                     path, spans.len ? start_idx + 1 : 0, end_idx, spans.len);
        }
        agent_buf_puts(&out, hdr);
        for (int i = start_idx; i < end_idx; i++) {
            agent_line_span sp = spans.v[i];
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "%d ", i + 1);
            agent_buf_puts(&out, prefix);
            agent_buf_append(&out, data + sp.start, sp.content_end - sp.start);
            agent_buf_puts(&out, "\n");
        }
    }
    if (set_more) {
        if (end_idx < spans.len) agent_worker_set_more(w, path, end_idx + 1, bare);
        else agent_worker_set_more(w, NULL, 0, false);
    }
    agent_line_spans_free(&spans);
    free(data);
    return agent_buf_take(&out);
}



char *agent_tool_read(agent_worker *w, const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    bool whole = agent_parse_bool_default(agent_tool_arg_value(call, "whole"), false);
    int start = agent_parse_int_default(agent_tool_arg_value(call, "start_line"),
                                        1, 1, INT_MAX);
    int count = agent_parse_int_default(agent_tool_arg_value(call, "max_lines"),
                                        AGENT_READ_DEFAULT_LINES, 1, INT_MAX);
    bool raw = agent_parse_bool_default(agent_tool_arg_value(call, "raw"), false);
    return agent_read_range(w, path, start, count, whole, raw, true);
}



char *agent_tool_more(agent_worker *w, const agent_tool_call *call) {
    int count = agent_parse_int_default(agent_tool_arg_value(call, "count"),
                                        AGENT_READ_DEFAULT_LINES, 1, INT_MAX);
    if (!w->more_valid) return xstrdup("Tool error: no previous output to continue\n");
    return agent_read_range(w, w->more_path, w->more_next_line, count, false,
                            w->more_bare, true);
}



char *agent_tool_write(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *path = agent_tool_arg_value(call, "path");
    const char *content = agent_tool_arg_value(call, "content");
    if (!path || !path[0]) return xstrdup("Tool error: write requires path\n");
    if (!content) return xstrdup("Tool error: write requires content\n");
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: open for write failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    size_t len = strlen(content);
    size_t wr = fwrite(content, 1, len, fp);
    int close_rc = fclose(fp);
    if (wr != len || close_rc != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: write failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 160];
    snprintf(msg, sizeof(msg), "Wrote %zu bytes to %s\n", len, path);
    return xstrdup(msg);
}



char *agent_tool_list(const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    DIR *dir = opendir(path);
    if (!dir) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: opendir failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    agent_buf out = {0};
    char hdr[PATH_MAX + 64];
    snprintf(hdr, sizeof(hdr), "%s:\n", path);
    agent_buf_puts(&out, hdr);
    struct dirent *de;
    int shown = 0;
    while ((de = readdir(dir)) != NULL && shown < 300) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, de->d_name);
        struct stat st;
        if (lstat(full, &st) != 0) continue;
        char type = S_ISDIR(st.st_mode) ? 'd' :
                    S_ISLNK(st.st_mode) ? 'l' :
                    S_ISREG(st.st_mode) ? '-' : '?';
        char line[PATH_MAX + 96];
        snprintf(line, sizeof(line), "%c %10lld %s%s\n", type,
                 (long long)st.st_size, de->d_name, S_ISDIR(st.st_mode) ? "/" : "");
        agent_buf_puts(&out, line);
        shown++;
    }
    if (de) agent_buf_puts(&out, "... more entries omitted ...\n");
    closedir(dir);
    return agent_buf_take(&out);
}

