#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ds4_agent_context.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DS4_AGENT_CONTEXT_MAX_META_BYTES (1024 * 1024)
#define DS4_AGENT_CONTEXT_MAX_SIDE_EFFECTS 64

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} ds4_agent_context_buf;

static void ctx_set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (!err || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static void *ctx_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "ds4-agent-context: out of memory\n");
        abort();
    }
    return p;
}

static void *ctx_xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) {
        fprintf(stderr, "ds4-agent-context: out of memory\n");
        abort();
    }
    return p;
}

static char *ctx_xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *out = ctx_xmalloc(n + 1);
    memcpy(out, s, n + 1);
    return out;
}

static void ctx_buf_append(ds4_agent_context_buf *b, const char *s, size_t n) {
    if (n == 0) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = ctx_xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void ctx_buf_puts(ds4_agent_context_buf *b, const char *s) {
    if (s) ctx_buf_append(b, s, strlen(s));
}

static char *ctx_buf_take(ds4_agent_context_buf *b) {
    if (!b->ptr) return ctx_xstrdup("");
    char *out = b->ptr;
    b->ptr = NULL;
    b->len = 0;
    b->cap = 0;
    return out;
}

static int ctx_read_file_bytes(const char *path, char **data, size_t *len,
                               char *err, size_t err_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        ctx_set_err(err, err_len, "%s", strerror(errno));
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        ctx_set_err(err, err_len, "%s", strerror(errno));
        fclose(fp);
        return -1;
    }
    long sz = ftell(fp);
    if (sz < 0) {
        ctx_set_err(err, err_len, "%s", strerror(errno));
        fclose(fp);
        return -1;
    }
    if ((unsigned long)sz > DS4_AGENT_CONTEXT_MAX_META_BYTES) {
        ctx_set_err(err, err_len, "metadata file too large: %s", path);
        fclose(fp);
        return -1;
    }
    rewind(fp);
    char *buf = ctx_xmalloc((size_t)sz + 1);
    size_t got = fread(buf, 1, (size_t)sz, fp);
    if (got != (size_t)sz && ferror(fp)) {
        ctx_set_err(err, err_len, "%s", strerror(errno));
        free(buf);
        fclose(fp);
        return -1;
    }
    buf[got] = '\0';
    fclose(fp);
    if (data) *data = buf; else free(buf);
    if (len) *len = got;
    return 0;
}

void ds4_agent_context_meta_free(ds4_agent_context_meta *m) {
    if (!m) return;
    free(m->label);
    free(m->kv_file);
    free(m->memory_file);
    memset(m, 0, sizeof(*m));
}

bool ds4_agent_context_id_valid(const char *id) {
    if (!id || strlen(id) != 40) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)id[i])) return false;
    }
    return true;
}

bool ds4_agent_context_file_component_safe(const char *s) {
    if (!s || !s[0]) return false;
    for (const char *p = s; *p; p++) {
        if (*p == '/' || *p == '\\') return false;
    }
    return true;
}

char *ds4_agent_context_file_name(const char id[41], const char *suffix) {
    ds4_agent_context_buf b = {0};
    ctx_buf_append(&b, id, 40);
    ctx_buf_puts(&b, suffix);
    return ctx_buf_take(&b);
}

char *ds4_agent_context_path_for_file(const char *context_dir, const char *file) {
    if (!context_dir || !context_dir[0]) return ctx_xstrdup(file ? file : "");
    if (!file || !file[0]) return ctx_xstrdup(context_dir);
    size_t dir_len = strlen(context_dir);
    bool need_sep = context_dir[dir_len - 1] != '/';
    ds4_agent_context_buf b = {0};
    ctx_buf_puts(&b, context_dir);
    if (need_sep) ctx_buf_puts(&b, "/");
    ctx_buf_puts(&b, file);
    return ctx_buf_take(&b);
}

char *ds4_agent_context_limited_strdup(const char *s, size_t max) {
    if (!s) return ctx_xstrdup("");
    size_t n = strlen(s);
    if (n > max) n = max;
    char *out = ctx_xmalloc(n + 1);
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

char *ds4_agent_context_oneline(const char *s, size_t max) {
    char *out = ds4_agent_context_limited_strdup(s, max);
    for (char *p = out; *p; p++) {
        if (*p == '\n' || *p == '\r' || *p == '\t') *p = ' ';
    }
    return out;
}

static void ctx_json_escape(ds4_agent_context_buf *b, const char *s) {
    if (!s) return;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '\\': ctx_buf_puts(b, "\\\\"); break;
        case '"': ctx_buf_puts(b, "\\\""); break;
        case '\n': ctx_buf_puts(b, "\\n"); break;
        case '\r': ctx_buf_puts(b, "\\r"); break;
        case '\t': ctx_buf_puts(b, "\\t"); break;
        default:
            if (*p < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof(tmp), "\\u%04x", *p);
                ctx_buf_puts(b, tmp);
            } else {
                char c = (char)*p;
                ctx_buf_append(b, &c, 1);
            }
            break;
        }
    }
}

static bool ctx_write_atomic_text(const char *path, const char *text,
                                  char *err, size_t err_len) {
    ds4_agent_context_buf tmpl = {0};
    ctx_buf_puts(&tmpl, path);
    ctx_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = ctx_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        ctx_set_err(err, err_len, "%s", strerror(errno));
        free(tmp);
        return false;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        ctx_set_err(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        free(tmp);
        return false;
    }
    size_t len = strlen(text ? text : "");
    errno = 0;
    bool ok = fwrite(text ? text : "", 1, len, fp) == len && fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    if (!ok) {
        ctx_set_err(err, err_len, "%s",
                    saved_errno ? strerror(saved_errno) : "write failed");
        unlink(tmp);
    }
    free(tmp);
    return ok;
}

bool ds4_agent_context_write_meta(const ds4_agent_context_meta *m,
                                  const char *meta_path,
                                  char *err, size_t err_len) {
    ds4_agent_context_buf b = {0};
    char num[80];
    ctx_buf_puts(&b, "{\n");
    ctx_buf_puts(&b, "  \"id\": \"");
    ctx_json_escape(&b, m->id);
    ctx_buf_puts(&b, "\",\n  \"label\": \"");
    ctx_json_escape(&b, m->label ? m->label : "");
    ctx_buf_puts(&b, "\",\n");
    snprintf(num, sizeof(num), "  \"created_at\": %" PRIu64 ",\n", m->created_at);
    ctx_buf_puts(&b, num);
    snprintf(num, sizeof(num), "  \"world_epoch\": %" PRIu64 ",\n", m->world_epoch);
    ctx_buf_puts(&b, num);
    snprintf(num, sizeof(num), "  \"transcript_tokens\": %d,\n", m->transcript_tokens);
    ctx_buf_puts(&b, num);
    ctx_buf_puts(&b, "  \"kv_path\": \"");
    ctx_json_escape(&b, m->kv_file ? m->kv_file : "");
    ctx_buf_puts(&b, "\",\n  \"memory_path\": \"");
    ctx_json_escape(&b, m->memory_file ? m->memory_file : "");
    ctx_buf_puts(&b, "\",\n  \"memory_sha1\": null\n}\n");
    char *text = ctx_buf_take(&b);
    bool ok = ctx_write_atomic_text(meta_path, text, err, err_len);
    free(text);
    return ok;
}

static const char *ctx_json_skip_string(const char *p) {
    if (!p || *p != '"') return p;
    p++;
    while (*p) {
        if (*p == '\\' && p[1]) {
            p += 2;
            continue;
        }
        if (*p == '"') return p + 1;
        p++;
    }
    return p;
}

static bool ctx_json_key_matches(const char *start, const char *end,
                                 const char *key) {
    const char *p = start;
    const char *k = key;
    while (p < end) {
        char c = *p++;
        if (c == '\\' && p < end) c = *p++;
        if (*k != c) return false;
        k++;
    }
    return *k == '\0';
}

static const char *ctx_json_find_value(const char *json, const char *key) {
    const char *p = json;
    while (p && *p) {
        if (*p != '"') {
            p++;
            continue;
        }
        const char *start = p + 1;
        const char *after = ctx_json_skip_string(p);
        if (!after || after == p || after[-1] != '"') return NULL;
        const char *end = after - 1;
        const char *q = after;
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q == ':' && ctx_json_key_matches(start, end, key)) {
            q++;
            while (*q && isspace((unsigned char)*q)) q++;
            return q;
        }
        p = after;
    }
    return NULL;
}

static bool ctx_json_get_string(const char *json, const char *key, char **out) {
    const char *p = ctx_json_find_value(json, key);
    if (!p || *p != '"') return false;
    p++;
    ds4_agent_context_buf b = {0};
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (!*p) break;
            switch (*p) {
            case 'n': ctx_buf_puts(&b, "\n"); break;
            case 'r': ctx_buf_puts(&b, "\r"); break;
            case 't': ctx_buf_puts(&b, "\t"); break;
            case '\\': ctx_buf_puts(&b, "\\"); break;
            case '"': ctx_buf_puts(&b, "\""); break;
            default: ctx_buf_append(&b, p, 1); break;
            }
            p++;
        } else {
            ctx_buf_append(&b, p, 1);
            p++;
        }
    }
    if (*p != '"') {
        free(b.ptr);
        return false;
    }
    *out = ctx_buf_take(&b);
    return true;
}

static bool ctx_json_get_u64(const char *json, const char *key, uint64_t *out) {
    const char *p = ctx_json_find_value(json, key);
    if (!p || !isdigit((unsigned char)*p)) return false;
    char *end = NULL;
    unsigned long long v = strtoull(p, &end, 10);
    if (end == p) return false;
    *out = (uint64_t)v;
    return true;
}

static bool ctx_json_get_int(const char *json, const char *key, int *out) {
    uint64_t v = 0;
    if (!ctx_json_get_u64(json, key, &v) || v > (uint64_t)INT_MAX) return false;
    *out = (int)v;
    return true;
}

bool ds4_agent_context_read_meta_file(const char *path,
                                      ds4_agent_context_meta *m,
                                      char *err, size_t err_len) {
    char *json = NULL;
    size_t len = 0;
    if (ctx_read_file_bytes(path, &json, &len, err, err_len) != 0) return false;
    (void)len;
    memset(m, 0, sizeof(*m));
    char *id = NULL;
    bool ok = ctx_json_get_string(json, "id", &id);
    if (ok) {
        if (strlen(id) < sizeof(m->id)) snprintf(m->id, sizeof(m->id), "%s", id);
        else ok = false;
        free(id);
        if (ok) ok = ds4_agent_context_id_valid(m->id);
    }
    if (ok && !ctx_json_get_string(json, "label", &m->label))
        m->label = ctx_xstrdup("");
    if (ok && !ctx_json_get_u64(json, "created_at", &m->created_at))
        ok = false;
    if (ok && !ctx_json_get_u64(json, "world_epoch", &m->world_epoch))
        ok = false;
    if (ok && !ctx_json_get_int(json, "transcript_tokens", &m->transcript_tokens))
        ok = false;
    if (ok && !ctx_json_get_string(json, "kv_path", &m->kv_file))
        m->kv_file = ds4_agent_context_file_name(m->id, ".kv");
    if (ok && !ctx_json_get_string(json, "memory_path", &m->memory_file))
        m->memory_file = ds4_agent_context_file_name(m->id, ".memory.md");
    if (ok && !ds4_agent_context_file_component_safe(m->kv_file)) ok = false;
    if (ok && m->memory_file && m->memory_file[0] &&
        !ds4_agent_context_file_component_safe(m->memory_file)) ok = false;
    if (!ok) {
        ctx_set_err(err, err_len, "invalid context metadata: %s", path);
        ds4_agent_context_meta_free(m);
    }
    free(json);
    return ok;
}

bool ds4_agent_context_meta_filename(const char *name) {
    size_t n = strlen(name);
    static const char suffix[] = ".meta.json";
    size_t s = sizeof(suffix) - 1;
    return n > s && !strcmp(name + n - s, suffix);
}

int ds4_agent_context_count_checkpoints(const char *context_dir) {
    DIR *d = opendir(context_dir);
    if (!d) return 0;
    int count = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (ds4_agent_context_meta_filename(de->d_name)) count++;
    }
    closedir(d);
    return count;
}

uint64_t ds4_agent_context_max_world_epoch(const char *context_dir) {
    DIR *d = opendir(context_dir);
    if (!d) return 0;
    uint64_t max_epoch = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (!ds4_agent_context_meta_filename(de->d_name)) continue;
        char *meta_path = ds4_agent_context_path_for_file(context_dir, de->d_name);
        ds4_agent_context_meta m = {0};
        char err[160] = {0};
        if (ds4_agent_context_read_meta_file(meta_path, &m, err, sizeof(err)) &&
            m.world_epoch > max_epoch)
            max_epoch = m.world_epoch;
        ds4_agent_context_meta_free(&m);
        free(meta_path);
    }
    closedir(d);
    return max_epoch;
}

char *ds4_agent_context_full_kv_path(const char *context_dir,
                                     const ds4_agent_context_meta *m) {
    return ds4_agent_context_path_for_file(context_dir, m->kv_file);
}

char *ds4_agent_context_full_memory_path(const char *context_dir,
                                         const ds4_agent_context_meta *m) {
    if (!m->memory_file || !m->memory_file[0]) return NULL;
    return ds4_agent_context_path_for_file(context_dir, m->memory_file);
}

bool ds4_agent_context_find_checkpoint(const char *context_dir,
                                       const char *prefix,
                                       ds4_agent_context_meta *found,
                                       char **meta_path_out,
                                       char **kv_path_out,
                                       char *err, size_t err_len) {
    if (!prefix || !prefix[0]) {
        ctx_set_err(err, err_len, "context id is required");
        return false;
    }
    size_t prefix_len = strlen(prefix);
    DIR *d = opendir(context_dir);
    if (!d) {
        ctx_set_err(err, err_len, "no context checkpoints found");
        return false;
    }
    bool matched = false;
    bool ambiguous = false;
    ds4_agent_context_meta best = {0};
    char *best_meta_path = NULL;
    char *best_kv_path = NULL;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (!ds4_agent_context_meta_filename(de->d_name)) continue;
        char *meta_path = ds4_agent_context_path_for_file(context_dir, de->d_name);
        ds4_agent_context_meta m = {0};
        char parse_err[160] = {0};
        if (!ds4_agent_context_read_meta_file(meta_path, &m, parse_err,
                                              sizeof(parse_err))) {
            free(meta_path);
            continue;
        }
        if (!strncmp(m.id, prefix, prefix_len)) {
            if (matched) {
                ambiguous = true;
                ds4_agent_context_meta_free(&m);
                free(meta_path);
                break;
            }
            matched = true;
            best = m;
            best_meta_path = meta_path;
            best_kv_path = ds4_agent_context_full_kv_path(context_dir, &best);
        } else {
            ds4_agent_context_meta_free(&m);
            free(meta_path);
        }
    }
    closedir(d);
    if (ambiguous) {
        ds4_agent_context_meta_free(&best);
        free(best_meta_path);
        free(best_kv_path);
        ctx_set_err(err, err_len, "context id prefix is ambiguous: %s", prefix);
        return false;
    }
    if (!matched) {
        ctx_set_err(err, err_len, "context checkpoint not found: %s", prefix);
        return false;
    }
    *found = best;
    if (meta_path_out) *meta_path_out = best_meta_path; else free(best_meta_path);
    if (kv_path_out) *kv_path_out = best_kv_path; else free(best_kv_path);
    return true;
}

void ds4_agent_side_effects_free(ds4_agent_side_effects *effects) {
    if (!effects) return;
    ds4_agent_side_effect *e = effects->head;
    while (e) {
        ds4_agent_side_effect *next = e->next;
        free(e->kind);
        free(e->detail);
        free(e);
        e = next;
    }
    effects->head = NULL;
    effects->count = 0;
    effects->evicted_count = 0;
    effects->latest_evicted_epoch = 0;
}

uint64_t ds4_agent_side_effects_note(ds4_agent_side_effects *effects,
                                     uint64_t current_epoch,
                                     const char *kind,
                                     const char *detail) {
    if (!effects) return current_epoch;
    uint64_t next_epoch = current_epoch == UINT64_MAX ? current_epoch : current_epoch + 1;
    ds4_agent_side_effect *e = ctx_xmalloc(sizeof(*e));
    memset(e, 0, sizeof(*e));
    e->epoch = next_epoch;
    e->kind = ctx_xstrdup(kind && kind[0] ? kind : "tool");
    e->detail = ctx_xstrdup(detail && detail[0] ? detail : "");
    e->next = effects->head;
    effects->head = e;
    effects->count++;

    while (effects->count > DS4_AGENT_CONTEXT_MAX_SIDE_EFFECTS) {
        ds4_agent_side_effect **link = &effects->head;
        while (*link && (*link)->next) link = &(*link)->next;
        if (!*link) break;
        ds4_agent_side_effect *old = *link;
        *link = NULL;
        effects->evicted_count++;
        if (old->epoch > effects->latest_evicted_epoch)
            effects->latest_evicted_epoch = old->epoch;
        free(old->kind);
        free(old->detail);
        free(old);
        effects->count--;
    }

    return next_epoch;
}

char *ds4_agent_side_effects_summary_since(const ds4_agent_side_effects *effects,
                                           uint64_t epoch) {
    ds4_agent_context_buf b = {0};
    if (effects && effects->latest_evicted_epoch > epoch) {
        char line[256];
        snprintf(line, sizeof(line),
                 "Known side effects after checkpoint may be incomplete: "
                 "%" PRIu64 " older side effect(s) were dropped from memory "
                 "up to epoch=%" PRIu64 ".\n",
                 effects->evicted_count, effects->latest_evicted_epoch);
        ctx_buf_puts(&b, line);
    }
    int shown = 0;
    for (const ds4_agent_side_effect *e = effects ? effects->head : NULL; e; e = e->next) {
        if (e->epoch <= epoch) continue;
        if (shown == 0) ctx_buf_puts(&b, "Known side effects after checkpoint:\n");
        char *detail = ds4_agent_context_oneline(e->detail, 180);
        char line[320];
        snprintf(line, sizeof(line), "- epoch=%" PRIu64 " %s %s\n",
                 e->epoch, e->kind ? e->kind : "tool", detail);
        ctx_buf_puts(&b, line);
        free(detail);
        shown++;
        if (shown >= 8) {
            ctx_buf_puts(&b, "- ... more side effects omitted ...\n");
            break;
        }
    }
    return ctx_buf_take(&b);
}

bool ds4_agent_context_no_running_bash_guard(const char *action,
                                             int running_bash_jobs,
                                             char *err,
                                             size_t err_len) {
    if (running_bash_jobs <= 0) return true;
    ctx_set_err(err, err_len,
                "context %s denied because %d bash job(s) are still running; "
                "use bash_status or bash_stop first",
                action && action[0] ? action : "operation",
                running_bash_jobs);
    return false;
}

bool ds4_agent_context_restore_epoch_guard(uint64_t current_epoch,
                                           uint64_t checkpoint_epoch,
                                           bool allow_side_effect_mismatch,
                                           char *err,
                                           size_t err_len) {
    if (current_epoch == checkpoint_epoch || allow_side_effect_mismatch)
        return true;
    ctx_set_err(err, err_len,
                "restore would rewind model context from world_epoch=%" PRIu64
                " to %" PRIu64 ", but external side effects may still exist. "
                "Revert or inspect those effects, or call context restore with "
                "allow_side_effect_mismatch=true.",
                current_epoch, checkpoint_epoch);
    return false;
}

char *ds4_agent_context_restore_expected_metrics_line(
        const ds4_agent_context_restore_metrics *metrics) {
    ds4_agent_context_buf b = {0};
    char line[384];
    int checkpoint_tokens = metrics ? metrics->checkpoint_tokens : 0;
    int notice_tokens = metrics ? metrics->restore_notice_tokens : 0;
    int restored_tokens = metrics ? metrics->restored_tokens : 0;
    if (checkpoint_tokens < 0) checkpoint_tokens = 0;
    if (notice_tokens < 0) notice_tokens = 0;
    if (restored_tokens < 0) restored_tokens = 0;
    snprintf(line, sizeof(line),
             "KV restore expected metrics: checkpoint_tokens=%d expected_restore_notice_tokens=%d expected_restored_tokens=%d expected_prefill_suffix_tokens=%d expected_full_prefill_tokens_without_kv=%d expected_saved_prefill_tokens=%d.\n",
             checkpoint_tokens, notice_tokens, restored_tokens,
             notice_tokens, restored_tokens, checkpoint_tokens);
    ctx_buf_puts(&b, line);
    return ctx_buf_take(&b);
}
