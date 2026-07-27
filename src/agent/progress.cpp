#include "pulsar_agent_internal.h"



/* ============================================================================
 * Worker Progress And Generic Buffers
 * ============================================================================
 */

void worker_progress_cb(void *ud, const char *event, int current, int total) {
    (void)total;
    agent_worker *w = (agent_worker *)ud;
    if (!w || !event) return;
    if (strcmp(event, "prefill_chunk") && strcmp(event, "prefill_display")) return;
    pthread_mutex_lock(&w->mu);
    int done = current - w->progress_base;
    if (done < 0) done = 0;
    if (done > w->status.prefill_total) done = w->status.prefill_total;
    w->status.prefill_done = done;
    double elapsed = agent_now_sec() - w->progress_started_at;
    w->status.prefill_tps =
        done > 0 && elapsed > 0.0 ? (double)done / elapsed : 0.0;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



bool worker_should_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool interrupt = w->interrupt || w->stop;
    pthread_mutex_unlock(&w->mu);
    return interrupt;
}



/* Ctrl+C is a latched request consumed by the worker.  Once an interrupted
 * operation has reached a stable append-only boundary and is about to publish
 * IDLE, the request must be acknowledged; otherwise the editor can observe an
 * idle worker with a stale interrupt still pending. */
void worker_clear_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    pthread_mutex_unlock(&w->mu);
}



bool agent_err_is_interrupted(const char *err) {
    return err && !strcmp(err, "interrupted");
}



bool worker_cancel_session_cb(void *ud) {
    return worker_should_interrupt((agent_worker *)ud);
}



void agent_buf_append(agent_buf *b, const char *s, size_t n) {
    if (!n || b->truncated) return;
    const size_t max = 128 * 1024;
    if (b->len + n > max) {
        n = max > b->len ? max - b->len : 0;
        b->truncated = true;
    }
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = (char *)agent_xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}



void agent_buf_puts(agent_buf *b, const char *s) {
    agent_buf_append(b, s, strlen(s));
}



char *agent_buf_take(agent_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}



bool agent_tokens_equal(const pulsar_tokens *a, const pulsar_tokens *b) {
    if (!a || !b || a->len != b->len) return false;
    for (int i = 0; i < a->len; i++) {
        if (a->v[i] != b->v[i]) return false;
    }
    return true;
}



bool agent_mkdir_p(const char *path) {
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



char *agent_default_cache_dir(void) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    agent_buf b = {0};
    agent_buf_puts(&b, home);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') agent_buf_puts(&b, "/");
    agent_buf_puts(&b, ".ds4/kvcache");
    return agent_buf_take(&b);
}



char *agent_kv_path_for_sha(const char *dir, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return pulsar_kvstore_path_join(dir, name);
}



static void agent_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}



/* Agent session IDs are intentionally independent from the rendered transcript:
 * once a session has a title and creation time, resaving it keeps the same file
 * name while the transcript and KV payload evolve. */
void agent_session_identity_sha(const char *title, uint64_t created_at,
                                       char sha_out[41]) {
    size_t title_len = title ? strlen(title) : 0;
    agent_buf b = {0};
    agent_buf_append(&b, title ? title : "", title_len);
    uint8_t ts[8];
    agent_le_put64(ts, created_at);
    agent_buf_append(&b, (const char *)ts, sizeof(ts));
    pulsar_kvstore_sha1_bytes_hex(b.ptr ? b.ptr : "", b.len, sha_out);
    free(b.ptr);
}



void agent_worker_clear_session_identity(agent_worker *w) {
    w->session_sha[0] = '\0';
    free(w->session_title);
    w->session_title = NULL;
    w->session_created_at = 0;
    free(w->legacy_session_path_to_delete);
    w->legacy_session_path_to_delete = NULL;
}



void agent_kv_session_meta_free(agent_kv_session_meta *m) {
    free(m->title);
    memset(m, 0, sizeof(*m));
}

