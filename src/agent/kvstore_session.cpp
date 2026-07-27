#include "ds4_agent_internal.h"



/* ============================================================================
 * Agent KV Store And Session Persistence
 * ============================================================================
 */

char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes);



/* Agent sessions deliberately use a different policy from ds4-server:
 *
 * - sysprompt.kv is a fixed bootstrap checkpoint for the current tool/system
 *   prompt.  Because its name is fixed, the current rendered text is compared
 *   with the text stored in the file before loading.  A mismatch simply rebuilds
 *   and overwrites the file.
 * - conversation sessions are explicit saves only.  Their stable file name is
 *   SHA1(title || created_at_le64).kv, where title is the first user prompt and
 *   created_at is preserved across future saves.  The title is stored in an
 *   agent-only trailer after the KV payload.
 *
 * The DS4 payload stores the exact token sequence and graph state.  The rendered
 * text is retained for listing, history rendering, and stripped-session rebuilds. */
bool agent_kv_read_text(FILE *fp, uint32_t text_bytes,
                               char **text_out, char *err, size_t err_len) {
    char *text = (char *)agent_xmalloc((size_t)text_bytes + 1);
    if (fread(text, 1, text_bytes, fp) != text_bytes) {
        if (err && err_len) snprintf(err, err_len, "truncated cached text");
        free(text);
        return false;
    }
    text[text_bytes] = '\0';
    *text_out = text;
    return true;
}



bool agent_kv_write_title_trailer(FILE *fp, const char *title,
                                         char *err, size_t err_len) {
    size_t title_len = title ? strlen(title) : 0;
    if (title_len > UINT32_MAX) {
        snprintf(err, err_len, "agent session title is too large");
        return false;
    }
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)title_len);
    return fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
           fwrite(title ? title : "", 1, title_len, fp) == title_len;
}



/* Read the optional agent title trailer without disturbing the payload cursor.
 * The caller is positioned just after rendered text, which is also the payload
 * start expected by ds4_session_load_payload(). */
bool agent_kv_read_title_trailer(FILE *fp, const ds4_kvstore_entry *hdr,
                                        char **title_out,
                                        char *err, size_t err_len) {
    off_t payload_pos = ftello(fp);
    if (payload_pos < 0) {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }
    if (hdr->payload_bytes > (uint64_t)LLONG_MAX ||
        fseeko(fp, (off_t)hdr->payload_bytes, SEEK_CUR) != 0)
    {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }

    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) {
        if (err && err_len) snprintf(err, err_len, "missing agent session title trailer");
        fseeko(fp, payload_pos, SEEK_SET);
        return false;
    }
    uint32_t title_bytes = ds4_kvstore_le_get32(tb);
    char *title = (char *)agent_xmalloc((size_t)title_bytes + 1);
    if (fread(title, 1, title_bytes, fp) != title_bytes) {
        if (err && err_len) snprintf(err, err_len, "truncated agent session title trailer");
        free(title);
        fseeko(fp, payload_pos, SEEK_SET);
        return false;
    }
    title[title_bytes] = '\0';
    if (fseeko(fp, payload_pos, SEEK_SET) != 0) {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        free(title);
        return false;
    }
    *title_out = title;
    return true;
}



void agent_kv_identity_sha(const ds4_kvstore_entry *hdr,
                                  const char *text, uint32_t text_bytes,
                                  const char *title,
                                  char sha_out[41]) {
    if (hdr->ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE) {
        agent_session_identity_sha(title ? title : "", hdr->created_at, sha_out);
    } else {
        ds4_kvstore_sha1_bytes_hex(text, text_bytes, sha_out);
    }
}



/* Load a KV file and optionally verify either its session identity or exact
 * rendered text.  sysprompt.kv uses exact text because the file name is fixed;
 * saved sessions use their filename SHA: modern agent sessions hash the title
 * trailer plus created_at, while legacy sessions still hash rendered text. */
bool agent_kv_load_path(agent_worker *w, const char *path,
                               const char *expected_sha,
                               const char *expected_text,
                               size_t expected_text_len,
                               ds4_tokens *loaded_tokens,
                               agent_kv_session_meta *meta_out,
                               char *err, size_t err_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    if (!ok) snprintf(err, err_len, "invalid KV header");

    char *text = NULL;
    if (ok) ok = agent_kv_read_text(fp, text_bytes, &text, err, err_len);
    char *title = NULL;
    bool has_title = ok && (hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE);
    if (has_title)
        ok = agent_kv_read_title_trailer(fp, &hdr, &title, err, err_len);
    uint32_t expected_tokens = hdr.tokens;
    if (ok && hdr.payload_bytes != 0 &&
        hdr.model_id != (uint8_t)ds4_engine_model_id(w->engine))
    {
        snprintf(err, err_len, "KV checkpoint was written for a different model");
        ok = false;
    }
    if (ok && hdr.payload_bytes != 0 &&
        hdr.quant_bits != (uint8_t)ds4_engine_routed_quant_bits(w->engine))
    {
        snprintf(err, err_len, "KV checkpoint was written for a different quantization");
        ok = false;
    }
    if (ok && expected_text) {
        if ((size_t)text_bytes != expected_text_len ||
            memcmp(text, expected_text, expected_text_len) != 0)
        {
            snprintf(err, err_len, "cached text does not match current system prompt");
            ok = false;
        }
    }
    if (ok && expected_sha) {
        char actual_sha[41];
        agent_kv_identity_sha(&hdr, text, text_bytes, title, actual_sha);
        if (strcmp(actual_sha, expected_sha)) {
            snprintf(err, err_len, "cached session identity does not match file name");
            ok = false;
        }
    }

    char load_err[160] = {0};
    if (ok && hdr.payload_bytes == 0) {
        ds4_tokens rebuilt = {0};
        ds4_tokenize_rendered_chat(w->engine, text, &rebuilt);
        expected_tokens = (uint32_t)rebuilt.len;
        if (agent_worker_sync_tokens(w, &rebuilt, true, err, err_len) != 0) {
            ds4_session_invalidate(w->session);
            ok = false;
        }
        ds4_tokens_free(&rebuilt);
    } else if (ok &&
               ds4_session_load_payload(w->session, fp, hdr.payload_bytes,
                                        load_err, sizeof(load_err)) != 0)
    {
        snprintf(err, err_len, "%s", load_err[0] ? load_err : "failed to load KV payload");
        ds4_session_invalidate(w->session);
        ok = false;
    }
    fclose(fp);

    if (ok) {
        const ds4_tokens *live = ds4_session_tokens(w->session);
        if (!live || live->len != (int)expected_tokens) {
            snprintf(err, err_len, "KV payload token count mismatch");
            ds4_session_invalidate(w->session);
            ok = false;
        } else if (loaded_tokens) {
            ds4_tokens_free(loaded_tokens);
            ds4_tokens_copy(loaded_tokens, live);
        }
        if (meta_out) {
            agent_kv_session_meta_free(meta_out);
            meta_out->has_title_trailer = has_title;
            meta_out->legacy_identity = !has_title;
            meta_out->created_at = hdr.created_at;
            agent_kv_identity_sha(&hdr, text, text_bytes, title, meta_out->sha);
            meta_out->title = has_title ?
                xstrdup(title) :
                agent_session_title_from_text(text, text_bytes, 0);
        }
    }
    free(title);
    free(text);
    return ok;
}



/* Save the current live KV under the rendered transcript identity.  The caller
 * decides the policy: fixed sysprompt path or SHA-named session path. */
static bool agent_kv_save_path(agent_worker *w, const char *path,
                               const ds4_tokens *tokens,
                               const char *reason,
                               char sha_out[41],
                               const char *session_title,
                               uint64_t session_created_at,
                               char *err, size_t err_len) {
    const ds4_tokens *live = ds4_session_tokens(w->session);
    if (!agent_tokens_equal(live, tokens)) {
        snprintf(err, err_len, "live KV state does not match session transcript");
        return false;
    }
    const int quant_bits = ds4_engine_routed_quant_bits(w->engine);
    if (quant_bits != 2 && quant_bits != 4) {
        snprintf(err, err_len, "unsupported routed quantization for KV save");
        return false;
    }
    const int model_id = ds4_engine_model_id(w->engine);

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, tokens, &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render KV text key");
        return false;
    }
    if (text_len > UINT32_MAX) {
        snprintf(err, err_len, "rendered KV text key is too large");
        free(text);
        return false;
    }
    const bool session_identity = session_title != NULL;
    uint64_t now = (uint64_t)time(NULL);
    uint64_t created_at = session_identity && session_created_at ?
        session_created_at : now;
    char sha[41];
    if (session_identity)
        agent_session_identity_sha(session_title, created_at, sha);
    else
        ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    if (sha_out) memcpy(sha_out, sha, sizeof(sha));

    ds4_session_payload_file staged = {0};
    char save_err[160] = {0};
    if (ds4_session_stage_payload(w->session, &staged,
                                  save_err, sizeof(save_err)) != 0) {
        snprintf(err, err_len, "%s",
                 save_err[0] ? save_err : "session has no valid KV payload");
        free(text);
        return false;
    }
    uint64_t payload_bytes = staged.bytes;

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        return false;
    }

    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        return false;
    }

    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, (uint8_t)model_id, (uint8_t)quant_bits,
                            ds4_kvstore_reason_code(reason),
                            session_identity ? DS4_KVSTORE_EXT_SESSION_TITLE : 0,
                            (uint32_t)tokens->len, 0,
                            (uint32_t)ds4_session_ctx(w->session),
                            created_at, now, payload_bytes);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);

    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              ds4_session_write_staged_payload(&staged, fp,
                                               save_err, sizeof(save_err)) == 0 &&
              (!session_identity ||
               agent_kv_write_title_trailer(fp, session_title,
                                            save_err, sizeof(save_err))) &&
              fflush(fp) == 0;
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
        snprintf(err, err_len, "%s",
                 saved_errno ? strerror(saved_errno) :
                 (save_err[0] ? save_err : "failed to write KV file"));
        unlink(tmp);
    }

    ds4_session_payload_file_free(&staged);
    free(tmp);
    free(text);
    return ok;
}



void agent_worker_build_system_tokens(agent_worker *w, ds4_tokens *out) {
    ds4_chat_begin(w->engine, out);
    if (w->cfg->gen.think_mode == DS4_THINK_MAX &&
        effective_think_mode(w->cfg) == DS4_THINK_MAX)
        ds4_chat_append_max_effort_prefix(w->engine, out);
    agent_append_system_prompt(w->engine, out, w->cfg->gen.system);
}



void agent_publish_system_status(agent_worker *w, const char *msg) {
    if (w->cfg->non_interactive) return;
    if (isatty(STDOUT_FILENO)) {
        static const char marker[] = "\x1b[33m✦ \x1b[38;5;218m";
        agent_publish(w, marker, sizeof(marker) - 1);
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\x1b[0m\n", strlen("\x1b[0m\n"));
    } else {
        agent_publish(w, "✦ ", strlen("✦ "));
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\n", 1);
    }
}



void agent_publishf_system_status(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) {
        agent_publish_system_status(w, stack);
        return;
    }

    char *heap = (char *)agent_xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish_system_status(w, heap);
    free(heap);
}






/* When a model turn finishes with a tool call, queued user messages should not
 * preempt that tool.  The worker asks the UI thread for the queue contents only
 * after the tool result is appended, so the next model input can contain both
 * the tool observation and the user's pending correction. */
char *worker_request_queued_user_drain(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->queued_user_drain_pending = true;
    w->queued_user_drain_answered = false;
    free(w->queued_user_drain_text);
    w->queued_user_drain_text = NULL;
    agent_wake_locked(w);
    pthread_cond_signal(&w->cond);
    while (!w->stop && !w->queued_user_drain_answered)
        pthread_cond_wait(&w->cond, &w->mu);
    char *text = w->queued_user_drain_text;
    w->queued_user_drain_text = NULL;
    w->queued_user_drain_pending = false;
    w->queued_user_drain_answered = false;
    pthread_mutex_unlock(&w->mu);
    return text;
}



bool worker_take_queued_user_drain_request(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool pending = w->queued_user_drain_pending;
    if (pending) w->queued_user_drain_pending = false;
    pthread_mutex_unlock(&w->mu);
    return pending;
}



void worker_answer_queued_user_drain(agent_worker *w, char *text) {
    pthread_mutex_lock(&w->mu);
    free(w->queued_user_drain_text);
    w->queued_user_drain_text = text;
    w->queued_user_drain_answered = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



/* Synchronize the live DS4 session to a transcript.  This is the agent's main
 * cache-saving operation: if the requested transcript extends the live session,
 * only the suffix is prefetched; otherwise the DS4 session rebuilds from the
 * longest common prefix it can retain. */
int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len) {
    int old_pos = ds4_session_pos(w->session);
    int common = ds4_session_common_prefix(w->session, tokens);
    int cached = common == old_pos && tokens->len >= old_pos ? common : 0;
    int suffix = tokens->len - cached;
    if (suffix < 0) suffix = tokens->len;

    if (publish_progress) {
        pthread_mutex_lock(&w->mu);
        unsigned prefill_label = w->status.state == AGENT_WORKER_PREFILL ?
            w->status.prefill_label : agent_next_prefill_label();
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->progress_started_at = agent_now_sec();
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_label = prefill_label;
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }

    ds4_session_set_progress(w->session, publish_progress ? worker_progress_cb : NULL,
                             publish_progress ? w : NULL);
    ds4_session_set_display_progress(w->session,
                                     publish_progress ? worker_progress_cb : NULL,
                                     publish_progress ? w : NULL);
    ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
    int rc = ds4_session_sync(w->session, tokens, err, err_len);
    ds4_session_set_cancel(w->session, NULL, NULL);
    ds4_session_set_progress(w->session, NULL, NULL);
    ds4_session_set_display_progress(w->session, NULL, NULL);
    return rc;
}



/* Start a new session at the system/tool prompt.  A fixed sysprompt.kv
 * checkpoint avoids paying this prefill cost repeatedly, but only when the
 * rendered prompt text still matches the file.  The same fixed path is shared
 * by Flash and Pro; agent_kv_load_path() checks the model id, so switching
 * model families rebuilds this cache instead of restoring incompatible KV. */
bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len) {
    ds4_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &sys, &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render system prompt");
        ds4_tokens_free(&sys);
        return false;
    }

    bool loaded = false;
    char load_err[160] = {0};
    if (w->sysprompt_path) {
        loaded = agent_kv_load_path(w, w->sysprompt_path, NULL,
                                    text, text_len, &w->transcript,
                                    NULL,
                                    load_err, sizeof(load_err));
        if (loaded) {
            agent_trace(w, "sysprompt kv hit file=%s tokens=%d",
                        w->sysprompt_path, w->transcript.len);
        }
    }

    if (!loaded) {
        if (w->sysprompt_path)
            agent_publish_system_status(w, "Updating system prompt cache...");
        ds4_tokens_free(&w->transcript);
        ds4_tokens_copy(&w->transcript, &sys);
        if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) {
            free(text);
            ds4_tokens_free(&sys);
            return false;
        }
        if (w->sysprompt_path) {
            char save_err[160] = {0};
            char ignored_sha[41];
            if (!agent_kv_save_path(w, w->sysprompt_path, &w->transcript,
                                    "agent-system", ignored_sha,
                                    NULL, 0,
                                    save_err, sizeof(save_err)))
            {
                if (w->cfg->non_interactive) {
                    fprintf(stderr, "ds4-agent: failed to save system prompt KV: %s\n",
                            save_err);
                } else {
                    agent_buf b = {0};
                    agent_buf_puts(&b, "\nds4-agent: failed to save system prompt KV: ");
                    agent_buf_puts(&b, save_err);
                    agent_buf_puts(&b, "\n");
                    char *msg = agent_buf_take(&b);
                    agent_publish(w, msg, strlen(msg));
                    free(msg);
                }
            } else {
                agent_trace(w, "sysprompt kv stored file=%s tokens=%d",
                            w->sysprompt_path, w->transcript.len);
            }
        }
    }

    agent_worker_note_system_prompt_seen(w);
    pthread_mutex_lock(&w->mu);
    w->user_activity = false;
    w->session_dirty = false;
    w->status.state = AGENT_WORKER_IDLE;
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = false;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
    w->datetime_context_injected = false;
    agent_worker_clear_session_identity(w);
    free(text);
    ds4_tokens_free(&sys);
    return true;
}



bool agent_worker_has_user_session(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity;
    pthread_mutex_unlock(&w->mu);
    return yes;
}



bool agent_worker_needs_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity && w->session_dirty;
    pthread_mutex_unlock(&w->mu);
    return yes;
}



/* Save the current session under its stable agent identity.  The worker owns
 * the live KV, so busy /save requests are deferred until a stable append-only
 * point and then executed by the worker thread. */
bool agent_worker_save_session_now(agent_worker *w, char sha_out[41],
                                          int *tokens_out,
                                          char *err, size_t err_len) {
    if (!agent_worker_has_user_session(w)) {
        snprintf(err, err_len, "nothing to save");
        return false;
    }

    if (agent_worker_sync_tokens(w, &w->transcript, false, err, err_len) != 0)
        return false;
    if (!agent_mkdir_p(w->cache_dir)) {
        snprintf(err, err_len, "failed to create %s", w->cache_dir);
        return false;
    }

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return false;
    }
    if (!w->session_title) {
        w->session_title = agent_session_title_from_text(text, text_len, 0);
    }
    if (w->session_created_at == 0)
        w->session_created_at = (uint64_t)time(NULL);

    char sha[41];
    agent_session_identity_sha(w->session_title, w->session_created_at, sha);
    char *path = agent_kv_path_for_sha(w->cache_dir, sha);

    bool ok = agent_kv_save_path(w, path, &w->transcript,
                                 "agent-session", sha_out,
                                 w->session_title, w->session_created_at,
                                 err, err_len);
    if (ok) {
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        if (w->legacy_session_path_to_delete &&
            strcmp(w->legacy_session_path_to_delete, path) != 0)
        {
            unlink(w->legacy_session_path_to_delete);
        }
        free(w->legacy_session_path_to_delete);
        w->legacy_session_path_to_delete = NULL;
        pthread_mutex_lock(&w->mu);
        w->session_dirty = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
        if (tokens_out) *tokens_out = w->transcript.len;
    }
    free(path);
    free(text);
    return ok;
}



bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len) {
    if (!worker_is_idle(w)) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    int tokens = 0;
    bool ok = agent_worker_save_session_now(w, sha, &tokens, err, err_len);
    if (ok) printf("saved session %.8s (%d tokens)\n", sha, tokens);
    return ok;
}

