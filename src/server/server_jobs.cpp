/* Job lifecycle around ONE request (split move-only from generate.c):
 * the resumable gen_state machine (prefill -> decode -> finish), its
 * prefill-progress/keepalive plumbing, tool-checkpoint canonicalization
 * and thinking-checkpoint remembering, protocol stream emission glue,
 * and the generate_job_* driver stepped by the scheduler
 * (server_sched.c). gen_state/gen_phase live in pulsar_server_internal.h
 * because the scheduler TU steps jobs by phase and drives the batched
 * lanes through the batch_* fields. */
#include "pulsar_server_internal.h"



/* Live recovery for a tool call started inside an unclosed <think> block.
 *
 * The model sometimes opens a DSML stanza without closing its thinking first.
 * Waiting for a </think> that never comes stalls the turn: the marker is never
 * scanned as executable and the block is dropped at parse time.  Instead of
 * rewriting sampled context, recover forward: force-feed "</think>" plus a
 * blank line and let the model continue.  Measured on the real model, that
 * position predicts a fresh stanza opening so strongly that the model
 * restarts the call cleanly on the executable side of the close.  Re-emitting
 * the stanza opening ourselves was tried and is counterproductive: with the
 * dangling opening right before the close and a forced copy right after it,
 * the model reads the call as already made and ends the turn.  The dangling
 * opening stays harmlessly inside reasoning.
 *
 * Detection works on accumulated text, so the tokenization of the marker does
 * not matter, and it triggers only on a complete stanza opening: a lone "<"
 * or a partial marker keeps decoding untouched, while *scan_from holds back
 * far enough that an opening split across future tokens is still seen from
 * its first byte.  The forced text is tokenized with the rendered-chat
 * tokenizer so </think> maps to its special token.
 *
 * Returns 1 when an injection was performed (text extended, thinking closed),
 * 0 when there is nothing to do or no budget, -1 on eval failure. */
int server::chat_think_tool_recovery(session_slot *sl,
                                    buf *text,
                                    thinking_state *thinking,
                                    size_t *scan_from,
                                    int *completion,
                                    int max_tokens,
                                    char *err,
                                    size_t errlen) {
    auto *s = this;
    (void)sl; /* slot is a pure bank descriptor; the session is s->sess */
    if (!thinking->inside || !text->ptr) return 0;
    if (*scan_from > text->len) *scan_from = text->len;
    if (!find_any_tool_start(text->ptr + *scan_from)) {
        const size_t hold = 80; /* > longest stanza opening */
        *scan_from = text->len > hold ? text->len - hold : 0;
        return 0;
    }

    const char *inject = "</think>\n\n";
    const size_t inject_len = strlen(inject);
    pulsar_tokens toks = {0};
    pulsar_tokenize_rendered_chat(s->engine, inject, &toks);

    const int room = pulsar_session_ctx(s->sess) - pulsar_session_pos(s->sess);
    if (toks.len <= 0 ||
        toks.len >= room ||
        *completion + toks.len >= max_tokens) {
        /* Not enough budget to recover; leave the stream as generated and let
         * the parse-time fallback deal with it.  Skip past this marker so the
         * scan does not retry it every token. */
        pulsar_tokens_free(&toks);
        *scan_from = text->len;
        return 0;
    }

    for (int i = 0; i < toks.len; i++) {
        if (pulsar_session_eval(s->sess, toks.v[i], err, errlen) != 0) {
            pulsar_tokens_free(&toks);
            return -1;
        }
        (*completion)++;
    }
    buf_append(text, inject, inject_len);
    thinking->feed(inject, inject_len);
    *scan_from = text->len;
    pulsar_tokens_free(&toks);
    return 1;
}



bool server::append_rendered_suffix_to_live_session(session_slot *sl,
                                                   const char *suffix,
                                                   int *tokens_appended,
                                                   char *err, size_t errlen) {
    auto *s = this;
    (void)sl; /* slot is a pure bank descriptor; the session is s->sess */
    if (tokens_appended) *tokens_appended = 0;
    if (!s || !suffix || !suffix[0]) return true;
    const pulsar_tokens *live = pulsar_session_tokens(s->sess);
    if (!live) {
        if (err && errlen) snprintf(err, errlen, "live session is unavailable");
        return false;
    }

    pulsar_tokens target = {0};
    build_prompt_from_exact_prefix_and_text_suffix(s->engine, live, suffix, &target);
    const int before = pulsar_session_pos(s->sess);
    bool ok = pulsar_session_sync(s->sess, &target, err, errlen) == 0;
    if (ok && tokens_appended) {
        int delta = pulsar_session_pos(s->sess) - before;
        *tokens_appended = delta > 0 ? delta : 0;
    }
    pulsar_tokens_free(&target);
    return ok;
}



bool server::continue_after_invalid_dsml(session_slot *sl,
                                        const request *r,
                                        const thinking_state *thinking,
                                        const char *detail,
                                        int *tokens_appended,
                                        char *err, size_t errlen) {
    auto *s = this;
    char *suffix = build_invalid_dsml_tool_error_suffix(r, thinking, detail);
    bool ok = s->append_rendered_suffix_to_live_session(sl, suffix,
                                                     tokens_appended,
                                                     err, errlen);
    free(suffix);
    return ok;
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
    server_log(PULSAR_LOG_TOOL,
               "pulsar-server: tool calls ctx=%s%s%s n=%d raw_dsml=%d ids=[%s] names=[%s]",
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
    server_prefill_progress *p = (server_prefill_progress *)ud;
    if (!p || !event) return;
    const bool is_chunk = strcmp(event, "prefill_chunk") == 0;
    const bool is_display = strcmp(event, "prefill_display") == 0;
    if (!is_chunk && !is_display) return;

    double now = server_now_sec();
    /* Keep the HTTP/SSE connection alive while prefill runs.  We write the SSE
     * response headers the first time the callback fires and then emit a
     * comment line (`:` prefix, ignored by SSE clients) every few seconds.
     * Best-effort: if the client has already gone away, the writes fail
     * silently and the outer code will discover the closed socket the next
     * time it tries to stream a real event. */
    if (p->stream && p->fd >= 0 && !p->stream_failed) {
        if (!p->headers_sent) {
            p->headers_sent = true;
            if (sse_headers(p->fd, p->enable_cors)) {
                p->last_keepalive = now;
            } else {
                p->stream_failed = true;
            }
        } else if (now - p->last_keepalive >= 5.0) {
            static const char ka[] = ": prefill\n\n";
            if (send_all(p->fd, ka, sizeof(ka) - 1)) {
                p->last_keepalive = now;
            } else {
                p->stream_failed = true;
            }
        }
    }
    if (is_display) return;
    double elapsed = now - p->t0;
    if (p->seen && current == p->last_current) {
        if (p->srv && p->slot && current > p->cached_tokens) {
            p->srv->kv_cache_maybe_store_continued(p->slot);
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
    server_log(PULSAR_LOG_PREFILL,
               "pulsar-server: %s ctx=%s%s%s %s chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs",
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
    if (p->srv && p->slot && current > p->cached_tokens) {
        p->srv->kv_cache_maybe_store_continued(p->slot);
    }
}



void server::send_prefill_failure_response(const job *j,
                                          const server_prefill_progress *progress,
                                          const char *ctx, const char *flags,
                                          const char *err) {
    auto *s = this;
    const char *kind = j->req.kind == REQ_CHAT ? "chat" : "completion";
    if (j->req.stream && progress && progress->headers_sent) {
        if (progress->stream_failed) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s%s%s prefill failed after stream closed: %s",
                       kind, ctx, flags && flags[0] ? " " : "",
                       flags && flags[0] ? flags : "", err);
            return;
        }
        if (!sse_error_event(j->fd, &j->req, err)) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s%s%s prefill SSE error failed: %s",
                       kind, ctx, flags && flags[0] ? " " : "",
                       flags && flags[0] ? flags : "", err);
        }
        return;
    }
    http_error(j->fd, s->enable_cors, 500, err);
}



void server::remember_thinking_checkpoint(session_slot *sl,
                                         const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content) {
    auto *s = this;
    /* The key must byte-match what render_chat_prompt_text emits for this turn
     * once it becomes historical on the next request.  With tools advertised
     * (tool_context) a stripped historical assistant turn renders
     * "<think></think>{content}<eos>", so keep prompt_text's trailing "<think>"
     * and append the empty-reasoning, no-calls suffix.  Without tools the turn
     * renders "</think>{content}<eos>" (no "<think>") — the toolless form. */
    char *visible = NULL;
    if (j->req.has_tools) {
        if (!j->req.prompt_text || !pulsar_think_mode_enabled(j->req.think_mode))
            return;
        char *suffix = build_tool_checkpoint_suffix(&j->req, content, "", NULL);
        buf b = {0};
        buf_puts(&b, j->req.prompt_text);
        buf_puts(&b, suffix);
        free(suffix);
        visible = buf_take(&b);
    } else {
        visible = build_toolless_thinking_visible_text(&j->req, content);
    }
    if (!visible) return;

    s->thinking_live_remember(sl, visible);
    server_log(PULSAR_LOG_KVCACHE,
               "pulsar-server: thinking live checkpoint remembered ctx=%s live=%d visible=%zu",
               ctx, pulsar_session_pos(s->sess), strlen(visible));
    s->trace_event(trace_id,
                "thinking live checkpoint remembered: live=%d visible=%zu",
                pulsar_session_pos(s->sess), strlen(visible));
    free(visible);
}

/* Tool-call finish WITH thinking on: the model emitted <think>reasoning</think>
 * before the DSML tool call, so the reasoning tokens sit in the live KV.  We
 * remember the exact bytes the NEXT request will render for this turn as a visible
 * key, keeping the live tokens (reasoning included) as the sampled frontier.  The
 * next request byte-matches the key and continues from live KV — no rewrite, no
 * rebuild — and, critically, an evicted-then-reloaded checkpoint is keyed by that
 * same visible transcript on disk (kv_cache_store_current).
 *
 * render_chat_prompt_text ALWAYS re-renders the reasoning inside <think>…</think>
 * for a tool-context turn (prompt_render.c: `tool_context || i > last_user_idx`),
 * because agentic clients (opencode et al.) replay reasoning_content verbatim so
 * the model keeps its chain of thought across tool rounds.  So the key MUST carry
 * the reasoning too — an earlier version dropped it (<think></think>), which byte-
 * diverges from every reasoning-preserving replay at the first reasoning byte and
 * made the live alias AND the disk key miss, forcing a full cold re-prefill of the
 * whole conversation on eviction (opencode's ~4-minute-per-message symptom).  The
 * toolless thinking path (remember_thinking_checkpoint) still strips: it only fires
 * for non-tool-context requests (should_remember_thinking_checkpoint bails when
 * prompt_preserves_reasoning), i.e. clients that DO drop reasoning on replay. */
void server::remember_tool_thinking_checkpoint(session_slot *sl,
                                              const job *j, const char *ctx,
                                              uint64_t trace_id, const char *content,
                                              const char *reasoning,
                                              const tool_calls *calls) {
    auto *s = this;
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;
    if (!pulsar_think_mode_enabled(j->req.think_mode)) return;

    /* Visible key = prompt_text (ends "<｜Assistant｜><think>") + reasoning-preserved
     * suffix "{reasoning}</think>{content}{DSML}<EOS>" — byte-identical to what
     * render_chat_prompt_text emits for this tool turn on the next request (DSML from
     * the id-keyed raw sample via tool_memory, reasoning replayed verbatim) and to
     * the sampled bytes already in the live KV. */
    char *suffix = build_tool_checkpoint_suffix(&j->req, content,
                                                reasoning ? reasoning : "", calls);
    buf visible = {0};
    buf_puts(&visible, j->req.prompt_text);
    buf_puts(&visible, suffix);
    if (visible.ptr) {
        s->thinking_live_remember(sl, visible.ptr);
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: tool thinking checkpoint remembered ctx=%s live=%d visible=%zu",
                   ctx, pulsar_session_pos(s->sess), visible.len);
        s->trace_event(trace_id,
                    "tool thinking checkpoint remembered: live=%d visible=%zu",
                    pulsar_session_pos(s->sess), visible.len);
    }
    free(suffix);
    buf_free(&visible);
}



/* After a successful tool-call finish, make the live checkpoint match what the
 * next request will render.  Usually that is just the exact DSML remembered by
 * tool id.  If a client sends a tool call without an id we know, the fallback
 * renderer still builds valid DSML from JSON, and this function either rewrites
 * the short suffix in place or reloads an older disk checkpoint before replay. */
void server::canonicalize_tool_checkpoint(session_slot *sl,
                                         const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content,
                                         const char *reasoning, const tool_calls *calls) {
    auto *s = this;
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;

    char *suffix_text = build_tool_checkpoint_suffix(&j->req, content, reasoning, calls);

    buf rendered = {0};
    buf_puts(&rendered, j->req.prompt_text);
    buf_puts(&rendered, suffix_text);

    pulsar_tokens canonical = {0};
    pulsar_tokenize_rendered_chat(s->engine, rendered.ptr ? rendered.ptr : "", &canonical);
    const int live_len = pulsar_session_pos(s->sess);
    const int common = pulsar_session_common_prefix(s->sess, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    size_t live_text_len;
    live_text_len = 0;
    char *live_text;
    live_text = render_tokens_text(s->engine, pulsar_session_tokens(s->sess), &live_text_len);
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
        s->trace_event(trace_id,
                    "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d",
                    common, j->req.prompt.len, live_len, canonical.len);
        goto done;
    }

    char err[160];
    memset(err, 0, sizeof(err));
    pulsar_session_rewrite_result rr;
    rr = pulsar_session_rewrite_from_common(s->sess, &canonical, common,
                                         err, sizeof(err));
    if (rr == PULSAR_SESSION_REWRITE_OK) {
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d",
                   ctx, common, live_len, canonical.len);
        s->trace_event(trace_id,
                    "tool checkpoint canonicalized: common=%d live=%d canonical=%d",
                    common, live_len, canonical.len);
    } else if (rr == PULSAR_SESSION_REWRITE_REBUILD_NEEDED) {
        /* The generated DSML suffix and the canonical prompt share a prefix,
         * but the generated tail is too large to overwrite safely inside the
         * live raw-window ring.  Prefer an older disk checkpoint over replaying
         * a very long conversation from token zero. */
        char *path = NULL;
        pulsar_tokens effective = {0};
        int loaded = s->kv_cache_try_load_text(sl, rendered.ptr ? rendered.ptr : "",
                                            &effective, &path, NULL, false);
        if (loaded == 0) pulsar_session_invalidate(s->sess);

        char sync_err[160] = {0};
        const pulsar_tokens *sync_prompt = loaded > 0 ? &effective : &canonical;
        char rebuild_ctx[48];
        request_ctx_span(rebuild_ctx, sizeof(rebuild_ctx), loaded, sync_prompt->len);
        int replay_tokens = sync_prompt->len - loaded;
        if (replay_tokens < 0) replay_tokens = sync_prompt->len;
        int canonical_tail_tokens = canonical.len - common;
        if (canonical_tail_tokens < 0) canonical_tail_tokens = canonical.len;
        int discarded_live_tokens = live_len - common;
        if (discarded_live_tokens < 0) discarded_live_tokens = 0;
        const char *source = loaded > 0 ? "disk" : "full";
        const double rebuild_t0 = server_now_sec();
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: tool checkpoint canonicalization needs %d tokens rebuild ctx=%s request_ctx=%s reason=canonical-tail-rewrite tail=%d discard=%d common=%d live=%d target=%d cached=%d source=%s%s%s",
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
            .slot = sl,
            .kind = j->req.kind,
            .prompt_tokens = sync_prompt->len,
            .cached_tokens = loaded,
            .phase = "tool checkpoint rebuild",
            .has_tools = j->req.has_tools,
            .t0 = rebuild_t0,
            .fd = j->fd,
            .stream = j->req.stream,
            .enable_cors = s->enable_cors,
            /* Tool checkpoint rebuild only runs after the response stream is
             * already in flight, so the SSE headers were sent long ago.
             * Pre-arm the flag so the progress callback only emits keepalive
             * comments and never tries to write a second set of headers. */
            .headers_sent = true,
        };
        snprintf(rebuild_progress.ctx, sizeof(rebuild_progress.ctx), "%s", rebuild_ctx);
        pulsar_session_set_progress(s->sess, server_progress_cb, &rebuild_progress);
        pulsar_session_set_display_progress(s->sess, server_progress_cb, &rebuild_progress);
        if (pulsar_session_sync(s->sess, sync_prompt, sync_err, sizeof(sync_err)) == 0) {
            pulsar_session_set_progress(s->sess, NULL, NULL);
            pulsar_session_set_display_progress(s->sess, NULL, NULL);
            const double rebuild_sec = server_now_sec() - rebuild_t0;
            if (loaded > 0) {
                server_log(PULSAR_LOG_KVCACHE,
                           "pulsar-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=disk cached=%d replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, loaded, replay_tokens, canonical.len, rebuild_sec);
                s->trace_event(trace_id,
                            "tool checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s",
                            common, live_len, canonical.len, loaded, path ? path : "");
            } else {
                server_log(PULSAR_LOG_KVCACHE,
                           "pulsar-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=full cached=0 replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, replay_tokens, canonical.len, rebuild_sec);
                s->trace_event(trace_id,
                            "tool checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s",
                            common, live_len, canonical.len, err);
            }
        } else {
            pulsar_session_set_progress(s->sess, NULL, NULL);
            pulsar_session_set_display_progress(s->sess, NULL, NULL);
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: tool checkpoint rebuild failed ctx=%s request_ctx=%s source=%s cached=%d replay=%d target=%d error=\"%s\"",
                       rebuild_ctx, ctx, source, loaded, replay_tokens,
                       canonical.len, sync_err);
            s->trace_event(trace_id, "tool checkpoint canonicalization failed after rebuild request: %s", sync_err);
        }
        pulsar_tokens_free(&effective);
        free(path);
    } else {
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        s->trace_event(trace_id, "tool checkpoint canonicalization failed: %s", err);
    }

done:
    pulsar_tokens_free(&canonical);
    buf_free(&rendered);
    free(suffix_text);
}



/* =========================================================================
 * Resumable per-slot generation (multi-session increment 2).
 * =========================================================================
 *
 * The old run-to-completion generate_job() is restructured into a state
 * machine the worker steps in bounded quanta:
 *
 *   GEN_PREFILL_COLD -> GEN_PREFILL_MAIN -> GEN_DECODE_INIT -> GEN_DECODE
 *        (one engine chunk per quantum)         (K tokens per quantum)
 *                                                       |
 *                              GEN_DONE <- GEN_FINISH <-+
 *                                              |
 *                (tool-error recovery, the old goto decode_again)
 *                                              v
 *                                       GEN_DECODE_INIT
 *
 * Everything that must survive across quanta lives in gen_state, hung off the
 * slot. A prefill quantum uses the engine's own chunk boundaries: a cancel
 * callback interrupts pulsar_session_sync() after one completed chunk (only when
 * pulsar_session_prefill_quantum_min_suffix() says resumption is bit-exact) and
 * the next quantum re-issues the sync, which resumes from the checkpoint.
 * A decode quantum runs the sampling loop for at most
 * PULSAR_SERVER_DECODE_QUANTUM_TOKENS tokens. Between quanta the session is not
 * touched, so the spec-decode carry (spec_carry_* in session.c) and sampling
 * rng stay valid; with one slot the quanta run back-to-back and the output is
 * byte-identical to the old function.
 *
 * Bounded exceptions that INTENTIONALLY stay run-to-completion inside one
 * quantum (kept in increment 3's scheduler by design): the tool-error
 * recovery syncs (short model-visible suffix append) and
 * canonicalize_tool_checkpoint's rebuild in GEN_FINISH. Both are rare repair
 * paths that must observe a consistent session frontier; a co-scheduled slot
 * simply waits out the (bounded) repair. The final logits-writing prefill
 * chunk likewise always completes within its quantum — the engine's cancel
 * check only interrupts when enough suffix remains for a bit-exact resume
 * (pulsar_session_prefill_quantum_min_suffix).
 *
 * The largest quantum overshoot in the system is none of the above: it is
 * lazy slot provisioning (provision_slot in the scheduler below), whose
 * pulsar_session_create is a multi-GiB allocation that can take SECONDS and
 * stalls every bound slot for its duration — larger than the DSpark fused
 * step's ≤17-token burst. Deliberate: all GPU work stays on this one thread
 * (CUDA-state audit, pulsar_server_internal.h).
 */



/* Chunk-note wrapper around server_progress_cb: counts completed prefill
 * chunks in the CURRENT pulsar_session_sync call so the cancel callback can
 * interrupt after exactly one chunk. Counters are reset before each sync. */
static void gen_prefill_progress_cb(void *ud, const char *event, int current, int total) {
    gen_state *g = (gen_state *)ud;
    if (event && strcmp(event, "prefill_chunk") == 0) {
        if (g->prefill_last_current >= 0 && current > g->prefill_last_current) {
            g->prefill_chunks_done++;
        }
        if (current > g->prefill_last_current) g->prefill_last_current = current;
        g->prefill_total = total;
    }
    server_progress_cb(&g->progress, event, current, total);
}



/* One prefill chunk per quantum. Only interrupt when the engine guarantees
 * bit-exact resumption AND enough suffix remains that the resumed sync takes
 * the batched chunk path rather than the single-token tail path (see
 * pulsar_session_prefill_quantum_min_suffix). */
/* Has the client gone away?  A request whose peer has disconnected is pure
 * waste: on this box a single stream is ~16 t/s, so an abandoned long generation
 * burns minutes of EXCLUSIVE GPU that a live request could have used.  Streaming
 * requests already notice via a failed write (send_all), but a NON-streaming
 * request writes nothing until the very end, so nothing detected the disconnect
 * at all — it ran to max_tokens for a socket no one was reading.
 *
 * Polled non-blocking, and only at coarse boundaries (once per decode quantum,
 * ~1/s, and once before a queued job starts) — never per token.
 *
 * POLLRDHUP is the signal that matters: a client that times out or is Ctrl-C'd
 * sends FIN, which shows up as POLLRDHUP and NOT as POLLHUP (POLLHUP needs a
 * full teardown/RST).  The deliberate trade: a client that half-closes its write
 * side while still reading the response would be treated as gone.  HTTP clients
 * do not do that while awaiting a response, and the same assumption is standard
 * in production servers — but PULSAR_ABORT_ON_DISCONNECT=0 restores the old
 * run-to-completion behavior without a rebuild if some client ever misbehaves.
 * The env is read ONCE (project rule: no per-token getenv). */
static bool gen_client_disconnected(int fd) {
    if (fd < 0) return false;
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("DS4_ABORT_ON_DISCONNECT");
        enabled = (e && e[0] == '0') ? 0 : 1;
    }
    if (!enabled) return false;
    struct pollfd p;
    p.fd = fd;
    p.events = POLLRDHUP;
    p.revents = 0;
    if (poll(&p, 1, 0) <= 0) return false;   /* 0 = quiet = still connected */
    return (p.revents & (POLLERR | POLLHUP | POLLNVAL | POLLRDHUP)) != 0;
}


static bool gen_prefill_cancel_cb(void *ud) {
    const gen_state *g = (const gen_state *)ud;
    /* A client that cancelled or hung up during a long prefill must stop the
     * engine — otherwise opencode's deep-context prefills run to completion after
     * a cancel, burning the GPU and a bank. Polled between chunks (not per token);
     * gen_step_prefill abandons on the resulting interrupt. */
    if (gen_client_disconnected(g->j->fd)) return true;
    if (g->prefill_min_suffix == 0) return false;
    if (g->prefill_chunks_done < 1) return false;
    if (g->prefill_last_current < 0 || g->prefill_total <= g->prefill_last_current) return false;
    return (uint32_t)(g->prefill_total - g->prefill_last_current) >= g->prefill_min_suffix;
}



/* Shared failure epilogue for both prefill phases (the old duplicated blocks
 * after each pulsar_session_sync failure). Token vectors and the disk path are
 * freed centrally by gen_state_free. */
void server::gen_prefill_fail(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    pulsar_session_set_cancel(s->sess, NULL, NULL);
    pulsar_session_set_progress(s->sess, NULL, NULL);
    pulsar_session_set_display_progress(s->sess, NULL, NULL);
    s->kv_cache_tracker_bind(sl);
    kv_cache_restore_suppressed_continued(&s->kv, g->suppressed_continued_last,
                                          g->cold_store_len);
    s->kv_cache_tracker_flush(sl);
    s->kv_cache_discard_failed_disk_entry(sl, g->disk_cache_path);
    s->trace_event(g->trace_id, "prefill failed: %s", g->err);
    s->send_prefill_failure_response(g->j, &g->progress, g->ctx_span,
                                  g->req_flags, g->err);
    g->phase = GEN_DONE;
}



/* Resolve the prompt against every cache layer and decide the prefill plan.
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
void server::gen_begin(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    /* Tier-2: install this slot's bank before ANY s->sess touch below (all the
     * pos/common-prefix/tokens reads and the prefill sync run against the live
     * bank). No-op in classic mode / when already live. Finding 1: a failed spill
     * restore (KV unrecoverable) must fail the request, not run against another
     * bank's KV. */
    if (!s->bank_switch(sl->bank)) {
        snprintf(g->err, sizeof g->err,
                 "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
        s->gen_prefill_fail(sl);
        return;
    }
    const int old_pos = pulsar_session_pos(s->sess);
    const int common = pulsar_session_common_prefix(s->sess, &j->req.prompt);
    trace_cache_diag cache_diag = {0};
    trace_cache_capture(&cache_diag, pulsar_session_tokens(s->sess),
                        &j->req.prompt, old_pos, common);
    pulsar_tokens effective_prompt = {0};
    const pulsar_tokens *prompt_for_sync = &j->req.prompt;
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
    int cached = s->responses_live_visible_prefix_prompt(sl, &j->req, old_pos,
                                                      &effective_prompt);
    const char *cache_source = cached > 0 ? "responses-visible" : "none";
    if (cached > 0) {
        responses_live_match = "visible-prefix";
        if (s->responses_live_matches_request(sl, &j->req.responses_live_call_ids,
                                           old_pos))
        {
            responses_live_match_ids = j->req.responses_live_call_ids.len;
        }
    }
    if (cached == 0) {
        cached = s->responses_live_continuation_prompt(sl, &j->req, old_pos,
                                                    &effective_prompt,
                                                    &responses_live_match_ids);
        cache_source = cached > 0 ? "responses-tool-output" : "none";
        if (cached > 0) responses_live_match = "tool-output-ids";
    }
    if (cached > 0) {
        responses_live_continuation = true;
        prompt_for_sync = &effective_prompt;
    } else {
        cached = s->anthropic_live_continuation_prompt(sl, &j->req, old_pos,
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
        pulsar_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Responses continuation state is not available; retry by replaying the full input history");
        g->phase = GEN_DONE;
        return;
    } else if (cached == 0 && j->req.api == API_ANTHROPIC &&
               j->req.anthropic_requires_live_tool_state)
    {
        pulsar_tokens_free(&effective_prompt);
        http_error(j->fd, s->enable_cors, 409,
                   "Anthropic continuation state is not available; retry by replaying the full messages history");
        g->phase = GEN_DONE;
        return;
    } else if (cached == 0) {
        cached = common == old_pos && j->req.prompt.len >= old_pos ? common : 0;
        cache_source = cached > 0 ? "memory-token" : "none";
    }
    if (cached == 0) {
        int thinking_cached =
            s->thinking_live_visible_prefix_prompt(sl, &j->req, old_pos,
                                                &effective_prompt);
        if (thinking_cached > 0) {
            cached = thinking_cached;
            cache_source = "thinking-visible";
            thinking_live_continuation = true;
            prompt_for_sync = &effective_prompt;
        }
    }
    int disk_cached = 0;
    uint8_t disk_cache_ext_flags = 0;
    if (cached == 0) {
        int text_cached = s->live_text_prefix_prompt(sl, &j->req, &effective_prompt);
        if (text_cached > 0) {
            cached = text_cached;
            cache_source = "memory-text";
            prompt_for_sync = &effective_prompt;
        }
    }
    if (cached == 0 && old_pos > 0) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: live kv cache miss%s live=%d prompt=%d common=%d reason=%s",
                   responses_protocol ? " RESPPROTO" : "",
                   old_pos, j->req.prompt.len, common,
                   trace_cache_miss_reason(&cache_diag));
    }
    if (cached == 0) sl->continued_last_store_tokens = 0;
    if (s->kv.enabled && cached == 0 && old_pos >= s->kv.opt.min_tokens) {
        /* Loading a disk snapshot replaces the live GPU session.  Persist the
         * current checkpoint first, otherwise a cache hit for an older prefix
         * would silently discard the newer conversation state. */
        s->kv_cache_store_current(sl, "evict");
    }
    if (cached == 0) {
        disk_cached = s->kv_cache_try_load(sl, &j->req, &effective_prompt,
                                        &g->disk_cache_path,
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
     * effective prompt suffix evaluated by pulsar_session_sync() is written into
     * the live KV cache and can be reused by the next request. */
    j->req.cache_read_tokens = cached;
    j->req.cache_write_tokens = prompt_tokens > cached ? prompt_tokens - cached : 0;

    /* Prometheus /metrics: prompt-throughput + prefix-cache-hit counters. */
    pthread_mutex_lock(&s->mu);
    s->m_prompt_tokens += (uint64_t)(prompt_tokens > 0 ? prompt_tokens : 0);
    s->m_prefix_queries += (uint64_t)(prompt_tokens > 0 ? prompt_tokens : 0);
    s->m_prefix_hits += (uint64_t)(cached > 0 ? cached : 0);
    pthread_mutex_unlock(&s->mu);

    g->prompt_tokens = prompt_tokens;
    g->t0 = server_now_sec();
    g->trace_id = s->trace_begin(j, cached, prompt_tokens, &cache_diag,
                              cache_source, disk_cached, g->disk_cache_path);
    request_ctx_span(g->ctx_span, sizeof(g->ctx_span), cached, prompt_tokens);
    g->progress = (server_prefill_progress){
        .srv = s,
        .slot = sl,
        .kind = j->req.kind,
        .prompt_tokens = prompt_tokens,
        .cached_tokens = cached,
        .has_tools = j->req.has_tools,
        .responses_protocol = responses_protocol,
        .t0 = g->t0,
        .fd = j->fd,
        .stream = j->req.stream,
        .enable_cors = s->enable_cors,
    };
    snprintf(g->progress.ctx, sizeof(g->progress.ctx), "%s", g->ctx_span);
    log_flags(g->req_flags, sizeof(g->req_flags), responses_protocol,
              j->req.has_tools, false, false, false);
    if (responses_live_continuation) {
        server_log(PULSAR_LOG_PREFILL,
                   "pulsar-server: responses live continuation RESPPROTO match=%s ids=%d cached=%d prompt=%d",
                   responses_live_match ? responses_live_match : "unknown",
                   responses_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (anthropic_live_continuation) {
        server_log(PULSAR_LOG_PREFILL,
                   "pulsar-server: anthropic live continuation match=tool-output-ids ids=%d cached=%d prompt=%d",
                   anthropic_live_match_ids,
                   cached,
                   prompt_tokens);
    } else if (thinking_live_continuation) {
        server_log(PULSAR_LOG_PREFILL,
                   "pulsar-server: thinking live continuation match=visible-prefix cached=%d prompt=%d",
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
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: responses replay RESPPROTO missing reasoning state; continuing from visible history source=%s cached=%d prompt=%d",
                   cache_source,
                   cached,
                   prompt_tokens);
        s->trace_event(g->trace_id,
                    "responses replay missing reasoning state; continuing from visible history source=%s cached=%d",
                    cache_source, cached);
    }
    server_log(PULSAR_LOG_PREFILL,
               "pulsar-server: %s ctx=%s%s%s prompt start",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               g->ctx_span,
               g->req_flags[0] ? " " : "",
               g->req_flags);
    pulsar_session_set_progress(s->sess, gen_prefill_progress_cb, g);
    pulsar_session_set_display_progress(s->sess, server_progress_cb, &g->progress);

    int cold_store_len = 0;
    if (cached == 0 &&
        s->kv.enabled &&
        prompt_for_sync->len >= s->kv.opt.min_tokens &&
        s->kv.opt.cold_max_tokens > 0 &&
        prompt_for_sync->len <= s->kv.opt.cold_max_tokens)
    {
        const int anchor = kv_cache_chat_anchor_pos(&s->kv, prompt_for_sync,
                                                    pulsar_token_user(s->engine),
                                                    pulsar_token_assistant(s->engine));
        cold_store_len = anchor >= s->kv.opt.min_tokens ?
                         anchor : kv_cache_store_len(&s->kv, prompt_for_sync->len);
    }
    g->cold_store_len = cold_store_len;
    g->suppressed_continued_last = -1;
    if (cold_store_len >= s->kv.opt.min_tokens) {
        /* A cold checkpoint can land exactly on the continued-checkpoint
         * frontier.  The prefill progress callback would then write the same
         * prefix as "continued" while we are intentionally stopping there to
         * write it as "cold".  Mark the frontier as already handled before the
         * sync reaches it; if the cold write fails, restore the old schedule so
         * a later continued write can still try. */
        s->kv_cache_tracker_bind(sl);
        g->suppressed_continued_last =
            kv_cache_suppress_continued_store(&s->kv, cold_store_len);
        s->kv_cache_tracker_flush(sl);
    }

    /* Transfer prompt ownership into the slot state; the prefill phases run in
     * later quanta. */
    g->effective_prompt = effective_prompt;
    g->prompt_for_sync = prompt_for_sync == &effective_prompt ?
                         &g->effective_prompt : prompt_for_sync;
    g->responses_protocol = responses_protocol;
    g->responses_live_continuation = responses_live_continuation;
    g->anthropic_live_continuation = anthropic_live_continuation;
    g->thinking_live_continuation = thinking_live_continuation;

    /* Prefill quantum policy: interrupt the engine's chunk loop only when
     * resumption is bit-exact for this session (see gen_prefill_cancel_cb). The
     * cancel callback itself is armed per-sync in gen_step_prefill, NOT here: in
     * pool mode every slot shares the one pool session (s->sess), so a once-per-job set would be
     * clobbered by the next job the worker binds before prefilling this one. */
    g->prefill_min_suffix = pulsar_session_prefill_quantum_min_suffix(s->sess);

    if (s->kv.enabled &&
        g->cold_store_len >= s->kv.opt.min_tokens &&
        g->cold_store_len < g->prompt_for_sync->len)
    {
        tokens_copy_prefix(&g->cold_prefix, g->prompt_for_sync, g->cold_store_len);
        g->phase = GEN_PREFILL_COLD;
    } else {
        g->phase = GEN_PREFILL_MAIN;
    }
}



/* One prefill quantum: (re-)issue the sync toward the phase's target; the
 * cancel callback stops it after one completed chunk and the checkpoint
 * carries the progress to the next quantum. */
void server::gen_step_prefill(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    const bool cold = g->phase == GEN_PREFILL_COLD;
    const pulsar_tokens *target = cold ? &g->cold_prefix : g->prompt_for_sync;

    /* Arm the cancel callback on THIS slot's gen_state right before the sync.
     * In pool mode every slot shares the one pool session (s->sess), and the worker binds several
     * jobs (each of which would set the callback) before prefilling any of them,
     * so a once-per-job set in gen_begin leaves the LAST-bound slot's callback on
     * the shared session. An earlier slot's prefill would then yield on ANOTHER
     * slot's progress and, interrupted before its own first chunk, be misread as a
     * fatal error ("interrupted", HTTP 500). The worker prefills serially, so
     * setting it here binds the correct callback for this exact sync. */
    pulsar_session_set_cancel(s->sess, gen_prefill_cancel_cb, g);

    g->prefill_chunks_done = 0;
    g->prefill_last_current = -1;
    g->prefill_total = 0;
    const int rc = pulsar_session_sync(s->sess, target, g->err, sizeof(g->err));
    if (rc == PULSAR_SESSION_SYNC_INTERRUPTED) {
        if (gen_client_disconnected(g->j->fd)) {
            /* Client cancelled mid-prefill: abandon rather than resume or fail, so
             * a long deep-context prefill stops promptly on disconnect instead of
             * running to completion (the decode loop already abandons this way). */
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: client disconnected during prefill, abandoning");
            snprintf(g->err, sizeof(g->err), "client disconnected");
            s->gen_prefill_fail(sl);
            return;
        }
        if (g->prefill_chunks_done > 0) return; /* voluntary yield; resume next quantum */
        /* Interrupted without progress cannot be our cancel callback; fail
         * rather than risk a live-lock re-issuing the same sync forever. */
        s->gen_prefill_fail(sl);
        return;
    }
    if (rc != 0) {
        s->gen_prefill_fail(sl);
        return;
    }

    if (cold) {
        s->kv_cache_tracker_bind(sl);
        if (s->kv_cache_store_live_prefix(sl, g->prompt_for_sync, g->cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, g->cold_store_len);
            g->suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, g->suppressed_continued_last,
                                                  g->cold_store_len);
            g->suppressed_continued_last = -1;
        }
        s->kv_cache_tracker_flush(sl);
        pulsar_tokens_free(&g->cold_prefix);
        g->phase = GEN_PREFILL_MAIN;
        return; /* the cold store is a quantum boundary of its own */
    }

    pulsar_session_set_cancel(s->sess, NULL, NULL);
    s->gen_stream_begin(sl);
}

/* Runs once, in the same quantum that completed the main prefill: clear stale
 * live bindings, persist checkpoints, emit response identity, and start the
 * protocol stream projections that persist across all decode quanta. */
void server::gen_stream_begin(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    free(g->disk_cache_path);
    g->disk_cache_path = NULL;
    /* Once a non-live request wins, old protocol live bindings are stale. Keep
     * a binding only when this request explicitly continued from it. */
    if (!g->responses_live_continuation) s->responses_live_clear(sl);
    if (!g->anthropic_live_continuation) s->anthropic_live_clear(sl);
    if (!g->thinking_live_continuation) s->thinking_live_clear(sl);
    pulsar_session_set_progress(s->sess, NULL, NULL);
    pulsar_session_set_display_progress(s->sess, NULL, NULL);
    s->kv_cache_maybe_store_continued(sl);
    server_log(PULSAR_LOG_PREFILL,
               "pulsar-server: %s ctx=%s%s%s prompt done %.3fs",
               j->req.kind == REQ_CHAT ? "chat" : "completion",
               g->ctx_span,
               g->req_flags[0] ? " " : "",
               g->req_flags,
               server_now_sec() - g->t0);
    if (g->cold_store_len == g->prompt_for_sync->len) {
        s->kv_cache_tracker_bind(sl);
        if (s->kv_cache_store_live_prefix(sl, g->prompt_for_sync, g->cold_store_len, "cold")) {
            kv_cache_note_store(&s->kv, g->cold_store_len);
            g->suppressed_continued_last = -1;
        } else {
            kv_cache_restore_suppressed_continued(&s->kv, g->suppressed_continued_last,
                                                  g->cold_store_len);
        }
        s->kv_cache_tracker_flush(sl);
    }
    snprintf(g->id, sizeof(g->id), "%s-%llu",
             j->req.kind == REQ_CHAT ? "chatcmpl" : "cmpl",
             (unsigned long long)++s->seq);

    g->structured_stream = request_uses_structured_stream(&j->req);
    g->openai_live_chat = request_uses_openai_live_stream(&j->req);
    g->responses_live_chat = request_uses_responses_live_stream(&j->req);
    g->responses_created_at = (long)time(NULL);
    if (j->req.stream) {
        if (g->progress.stream_failed) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s%s%s stream closed during prefill",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
            g->phase = GEN_DONE;
            return;
        }
        /* The prefill progress callback may have already sent the SSE headers
         * to keep the connection alive during a long prefill. Only emit them
         * here when prefill never fired (e.g. fully cached prompt). */
        if (!g->progress.headers_sent && !sse_headers(j->fd, s->enable_cors)) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s%s%s sse headers failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
            g->phase = GEN_DONE;
            return;
        }
        g->progress.headers_sent = true;
        if (j->req.api == API_ANTHROPIC &&
            !anthropic_sse_start_live(j->fd, &j->req, g->id,
                                      g->prompt_tokens, &g->anthropic_live)) {
            server_log(PULSAR_LOG_GENERATION, "pulsar-server: chat ctx=%s anthropic stream start failed", g->ctx_span);
            g->phase = GEN_DONE;
            return;
        }
        if (j->req.api == API_OPENAI && j->req.kind == REQ_CHAT &&
            !sse_chunk(j->fd, &j->req, g->id, NULL, NULL)) {
            server_log(PULSAR_LOG_GENERATION, "pulsar-server: chat ctx=%s openai role chunk failed", g->ctx_span);
            g->phase = GEN_DONE;
            return;
        }
        if (g->openai_live_chat) openai_stream_start(&j->req, &g->openai_live);
        if (g->responses_live_chat) {
            responses_stream_init(&j->req, &g->responses_live);
            g->responses_live.active = true;
            if (!responses_sse_created(j->fd, &j->req, &g->responses_live, g->responses_created_at)) {
                server_log(PULSAR_LOG_GENERATION,
                           "pulsar-server: chat ctx=%s%s%s responses created event failed",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags);
                g->phase = GEN_DONE;
                return;
            }
        }
    }

    g->dsml_recovery_attempted = false;
    g->rng = j->req.seed ? j->req.seed :
        (((uint64_t)time(NULL) << 32) ^ ((uint64_t)s->seq << 1) ^ (uint64_t)(uintptr_t)j);
    g->phase = GEN_DECODE_INIT;
}



/* (Re)initialize a decode attempt: the body of the old decode_again label.
 * Runs both for a fresh request and after a tool-error recovery appended a
 * model-visible correction to the live session. */
void server::gen_decode_init(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    buf_free(&g->text);
    g->plain_stream_pos = 0;
    g->stop_scan_from = 0;
    g->finish = "length";
    g->completion = 0;
    g->max_tokens = j->req.max_tokens;
    int room = pulsar_session_ctx(s->sess) - pulsar_session_pos(s->sess);
    g->saw_tool_start = false;
    g->saw_tool_end = false;
    g->saw_orphan_tool_end = false;
    g->tool_scan_from = 0;
    g->next_tool_progress = 128;
    g->next_decode_log = 50;
    if (g->max_tokens < 0) g->max_tokens = 0;
    if (g->max_tokens > room) g->max_tokens = room;
    s->trace_event(g->trace_id, "prefill done; decode_max=%d ctx_room=%d", g->max_tokens, room);
    g->decode_t0 = server_now_sec();
    /* Snapshot the session's cumulative DSpark counters so gen_step_finish can
     * diff them into this request's accept-rate/tokens-per-step. Re-snapshotting
     * here (decode_again also lands here) keeps the diff consistent with the
     * reset completion/decode_t0 for the attempt that actually finishes. */
    pulsar_session_spec_metrics(s->sess, &g->spec_start);
    g->last_decode_log_t = g->decode_t0;
    g->last_decode_log_completion = 0;
    g->thinking = thinking_state_from_prompt(&j->req);
    g->thinking_gates_tool_markers = pulsar_think_mode_enabled(j->req.think_mode);
    g->tool_scan_waiting_for_think_close =
        g->thinking_gates_tool_markers && g->thinking.inside;
    g->think_recovery_scan_from = 0;
    g->think_tool_recovery_enabled =
        getenv("DS4_SERVER_DISABLE_THINK_TOOL_RECOVERY") == NULL;
    g->dspark_spec_enabled = getenv("DS4_DSPARK_DISABLE") == NULL;
    dsml_decode_tracker_init(&g->dsml_tracker);

    /* tool_choice="required": the prompt was prefilled into an open DSML
     * tool_calls block (thinking skipped: prompt ends "</think>\n\n<tool_calls>").
     * Seed the output with that exact prefix — including the closing </think> so
     * the (thinking-mode) parser sees reasoning end and a complete tool block —
     * and prime the trackers to "inside tool call"; the model now generates only
     * the invoke body. */
    if (j->req.kind == REQ_CHAT && j->req.force_tool_call) {
        request_forced_tool_seed(&j->req, &g->text);
        g->saw_tool_start = true;
        g->tool_scan_waiting_for_think_close = false;
        dsml_decode_tracker_update(&g->dsml_tracker, g->text.ptr, g->text.len);
        g->tool_scan_from = g->text.len;
        g->plain_stream_pos = g->text.len;
    }
    g->phase = GEN_DECODE;
}



/* Sampling contract: request_init() pre-fills the engine defaults, so the
 * request values are already correct for non-thinking requests. In thinking
 * mode the engine defaults are re-asserted, but ONLY for parameters the
 * client left absent (per-param has_* flags set in api_parse.c); anything the
 * client sent explicitly is respected as-is. That includes an explicit
 * temperature==0, which selects greedy decode so DSpark speculative decode
 * (greedy-only) can engage. Tool-call payload forcing (temperature=0 while
 * decoding structured tool output, in gen_step_decode) is a separate,
 * deliberate override applied on top of this. */
void gen_resolve_sampling(const request *req, float *temperature,
                                 int *top_k, float *top_p, float *min_p) {
    *temperature = req->temperature;
    *top_k = req->top_k;
    *top_p = req->top_p;
    *min_p = req->min_p;
    if (pulsar_think_mode_enabled(req->think_mode)) {
        if (!req->has_temperature) *temperature = PULSAR_DEFAULT_TEMPERATURE;
        if (!req->has_top_k) *top_k = 0;
        if (!req->has_top_p) *top_p = PULSAR_DEFAULT_TOP_P;
        if (!req->has_min_p) *min_p = PULSAR_DEFAULT_MIN_P;
    }
}



/* Emit one already-decoded token into the response stream: append it to the
 * accumulated text, feed the thinking/DSML trackers, run stop-string and
 * tool-marker detection, and drive every active protocol stream projection
 * (plain SSE / OpenAI / Anthropic / Responses). Returns true when the decode
 * loop must STOP after this token (EOS, a stop string, a completed tool_calls
 * block, or a client write error), with g->finish (and g->err on error) set.
 *
 * Factored out of gen_step_decode's per-token inner loop (plan Tier-2 Step 5)
 * so both drivers share ONE emit path: the classic spec/plain decode loop
 * below AND the batched multi-session scatter (which samples each live bank's
 * row on the host, then calls this to stream that bank's slot). It touches
 * ONLY host state hung off sl->gen + j->req + the client fd — no engine/CUDA
 * call except pulsar_token_text and the tool-recovery sync (chat_think_tool_
 * recovery, which the batched path never reaches because n>=2 is plain,
 * tool-gated decode by contract). Behavior for the single-session path is
 * byte-identical to the pre-factoring inner loop. */
bool server::gen_emit_token(session_slot *sl, int token) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    if (token == pulsar_token_eos(s->engine)) {
        g->finish = "stop";
        return true;
    }

    size_t piece_len = 0;
    char *piece = pulsar_token_text(s->engine, token, &piece_len);
    g->completion++;

    s->trace_piece(g->trace_id, piece, piece_len);
    buf_append(&g->text, piece, piece_len);
    g->thinking.feed(piece, piece_len);
    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        dsml_decode_tracker_update(&g->dsml_tracker, g->text.ptr, g->text.len);
    }

    size_t stop_pos = 0, stop_len = 0;
    bool hit_stop = stop_list_find_from(&j->req.stops, g->text.ptr,
                                        g->stop_scan_from,
                                        &stop_pos, &stop_len);
    size_t stream_len = hit_stop ?
        stop_pos : stop_list_stream_safe_len(&j->req.stops, g->text.len);
    if (stream_len > g->text.len) stream_len = g->text.len;
    stream_len = utf8_stream_safe_len(g->text.ptr, g->plain_stream_pos,
                                      stream_len, hit_stop);
    if (!hit_stop && j->req.stops.max_len > 1) {
        const size_t hold = j->req.stops.max_len - 1;
        g->stop_scan_from = g->text.len > hold ? g->text.len - hold : 0;
    }

    if (j->req.stream && !g->structured_stream && stream_len > g->plain_stream_pos) {
        char *delta = xstrndup(g->text.ptr + g->plain_stream_pos, stream_len - g->plain_stream_pos);
        bool ok = sse_chunk(j->fd, &j->req, g->id, delta, NULL);
        free(delta);
        if (!ok) {
            g->finish = "error";
            snprintf(g->err, sizeof(g->err), "client stream write failed");
            free(piece);
            return true;
        }
        g->plain_stream_pos = stream_len;
    }
    if (j->req.stream && j->req.api == API_ANTHROPIC &&
        !anthropic_sse_stream_update(j->fd, s, &j->req, g->id,
                                     &g->anthropic_live, g->text.ptr, stream_len,
                                     false)) {
        g->finish = "error";
        snprintf(g->err, sizeof(g->err), "client stream write failed");
        free(piece);
        return true;
    }
    if (g->openai_live_chat &&
        !openai_sse_stream_update(j->fd, s, &j->req, g->id,
                                  &g->openai_live, g->text.ptr, stream_len,
                                  false)) {
        g->finish = "error";
        snprintf(g->err, sizeof(g->err), "client stream write failed");
        free(piece);
        return true;
    }
    if (g->responses_live_chat &&
        !responses_sse_stream_update(j->fd, &j->req,
                                     &g->responses_live, g->text.ptr, stream_len,
                                     false)) {
        g->finish = "error";
        snprintf(g->err, sizeof(g->err), "client stream write failed");
        free(piece);
        return true;
    }
    free(piece);

    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        if (g->thinking_gates_tool_markers && g->thinking.inside) {
            /* A DSML block inside reasoning is not executable.  This is
             * the live guard: do not let a quoted or mistaken marker in
             * <think> stop decoding as a real tool call.  A complete
             * stanza opening, however, almost always means the model
             * forgot to close its thinking; recover by forcing the
             * close so the model restarts the call on the executable
             * side. */
            const int recovered = g->think_tool_recovery_enabled ?
                s->chat_think_tool_recovery(sl, &g->text, &g->thinking,
                                         &g->think_recovery_scan_from,
                                         &g->completion, g->max_tokens,
                                         g->err, sizeof(g->err)) : 0;
            if (recovered < 0) {
                g->finish = "error";
                return true;
            }
            if (recovered) {
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: chat ctx=%s%s%s tool call inside unclosed <think>; "
                           "forced </think> after %d generated tokens",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           g->completion);
                s->trace_event(g->trace_id,
                            "think tool recovery after %d generated tokens",
                            g->completion);
                dsml_decode_tracker_update(&g->dsml_tracker, g->text.ptr, g->text.len);
                g->tool_scan_waiting_for_think_close = true;
            } else {
                g->tool_scan_waiting_for_think_close = true;
                g->tool_scan_from = g->text.len;
            }
        } else {
            if (g->tool_scan_waiting_for_think_close) {
                const char *think_end = find_last_substr(g->text.ptr, "</think>");
                g->tool_scan_from = think_end ? (size_t)((think_end + 8) - g->text.ptr) : g->text.len;
                if (g->tool_scan_from > g->text.len) g->tool_scan_from = g->text.len;
                g->tool_scan_waiting_for_think_close = false;
            }
            if (g->tool_scan_from > g->text.len) g->tool_scan_from = g->text.len;
            const char *tool_scan = g->text.ptr ? g->text.ptr + g->tool_scan_from : "";
            bool orphan_end = false;
            bool old_start = g->saw_tool_start;
            bool old_end = g->saw_tool_end;
            observe_tool_markers(tool_scan, &g->saw_tool_start, &g->saw_tool_end, &orphan_end);
            if (orphan_end && !g->saw_orphan_tool_end) {
                g->saw_orphan_tool_end = true;
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: chat ctx=%s%s%s ignored orphan tool-call end marker after %d generated tokens",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           g->completion);
                s->trace_event(g->trace_id,
                            "ignored orphan tool-call end marker after %d generated tokens",
                            g->completion);
            }
            if (g->saw_tool_start && !old_start) {
                s->trace_event(g->trace_id, "entered tool-call block after %d generated tokens", g->completion);
            }
            if (g->saw_tool_end && !old_end) {
                s->trace_event(g->trace_id, "closed tool-call block after %d generated tokens", g->completion);
            }
            const size_t marker_hold = 80;
            size_t hold_from = g->text.len > marker_hold ? g->text.len - marker_hold : 0;
            if (hold_from > g->tool_scan_from) g->tool_scan_from = hold_from;
            if (s->trace && g->completion >= g->next_tool_progress) {
                s->trace_event(g->trace_id,
                            "progress gen=%d dsml_start=%d dsml_end=%d",
                            g->completion, g->saw_tool_start ? 1 : 0, g->saw_tool_end ? 1 : 0);
                g->next_tool_progress += 128;
            }
        }
    }

    if (g->completion >= g->next_decode_log) {
        log_decode_progress(j->req.kind, g->prompt_tokens, g->completion,
                            g->responses_protocol,
                            j->req.has_tools,
                            g->thinking.inside,
                            g->saw_tool_start,
                            g->saw_tool_end,
                            g->decode_t0,
                            &g->last_decode_log_t,
                            &g->last_decode_log_completion);
        g->next_decode_log += 50;
    }

    if (hit_stop) {
        (void)stop_len;
        g->finish = "stop";
        g->text.len = stop_pos;
        g->text.ptr[g->text.len] = '\0';
        pulsar_session_invalidate(s->sess);
        return true;
    }

    if (j->req.kind == REQ_CHAT && j->req.has_tools && g->saw_tool_end) {
        g->finish = "tool_calls";
        return true;
    }
    return false;
}



/* One decode quantum: run the sampling loop for at most
 * PULSAR_SERVER_DECODE_QUANTUM_TOKENS generated tokens, then yield with all loop
 * state parked in gen_state. The session is untouched between quanta, so
 * resuming is exactly the next iteration of the old run-to-completion loop. */
void server::gen_step_decode(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;
    const int quantum_start = g->completion;
    bool stop_decode = false;

    /* Once per quantum (~1/s): stop generating for a client that has gone away.
     * Falls through the normal GEN_FINISH epilogue so the slot, KV checkpoint and
     * metrics are released exactly as on a natural stop; the final write simply
     * fails harmlessly on the dead fd. */
    if (gen_client_disconnected(j->fd)) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: client disconnected, abandoning generation after %d tokens",
                   g->completion);
        g->phase = GEN_FINISH;
        return;
    }

    while (!g_stop_requested && g->completion < g->max_tokens &&
           pulsar_session_pos(s->sess) < pulsar_session_ctx(s->sess)) {
        if (g->completion - quantum_start >= PULSAR_SERVER_DECODE_QUANTUM_TOKENS) {
            sl->tokens_emitted += (uint64_t)(g->completion - quantum_start);
            return; /* quantum exhausted; phase stays GEN_DECODE */
        }
        dsml_decode_state dsml_state = j->req.kind == REQ_CHAT && j->req.has_tools ?
            g->dsml_tracker.decode : DSML_DECODE_OUTSIDE;
        const bool in_tool_call = dsml_decode_state_is_tool(dsml_state);
        if (!(j->req.kind == REQ_CHAT && j->req.has_tools && (g->saw_tool_start || in_tool_call))) {
            s->kv_cache_maybe_store_continued(sl);
        }
        float temperature, top_p, min_p;
        int top_k;
        gen_resolve_sampling(&j->req, &temperature, &top_k, &top_p, &min_p);
        if (in_tool_call && !dsml_decode_state_uses_payload_sampling(dsml_state)) {
            temperature = 0.0f;
        }
        int token;
        int toks[17];
        int ntok = 0;
        if (pulsar_engine_has_dspark(s->engine) && g->dspark_spec_enabled) {
            /* the speculative block owns sampling (exact sampled acceptance at
             * any temperature; greedy degenerates to the argmax rule) */
            ntok = pulsar_session_generate_speculative(s->sess,
                                                    temperature, top_k, top_p, min_p,
                                                    &g->rng,
                                                    g->max_tokens - g->completion,
                                                    pulsar_token_eos(s->engine),
                                                    toks,
                                                    (int)(sizeof(toks) / sizeof(toks[0])),
                                                    g->err,
                                                    sizeof(g->err));
            if (ntok < 0) {
                g->finish = "error";
                break;
            }
        } else {
            token = pulsar_session_sample(s->sess, temperature, top_k, top_p, min_p, &g->rng);
            if (token == pulsar_token_eos(s->engine)) {
                g->finish = "stop";
                break;
            }
            if (pulsar_session_eval(s->sess, token, g->err, sizeof(g->err)) != 0) {
                g->finish = "error";
                break;
            }
            toks[0] = token;
            ntok = 1;
        }

        /* TTFT: stamp the moment the first output token is available. One guarded
         * clock read for the whole request (branch is predictably not-taken after
         * the first token), so no meaningful decode-loop overhead. */
        if (g->first_token_t == 0.0 && ntok > 0) g->first_token_t = server_now_sec();

        for (int ti = 0; ti < ntok && g->completion < g->max_tokens; ti++) {
            if (s->gen_emit_token(sl, toks[ti])) {
                stop_decode = true;
                break;
            }
        }
        if (stop_decode) break;
    }

    sl->tokens_emitted += (uint64_t)(g->completion - quantum_start);
    g->phase = GEN_FINISH;
}



/* Post-decode epilogue: tool repair/recovery, final parse, protocol live
 * state, checkpoints, the final response, and logging. Recovery paths loop
 * back to GEN_DECODE_INIT (the old goto decode_again). */
void server::gen_step_finish(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    job *j = g->j;

    if (g_stop_requested && strcmp(g->finish, "error") != 0) {
        g->finish = "error";
        snprintf(g->err, sizeof(g->err), "shutdown requested");
    }

    if (j->req.kind == REQ_CHAT && j->req.has_tools &&
        g->saw_tool_start && !g->saw_tool_end && strcmp(g->finish, "error") != 0)
    {
        /* Deterministically complete a simple truncation.  Anything more than
         * missing closing tags stays model-owned: for non-streaming requests,
         * append a tool error plus prompt reminder to the live session and let
         * the model issue a fresh call. */
        bool completed_truncation = false;
        buf repaired = {0};
        if (try_repair_dsml(g->text.ptr, g->text.len, &repaired)) {
            /* Parse repaired text to verify it produces valid tool calls */
            tool_calls test_calls = {0};
            char *test_content = NULL;
            char *test_reasoning = NULL;
            bool repair_ok = parse_generated_message_ex(repaired.ptr, false, &test_content, &test_reasoning, &test_calls);
            free(test_content);
            free(test_reasoning);
            if (repair_ok && test_calls.len > 0) {
                /* Repair succeeded - replace text with repaired version */
                free(g->text.ptr);
                g->text.ptr = buf_take(&repaired);
                g->text.len = strlen(g->text.ptr);
                g->text.cap = g->text.len ? g->text.len + 1 : 0;
                g->saw_tool_end = true;
                completed_truncation = true;
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: chat ctx=%s%s%s repaired unterminated tool call (%d calls recovered)",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           test_calls.len);
                s->trace_event(g->trace_id, "repaired unterminated tool call (%d calls recovered)", test_calls.len);
            }
            tool_calls_free(&test_calls);
        }
        if (!completed_truncation) {
            if (!j->req.stream && !g->dsml_recovery_attempted) {
                int recovery_tokens = 0;
                char recovery_err[160] = {0};
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: chat ctx=%s%s%s unterminated tool call; continuing with model-visible tool error",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags);
                s->trace_event(g->trace_id,
                            "unterminated tool call; continuing with model-visible tool error");
                if (s->continue_after_invalid_dsml(sl, &j->req, &g->thinking,
                                                "unterminated tool call",
                                                &recovery_tokens,
                                                recovery_err,
                                                sizeof(recovery_err)))
                {
                    g->dsml_recovery_attempted = true;
                    server_log(PULSAR_LOG_GENERATION,
                               "pulsar-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                               g->ctx_span,
                               g->req_flags[0] ? " " : "",
                               g->req_flags,
                               recovery_tokens);
                    s->trace_event(g->trace_id,
                                "tool-error continuation appended %d tokens",
                                recovery_tokens);
                    buf_free(&repaired);
                    buf_free(&g->text);
                    g->phase = GEN_DECODE_INIT; /* the old goto decode_again */
                    return;
                }
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "invalid tool call recovery failed: %s",
                         recovery_err[0] ? recovery_err : "unknown error");
            } else {
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "unterminated tool call");
            }
        }
        buf_free(&repaired);
    }

    if (g->completion > g->last_decode_log_completion) {
        log_decode_progress(j->req.kind, g->prompt_tokens, g->completion,
                            g->responses_protocol,
                            j->req.has_tools,
                            g->thinking.inside,
                            g->saw_tool_start,
                            g->saw_tool_end,
                            g->decode_t0,
                            &g->last_decode_log_t,
                            &g->last_decode_log_completion);
    }

    if (j->req.stream && !g->structured_stream && g->text.len > g->plain_stream_pos) {
        char *tail = xstrndup(g->text.ptr + g->plain_stream_pos, g->text.len - g->plain_stream_pos);
        if (!sse_chunk(j->fd, &j->req, g->id, tail, NULL)) g->finish = "error";
        free(tail);
    }

    tool_calls parsed_calls = {0};
    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    const char *final_finish = g->finish;
    bool recovered_tool_parse_failure = false;
    if (j->req.kind == REQ_CHAT) {
        bool parsed_ok = parse_generated_message_for_response(
            g->text.ptr ? g->text.ptr : "",
            j->req.has_tools,
            g->saw_tool_start,
            pulsar_think_mode_enabled(j->req.think_mode),
            &final_finish,
            g->err,
            sizeof(g->err),
            &parsed_content,
            &parsed_reasoning,
            &parsed_calls,
            &recovered_tool_parse_failure);
        if (!parsed_ok && recovered_tool_parse_failure && j->req.has_tools && g->saw_tool_start) {
            /* parse_generated_message failed even though DSML was present.
             * Semantic repair is intentionally avoided: if the parser cannot
             * execute the block, feed the model a tool error and the protocol
             * reminder so it owns the corrected next action. */
            if (!j->req.stream && !g->dsml_recovery_attempted) {
                int recovery_tokens = 0;
                char recovery_err[160] = {0};
                const char *detail = g->err[0] ? g->err : "invalid tool call";
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: chat ctx=%s%s%s invalid tool call; continuing with model-visible tool error",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags);
                s->trace_event(g->trace_id,
                            "invalid tool call; continuing with model-visible tool error");
                if (s->continue_after_invalid_dsml(sl, &j->req, &g->thinking,
                                                detail,
                                                &recovery_tokens,
                                                recovery_err,
                                                sizeof(recovery_err)))
                {
                    g->dsml_recovery_attempted = true;
                    server_log(PULSAR_LOG_GENERATION,
                               "pulsar-server: chat ctx=%s%s%s tool-error continuation appended %d tokens",
                               g->ctx_span,
                               g->req_flags[0] ? " " : "",
                               g->req_flags,
                               recovery_tokens);
                    s->trace_event(g->trace_id,
                                "tool-error continuation appended %d tokens",
                                recovery_tokens);
                    free(parsed_content);
                    free(parsed_reasoning);
                    tool_calls_free(&parsed_calls);
                    buf_free(&g->text);
                    g->phase = GEN_DECODE_INIT; /* the old goto decode_again */
                    return;
                }
                final_finish = "error";
                snprintf(g->err, sizeof(g->err), "invalid tool call recovery failed: %s",
                         recovery_err[0] ? recovery_err : "unknown error");
            }
            if (!parsed_ok) {
                /* Print raw DSML snippet for debugging */
                size_t dsml_snippet_len = 0;
                const char *dsml_start = NULL;
                const char *p;
                for (p = g->text.ptr; p && (size_t)(p - g->text.ptr) < g->text.len - 20; p++) {
                    if ((strncmp(p, PULSAR_TOOL_CALLS_START, strlen(PULSAR_TOOL_CALLS_START)) == 0) ||
                        (strncmp(p, PULSAR_TOOL_CALLS_START_SHORT, strlen(PULSAR_TOOL_CALLS_START_SHORT)) == 0) ||
                        (strncmp(p, "<tool_calls>", 12) == 0)) {
                        dsml_start = p;
                        break;
                    }
                }
                if (dsml_start) {
                    dsml_snippet_len = g->text.len - (dsml_start - g->text.ptr);
                    if (dsml_snippet_len > 500) dsml_snippet_len = 500;
                }
                /* Also log a snippet of the full text to see what the model output */
                size_t text_snippet_len = g->text.len > 300 ? 300 : g->text.len;
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: chat ctx=%s%s%s invalid tool call returned as assistant text finish=%s [text_len=%zu saw_start=%d saw_end=%d text_snippet: %.*s]",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           final_finish,
                           g->text.len,
                           g->saw_tool_start,
                           g->saw_tool_end,
                           (int)text_snippet_len,
                           g->text.ptr ? g->text.ptr : "(null)");
                server_log(PULSAR_LOG_WARNING,
                           "pulsar-server: chat ctx=%s%s%s invalid tool call dsml_snippet: %.*s",
                           g->ctx_span,
                           g->req_flags[0] ? " " : "",
                           g->req_flags,
                           (int)dsml_snippet_len,
                           dsml_start ? dsml_start : "(none)");
                s->trace_event(g->trace_id,
                            "invalid tool call returned as assistant text finish=%s",
                            final_finish);
            }
        }
        if (parsed_calls.len) {
            if (g->openai_live_chat) apply_openai_stream_tool_ids(&parsed_calls, &g->openai_live);
            if (j->req.api == API_ANTHROPIC && j->req.stream)
                apply_anthropic_stream_tool_ids(&parsed_calls, &g->anthropic_live);
            s->assign_tool_call_ids(&parsed_calls, j->req.api);
            s->tool_memory_remember(&parsed_calls);
            final_finish = "tool_calls";
        } else if (j->req.api == API_RESPONSES) {
            s->responses_live_clear(sl);
        }
    }
    log_tool_calls_summary(g->ctx_span, &parsed_calls,
                           g->responses_protocol);

    /* Populate the additive per-response "timings" block from counters the
     * worker already kept. Pure metadata assembled once at finish, off any hot
     * path; the emitter derives the rates. */
    {
        const double finish_t = server_now_sec();
        req_timings *t = &j->req.timings;
        t->ttft_s = g->first_token_t > 0.0 ? g->first_token_t - g->t0 : 0.0;
        t->prefill_s = g->decode_t0 > g->t0 ? g->decode_t0 - g->t0 : 0.0;
        t->decode_s = finish_t > g->decode_t0 ? finish_t - g->decode_t0 : 0.0;
        t->prompt_n = g->prompt_tokens;
        t->cached_n = j->req.cache_read_tokens;
        t->decode_n = g->completion;
        pulsar_spec_metrics spec_end;
        pulsar_session_spec_metrics(s->sess, &spec_end);
        t->spec_gen = spec_end.gen_tokens - g->spec_start.gen_tokens;
        t->spec_accepted = spec_end.accepted_tokens - g->spec_start.accepted_tokens;
        t->spec_draft = spec_end.draft_tokens - g->spec_start.draft_tokens;
        t->spec_drafts = spec_end.num_drafts - g->spec_start.num_drafts;
        t->spec_active = t->spec_gen > 0; /* the fused spec loop ran this request */
        t->valid = true;
    }

    s->trace_finish(g->trace_id, &j->req, final_finish, g->completion,
                 g->saw_tool_start, g->saw_tool_end,
                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                 parsed_reasoning, &parsed_calls, server_now_sec() - g->t0);

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
            s->responses_live_remember(sl, visible.ptr ? visible.ptr : "",
                                    parsed_calls.len ? &parsed_calls : NULL);
            buf_free(&visible);
            free(visible_suffix);
        } else {
            s->responses_live_clear(sl);
        }
    }
    if (j->req.api == API_ANTHROPIC) {
        if (parsed_calls.len && strcmp(final_finish, "error") &&
            strcmp(final_finish, "length"))
        {
            s->anthropic_live_remember(sl, &parsed_calls);
        } else {
            s->anthropic_live_clear(sl);
        }
    }

    if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        pulsar_think_mode_enabled(j->req.think_mode) &&
        !j->req.force_tool_call)
    {
        /* Tool call with thinking on: the reasoning is in the live KV, and the
         * client replays this turn with the reasoning preserved (agentic clients
         * echo reasoning_content so the model keeps its chain of thought), so we
         * remember the reasoning-PRESERVED bytes the next request will render as a
         * key and keep the live tokens — the next request byte-matches the key and
         * continues from live KV (or a disk-reloaded checkpoint keyed by the same
         * visible transcript) with no rebuild. */
        s->remember_tool_thinking_checkpoint(sl, j, g->ctx_span, g->trace_id,
                                          parsed_content ? parsed_content : "",
                                          parsed_reasoning, &parsed_calls);
    } else if (j->req.kind == REQ_CHAT && parsed_calls.len &&
        j->req.api != API_RESPONSES &&
        s->should_canonicalize_tool_checkpoint(&parsed_calls))
    {
        /* Chat/completions has no protocol object that binds the next request
         * to this live KV state.  Canonicalize only the fallback tool-call
         * path where we lack exact sampled DSML replay; when raw DSML is known,
         * replaying those bytes keeps future prompts aligned without rebuilding
         * hidden reasoning.  Responses deliberately skips this path because its
         * previous_response_id contract binds the next turn to live state. */
        s->canonicalize_tool_checkpoint(sl, j, g->ctx_span, g->trace_id,
                                     parsed_content ? parsed_content : "",
                                     parsed_reasoning, &parsed_calls);
        s->thinking_live_clear(sl);
    } else if (parsed_calls.len) {
        s->thinking_live_clear(sl);
    } else if (!parsed_calls.len &&
               should_remember_thinking_checkpoint(&j->req, &g->thinking, final_finish)) {
        s->remember_thinking_checkpoint(sl, j, g->ctx_span, g->trace_id,
                                     parsed_content ? parsed_content : "");
    } else if (!parsed_calls.len) {
        s->thinking_live_clear(sl);
    }

    if (j->req.stream) {
        bool response_ok = true;
        if (j->req.api == API_ANTHROPIC) {
            response_ok = anthropic_sse_finish_live(j->fd, s, &j->req, g->id, &g->anthropic_live,
                                                    g->text.ptr ? g->text.ptr : "", g->text.len,
                                                    &parsed_calls, final_finish, g->completion);
        } else if (g->openai_live_chat) {
            response_ok = openai_sse_finish_live(j->fd, s, &j->req, g->id, &g->openai_live,
                                                 g->text.ptr ? g->text.ptr : "", g->text.len,
                                                 &parsed_calls, final_finish,
                                                 g->prompt_tokens, g->completion);
        } else if (g->responses_live_chat) {
            /* If parse recovered a malformed tool call back to plain text,
             * pass parsed_content so the streaming tail can be flushed; in
             * the normal path parsed_content is the assistant text we already
             * streamed and the diff is empty. */
            const char *recover =
                recovered_tool_parse_failure ? parsed_content : NULL;
            response_ok = responses_sse_finish_live(j->fd, &j->req, &g->responses_live,
                                                    g->text.ptr ? g->text.ptr : "", g->text.len,
                                                    recover,
                                                    &parsed_calls, final_finish,
                                                    g->prompt_tokens, g->completion,
                                                    g->responses_created_at);
        } else if (g->structured_stream) {
            response_ok = sse_chat_finish(j->fd, &j->req, g->id,
                                          parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                                          parsed_reasoning,
                                          &parsed_calls, final_finish,
                                          g->prompt_tokens, g->completion);
        } else {
            response_ok = sse_chunk(j->fd, &j->req, g->id, NULL, final_finish) &&
                          sse_done(j->fd, &j->req, g->id, g->prompt_tokens, g->completion);
        }
        if (!response_ok) {
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: %s ctx=%s%s%s final stream failed",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->req_flags[0] ? " " : "",
                       g->req_flags);
        }
    } else if (j->req.api == API_ANTHROPIC) {
        anthropic_final_response(j->fd, s->enable_cors, &j->req, g->id,
                                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 g->prompt_tokens, g->completion);
    } else if (j->req.api == API_RESPONSES) {
        responses_final_response(j->fd, s->enable_cors, &j->req, g->id,
                                 parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                                 parsed_reasoning,
                                 &parsed_calls, final_finish,
                                 g->prompt_tokens, g->completion);
    } else {
        final_response(j->fd, s->enable_cors, &j->req, g->id,
                       parsed_content ? parsed_content : (g->text.ptr ? g->text.ptr : ""),
                       parsed_reasoning,
                       &parsed_calls, final_finish,
                       g->prompt_tokens, g->completion);
    }
    if (j->req.kind == REQ_CHAT && j->req.has_tools) {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  g->responses_protocol,
                  true,
                  g->thinking.inside,
                  g->saw_tool_start,
                  g->saw_tool_end);
        if (!strcmp(final_finish, "error") && g->err[0]) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: chat ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       g->err,
                       server_now_sec() - g->t0);
        } else {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: chat ctx=%s gen=%d%s%s finish=%s %.3fs",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       server_now_sec() - g->t0);
        }
    } else {
        char flags[80];
        log_flags(flags, sizeof(flags),
                  g->responses_protocol,
                  j->req.has_tools,
                  g->thinking.inside,
                  false,
                  false);
        if (!strcmp(final_finish, "error") && g->err[0]) {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s gen=%d%s%s finish=%s error=\"%s\" %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       g->err,
                       server_now_sec() - g->t0);
        } else {
            server_log(PULSAR_LOG_GENERATION,
                       "pulsar-server: %s ctx=%s gen=%d%s%s finish=%s %.3fs",
                       j->req.kind == REQ_CHAT ? "chat" : "completion",
                       g->ctx_span,
                       g->completion,
                       flags[0] ? " " : "",
                       flags,
                       final_finish,
                       server_now_sec() - g->t0);
        }
    }
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&parsed_calls);
    g->phase = GEN_DONE;
}



/* ---- state-machine driver: bind, step, unbind ---- */

void server::gen_state_free(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    if (!g) return;
    (void)s;
    /* Callback safety: no gen_state pointer may remain installed anywhere. */
    pulsar_session_set_cancel(s->sess, NULL, NULL);
    pulsar_session_set_progress(s->sess, NULL, NULL);
    pulsar_session_set_display_progress(s->sess, NULL, NULL);
    anthropic_stream_free(&g->anthropic_live);
    openai_stream_free(&g->openai_live);
    responses_stream_free(&g->responses_live);
    buf_free(&g->text);
    pulsar_tokens_free(&g->effective_prompt);
    pulsar_tokens_free(&g->cold_prefix);
    pulsar_tokens_free(&g->batch_pending);
    free(g->disk_cache_path);
    slot_writer_free(&g->writer);
    free(g);
    sl->gen = NULL;
}



/* Bind a dequeued job to the slot and resolve its prompt (the first quantum). */
void server::generate_job_begin(session_slot *sl, job *j) {
    auto *s = this;
    gen_state *g = (gen_state *)server_xmalloc(sizeof(*g));
    memset(g, 0, sizeof(*g));
    g->j = j;
    g->prompt_for_sync = &j->req.prompt;
    g->finish = "length";
    g->suppressed_continued_last = -1;
    sl->gen = g;
    sl->active_job = j;
    sl->state = SLOT_PREFILLING;
    /* All client writes for this job (worker thread only) become non-blocking
     * and deferred; drained in generate_job_end. */
    slot_writer_init(&g->writer, j->fd);
    slot_writer_install(&g->writer);
    s->gen_begin(sl);
}



/* Advance the job by one quantum. */
void server::generate_job_step(session_slot *sl) {
    auto *s = this;
    gen_state *g = sl->gen;
    /* Tier-2: install this slot's bank before any engine work this quantum.
     * After another slot (or a fresh bind) was serviced in between, the pool's
     * live bank may be someone else's; switch back to ours. No-op in classic
     * mode / when already live. Finding 1: fail the request on a failed spill
     * restore rather than run engine work against the wrong bank's KV. */
    if (!s->bank_switch(sl->bank)) {
        snprintf(g->err, sizeof g->err,
                 "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
        g->finish = "error";
        g->phase = GEN_FINISH;
        return;
    }
    /* Tier-2: a slot leaving the batched lane (now at GEN_FINISH) has its bank
     * installed above (bank_state_restore cleared the multiseq poison and
     * installed the driver-maintained device counters); catch the host
     * checkpoint up to the tokens multiseq committed, so gen_step_finish's
     * store/continuation see the true frontier. */
    if (g->batch_active) {
        if (g->batch_pending.len > 0)
            pulsar_session_note_committed_tokens(s->sess, g->batch_pending.v,
                                              g->batch_pending.len);
        pulsar_tokens_free(&g->batch_pending);
        g->batch_active = false;
        g->batch_feed_valid = false;
    }
    /* The installed slot writer is worker-thread-local and shared across
     * slots; re-install this slot's writer so send_all() routes through the
     * right deferral queue after another slot (or a fresh bind) was serviced
     * in between. */
    slot_writer_install(&g->writer);
    /* Push any bytes a slow client deferred before spending GPU time. */
    slot_writer_flush(&g->writer);
    switch (g->phase) {
    case GEN_PREFILL_COLD:
    case GEN_PREFILL_MAIN:
        sl->state = SLOT_PREFILLING;
        s->gen_step_prefill(sl);
        break;
    case GEN_DECODE_INIT:
        sl->state = SLOT_DECODING;
        s->gen_decode_init(sl);
        if (g->phase != GEN_DECODE) break;
        /* fall through into the first decode quantum */
        s->gen_step_decode(sl);
        break;
    case GEN_DECODE:
        sl->state = SLOT_DECODING;
        s->gen_step_decode(sl);
        break;
    case GEN_FINISH:
        s->gen_step_finish(sl);
        break;
    case GEN_DONE:
        break;
    }
}



/* Unbind: drain deferred client bytes, free the resumable state. */
void server::generate_job_end(session_slot *sl) {
    auto *s = this;
    if (sl->gen) slot_writer_drain(&sl->gen->writer);
    s->gen_state_free(sl);
    sl->active_job = NULL;
    sl->state = SLOT_IDLE;
    sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
}

