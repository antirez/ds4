#include "ds4_agent_internal.h"



/* ============================================================================
 * Model Worker Thread
 * ============================================================================
 */

/* DSML structure is a machine-readable grammar, so once the model has clearly
 * started a tool stanza we decode grammar bytes greedily.  Parameter values are
 * different: they can be shell commands, code, file contents, or edit bodies,
 * and should keep the configured sampling behavior.  The only exception is a
 * parameter closing tag once it is clearly DSML syntax, not ordinary text such
 * as HTML/XML/code containing "</".
 *
 * This helper is intentionally derived only from the current streaming parser
 * state.  The state object is local to one assistant round, so malformed output,
 * EOS, Ctrl+C, or the next turn cannot accidentally leave sampling greedy. */
static bool agent_stream_wants_greedy_sampling(const agent_stream_renderer *sr) {
    if (!sr || !sr->parser) return false;
    if (sr->parser->state == AGENT_DSML_ERROR ||
        sr->parser->state == AGENT_DSML_DONE)
        return false;

    /* A possible opening marker is being held back by the start detector.  A
     * single '<' is too common in prose/code to justify forcing argmax; after
     * the second byte, the buffered prefix still matching here is specifically
     * DSML-shaped ("<｜..." or the tolerated "<D..." typo). */
    if (sr->dsml_start_len > 1) return true;
    if (!sr->dsml_active) return false;

    if (sr->parser->state == AGENT_DSML_STRUCTURAL)
        return true;
    if (sr->parser->state != AGENT_DSML_PARAM_VALUE)
        return false;

    return sr->parser->param_close_prefix;
}



static int worker_sample_with_mode(agent_worker *w, const agent_config *cfg,
                                   bool greedy, uint64_t *rng) {
    return ds4_session_sample(w->session,
                              greedy ? 0.0f : cfg->gen.temperature,
                              0,
                              greedy ? 1.0f : cfg->gen.top_p,
                              greedy ? 0.0f : cfg->gen.min_p,
                              rng);
}



static void worker_set_greedy_sampling(agent_worker *w, bool greedy) {
    pthread_mutex_lock(&w->mu);
    if (w->status.greedy_sampling != greedy) {
        w->status.greedy_sampling = greedy;
        agent_wake_locked(w);
    }
    pthread_mutex_unlock(&w->mu);
}



/* Run one user turn until the assistant stops or returns a tool call.  Tool
 * results are appended to the transcript and the loop continues, which gives
 * the model native DSML tool iteration without a client/server protocol. */
static int worker_run_turn(agent_worker *w, const char *user_text) {
    agent_config *cfg = w->cfg;
    ds4_think_mode think_mode = effective_think_mode(cfg);
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    w->status.error[0] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    char compact_err[160] = {0};
    if (!agent_worker_compact_if_needed(w, "soft limit before user turn",
                                        compact_err, sizeof(compact_err)))
    {
        if (agent_err_is_interrupted(compact_err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
        return 1;
    }
    agent_worker_maybe_append_datetime_context(w);
    agent_trace_text(w, "user", user_text ? user_text : "",
                     user_text ? strlen(user_text) : 0);
    if (!w->session_title) {
        w->session_title = agent_session_title_from_prompt(user_text, 0);
        w->session_created_at = (uint64_t)time(NULL);
        agent_session_identity_sha(w->session_title, w->session_created_at,
                                   w->session_sha);
    }
    ds4_chat_append_message(w->engine, &w->transcript, "user", user_text);

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    pthread_mutex_lock(&w->mu);
    w->user_activity = true;
    w->session_dirty = true;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    /* A user turn may contain any number of assistant/tool/assistant rounds.
     * Coding agents naturally perform long read/edit/test loops, so there is
     * deliberately no artificial "too many tool calls" ceiling here: context
     * pressure, compaction, user Ctrl+C, and the model's final answer are the
     * real stopping conditions.  The transcript is the single source of truth:
     * after a DSML stanza completes we terminate that assistant message, append
     * the tool result as a tool message, then ask the model to continue. */
    for (int tool_round = 0; ; tool_round++) {
        if (tool_round > 0 &&
            !agent_worker_compact_if_needed(w, "soft limit before tool continuation",
                                            compact_err, sizeof(compact_err)))
        {
            if (agent_err_is_interrupted(compact_err)) {
                worker_clear_interrupt(w);
                agent_set_status(w, AGENT_WORKER_IDLE);
                return 0;
            }
            agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
            return 1;
        }
        agent_worker_maybe_append_system_prompt_reminder(w);
        ds4_chat_append_assistant_prefix(w->engine, &w->transcript, think_mode);

        const ds4_tokens *prompt_for_sync = &w->transcript;
        int old_pos = ds4_session_pos(w->session);
        int common = ds4_session_common_prefix(w->session, &w->transcript);
        int cached = common == old_pos && w->transcript.len >= old_pos ? common : 0;

        int suffix = prompt_for_sync->len - cached;
        agent_trace(w, "prefill tool_round=%d transcript=%d prompt=%d cached=%d suffix=%d think=%s",
                    tool_round, w->transcript.len, prompt_for_sync->len,
                    cached, suffix, ds4_think_mode_name(think_mode));
        agent_trace_tokens(w, "prefill_suffix", prompt_for_sync, cached);

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
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        char err[160];
        ds4_session_set_progress(w->session, worker_progress_cb, w);
        ds4_session_set_display_progress(w->session, worker_progress_cb, w);
        ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
        int sync_rc = ds4_session_sync(w->session, prompt_for_sync, err, sizeof(err));
        ds4_session_set_cancel(w->session, NULL, NULL);
        ds4_session_set_progress(w->session, NULL, NULL);
        ds4_session_set_display_progress(w->session, NULL, NULL);
        if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED) {
            agent_publish_system_status(
                w, "Model reading interrupted; the model may only be aware of the prefix processed so far.");
            ds4_tokens_push(&w->transcript, ds4_token_eos(w->engine));
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (sync_rc != 0) {
            agent_set_error(w, err);
            return 1;
        }

        int max_tokens = cfg->gen.n_predict;
        int room = ds4_session_ctx(w->session) - ds4_session_pos(w->session);
        if (room <= 1) max_tokens = 0;
        else if (max_tokens > room - 1) max_tokens = room - 1;

        bool use_color = isatty(STDOUT_FILENO) != 0;
        agent_token_renderer renderer = {
            .engine = w->engine,
            .worker = w,
            .format_thinking = ds4_think_mode_enabled(think_mode),
            .format_markdown = use_color,
            .in_think = ds4_think_mode_enabled(think_mode),
            .use_color = use_color,
            .last_output_newline = true,
        };
        agent_dsml_parser dsml = {.state = AGENT_DSML_SEARCH};
        agent_stream_renderer stream = {
            .renderer = &renderer,
            .parser = &dsml,
            .in_think = ds4_think_mode_enabled(think_mode),
        };
        agent_edit_upto_forcer upto_forcer = {0};
        bool got_tool = false;
        bool malformed_tool = false;
        bool early_tool_error = false;
        int generated = 0;
        double t0 = agent_now_sec();

        pthread_mutex_lock(&w->mu);
        w->status.state = AGENT_WORKER_GENERATING;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);

        bool status_greedy_sampling = false;
        while (generated < max_tokens && !worker_should_interrupt(w)) {
            bool greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }
            int token = worker_sample_with_mode(w, cfg, greedy_sampling, &rng);
            if (token == ds4_token_eos(w->engine)) break;

            size_t text_len = 0;
            char *text = ds4_token_text(w->engine, token, &text_len);
            if (agent_edit_upto_forcer_should_replace(&upto_forcer, &dsml,
                                                       text, text_len))
            {
                agent_trace(w, "edit old auto-upto replaced token=%d text=%.*s",
                            token, (int)(text_len > 80 ? 80 : text_len), text);
                free(text);
                if (worker_force_generated_text(w, "[upto]\n", max_tokens,
                                                &generated, t0, &stream,
                                                err, sizeof(err)) != 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
            } else {
                free(text);
                if (worker_accept_generated_token(w, token, &generated, t0,
                                                  &stream, err, sizeof(err)) != 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
            }

            greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }

            if (dsml.state == AGENT_DSML_DONE) {
                got_tool = true;
                break;
            }
            if (stream.tool_preflight_error) {
                early_tool_error = true;
                break;
            }
            if (dsml.state == AGENT_DSML_ERROR) {
                malformed_tool = true;
                break;
            }
            if (stream.dsml_in_think) {
                malformed_tool = true;
                break;
            }
        }

        bool interrupted = worker_should_interrupt(w);
        agent_stream_text(&stream, NULL, 0, true);
        renderer_finish(&renderer);
        worker_set_greedy_sampling(w, false);
        if (interrupted) {
            ds4_tokens_push(&w->transcript, ds4_token_eos(w->engine));
            agent_dsml_parser_free(&dsml);
            agent_publish_system_status(w, "Stopped by user");
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (stream.dsml_in_think) {
            got_tool = false;
            malformed_tool = true;
            early_tool_error = false;
            snprintf(dsml.error, sizeof(dsml.error),
                     "tool calling is not allowed inside <think></think>");
        } else if (!malformed_tool && dsml.state == AGENT_DSML_ERROR) {
            malformed_tool = true;
        } else if (!got_tool && !malformed_tool && !early_tool_error &&
                   !interrupted &&
                   (dsml.state == AGENT_DSML_STRUCTURAL ||
                    dsml.state == AGENT_DSML_PARAM_VALUE))
        {
            malformed_tool = true;
            snprintf(dsml.error, sizeof(dsml.error),
                     "incomplete DSML tool call");
        }

        ds4_tokens_push(&w->transcript, ds4_token_eos(w->engine));

        if (!got_tool && !malformed_tool && !early_tool_error) {
            agent_dsml_parser_free(&dsml);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }

        char *tool_result;
        if (early_tool_error) {
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: ");
            agent_buf_puts(&b, stream.tool_preflight_error_msg[0] ?
                           stream.tool_preflight_error_msg :
                           "edit old selector failed before new was generated");
            agent_buf_puts(&b, "\n");
            tool_result = agent_buf_take(&b);
        } else if (malformed_tool) {
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: invalid DSML tool call: ");
            agent_buf_puts(&b, dsml.error[0] ? dsml.error : "parse error");
            agent_buf_puts(&b, "\n");
            agent_buf_puts(&b, agent_dsml_syntax_reminder);
            tool_result = agent_buf_take(&b);
        } else {
            tool_result = agent_execute_tool_calls(w, &dsml.calls);
        }
        int projected_tokens = 0;
        if (!agent_tool_result_fits_context(w, tool_result,
                                            AGENT_TOOL_RESULT_RESERVE_TOKENS,
                                            &projected_tokens))
        {
            if (!agent_worker_compact(w, "tool result would exceed context",
                                      compact_err, sizeof(compact_err)))
            {
                free(tool_result);
                agent_dsml_parser_free(&dsml);
                if (agent_err_is_interrupted(compact_err)) {
                    worker_clear_interrupt(w);
                    agent_set_status(w, AGENT_WORKER_IDLE);
                    return 0;
                }
                agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
                return 1;
            }
            if (!agent_tool_result_fits_context(w, tool_result,
                                                AGENT_TOOL_RESULT_RESERVE_TOKENS,
                                                &projected_tokens))
            {
                free(tool_result);
                agent_buf b = {0};
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "Tool error: tool result still does not fit after context compaction "
                         "(projected_prompt=%d tokens, ctx=%d, reserve=%d). "
                         "Retry with a smaller read/search/bash output.\n",
                         projected_tokens, w->cfg->gen.ctx_size,
                         AGENT_TOOL_RESULT_RESERVE_TOKENS);
                agent_buf_puts(&b, msg);
                tool_result = agent_buf_take(&b);
                if (!agent_tool_result_fits_context(w, tool_result, 16, NULL)) {
                    free(tool_result);
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, "context full after compaction");
                    return 1;
                }
            }
        }
        ds4_chat_append_message(w->engine, &w->transcript, "tool", tool_result);
        free(tool_result);
        agent_dsml_parser_free(&dsml);

        char *queued_user = worker_request_queued_user_drain(w);
        if (queued_user && queued_user[0]) {
            agent_trace_text(w, "queued_user", queued_user, strlen(queued_user));
            ds4_chat_append_message(w->engine, &w->transcript, "user", queued_user);
            pthread_mutex_lock(&w->mu);
            w->user_activity = true;
            w->session_dirty = true;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);
        }
        free(queued_user);
    }
}



void worker_request_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->save_requested = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



void worker_request_compact(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->compact_requested = true;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}



static bool worker_take_save_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->save_requested;
    w->save_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}



static bool worker_take_compact_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->compact_requested;
    w->compact_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}



static void worker_run_deferred_save(agent_worker *w) {
    if (!worker_take_save_requested(w)) return;
    agent_set_status(w, AGENT_WORKER_SAVING);
    char err[160] = {0};
    char sha[41];
    int tokens = 0;
    if (agent_worker_save_session_now(w, sha, &tokens, err, sizeof(err)))
        agent_publishf(w, "\nsaved session %.8s (%d tokens)\n", sha, tokens);
    else
        agent_publishf(w, "\nsave failed: %s\n", err[0] ? err : "unknown error");
    agent_set_status(w, AGENT_WORKER_IDLE);
}



static void worker_run_deferred_compact(agent_worker *w) {
    if (!worker_take_compact_requested(w)) return;
    if (!agent_worker_has_user_session(w)) {
        agent_publishf(w, "\ncompact skipped: nothing to compact\n");
        return;
    }

    int before = w->transcript.len;
    char err[160] = {0};
    if (agent_worker_compact(w, "user requested compaction", err, sizeof(err))) {
        if (w->transcript.len != before) {
            pthread_mutex_lock(&w->mu);
            w->session_dirty = true;
            agent_wake_locked(w);
            pthread_mutex_unlock(&w->mu);
        } else {
            agent_publishf(w, "\ncompact skipped: nothing to compact\n");
        }
        agent_set_status(w, AGENT_WORKER_IDLE);
    } else {
        if (agent_err_is_interrupted(err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return;
        }
        agent_set_error(w, err[0] ? err : "context compaction failed");
    }
}



/* Worker thread entry point.  The UI thread submits plain user text; this
 * thread owns all DS4 session mutation, tool execution, and compaction. */
void *worker_main(void *arg) {
    agent_worker *w = (agent_worker *)arg;
    agent_trace(w, "agent worker start ctx=%d backend=%s model=%s trace=%s",
                w->cfg->gen.ctx_size,
                ds4_backend_name(w->cfg->engine.backend),
                w->cfg->engine.model_path ? w->cfg->engine.model_path : "",
                w->cfg->gen.trace_path ? w->cfg->gen.trace_path : "");
    char init_err[160] = {0};
    if (!agent_worker_reset_to_sysprompt(w, init_err, sizeof(init_err))) {
        agent_set_error(w, init_err[0] ? init_err : "failed to initialize system prompt");
    }
    agent_trace_tokens(w, "initial_system_prompt", &w->transcript, 0);
    pthread_mutex_lock(&w->mu);
    w->initialized = true;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    while (true) {
        pthread_mutex_lock(&w->mu);
        while (!w->stop && !w->cmd_text && !w->save_requested &&
               !w->compact_requested)
            pthread_cond_wait(&w->cond, &w->mu);
        if (w->stop) {
            pthread_mutex_unlock(&w->mu);
            break;
        }
        if (!w->cmd_text && w->save_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_save(w);
            continue;
        }
        if (!w->cmd_text && w->compact_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_compact(w);
            continue;
        }
        char *cmd = w->cmd_text;
        w->cmd_text = NULL;
        pthread_mutex_unlock(&w->mu);

        worker_run_turn(w, cmd);
        free(cmd);
        worker_run_deferred_compact(w);
        worker_run_deferred_save(w);
    }

    agent_set_status(w, AGENT_WORKER_STOPPED);
    return NULL;
}

