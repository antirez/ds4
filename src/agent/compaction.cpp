#include "ds4_agent_internal.h"



/* ============================================================================
 * Context Compaction
 * ============================================================================
 *
 * Compaction asks the model for durable task state, then rebuilds the live
 * transcript as: system prompt + summary + recent verbatim tail.  This keeps
 * the active KV usable while avoiding unbounded transcript growth.
 */

/* Decide when to compact before an ordinary turn or before appending a large
 * tool result.  The fixed free-token threshold is capped proportionally for
 * smaller contexts so tests with tiny contexts still compact rather than fail. */
static bool agent_worker_should_compact(agent_worker *w) {
    int ctx = w->cfg->gen.ctx_size;
    int used = w->transcript.len;
    if (ctx <= 0 || used <= 0) return false;
    if (used >= (ctx * AGENT_COMPACT_SOFT_PERCENT) / 100) return true;
    int free_threshold = AGENT_COMPACT_MIN_FREE_TOKENS;
    int proportional = ctx / 4;
    if (free_threshold > proportional) free_threshold = proportional;
    return ctx - used <= free_threshold;
}



static int agent_special_token_id(ds4_engine *engine, const char *rendered) {
    ds4_tokens t = {0};
    ds4_tokenize_rendered_chat(engine, rendered, &t);
    int id = t.len == 1 ? t.v[0] : -1;
    ds4_tokens_free(&t);
    return id;
}



/* Pick a recent verbatim tail for the compacted transcript.  Prefer a user
 * boundary inside the budget so the rebuilt context starts at a natural turn. */
static int agent_compact_tail_start(agent_worker *w, int bottom, int sys_len) {
    int tail_budget = w->cfg->gen.ctx_size / AGENT_COMPACT_TAIL_DIVISOR;
    if (tail_budget > AGENT_COMPACT_TAIL_CAP_TOKENS)
        tail_budget = AGENT_COMPACT_TAIL_CAP_TOKENS;
    if (tail_budget < 1) tail_budget = 1;

    int target = bottom - tail_budget;
    if (target < sys_len) target = sys_len;

    int user_id = agent_special_token_id(w->engine, "<｜User｜>");
    if (user_id < 0) return target;

    for (int i = target; i < bottom; i++) {
        if (w->transcript.v[i] == user_id) return i;
    }
    return target;
}



static void agent_tokens_append_range(ds4_tokens *dst, const ds4_tokens *src,
                                      int start, int end) {
    if (start < 0) start = 0;
    if (end > src->len) end = src->len;
    for (int i = start; i < end; i++) ds4_tokens_push(dst, src->v[i]);
}



/* Build the private prompt used to ask the model for durable state.  The prompt
 * explicitly forbids tool calls because the result is consumed internally, not
 * delivered as an assistant turn. */
static char *agent_compact_make_prompt(const char *reason) {
    agent_buf b = {0};
    agent_buf_puts(&b,
        "Internal ds4-agent context compaction request. This is not a user request.\n"
        "Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n"
        "- user goals, constraints, and preferences\n"
        "- files inspected or edited\n"
        "- commands run and important results\n"
        "- decisions, rejected approaches, known bugs, and pending next steps\n"
        "- reloadable bulky data with exact paths/ranges/commands when available\n\n"
        "Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n"
        "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n"
        "Output only the compact summary.\n");
    if (reason && reason[0]) {
        agent_buf_puts(&b, "\nCompaction reason: ");
        agent_buf_puts(&b, reason);
        agent_buf_puts(&b, "\n");
    }
    return agent_buf_take(&b);
}



/* Perform the full compaction exchange and rebuild the live DS4 session from
 * the compacted transcript.  Any failure invalidates live KV because the model
 * may have just seen private compaction instructions that are not part of the
 * real conversation. */
bool agent_worker_compact(agent_worker *w, const char *reason,
                                 char *err, size_t err_len) {
    const int bottom = w->transcript.len;
    if (bottom <= 0) return true;

    ds4_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);
    if (bottom <= sys.len) {
        ds4_tokens_free(&sys);
        return true;
    }

    agent_publishf(w,
        "\n\x1b[1;95mCOMPACTING\x1b[0m %s: summarizing durable task state\n\x1b[38;5;245m",
        reason && reason[0] ? reason : "context");

    char *prompt_text = agent_compact_make_prompt(reason);
    ds4_tokens prompt = {0};
    ds4_tokens_copy(&prompt, &w->transcript);
    ds4_chat_append_message(w->engine, &prompt, "user", prompt_text);
    free(prompt_text);
    ds4_chat_append_assistant_prefix(w->engine, &prompt, DS4_THINK_NONE);

    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_COMPACTING;
    w->progress_started_at = agent_now_sec();
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = false;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);

    int summary_room = w->cfg->gen.ctx_size - prompt.len - 1;
    if (summary_room < 256) {
        snprintf(err, err_len, "not enough context left to request compaction summary");
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        return false;
    }
    int summary_max = summary_room < AGENT_COMPACT_SUMMARY_MAX_TOKENS ?
                      summary_room : AGENT_COMPACT_SUMMARY_MAX_TOKENS;

    ds4_session_set_progress(w->session, worker_progress_cb, w);
    ds4_session_set_display_progress(w->session, worker_progress_cb, w);
    ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
    int sync_rc = ds4_session_sync(w->session, &prompt, err, err_len);
    ds4_session_set_cancel(w->session, NULL, NULL);
    ds4_session_set_progress(w->session, NULL, NULL);
    ds4_session_set_display_progress(w->session, NULL, NULL);
    if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED) {
        ds4_session_invalidate(w->session);
        snprintf(err, err_len, "interrupted");
        agent_publish_system_status(
            w, "Compaction interrupted; keeping the previous conversation state.");
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        worker_clear_interrupt(w);
        return false;
    }
    if (sync_rc != 0) {
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        agent_publish(w, "\x1b[0m\n", 5);
        return false;
    }

    /* From here until the final rebuild, the live KV contains the internal
     * compaction prompt/summary, while w->transcript still contains the real
     * conversation.  If anything fails, invalidate live KV so the next turn
     * cannot accidentally continue from the private compaction exchange. */
    agent_buf summary = {0};
    char eval_err[160] = {0};
    int think_end_id = agent_special_token_id(w->engine, "</think>");
    int dsml_id = agent_special_token_id(w->engine, "｜DSML｜");
    double t0 = agent_now_sec();
    for (int i = 0; i < summary_max; i++) {
        if (worker_should_interrupt(w)) {
            snprintf(err, err_len, "interrupted");
            ds4_session_invalidate(w->session);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&sys);
            free(summary.ptr);
            agent_publish(w, "\x1b[0m\n", 5);
            agent_publish_system_status(
                w, "Compaction interrupted; keeping the previous conversation state.");
            worker_clear_interrupt(w);
            return false;
        }
        int token = ds4_session_argmax(w->session);
        if (token == ds4_token_eos(w->engine)) break;
        if (token == think_end_id || token == dsml_id) {
            if (token == dsml_id && summary.len && summary.ptr[summary.len - 1] == '<') {
                summary.ptr[--summary.len] = '\0';
            }
            agent_trace(w, "compaction summary stopped before control token id=%d", token);
            break;
        }
        if (ds4_session_eval(w->session, token, eval_err, sizeof(eval_err)) != 0) {
            snprintf(err, err_len, "%s", eval_err);
            ds4_session_invalidate(w->session);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&sys);
            free(summary.ptr);
            agent_publish(w, "\x1b[0m\n", 5);
            return false;
        }

        size_t text_len = 0;
        char *text = ds4_token_text(w->engine, token, &text_len);
        agent_buf_append(&summary, text, text_len);
        agent_publish(w, text, text_len);
        free(text);

        double dt = agent_now_sec() - t0;
        pthread_mutex_lock(&w->mu);
        w->status.generated = i + 1;
        w->status.gen_tps = dt > 0.0 ? (double)(i + 1) / dt : 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }
    agent_publish(w, "\x1b[0m\n", 5);
    ds4_tokens_free(&prompt);

    if (!summary.ptr || !summary.ptr[0]) {
        snprintf(err, err_len, "compaction summary was empty");
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&sys);
        free(summary.ptr);
        return false;
    }

    int tail_start = agent_compact_tail_start(w, bottom, sys.len);
    ds4_tokens compacted = {0};
    ds4_tokens_copy(&compacted, &sys);

    agent_buf summary_msg = {0};
    agent_buf_puts(&summary_msg,
        "\n\n[ds4-agent compacted earlier conversation. Durable task-state summary follows.]\n");
    agent_buf_puts(&summary_msg, summary.ptr);
    if (summary_msg.len && summary_msg.ptr[summary_msg.len - 1] != '\n')
        agent_buf_puts(&summary_msg, "\n");
    agent_buf_puts(&summary_msg, "[End compacted summary. Recent conversation continues verbatim below.]\n\n");
    ds4_chat_append_message(w->engine, &compacted, "system", summary_msg.ptr);
    free(summary_msg.ptr);
    free(summary.ptr);

    agent_tokens_append_range(&compacted, &w->transcript, tail_start, bottom);

    agent_publishf(w,
        "\x1b[1;95mCOMPACTING\x1b[0m rebuilding context: old=%d summary+tail=%d tail=%d\n",
        bottom, compacted.len, bottom - tail_start);

    ds4_tokens old_transcript = {0};
    ds4_tokens_copy(&old_transcript, &w->transcript);
    ds4_tokens_free(&w->transcript);
    w->transcript = compacted;
    if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) {
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&w->transcript);
        w->transcript = old_transcript;
        ds4_tokens_free(&sys);
        return false;
    }
    agent_worker_note_system_prompt_seen(w);
    ds4_tokens_free(&old_transcript);
    ds4_tokens_free(&sys);
    char *bash_update = agent_bash_jobs_compaction_observation(w);
    if (bash_update) {
        ds4_chat_append_message(w->engine, &w->transcript, "tool", bash_update);
        w->session_dirty = true;
        agent_trace_text(w, "tool-after-compaction", bash_update, strlen(bash_update));
        agent_publish(w, "\x1b[90mCOMPACTING added bash job update after rebuild\x1b[0m\n",
                      strlen("\x1b[90mCOMPACTING added bash job update after rebuild\x1b[0m\n"));
        free(bash_update);
    }
    agent_trace(w, "compacted reason=\"%s\" old=%d new=%d tail_start=%d tail=%d",
                reason ? reason : "", bottom, w->transcript.len,
                tail_start, bottom - tail_start);
    return true;
}



bool agent_worker_compact_if_needed(agent_worker *w, const char *reason,
                                           char *err, size_t err_len) {
    if (!agent_worker_should_compact(w)) return true;
    return agent_worker_compact(w, reason, err, err_len);
}



int worker_accept_generated_token(agent_worker *w,
                                         int token,
                                         int *generated,
                                         double t0,
                                         agent_stream_renderer *stream,
                                         char *err,
                                         size_t err_len) {
    if (ds4_session_eval(w->session, token, err, err_len) != 0)
        return 1;

    ds4_tokens_push(&w->transcript, token);

    size_t text_len = 0;
    char *text = ds4_token_text(w->engine, token, &text_len);
    agent_trace_token(w, token, text, text_len, *generated + 1);
    agent_stream_text(stream, text, text_len, false);
    free(text);
    (*generated)++;

    double dt = agent_now_sec() - t0;
    pthread_mutex_lock(&w->mu);
    w->status.generated = *generated;
    w->status.gen_tps = dt > 0.0 ? (double)*generated / dt : 0.0;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
    return 0;
}



int worker_force_generated_text(agent_worker *w,
                                       const char *text,
                                       int max_tokens,
                                       int *generated,
                                       double t0,
                                       agent_stream_renderer *stream,
                                       char *err,
                                       size_t err_len) {
    ds4_tokens tokens = {0};
    ds4_tokenize_text(w->engine, text, &tokens);
    if (tokens.len > max_tokens - *generated) {
        snprintf(err, err_len, "not enough generation room to force %s", text);
        ds4_tokens_free(&tokens);
        return 1;
    }
    for (int i = 0; i < tokens.len && *generated < max_tokens; i++) {
        if (worker_accept_generated_token(w, tokens.v[i], generated, t0,
                                          stream, err, err_len) != 0) {
            ds4_tokens_free(&tokens);
            return 1;
        }
    }
    ds4_tokens_free(&tokens);
    return 0;
}

