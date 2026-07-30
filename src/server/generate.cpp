#include "pulsar_server_internal.h"



bool thinking_state::tail_ends_with(const char *s) const {
    const auto *st = this;
    int n = (int)strlen(s);
    return st->tail_len >= n && !memcmp(st->tail + st->tail_len - n, s, (size_t)n);
}



void thinking_state::feed(const char *p, size_t len) {
    auto *st = this;
    if (!st || !p) return;
    for (size_t i = 0; i < len; i++) {
        if (st->tail_len == (int)sizeof(st->tail)) {
            memmove(st->tail, st->tail + 1, sizeof(st->tail) - 1);
            st->tail_len--;
        }
        st->tail[st->tail_len++] = p[i];
        if (st->tail_ends_with("<think>")) st->inside = true;
        else if (st->tail_ends_with("</think>")) st->inside = false;
    }
}



thinking_state thinking_state_from_prompt(const request *r) {
    thinking_state st = {0};
    if (r && r->prompt_text) {
        st.feed(r->prompt_text, strlen(r->prompt_text));
    } else if (r && pulsar_think_mode_enabled(r->think_mode)) {
        st.inside = true;
    }
    return st;
}



static char *rendered_chat_system_region(const char *prompt_text) {
    if (!prompt_text) return xstrdup("");
    const char *p = prompt_text;
    const char *bos = PULSAR_SERVER_RENDER_BOS;
    const size_t bos_len = strlen(bos);
    if (!strncmp(p, bos, bos_len)) p += bos_len;
    const char *max_prefix = pulsar_think_max_prefix();
    const size_t max_prefix_len = strlen(max_prefix);
    if (max_prefix_len && !strncmp(p, max_prefix, max_prefix_len)) {
        p += max_prefix_len;
    }
    while (*p && isspace((unsigned char)*p)) p++;

    const char *user = strstr(p, "<｜User｜>");
    const char *assistant = strstr(p, "<｜Assistant｜>");
    const char *end = NULL;
    if (user && assistant) end = user < assistant ? user : assistant;
    else end = user ? user : assistant;
    if (!end) end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}



char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail) {
    char *system = rendered_chat_system_region(r ? r->prompt_text : NULL);
    buf tool_error = {0};
    buf_puts(&tool_error, "Tool error: invalid DSML tool call");
    if (detail && detail[0]) {
        buf_puts(&tool_error, ": ");
        buf_puts(&tool_error, detail);
    }
    buf_puts(&tool_error,
             "\nThe previous assistant output was not executed because the DSML syntax was malformed. "
             "Emit a new valid DSML tool call, or answer normally if no tool is needed.");
    if (system && system[0]) {
        buf_puts(&tool_error, "\n\nSystem prompt reminder:\n");
        buf_puts(&tool_error, system);
    }

    buf suffix = {0};
    if (r && pulsar_think_mode_enabled(r->think_mode) && thinking && thinking->inside) {
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, "<｜end▁of▁sentence｜><｜User｜><tool_result>");
    append_tool_result_text(&suffix, tool_error.ptr ? tool_error.ptr : "");
    buf_puts(&suffix, "</tool_result><｜Assistant｜>");
    buf_puts(&suffix, r && pulsar_think_mode_enabled(r->think_mode) ? "<think>" : "</think>");

    free(system);
    buf_free(&tool_error);
    return buf_take(&suffix);
}



bool should_remember_thinking_checkpoint(const request *r,
                                                const thinking_state *thinking,
                                                const char *finish) {
    if (!r || r->kind != REQ_CHAT) return false;
    /* has_tools is NOT a disqualifier: a client can advertise tools and still
     * strip reasoning on replay (openwebui).  prompt_preserves_reasoning now
     * reflects the client's ACTUAL replay behavior, so it is the sole gate;
     * remember_thinking_checkpoint renders the tool-context vs toolless form. */
    if (r->prompt_preserves_reasoning) return false;
    if (!pulsar_think_mode_enabled(r->think_mode)) return false;
    if (finish && (!strcmp(finish, "error") || !strcmp(finish, "length"))) return false;
    if (thinking && thinking->inside) return false;
    return true;
}



char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls) {
    buf suffix = {0};
    if (pulsar_think_mode_enabled(r->think_mode)) {
        buf_puts(&suffix, reasoning ? reasoning : "");
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_dsml_tool_calls_text(&suffix, calls);
    buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    return buf_take(&suffix);
}



char *build_responses_visible_assistant_suffix(const request *r,
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
    if (pulsar_think_mode_enabled(r->think_mode)) {
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
char *build_toolless_thinking_visible_text(const request *r,
                                                  const char *content) {
    if (!r || !r->prompt_text) return NULL;
    if (!pulsar_think_mode_enabled(r->think_mode)) return NULL;

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



bool server::should_canonicalize_tool_checkpoint(const tool_calls *calls) const {
    const auto *s = this; (void)s;
    if (!calls || calls->len == 0) return false;
    if (s && !s->disable_exact_dsml_tool_replay &&
        calls->raw_dsml && calls->raw_dsml[0])
    {
        return false;
    }
    return true;
}

