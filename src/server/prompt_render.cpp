#include "pulsar_server_internal.h"



static void append_tools_prompt_text(buf *b, const char *tool_schemas) {
    if (!tool_schemas || !tool_schemas[0]) return;
    buf_puts(b,
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user question. "
        "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
        "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n\n"
        "String parameters should be specified as raw text and set `string=\"true\"`. "
        "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. "
        "Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. "
        "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n");
    buf_puts(b, tool_schemas);
    buf_puts(b, "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. "
                "Use the exact parameter names from the schemas.");
}



void json_escape(buf *b, const char *s);



static void json_args_free(json_args *args) {
    for (int i = 0; i < args->len; i++) {
        free(args->v[i].key);
        free(args->v[i].value);
    }
    free(args->v);
    memset(args, 0, sizeof(*args));
}



static void json_args_push(json_args *args, json_arg arg) {
    if (args->len == args->cap) {
        args->cap = args->cap ? args->cap * 2 : 8;
        args->v = (json_arg *)server_xrealloc(args->v, (size_t)args->cap * sizeof(args->v[0]));
    }
    args->v[args->len++] = arg;
}



static int json_args_find_unused(json_args *args, const char *key) {
    if (!key) return -1;
    for (int i = 0; i < args->len; i++) {
        if (!args->v[i].used && args->v[i].key && !strcmp(args->v[i].key, key)) return i;
    }
    return -1;
}



static bool json_args_parse(const char *json, json_args *args) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        bool is_string = false;
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') goto bad;
        p++;
        json_ws(&p);
        if (*p == '"') {
            is_string = true;
            if (!json_string(&p, &value)) goto bad;
        } else {
            char *raw = NULL;
            if (!json_raw_value(&p, &raw)) goto bad;
            value = json_minify_raw_value(raw);
            free(raw);
        }

        json_arg arg;
        arg = {.key = key, .value = value, .is_string = is_string};
        json_args_push(args, arg);
        key = value = NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
        continue;
bad:
        free(key);
        free(value);
        json_args_free(args);
        return false;
    }
    if (*p != '}') {
        json_args_free(args);
        return false;
    }
    return true;
}



static void append_dsml_attr_escaped(buf *b, const char *s) {
    for (s = s ? s : ""; *s; s++) {
        if (*s == '&') buf_puts(b, "&amp;");
        else if (*s == '<') buf_puts(b, "&lt;");
        else if (*s == '>') buf_puts(b, "&gt;");
        else if (*s == '"') buf_puts(b, "&quot;");
        else buf_putc(b, *s);
    }
}



static void append_dsml_parameter_text(buf *b, const char *s) {
    const char *end = "</｜DSML｜parameter>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "&lt;");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}



void append_tool_result_text(buf *b, const char *s) {
    /* Tool output is data.  DeepSeek's renderer keeps it as ordinary text inside
     * <tool_result>...</tool_result>, so preserving literal '<', '>' and '&' is
     * important for read-file tools and shell output.  The only delimiter we must
     * protect is the wrapper's own closing tag; otherwise a file containing that
     * exact sentinel would terminate the result early. */
    const char *end = "</tool_result>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "&lt;");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}



static void append_dsml_json_literal(buf *b, const char *s) {
    const char *end = "</｜DSML｜parameter>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "\\u003c");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}



static void append_dsml_arg(buf *b, const json_arg *arg) {
    buf_puts(b, "<｜DSML｜parameter name=\"");
    append_dsml_attr_escaped(b, arg->key);
    buf_puts(b, "\" string=\"");
    buf_puts(b, arg->is_string ? "true" : "false");
    buf_puts(b, "\">");
    if (arg->is_string) append_dsml_parameter_text(b, arg->value);
    else append_dsml_json_literal(b, arg->value);
    buf_puts(b, "</｜DSML｜parameter>\n");
}



bool append_dsml_arguments_from_json(buf *b, const char *json, const tool_schema_order *order) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) return false;
    if (order) {
        for (int i = 0; i < order->len; i++) {
            int idx = json_args_find_unused(&args, order->prop[i]);
            if (idx < 0) continue;
            append_dsml_arg(b, &args.v[idx]);
            args.v[idx].used = true;
        }
    }
    for (int i = 0; i < args.len; i++) {
        if (args.v[i].used) continue;
        append_dsml_arg(b, &args.v[i]);
    }
    json_args_free(&args);
    return true;
}



static void append_json_arg_pair(buf *b, const json_arg *arg) {
    json_escape(b, arg->key);
    buf_puts(b, ":");
    if (arg->is_string) json_escape(b, arg->value);
    else buf_puts(b, arg->value);
}



void append_json_object_or_empty(buf *b, const char *json) {
    json_args args = {0};
    if (!json_args_parse(json, &args)) {
        buf_puts(b, "{}");
        return;
    }
    buf_putc(b, '{');
    bool wrote = false;
    for (int i = 0; i < args.len; i++) {
        if (wrote) buf_putc(b, ',');
        append_json_arg_pair(b, &args.v[i]);
        wrote = true;
    }
    buf_putc(b, '}');
    json_args_free(&args);
}



void append_dsml_tool_calls_text(buf *b, const tool_calls *calls) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_dsml && calls->raw_dsml[0]) {
        buf_puts(b, calls->raw_dsml);
        return;
    }
    buf_puts(b, "\n\n<｜DSML｜tool_calls>\n");
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        buf_puts(b, "<｜DSML｜invoke name=\"");
        append_dsml_attr_escaped(b, tc->name);
        buf_puts(b, "\">\n");
        if (!append_dsml_arguments_from_json(b, tc->arguments, NULL)) {
            buf_puts(b, "<｜DSML｜parameter name=\"arguments\" string=\"true\">");
            append_dsml_parameter_text(b, tc->arguments);
            buf_puts(b, "</｜DSML｜parameter>\n");
        }
        buf_puts(b, "</｜DSML｜invoke>\n");
    }
    buf_puts(b, "</｜DSML｜tool_calls>");
}



static bool role_is_system(const char *role) {
    return !strcmp(role, "system") || !strcmp(role, "developer");
}



static bool role_is_user_like(const char *role) {
    return !strcmp(role, "user") || !strcmp(role, "tool") || !strcmp(role, "function");
}



bool chat_history_uses_tool_context(const chat_msgs *msgs,
                                           const char *tool_schemas) {
    if (tool_schemas && tool_schemas[0]) return true;
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if ((!strcmp(m->role, "assistant") && m->calls.len > 0) ||
            !strcmp(m->role, "tool") || !strcmp(m->role, "function"))
        {
            return true;
        }
    }
    return false;
}


/* Does this client REPLAY prior reasoning on the next turn?  This governs the
 * thinking checkpoint (should_remember_thinking_checkpoint): a reasoning-
 * preserving client (opencode et al.) echoes reasoning_content so its replay
 * byte-matches the live KV directly and no stripped checkpoint is wanted; a
 * reasoning-stripping client (openwebui) drops it, so the durable key must be
 * the reasoning-STRIPPED transcript.
 *
 * The old code keyed this off chat_history_uses_tool_context — i.e. "tools are
 * advertised" — on the assumption that tool-advertising clients are agentic and
 * preserve reasoning.  openwebui breaks that: it advertises a tool list AND
 * strips reasoning, so every thinking turn was stored token-text-with-reasoning
 * and its stripped replay cold-re-prefilled the whole conversation.
 *
 * The ground truth is observable: if any prior assistant turn carries non-empty
 * reasoning the client preserves; if we have seen assistant turns and NONE do,
 * it strips.  Only with no assistant history yet can we not tell — there we fall
 * back to the tool-context heuristic (unchanged first-turn behavior; the signal
 * self-corrects from the second turn on, where the deep re-prefills hurt). */
bool chat_history_preserves_reasoning(const chat_msgs *msgs,
                                             const char *tool_schemas) {
    bool saw_assistant = false;
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (strcmp(m->role, "assistant")) continue;
        saw_assistant = true;
        if (m->reasoning && m->reasoning[0]) return true;
    }
    if (saw_assistant) return false;
    return chat_history_uses_tool_context(msgs, tool_schemas);
}



char *render_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                     const tool_schema_orders *tool_orders,
                                     pulsar_think_mode think_mode) {
    (void)tool_orders;
    const bool think = pulsar_think_mode_enabled(think_mode);
    const bool tool_context = chat_history_uses_tool_context(msgs, tool_schemas);
    int last_user_idx = -1;
    buf system = {0};
    /* Render tool schemas before the client system content so
     * --kv-cache-boundary-trim-tokens chops a dynamic tail from the client
     * message instead of the much larger tool-schema region. */
    if (tool_schemas && tool_schemas[0]) {
        append_tools_prompt_text(&system, tool_schemas);
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!role_is_system(m->role)) continue;
        if (system.len) buf_puts(&system, "\n\n");
        buf_puts(&system, m->content ? m->content : "");
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_user_like(m->role)) last_user_idx = i;
    }

    buf out = {0};
    buf_puts(&out, PULSAR_SERVER_RENDER_BOS);
    if (think_mode == PULSAR_THINK_MAX) buf_puts(&out, pulsar_think_max_prefix());
    buf_puts(&out, system.ptr ? system.ptr : "");

    bool pending_assistant = false;
    bool pending_tool_result = false;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            continue;
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<｜User｜>");
            buf_puts(&out, m->content ? m->content : "");
            pending_assistant = true;
            pending_tool_result = false;
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            if (!pending_tool_result) buf_puts(&out, "<｜User｜>");
            buf_puts(&out, "<tool_result>");
            append_tool_result_text(&out, m->content);
            buf_puts(&out, "</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if (!strcmp(m->role, "assistant")) {
            if (pending_assistant) {
                buf_puts(&out, "<｜Assistant｜>");
                if (think) {
                    if (tool_context || i > last_user_idx) {
                        buf_puts(&out, "<think>");
                        buf_puts(&out, m->reasoning ? m->reasoning : "");
                        buf_puts(&out, "</think>");
                    } else {
                        buf_puts(&out, "</think>");
                    }
                } else {
                    buf_puts(&out, "</think>");
                }
            }
            buf_puts(&out, m->content ? m->content : "");
            append_dsml_tool_calls_text(&out, &m->calls);
            buf_puts(&out, "<｜end▁of▁sentence｜>");
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if (pending_assistant) {
        buf_puts(&out, "<｜Assistant｜>");
        buf_puts(&out, think ? "<think>" : "</think>");
    }

    buf_free(&system);
    return buf_take(&out);
}



/* Render only the semantic tail that must be appended to the live KV for a
 * tool-result continuation.
 *
 * In the common agent tool path, the previous assistant tool-call turn is
 * already in the model session, including hidden thinking and exact sampled
 * DSML.  The next request provides only the tool results, either as OpenAI
 * Responses tool-output items or Anthropic user content blocks.  Re-rendering
 * the assistant call here would duplicate it and destroy cache alignment, so
 * this function starts at the first new item and emits only:
 *
 *   previous EOS, tool results, and the next assistant prefix.
 *
 * This is intentionally independent from req.prompt's already-tokenized suffix:
 * suffix tokenization happens later after the cache decision, using the live
 * token prefix as the boundary.  That avoids BPE merges across the visible
 * replay/live-KV boundary. */
static char *render_live_tool_tail(const chat_msgs *msgs, int start,
                                   pulsar_think_mode think_mode) {
    const bool think = pulsar_think_mode_enabled(think_mode);
    buf out = {0};
    buf_puts(&out, "<｜end▁of▁sentence｜>");

    bool pending_assistant = false;
    bool pending_tool_result = false;
    for (int i = start; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            continue;
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<｜User｜>");
            buf_puts(&out, m->content ? m->content : "");
            pending_assistant = true;
            pending_tool_result = false;
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            if (!pending_tool_result) buf_puts(&out, "<｜User｜>");
            buf_puts(&out, "<tool_result>");
            append_tool_result_text(&out, m->content);
            buf_puts(&out, "</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if (!strcmp(m->role, "assistant")) {
            if (pending_assistant) {
                buf_puts(&out, "<｜Assistant｜>");
                if (think) {
                    buf_puts(&out, "<think>");
                    buf_puts(&out, m->reasoning ? m->reasoning : "");
                    buf_puts(&out, "</think>");
                } else {
                    buf_puts(&out, "</think>");
                }
            }
            buf_puts(&out, m->content ? m->content : "");
            append_dsml_tool_calls_text(&out, &m->calls);
            buf_puts(&out, "<｜end▁of▁sentence｜>");
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if (pending_assistant) {
        buf_puts(&out, "<｜Assistant｜>");
        buf_puts(&out, think ? "<think>" : "</think>");
    }
    return buf_take(&out);
}



static bool chat_msg_has_call_id(const chat_msg *m, const char *id) {
    if (!m || !id || !id[0] || strcmp(m->role, "assistant")) return false;
    for (int i = 0; i < m->calls.len; i++) {
        if (m->calls.v[i].id && !strcmp(m->calls.v[i].id, id)) return true;
    }
    return false;
}



static void chat_msg_collect_tool_call_ids(const chat_msg *m, stop_list *ids) {
    if (!m || !ids) return;
    id_list_push_unique(ids, m->tool_call_id);
    for (int i = 0; i < m->tool_call_ids_len; i++) {
        id_list_push_unique(ids, m->tool_call_ids[i]);
    }
}



static const chat_msg *responses_find_prior_call_msg(const chat_msgs *msgs,
                                                     int before,
                                                     const char *id) {
    if (!msgs || !id || !id[0]) return NULL;
    if (before > msgs->len) before = msgs->len;
    for (int i = before - 1; i >= 0; i--) {
        if (chat_msg_has_call_id(&msgs->v[i], id)) return &msgs->v[i];
    }
    return NULL;
}



/* Validate Responses tool outputs before rendering.
 *
 * A tool output with a call_id is meaningful only if either:
 *   1. DS4 still has the matching live assistant call in memory, or
 *   2. the same request replays the prior assistant call item.
 *
 * Case 1 is the fast, protocol-native continuation path: keep the live KV and
 * append only the tool result.  Case 2 is stateless replay after restart or
 * branching.  In thinking mode, case 2 is less faithful if the replay omits
 * reasoning state for the assistant call.  Official Responses clients can
 * carry that state with reasoning items / encrypted reasoning content; when
 * they do not, the request is still renderable as visible history.  Mark that
 * condition so generate_job() can prefer live / visible checkpoints and emit a
 * warning if it must fall back to visible replay instead of aborting the
 * session. */
bool responses_validate_tool_outputs(server *s, const chat_msgs *msgs,
                                            pulsar_think_mode think_mode,
                                            bool *requires_live_tool_state,
                                            bool *requires_live_reasoning,
                                            char *err, size_t errlen) {
    if (!msgs) return true;
    if (requires_live_tool_state) *requires_live_tool_state = false;
    if (requires_live_reasoning) *requires_live_reasoning = false;
    const bool needs_reasoning = pulsar_think_mode_enabled(think_mode);
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (strcmp(m->role, "tool") && strcmp(m->role, "function")) continue;

        stop_list ids = {0};
        chat_msg_collect_tool_call_ids(m, &ids);
        for (int j = 0; j < ids.len; j++) {
            const char *id = ids.v[j];
            const bool live_known = responses_live_has_call_id(s, id);
            const chat_msg *prior = responses_find_prior_call_msg(msgs, i, id);
            if (!live_known && !prior) {
                snprintf(err, errlen,
                         "Responses continuation state is not available for call_id %s; retry by replaying the full input history",
                         id);
                id_list_free(&ids);
                return false;
            }
            if (!prior) {
                if (requires_live_tool_state) *requires_live_tool_state = true;
                continue;
            }
            if (needs_reasoning &&
                (!prior->reasoning || !prior->reasoning[0]))
            {
                if (requires_live_reasoning) *requires_live_reasoning = true;
            }
        }
        id_list_free(&ids);
    }
    return true;
}



/* Record the call ids and suffix candidate for a live Responses continuation.
 *
 * This only prepares evidence.  generate_job() later checks that the live
 * server state is still exactly at the remembered token frontier before using
 * it.  If another request already replaced the session, normal token/text/disk
 * prefix matching handles the request instead. */
void responses_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs) {
    if (!r || r->api != API_RESPONSES || !msgs || msgs->len == 0) return;

    int tail_start = msgs->len;
    while (tail_start > 0) {
        const chat_msg *m = &msgs->v[tail_start - 1];
        if (strcmp(m->role, "tool") && strcmp(m->role, "function")) break;
        tail_start--;
    }
    if (tail_start == msgs->len) return;

    stop_list_clear(&r->responses_live_call_ids);
    if (tail_start > 0) {
        const int anchor = tail_start - 1;
        const chat_msg *assistant = &msgs->v[anchor];
        if (strcmp(assistant->role, "assistant") || assistant->calls.len == 0) return;
        for (int i = 0; i < assistant->calls.len; i++) {
            id_list_push_unique(&r->responses_live_call_ids, assistant->calls.v[i].id);
        }
    } else {
        for (int i = tail_start; i < msgs->len; i++) {
            chat_msg_collect_tool_call_ids(&msgs->v[i], &r->responses_live_call_ids);
        }
    }
    if (r->responses_live_call_ids.len == 0) return;

    free(r->responses_live_suffix_text);
    r->responses_live_suffix_text =
        render_live_tool_tail(msgs, tail_start, r->think_mode);
}



static bool anthropic_msg_is_tool_result_tail(const chat_msg *m) {
    return m && !strcmp(m->role, "user") &&
           ((m->tool_call_id && m->tool_call_id[0]) ||
            m->tool_call_ids_len > 0);
}



/* Validate Anthropic tool results before rendering.
 *
 * A tool_result.tool_use_id is valid if it is either still bound to the live
 * Anthropic assistant tool-call frontier or the same request replays the prior
 * assistant tool_use block.  The first case is the fast path: keep the sampled
 * KV and append only the tool-result suffix.  The second case is a normal
 * stateless replay, where exact DSML tool memory can restore the sampled tool
 * bytes before prefix matching.  A tool-result-only request with an unknown
 * live id has no safe prefix to reconstruct, so report a clear client error. */
bool anthropic_validate_tool_results(server *s, const chat_msgs *msgs,
                                            bool *requires_live_tool_state,
                                            char *err, size_t errlen) {
    if (requires_live_tool_state) *requires_live_tool_state = false;
    if (!msgs) return true;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!anthropic_msg_is_tool_result_tail(m)) continue;

        stop_list ids = {0};
        chat_msg_collect_tool_call_ids(m, &ids);
        for (int j = 0; j < ids.len; j++) {
            const char *id = ids.v[j];
            const bool live_known = anthropic_live_has_call_id(s, id);
            const chat_msg *prior = responses_find_prior_call_msg(msgs, i, id);
            if (!live_known && !prior) {
                snprintf(err, errlen,
                         "Anthropic continuation state is not available for tool_use_id %s; retry by replaying the full messages history",
                         id);
                id_list_free(&ids);
                return false;
            }
            if (!prior && requires_live_tool_state) {
                *requires_live_tool_state = true;
            }
        }
        id_list_free(&ids);
    }
    return true;
}



/* Prepare the Anthropic live-tool fast path.
 *
 * Anthropic's visible replay normally includes the assistant tool_use JSON and
 * the user tool_result.  That replay is still only a description of what the
 * model sampled.  If the incoming tool_result IDs match the live sampled
 * frontier, generate_job() can skip replay matching entirely and append just
 * EOS + tool_result + next assistant prefix to the real KV. */
void anthropic_prepare_live_continuation(request *r,
                                                const chat_msgs *msgs) {
    if (!r || r->api != API_ANTHROPIC || !msgs || msgs->len == 0) return;

    int tail_end = msgs->len;
    while (tail_end > 0 && role_is_system(msgs->v[tail_end - 1].role)) tail_end--;
    int tail_start = tail_end;
    while (tail_start > 0 &&
           anthropic_msg_is_tool_result_tail(&msgs->v[tail_start - 1]))
    {
        tail_start--;
    }
    if (tail_start == tail_end) return;

    stop_list_clear(&r->anthropic_live_call_ids);
    for (int i = tail_start; i < msgs->len; i++) {
        chat_msg_collect_tool_call_ids(&msgs->v[i], &r->anthropic_live_call_ids);
    }
    if (r->anthropic_live_call_ids.len == 0) return;

    free(r->anthropic_live_suffix_text);
    r->anthropic_live_suffix_text =
        render_live_tool_tail(msgs, tail_start, r->think_mode);
}

