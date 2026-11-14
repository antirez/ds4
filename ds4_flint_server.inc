/* Recursive Flint (fork-owned): the server's chat renderer, following
 * rec_mlx/chat_template.jinja (docs/RECURSIVE_FLINT.md step 5).  Textually
 * included into ds4_server.c before render_chat_prompt_text_for_syntax.
 *
 * Flint's tool-call syntax is GLM's (<tool_call>NAME<arg_key>K</arg_key>
 * <arg_value>V</arg_value></tool_call>), so a Flint engine reports
 * SERVER_MODEL_SYNTAX_GLM and reuses the GLM reply parser and the live
 * tool-call tracker; only the prompt text differs:
 *
 *   <|system|>TEXT  <|developer|>TEXT  <|user|>TEXT, in message order, with
 *   the tool preamble (a <|system|> turn) before the first non-system
 *   message; <|assistant|>[<think>R</think>]CONTENT[<tool_call>...]; tool
 *   results as <|observation|><tool_response>TEXT</tool_response>; no BOS,
 *   no newlines between turns.  Non-assistant text has <think> tags removed
 *   and role tags HTML-escaped, as the template's flint_text macro does.
 *   The generation prompt is <|assistant|>: rec's grammar then forces the
 *   first token to <think>, so thinking on prefills <think>; thinking off
 *   prefills <think></think> (the template's own rendering of an assistant
 *   turn without reasoning content). */

/* Python's json.dumps(value, ensure_ascii=False) over the client's JSON
 * text: ", " and ": " separators, keys in order, control characters as
 * \uXXXX, everything else raw (the schema line the model was trained on). */
static void flint_json_py_string(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        const unsigned char c = (unsigned char)*s;
        if (c == '"') buf_puts(b, "\\\"");
        else if (c == '\\') buf_puts(b, "\\\\");
        else if (c == '\n') buf_puts(b, "\\n");
        else if (c == '\r') buf_puts(b, "\\r");
        else if (c == '\t') buf_puts(b, "\\t");
        else if (c == '\b') buf_puts(b, "\\b");
        else if (c == '\f') buf_puts(b, "\\f");
        else if (c < 0x20) buf_printf(b, "\\u%04x", (unsigned)c);
        else buf_putc(b, (char)c);
    }
    buf_putc(b, '"');
}

static bool flint_json_py_value(buf *b, const char **p) {
    json_ws(p);
    if (**p == '{' || **p == '[') {
        const char close = **p == '{' ? '}' : ']';
        const bool object = **p == '{';
        buf_putc(b, **p);
        (*p)++;
        json_ws(p);
        bool first = true;
        while (**p && **p != close) {
            if (!first) buf_puts(b, ", ");
            first = false;
            if (object) {
                char *key = NULL;
                if (!json_string(p, &key)) return false;
                flint_json_py_string(b, key);
                free(key);
                json_ws(p);
                if (**p != ':') return false;
                (*p)++;
                buf_puts(b, ": ");
            }
            if (!flint_json_py_value(b, p)) return false;
            json_ws(p);
            if (**p == ',') { (*p)++; json_ws(p); }
        }
        if (**p != close) return false;
        (*p)++;
        buf_putc(b, close);
        return true;
    }
    if (**p == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        flint_json_py_string(b, s);
        free(s);
        return true;
    }
    const char *start = *p;
    while (**p && (isalnum((unsigned char)**p) || **p == '-' || **p == '+' || **p == '.')) (*p)++;
    if (*p == start) return false;
    buf_append(b, start, (size_t)(*p - start));
    return true;
}

/* The template's flint_text: drops NULs and <think> tags, escapes role tags. */
static void append_flint_text(buf *b, const char *s) {
    static const char *const tags[] = {
        "<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|developer|>",
        "<|user|>", "<|assistant|>", "<|observation|>",
    };
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, "<think>", 7)) { s += 7; continue; }
        if (!strncmp(s, "</think>", 8)) { s += 8; continue; }
        if (s[0] == '<' && s[1] == '|') {
            bool matched = false;
            for (size_t i = 0; i < sizeof(tags) / sizeof(tags[0]); i++) {
                const size_t n = strlen(tags[i]);
                if (!strncmp(s, tags[i], n)) {
                    buf_puts(b, "&lt;");
                    buf_append(b, tags[i] + 1, n - 2);
                    buf_puts(b, "&gt;");
                    s += n;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        buf_putc(b, *s++);
    }
}

static void append_flint_tools_prompt_text(buf *b, const char *tool_schemas) {
    if (!tool_schemas || !tool_schemas[0]) return;
    buf_puts(b,
        "<|system|>\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>");
    const char *p = tool_schemas;
    json_ws(&p);
    while (*p) {
        char *raw = NULL;
        if (!json_raw_value(&p, &raw)) break;
        /* no separators: HF's template environment trims the block newlines */
        buf py = {0};
        const char *q = raw;
        if (flint_json_py_value(&py, &q)) buf_append(b, py.ptr, py.len);
        else buf_puts(b, raw);
        buf_free(&py);
        free(raw);
        json_ws(&p);
    }
    buf_puts(b,
        "</tools>\n\n"
        "For each function call, output the function name and arguments within the following XML format:\n"
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n");
}

/* <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>:
 * string values raw, other values as Python JSON, keys in schema order. */
static void append_flint_tool_calls_text(buf *b, const tool_calls *calls, const tool_schema_orders *tool_orders) {
    if (!calls || calls->len == 0) return;
    if (calls->raw_tool_text && calls->raw_tool_text[0]) {
        buf_puts(b, calls->raw_tool_text);
        return;
    }
    for (int i = 0; i < calls->len; i++) {
        const tool_call *tc = &calls->v[i];
        const tool_schema_order *order = tool_schema_orders_find(tool_orders, tc->name);
        buf_puts(b, "<tool_call>");
        buf_puts(b, tc->name ? tc->name : "");
        json_args args = {0};
        if (tc->arguments && json_args_parse(tc->arguments, &args)) {
            for (int pass = 0; pass < 2; pass++) {
                for (int k = 0; k < (pass == 0 && order ? order->len : args.len); k++) {
                    int idx = k;
                    if (pass == 0) {
                        idx = json_args_find_unused(&args, order->prop[k]);
                        if (idx < 0) continue;
                    } else if (args.v[idx].used) {
                        continue;
                    }
                    const json_arg *arg = &args.v[idx];
                    buf_puts(b, "<arg_key>");
                    append_glm_arg_key_text(b, arg->key);
                    buf_puts(b, "</arg_key><arg_value>");
                    if (arg->is_string) {
                        append_glm_arg_value_text(b, arg->value);
                    } else {
                        buf py = {0};
                        const char *q = arg->value;
                        if (flint_json_py_value(&py, &q)) buf_append(b, py.ptr, py.len);
                        else buf_puts(b, arg->value);
                        buf_free(&py);
                    }
                    buf_puts(b, "</arg_value>");
                    args.v[idx].used = true;
                }
            }
            json_args_free(&args);
        } else if (tc->arguments && tc->arguments[0]) {
            buf_puts(b, "<arg_key>arguments</arg_key><arg_value>");
            append_glm_arg_value_text(b, tc->arguments);
            buf_puts(b, "</arg_value>");
        }
        buf_puts(b, "</tool_call>");
    }
}

static void append_flint_tool_result(buf *out, const chat_msg *m) {
    buf_puts(out, "<|observation|>");
    if (m->role && !strcmp(m->role, "user") && m->tool_call_ids_len > 0) {
        append_glm_tool_result_message(out, m);   /* Anthropic tool_result blocks */
        return;
    }
    buf text = {0};
    append_flint_text(&text, m->content);
    buf_puts(out, "<tool_response>");
    if (text.ptr) append_glm_tool_response_text(out, text.ptr);
    buf_puts(out, "</tool_response>");
    buf_free(&text);
}

/* Messages from `start` on; returns whether an assistant turn is pending. */
static bool append_flint_conversation(buf *out, const chat_msgs *msgs, int start, const char *tool_schemas,
                                      const tool_schema_orders *tool_orders, ds4_think_mode think_mode) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool tool_context = chat_history_uses_tool_context(msgs, tool_schemas);
    int last_user_idx = -1;
    for (int i = 0; msgs && i < msgs->len; i++) {
        if (role_is_user_like(msgs->v[i].role)) last_user_idx = i;
    }
    bool tools_pending = tool_schemas && tool_schemas[0] && start == 0;
    bool pending_assistant = start > 0;
    for (int i = start; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (tools_pending && strcmp(m->role, "system") != 0) {
            append_flint_tools_prompt_text(out, tool_schemas);
            tools_pending = false;
        }
        if (!strcmp(m->role, "system") || !strcmp(m->role, "developer")) {
            buf_puts(out, m->role[0] == 's' ? "<|system|>" : "<|developer|>");
            append_flint_text(out, m->content);
        } else if (chat_msg_is_glm_tool_result(m)) {
            append_flint_tool_result(out, m);
            pending_assistant = true;
        } else if (!strcmp(m->role, "assistant")) {
            buf_puts(out, "<|assistant|>");
            const char *content = m->content ? m->content : "";
            const bool preserve = think && (tool_context || i > last_user_idx);
            if (preserve && m->reasoning && m->reasoning[0] && !text_starts_with_think_tag(content)) {
                buf_puts(out, "<think>");
                buf_puts(out, m->reasoning);
                buf_puts(out, "</think>");
            }
            buf_puts(out, content);
            append_flint_tool_calls_text(out, &m->calls, tool_orders);
            pending_assistant = false;
        } else {
            buf_puts(out, "<|user|>");
            append_flint_text(out, m->content);
            pending_assistant = true;
        }
    }
    if (tools_pending) append_flint_tools_prompt_text(out, tool_schemas);
    if (pending_assistant) {
        buf_puts(out, "<|assistant|>");
        buf_puts(out, think ? "<think>" : "<think></think>");
    }
    return pending_assistant;
}

static char *render_flint_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas,
                                           const tool_schema_orders *tool_orders, ds4_think_mode think_mode) {
    buf out = {0};
    append_flint_conversation(&out, msgs, 0, tool_schemas, tool_orders, think_mode);
    /* tests/flint/check_template.py: DS4_FLINT_PROMPT_DUMP=FILE appends each
     * rendered prompt, NUL-terminated, for comparison with rec's template */
    const char *dump = getenv("DS4_FLINT_PROMPT_DUMP");
    if (dump && dump[0]) {
        FILE *f = fopen(dump, "ab");
        if (f) { fwrite(out.ptr ? out.ptr : "", 1, out.len, f); fputc(0, f); fclose(f); }
    }
    return buf_take(&out);
}

static char *render_flint_live_tool_tail(const chat_msgs *msgs, int start, const tool_schema_orders *tool_orders,
                                         ds4_think_mode think_mode) {
    buf out = {0};
    append_flint_conversation(&out, msgs, start, NULL, tool_orders, think_mode);
    return buf_take(&out);
}

/* generation_config.json: temperature 1.0, top_p 0.95, top_k 20 for the
 * knobs a request leaves out (ds4's own defaults are 1.0 / 1.0 / 0). */
static void flint_server_sampling_defaults(const request *r, float *temperature, int *top_k, float *top_p) {
    if (!r->temperature_set) *temperature = 1.0f;
    if (!r->top_k_set) *top_k = 20;
    if (!r->top_p_set) *top_p = 0.95f;
}

static bool flint_model_alias_disables_thinking(const char *model) {
    return model && (!strcmp(model, "flint-chat") || !strcmp(model, "flint-no-think") ||
                     !strcmp(model, "flint-nothink") || !strcmp(model, "recursive/flint-chat"));
}

static bool flint_model_alias_enables_thinking(const char *model) {
    return model && (!strcmp(model, "flint-reasoner") || !strcmp(model, "recursive/flint-reasoner"));
}

static bool flint_model_alias_known(const char *id) {
    return id && (!strcmp(id, "flint") || !strcmp(id, "recursive/flint") ||
                  flint_model_alias_disables_thinking(id) || flint_model_alias_enables_thinking(id));
}

/* Live KV reuse when clients send assistant turns back without reasoning.
 *
 * rec's template renders a history assistant turn as
 * <|assistant|>[<think>R</think>]CONTENT[<tool_call>...</tool_call>]: without
 * reasoning_content there is no think block at all, while the live KV holds
 * the sampled <think>R</think>.  Agent clients (tau2, most OpenAI-style
 * harnesses) echo content and tool calls byte for byte and drop the
 * reasoning, so the next request's token prefix parts from the live KV at
 * the <think> of the generation prompt and every turn re-prefills the whole
 * conversation.  The server's thinking-visible key (Qwen's path) fixes this
 * without touching the KV: remember the bytes the next request will render
 * for the current frontier, and continue from live KV with only the new
 * suffix tokenized.  For Flint that key is the generation prompt without its
 * trailing <think> (or <think></think> with thinking off), the content as
 * returned, and the sampled tool-call text, which the renderer replays by id.
 * The stop token is not in the KV; the suffix (<|user|>..., <|observation|>...)
 * evaluates it. */
static char *build_flint_visible_text(const request *r, const char *content, const tool_calls *calls) {
    if (!r || !r->prompt_text || r->kind != REQ_CHAT || r->image_count != 0) return NULL;
    if (r->api == API_RESPONSES || r->api == API_ANTHROPIC) return NULL;
    const size_t pt_len = strlen(r->prompt_text);
    static const char *const gens[] = { "<|assistant|><think></think>", "<|assistant|><think>" };
    size_t cut = 0;
    for (size_t i = 0; i < sizeof(gens) / sizeof(gens[0]) && !cut; i++) {
        const size_t n = strlen(gens[i]);
        if (pt_len >= n && !memcmp(r->prompt_text + pt_len - n, gens[i], n)) cut = pt_len - n + strlen("<|assistant|>");
    }
    if (!cut) return NULL;
    if (calls && calls->len > 0 && !(calls->raw_tool_text && calls->raw_tool_text[0])) return NULL;
    buf visible = {0};
    buf_append(&visible, r->prompt_text, cut);
    buf_puts(&visible, content ? content : "");
    if (calls && calls->len > 0) buf_puts(&visible, calls->raw_tool_text);
    return buf_take(&visible);
}

/* The gate for text answers: unlike GLM and DeepSeek, Flint's clients drop
 * the reasoning in tool conversations too, and thinking off needs the key as
 * well (the generation prompt's <think></think> is not in the history). */
static bool flint_should_remember_visible_checkpoint(const request *r, bool inside_thinking, const char *finish) {
    if (!r || r->kind != REQ_CHAT || r->api == API_RESPONSES || r->api == API_ANTHROPIC) return false;
    if (finish && (!strcmp(finish, "error") || !strcmp(finish, "length"))) return false;
    return !inside_thinking;
}

struct server_slot;   /* defined after this include */
static void thinking_live_remember(server *s, struct server_slot *slot, const char *visible_text, const request *req);
static void server_log(ds4_log_type type, const char *fmt, ...);

/* Tool-call turns stop at </tool_call>, so the frontier is the raw call text. */
static bool remember_flint_tool_turn_visible_checkpoint(server *s, struct server_slot *slot, const request *req,
                                                        const char *ctx, const char *finish, bool inside_thinking,
                                                        const char *content, const tool_calls *calls) {
    if (!finish || strcmp(finish, "tool_calls") != 0 || inside_thinking) return false;
    char *visible = build_flint_visible_text(req, content, calls);
    if (!visible) return false;
    thinking_live_remember(s, slot, visible, req);
    server_log(DS4_LOG_KVCACHE,
               "ds4-server: flint tool-turn visible checkpoint remembered ctx=%s visible=%zu",
               ctx, strlen(visible));
    free(visible);
    return true;
}

/* Argument values.  The template renders a tool-call argument as
 * `v | tojson if v is not string else v`, so the model writes string
 * values raw and everything else (arrays, objects, numbers, booleans,
 * null) as JSON text between <arg_value> tags.  The GLM reply parser
 * takes every value as a string, which hands clients {"item_ids":
 * "[\"1\"]"} for an array parameter.  Invert the template rule the way
 * the Qwen parser does: a parameter the tool schema declares as a
 * string stays text whatever it looks like; any other value that parses
 * as one complete non-string JSON value is passed through as JSON. */
static bool qwen_param_value_is_json(const char *s);   /* defined after this include */
static const char *flint_glm_arg_is_string(const tool_schema_orders *orders, const char *name,
                                           const char *key, const char *value) {
    const tool_schema_order *order = tool_schema_orders_find(orders, name);
    for (int i = 0; order && i < order->len; i++) {
        if (strcmp(order->prop[i], key)) continue;
        json_args schema = {0};
        bool declared_string = false;
        if (json_args_parse(order->prop_schema[i], &schema)) {
            for (int j = 0; j < schema.len; j++) {
                const json_arg *arg = &schema.v[j];
                if (!strcmp(arg->key, "type") && arg->is_string && !strcmp(arg->value, "string"))
                    declared_string = true;
            }
        }
        json_args_free(&schema);
        if (declared_string) return "true";
        break;
    }
    return qwen_param_value_is_json(value) ? "false" : "true";
}
