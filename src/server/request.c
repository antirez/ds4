#include "ds4_server_internal.h"



void random_tool_id(char *dst, size_t dstlen, api_style api) {
    unsigned char bytes[16];
    const char *prefix = api == API_ANTHROPIC ? "toolu_" : "call_";
    size_t pos = snprintf(dst, dstlen, "%s", prefix);
    if (pos >= dstlen) return;

    if (!random_bytes(bytes, sizeof(bytes))) {
        /* Fail closed: tool-call IDs must not be predictable, and with
         * getrandom() + /dev/urandom both unavailable something is deeply
         * wrong with the host. */
        ds4_die("random_bytes failed; cannot generate tool-call ids");
    }

    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(bytes) && pos + 2 < dstlen; i++) {
        dst[pos++] = hex[bytes[i] >> 4];
        dst[pos++] = hex[bytes[i] & 15];
    }
    dst[pos] = '\0';
}



/* The exact byte sequence a forced tool call is seeded with: close thinking,
 * open the tool_calls block, and (when a specific tool was requested) open
 * the named invoke. Used for BOTH the prompt suffix and the generate_job
 * output seed - they must stay byte-identical or the DSML tracker and the
 * prompt disagree about parser state. */
void request_forced_tool_seed(const request *r, buf *out) {
    buf_puts(out, "</think>\n\n" DS4_TOOL_CALLS_START "\n");
    if (r->forced_tool_name && r->forced_tool_name[0]) {
        buf_puts(out, DS4_INVOKE_START " name=\"");
        buf_puts(out, r->forced_tool_name);
        buf_puts(out, "\">\n");
    }
}

/* Rewrite the rendered prompt tail for a forced tool call: drop a trailing
 * "<think>" opener, then append the forced-tool seed (which begins with the
 * "</think>" close; if the render already ended with one, skip the extra). */
void request_apply_forced_tool_prefill(request *r) {
    if (!r->force_tool_call || !r->prompt_text) return;
    buf pt = {0};
    const char *base = r->prompt_text;
    size_t blen = strlen(base);
    if (blen >= 7 && !memcmp(base + blen - 7, "<think>", 7)) blen -= 7;
    buf_append(&pt, base, blen);
    buf seed = {0};
    request_forced_tool_seed(r, &seed);
    if (pt.len >= 8 && !memcmp(pt.ptr + pt.len - 8, "</think>", 8)) {
        buf_puts(&pt, seed.ptr + 8);   /* already closed: skip seed's close */
    } else {
        buf_puts(&pt, seed.ptr);
    }
    buf_free(&seed);
    free(r->prompt_text);
    r->prompt_text = buf_take(&pt);
}

void tool_memory_attach_to_messages(server *s, chat_msgs *msgs,
                                           tool_replay_stats *stats);


bool tool_memory_has_id(server *s, const char *id);


void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs);



void stop_list_clear(stop_list *stops);


bool id_list_contains(const stop_list *ids, const char *id);


void id_list_push_unique(stop_list *ids, const char *id);


void id_list_free(stop_list *ids);


bool responses_live_has_call_id(server *s, const char *id);


bool anthropic_live_has_call_id(server *s, const char *id);



static void tool_call_free(tool_call *tc) {
    free(tc->id);
    free(tc->name);
    free(tc->arguments);
    memset(tc, 0, sizeof(*tc));
}



void tool_calls_free(tool_calls *calls) {
    for (int i = 0; i < calls->len; i++) tool_call_free(&calls->v[i]);
    free(calls->raw_dsml);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}



void tool_calls_push(tool_calls *calls, tool_call tc) {
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 4;
        calls->v = server_xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = tc;
}



void chat_msg_add_tool_call_id(chat_msg *m, const char *id) {
    if (!m || !id || !id[0]) return;
    if (!m->tool_call_id) m->tool_call_id = xstrdup(id);
    for (int i = 0; i < m->tool_call_ids_len; i++) {
        if (m->tool_call_ids[i] && !strcmp(m->tool_call_ids[i], id)) return;
    }
    if (m->tool_call_ids_len == m->tool_call_ids_cap) {
        m->tool_call_ids_cap = m->tool_call_ids_cap ? m->tool_call_ids_cap * 2 : 2;
        m->tool_call_ids = server_xrealloc(m->tool_call_ids,
            (size_t)m->tool_call_ids_cap * sizeof(m->tool_call_ids[0]));
    }
    m->tool_call_ids[m->tool_call_ids_len++] = xstrdup(id);
}



static void chat_msg_free(chat_msg *m) {
    free(m->role);
    free(m->content);
    free(m->reasoning);
    free(m->tool_call_id);
    for (int i = 0; i < m->tool_call_ids_len; i++) free(m->tool_call_ids[i]);
    free(m->tool_call_ids);
    tool_calls_free(&m->calls);
    memset(m, 0, sizeof(*m));
}



void chat_msgs_free(chat_msgs *msgs) {
    for (int i = 0; i < msgs->len; i++) chat_msg_free(&msgs->v[i]);
    free(msgs->v);
    memset(msgs, 0, sizeof(*msgs));
}



void chat_msgs_push(chat_msgs *msgs, chat_msg msg) {
    if (msgs->len == msgs->cap) {
        msgs->cap = msgs->cap ? msgs->cap * 2 : 8;
        msgs->v = server_xrealloc(msgs->v, (size_t)msgs->cap * sizeof(msgs->v[0]));
    }
    msgs->v[msgs->len++] = msg;
}



static void tool_schema_order_free(tool_schema_order *o) {
    free(o->name);
    free(o->wire_name);
    free(o->tool_namespace);
    for (int i = 0; i < o->len; i++) free(o->prop[i]);
    free(o->prop);
    memset(o, 0, sizeof(*o));
}



void tool_schema_orders_free(tool_schema_orders *orders) {
    for (int i = 0; i < orders->len; i++) tool_schema_order_free(&orders->v[i]);
    free(orders->v);
    memset(orders, 0, sizeof(*orders));
}



static void tool_schema_order_prop_push(tool_schema_order *o, char *prop) {
    if (o->len == o->cap) {
        o->cap = o->cap ? o->cap * 2 : 8;
        o->prop = server_xrealloc(o->prop, (size_t)o->cap * sizeof(o->prop[0]));
    }
    o->prop[o->len++] = prop;
}



static int tool_schema_orders_find_index(const tool_schema_orders *orders, const char *name) {
    if (!orders || !name) return -1;
    for (int i = 0; i < orders->len; i++) {
        if (orders->v[i].name && !strcmp(orders->v[i].name, name)) return i;
    }
    return -1;
}



static void tool_schema_orders_push(tool_schema_orders *orders, tool_schema_order order) {
    int idx = tool_schema_orders_find_index(orders, order.name);
    if (idx >= 0) {
        tool_schema_order_free(&orders->v[idx]);
        orders->v[idx] = order;
        return;
    }
    if (orders->len == orders->cap) {
        orders->cap = orders->cap ? orders->cap * 2 : 8;
        orders->v = server_xrealloc(orders->v, (size_t)orders->cap * sizeof(orders->v[0]));
    }
    orders->v[orders->len++] = order;
}



const tool_schema_order *tool_schema_orders_find(const tool_schema_orders *orders, const char *name) {
    int idx = tool_schema_orders_find_index(orders, name);
    return idx >= 0 ? &orders->v[idx] : NULL;
}



void request_init(request *r, req_kind kind, int max_tokens) {
    memset(r, 0, sizeof(*r));
    r->kind = kind;
    r->api = API_OPENAI;
    r->model = xstrdup("deepseek-v4-flash");
    r->max_tokens = max_tokens;
    r->top_k = 0;
    r->temperature = DS4_DEFAULT_TEMPERATURE;
    r->top_p = DS4_DEFAULT_TOP_P;
    r->min_p = DS4_DEFAULT_MIN_P;
    r->think_mode = DS4_THINK_HIGH;
}



void request_free(request *r) {
    ds4_tokens_free(&r->prompt);
    free(r->model);
    free(r->forced_tool_name);
    for (int i = 0; i < r->stops.len; i++) free(r->stops.v[i]);
    free(r->stops.v);
    free(r->raw_body);
    free(r->prompt_text);
    stop_list_clear(&r->responses_live_call_ids);
    free(r->responses_live_call_ids.v);
    free(r->responses_live_suffix_text);
    stop_list_clear(&r->anthropic_live_call_ids);
    free(r->anthropic_live_call_ids.v);
    free(r->anthropic_live_suffix_text);
    tool_schema_orders_free(&r->tool_orders);
    memset(r, 0, sizeof(*r));
}



ds4_think_mode think_mode_from_enabled(bool enabled, ds4_think_mode effort) {
    if (!enabled || effort == DS4_THINK_NONE) return DS4_THINK_NONE;
    return effort == DS4_THINK_MAX ? DS4_THINK_MAX : DS4_THINK_HIGH;
}



bool parse_reasoning_effort_name(const char *s, ds4_think_mode *out) {
    if (!s) return false;
    if (!strcmp(s, "max")) {
        *out = DS4_THINK_MAX;
        return true;
    }
    if (!strcmp(s, "xhigh") || !strcmp(s, "high") ||
        !strcmp(s, "medium") || !strcmp(s, "low") ||
        !strcmp(s, "minimal"))
    {
        /* DS4 only exposes HIGH and MAX above zero, so "minimal" collapses to
         * the smallest non-zero level (HIGH). Callers that need *no* reasoning
         * must use "none" instead. */
        *out = DS4_THINK_HIGH;
        return true;
    }
    if (!strcmp(s, "none")) {
        *out = DS4_THINK_NONE;
        return true;
    }
    return false;
}



bool parse_reasoning_effort_value(const char **p, ds4_think_mode *out) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    char *effort = NULL;
    if (!json_string(p, &effort)) return false;
    bool ok = parse_reasoning_effort_name(effort, out);
    free(effort);
    return ok;
}



bool parse_thinking_control_value(const char **p, bool *thinking_enabled) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p == 't' || **p == 'f') return json_bool(p, thinking_enabled);
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "type")) {
            char *type = NULL;
            if (!json_string(p, &type)) {
                free(key);
                return false;
            }
            if (!strcmp(type, "enabled")) *thinking_enabled = true;
            else if (!strcmp(type, "disabled")) *thinking_enabled = false;
            free(type);
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}



bool parse_output_config_effort(const char **p, ds4_think_mode *effort) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "effort")) {
            if (!parse_reasoning_effort_value(p, effort)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}



bool model_alias_disables_thinking(const char *model) {
    return model && !strcmp(model, "deepseek-chat");
}



bool model_alias_enables_thinking(const char *model) {
    return model && !strcmp(model, "deepseek-reasoner");
}



const char *server_model_id_from_engine(ds4_engine *engine) {
    return ds4_engine_model_id(engine) == 1 ?
           "deepseek-v4-pro" : "deepseek-v4-flash";
}

const char *server_served_model_id(const server *s) {
    return s->served_model_id ? s->served_model_id
                              : server_model_id_from_engine(s->engine);
}

const char *server_served_model_name(const server *s) {
    return s->served_model_name ? s->served_model_name
                                : ds4_engine_model_name(s->engine);
}



void stop_list_clear(stop_list *stops) {
    for (int i = 0; i < stops->len; i++) free(stops->v[i]);
    stops->len = 0;
    stops->max_len = 0;
}



void stop_list_push(stop_list *stops, char *s) {
    if (!s || !s[0]) {
        free(s);
        return;
    }
    if (stops->len == stops->cap) {
        stops->cap = stops->cap ? stops->cap * 2 : 4;
        stops->v = server_xrealloc(stops->v, (size_t)stops->cap * sizeof(stops->v[0]));
    }
    size_t n = strlen(s);
    if (n > stops->max_len) stops->max_len = n;
    stops->v[stops->len++] = s;
}



bool parse_stop(const char **p, stop_list *out) {
    json_ws(p);
    stop_list_clear(out);
    if (**p == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        stop_list_push(out, s);
        return true;
    }
    if (**p != '[') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) return false;
            stop_list_push(out, s);
        } else if (!json_skip_value(p)) {
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}



bool stop_list_find_from(const stop_list *stops, const char *text,
                                size_t from, size_t *pos, size_t *len) {
    if (!stops->len || !text) return false;
    bool found = false;
    size_t best_pos = 0, best_len = 0;
    for (int i = 0; i < stops->len; i++) {
        const char *p = strstr(text + from, stops->v[i]);
        if (!p) continue;
        size_t ppos = (size_t)(p - text);
        size_t plen = strlen(stops->v[i]);
        if (!found || ppos < best_pos) {
            found = true;
            best_pos = ppos;
            best_len = plen;
        }
    }
    if (!found) return false;
    *pos = best_pos;
    *len = best_len;
    return true;
}



size_t stop_list_stream_safe_len(const stop_list *stops, size_t text_len) {
    /* Streaming cannot emit the last max_stop_len-1 bytes yet: a stop sequence
     * may start there and finish in the next token.  The final flush releases
     * this small tail once generation ends without a stop hit. */
    if (!stops->len || stops->max_len <= 1) return text_len;
    const size_t hold = stops->max_len - 1;
    return text_len > hold ? text_len - hold : 0;
}



static int utf8_expected_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}



/* Tokenizers can split a multi-byte UTF-8 character across two tokens.  If an
 * SSE delta ends at that boundary, some clients replace the incomplete byte
 * sequence with U+FFFD and later send the corrupted text back, destroying KV
 * cache prefix matches.  Hold only the trailing incomplete character; the next
 * generated token will complete it. */
size_t utf8_stream_safe_len(const char *s, size_t start,
                                   size_t limit, bool final) {
    if (final || !s || limit <= start) return limit;

    size_t p = limit;
    int cont = 0;
    while (p > start && cont < 4 &&
           (((unsigned char)s[p - 1] & 0xc0) == 0x80))
    {
        p--;
        cont++;
    }

    if (p == limit) {
        return utf8_expected_len((unsigned char)s[limit - 1]) > 1 ?
               limit - 1 : limit;
    }
    if (p == start && (((unsigned char)s[p] & 0xc0) == 0x80)) return start;

    size_t lead = p - 1;
    int need = utf8_expected_len((unsigned char)s[lead]);
    return (limit - lead) < (size_t)need ? lead : limit;
}



bool parse_stream_options(const char **p, bool *include_usage) {
    json_ws(p);
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "include_usage")) {
            if (!json_bool(p, include_usage)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}



static bool parse_function_call(const char **p, tool_call *tc) {
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') {
            free(key);
            goto bad;
        }
        (*p)++;
        if (!strcmp(key, "name")) {
            free(tc->name);
            if (!json_string(p, &tc->name)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "arguments")) {
            free(tc->arguments);
            json_ws(p);
            if (**p == '"') {
                if (!json_string(p, &tc->arguments)) {
                    free(key);
                    goto bad;
                }
            } else if (!json_raw_value(p, &tc->arguments)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;
    return true;
bad:
    return false;
}



static bool parse_tool_calls_value(const char **p, tool_calls *calls) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        tool_call tc = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto bad;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto bad;
            }
            (*p)++;
            if (!strcmp(key, "id")) {
                free(tc.id);
                if (!json_string(p, &tc.id)) {
                    free(key);
                    goto bad;
                }
            } else if (!strcmp(key, "function")) {
                if (!parse_function_call(p, &tc)) {
                    free(key);
                    goto bad;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto bad;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto bad;
        (*p)++;
        if (tc.name && tc.arguments) {
            tool_calls_push(calls, tc);
            memset(&tc, 0, sizeof(tc));
        }
        tool_call_free(&tc);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
bad:
        tool_call_free(&tc);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}



static void append_raw_json_line(buf *b, const char *json) {
    if (!json || !json[0]) return;
    if (b->len) buf_putc(b, '\n');
    buf_puts(b, json);
}



void json_escape(buf *b, const char *s);



static char *openai_function_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        char *value = NULL;
        if (!json_string(&p, &key)) return NULL;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return NULL;
        }
        p++;
        if (!strcmp(key, "function")) {
            free(key);
            if (!json_raw_value(&p, &value)) return NULL;
            return value;
        }
        free(key);
        if (!json_skip_value(&p)) return NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return NULL;
}



static char *responses_special_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;

    char *type = NULL;
    char *description = NULL;
    char *parameters = NULL;
    char *out = NULL;

    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(&p, &type)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "description")) {
            free(description);
            if (!json_string(&p, &description)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "parameters")) {
            free(parameters);
            if (!json_raw_value(&p, &parameters)) {
                free(key);
                goto done;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }

    if (type && !strcmp(type, "tool_search")) {
        buf b = {0};
        buf_puts(&b, "{\"name\":\"tool_search\",\"description\":");
        json_escape(&b, description ? description : "Search available tools.");
        buf_puts(&b, ",\"parameters\":");
        buf_puts(&b, parameters ? parameters :
                 "{\"type\":\"object\",\"properties\":{}}");
        buf_putc(&b, '}');
        out = buf_take(&b);
    }

done:
    free(type);
    free(description);
    free(parameters);
    return out;
}



static char *responses_namespace_function_schema_from_tool(const char *raw,
                                                           const char *namespace,
                                                           char **wire_name) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;

    char *type = NULL;
    char *name = NULL;
    char *description = NULL;
    char *parameters = NULL;
    char *out = NULL;

    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(&p, &type)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(&p, &name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "description")) {
            free(description);
            if (!json_string(&p, &description)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "parameters") || !strcmp(key, "input_schema")) {
            free(parameters);
            if (!json_raw_value(&p, &parameters)) {
                free(key);
                goto done;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }

    if ((!type || !strcmp(type, "function")) && namespace && name && name[0]) {
        buf prompt_name = {0};
        buf_puts(&prompt_name, namespace);
        buf_puts(&prompt_name, name);

        buf b = {0};
        buf_puts(&b, "{\"name\":");
        json_escape(&b, prompt_name.ptr ? prompt_name.ptr : name);
        buf_puts(&b, ",\"description\":");
        json_escape(&b, description ? description : "");
        buf_puts(&b, ",\"parameters\":");
        buf_puts(&b, parameters ? parameters :
                 "{\"type\":\"object\",\"properties\":{}}");
        buf_putc(&b, '}');
        out = buf_take(&b);
        if (wire_name) *wire_name = xstrdup(name);
        buf_free(&prompt_name);
    }

done:
    free(type);
    free(name);
    free(description);
    free(parameters);
    return out;
}



static bool parse_schema_properties(const char *json, tool_schema_order *order) {
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            return false;
        }
        p++;
        if (!strcmp(key, "properties")) {
            free(key);
            json_ws(&p);
            if (*p != '{') return false;
            p++;
            json_ws(&p);
            while (*p && *p != '}') {
                char *prop = NULL;
                if (!json_string(&p, &prop)) return false;
                json_ws(&p);
                if (*p != ':') {
                    free(prop);
                    return false;
                }
                p++;
                tool_schema_order_prop_push(order, prop);
                if (!json_skip_value(&p)) return false;
                json_ws(&p);
                if (*p == ',') p++;
                json_ws(&p);
            }
            if (*p != '}') return false;
            p++;
        } else {
            free(key);
            if (!json_skip_value(&p)) return false;
        }
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return *p == '}';
}



static void tool_schema_orders_add_json_wire(tool_schema_orders *orders,
                                             const char *json,
                                             const char *namespace,
                                             const char *wire_name,
                                             bool responses_tool_search) {
    if (!orders || !json) return;
    const char *p = json;
    json_ws(&p);
    if (*p != '{') return;
    p++;
    tool_schema_order order = {0};
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "name")) {
            free(order.name);
            if (!json_string(&p, &order.name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "input_schema") || !strcmp(key, "parameters")) {
            char *schema = NULL;
            if (!json_raw_value(&p, &schema)) {
                free(key);
                goto done;
            }
            parse_schema_properties(schema, &order);
            free(schema);
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (order.name) {
        if (namespace && namespace[0]) order.tool_namespace = xstrdup(namespace);
        if (wire_name && wire_name[0]) order.wire_name = xstrdup(wire_name);
        order.responses_tool_search = responses_tool_search;
        tool_schema_orders_push(orders, order);
        memset(&order, 0, sizeof(order));
    }
done:
    tool_schema_order_free(&order);
}



void tool_schema_orders_add_json(tool_schema_orders *orders, const char *json) {
    tool_schema_orders_add_json_wire(orders, json, NULL, NULL, false);
}



static bool append_responses_namespace_tool_schemas(buf *schemas,
                                                    tool_schema_orders *orders,
                                                    const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return false;
    p++;

    char *type = NULL;
    char *name = NULL;
    char *tools = NULL;
    bool appended = false;

    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto done;
        json_ws(&p);
        if (*p != ':') {
            free(key);
            goto done;
        }
        p++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(&p, &type)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(&p, &name)) {
                free(key);
                goto done;
            }
        } else if (!strcmp(key, "tools")) {
            free(tools);
            if (!json_raw_value(&p, &tools)) {
                free(key);
                goto done;
            }
        } else if (!json_skip_value(&p)) {
            free(key);
            goto done;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }

    if (!type || strcmp(type, "namespace") || !name || !tools) goto done;

    const char *tp = tools;
    json_ws(&tp);
    if (*tp != '[') goto done;
    tp++;
    json_ws(&tp);
    while (*tp && *tp != ']') {
        char *tool_raw = NULL;
        if (!json_raw_value(&tp, &tool_raw)) goto done;
        char *wire_name = NULL;
        char *schema =
            responses_namespace_function_schema_from_tool(tool_raw, name, &wire_name);
        if (schema) {
            append_raw_json_line(schemas, schema);
            tool_schema_orders_add_json_wire(orders, schema, name, wire_name, false);
            appended = true;
        }
        free(schema);
        free(wire_name);
        free(tool_raw);
        json_ws(&tp);
        if (*tp == ',') tp++;
        json_ws(&tp);
    }

done:
    free(type);
    free(name);
    free(tools);
    return appended;
}



/* OpenAI wraps tools as {"type":"function","function":{...}}. Anthropic sends
 * the function schema directly as {"name":...,"input_schema":...}. The DS4
 * prompt wants one raw function schema per line, so unwrap OpenAI tools and keep
 * already-direct schemas unchanged. Responses can additionally group tools in a
 * namespace item; those are flattened for DSML prompt rendering while preserving
 * their client-facing name and namespace for response output. */
bool parse_tools_value(const char **p, char **out, tool_schema_orders *orders) {
    json_ws(p);
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') return false;
    (*p)++;
    buf schemas = {0};

    json_ws(p);
    while (**p && **p != ']') {
        char *raw = NULL;
        if (!json_raw_value(p, &raw)) goto bad;
        char *function = openai_function_schema_from_tool(raw);
        if (function) {
            append_raw_json_line(&schemas, function);
            tool_schema_orders_add_json(orders, function);
        } else if (!append_responses_namespace_tool_schemas(&schemas, orders, raw)) {
            char *special = responses_special_schema_from_tool(raw);
            if (special) {
                append_raw_json_line(&schemas, special);
                tool_schema_orders_add_json_wire(orders, special,
                                                 NULL, NULL, true);
            } else {
                append_raw_json_line(&schemas, raw);
                tool_schema_orders_add_json(orders, raw);
            }
            free(special);
        }
        free(function);
        free(raw);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto bad;
    (*p)++;
    *out = buf_take(&schemas);
    return true;
bad:
    buf_free(&schemas);
    return false;
}



bool parse_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto fail;
            }
            (*p)++;
            if (!strcmp(key, "role")) {
                free(msg.role);
                if (!json_string(p, &msg.role)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "content")) {
                free(msg.content);
                if (!json_content(p, &msg.content)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "reasoning_content")) {
                free(msg.reasoning);
                if (!json_content(p, &msg.reasoning)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "tool_call_id")) {
                char *id = NULL;
                if (!json_string(p, &id)) {
                    free(key);
                    goto fail;
                }
                chat_msg_add_tool_call_id(&msg, id);
                free(id);
            } else if (!strcmp(key, "tool_calls")) {
                tool_calls_free(&msg.calls);
                if (!parse_tool_calls_value(p, &msg.calls)) {
                    free(key);
                    goto fail;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto fail;
        (*p)++;
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
        chat_msgs_push(msgs, msg);
        memset(&msg, 0, sizeof(msg));
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
fail:
        chat_msg_free(&msg);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}



void append_tool_result_text(buf *b, const char *s);



static bool append_anthropic_block_content(buf *dst, const char *text) {
    if (!text || !text[0]) return true;
    buf_puts(dst, text);
    return true;
}



/* Anthropic content is block-structured, while the engine consumes one compact
 * chat_msg per role.  Parsing collapses text/thinking into strings, converts
 * assistant tool_use blocks to tool_calls, and keeps tool_result blocks as
 * escaped text because DS4 sees tool results in its chat template. */
static bool parse_anthropic_content_block(const char **p, const char *role, chat_msg *msg) {
    (void)role;
    if (**p != '{') return false;
    (*p)++;
    char *type = NULL;
    char *text = NULL;
    char *thinking = NULL;
    char *id = NULL;
    char *name = NULL;
    char *input = NULL;
    char *tool_result = NULL;

    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') {
            free(key);
            goto bad;
        }
        (*p)++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(p, &type)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "text")) {
            free(text);
            if (!json_content(p, &text)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "thinking")) {
            free(thinking);
            if (!json_content(p, &thinking)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "id") || !strcmp(key, "tool_use_id")) {
            free(id);
            if (!json_string(p, &id)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(p, &name)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "input")) {
            free(input);
            if (!json_raw_value(p, &input)) {
                free(key);
                goto bad;
            }
        } else if (!strcmp(key, "content")) {
            free(tool_result);
            if (!json_content(p, &tool_result)) {
                free(key);
                goto bad;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;

    /* JSON object member order is not meaningful.  Some Anthropic-compatible
     * clients serialize a message as {"content": ..., "role": ...}, so the
     * caller may not know the enclosing role yet while parsing content blocks.
     * Classify protocol blocks by their own "type" field; later rendering and
     * validation use the final message role. */
    if (type && !strcmp(type, "tool_use")) {
        tool_call tc = {0};
        tc.id = id ? xstrdup(id) : NULL;
        tc.name = name ? xstrdup(name) : xstrdup("");
        tc.arguments = input ? xstrdup(input) : xstrdup("{}");
        tool_calls_push(&msg->calls, tc);
    } else if (type && !strcmp(type, "tool_result")) {
        chat_msg_add_tool_call_id(msg, id);
        buf b = {0};
        buf_puts(&b, msg->content ? msg->content : "");
        buf_puts(&b, "<tool_result>");
        append_tool_result_text(&b, tool_result);
        buf_puts(&b, "</tool_result>");
        free(msg->content);
        msg->content = buf_take(&b);
    } else {
        if (text) {
            buf b = {0};
            buf_puts(&b, msg->content ? msg->content : "");
            append_anthropic_block_content(&b, text);
            free(msg->content);
            msg->content = buf_take(&b);
        }
        if (thinking) {
            buf b = {0};
            buf_puts(&b, msg->reasoning ? msg->reasoning : "");
            append_anthropic_block_content(&b, thinking);
            free(msg->reasoning);
            msg->reasoning = buf_take(&b);
        }
    }

    free(type);
    free(text);
    free(thinking);
    free(id);
    free(name);
    free(input);
    free(tool_result);
    return true;
bad:
    free(type);
    free(text);
    free(thinking);
    free(id);
    free(name);
    free(input);
    free(tool_result);
    return false;
}



static bool parse_anthropic_content(const char **p, chat_msg *msg) {
    json_ws(p);
    if (**p == '"') return json_string(p, &msg->content);
    if (json_lit(p, "null")) {
        msg->content = xstrdup("");
        return true;
    }
    if (**p != '[') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) return false;
            buf b = {0};
            buf_puts(&b, msg->content ? msg->content : "");
            buf_puts(&b, s);
            free(msg->content);
            msg->content = buf_take(&b);
            free(s);
        } else if (**p == '{') {
            if (!parse_anthropic_content_block(p, msg->role ? msg->role : "", msg)) return false;
        } else if (!json_skip_value(p)) {
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    if (!msg->content) msg->content = xstrdup("");
    return true;
}



bool parse_anthropic_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;

    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto fail;
            json_ws(p);
            if (**p != ':') {
                free(key);
                goto fail;
            }
            (*p)++;
            if (!strcmp(key, "role")) {
                free(msg.role);
                if (!json_string(p, &msg.role)) {
                    free(key);
                    goto fail;
                }
            } else if (!strcmp(key, "content")) {
                free(msg.content);
                msg.content = NULL;
                if (!parse_anthropic_content(p, &msg)) {
                    free(key);
                    goto fail;
                }
            } else if (!json_skip_value(p)) {
                free(key);
                goto fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') goto fail;
        (*p)++;
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
        chat_msgs_push(msgs, msg);
        memset(&msg, 0, sizeof(msg));
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
        continue;
fail:
        chat_msg_free(&msg);
        return false;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}



static bool anthropic_system_part_is_private(const char *s) {
    return s && !strncmp(s, "x-anthropic-", 12);
}



static void append_anthropic_system_part(buf *b, const char *s) {
    if (!s || !s[0] || anthropic_system_part_is_private(s)) return;
    if (b->len && b->ptr[b->len - 1] != '\n') buf_putc(b, '\n');
    buf_puts(b, s);
}



static bool parse_anthropic_system_object(const char **p, buf *out) {
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') {
            free(key);
            return false;
        }
        (*p)++;
        if (!strcmp(key, "text")) {
            char *text = NULL;
            if (!json_string(p, &text)) {
                free(key);
                return false;
            }
            append_anthropic_system_part(out, text);
            free(text);
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}



bool parse_anthropic_system(const char **p, char **out) {
    json_ws(p);
    buf b = {0};
    if (**p == '"') {
        char *text = NULL;
        if (!json_string(p, &text)) return false;
        append_anthropic_system_part(&b, text);
        free(text);
        *out = buf_take(&b);
        return true;
    }
    if (json_lit(p, "null")) {
        *out = xstrdup("");
        return true;
    }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *text = NULL;
            if (!json_string(p, &text)) goto bad;
            append_anthropic_system_part(&b, text);
            free(text);
        } else if (**p == '{') {
            if (!parse_anthropic_system_object(p, &b)) goto bad;
        } else if (!json_skip_value(p)) {
            goto bad;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') goto bad;
    (*p)++;
    *out = buf_take(&b);
    return true;
bad:
    buf_free(&b);
    return false;
}

