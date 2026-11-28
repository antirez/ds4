/* Inkling 2 Mini chat layer for ds4_server.c (included from there; see docs/INKLING2.md, "Chat").
 *
 * The model has no generation prompt.  A prompt is a list of messages
 *     <role token><kind token>text<|end_message|>
 * (role: message_system/user/model/tool, kind: content_text/content_thinking/
 * content_xml/content_invoke_tool_json) and the model itself starts its turn with
 * <|message_model|>.  This file renders a request exactly as the official
 * chat_template.jinja does (checked token for token against llama.cpp's
 * /apply-template + /tokenize), parses generated text back into reasoning, content
 * and tool calls the way llama.cpp's common/parsers/inkling.cpp does, and tracks a
 * streamed generation block by block.
 *
 * Everything here works on plain text; special tokens are written as their text
 * and are cut out by the tokenizer (ds4_tokenize_rendered_chat), like llama.cpp
 * does for the rendered prompt. */

#define IK_END       "<|end_message|>"
#define IK_HDR_MODEL "<|message_model|>"
#define IK_THINK     "<|content_thinking|>"
#define IK_TEXT      "<|content_text|>"
#define IK_TOOLJSON  "<|content_invoke_tool_json|>"
#define IK_EOS_TEXT  "<|content_model_end_sampling|>"

/* ---- JSON canonicalisation (jinja tojson as in llama.cpp common/jinja/value.cpp) ---- */

static void ik_json_put_string(buf *b, const char *s, size_t n) {
    buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\b': buf_puts(b, "\\b"); break;
        case '\f': buf_puts(b, "\\f"); break;
        case '\n': buf_puts(b, "\\n"); break;
        case '\r': buf_puts(b, "\\r"); break;
        case '\t': buf_puts(b, "\\t"); break;
        default:
            if (c < 0x20) buf_printf(b, "\\u%04x", c);
            else buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

/* A double as Python's repr()/llama.cpp float_repr writes it: shortest digits that read back,
 * fixed notation for 1e-4 <= |x| < 1e16, else d.ddde+XX. */
static void ik_float_repr(buf *b, double val) {
    if (val == 0.0) {
        buf_puts(b, signbit(val) ? "-0.0" : "0.0");
        return;
    }
    if (isnan(val)) { buf_puts(b, "NaN"); return; }
    if (isinf(val)) { buf_puts(b, val < 0 ? "-Infinity" : "Infinity"); return; }
    const double mag = fabs(val);
    char digits[40];
    int exp10 = 0;
    for (int prec = 0; prec <= 16; prec++) {
        char t[64];
        snprintf(t, sizeof(t), "%.*e", prec, mag);
        const char *e = strchr(t, 'e');
        if (strtod(t, NULL) != mag && prec != 16) continue;
        size_t n = 0;
        for (const char *q = t; q < e; q++)
            if (*q >= '0' && *q <= '9') digits[n++] = *q;
        digits[n] = '\0';
        exp10 = atoi(e + 1);
        break;
    }
    size_t n = strlen(digits);
    while (n > 1 && digits[n - 1] == '0') digits[--n] = '\0';
    const int point = exp10 + 1;
    if (signbit(val)) buf_putc(b, '-');
    if (point > -4 && point <= 16) {
        if (point <= 0) {
            buf_puts(b, "0.");
            for (int i = 0; i < -point; i++) buf_putc(b, '0');
            buf_puts(b, digits);
        } else if ((size_t)point >= n) {
            buf_puts(b, digits);
            for (size_t i = n; i < (size_t)point; i++) buf_putc(b, '0');
            buf_puts(b, ".0");
        } else {
            buf_append(b, digits, (size_t)point);
            buf_putc(b, '.');
            buf_puts(b, digits + point);
        }
    } else {
        buf_putc(b, digits[0]);
        if (n > 1) {
            buf_putc(b, '.');
            buf_puts(b, digits + 1);
        }
        buf_printf(b, "e%c%02d", exp10 < 0 ? '-' : '+', abs(exp10));
    }
}

static void ik_json_put_number(buf *b, const char *s, size_t n) {
    char tmp[512];
    if (n >= sizeof(tmp)) {   /* absurdly long literal: keep it as written */
        buf_append(b, s, n);
        return;
    }
    memcpy(tmp, s, n);
    tmp[n] = '\0';
    bool is_float = false;
    for (size_t i = 0; i < n; i++)
        if (tmp[i] == '.' || tmp[i] == 'e' || tmp[i] == 'E') is_float = true;
    if (!is_float) {
        errno = 0;
        char *end = NULL;
        const long long v = strtoll(tmp, &end, 10);
        if (errno == 0 && *end == '\0') {
            if (v == 0 && tmp[0] == '-') buf_puts(b, "-0.0");   /* nlohmann reads -0 as the float -0.0 */
            else buf_printf(b, "%lld", v);
            return;
        }
        errno = 0;
        end = NULL;
        if (tmp[0] != '-') {
            (void)strtoull(tmp, &end, 10);
            if (errno == 0 && *end == '\0') {   /* fits uint64: llama.cpp wraps it to int64, the HF template prints it */
                buf_puts(b, tmp);
                return;
            }
        }
    }
    ik_float_repr(b, strtod(tmp, NULL));
}

typedef struct {
    char *key;
    const char *val;
    int ord;           /* position in the source, for dropping repeated keys */
} ik_member;

/* qsort order: by key bytes, then by source position (so the first and the last of a repeated key are at the group ends) */
static int ik_member_qcmp(const void *pa, const void *pb) {
    const ik_member *a = pa, *b = pb;
    const int c = strcmp(a->key, b->key);
    return c ? c : (a->ord > b->ord) - (a->ord < b->ord);
}

/* A repeated key keeps its first position and takes the last value (nlohmann).  O(n log n): sort a copy, mark the later
 * members of every group dead, then compact the original order. */
static void ik_members_dedupe(ik_member *v, int *n_io) {
    const int n = *n_io;
    if (n < 2) return;
    ik_member *tmp = xmalloc((size_t)n * sizeof(*tmp));
    memcpy(tmp, v, (size_t)n * sizeof(*tmp));
    qsort(tmp, (size_t)n, sizeof(*tmp), ik_member_qcmp);
    bool dups = false;
    for (int i = 0; i < n;) {
        int j = i + 1;
        while (j < n && !strcmp(tmp[i].key, tmp[j].key)) j++;
        if (j - i > 1) {
            dups = true;
            v[tmp[i].ord].val = tmp[j - 1].val;                      /* first position, last value */
            for (int k = i + 1; k < j; k++) { free(v[tmp[k].ord].key); v[tmp[k].ord].key = NULL; }
        }
        i = j;
    }
    free(tmp);
    if (!dups) return;
    int w = 0;
    for (int i = 0; i < n; i++)
        if (v[i].key) v[w++] = v[i];
    *n_io = w;
}

/* Reads the members of the object at *p (positioned on '{'); values stay as pointers into the source. */
static bool ik_json_members(const char **p, ik_member **out, int *n_out) {
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    ik_member *v = NULL;
    int n = 0, cap = 0;
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
        json_ws(p);
        const char *val = *p;
        if (!json_skip_value(p)) {
            free(key);
            goto bad;
        }
        if (n == cap) {
            cap = cap ? cap * 2 : 8;
            v = xrealloc(v, (size_t)cap * sizeof(v[0]));
        }
        v[n].key = key;
        v[n].val = val;
        v[n].ord = n;
        n++;
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;
    ik_members_dedupe(v, &n);
    *out = v;
    *n_out = n;
    return true;
bad:
    for (int i = 0; i < n; i++) free(v[i].key);
    free(v);
    return false;
}

static void ik_members_free(ik_member *v, int n) {
    for (int i = 0; i < n; i++) free(v[i].key);
    free(v);
}

static void ik_members_sort(ik_member *v, int n) {   /* by key bytes; keys are unique after ik_json_members */
    qsort(v, (size_t)n, sizeof(*v), ik_member_qcmp);
}

/* Dump one JSON value compactly ("," ":" separators, no ASCII escaping).  sort_keys sorts every object
 * level, as tojson(sort_keys=true) does. */
static bool ik_json_dump_depth(buf *b, const char **p, bool sort_keys, int depth) {
    if (depth > 64) return false;
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        ik_json_put_string(b, s, strlen(s));
        free(s);
        return true;
    }
    if (**p == '{') {
        ik_member *v = NULL;
        int n = 0;
        if (!ik_json_members(p, &v, &n)) return false;
        if (sort_keys) ik_members_sort(v, n);
        buf_putc(b, '{');
        bool ok = true;
        for (int i = 0; i < n && ok; i++) {
            if (i) buf_putc(b, ',');
            ik_json_put_string(b, v[i].key, strlen(v[i].key));
            buf_putc(b, ':');
            const char *q = v[i].val;
            ok = ik_json_dump_depth(b, &q, sort_keys, depth + 1);
        }
        buf_putc(b, '}');
        ik_members_free(v, n);
        return ok;
    }
    if (**p == '[') {
        (*p)++;
        buf_putc(b, '[');
        json_ws(p);
        bool first = true;
        while (**p && **p != ']') {
            if (!first) buf_putc(b, ',');
            first = false;
            if (!ik_json_dump_depth(b, p, sort_keys, depth + 1)) return false;
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != ']') return false;
        (*p)++;
        buf_putc(b, ']');
        return true;
    }
    if (json_lit(p, "true")) { buf_puts(b, "true"); return true; }
    if (json_lit(p, "false")) { buf_puts(b, "false"); return true; }
    if (json_lit(p, "null")) { buf_puts(b, "null"); return true; }
    const char *start = *p;
    if (**p == '-') (*p)++;
    while (**p && (isdigit((unsigned char)**p) || **p == '.' || **p == 'e' || **p == 'E' ||
                   **p == '+' || **p == '-')) (*p)++;
    if (*p == start) return false;
    ik_json_put_number(b, start, (size_t)(*p - start));
    return true;
}

/* Is the JSON text an object (after optional whitespace)? */
static bool ik_json_is_object(const char *json) {
    const char *p = json ? json : "";
    json_ws(&p);
    if (*p != '{') return false;
    const char *q = p;
    if (!json_skip_value(&q)) return false;
    json_ws(&q);
    return *q == '\0';
}

/* ---- tool declaration: [{"description":..,"name":..,"parameters":..,"type":"function"},..] ---- */

static bool ik_render_tool_specs(buf *b, const char *tool_schemas) {
    buf_putc(b, '[');
    bool first = true;
    const char *p = tool_schemas ? tool_schemas : "";
    json_ws(&p);
    while (*p) {
        char *raw = NULL;
        if (!json_raw_value(&p, &raw)) return false;
        const char *q = raw;
        ik_member *m = NULL;
        int n = 0;
        if (!ik_json_members(&q, &m, &n)) {
            free(raw);
            return false;
        }
        const char *desc = NULL, *name = NULL, *params = NULL;
        for (int i = 0; i < n; i++) {
            if (!strcmp(m[i].key, "description")) desc = m[i].val;
            else if (!strcmp(m[i].key, "name")) name = m[i].val;
            else if (!strcmp(m[i].key, "parameters")) params = m[i].val;
            else if (!strcmp(m[i].key, "input_schema") && !params) params = m[i].val;   /* Anthropic tools */
        }
        if (!first) buf_putc(b, ',');
        first = false;
        buf_puts(b, "{\"description\":");
        /* jinja: `fn.description if fn.description is defined and fn.description else ""` */
        bool desc_done = false;
        if (desc) {
            const char *d = desc;
            json_ws(&d);
            if (strncmp(d, "null", 4) && strncmp(d, "false", 5) && strncmp(d, "\"\"", 2) &&
                strncmp(d, "0", 1) && strncmp(d, "[]", 2) && strncmp(d, "{}", 2)) {
                ik_json_dump_depth(b, &d, true, 0);
                desc_done = true;
            }
        }
        if (!desc_done) buf_puts(b, "\"\"");
        buf_puts(b, ",\"name\":");
        if (name) {
            const char *d = name;
            if (!ik_json_dump_depth(b, &d, true, 0)) buf_puts(b, "\"\"");
        } else {
            buf_puts(b, "null");
        }
        buf_puts(b, ",\"parameters\":");
        bool params_done = false;
        if (params) {
            const char *d = params;
            json_ws(&d);
            if (strncmp(d, "null", 4) && strncmp(d, "false", 5)) {
                ik_json_dump_depth(b, &d, true, 0);
                params_done = true;
            }
        }
        if (!params_done) buf_puts(b, "{}");
        buf_puts(b, ",\"type\":\"function\"}");
        ik_members_free(m, n);
        free(raw);
        json_ws(&p);
    }
    buf_putc(b, ']');
    return true;
}

/* {"k":v,...} of a tool call's arguments: keys sorted (dictsort), values compact in their own order. */
static bool ik_render_call_args(buf *b, const char *args) {
    const char *p = args ? args : "";
    json_ws(&p);
    if (!*p || !strncmp(p, "null", 4)) {
        buf_puts(b, "{}");
        return true;
    }
    ik_member *m = NULL;
    int n = 0;
    if (!ik_json_members(&p, &m, &n)) return false;
    ik_members_sort(m, n);
    buf_putc(b, '{');
    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
        if (i) buf_putc(b, ',');
        ik_json_put_string(b, m[i].key, strlen(m[i].key));
        buf_putc(b, ':');
        const char *q = m[i].val;
        ok = ik_json_dump_depth(b, &q, false, 0);
    }
    buf_putc(b, '}');
    ik_members_free(m, n);
    return ok;
}

/* ---- rendering ---- */

static const char *ik_role_token(const char *role) {
    if (!role) return NULL;
    if (!strcmp(role, "system") || !strcmp(role, "developer")) return "<|message_system|>";
    if (!strcmp(role, "user")) return "<|message_user|>";
    if (!strcmp(role, "assistant")) return "<|message_model|>";
    if (!strcmp(role, "tool") || !strcmp(role, "function")) return "<|message_tool|>";
    return NULL;
}

static void ik_put_block(buf *out, const char *role_tok, const char *kind, const char *text, size_t n) {
    buf_puts(out, role_tok);
    buf_puts(out, kind);
    buf_append(out, text, n);
    buf_puts(out, IK_END);
}

static void ik_put_effort(buf *out, ds4_think_mode mode) {
    buf_puts(out, "<|message_system|><|content_text|>Thinking effort level: ");
    buf_puts(out, ds4_inkling2_effort_text(mode));
    buf_puts(out, IK_END);
}

/* id -> name of the last tool call with that id, over the whole conversation, built once per render (a search per tool message made a
 * conversation with thousands of tool results quadratic). */
typedef struct {
    const char **id, **name;
    size_t cap;
} ik_name_map;

static uint64_t ik_id_hash(const char *s) {
    uint64_t h = 1469598103934665603ull;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; }
    return h;
}

static void ik_name_map_build(ik_name_map *mp, const chat_msgs *msgs) {
    size_t n = 0;
    for (int i = 0; i < msgs->len; i++)
        if (msgs->v[i].role && !strcmp(msgs->v[i].role, "assistant")) n += (size_t)msgs->v[i].calls.len;
    mp->cap = 16;
    while (mp->cap < n * 2) mp->cap *= 2;
    mp->id = xmalloc(mp->cap * sizeof(mp->id[0]));
    mp->name = xmalloc(mp->cap * sizeof(mp->name[0]));
    memset(mp->id, 0, mp->cap * sizeof(mp->id[0]));
    memset(mp->name, 0, mp->cap * sizeof(mp->name[0]));
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *a = &msgs->v[i];
        if (!a->role || strcmp(a->role, "assistant")) continue;
        for (int k = 0; k < a->calls.len; k++) {
            const tool_call *tc = &a->calls.v[k];
            if (!tc->id || !tc->name) continue;
            size_t at = (size_t)(ik_id_hash(tc->id) & (mp->cap - 1));
            while (mp->id[at] && strcmp(mp->id[at], tc->id)) at = (at + 1) & (mp->cap - 1);
            mp->id[at] = tc->id;
            mp->name[at] = tc->name;   /* a later call with the same id wins, as in the linear search */
        }
    }
}

static const char *ik_name_map_find(const ik_name_map *mp, const char *id) {
    size_t at = (size_t)(ik_id_hash(id) & (mp->cap - 1));
    while (mp->id[at]) {
        if (!strcmp(mp->id[at], id)) return mp->name[at];
        at = (at + 1) & (mp->cap - 1);
    }
    return NULL;
}

/* The name a tool message is filed under: its own `name`, else the function name of the tool call with its id. */
static const char *ik_tool_message_name(const chat_msgs *msgs, const chat_msg *m, const ik_name_map *mp) {
    if (m->name && m->name[0]) return m->name;
    const char *id = m->tool_call_id;
    if (!id && m->tool_call_ids_len > 0) id = m->tool_call_ids[0];
    if (!id || !id[0]) return NULL;
    if (mp && mp->cap) return ik_name_map_find(mp, id);
    const char *found = NULL;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *a = &msgs->v[i];
        if (strcmp(a->role, "assistant")) continue;
        for (int k = 0; k < a->calls.len; k++)
            if (a->calls.v[k].id && !strcmp(a->calls.v[k].id, id) && a->calls.v[k].name) found = a->calls.v[k].name;
    }
    return found;
}

/* Text of the content parts of a message (a string is one part). */
static void ik_put_content_parts(buf *out, const char *head, const chat_msg *m) {
    const char *content = m->content ? m->content : "";
    if (m->n_parts >= 2) {
        size_t start = 0;
        const size_t clen = strlen(content);
        for (int i = 0; i < m->n_parts; i++) {
            size_t end = m->part_ends[i];
            if (end > clen) end = clen;
            ik_put_block(out, head, IK_TEXT, content + start, end - start);
            start = end;
        }
        return;
    }
    ik_put_block(out, head, IK_TEXT, content, strlen(content));
}

/* Returns false (and fills err) on what the template rejects: unknown role, tool arguments that are not a JSON object. */
static bool ik_validate_messages(const chat_msgs *msgs, char *err, size_t errlen) {
    if (!msgs || msgs->len == 0) {
        snprintf(err, errlen, "messages must not be empty");
        return false;
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!ik_role_token(m->role)) {
            snprintf(err, errlen, "Unknown message role: %s", m->role ? m->role : "");
            return false;
        }
        if (strcmp(m->role, "assistant")) continue;
        for (int k = 0; k < m->calls.len; k++) {
            const char *a = m->calls.v[k].arguments;
            const char *q = a ? a : "";
            json_ws(&q);
            if (!*q || (!ik_json_is_object(q) && strncmp(q, "null", 4))) {
                snprintf(err, errlen, "tool call arguments must be a JSON object");
                return false;
            }
            buf tmp = {0};   /* the renderer stops at 64 levels: refuse deeper arguments instead of rendering broken JSON */
            const char *dq = q;
            const bool depth_ok = ik_json_dump_depth(&tmp, &dq, false, 0);
            buf_free(&tmp);
            if (!depth_ok) {
                snprintf(err, errlen, "tool call arguments are invalid or nested more than 64 levels");
                return false;
            }
        }
    }
    return true;
}

/* The tool list must render: one entry that is not an object, or schemas nested more than 64 levels, used to drop every tool silently. */
static bool ik_validate_tools(const char *tool_schemas, char *err, size_t errlen) {
    if (!tool_schemas || !tool_schemas[0]) return true;
    buf tmp = {0};
    const bool ok = ik_render_tool_specs(&tmp, tool_schemas);
    buf_free(&tmp);
    if (!ok) snprintf(err, errlen, "tools must be objects whose schemas nest at most 64 levels");
    return ok;
}

static char *render_inkling_chat_prompt_text(const chat_msgs *msgs, const char *tool_schemas, ds4_think_mode think_mode) {
    buf out = {0};
    if (tool_schemas && tool_schemas[0]) {
        buf_puts(&out, "<|message_system|>tool_declare<|content_xml|>");
        if (!ik_render_tool_specs(&out, tool_schemas)) {   /* unparsable schema: declare nothing rather than garbage */
            buf_free(&out);
        } else {
            buf_puts(&out, IK_END);
        }
    }
    bool effort_done = false;
    ik_name_map names = {0};
    if (msgs) ik_name_map_build(&names, msgs);
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        const char *rt = ik_role_token(m->role);
        if (!rt) continue;
        const bool is_system = !strcmp(rt, "<|message_system|>");
        const bool is_assistant = !strcmp(rt, IK_HDR_MODEL);
        const bool is_tool = !strcmp(rt, "<|message_tool|>");
        if (!effort_done && !is_system) {
            ik_put_effort(&out, think_mode);
            effort_done = true;
        }
        buf head = {0};
        buf_puts(&head, rt);
        if (is_tool) {
            const char *name = ik_tool_message_name(msgs, m, &names);
            if (name) buf_puts(&head, name);
        }
        const bool has_reasoning = is_assistant && m->reasoning && m->reasoning[0];
        const bool has_calls = is_assistant && m->calls.len > 0;
        if (has_reasoning) ik_put_block(&out, head.ptr, IK_THINK, m->reasoning, strlen(m->reasoning));
        const bool has_content = m->content && m->content[0];
        if (has_content || m->n_parts >= 2 || !(is_assistant && (has_calls || has_reasoning)))
            ik_put_content_parts(&out, head.ptr, m);
        if (has_calls) {
            for (int k = 0; k < m->calls.len; k++) {
                const tool_call *tc = &m->calls.v[k];
                const char *name = tc->name ? tc->name : "";
                buf_puts(&out, IK_HDR_MODEL);
                buf_puts(&out, name);
                buf_puts(&out, IK_TOOLJSON);
                buf_puts(&out, "{\"name\":");
                ik_json_put_string(&out, name, strlen(name));
                buf_puts(&out, ",\"args\":");
                if (!ik_render_call_args(&out, tc->arguments)) buf_puts(&out, "{}");
                buf_puts(&out, "}" IK_END);
            }
        }
        if (is_assistant) buf_puts(&out, IK_EOS_TEXT);
        buf_free(&head);
    }
    if (!effort_done) ik_put_effort(&out, think_mode);
    free(names.id);
    free(names.name);
    return buf_take(&out);
}

/* ---- parsing generated text ---- */

static bool ik_starts(const char *s, const char *lit) {
    return !strncmp(s, lit, strlen(lit));
}

static void ik_trim_ws(const char **s, const char **e) {
    while (*s < *e && isspace((unsigned char)**s)) (*s)++;
    while (*e > *s && isspace((unsigned char)(*e)[-1])) (*e)--;
}

/* The tool call JSON {"name":..,"args":{..}} (after the marker) as a call: name, and the args text as written (trimmed).
 * False for what the turn parser calls malformed: not JSON, no name, args that are neither an object nor null.  The
 * last of repeated keys wins.  The stream tracker checks every call it streamed against this at the end marker. */
static bool ik_parse_tool_json(const char *body, tool_call *out) {
    const char *q = body;
    ik_member *m = NULL;
    int n = 0;
    if (!ik_json_members(&q, &m, &n)) return false;
    tool_call tc = {0};
    for (int i = 0; i < n; i++) {
        if (!strcmp(m[i].key, "name")) {
            const char *v = m[i].val;
            char *nm = NULL;
            if (json_string(&v, &nm)) { free(tc.name); tc.name = nm; }
        } else if (!strcmp(m[i].key, "args")) {
            const char *v = m[i].val;
            const char *s0 = v;
            if (json_skip_value(&v)) {
                const char *s1 = v;
                ik_trim_ws(&s0, &s1);
                free(tc.arguments);
                tc.arguments = xstrndup(s0, (size_t)(s1 - s0));
            }
        }
    }
    ik_members_free(m, n);
    if (!tc.name) { tool_call_free(&tc); return false; }
    if (!tc.arguments) tc.arguments = xstrdup("{}");
    /* arguments that are not a JSON object: the request validator refuses them on the next turn, so this is a malformed call */
    const char *aq = tc.arguments;
    json_ws(&aq);
    if (!ik_json_is_object(aq) && strncmp(aq, "null", 4)) { tool_call_free(&tc); return false; }
    *out = tc;
    return true;
}

/* The model's turn is a list of blocks, each opened by <|message_model|> and closed by <|end_message|>:
 *   <|content_thinking|>reasoning   <|content_text|>answer   NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}
 * Reasoning blocks and answer blocks are concatenated, a tool call takes its name from the JSON.  An unclosed
 * last block (output cut off) keeps what was generated, except a tool call, which is dropped.  Text that does not
 * start with a block header fails the parse (the caller then returns it as plain content).  With want_tools off a
 * tool block is skipped, as llama.cpp does when no tools are active.  Returns false only for a malformed tool call
 * or text that is not a block list. */
static bool parse_inkling_generated_message_ex(const char *text, bool want_tools,
                                               char **content_out, char **reasoning_out,
                                               tool_calls *calls) {
    const char *p = text ? text : "";
    buf content = {0}, reasoning = {0};
    bool ok = true;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        if (ik_starts(p, IK_EOS_TEXT)) {
            p += strlen(IK_EOS_TEXT);
            continue;
        }
        if (!ik_starts(p, IK_HDR_MODEL)) {
            ok = false;
            break;
        }
        p += strlen(IK_HDR_MODEL);
        const char *end = strstr(p, IK_END);
        const bool closed = end != NULL;
        const char *stop = closed ? end : p + strlen(p);
        if (ik_starts(p, IK_THINK)) {
            buf_append(&reasoning, p + strlen(IK_THINK), (size_t)(stop - p) - strlen(IK_THINK));
        } else if (ik_starts(p, IK_TEXT)) {
            buf_append(&content, p + strlen(IK_TEXT), (size_t)(stop - p) - strlen(IK_TEXT));
        } else {
            const char *tj = strstr(p, IK_TOOLJSON);
            if (!tj || tj >= stop) {
                /* a header followed by something that is neither block kind: not a model block list */
                if (closed || !strncmp(p, "<|", 2)) { ok = false; break; }
                break;   /* partial header of a cut-off turn */
            }
            if (!closed) break;   /* cut-off tool call: nothing to run */
            const char *body = tj + strlen(IK_TOOLJSON);
            if (want_tools) {
                tool_call tc = {0};
                if (!ik_parse_tool_json(body, &tc)) { ok = false; break; }
                tool_calls_push(calls, tc);
            }
        }
        if (!closed) break;
        p = end + strlen(IK_END);
    }
    if (!ok) {
        buf_free(&content);
        buf_free(&reasoning);
        return false;
    }
    free(*content_out);
    free(*reasoning_out);
    *content_out = buf_take(&content);
    *reasoning_out = reasoning.len ? buf_take(&reasoning) : NULL;
    buf_free(&reasoning);
    return true;
}

/* ---- Anthropic Messages request mapping ----
 *
 * Anthropic `thinking:{type:"enabled",budget_tokens:N}` has no counterpart in the template, which only has the
 * effort number.  The budget picks an effort tier (an explicit output_config.effort or reasoning_effort wins):
 *   N < 2048 -> low (0.2)   N < 8192 -> medium (0.7)   N < 32000 -> high (0.9)   N >= 32000 -> max (0.99)
 * `enabled` without a budget is high, `disabled` is none (the model still thinks a little at none). */

/* budget_tokens of a raw `thinking` object, or -1 */
static int ik_anthropic_thinking_budget(const char *raw) {
    const char *q = raw ? raw : "";
    json_ws(&q);
    if (*q != '{') return -1;
    ik_member *m = NULL;
    int n = 0, budget = -1;
    if (!ik_json_members(&q, &m, &n)) return -1;
    for (int i = 0; i < n; i++) {
        if (strcmp(m[i].key, "budget_tokens")) continue;
        const char *v = m[i].val;
        double b = 0.0;
        if (json_number(&v, &b) && b >= 0.0) budget = b > 2e9 ? 2000000000 : (int)b;   /* a negative budget is no budget */
    }
    ik_members_free(m, n);
    return budget;
}

static ds4_think_mode ik_anthropic_budget_mode(int budget) {
    if (budget < 0) return DS4_THINK_HIGH;
    if (budget < 2048) return DS4_THINK_LOW;
    if (budget < 8192) return DS4_THINK_MEDIUM;
    if (budget < 32000) return DS4_THINK_HIGH;
    return DS4_THINK_MAX;
}

/* The Anthropic parser folds tool_result blocks into the user message that carries them, as "<tool_result>..</tool_result>"
 * spans.  The template wants what an OpenAI request has: one tool message per result, then the rest of the user
 * text as a user message of its own.  Done in place. */
static void ik_split_tool_results(chat_msgs *msgs) {
    chat_msgs out = {0};
    for (int i = 0; i < msgs->len; i++) {
        chat_msg *m = &msgs->v[i];
        if (!m->role || strcmp(m->role, "user") || m->tool_results_len == 0) {
            chat_msgs_push(&out, *m);
            continue;
        }
        const char *content = m->content ? m->content : "";
        const size_t clen = strlen(content);
        buf rest = {0};
        size_t *pe = NULL;      /* part ends inside `rest`: one block per text block, as for an array content in chat */
        int n_pe = 0, cap_pe = 0, pi = 0;
        size_t at = 0;
        for (int k = 0; k <= m->tool_results_len; k++) {
            size_t seg_end = clen;
            if (k < m->tool_results_len) {
                const tool_result_span *sp = &m->tool_results[k];
                seg_end = sp->begin < clen ? sp->begin : clen;
            }
            /* the text between the previous result and this one, cut at the text block ends */
            size_t s0 = at;
            while (s0 < seg_end) {
                size_t s1 = seg_end;
                while (pi < m->n_parts && m->part_ends[pi] <= s0) pi++;   /* part ends are ascending and s0 only grows */
                if (pi < m->n_parts && m->part_ends[pi] < s1) s1 = m->part_ends[pi];
                buf_append(&rest, content + s0, s1 - s0);
                if (n_pe == cap_pe) { cap_pe = cap_pe ? cap_pe * 2 : 16; pe = xrealloc(pe, (size_t)cap_pe * sizeof(pe[0])); }
                pe[n_pe++] = rest.len;
                s0 = s1;
            }
            if (k < m->tool_results_len) {
                const tool_result_span *sp = &m->tool_results[k];
                size_t e = sp->end < clen ? sp->end : clen;
                if (e > at) at = e;
                if (seg_end > at) at = seg_end;
                chat_msg tm = {0};
                tm.role = xstrdup("tool");
                tm.content = xstrdup(sp->content ? sp->content : "");
                if (sp->id && sp->id[0]) tm.tool_call_id = xstrdup(sp->id);
                chat_msgs_push(&out, tm);
            }
        }
        const bool has_rest = rest.len > 0 || m->images.len > 0;
        if (!has_rest) free(pe);
        if (has_rest) {
            chat_msg um = {0};
            um.role = xstrdup("user");
            um.content = rest.len ? buf_take(&rest) : xstrdup("");
            if (n_pe >= 2) { um.part_ends = pe; um.n_parts = n_pe; um.cap_parts = n_pe; } else free(pe);
            um.images = m->images;          /* moves with the text */
            memset(&m->images, 0, sizeof(m->images));
            chat_msgs_push(&out, um);
        }
        buf_free(&rest);
        chat_msg_free(m);
    }
    free(msgs->v);
    *msgs = out;
}

/* ---- streaming ----
 *
 * The raw text grows; ik_stream_step() emits reasoning and answer text as it arrives and holds back the
 * bytes that could still turn out to be the start of a marker.  With want_tools set, tool calls are streamed
 * too: a "tool_name" event once the call's JSON has shown its name and the start of its args object, then
 * "tool_args" events with the raw text of the args object as it is generated, then "tool_done" when the object
 * closes.  The concatenated tool_args text is byte for byte the arguments the non-streaming parser returns.
 * A call whose JSON does not have the shape {"name":"..","args":{..}} (other key order, args not an object,
 * a header that never reaches the marker) is not streamed: the stream stops streaming calls from there on
 * and `tools_streamed` tells the caller how many calls it already sent; the caller sends the rest whole at
 * the end of the turn from the parsed message.  A call cut off by the token limit after its name was sent
 * stays partial on the wire (the parser drops such a call, and the finish reason is "length"). */

typedef enum {
    IK_ST_HEADER,   /* expecting <|message_model|> */
    IK_ST_KIND,     /* after the header: thinking, text or a tool name */
    IK_ST_THINK,
    IK_ST_TEXT,
    IK_ST_TOOL,     /* name or JSON of a tool call: sent at the end */
    IK_ST_RAW,      /* not a block list: pass everything through as content */
} ik_stream_state;

typedef enum {
    IK_TP_MARKER,   /* looking for the tool JSON marker after the header name */
    IK_TP_HEAD,     /* waiting for {"name":"..","args":{ */
    IK_TP_ARGS,     /* streaming the args object */
    IK_TP_TAIL,     /* args sent: skip to the end marker */
} ik_tool_phase;

typedef struct {
    ik_stream_state state;
    size_t pos;
    bool want_tools;        /* set by the caller before the first step; zero-initialised streams do not stream calls */
    bool tool_broken;       /* a call could not be streamed: no later call is streamed either */
    int tools_streamed;     /* calls whose name was sent */
    int tool_index;         /* index of the tool block being read */
    ik_tool_phase tp;
    size_t args_emit;       /* args text already sent up to here (offset in raw) */
    size_t args_scan;       /* args text scanned up to here */
    int depth;
    bool in_str, esc;
    size_t body_start;      /* offset in raw of the JSON after the tool marker */
    size_t args_from;       /* offset of the args object */
    int tool_bad;           /* streamed calls the turn parser does not agree with (checked at the end marker) */
} ik_stream;

/* Reads {"name":"..","args":{ at s[0..avail).  1: found (name allocated, *args_off = offset of the args '{');
 * 0: need more text; -1: not that shape. */
static int ik_tool_head(const char *s, size_t avail, char **name, size_t *args_off) {
    size_t i = 0;
#define IK_TH_WS() do { while (i < avail && isspace((unsigned char)s[i])) i++; if (i >= avail) return 0; } while (0)
#define IK_TH_LIT(lit) do { const size_t n_ = strlen(lit); \
        if (avail - i < n_) return !strncmp(s + i, lit, avail - i) ? 0 : -1; \
        if (strncmp(s + i, lit, n_)) return -1; i += n_; } while (0)
    IK_TH_WS();
    IK_TH_LIT("{");
    IK_TH_WS();
    IK_TH_LIT("\"name\"");
    IK_TH_WS();
    IK_TH_LIT(":");
    IK_TH_WS();
    if (s[i] != '"') return -1;
    size_t j = i + 1;
    for (;;) {
        if (j >= avail) return 0;
        if (s[j] == '\\') { j += 2; continue; }
        if (s[j] == '"') break;
        j++;
    }
    {
        char *copy = xstrndup(s + i, j - i + 1);
        const char *q = copy;
        char *nm = NULL;
        const bool ok = json_string(&q, &nm);
        free(copy);
        if (!ok) { free(nm); return -1; }
        *name = nm;
    }
    i = j + 1;
#define IK_TH_FAIL(rc) do { free(*name); *name = NULL; return (rc); } while (0)
    {
        size_t k = i;
        while (k < avail && isspace((unsigned char)s[k])) k++;
        if (k >= avail) IK_TH_FAIL(0);
        if (s[k] != ',') IK_TH_FAIL(-1);
        k++;
        while (k < avail && isspace((unsigned char)s[k])) k++;
        static const char key[] = "\"args\"";
        const size_t n_ = sizeof(key) - 1;
        if (avail - k < n_) IK_TH_FAIL(!strncmp(s + k, key, avail - k) ? 0 : -1);
        if (strncmp(s + k, key, n_)) IK_TH_FAIL(-1);
        k += n_;
        while (k < avail && isspace((unsigned char)s[k])) k++;
        if (k >= avail) IK_TH_FAIL(0);
        if (s[k] != ':') IK_TH_FAIL(-1);
        k++;
        while (k < avail && isspace((unsigned char)s[k])) k++;
        if (k >= avail) IK_TH_FAIL(0);
        if (s[k] != '{') IK_TH_FAIL(-1);
        *args_off = k;
    }
#undef IK_TH_FAIL
#undef IK_TH_WS
#undef IK_TH_LIT
    return 1;
}

static bool ik_is_partial_of(const char *s, size_t avail, const char *lit) {
    return avail < strlen(lit) && !strncmp(s, lit, avail);
}

/* Advances the stream state over raw[0..raw_len).  Each emit callback gets (field, text, len). */
typedef bool (*ik_emit_fn)(void *ctx, const char *field, const char *text, size_t len);

static bool ik_stream_step(ik_stream *st, const char *raw, size_t raw_len, bool final,
                           ik_emit_fn emit, void *ctx) {
    for (;;) {
        const char *s = raw + st->pos;
        const size_t avail = raw_len - st->pos;
        switch (st->state) {
        case IK_ST_HEADER:
            if (!avail) return true;
            if (isspace((unsigned char)*s)) { st->pos++; continue; }   /* the non-stream parser skips whitespace between blocks too */
            if (avail >= strlen(IK_EOS_TEXT) && ik_starts(s, IK_EOS_TEXT)) { st->pos += strlen(IK_EOS_TEXT); continue; }
            if (avail >= strlen(IK_HDR_MODEL) && ik_starts(s, IK_HDR_MODEL)) {
                st->pos += strlen(IK_HDR_MODEL);
                st->state = IK_ST_KIND;
                continue;
            }
            if (!final && (ik_is_partial_of(s, avail, IK_HDR_MODEL) || ik_is_partial_of(s, avail, IK_EOS_TEXT)))
                return true;
            st->state = IK_ST_RAW;
            continue;
        case IK_ST_KIND:
            if (avail >= strlen(IK_THINK) && ik_starts(s, IK_THINK)) {
                st->pos += strlen(IK_THINK);
                st->state = IK_ST_THINK;
                continue;
            }
            if (avail >= strlen(IK_TEXT) && ik_starts(s, IK_TEXT)) {
                st->pos += strlen(IK_TEXT);
                st->state = IK_ST_TEXT;
                continue;
            }
            if (!avail) return true;
            if (!final && (ik_is_partial_of(s, avail, IK_THINK) || ik_is_partial_of(s, avail, IK_TEXT)))
                return true;
            st->state = IK_ST_TOOL;
            continue;
        case IK_ST_THINK:
        case IK_ST_TEXT: {
            const char *field = st->state == IK_ST_THINK ? "reasoning_content" : "content";
            const char *end = NULL;
            for (size_t i = 0; i + strlen(IK_END) <= avail; i++)
                if (!memcmp(s + i, IK_END, strlen(IK_END))) { end = s + i; break; }
            size_t take;
            if (end) take = (size_t)(end - s);
            else if (final) take = avail;
            else {
                const size_t hold = strlen(IK_END) - 1;
                take = avail > hold ? avail - hold : 0;
                take = utf8_stream_safe_len(s, 0, take, false);
            }
            if (take && !emit(ctx, field, s, take)) return false;
            st->pos += take;
            if (!end) return true;
            st->pos += strlen(IK_END);
            st->state = IK_ST_HEADER;
            continue;
        }
        case IK_ST_TOOL: {
            const char *end = NULL;
            if (st->want_tools && !st->tool_broken && st->tp != IK_TP_TAIL) {
                if (st->tp == IK_TP_MARKER) {
                    const char *mk = NULL;
                    for (size_t i = 0; i < avail && !mk; i++) {
                        if (s[i] != '<') continue;
                        if (avail - i >= strlen(IK_END) && !memcmp(s + i, IK_END, strlen(IK_END))) break;   /* the block ended without a JSON marker */
                        if (avail - i >= strlen(IK_TOOLJSON) && !memcmp(s + i, IK_TOOLJSON, strlen(IK_TOOLJSON))) mk = s + i;
                    }
                    if (mk) {
                        st->pos += (size_t)(mk - s) + strlen(IK_TOOLJSON);
                        st->tp = IK_TP_HEAD;
                        continue;
                    }
                    bool ended = false;
                    for (size_t i = 0; i + strlen(IK_END) <= avail; i++)
                        if (s[i] == '<' && !memcmp(s + i, IK_END, strlen(IK_END))) { ended = true; break; }
                    if (ended || avail > 4096) { st->tool_broken = true; continue; }
                    if (final) { st->pos = raw_len; return true; }
                    return true;   /* wait for the marker; the header name is short */
                }
                if (st->tp == IK_TP_HEAD) {
                    char *name = NULL;
                    size_t args_off = 0;
                    const int rc = ik_tool_head(s, avail, &name, &args_off);
                    if (rc < 0) { st->tool_broken = true; continue; }
                    if (rc == 0) {
                        if (final) { st->pos = raw_len; return true; }
                        if (avail > 65536) { st->tool_broken = true; continue; }   /* a name that long is not a name */
                        return true;
                    }
                    st->tools_streamed++;   /* counted before the event: the emit callback reads the index as tools_streamed - 1 */
                    const bool ok = emit(ctx, "tool_name", name, strlen(name));
                    free(name);
                    if (!ok) return false;
                    st->body_start = st->pos;
                    st->args_from = st->pos + args_off;
                    st->args_emit = st->args_scan = st->pos + args_off;
                    st->depth = 0;
                    st->in_str = st->esc = false;
                    st->tp = IK_TP_ARGS;
                    continue;
                }
                /* IK_TP_ARGS: scan the args object, send what is complete */
                size_t i = st->args_scan;
                bool closed = false;
                for (; i < raw_len && !closed; i++) {
                    const char c = raw[i];
                    if (st->in_str) {
                        if (st->esc) st->esc = false;
                        else if (c == '\\') st->esc = true;
                        else if (c == '"') st->in_str = false;
                    } else if (c == '"') st->in_str = true;
                    else if (c == '{' || c == '[') st->depth++;
                    else if (c == '}' || c == ']') {
                        if (--st->depth == 0) closed = true;
                    }
                }
                st->args_scan = i;
                size_t upto = (closed || final) ? i : utf8_stream_safe_len(raw, st->args_emit, i, false);
                if (upto > st->args_emit) {
                    if (!emit(ctx, "tool_args", raw + st->args_emit, upto - st->args_emit)) return false;
                    st->args_emit = upto;
                }
                if (!closed) { st->pos = st->args_emit; return true; }
                if (!emit(ctx, "tool_done", "", 0)) return false;
                st->pos = i;
                st->tp = IK_TP_TAIL;
                continue;
            }
            for (size_t i = 0; i + strlen(IK_END) <= avail; i++)
                if (!memcmp(s + i, IK_END, strlen(IK_END))) { end = s + i; break; }
            if (!end) {
                /* keep the last strlen(IK_END)-1 bytes: the end marker may arrive split across two calls */
                const size_t hold = strlen(IK_END) - 1;
                st->pos = final || avail <= hold ? (final ? raw_len : st->pos) : raw_len - hold;
                return true;
            }
            if (st->want_tools && !st->tool_broken && st->tp == IK_TP_TAIL) {
                /* the call is complete: it must be the call the turn parser will return, or the stream sent a wrong one */
                char *body = xstrndup(raw + st->body_start, (size_t)(end - raw) - st->body_start);
                tool_call tc = {0};
                char *name = NULL;
                size_t ao = 0;
                const bool parsed = ik_parse_tool_json(body, &tc);
                const bool head = ik_tool_head(body, strlen(body), &name, &ao) == 1;
                if (!parsed || !head || strcmp(tc.name, name) || strlen(tc.arguments) > st->args_scan - st->args_from ||
                    memcmp(tc.arguments, raw + st->args_from, strlen(tc.arguments)) ||
                    st->args_scan - st->args_from != strlen(tc.arguments))
                    st->tool_bad++;
                if (parsed) tool_call_free(&tc);
                free(name);
                free(body);
            }
            st->pos += (size_t)(end - s) + strlen(IK_END);
            st->state = IK_ST_HEADER;
            st->tool_index++;
            st->tp = IK_TP_MARKER;
            continue;
        }
        case IK_ST_RAW:
            if (avail) {
                size_t take = final ? avail : utf8_stream_safe_len(s, 0, avail, false);
                if (take && !emit(ctx, "content", s, take)) return false;
                st->pos += take;
            }
            return true;
        }
    }
}
