/* Model-free tests for the Inkling 2 chat layer (ds4_inkling2_chat.inc), included by ds4_server.c in the unit-test build.
 *
 * Reference vectors: tests/test-vectors/inkling2-chat (one JSON file per case), captured from llama.cpp (see the README there).  Each
 * file holds an OpenAI chat request, the prompt llama.cpp rendered for it (/apply-template), and for tool cases the raw
 * text the model generated plus the message llama.cpp parsed from it.  The tests check that ds4 renders the same
 * prompt text and parses the same message.  Token ids are compared by tests/inkling2_chat_tokens.py (it needs a GGUF).
 *
 *   DS4_INKLING2_CHATVEC   directory of vectors (default tests/test-vectors/inkling2-chat)
 *   DS4_INKLING2_DUMP_DIR  write every rendered prompt to DIR/<case>.txt (input of inkling2_chat_tokens.py) */

#include <dirent.h>

typedef struct {
    char *name;
    char *request;
    char *prompt;          /* what llama.cpp rendered; NULL when it returned an error */
    char *prompt_ds4;      /* set when ds4 deliberately differs (llama.cpp bug); compared instead of prompt */
    char *ds4;             /* "match" (default), "error" (ds4 must refuse the request), "skip" (documented difference) */
    char *raw;             /* generated text with special tokens */
    char *content;
    char *reasoning;
    bool has_expect;
    tool_calls calls;
} ik_fixture;

static void ik_fixture_free(ik_fixture *f) {
    free(f->name); free(f->request); free(f->prompt); free(f->prompt_ds4); free(f->ds4);
    free(f->raw); free(f->content); free(f->reasoning);
    tool_calls_free(&f->calls);
    memset(f, 0, sizeof(*f));
}

static bool ik_fixture_string_or_null(const char **p, char **out) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    free(*out);
    return json_string(p, out);
}

static bool ik_fixture_parse(const char *text, ik_fixture *f) {
    const char *p = text;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p != ':') { free(key); return false; }
        p++;
        bool ok = true;
        if (!strcmp(key, "name")) ok = ik_fixture_string_or_null(&p, &f->name);
        else if (!strcmp(key, "request")) { free(f->request); ok = json_raw_value(&p, &f->request); }
        else if (!strcmp(key, "prompt")) ok = ik_fixture_string_or_null(&p, &f->prompt);
        else if (!strcmp(key, "prompt_ds4")) ok = ik_fixture_string_or_null(&p, &f->prompt_ds4);
        else if (!strcmp(key, "ds4")) ok = ik_fixture_string_or_null(&p, &f->ds4);
        else if (!strcmp(key, "raw_generation")) ok = ik_fixture_string_or_null(&p, &f->raw);
        else if (!strcmp(key, "expect")) {
            json_ws(&p);
            if (json_lit(&p, "null")) { free(key); json_ws(&p); if (*p == ',') p++; json_ws(&p); continue; }
            f->has_expect = true;
            if (*p != '{') { free(key); return false; }
            p++;
            json_ws(&p);
            while (*p && *p != '}') {
                char *k2 = NULL;
                if (!json_string(&p, &k2)) { free(key); return false; }
                json_ws(&p);
                if (*p != ':') { free(k2); free(key); return false; }
                p++;
                if (!strcmp(k2, "content")) ok = ik_fixture_string_or_null(&p, &f->content);
                else if (!strcmp(k2, "reasoning_content")) ok = ik_fixture_string_or_null(&p, &f->reasoning);
                else if (!strcmp(k2, "calls")) {
                    json_ws(&p);
                    if (*p != '[') ok = false;
                    else {
                        p++;
                        json_ws(&p);
                        while (ok && *p && *p != ']') {
                            tool_call tc = {0};
                            if (*p != '{') { ok = false; break; }
                            p++;
                            json_ws(&p);
                            while (*p && *p != '}') {
                                char *k3 = NULL;
                                if (!json_string(&p, &k3)) { ok = false; break; }
                                json_ws(&p);
                                if (*p != ':') { free(k3); ok = false; break; }
                                p++;
                                if (!strcmp(k3, "name")) ok = json_string_replace(&p, &tc.name);
                                else if (!strcmp(k3, "arguments")) ok = json_string_replace(&p, &tc.arguments);
                                else ok = json_skip_value(&p);
                                free(k3);
                                if (!ok) break;
                                json_ws(&p);
                                if (*p == ',') p++;
                                json_ws(&p);
                            }
                            if (ok && *p == '}') p++; else ok = false;
                            tool_calls_push(&f->calls, tc);
                            json_ws(&p);
                            if (*p == ',') p++;
                            json_ws(&p);
                        }
                        if (ok && *p == ']') p++; else ok = false;
                    }
                } else ok = json_skip_value(&p);
                free(k2);
                if (!ok) break;
                json_ws(&p);
                if (*p == ',') p++;
                json_ws(&p);
            }
            if (ok && *p == '}') p++; else ok = false;
        } else ok = json_skip_value(&p);
        free(key);
        if (!ok) return false;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return *p == '}';
}

static char *ik_test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *s = xmalloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, fp) != (size_t)n) { free(s); fclose(fp); return NULL; }
    s[n] = '\0';
    fclose(fp);
    return s;
}

static size_t ik_first_diff(const char *a, const char *b) {
    size_t i = 0;
    while (a[i] && a[i] == b[i]) i++;
    return i;
}

static bool ik_same_json(const char *a, const char *b) {
    buf x = {0}, y = {0};
    const char *pa = a, *pb = b;
    bool ok = ik_json_dump_depth(&x, &pa, true, 0) && ik_json_dump_depth(&y, &pb, true, 0) &&
              x.ptr && y.ptr && !strcmp(x.ptr, y.ptr);
    buf_free(&x);
    buf_free(&y);
    return ok;
}

typedef struct {
    buf reasoning, content;
    buf names[16], args[16];   /* streamed tool calls */
    int n_calls, n_done;
    int bad_events;            /* events out of order */
} ik_collect;

static bool ik_collect_emit(void *ctx, const char *field, const char *text, size_t len) {
    ik_collect *c = ctx;
    if (!strcmp(field, "tool_name")) {
        if (c->n_calls >= 16 || c->n_done != c->n_calls) { c->bad_events++; return true; }
        buf_append(&c->names[c->n_calls++], text, len);
    } else if (!strcmp(field, "tool_args")) {
        if (c->n_calls == 0 || c->n_done == c->n_calls) { c->bad_events++; return true; }
        buf_append(&c->args[c->n_calls - 1], text, len);
    } else if (!strcmp(field, "tool_done")) {
        if (c->n_calls == 0 || c->n_done == c->n_calls) c->bad_events++;
        else c->n_done++;
    } else {
        buf_append(!strcmp(field, "content") ? &c->content : &c->reasoning, text, len);
    }
    return true;
}

static void ik_collect_free(ik_collect *c) {
    buf_free(&c->content);
    buf_free(&c->reasoning);
    for (int i = 0; i < 16; i++) { buf_free(&c->names[i]); buf_free(&c->args[i]); }
}

/* Feed the raw text to the stream parser in pieces of `step` bytes (the server hands it a growing prefix). */
static ik_stream ik_stream_collect_ex(const char *raw, size_t step, bool want_tools, ik_collect *out) {
    ik_stream st = {0};
    st.want_tools = want_tools;
    const size_t n = strlen(raw);
    for (size_t at = step; at < n; at += step) ik_stream_step(&st, raw, at, false, ik_collect_emit, out);
    ik_stream_step(&st, raw, n, true, ik_collect_emit, out);
    return st;
}

static void ik_stream_collect(const char *raw, size_t step, ik_collect *out) {
    ik_stream_collect_ex(raw, step, false, out);
}

/* The streamed calls must be the parsed calls: same count, names, and args text byte for byte. */
static bool ik_streamed_calls_match(const ik_collect *c, const ik_stream *st, const tool_calls *calls,
                                    const char *content, const char *reasoning) {
    if (c->bad_events || st->tool_bad || st->tools_streamed != c->n_calls || c->n_done > c->n_calls) return false;
    if (c->n_calls != calls->len || c->n_done != c->n_calls) return false;
    for (int k = 0; k < calls->len; k++) {
        if (strcmp(c->names[k].ptr ? c->names[k].ptr : "", calls->v[k].name)) return false;
        if (strcmp(c->args[k].ptr ? c->args[k].ptr : "", calls->v[k].arguments)) return false;
    }
    return !strcmp(c->content.ptr ? c->content.ptr : "", content ? content : "") &&
           !strcmp(c->reasoning.ptr ? c->reasoning.ptr : "", reasoning ? reasoning : "");
}

/* Every split of the raw text into two pieces, and every step size: the streamed calls equal the parsed ones. */
static int ik_stream_check_all_splits(bool want, const char *raw, const char *content, const char *reasoning, const tool_calls *calls) {
    int bad = 0;
    const size_t n = strlen(raw);
    for (size_t at = 0; at <= n; at++) {
        ik_collect c = {0};
        ik_stream st = {0};
        st.want_tools = want;
        if (at) ik_stream_step(&st, raw, at, false, ik_collect_emit, &c);
        ik_stream_step(&st, raw, n, true, ik_collect_emit, &c);
        if (!ik_streamed_calls_match(&c, &st, calls, content, reasoning)) bad++;
        ik_collect_free(&c);
    }
    static const size_t steps[] = {1, 2, 3, 5, 17, 64, 4096};
    for (size_t si = 0; si < sizeof(steps) / sizeof(steps[0]); si++) {
        ik_collect c = {0};
        ik_stream st = ik_stream_collect_ex(raw, steps[si], want, &c);
        if (!ik_streamed_calls_match(&c, &st, calls, content, reasoning)) bad++;
        ik_collect_free(&c);
    }
    return bad;
}

static void test_inkling2_reference_vectors(void) {
    const char *dir = getenv("DS4_INKLING2_CHATVEC");
    if (!dir || !dir[0]) dir = "tests/test-vectors/inkling2-chat";
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "inkling2 reference vectors: %s not found, skipped\n", dir);
        return;
    }
    const char *dump_dir = getenv("DS4_INKLING2_DUMP_DIR");
    char **names = NULL;
    int n_names = 0, cap = 0;
    for (struct dirent *de; (de = readdir(d)) != NULL;) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".json")) continue;
        if (n_names == cap) { cap = cap ? cap * 2 : 128; names = xrealloc(names, (size_t)cap * sizeof(names[0])); }
        names[n_names++] = xstrdup(de->d_name);
    }
    closedir(d);
    for (int i = 1; i < n_names; i++) {   /* sort by name */
        char *x = names[i];
        int j = i;
        while (j > 0 && strcmp(names[j - 1], x) > 0) { names[j] = names[j - 1]; j--; }
        names[j] = x;
    }
    TEST_ASSERT(n_names >= 40);

    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    int n_prompt = 0, n_prompt_ok = 0, n_err = 0, n_err_ok = 0, n_skip = 0, n_pair = 0, n_pair_ok = 0, n_stream_ok = 0;
    int n_tool_stream_ok = 0, n_tool_stream_calls = 0;
    for (int i = 0; i < n_names; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        char *text = ik_test_read_file(path);
        ik_fixture f = {0};
        TEST_ASSERT(text && ik_fixture_parse(text, &f) && f.name && f.request);
        free(text);
        if (!f.name || !f.request) { ik_fixture_free(&f); continue; }
        request r;
        char err[256] = {0};
        const bool ok = parse_chat_request(NULL, NULL, f.request, 128, 32768, &r, err, sizeof(err));
        const bool want_err = f.ds4 && !strcmp(f.ds4, "error");
        if (f.ds4 && !strcmp(f.ds4, "skip")) {
            n_skip++;
            if (ok) request_free(&r);
        } else if (want_err) {
            n_err++;
            if (!ok) n_err_ok++;
            else { fprintf(stderr, "inkling2 vector %s: request should have been refused\n", f.name); request_free(&r); }
        } else if (!ok) {
            fprintf(stderr, "inkling2 vector %s: request refused: %s\n", f.name, err);
            TEST_ASSERT(false);
        } else {
            const char *want = f.prompt_ds4 ? f.prompt_ds4 : f.prompt;
            n_prompt++;
            if (want && r.prompt_text && !strcmp(want, r.prompt_text)) {
                n_prompt_ok++;
            } else if (want) {
                const char *got = r.prompt_text ? r.prompt_text : "";
                const size_t at = ik_first_diff(want, got);
                fprintf(stderr, "inkling2 vector %s: prompt differs at byte %zu\n  want ...%.60s\n  got  ...%.60s\n",
                        f.name, at, want + (at > 20 ? at - 20 : 0), got + (at > 20 ? at - 20 : 0));
            }
            if (dump_dir && dump_dir[0] && r.prompt_text) {
                char out[1100];
                snprintf(out, sizeof(out), "%s/%s.txt", dump_dir, f.name);
                FILE *fp = fopen(out, "wb");
                if (fp) { fwrite(r.prompt_text, 1, strlen(r.prompt_text), fp); fclose(fp); }
            }
            if (f.raw && f.has_expect) {
                n_pair++;
                char *content = NULL, *reasoning = NULL;
                tool_calls calls = {0};
                bool good = parse_inkling_generated_message_ex(f.raw, r.has_tools, &content, &reasoning, &calls);
                if (!good) fprintf(stderr, "inkling2 vector %s: generation did not parse\n", f.name);
                good = good && !strcmp(content ? content : "", f.content ? f.content : "") &&
                       !strcmp(reasoning ? reasoning : "", f.reasoning ? f.reasoning : "") &&
                       calls.len == f.calls.len;
                for (int k = 0; good && k < calls.len; k++)
                    good = !strcmp(calls.v[k].name, f.calls.v[k].name) &&
                           ik_same_json(calls.v[k].arguments, f.calls.v[k].arguments);
                if (good) n_pair_ok++;
                else fprintf(stderr, "inkling2 vector %s: parsed message differs (content %zu/%zu bytes, reasoning %zu/%zu, calls %d/%d)\n",
                             f.name, content ? strlen(content) : 0, f.content ? strlen(f.content) : 0,
                             reasoning ? strlen(reasoning) : 0, f.reasoning ? strlen(f.reasoning) : 0, calls.len, f.calls.len);
                /* streaming: the same raw text in pieces of any size gives the same reasoning and content */
                static const size_t steps[] = {1, 2, 5, 17, 64, 4096};
                bool stream_good = true;
                for (size_t si = 0; si < sizeof(steps) / sizeof(steps[0]); si++) {
                    ik_collect c = {0};
                    ik_stream_collect(f.raw, steps[si], &c);
                    if (strcmp(c.content.ptr ? c.content.ptr : "", content ? content : "") ||
                        strcmp(c.reasoning.ptr ? c.reasoning.ptr : "", reasoning ? reasoning : ""))
                        stream_good = false;
                    buf_free(&c.content);
                    buf_free(&c.reasoning);
                }
                {   /* incremental tool calls: the same calls as the parse, under every split */
                    const int bad = ik_stream_check_all_splits(r.has_tools, f.raw, content, reasoning, &calls);
                    if (bad) fprintf(stderr, "inkling2 vector %s: streamed tool calls differ in %d splits\n", f.name, bad);
                    else n_tool_stream_ok++;
                    if (calls.len) n_tool_stream_calls++;
                    if (bad) stream_good = false;
                }
                if (stream_good) n_stream_ok++;
                else fprintf(stderr, "inkling2 vector %s: streamed text differs from the parsed message\n", f.name);
                free(content);
                free(reasoning);
                tool_calls_free(&calls);
            }
            request_free(&r);
        }
        ik_fixture_free(&f);
    }
    g_test_force_syntax = -1;
    g_effort_fine_names = false;
    fprintf(stderr, "inkling2 reference vectors: %d files, prompts %d/%d, refused %d/%d, skipped %d, parsed %d/%d, streamed %d/%d\n",
            n_names, n_prompt_ok, n_prompt, n_err_ok, n_err, n_skip, n_pair_ok, n_pair, n_stream_ok, n_pair);
    TEST_ASSERT(n_prompt_ok == n_prompt);
    TEST_ASSERT(n_err_ok == n_err);
    TEST_ASSERT(n_pair_ok == n_pair);
    fprintf(stderr, "inkling2 tool-call streaming: %d/%d vectors identical under every split (%d with calls)\n",
            n_tool_stream_ok, n_pair, n_tool_stream_calls);
    TEST_ASSERT(n_stream_ok == n_pair);
    for (int i = 0; i < n_names; i++) free(names[i]);
    free(names);
}

/* ---- hand written cases ---- */

static char *ik_test_render(const char *body, ds4_think_mode forced_mode, bool use_forced) {
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    request r;
    char err[256] = {0};
    bool ok = parse_chat_request(NULL, NULL, body, 128, 32768, &r, err, sizeof(err));
    g_test_force_syntax = -1;
    g_effort_fine_names = false;
    TEST_ASSERT(ok);
    if (!ok) return xstrdup("");
    if (use_forced) r.think_mode = forced_mode;
    char *out = xstrdup(r.prompt_text);
    request_free(&r);
    return out;
}

static void test_inkling2_effort_mapping(void) {
    TEST_ASSERT(!strcmp(ds4_inkling2_effort_text(DS4_THINK_NONE), "0"));
    TEST_ASSERT(!strcmp(ds4_inkling2_effort_text(DS4_THINK_LOW), "0.2"));
    TEST_ASSERT(!strcmp(ds4_inkling2_effort_text(DS4_THINK_MEDIUM), "0.7"));
    TEST_ASSERT(!strcmp(ds4_inkling2_effort_text(DS4_THINK_HIGH), "0.9"));
    TEST_ASSERT(!strcmp(ds4_inkling2_effort_text(DS4_THINK_MAX), "0.99"));
    ds4_think_mode m;
    TEST_ASSERT(ds4_think_mode_parse_level("10", &m) && !strcmp(ds4_inkling2_effort_text(m), "0.1"));
    TEST_ASSERT(ds4_think_mode_parse_level("55", &m) && !strcmp(ds4_inkling2_effort_text(m), "0.55"));
    TEST_ASSERT(ds4_think_mode_parse_level("5", &m) && !strcmp(ds4_inkling2_effort_text(m), "0.05"));
    TEST_ASSERT(ds4_think_mode_parse_level("100", &m) && !strcmp(ds4_inkling2_effort_text(m), "0.99"));
    TEST_ASSERT(ds4_think_mode_parse_level("0", &m) && !strcmp(ds4_inkling2_effort_text(m), "0"));

    /* request strings, as llama.cpp's template reads them */
    static const struct { const char *effort, *shown; } cases[] = {
        {"none", "0"}, {"minimal", "0.1"}, {"low", "0.2"}, {"medium", "0.7"}, {"high", "0.9"},
        {"xhigh", "0.99"}, {"max", "0.99"}, {NULL, "0.9"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char body[256];
        if (cases[i].effort)
            snprintf(body, sizeof(body), "{\"reasoning_effort\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", cases[i].effort);
        else
            snprintf(body, sizeof(body), "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        char *p = ik_test_render(body, DS4_THINK_HIGH, false);
        char want[256];
        snprintf(want, sizeof(want),
                 "<|message_system|><|content_text|>Thinking effort level: %s<|end_message|>"
                 "<|message_user|><|content_text|>hi<|end_message|>", cases[i].shown);
        if (strcmp(p, want)) fprintf(stderr, "effort %s: got %s\n", cases[i].effort ? cases[i].effort : "(default)", p);
        TEST_ASSERT(!strcmp(p, want));
        free(p);
    }
    /* thinking switched off (model -chat, thinking:false) is effort none; max keeps 0.99 on a small context */
    char *p = ik_test_render("{\"thinking\":{\"type\":\"disabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", DS4_THINK_HIGH, false);
    TEST_ASSERT(strstr(p, "Thinking effort level: 0<|end_message|>") != NULL);
    free(p);
    p = ik_test_render("{\"model\":\"inkling-2-mini-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", DS4_THINK_HIGH, false);
    TEST_ASSERT(strstr(p, "Thinking effort level: 0<|end_message|>") != NULL);
    free(p);
    p = ik_test_render("{\"model\":\"inkling-2-mini-reasoner\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", DS4_THINK_HIGH, false);
    TEST_ASSERT(strstr(p, "Thinking effort level: 0.9<|end_message|>") != NULL);
    free(p);
    p = ik_test_render("{\"reasoning_effort\":\"max\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", DS4_THINK_HIGH, false);
    TEST_ASSERT(strstr(p, "Thinking effort level: 0.99<|end_message|>") != NULL);
    free(p);
    /* unknown names are refused, JSON numbers are ignored like llama.cpp does */
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    request r;
    char err[128];
    TEST_ASSERT(!parse_chat_request(NULL, NULL, "{\"reasoning_effort\":\"bogus\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", 8, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(parse_chat_request(NULL, NULL, "{\"reasoning_effort\":0.5,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", 8, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(r.prompt_text, "effort level: 0.9<") != NULL);
    request_free(&r);
    g_test_force_syntax = -1;
    g_effort_fine_names = false;
}

static void test_inkling2_model_ids(void) {
    TEST_ASSERT(server_model_alias_known("inkling-2-mini"));
    TEST_ASSERT(server_model_alias_known("inkling-2-mini-chat"));
    TEST_ASSERT(server_model_alias_known("inkling-2-mini-reasoner"));
    TEST_ASSERT(model_alias_disables_thinking("inkling-2-mini-chat"));
    TEST_ASSERT(!model_alias_disables_thinking("inkling-2-mini"));
    TEST_ASSERT(model_alias_enables_thinking("inkling-2-mini-reasoner"));
    TEST_ASSERT(!model_alias_enables_thinking("inkling-2-mini"));
}

static void test_inkling2_render_details(void) {
    /* tool call with unsorted keys, nested values keep their order, numbers as the template prints them */
    const char *body =
        "{\"reasoning_effort\":\"none\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"go\"},"
        "{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"hmm\",\"tool_calls\":[{\"id\":\"a\",\"type\":\"function\","
        "\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"z\\\":{\\\"b\\\":1,\\\"a\\\":[1.0,2.50,1e3,-0,10000000000000000000000]},\\\"a\\\":\\\"x\\\\ty\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"a\",\"content\":\"r\"}]}";
    char *p = ik_test_render(body, DS4_THINK_NONE, false);
    const char *want =
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        "<|message_user|><|content_text|>go<|end_message|>"
        "<|message_model|><|content_thinking|>hmm<|end_message|>"
        "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":\"x\\ty\",\"z\":{\"b\":1,\"a\":[1.0,2.5,1000.0,-0.0,1e+22]}}}<|end_message|>"
        "<|content_model_end_sampling|>"
        "<|message_tool|>f<|content_text|>r<|end_message|>";
    if (strcmp(p, want)) fprintf(stderr, "got:  %s\nwant: %s\n", p, want);
    TEST_ASSERT(!strcmp(p, want));
    free(p);

    /* array content: one block per part; a tool's own name wins over the id lookup */
    p = ik_test_render("{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\",\"text\":\"b\"}]},"
                       "{\"role\":\"tool\",\"name\":\"n\",\"content\":\"r\"}]}", DS4_THINK_HIGH, false);
    TEST_ASSERT(strstr(p, "<|message_user|><|content_text|>a<|end_message|><|message_user|><|content_text|>b<|end_message|>"
                          "<|message_tool|>n<|content_text|>r<|end_message|>") != NULL);
    free(p);

    /* the template wants JSON-object arguments; unknown roles are errors */
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    request r;
    char err[128] = {0};
    TEST_ASSERT(!parse_chat_request(NULL, NULL,
        "{\"messages\":[{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"a\",\"function\":{\"name\":\"f\",\"arguments\":\"{oops\"}}]}]}",
        8, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "JSON object") != NULL);
    TEST_ASSERT(!parse_chat_request(NULL, NULL, "{\"messages\":[{\"role\":\"wizard\",\"content\":\"x\"}]}", 8, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Unknown message role") != NULL);
    /* Anthropic and Responses are served for this family; a durable previous_response_id is refused (as for Qwen) */
    TEST_ASSERT(parse_anthropic_request(NULL, NULL, "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", 8, 4096, &r, err, sizeof(err)));
    request_free(&r);
    TEST_ASSERT(parse_responses_request(NULL, NULL, "{\"input\":\"x\"}", 8, 4096, &r, err, sizeof(err)));
    request_free(&r);
    TEST_ASSERT(!parse_responses_request(NULL, NULL, "{\"input\":\"x\",\"previous_response_id\":\"resp_1\"}", 8, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "previous_response_id") != NULL);
    /* Anthropic thinking budget -> effort tier; an explicit effort wins; disabled is none; max stays max on a small context */
    static const struct { const char *extra; ds4_think_mode want; } thinking_cases[] = {
        {"", DS4_THINK_HIGH},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024}", DS4_THINK_LOW},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":2048}", DS4_THINK_MEDIUM},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":8191}", DS4_THINK_MEDIUM},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":8192}", DS4_THINK_HIGH},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":31999}", DS4_THINK_HIGH},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":32000}", DS4_THINK_MAX},
        {",\"thinking\":{\"type\":\"enabled\"}", DS4_THINK_HIGH},
        {",\"thinking\":{\"type\":\"disabled\",\"budget_tokens\":9999}", DS4_THINK_NONE},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024},\"output_config\":{\"effort\":\"high\"}", DS4_THINK_HIGH},
        {",\"output_config\":{\"effort\":\"max\"}", DS4_THINK_MAX},
        {",\"thinking\":{\"budget_tokens\":40000,\"type\":\"enabled\"},\"reasoning_effort\":\"low\"", DS4_THINK_LOW},
    };
    for (size_t i = 0; i < sizeof(thinking_cases) / sizeof(thinking_cases[0]); i++) {
        char body[400];
        snprintf(body, sizeof(body), "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]%s}", thinking_cases[i].extra);
        TEST_ASSERT(parse_anthropic_request(NULL, NULL, body, 8, 1024, &r, err, sizeof(err)));
        if (r.think_mode != thinking_cases[i].want) fprintf(stderr, "anthropic thinking case %zu: mode %d want %d\n", i, (int)r.think_mode, (int)thinking_cases[i].want);
        TEST_ASSERT(r.think_mode == thinking_cases[i].want);
        request_free(&r);
    }
    /* tool_result blocks become tool messages after the call that made them; the system prompt comes first */
    {
        const char *body =
            "{\"max_tokens\":8,\"system\":\"S\",\"tools\":[{\"name\":\"f\",\"description\":\"d\",\"input_schema\":{\"type\":\"object\"}}],\"messages\":["
            "{\"role\":\"user\",\"content\":\"q\"},"
            "{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"f\",\"input\":{\"a\":1}}]},"
            "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t1\",\"content\":\"R\"},{\"type\":\"text\",\"text\":\"more\"}]}]}";
        TEST_ASSERT(parse_anthropic_request(NULL, NULL, body, 8, 4096, &r, err, sizeof(err)));
        const char *pt = r.prompt_text;
        const char *sys = strstr(pt, "<|message_system|><|content_text|>S<|end_message|>");
        const char *tool = strstr(pt, "<|message_tool|>f<|content_text|>R<|end_message|><|message_user|><|content_text|>more<|end_message|>");
        TEST_ASSERT(sys && tool && sys < tool);
        TEST_ASSERT(strstr(pt, "\"parameters\":{\"type\":\"object\"}") != NULL);   /* input_schema is declared as parameters */
        request_free(&r);
    }
    g_test_force_syntax = -1;
    g_effort_fine_names = false;
}

static void test_inkling2_float_repr(void) {
    static const struct { double v; const char *s; } cases[] = {
        {1.0, "1.0"}, {0.5, "0.5"}, {3.14, "3.14"}, {1e-7, "1e-07"}, {1.5e10, "15000000000.0"}, {1e16, "1e+16"},
        {123456789012345.0, "123456789012345.0"}, {0.0001, "0.0001"}, {0.00001, "1e-05"}, {-2.5, "-2.5"}, {0.1, "0.1"},
        {100.0, "100.0"}, {1.7976931348623157e308, "1.7976931348623157e+308"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        buf b = {0};
        ik_float_repr(&b, cases[i].v);
        if (strcmp(b.ptr, cases[i].s)) fprintf(stderr, "float %.17g: got %s want %s\n", cases[i].v, b.ptr, cases[i].s);
        TEST_ASSERT(!strcmp(b.ptr, cases[i].s));
        buf_free(&b);
    }
}

static void test_inkling2_parse_cases(void) {
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    /* thinking, answer, two calls; the name comes from the JSON, trailing EOS text is ignored */
    TEST_ASSERT(parse_inkling_generated_message_ex(
        "<|message_model|><|content_thinking|>t1<|end_message|><|message_model|><|content_text|>A<|end_message|>"
        "<|message_model|><|content_thinking|>t2<|end_message|><|message_model|><|content_text|>B<|end_message|>"
        "<|message_model|>wrongname<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"x\":1}}<|end_message|>"
        "<|message_model|><|content_invoke_tool_json|>{\"name\":\"g\",\"args\":{}}<|end_message|><|content_model_end_sampling|>",
        true, &content, &reasoning, &calls));
    TEST_ASSERT(!strcmp(content, "AB") && !strcmp(reasoning, "t1t2"));
    TEST_ASSERT(calls.len == 2 && !strcmp(calls.v[0].name, "f") && !strcmp(calls.v[0].arguments, "{\"x\":1}"));
    TEST_ASSERT(!strcmp(calls.v[1].name, "g") && !strcmp(calls.v[1].arguments, "{}"));
    tool_calls_free(&calls);
    free(content); free(reasoning); content = reasoning = NULL;

    /* no tools active: tool blocks are dropped */
    TEST_ASSERT(parse_inkling_generated_message_ex(
        "<|message_model|><|content_text|>A<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>",
        false, &content, &reasoning, &calls));
    TEST_ASSERT(!strcmp(content, "A") && calls.len == 0 && reasoning == NULL);
    free(content); free(reasoning); content = reasoning = NULL;

    /* cut off inside a block: the text so far is kept; a cut-off tool call is not run */
    TEST_ASSERT(parse_inkling_generated_message_ex("<|message_model|><|content_thinking|>half a thou", true, &content, &reasoning, &calls));
    TEST_ASSERT(!strcmp(reasoning, "half a thou") && !strcmp(content, ""));
    free(content); free(reasoning); content = reasoning = NULL;
    TEST_ASSERT(parse_inkling_generated_message_ex("<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"ar", true, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 0);
    free(content); free(reasoning); content = reasoning = NULL;

    /* not a block list, or a call without a name: the parse fails and the caller returns the raw text */
    TEST_ASSERT(!parse_inkling_generated_message_ex("plain words", true, &content, &reasoning, &calls));
    TEST_ASSERT(!parse_inkling_generated_message_ex("<|message_model|>f<|content_invoke_tool_json|>{\"args\":{}}<|end_message|>", true, &content, &reasoning, &calls));
    tool_calls_free(&calls);

    /* via the server's response parser: finish reason and fallback */
    const char *finish = "stop";
    char err[64] = {0};
    bool recovered = false;
    TEST_ASSERT(parse_generated_message_for_response_for_syntax(SERVER_MODEL_SYNTAX_INKLING,
        "<|message_model|><|content_text|>hi<|end_message|>", true, false, true, &finish, err, sizeof(err),
        &content, &reasoning, &calls, &recovered, NULL));
    TEST_ASSERT(!strcmp(content, "hi"));
    free(content); free(reasoning); content = reasoning = NULL;
    TEST_ASSERT(!parse_generated_message_for_response_for_syntax(SERVER_MODEL_SYNTAX_INKLING,
        "no header here", true, false, true, &finish, err, sizeof(err),
        &content, &reasoning, &calls, &recovered, NULL));
    TEST_ASSERT(!strcmp(content, "no header here"));
    free(content); free(reasoning);
    tool_calls_free(&calls);
}


/* The parser cases of llama.cpp's own tests/test-chat.cpp ("Inkling 1 and 2"), with the expectations written there. */
static void test_inkling2_llamacpp_parser_cases(void) {
#define H "<|message_model|>"
#define E "<|end_message|>"
#define CALL "special_function<|content_invoke_tool_json|>{\"name\":\"special_function\",\"args\":{\"arg1\":1}}" E
    static const struct {
        const char *raw, *content, *reasoning;
        const char *names[2], *args[2];
    } cases[] = {
        { H "<|content_text|>Hello, world!\nWhat's up?" E, "Hello, world!\nWhat's up?", NULL, {NULL}, {NULL} },
        { H "<|content_thinking|>I'm thinking" E H "<|content_text|>Hello, world!\nWhat's up?" E,
          "Hello, world!\nWhat's up?", "I'm thinking", {NULL}, {NULL} },
        { H "<|content_thinking|>I was still thinking", "", "I was still thinking", {NULL}, {NULL} },
        { H "<|content_thinking|>I'm thinking" E H "<|content_text|>Hello, wor", "Hello, wor", "I'm thinking", {NULL}, {NULL} },
        { H CALL, "", NULL, {"special_function"}, {"{\"arg1\":1}"} },
        { H "<|content_thinking|>I should call it" E H "<|content_text|>On it." E H CALL,
          "On it.", "I should call it", {"special_function"}, {"{\"arg1\":1}"} },
        { H CALL H "special_function_with_opt<|content_invoke_tool_json|>{\"name\":\"special_function_with_opt\",\"args\":{\"arg1\":1,\"arg2\":2}}" E,
          "", NULL, {"special_function", "special_function_with_opt"}, {"{\"arg1\":1}", "{\"arg1\":1,\"arg2\":2}"} },
        { H "<|content_invoke_tool_json|>{\"name\":\"special_function\",\"args\":{\"arg1\":1}}" E,
          "", NULL, {"special_function"}, {"{\"arg1\":1}"} },
        { H "unknown_tool<|content_invoke_tool_json|>{\"name\": \"unknown_tool\", \"args\": {\"arg1\": 1}}" E,
          "", NULL, {"unknown_tool"}, {"{\"arg1\": 1}"} },
        { H "special<|content_invoke_tool_json|>{\"name\":\"special_function\",\"args\":{\"arg1\":1}}" E,
          "", NULL, {"special_function"}, {"{\"arg1\":1}"} },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *content = NULL, *reasoning = NULL;
        tool_calls calls = {0};
        TEST_ASSERT(parse_inkling_generated_message_ex(cases[i].raw, true, &content, &reasoning, &calls));
        TEST_ASSERT(!strcmp(content, cases[i].content));
        TEST_ASSERT(cases[i].reasoning ? reasoning && !strcmp(reasoning, cases[i].reasoning) : !reasoning);
        int want = 0;
        while (want < 2 && cases[i].names[want]) want++;
        TEST_ASSERT(calls.len == want);
        for (int k = 0; k < want && k < calls.len; k++) {
            TEST_ASSERT(!strcmp(calls.v[k].name, cases[i].names[k]));
            TEST_ASSERT(!strcmp(calls.v[k].arguments, cases[i].args[k]));
        }
        free(content);
        free(reasoning);
        tool_calls_free(&calls);
    }
#undef H
#undef E
#undef CALL
}

static void test_inkling2_stream_holds_markers(void) {
    /* marker pieces and multi-byte characters never leak into deltas */
    const char *raw = "<|message_model|><|content_thinking|>r\xc3\xa9sum\xc3\xa9<|end_message|><|message_model|><|content_text|>ok <|end_mess<|end_message|>";
    ik_stream st = {0};
    ik_collect c = {0};
    const size_t n = strlen(raw);
    for (size_t at = 1; at < n; at++) {
        ik_stream_step(&st, raw, at, false, ik_collect_emit, &c);
        /* every prefix of the emitted text must be valid UTF-8 and free of marker fragments */
        TEST_ASSERT(!c.reasoning.ptr || !strstr(c.reasoning.ptr, "<|"));
        TEST_ASSERT(!c.content.ptr || strstr(c.content.ptr, "<|end_message|>") == NULL);
        if (c.reasoning.ptr) TEST_ASSERT(utf8_stream_safe_len(c.reasoning.ptr, 0, c.reasoning.len, false) == c.reasoning.len);
    }
    ik_stream_step(&st, raw, n, true, ik_collect_emit, &c);
    TEST_ASSERT(c.reasoning.ptr && !strcmp(c.reasoning.ptr, "r\xc3\xa9sum\xc3\xa9"));
    TEST_ASSERT(c.content.ptr && !strcmp(c.content.ptr, "ok <|end_mess"));
    buf_free(&c.reasoning);
    buf_free(&c.content);

    /* text without a header is passed through as content */
    ik_collect raw_c = {0};
    ik_stream_collect("just text", 3, &raw_c);
    TEST_ASSERT(raw_c.content.ptr && !strcmp(raw_c.content.ptr, "just text"));
    buf_free(&raw_c.content);
    buf_free(&raw_c.reasoning);
}

/* Regression: whitespace between two blocks used to switch the stream tracker to pass-through (markers and tool JSON leaked into
 * content), and a tool block's end marker split across two calls was never found (later blocks swallowed).  The streamed text must
 * equal what the non-stream parser returns, for every split point. */
static void test_inkling2_stream_whitespace_and_split_end(void) {
    static const char *const raws[] = {
        "<|message_model|><|content_thinking|>t<|end_message|>\n<|message_model|><|content_text|>hi<|end_message|> \n\t"
            "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>\n",
        "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}<|end_message|>"
            "<|message_model|><|content_text|>after<|end_message|>",
        "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>\n"
            "<|message_model|><|content_invoke_tool_json|>{\"name\":\"g\",\"args\":{}}<|end_message|>  "
            "<|message_model|><|content_text|>done<|end_message|>",
    };
    for (size_t k = 0; k < sizeof(raws) / sizeof(raws[0]); k++) {
        char *pc = NULL, *pr = NULL;
        tool_calls calls = {0};
        TEST_ASSERT(parse_inkling_generated_message_ex(raws[k], true, &pc, &pr, &calls));
        const size_t n = strlen(raws[k]);
        for (size_t step = 1; step <= n; step++) {
            ik_collect c = {0};
            ik_stream_collect(raws[k], step, &c);
            TEST_ASSERT(!strcmp(c.content.ptr ? c.content.ptr : "", pc ? pc : ""));
            TEST_ASSERT(!strcmp(c.reasoning.ptr ? c.reasoning.ptr : "", pr ? pr : ""));
            buf_free(&c.content);
            buf_free(&c.reasoning);
        }
        /* every single split point: the text is handed over as two growing prefixes */
        for (size_t at = 1; at < n; at++) {
            ik_stream st = {0};
            ik_collect c = {0};
            ik_stream_step(&st, raws[k], at, false, ik_collect_emit, &c);
            ik_stream_step(&st, raws[k], n, true, ik_collect_emit, &c);
            TEST_ASSERT(!strcmp(c.content.ptr ? c.content.ptr : "", pc ? pc : ""));
            buf_free(&c.content);
            buf_free(&c.reasoning);
        }
        free(pc);
        free(pr);
        tool_calls_free(&calls);
    }
}

/* Regression: /v1/completions wrapped the prompt in the chat template and returned the block markers, and logprobs were ignored
 * silently.  Both are now refused with a clear 400 message for this family; null/false/0 values are accepted. */
static void test_inkling2_refused_endpoints(void) {
    request r;
    char err[256] = {0};
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    /* /v1/completions: the prompt is the text, no chat template, special tokens stay in it for the tokenizer */
    TEST_ASSERT(parse_completion_request(NULL, "{\"prompt\":\"hi <|message_user|> there\"}", 64, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.prompt_text && !strcmp(r.prompt_text, "hi <|message_user|> there"));
    TEST_ASSERT(!r.logprobs && r.think_mode == DS4_THINK_NONE);
    request_free(&r);
    TEST_ASSERT(parse_completion_request(NULL, "{\"prompt\":[\"a\"],\"logprobs\":3,\"max_tokens\":4}", 64, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(!strcmp(r.prompt_text, "a") && r.logprobs && r.top_logprobs == 3 && r.max_tokens == 4);
    request_free(&r);
    TEST_ASSERT(!parse_completion_request(NULL, "{\"prompt\":[\"a\",\"b\"],\"logprobs\":3,\"max_tokens\":4}", 64, 32768, &r, err, sizeof(err)));   /* several prompts: refused */
    TEST_ASSERT(parse_completion_request(NULL, "{\"prompt\":\"x\",\"logprobs\":0}", 64, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(r.logprobs && r.top_logprobs == 0);
    request_free(&r);
    TEST_ASSERT(parse_completion_request(NULL, "{\"prompt\":\"x\",\"logprobs\":null}", 64, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(!r.logprobs);
    request_free(&r);
    TEST_ASSERT(!parse_completion_request(NULL, "{\"prompt\":\"x\",\"logprobs\":-1}", 64, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(!parse_completion_request(NULL, "{\"prompt\":\"x\",\"logprobs\":21}", 64, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(!parse_completion_request(NULL, "{\"logprobs\":2}", 64, 32768, &r, err, sizeof(err)));   /* no prompt */
    /* chat completions: logprobs and top_logprobs are accepted and recorded */
    static const struct { const char *body; bool on; int top; } lp_cases[] = {
        {"{\"logprobs\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", true, 0},
        {"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"logprobs\":true,\"top_logprobs\":5}", true, 5},
        {"{\"top_logprobs\":3,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", true, 3},
        {"{\"logprobs\":false,\"top_logprobs\":0,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", false, 0},
        {"{\"logprobs\":false,\"top_logprobs\":5,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", false, 0},
    };
    for (size_t i = 0; i < sizeof(lp_cases) / sizeof(lp_cases[0]); i++) {
        TEST_ASSERT(parse_chat_request(NULL, NULL, lp_cases[i].body, 128, 32768, &r, err, sizeof(err)));
        TEST_ASSERT(r.logprobs == lp_cases[i].on && r.top_logprobs == lp_cases[i].top);
        request_free(&r);
    }
    err[0] = 0;
    TEST_ASSERT(!parse_chat_request(NULL, NULL, "{\"logprobs\":true,\"top_logprobs\":21,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", 128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "top_logprobs"));
    static const char *const ok[] = {
        "{\"logprobs\":false,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        "{\"logprobs\":null,\"top_logprobs\":null,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        "{\"top_logprobs\":0,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
    };
    for (size_t i = 0; i < sizeof(ok) / sizeof(ok[0]); i++) {
        TEST_ASSERT(parse_chat_request(NULL, NULL, ok[i], 128, 32768, &r, err, sizeof(err)));
        request_free(&r);
    }
    g_test_force_syntax = -1;
    g_effort_fine_names = false;
}

/* Regression: the JSON canonicaliser scanned every earlier key for each new key and used an insertion sort (40,000 keys = a 509 KB
 * body took 2.7 s of CPU, the Qwen path 0.008 s).  Repeated keys keep their first position and take the last value. */
static double ik_test_cpu_s(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }

static char *ik_test_dump(const char *json, bool sort_keys) {
    buf b = {0};
    const char *p = json;
    bool ok = ik_json_dump_depth(&b, &p, sort_keys, 0);
    TEST_ASSERT(ok);
    return buf_take(&b);
}

static void test_inkling2_canonical_json_scales(void) {
    char *o = ik_test_dump("{\"b\":1,\"a\":2,\"b\":3,\"c\":{\"y\":1,\"x\":2,\"y\":9},\"a\":4}", true);
    TEST_ASSERT(!strcmp(o, "{\"a\":4,\"b\":3,\"c\":{\"x\":2,\"y\":9}}"));
    free(o);
    o = ik_test_dump("{\"b\":1,\"a\":2,\"b\":3,\"a\":4}", false);
    TEST_ASSERT(!strcmp(o, "{\"b\":3,\"a\":4}"));
    free(o);
    const int n = 40000;
    buf big = {0};
    buf_putc(&big, '{');
    for (int i = 0; i < n; i++) {
        char item[64];
        /* descending keys: the worst case for an insertion sort; every 7th key is repeated */
        snprintf(item, sizeof(item), "%s\"k%06d\":%d", i ? "," : "", n - i, i);
        buf_puts(&big, item);
        if (i % 7 == 0) { snprintf(item, sizeof(item), ",\"k%06d\":-1", n - i); buf_puts(&big, item); }
    }
    buf_putc(&big, '}');
    const double t0 = ik_test_cpu_s();
    o = ik_test_dump(big.ptr, true);
    const double dt = ik_test_cpu_s() - t0;
    fprintf(stderr, "inkling2 canonical JSON, 40,000 keys: %.3f s CPU (was 2.7 s)\n", dt);
    TEST_ASSERT(dt < 1.0);
    TEST_ASSERT(!strncmp(o, "{\"k000001\":", 11) && strstr(o, "\"k040000\":-1}") != NULL);
    free(o);
    buf_free(&big);
}

/* Pull the JSON strings that follow `key` (e.g. "\"arguments\":") out of an SSE dump, in order. */
static char *ik_test_collect_strings(const char *sse, const char *key, int *count) {
    buf all = {0};
    int n = 0;
    for (const char *p = sse; (p = strstr(p, key)) != NULL;) {
        p += strlen(key);
        if (*p != '"') continue;
        char *s = NULL;
        const char *q = p;
        if (json_string(&q, &s)) { buf_puts(&all, s); n++; }
        free(s);
    }
    if (count) *count = n;
    return buf_take(&all);
}

/* A4 on the wire: the name arrives first in its own chunk, the arguments follow in pieces, and what the
 * pieces add up to is the parsed call's arguments, for a growing prefix of any step size. */
static void test_inkling2_openai_incremental_tool_stream(void) {
    const char *raw =
        "<|message_model|><|content_thinking|>I should call it.<|end_message|>"
        "<|message_model|><|content_text|>Checking ?\xc3\xa9\xe2\x82\xac now.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\": \"get_weather\", \"args\": {\"city\": \"Z\xc3\xbcrich \\\"x\\\"\", \"days\": [1, 2, {\"a\": null}]}}<|end_message|>"
        "<|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{}}<|end_message|>"
        "<|content_model_end_sampling|>";
    static const size_t steps[] = {1, 3, 7, 64};
    for (size_t si = 0; si < sizeof(steps) / sizeof(steps[0]); si++) {
        int sv[2];
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        if (sv[0] < 0 || sv[1] < 0) return;
        { int big = 4 << 20; setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &big, sizeof(big)); setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &big, sizeof(big)); }
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.api = API_OPENAI;
        r.stream = true;
        r.model_syntax = SERVER_MODEL_SYNTAX_INKLING;
        r.think_mode = DS4_THINK_HIGH;
        r.has_tools = true;
        openai_stream st;
        openai_stream_start(&r, &st);
        const size_t n = strlen(raw);
        for (size_t at = steps[si]; at < n; at += steps[si])
            TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_t", &st, raw, at, false));
        char *content = NULL, *reasoning = NULL;
        tool_calls calls = {0};
        TEST_ASSERT(parse_inkling_generated_message_ex(raw, true, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 2);
        TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_t", &st, raw, n, &calls, "tool_calls", 10, 8));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        int n_args = 0;
        char *args0 = ik_test_collect_strings(out, "\"function\":{\"arguments\":", &n_args);
        /* the two calls' args pieces are in one string here: split by call index below */
        const char *name0 = strstr(out, "\"index\":0,\"id\":");
        const char *name1 = strstr(out, "\"index\":1,\"id\":");
        TEST_ASSERT(name0 && name1 && name0 < name1);
        TEST_ASSERT(strstr(out, "\"name\":\"get_weather\",\"arguments\":\"\"") != NULL);
        TEST_ASSERT(strstr(out, "\"name\":\"get_time\",\"arguments\":\"\"") != NULL);
        /* every args piece sits between its call's name chunk and the next */
        const char *first_args = strstr(out, "\"function\":{\"arguments\":\"{");
        TEST_ASSERT(first_args && name0 < first_args);
        char *want_all = xstrdup("");
        for (int k = 0; k < calls.len; k++) {
            buf b = {0};
            buf_puts(&b, want_all);
            buf_puts(&b, calls.v[k].arguments);
            free(want_all);
            want_all = buf_take(&b);
        }
        TEST_ASSERT(!strcmp(args0, want_all));
        if (steps[si] == 1) TEST_ASSERT(n_args > 4);   /* really incremental, not one piece per call */
        TEST_ASSERT(strstr(out, "\"finish_reason\":\"tool_calls\"") != NULL);
        /* each call is announced once: no whole-call chunk after the streamed pieces */
        {
            int announced = 0;
            for (const char *p = out; (p = strstr(p, "\"function\":{\"name\":")) != NULL; p++) announced++;
            TEST_ASSERT(announced == 2);
        }
        free(args0);
        free(want_all);
        free(out);
        free(content);
        free(reasoning);
        tool_calls_free(&calls);
        close(sv[0]);
        close(sv[1]);
        request_free(&r);
    }
}

/* A1/A2: the same conversation sent as chat completions, Anthropic messages and Responses input renders to the
 * same prompt text (and so the same tokens).  Files from tests/inkling2_api_fixtures.py. */
static void test_inkling2_api_equivalence(void) {
    const char *dir = getenv("DS4_INKLING2_APIVEC");
    if (!dir || !dir[0]) dir = "tests/test-vectors/inkling2-api";
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "inkling2 api vectors: %s not found, skipped\n", dir);
        return;
    }
    int n_files = 0, n_same = 0, n_tools = 0;
    for (struct dirent *de; (de = readdir(d)) != NULL;) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".json")) continue;
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        char *text = ik_test_read_file(path);
        TEST_ASSERT(text != NULL);
        if (!text) continue;
        const char *q = text;
        ik_member *m = NULL;
        int nm = 0;
        TEST_ASSERT(ik_json_members(&q, &m, &nm));
        char *raw[3] = {NULL, NULL, NULL};
        static const char *keys[3] = {"chat", "anthropic", "responses"};
        for (int i = 0; i < nm; i++)
            for (int k = 0; k < 3; k++)
                if (!strcmp(m[i].key, keys[k])) { const char *v = m[i].val; json_raw_value(&v, &raw[k]); }
        char *prompts[3] = {NULL, NULL, NULL};
        int tools_flags[3] = {0, 0, 0};
        g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
        for (int k = 0; k < 3; k++) {
            request r;
            char err[256] = {0};
            bool ok = false;
            if (!raw[k]) continue;
            if (k == 0) ok = parse_chat_request(NULL, NULL, raw[k], 128, 32768, &r, err, sizeof(err));
            else if (k == 1) ok = parse_anthropic_request(NULL, NULL, raw[k], 128, 32768, &r, err, sizeof(err));
            else ok = parse_responses_request(NULL, NULL, raw[k], 128, 32768, &r, err, sizeof(err));
            if (!ok) fprintf(stderr, "inkling2 api vector %s (%s): refused: %s\n", de->d_name, keys[k], err);
            TEST_ASSERT(ok);
            if (!ok) continue;
            prompts[k] = xstrdup(r.prompt_text);
            tools_flags[k] = r.has_tools;
            const char *dump_dir = getenv("DS4_INKLING2_DUMP_DIR");   /* for tests/inkling2_api_tokens.py */
            if (dump_dir && dump_dir[0]) {
                char out[1200];
                snprintf(out, sizeof(out), "%s/api_%.*s.%s.txt", dump_dir, (int)(strlen(de->d_name) - 5), de->d_name, keys[k]);
                FILE *fp = fopen(out, "wb");
                if (fp) { fwrite(r.prompt_text, 1, strlen(r.prompt_text), fp); fclose(fp); }
            }
            request_free(&r);
        }
        g_test_force_syntax = -1;
        g_effort_fine_names = false;
        n_files++;
        const bool same = prompts[0] && prompts[1] && prompts[2] && !strcmp(prompts[0], prompts[1]) && !strcmp(prompts[0], prompts[2]) &&
                          tools_flags[0] == tools_flags[1] && tools_flags[0] == tools_flags[2];
        if (same) n_same++;
        else {
            fprintf(stderr, "inkling2 api vector %s: prompts differ\n", de->d_name);
            for (int k = 1; k < 3 && prompts[0] && prompts[k]; k++)
                if (strcmp(prompts[0], prompts[k])) {
                    size_t at = ik_first_diff(prompts[0], prompts[k]);
                    fprintf(stderr, "  chat vs %s at byte %zu\n    chat ...%.70s\n    other...%.70s\n", keys[k], at,
                            prompts[0] + (at > 20 ? at - 20 : 0), prompts[k] + (at > 20 ? at - 20 : 0));
                }
        }
        if (tools_flags[0]) n_tools++;
        for (int k = 0; k < 3; k++) { free(raw[k]); free(prompts[k]); }
        ik_members_free(m, nm);
        free(text);
    }
    closedir(d);
    fprintf(stderr, "inkling2 api vectors: %d conversations, chat == anthropic == responses prompt in %d (%d with tools)\n", n_files, n_same, n_tools);
    TEST_ASSERT(n_files >= 15);
    TEST_ASSERT(n_same == n_files);
}

/* Request-path regressions: each of these failed before its fix. */
static void test_inkling2_request_regressions(void) {
    request r;
    char err[256] = {0};
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    /* an output_config / reasoning_effort field that carries no effort must not reset an effort set by the other */
    static const struct { const char *extra; ds4_think_mode want; } eff[] = {
        {",\"reasoning_effort\":\"low\",\"output_config\":{\"format\":{\"type\":\"text\"}}", DS4_THINK_LOW},
        {",\"output_config\":{\"effort\":\"low\"},\"reasoning_effort\":null", DS4_THINK_LOW},
        {",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":-5}", DS4_THINK_HIGH},
    };
    for (size_t i = 0; i < sizeof(eff) / sizeof(eff[0]); i++) {
        char body[300];
        snprintf(body, sizeof(body), "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]%s}", eff[i].extra);
        TEST_ASSERT(parse_anthropic_request(NULL, NULL, body, 8, 4096, &r, err, sizeof(err)));
        TEST_ASSERT(r.think_mode == eff[i].want);
        request_free(&r);
    }
    /* several text blocks are several blocks in the prompt, also around tool results */
    TEST_ASSERT(parse_anthropic_request(NULL, NULL,
        "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"},{\"type\":\"text\",\"text\":\"World\"}]}]}",
        8, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(r.prompt_text, "<|message_user|><|content_text|>Hello<|end_message|><|message_user|><|content_text|>World<|end_message|>"));
    request_free(&r);
    TEST_ASSERT(parse_anthropic_request(NULL, NULL,
        "{\"max_tokens\":8,\"messages\":[{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"a\",\"name\":\"f\",\"input\":{}},{\"type\":\"tool_use\",\"id\":\"b\",\"name\":\"f\",\"input\":{}}]},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"pre\"},{\"type\":\"tool_result\",\"tool_use_id\":\"a\",\"content\":\"A\"},"
        "{\"type\":\"tool_result\",\"tool_use_id\":\"b\",\"content\":\"B\"},{\"type\":\"text\",\"text\":\"post1\"},{\"type\":\"text\",\"text\":\"post2\"}]}]}",
        8, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(r.prompt_text, "<|message_tool|>f<|content_text|>A<|end_message|><|message_tool|>f<|content_text|>B<|end_message|>"
                                      "<|message_user|><|content_text|>pre<|end_message|><|message_user|><|content_text|>post1<|end_message|>"
                                      "<|message_user|><|content_text|>post2<|end_message|>"));
    request_free(&r);
    /* completions: logprobs is validated for Inkling only, without an int cast of a huge number */
    TEST_ASSERT(!parse_completion_request(NULL, "{\"prompt\":\"x\",\"logprobs\":1e10}", 8, 4096, &r, err, sizeof(err)));
    TEST_ASSERT(!parse_completion_request(NULL, "{\"prompt\":\"x\",\"logprobs\":2.5}", 8, 4096, &r, err, sizeof(err)));
    /* (the other families skip the field; their path needs an engine to tokenize, so it is covered by the live runs) */
    g_test_force_syntax = -1;
    g_effort_fine_names = false;

    /* token strings: only well formed UTF-8 passes */
    static const struct { const char *s; size_t n; bool clean; } utf[] = {
        {"\xed\xa0\x80", 3, false}, {"\xe0\x80\x80", 3, false}, {"\xf4\x90\x80\x80", 4, false}, {"\xf5\x80\x80\x80", 4, false},
        {"\xf0\x80\x80\x80", 4, false}, {"\xc0\x80", 2, false}, {"\xe2\x82\xac", 3, true}, {"\xf0\x9f\x99\x82", 4, true}, {"\xed\x9f\xbf", 3, true},
    };
    for (size_t i = 0; i < sizeof(utf) / sizeof(utf[0]); i++) {
        buf b = {0};
        lp_put_token_string(&b, utf[i].s, utf[i].n);
        const bool has_fffd = strstr(b.ptr, "\xef\xbf\xbd") != NULL;
        TEST_ASSERT(has_fffd == !utf[i].clean);
        buf_free(&b);
    }

    /* tokens that start after a stop sequence's cut are not reported */
    {
        request lr;
        request_init(&lr, REQ_COMPLETION, 8);
        lr.logprobs = true;
        lr.lp = xmalloc(sizeof(*lr.lp));
        memset(lr.lp, 0, sizeof(*lr.lp));
        lr.lp->v = xmalloc(3 * sizeof(lr.lp->v[0]));
        lr.lp->n = lr.lp->cap = 3;
        for (int i = 0; i < 3; i++) {
            lr.lp->v[i].start = (size_t)i;
            lr.lp->v[i].entry = xstrdup("{}");
            lr.lp->v[i].ts = xstrdup("\"t\"");
            lr.lp->v[i].ls = xstrdup("-1.0");
            lr.lp->v[i].ps = xstrdup("{}");
        }
        lr.lp->off = 3;
        lr.lp_limited = true;
        lr.lp_limit = 2;
        buf b = {0};
        lp_append_json(&b, &lr);
        TEST_ASSERT(strstr(b.ptr, "\"text_offset\":[0,1]}") != NULL);
        buf_free(&b);
        request_free(&lr);
    }

    /* a streamed call the turn parser would not return is counted bad (at every split); a good one is not */
    static const struct { const char *raw; bool bad; } calls[] = {
        {"<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}<|end_message|>", false},
        {"<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":tru}}<|end_message|>", true},
        {"<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1]}<|end_message|>", true},
        {"<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1},\"args\":{\"b\":2}}<|end_message|>", true},
        {"<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1},\"name\":\"g\"}<|end_message|>", true},
        {"<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}<|end_message|>"
         "<|message_model|>g<|content_invoke_tool_json|>{\"name\":\"g\",\"args\":\"str\"}<|end_message|>", false},   /* not streamed at all: the server sees streamed 1 > parsed 0 */
    };
    for (size_t i = 0; i < sizeof(calls) / sizeof(calls[0]); i++) {
        const size_t n = strlen(calls[i].raw);
        for (size_t at = 0; at <= n; at++) {
            ik_collect c = {0};
            ik_stream st = {0};
            st.want_tools = true;
            if (at) ik_stream_step(&st, calls[i].raw, at, false, ik_collect_emit, &c);
            ik_stream_step(&st, calls[i].raw, n, true, ik_collect_emit, &c);
            TEST_ASSERT((st.tool_bad > 0) == calls[i].bad);
            ik_collect_free(&c);
        }
    }
}

/* ---- fuzzing: mutated request bodies through the four request parsers, mutated generations through the stream tracker.
 * Seconds from DS4_INKLING2_FUZZ_SECS (default 2; the minutes-long runs under -fsanitize=address,undefined set it higher),
 * seed from DS4_INKLING2_FUZZ_SEED.  The stream check also needs the same events from one whole step and from random splits. */
static uint64_t ik_fz_state = 0x9e3779b97f4a7c15ull;
static uint32_t ik_fz_rand(void) {
    ik_fz_state ^= ik_fz_state << 13;
    ik_fz_state ^= ik_fz_state >> 7;
    ik_fz_state ^= ik_fz_state << 17;
    return (uint32_t)(ik_fz_state >> 16);
}

static char *ik_fz_mutate(const char *s, size_t n, size_t *out_n) {
    static const char *const bits[] = {"\"", "{", "}", "[", "]", "\\", ",", ":", "null", "<|end_message|>", "<|message_model|>",
                                       "<|content_invoke_tool_json|>", "<|content_text|>", "<|content_thinking|>", "\xc3", "\xff", "\\u00e9",
                                       "{\"name\":\"f\",\"args\":{", "\"args\":", "<|content_model_end_sampling|>", " ", "\n"};
    buf b = {0};
    buf_append(&b, s, n);
    const int ops = 1 + (int)(ik_fz_rand() % 4);
    for (int k = 0; k < ops && b.len; k++) {
        const size_t at = ik_fz_rand() % b.len;
        switch (ik_fz_rand() % 6) {
        case 0: b.ptr[at] = (char)(ik_fz_rand() & 0xff); break;
        case 1: {   /* delete a range */
            size_t len = 1 + ik_fz_rand() % 40;
            if (at + len > b.len) len = b.len - at;
            memmove(b.ptr + at, b.ptr + at + len, b.len - at - len);
            b.len -= len;
            b.ptr[b.len] = 0;
            break;
        }
        case 2: {   /* duplicate a range */
            size_t len = 1 + ik_fz_rand() % 60;
            if (at + len > b.len) len = b.len - at;
            char *tmp = xstrndup(b.ptr + at, len);
            buf nb = {0};
            buf_append(&nb, b.ptr, at);
            buf_append(&nb, tmp, len);
            buf_append(&nb, b.ptr + at, b.len - at);
            free(tmp);
            buf_free(&b);
            b = nb;
            break;
        }
        case 3: case 4: {   /* insert a snippet */
            const char *bit = bits[ik_fz_rand() % (sizeof(bits) / sizeof(bits[0]))];
            buf nb = {0};
            buf_append(&nb, b.ptr, at);
            buf_puts(&nb, bit);
            buf_append(&nb, b.ptr + at, b.len - at);
            buf_free(&b);
            b = nb;
            break;
        }
        default: b.len = at; b.ptr[at] = 0; break;   /* truncate */
        }
    }
    for (size_t i = 0; i < b.len; i++) if (!b.ptr[i]) b.ptr[i] = 'x';   /* the parsers take C strings */
    *out_n = b.len;
    return buf_take(&b);
}

static bool ik_fz_same(const ik_collect *a, const ik_collect *b) {
    if (a->n_calls != b->n_calls || a->n_done != b->n_done || a->bad_events != b->bad_events) return false;
    if (strcmp(a->content.ptr ? a->content.ptr : "", b->content.ptr ? b->content.ptr : "")) return false;
    if (strcmp(a->reasoning.ptr ? a->reasoning.ptr : "", b->reasoning.ptr ? b->reasoning.ptr : "")) return false;
    for (int i = 0; i < a->n_calls && i < 16; i++) {
        if (strcmp(a->names[i].ptr ? a->names[i].ptr : "", b->names[i].ptr ? b->names[i].ptr : "")) return false;
        if (strcmp(a->args[i].ptr ? a->args[i].ptr : "", b->args[i].ptr ? b->args[i].ptr : "")) return false;
    }
    return true;
}

static void test_inkling2_fuzz(void) {
    const char *secs_env = getenv("DS4_INKLING2_FUZZ_SECS");
    const double secs = secs_env && secs_env[0] ? atof(secs_env) : 2.0;
    const char *seed_env = getenv("DS4_INKLING2_FUZZ_SEED");
    if (seed_env && seed_env[0]) ik_fz_state = strtoull(seed_env, NULL, 10) | 1;
    char *bodies[256], *raws[256];
    int n_bodies = 0, n_raws = 0;
    /* corpus: the API fixtures (three bodies each) and the generations of the reference vectors */
    DIR *d = opendir("tests/test-vectors/inkling2-api");
    for (struct dirent *de; d && (de = readdir(d)) != NULL && n_bodies < 240;) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".json")) continue;
        char path[1024];
        snprintf(path, sizeof(path), "tests/test-vectors/inkling2-api/%s", de->d_name);
        char *text = ik_test_read_file(path);
        if (!text) continue;
        const char *q = text;
        ik_member *m = NULL;
        int nm = 0;
        if (ik_json_members(&q, &m, &nm)) {
            for (int i = 0; i < nm; i++) {
                if (!strcmp(m[i].key, "name")) continue;
                const char *v = m[i].val;
                char *rv = NULL;
                if (json_raw_value(&v, &rv) && n_bodies < 240) bodies[n_bodies++] = rv; else free(rv);
            }
            ik_members_free(m, nm);
        }
        free(text);
    }
    if (d) closedir(d);
    d = opendir("tests/test-vectors/inkling2-chat");
    for (struct dirent *de; d && (de = readdir(d)) != NULL && n_raws < 250;) {
        size_t l = strlen(de->d_name);
        if (l < 6 || strcmp(de->d_name + l - 5, ".json")) continue;
        char path[1024];
        snprintf(path, sizeof(path), "tests/test-vectors/inkling2-chat/%s", de->d_name);
        char *text = ik_test_read_file(path);
        ik_fixture f = {0};
        if (text && ik_fixture_parse(text, &f) && f.raw && n_raws < 250) raws[n_raws++] = xstrdup(f.raw);
        ik_fixture_free(&f);
        free(text);
    }
    if (d) closedir(d);
    raws[n_raws++] = xstrdup("<|message_model|><|content_thinking|>t<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":[1,2,{\"b\":\"x\\\"y\"}]}}<|end_message|><|content_model_end_sampling|>");
    if (n_bodies == 0 || n_raws < 2) {
        fprintf(stderr, "inkling2 fuzz: corpus not found, skipped\n");
        return;
    }
    const double t0 = ik_test_cpu_s();
    long n_req = 0, n_req_ok = 0, n_stream = 0, n_stream_calls = 0, n_diff = 0;
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    while (ik_test_cpu_s() - t0 < secs) {
        for (int rep = 0; rep < 20; rep++) {
            size_t n = 0;
            const char *src = bodies[ik_fz_rand() % (uint32_t)n_bodies];
            char *mut = ik_fz_mutate(src, strlen(src), &n);
            request r;
            char err[256];
            bool ok;
            switch (ik_fz_rand() % 4) {
            case 0: ok = parse_chat_request(NULL, NULL, mut, 128, 4096, &r, err, sizeof(err)); break;
            case 1: ok = parse_anthropic_request(NULL, NULL, mut, 128, 4096, &r, err, sizeof(err)); break;
            case 2: ok = parse_responses_request(NULL, NULL, mut, 128, 4096, &r, err, sizeof(err)); break;
            default: ok = parse_completion_request(NULL, mut, 128, 4096, &r, err, sizeof(err)); break;
            }
            n_req++;
            if (ok) { n_req_ok++; request_free(&r); }
            free(mut);
        }
        for (int rep = 0; rep < 20; rep++) {
            size_t n = 0;
            const char *src = raws[ik_fz_rand() % (uint32_t)n_raws];
            char *mut = (ik_fz_rand() % 4) ? ik_fz_mutate(src, strlen(src), &n) : xstrdup(src);
            n = strlen(mut);
            const bool want = (ik_fz_rand() & 1) != 0;
            ik_collect whole = {0}, split = {0};
            ik_stream sw = {0}, ss = {0};
            sw.want_tools = ss.want_tools = want;
            ik_stream_step(&sw, mut, n, true, ik_collect_emit, &whole);
            size_t at = 0;
            while (at < n) {
                at += 1 + ik_fz_rand() % (n / 3 + 2);
                if (at >= n) break;
                ik_stream_step(&ss, mut, at, false, ik_collect_emit, &split);
            }
            ik_stream_step(&ss, mut, n, true, ik_collect_emit, &split);
            n_stream++;
            if (whole.n_calls) n_stream_calls++;
            if (!ik_fz_same(&whole, &split)) {
                if (n_diff++ < 3) fprintf(stderr, "inkling2 fuzz: stream differs between one step and a split run on: %.200s\n", mut);
            }
            /* the parser must not crash on the same text either */
            char *c = NULL, *rs = NULL;
            tool_calls calls = {0};
            if (parse_inkling_generated_message_ex(mut, want, &c, &rs, &calls)) { free(c); free(rs); }
            tool_calls_free(&calls);
            ik_collect_free(&whole);
            ik_collect_free(&split);
            free(mut);
        }
    }
    g_test_force_syntax = -1;
    g_effort_fine_names = false;
    fprintf(stderr, "inkling2 fuzz (%.1f s CPU, seed state %llu): %ld requests (%ld accepted), %ld generations (%ld with a streamed call), stream mismatches %ld\n",
            ik_test_cpu_s() - t0, (unsigned long long)ik_fz_state, n_req, n_req_ok, n_stream, n_stream_calls, n_diff);
    TEST_ASSERT(n_diff == 0);
    for (int i = 0; i < n_bodies; i++) free(bodies[i]);
    for (int i = 0; i < n_raws; i++) free(raws[i]);
}

/* Claude Code against an Inkling server: media in tool results become text placeholders instead of a 400 that stays in the
 * client's history, tool_reference blocks (tool search) become a line of text, and the context overflow error says what Claude Code
 * looks for to compact.  Other families behave as before. */
static void test_inkling2_claude_code_compat(void) {
    static const char *const body_fmt =
        "{\"model\":\"m\",\"max_tokens\":100,\"messages\":[{\"role\":\"user\",\"content\":\"read it\"},"
        "{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"Read\",\"input\":{}}]},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_1\",\"content\":[%s]}]}]}";
    static const struct { const char *block; const char *want; } cases[] = {
        {"{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"iVBORw0KGgo=\"}}", "[image omitted: this model cannot read it]"},
        {"{\"type\":\"document\",\"source\":{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"JVBERi0=\"},\"title\":\"x\"}", "[document omitted: this model cannot read it]"},
        {"{\"type\":\"tool_reference\",\"tool_name\":\"WebFetch\"}", "[tool loaded: WebFetch]"},
        {"{\"type\":\"search_result\",\"source\":\"s\",\"title\":\"t\",\"content\":[{\"type\":\"text\",\"text\":\"c\"}]}", "[search result omitted: this model cannot read it]"},
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        for (int syntax = 0; syntax < 2; syntax++) {
            buf bb = {0};
            buf_printf(&bb, body_fmt, cases[k].block);
            char *body = buf_take(&bb);
            g_test_force_syntax = syntax ? SERVER_MODEL_SYNTAX_INKLING : SERVER_MODEL_SYNTAX_QWEN;
            request r;
            char err[256] = "";
            const bool ok = parse_anthropic_request(NULL, NULL, body, 128, 4096, &r, err, sizeof(err));
            if (syntax) {
                TEST_ASSERT(ok);
                TEST_ASSERT(ok && r.prompt_text && strstr(r.prompt_text, cases[k].want) != NULL);
            } else {
                /* unchanged for the other families: a document, tool_reference or search_result in a tool result is still refused */
                if (strstr(cases[k].block, "\"image\"") == NULL) TEST_ASSERT(!ok);
            }
            if (ok) request_free(&r);
            free(body);
        }
    }
    /* an image the user sends directly is still refused for Inkling (no vision): the placeholder is only for tool results */
    {
        g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
        request r;
        char err[256] = "";
        const char *body = "{\"model\":\"m\",\"max_tokens\":10,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"iVBORw0KGgo=\"}}]}]}";
        const bool ok = parse_anthropic_request(NULL, NULL, body, 128, 4096, &r, err, sizeof(err));
        TEST_ASSERT(!ok);
        if (ok) request_free(&r);
    }
    g_test_force_syntax = -1;
    /* overflow wording */
    for (int syntax = 0; syntax < 2; syntax++) {
        int sv[2];
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        if (sv[0] < 0 || sv[1] < 0) return;
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.api = API_ANTHROPIC;
        r.model_syntax = syntax ? SERVER_MODEL_SYNTAX_INKLING : SERVER_MODEL_SYNTAX_QWEN;
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], false, &r, 55867, 32768));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        if (syntax) TEST_ASSERT(strstr(out, "prompt is too long: 55867 tokens > 32768 maximum") != NULL);
        else TEST_ASSERT(strstr(out, "Prompt has 55867 tokens, but the configured context size is 32768 tokens") != NULL);
        free(out);
        request_free(&r);
        close(sv[0]);
        close(sv[1]);
    }
}

/* A message with thousands of content blocks parses and renders in linear time for Inkling (it was quadratic: 20,000 blocks 3 s),
 * and a message with many tool call ids still finds duplicates. */
static void test_inkling2_many_blocks_are_linear(void) {
    double t[2] = {0, 0};
    const int ns[2] = {8000, 32000};
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    for (int w = 0; w < 2; w++) {
        const int n = ns[w];
        buf b = {0};
        buf_puts(&b, "{\"model\":\"m\",\"max_tokens\":100,\"tools\":[{\"name\":\"f\",\"input_schema\":{\"type\":\"object\"}}],\"messages\":[{\"role\":\"user\",\"content\":\"go\"},{\"role\":\"assistant\",\"content\":[");
        for (int i = 0; i < n; i++) buf_printf(&b, "%s{\"type\":\"tool_use\",\"id\":\"toolu_%d\",\"name\":\"f\",\"input\":{}}", i ? "," : "", i);
        buf_puts(&b, "]},{\"role\":\"user\",\"content\":[");
        for (int i = 0; i < n; i++) buf_printf(&b, "%s{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_%d\",\"content\":\"result %d\"},{\"type\":\"text\",\"text\":\"note %d\"}", i ? "," : "", i, i, i);
        buf_puts(&b, "]}]}");
        request r;
        char err[256] = "";
        const double t0 = ik_test_cpu_s();
        const bool ok = parse_anthropic_request(NULL, NULL, b.ptr, 128, 1 << 30, &r, err, sizeof(err));
        t[w] = ik_test_cpu_s() - t0;
        TEST_ASSERT(ok);
        if (ok) {
            TEST_ASSERT(r.prompt_text && strstr(r.prompt_text, "<|message_tool|>f<|content_text|>result 7<|end_message|>") != NULL);
            request_free(&r);
        }
        buf_free(&b);
    }
    g_test_force_syntax = -1;
    fprintf(stderr, "inkling2 many blocks: %d blocks %.3f s, %d blocks %.3f s\n", ns[0], t[0], ns[1], t[1]);
    TEST_ASSERT(t[1] < 0.6);                      /* was about 11 s */
    TEST_ASSERT(t[1] < 8.0 * (t[0] + 0.005));    /* 4x the blocks: linear is 4x, quadratic 16x */
    chat_msg m = {0};
    for (int rep = 0; rep < 2; rep++)
        for (int i = 0; i < 200; i++) {
            char id[32];
            snprintf(id, sizeof(id), "toolu_%d", i);
            chat_msg_add_tool_call_id(&m, id);
        }
    TEST_ASSERT(m.tool_call_ids_len == 200);
    chat_msg_free(&m);
}

/* Responses stream, thinking / answer / thinking: the second thinking block must not stream into the closed reasoning item (the
 * completed response then said "AC" while the item's done event said "A"). */
static void test_inkling2_responses_second_thinking_block(void) {
    const char *raw =
        "<|message_model|><|content_thinking|>AAA<|end_message|>"
        "<|message_model|><|content_text|>BBB<|end_message|>"
        "<|message_model|><|content_thinking|>CCC<|end_message|>"
        "<|content_model_end_sampling|>";
    static const size_t steps[] = {1, 5, 4096};
    for (size_t si = 0; si < sizeof(steps) / sizeof(steps[0]); si++) {
        int sv[2];
        TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        if (sv[0] < 0 || sv[1] < 0) return;
        int big = 4 << 20;
        setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
        setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &big, sizeof(big));
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.api = API_RESPONSES;
        r.stream = true;
        r.model_syntax = SERVER_MODEL_SYNTAX_INKLING;
        r.think_mode = DS4_THINK_HIGH;
        r.reasoning_summary_emit = true;
        responses_stream st;
        responses_stream_init(&r, &st);
        st.active = true;
        TEST_ASSERT(responses_sse_created(sv[0], &r, &st, 1234));
        const size_t n = strlen(raw);
        for (size_t at = steps[si]; at < n; at += steps[si]) TEST_ASSERT(responses_sse_stream_update(sv[0], &r, &st, raw, at, false));
        TEST_ASSERT(responses_sse_finish_live(sv[0], &r, &st, raw, n, NULL, NULL, "stop", 10, 6, 1234));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        const char *done = strstr(out, "\"type\":\"response.output_item.done\"");
        TEST_ASSERT(done != NULL);
        int deltas_after_done = 0, deltas = 0;
        for (const char *q = out; (q = strstr(q, "response.reasoning_summary_text.delta")) != NULL; q++) {
            deltas++;
            if (done && q > done) deltas_after_done++;
        }
        TEST_ASSERT(deltas >= 1 && deltas_after_done == 0);
        TEST_ASSERT(strstr(out, "CCC") == NULL);       /* the dropped block is nowhere in the stream */
        TEST_ASSERT(strstr(out, "AAA") != NULL && strstr(out, "BBB") != NULL);
        free(out);
        responses_stream_free(&st);
        request_free(&r);
        close(sv[0]);
        close(sv[1]);
    }
}

/* /v1/completions for Inkling: token-id prompts, a list of one prompt; lists of several prompts, echo, suffix, n and best_of are
 * refused with a message naming them.  Other families still ignore those fields. */
static void test_inkling2_completions_prompts(void) {
    static const struct { const char *body; bool ok; const char *msg; int ntok; } cases[] = {
        {"{\"prompt\":\"hello\"}", true, NULL, -1},
        {"{\"prompt\":[\"hello\"]}", true, NULL, -1},
        {"{\"prompt\":[1,2,3]}", true, NULL, 3},
        {"{\"prompt\":[[5,6]]}", true, NULL, 2},
        {"{\"prompt\":[\"a\",\"b\"]}", false, "more than one prompt", 0},
        {"{\"prompt\":[[1],[2]]}", false, "more than one prompt", 0},
        {"{\"prompt\":[1,\"a\"]}", false, "list of token ids", 0},
        {"{\"prompt\":5}", false, "list of token ids", 0},
        {"{\"prompt\":null}", false, "list of token ids", 0},
        {"{\"prompt\":[]}", false, "must not be empty", 0},
        {"{\"prompt\":[1,-2]}", false, "non-negative", 0},
        {"{\"prompt\":\"x\",\"echo\":true}", false, "echo is not supported", 0},
        {"{\"prompt\":\"x\",\"echo\":false}", true, NULL, -1},
        {"{\"prompt\":\"x\",\"suffix\":\"tail\"}", false, "suffix is not supported", 0},
        {"{\"prompt\":\"x\",\"suffix\":null}", true, NULL, -1},
        {"{\"prompt\":\"x\",\"n\":2}", false, "n is not supported", 0},
        {"{\"prompt\":\"x\",\"n\":1}", true, NULL, -1},
        {"{\"prompt\":\"x\",\"best_of\":3}", false, "best_of is not supported", 0},
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        for (int syntax = 1; syntax < 2; syntax++) {   /* the other families need an engine to tokenize: their code is untouched */
            g_test_force_syntax = syntax ? SERVER_MODEL_SYNTAX_INKLING : SERVER_MODEL_SYNTAX_QWEN;
            request r;
            char err[256] = "";
            const bool ok = parse_completion_request(NULL, cases[k].body, 128, 4096, &r, err, sizeof(err));
            if (syntax) {
                if (ok != cases[k].ok) fprintf(stderr, "completions case %zu: %s -> ok %d err %s\n", k, cases[k].body, (int)ok, err);
                TEST_ASSERT(ok == cases[k].ok);
                if (!ok && cases[k].msg) TEST_ASSERT(strstr(err, cases[k].msg) != NULL);
                if (ok && cases[k].ntok >= 0) TEST_ASSERT(r.prompt.len == cases[k].ntok);
            } else if (strstr(cases[k].body, "echo") || strstr(cases[k].body, "\"n\"") || strstr(cases[k].body, "best_of") || strstr(cases[k].body, "suffix")) {
                TEST_ASSERT(ok);   /* unchanged for the other families: the field is ignored */
            }
            if (ok) request_free(&r);
        }
    }
    g_test_force_syntax = -1;
}

/* A stop string holds the last bytes of the text back, so the end marker of the last call is only seen in the final step.  The
 * verdict "a streamed call is not what the turn parser returned" must be taken after that step (both stream shapes). */
static void test_inkling2_stream_verdict_waits_for_final_step(void) {
    static const struct { const char *raw; int parsed; bool bad; } cases[] = {
        /* the tracker streamed f({"a":1}); the parser reads the second "name" and returns g: bad */
        {"<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1},\"name\":\"g\"}<|end_message|>", 1, true},
        {"<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}<|end_message|>", 1, false},
    };
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        for (int hold = 0; hold <= 20; hold += (hold < 3 ? 1 : 17)) {
            for (int anth = 0; anth < 2; anth++) {
                int sv[2];
                TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
                if (sv[0] < 0 || sv[1] < 0) return;
                int big = 4 << 20;
                setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
                setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &big, sizeof(big));
                request r;
                request_init(&r, REQ_CHAT, 128);
                r.api = anth ? API_ANTHROPIC : API_OPENAI;
                r.stream = true;
                r.model_syntax = SERVER_MODEL_SYNTAX_INKLING;
                r.think_mode = DS4_THINK_HIGH;
                r.has_tools = true;
                openai_stream os;
                anthropic_stream as;
                if (anth) TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "m", 10, &as)); else openai_stream_start(&r, &os);
                const size_t n = strlen(cases[k].raw), held = n - (size_t)hold;
                const ik_stream *iks = anth ? &as.ik : &os.ik;
                bool ok = anth ? anthropic_sse_stream_update(sv[0], NULL, &r, "m", &as, cases[k].raw, held, false)
                               : openai_sse_stream_update(sv[0], NULL, &r, "c", &os, cases[k].raw, held, false);
                TEST_ASSERT(ok);
                if (hold > 0) TEST_ASSERT(iks->tool_bad == 0);   /* the end marker is still held back: nothing can be judged yet */
                bool bad = !cases[k].bad;
                TEST_ASSERT(inkling_stream_settle(sv[0], NULL, &r, anth ? "m" : "c", anth ? NULL : &os, anth ? &as : NULL,
                                                  cases[k].raw, n, cases[k].parsed, &bad));
                TEST_ASSERT(bad == cases[k].bad);
                /* settling twice is harmless: finish_live runs the same final step again */
                bad = !cases[k].bad;
                TEST_ASSERT(inkling_stream_settle(sv[0], NULL, &r, anth ? "m" : "c", anth ? NULL : &os, anth ? &as : NULL,
                                                  cases[k].raw, n, cases[k].parsed, &bad));
                TEST_ASSERT(bad == cases[k].bad);
                if (anth) anthropic_stream_free(&as); else openai_stream_free(&os);
                request_free(&r);
                close(sv[0]);
                close(sv[1]);
            }
        }
    }
    g_test_force_syntax = -1;
}

/* A tool call cut off at the end of the generation, with a lone UTF-8 continuation byte after the last complete token of the
 * arguments: the final step must send the same arguments whatever steps came before (found by the 900 s fuzz run, seed 6061). */
static void test_inkling2_cut_off_call_with_stray_byte(void) {
    static const char *const raws[] = {
        "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"city\":\"Oslo\",\"days\"\x8e",
        "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":\"x\xc3",
        "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":\"\xe2\x82\"",
    };
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    int bad = 0, runs = 0;
    for (size_t k = 0; k < sizeof(raws) / sizeof(raws[0]); k++) {
        const char *raw = raws[k];
        const size_t n = strlen(raw);
        for (int want = 0; want < 2; want++) {
            ik_collect whole = {0};
            ik_stream sw = {0};
            sw.want_tools = want;
            ik_stream_step(&sw, raw, n, true, ik_collect_emit, &whole);
            for (size_t at = 1; at < n; at++) {
                ik_collect split = {0};
                ik_stream ss = {0};
                ss.want_tools = want;
                ik_stream_step(&ss, raw, at, false, ik_collect_emit, &split);
                ik_stream_step(&ss, raw, n, true, ik_collect_emit, &split);
                runs++;
                if (!ik_fz_same(&whole, &split)) bad++;
                ik_collect_free(&split);
            }
            ik_collect_free(&whole);
        }
    }
    g_test_force_syntax = -1;
    TEST_ASSERT(runs > 0 && bad == 0);
}

/* Regressions for small request-path defects. */
static void test_inkling2_small_fixes(void) {
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    /* a tool call whose args are not a JSON object is a malformed call (the validator refuses them on the next turn) */
    TEST_ASSERT(!parse_inkling_generated_message_ex(
        "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":[1,2]}<|end_message|>", true, &content, &reasoning, &calls));
    tool_calls_free(&calls);
    calls = (tool_calls){0};
    TEST_ASSERT(parse_inkling_generated_message_ex(
        "<|message_model|><|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}<|end_message|>", true, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    tool_calls_free(&calls);
    free(content); free(reasoning);

    request r;
    char err[256] = {0};
    g_test_force_syntax = SERVER_MODEL_SYNTAX_INKLING;
    /* bogus reasoning_effort: the message names the field */
    TEST_ASSERT(!parse_chat_request(NULL, NULL, "{\"reasoning_effort\":\"bogus\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", 128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "reasoning_effort"));
    /* empty messages are refused */
    err[0] = 0;
    TEST_ASSERT(!parse_chat_request(NULL, NULL, "{\"messages\":[]}", 128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "must not be empty"));
    /* one entry in tools that is not an object used to drop every tool silently */
    err[0] = 0;
    TEST_ASSERT(!parse_chat_request(NULL, NULL,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{}}},5]}",
        128, 32768, &r, err, sizeof(err)));
    /* tool arguments nested more than 64 levels */
    buf deep = {0};
    buf_puts(&deep, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"");
    for (int i = 0; i < 100; i++) buf_puts(&deep, "{\\\"a\\\":");
    buf_puts(&deep, "1");
    for (int i = 0; i < 100; i++) buf_putc(&deep, '}');
    buf_puts(&deep, "\"}}]}]}");
    err[0] = 0;
    TEST_ASSERT(!parse_chat_request(NULL, NULL, deep.ptr, 128, 32768, &r, err, sizeof(err)));
    TEST_ASSERT(strstr(err, "64 levels"));
    buf_free(&deep);
    g_test_force_syntax = -1;
    g_effort_fine_names = false;
}

static void test_inkling2_chat(void) {
    test_inkling2_effort_mapping();
    test_inkling2_model_ids();
    test_inkling2_float_repr();
    test_inkling2_render_details();
    test_inkling2_parse_cases();
    test_inkling2_llamacpp_parser_cases();
    test_inkling2_stream_holds_markers();
    test_inkling2_stream_whitespace_and_split_end();
    test_inkling2_refused_endpoints();
    test_inkling2_canonical_json_scales();
    test_inkling2_small_fixes();
    test_inkling2_openai_incremental_tool_stream();
    test_inkling2_api_equivalence();
    test_inkling2_request_regressions();
    test_inkling2_fuzz();
    test_inkling2_cut_off_call_with_stray_byte();
    test_inkling2_stream_verdict_waits_for_final_step();
    test_inkling2_claude_code_compat();
    test_inkling2_completions_prompts();
    test_inkling2_responses_second_thinking_block();
    test_inkling2_many_blocks_are_linear();
    test_inkling2_reference_vectors();
}
