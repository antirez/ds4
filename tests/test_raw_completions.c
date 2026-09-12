/* Exercise the real server parser and live-prefix path without model weights.
 * Model-dependent operations and disk restoration are replaced; token
 * containers, JSON parsing, prompt construction and reuse are production code. */
#define ds4_engine_is_glm_dsa fixture_is_glm
#define ds4_engine_vocab_size fixture_vocab_size
#define ds4_chat_begin fixture_chat_begin
#define ds4_tokenize_rendered_chat fixture_tokenize
#define ds4_session_tokens fixture_session_tokens
#define ds4_session_rebase_vision_state fixture_rebase_vision
#define ds4_session_vision_prefix_matches fixture_vision_matches
#define ds4_kvstore_render_tokens_text fixture_render_tokens
#define ds4_session_has_vision_state fixture_has_vision
#define ds4_session_invalidate fixture_invalidate
#define ds4_kvstore_try_load_text fixture_load_text
#define ds4_token_is_stop fixture_is_stop
#define ds4_token_is_stop_for_think_mode fixture_is_stop_for_think_mode
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

struct ds4_engine { bool glm; };
struct ds4_session { ds4_tokens tokens; bool vision_matches; int invalidations; };

enum { BOS = 256, GMASK, SOP, THINK, MERGED_AB, MERGED_ABC, END_THINK, EOS, VOCAB_SIZE };

bool fixture_is_glm(ds4_engine *e) { return e && e->glm; }
int fixture_vocab_size(ds4_engine *e) { return e ? VOCAB_SIZE : 0; }

void fixture_chat_begin(ds4_engine *e, ds4_tokens *out) {
    ds4_tokens_push(out, e->glm ? GMASK : BOS);
    if (e->glm) ds4_tokens_push(out, SOP);
}

static const char *fixture_token_spelling(int token) {
    switch (token) {
    case BOS: return "<|begin_of_sentence|>";
    case GMASK: return "[gMASK]";
    case SOP: return "<sop>";
    case THINK: return "<think>";
    case MERGED_AB: return "ab";
    case MERGED_ABC: return "abc";
    case END_THINK: return "</think>";
    case EOS: return "<eos>";
    default: return NULL;
    }
}

void fixture_tokenize(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e;
    while (*text) {
        int match = -1;
        size_t match_len = 0;
        for (int token = BOS; token < VOCAB_SIZE; token++) {
            const char *spelling = fixture_token_spelling(token);
            size_t len = strlen(spelling);
            if (len > match_len && !strncmp(text, spelling, len)) {
                match = token;
                match_len = len;
            }
        }
        if (match >= 0) {
            ds4_tokens_push(out, match);
            text += match_len;
        } else {
            ds4_tokens_push(out, (unsigned char)*text++);
        }
    }
}

char *fixture_render_tokens(ds4_engine *e, const ds4_tokens *tokens,
                            size_t *out_len) {
    (void)e;
    buf text = {0};
    for (int i = 0; i < tokens->len; i++) {
        const char *spelling = fixture_token_spelling(tokens->v[i]);
        if (spelling) buf_puts(&text, spelling);
        else buf_putc(&text, (char)tokens->v[i]);
    }
    if (out_len) *out_len = text.len;
    return text.ptr ? text.ptr : xstrdup("");
}

const ds4_tokens *fixture_session_tokens(ds4_session *s) { return &s->tokens; }

bool fixture_rebase_vision(const ds4_session *s,
                          ds4_vision_span *images, size_t count) {
    (void)images;
    return s->vision_matches && count == 0;
}

bool fixture_vision_matches(const ds4_session *s,
                           const ds4_vision_span *images, size_t count) {
    (void)images;
    return s->vision_matches && count == 0;
}

bool fixture_has_vision(const ds4_session *s) { (void)s; return false; }
void fixture_invalidate(ds4_session *s) {
    s->invalidations++;
    ds4_tokens_free(&s->tokens);
}

static char *disk_lookup_text;
static int disk_lookups;

int fixture_load_text(ds4_kvstore *kc, ds4_engine *e, ds4_session *session,
                      const char *text, ds4_tokens *effective,
                      ds4_kvstore_load_result *result,
                      const ds4_kvstore_trailer_hooks *hooks,
                      bool responses_protocol) {
    (void)kc; (void)e; (void)hooks; (void)responses_protocol;
    disk_lookups++;
    free(disk_lookup_text);
    disk_lookup_text = text ? xstrdup(text) : NULL;
    /* The payload is supplied through session.tokens; the wrapper must
     * validate its actual tokens, not trust the disk entry's visible key. */
    ds4_tokens_push(effective, 'z');
    result->path = xstrdup("fixture.kv");
    result->ext_flags = KV_EXT_TOOL_MAP;
    return session->tokens.len;
}

bool fixture_is_stop(ds4_engine *e, int token) { (void)e; return token == EOS; }
bool fixture_is_stop_for_think_mode(ds4_engine *e, int token,
                                    ds4_think_mode mode) {
    return fixture_is_stop(e, token) ||
        (mode == DS4_THINK_NONE && (token == THINK || token == END_THINK));
}

static void expect_tokens(const ds4_tokens *actual, const int *expected, int n) {
    TEST_ASSERT(actual->len == n);
    for (int i = 0; i < actual->len && i < n; i++)
        TEST_ASSERT(actual->v[i] == expected[i]);
}

static void test_raw_strings(ds4_engine *e) {
    request r;
    char err[160] = {0};
    const char *body = "{\"prompt\":\"ab<think>\",\"max_tokens\":0,"
        "\"thinking\":true,\"think\":true,\"reasoning_effort\":\"max\"}";
    bool ok = parse_completion_request(e, body, 128, 4096, &r, err, sizeof(err));
    TEST_ASSERT(ok);
    if (!ok) return;
    const int deepseek[] = {BOS, MERGED_AB, THINK};
    const int glm[] = {GMASK, SOP, MERGED_AB, THINK};
    expect_tokens(&r.prompt, e->glm ? glm : deepseek, e->glm ? 4 : 3);
    TEST_ASSERT(r.prompt_text && !strcmp(r.prompt_text, "ab<think>"));
    TEST_ASSERT(r.think_mode == DS4_THINK_NONE);
    TEST_ASSERT(r.max_tokens == 0);
    request_free(&r);

    ok = parse_completion_request(e, "{\"prompt\":\"\"}", 128, 4096,
                                  &r, err, sizeof(err));
    TEST_ASSERT(ok);
    if (ok) {
        expect_tokens(&r.prompt, e->glm ? glm : deepseek, e->glm ? 2 : 1);
        request_free(&r);
    }
}

static void test_exact_token_ids(ds4_engine *e) {
    request r;
    char err[160] = {0};
    const int expected[] = {97, MERGED_AB, THINK};
    char body[96];
    snprintf(body, sizeof(body), "{\"prompt\":[97,%d,%d],\"stream\":true}",
             MERGED_AB, THINK);
    bool ok = parse_completion_request(e, body, 128, 4096, &r, err, sizeof(err));
    TEST_ASSERT(ok);
    if (ok) {
        expect_tokens(&r.prompt, expected, 3);
        TEST_ASSERT(r.prompt_text == NULL);
        TEST_ASSERT(r.stream);
        request_free(&r);
    }
    snprintf(body, sizeof(body), "{\"prompt\":[%d]}", VOCAB_SIZE);
    ok = parse_completion_request(e, body, 128, 4096, &r, err, sizeof(err));
    TEST_ASSERT(!ok);
    if (ok) request_free(&r);
}

static void test_raw_live_prefix(ds4_engine *e) {
    ds4_session session = {.vision_matches = true};
    server s = {.engine = e};
    server_slot slot = {.session = &session};
    request r;
    char err[160] = {0};
    bool ok = parse_completion_request(e, "{\"prompt\":\"abc\"}", 128, 4096,
                                      &r, err, sizeof(err));
    TEST_ASSERT(ok);
    if (!ok) return;
    /* Full prompt tokenization merges abc. Reuse must keep the cached ab
     * token and tokenize only c, rather than use the full-prompt token suffix. */
    const int full_deepseek[] = {BOS, MERGED_ABC};
    const int full_glm[] = {GMASK, SOP, MERGED_ABC};
    expect_tokens(&r.prompt, e->glm ? full_glm : full_deepseek, e->glm ? 3 : 2);
    fixture_chat_begin(e, &session.tokens);
    ds4_tokens_push(&session.tokens, MERGED_AB);
    ds4_tokens effective = {0};
    int cached = live_text_prefix_prompt(&s, &slot, &r, &effective);
    TEST_ASSERT(cached == session.tokens.len);
    const int reused_deepseek[] = {BOS, MERGED_AB, 'c'};
    const int reused_glm[] = {GMASK, SOP, MERGED_AB, 'c'};
    expect_tokens(&effective, e->glm ? reused_glm : reused_deepseek,
                  e->glm ? 4 : 3);
    ds4_tokens_free(&effective);

    session.tokens.v[session.tokens.len - 1] = 'z';
    TEST_ASSERT(live_text_prefix_prompt(&s, &slot, &r, &effective) == 0);
    TEST_ASSERT(effective.len == 0);
    session.tokens.v[session.tokens.len - 1] = MERGED_AB;
    session.vision_matches = false;
    TEST_ASSERT(live_text_prefix_prompt(&s, &slot, &r, &effective) == 0);
    TEST_ASSERT(effective.len == 0);
    session.vision_matches = true;

    /* A prior token-array request may omit start tokens; identical text is
     * insufficient to reuse that state for a string request which adds them. */
    ds4_tokens_free(&session.tokens);
    ds4_tokens_push(&session.tokens, MERGED_AB);
    TEST_ASSERT(live_text_prefix_prompt(&s, &slot, &r, &effective) == 0);
    TEST_ASSERT(effective.len == 0);
    ds4_tokens_free(&session.tokens);
    ds4_tokens_free(&effective);
    request_free(&r);
}

static void test_raw_disk_prefix(ds4_engine *e) {
    ds4_session session = {.vision_matches = true};
    server s = {.engine = e, .kv = {.enabled = true}};
    server_slot slot = {.session = &session};
    TEST_ASSERT(pthread_mutex_init(&s.inference_mu, NULL) == 0);
    TEST_ASSERT(pthread_mutex_init(&s.kv_mu, NULL) == 0);
    for (int explicit_start = 0; explicit_start < 2; explicit_start++) {
        const char *start = e->glm ? "[gMASK]<sop>" : "<|begin_of_sentence|>";
        char body[160], expected_text[160];
        snprintf(body, sizeof(body), "{\"prompt\":\"%sabc\"}",
                 explicit_start ? start : "");
        snprintf(expected_text, sizeof(expected_text), "%s%sabc", start,
                 explicit_start ? start : "");
        request r;
        char err[160] = {0};
        bool ok = parse_completion_request(e, body, 128, 4096, &r, err, sizeof(err));
        TEST_ASSERT(ok);
        if (!ok) continue;
        fixture_chat_begin(e, &session.tokens);
        if (explicit_start) fixture_chat_begin(e, &session.tokens);
        ds4_tokens_push(&session.tokens, MERGED_AB);
        ds4_tokens expected = {0}, effective = {0};
        ds4_tokens_copy(&expected, &session.tokens);
        ds4_tokens_push(&expected, 'c');
        char *path = NULL;
        uint8_t flags = 0;
        disk_lookups = 0;
        int loaded = kv_cache_try_load(&s, &slot, &r, &effective, &path, &flags);
        TEST_ASSERT(loaded == session.tokens.len);
        TEST_ASSERT(disk_lookups == 1);
        TEST_ASSERT(disk_lookup_text && !strcmp(disk_lookup_text, expected_text));
        expect_tokens(&effective, expected.v, expected.len);
        TEST_ASSERT(path && !strcmp(path, "fixture.kv"));
        TEST_ASSERT(flags == KV_EXT_TOOL_MAP);
        free(path);
        ds4_tokens_free(&expected);
        ds4_tokens_free(&effective);
        ds4_tokens_free(&session.tokens);

        if (!explicit_start) {
            for (int invalid = 0; invalid < 3; invalid++) {
                if (invalid == 1) {
                    fixture_chat_begin(e, &session.tokens);
                    ds4_tokens_push(&session.tokens, THINK);
                } else if (invalid == 2) {
                    ds4_tokens_push(&session.tokens, e->glm ? BOS : GMASK);
                }
                ds4_tokens_push(&session.tokens, MERGED_AB);
                path = NULL;
                flags = 0;
                slot.continued_last_store_tokens = 17;
                int invalidations = session.invalidations;
                loaded = kv_cache_try_load(&s, &slot, &r, &effective, &path, &flags);
                TEST_ASSERT(loaded == 0);
                TEST_ASSERT(session.invalidations == invalidations + 1);
                TEST_ASSERT(effective.len == 0);
                TEST_ASSERT(path == NULL);
                TEST_ASSERT(flags == 0);
                TEST_ASSERT(slot.continued_last_store_tokens == 0);
                free(path);
                ds4_tokens_free(&effective);
                ds4_tokens_free(&session.tokens);
            }
        }
        request_free(&r);
    }

    request r;
    char err[160] = {0};
    bool ok = parse_completion_request(e, "{\"prompt\":[97]}", 128, 4096,
                                      &r, err, sizeof(err));
    TEST_ASSERT(ok);
    if (ok) {
        ds4_tokens effective = {0};
        disk_lookups = 0;
        TEST_ASSERT(kv_cache_try_load(&s, &slot, &r, &effective, NULL, NULL) == 0);
        TEST_ASSERT(disk_lookups == 0);
        ds4_tokens_free(&effective);
        request_free(&r);
    }
    free(disk_lookup_text);
    disk_lookup_text = NULL;
    TEST_ASSERT(pthread_mutex_destroy(&s.kv_mu) == 0);
    TEST_ASSERT(pthread_mutex_destroy(&s.inference_mu) == 0);
}

static void test_raw_stop_tokens(ds4_engine *e) {
    request r = {.kind = REQ_COMPLETION, .think_mode = DS4_THINK_NONE};
    TEST_ASSERT(!request_token_is_stop(e, &r, THINK));
    TEST_ASSERT(!request_token_is_stop(e, &r, END_THINK));
    TEST_ASSERT(request_token_is_stop(e, &r, EOS));
    r.kind = REQ_CHAT;
    TEST_ASSERT(request_token_is_stop(e, &r, THINK));
    TEST_ASSERT(request_token_is_stop(e, &r, END_THINK));
    TEST_ASSERT(request_token_is_stop(e, &r, EOS));
    r.think_mode = DS4_THINK_HIGH;
    TEST_ASSERT(!request_token_is_stop(e, &r, THINK));
    TEST_ASSERT(!request_token_is_stop(e, &r, END_THINK));
    TEST_ASSERT(request_token_is_stop(e, &r, EOS));
}

int main(void) {
    for (int glm = 0; glm < 2; glm++) {
        ds4_engine engine = {.glm = glm != 0};
        test_raw_strings(&engine);
        test_exact_token_ids(&engine);
        test_raw_live_prefix(&engine);
        test_raw_disk_prefix(&engine);
        test_raw_stop_tokens(&engine);
    }
    if (test_failures) {
        fprintf(stderr, "raw completion tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("raw completion tests passed (model-free DeepSeek/GLM fixtures)");
    return 0;
}
