#define main quality_scorer_main
#define ds4_session_sync quality_test_sync
#include "../gguf-tools/quality-testing/score_official.c"
#undef main
#undef ds4_session_sync
#include <assert.h>

static int sync_calls, sync_lengths[2], sync_fail;
static const int *sync_tokens;

int quality_test_sync(ds4_session *session, const ds4_tokens *prompt,
                      char *err, size_t errlen) {
    (void)session; (void)err; (void)errlen;
    assert(sync_calls < 2 && prompt->v == sync_tokens);
    sync_lengths[sync_calls++] = prompt->len;
    return sync_fail && sync_calls == sync_fail ? 7 : 0;
}

static void check_continued_prefill(void) {
    int tokens[17] = {0};
    ds4_tokens prompt = {.v = tokens, .len = 17};
    char err[128];
    sync_tokens = tokens;
    assert(sync_prompt(NULL, &prompt, 0, err, sizeof(err)) == 0);
    assert(sync_calls == 1 && sync_lengths[0] == 17);
    const int suffixes[] = {1, 8, 16};
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(*suffixes); i++) {
        sync_calls = 0;
        assert(sync_prompt(NULL, &prompt, suffixes[i], err, sizeof(err)) == 0);
        assert(sync_calls == 2 && sync_lengths[0] == 17 - suffixes[i] && sync_lengths[1] == 17);
        assert(prompt.len == 17 && prompt.v == tokens);
    }
    for (sync_fail = 1; sync_fail <= 2; sync_fail++) {
        sync_calls = 0;
        assert(sync_prompt(NULL, &prompt, 8, err, sizeof(err)) == 7);
        assert(sync_calls == sync_fail && prompt.len == 17);
    }
    sync_fail = sync_calls = 0;
    assert(sync_prompt(NULL, &prompt, 17, err, sizeof(err)) != 0);
    assert(sync_prompt(NULL, &prompt, 18, err, sizeof(err)) != 0);
    assert(sync_prompt(NULL, &prompt, -1, err, sizeof(err)) != 0);
    prompt.len = 0;
    assert(sync_prompt(NULL, &prompt, 1, err, sizeof(err)) != 0);
    assert(sync_calls == 0);
}

int main(void) {
    check_continued_prefill();
    const char *invalid[] = {
        "{}",
        "{\"choices\":[]}",
        ("{\"choices\":[{\"logprobs\":null,\"message\":{\"content\":\"[text]\","
         "\"reasoning_details\":[{\"text\":\"reasoning\"}]}}]}"),
        "{\"choices\":[{\"logprobs\":{\"content\":[]}}]}",
        "{\"choices\":[{\"logprobs\":{\"content\":[{}]}}]}",
        "{\"choices\":[{\"message\":{\"logprobs\":{\"content\":[{\"logprob\":-1}]}}}]}",
        "{\"choices\":[{\"logprobs\":null},{\"logprobs\":{\"content\":[{\"logprob\":-1}]}}]}",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        api_ref ref;
        assert(!api_ref_parse(invalid[i], &ref));
        assert(ref.n_pos == 0);
        api_ref_free(&ref);
    }
    api_ref ref;
    assert(api_ref_parse(
        "{\"ignored\":{\"logprobs\":null},\"choices\":[{\"message\":{\"content\":\"[x]\"},"
        "\"logprobs\":{\"content\":[{\"logprob\":-0.5,\"top_logprobs\":["
        "{\"bytes\":[65],\"logprob\":-0.25}]},{\"logprob\":-1.5}]}}]}", &ref));
    assert(ref.n_pos == 2 && ref.pos[0].logprob == -0.5);
    assert(ref.pos[1].logprob == -1.5);
    assert(ref.pos[0].n_alts == 1);
    assert(ref.pos[0].alts[0].len == 1 && ref.pos[0].alts[0].bytes[0] == 'A');
    assert(ref.pos[0].alts[0].logprob == -0.25);
    api_ref_free(&ref);
    puts("quality API parser and continued-prefill scheduling: PASS");
    return 0;
}
