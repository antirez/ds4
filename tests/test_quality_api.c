#define main quality_scorer_main
#include "../gguf-tools/quality-testing/score_official.c"
#undef main
#include <assert.h>

int main(void) {
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
    puts("quality API parser: PASS");
    return 0;
}
