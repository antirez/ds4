/* Live qwen35moe session transitions against a separately rebuilt session. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cancel_budget { int remaining; };
struct generation_capture { int count, token; bool done; };
static void capture_token(void *ud, int token) {
    struct generation_capture *capture = ud;
    capture->count++;
    capture->token = token;
}
static void capture_done(void *ud) {
    ((struct generation_capture *)ud)->done = true;
}
static bool cancel_after(void *ud) {
    struct cancel_budget *budget = ud;
    return --budget->remaining <= 0;
}
static void require(bool condition, const char *what, const char *err) {
    if (!condition) {
        fprintf(stderr, "%s: %s\n", what, err ? err : "");
        exit(1);
    }
}
static void same_logits(ds4_session *a, ds4_session *b, int vocab, const char *stage) {
    float *x = malloc((size_t)vocab * sizeof(float));
    float *y = malloc((size_t)vocab * sizeof(float));
    require(x && y, "logit buffers", NULL);
    require(ds4_session_copy_logits(a, x, vocab) == vocab &&
            ds4_session_copy_logits(b, y, vocab) == vocab &&
            memcmp(x, y, (size_t)vocab * sizeof(float)) == 0,
            "session logits differ", stage);
    free(x);
    free(y);
}
static void sync_to(ds4_session *s, int *tokens, int len, char *err) {
    ds4_tokens prompt = {tokens, len, len};
    require(ds4_session_sync(s, &prompt, err, 256) == 0, "sync", err);
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int tokens[32], changed[24];
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_CPU, .n_threads = 6};
    ds4_engine *engine = NULL;
    ds4_session *a = NULL, *b = NULL, *fresh = NULL;
    char err[256] = "";
    require(ds4_engine_open(&engine, &opt) == 0, "open", NULL);
    ds4_tokens fixture = {0};
    ds4_tokenize_rendered_chat(engine,
        "<|im_start|>user\nCompare tokenizer output, attention caches, recurrent state, "
        "tool turns, and saved sessions across several precise continuations. "
        "Compare tokenizer output, attention caches, recurrent state, "
        "tool turns, and saved sessions across several precise continuations."
        "<|im_end|>\n", &fixture);
    require(fixture.len >= 32, "fixture tokenization", NULL);
    memcpy(tokens, fixture.v, sizeof(tokens));
    ds4_tokens_free(&fixture);
    require(ds4_session_create(&a, engine, 64) == 0 &&
            ds4_session_create(&b, engine, 64) == 0 &&
            ds4_session_create(&fresh, engine, 96) == 0, "create sessions", NULL);
    int vocab = ds4_engine_vocab_size(engine);

    sync_to(a, tokens, 16, err);
    FILE *payload = tmpfile();
    uint64_t bytes = ds4_session_payload_bytes(a);
    require(payload && bytes > 0 && ds4_session_save_payload(a, payload, err, sizeof(err)) == 0 &&
            fflush(payload) == 0 && fseek(payload, 0, SEEK_SET) == 0 &&
            ds4_session_load_payload(b, payload, bytes, err, sizeof(err)) == 0,
            "save/restore", err);
    same_logits(a, b, vocab, "after restore");
    require(ds4_session_eval(a, tokens[16], err, sizeof(err)) == 0 &&
            ds4_session_eval(b, tokens[16], err, sizeof(err)) == 0,
            "restored continuation", err);
    same_logits(a, b, vocab, "after restored decode");
    fclose(payload);

    sync_to(a, tokens, 24, err);
    sync_to(b, tokens, 24, err);
    sync_to(fresh, tokens, 24, err);
    same_logits(a, b, vocab, "after prefix extension");
    memcpy(changed, tokens, sizeof(changed));
    changed[8] = tokens[24];
    sync_to(a, changed, 24, err);
    ds4_session_invalidate(fresh);
    sync_to(fresh, changed, 24, err);
    same_logits(a, fresh, vocab, "after changed history");

    ds4_session_rewind(a, 12);
    require(ds4_session_eval(a, changed[12], err, sizeof(err)) == 0,
            "rewind/eval", err);
    ds4_session_invalidate(fresh);
    sync_to(fresh, changed, 12, err);
    require(ds4_session_eval(fresh, changed[12], err, sizeof(err)) == 0,
            "matching rewind continuation", err);
    same_logits(a, fresh, vocab, "after rewind");

    struct cancel_budget budget = {5};
    ds4_session_set_cancel(b, cancel_after, &budget);
    ds4_tokens prompt = {tokens, 16, 16};
    require(ds4_session_sync(b, &prompt, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED,
            "prefill cancellation", err);
    ds4_session_set_cancel(b, NULL, NULL);
    sync_to(b, tokens, 16, err);
    ds4_session_invalidate(fresh);
    sync_to(fresh, tokens, 16, err);
    same_logits(b, fresh, vocab, "after prefill cancellation");

    budget.remaining = 5;
    ds4_session_set_cancel(b, cancel_after, &budget);
    require(ds4_session_eval(b, tokens[16], err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED,
            "decode cancellation", err);
    ds4_session_set_cancel(b, NULL, NULL);
    sync_to(b, tokens, 17, err);
    ds4_session_invalidate(fresh);
    sync_to(fresh, tokens, 17, err);
    same_logits(b, fresh, vocab, "after decode cancellation");

    /* Exercise the public direct generator at its last context slot. Compare
     * with an actual session's next non-stop token so this checks the emission
     * boundary rather than accidentally passing on an immediate stop. */
    ds4_session_invalidate(fresh);
    int oracle_len, expected_next = -1;
    for (oracle_len = 16; oracle_len <= 32; oracle_len++) {
        sync_to(fresh, tokens, oracle_len, err);
        expected_next = ds4_session_argmax(fresh);
        if (expected_next >= 0 && !ds4_token_is_stop(engine, expected_next)) break;
    }
    require(oracle_len <= 32, "direct generation fixture needs a non-stop continuation", err);
    ds4_tokens short_prompt = {tokens, oracle_len, 32};
    struct generation_capture capture = {0};
    require(ds4_engine_generate_argmax(engine, &short_prompt, 4, oracle_len + 1,
                                      capture_token, capture_done, &capture, NULL, NULL) == 0,
            "direct generation at context boundary", err);
    require(capture.done && capture.count == 1,
            "direct generation exceeded remaining context", NULL);
    if (capture.count)
        require(capture.token == expected_next, "direct generation token differs", NULL);

    ds4_session_free(fresh);
    ds4_session_free(b);
    ds4_session_free(a);
    ds4_engine_close(engine);
    puts("qwen35moe restore, prefix replay, rewind, cancellation recovery: PASS");
    return 0;
}
