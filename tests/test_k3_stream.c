#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Run separate processes with the residency diagnostic disabled/enabled and
 * compare the binary traces. Each records every logit, not just greedy IDs. */
static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "test_k3_stream: %s\n", message);
        exit(1);
    }
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void record(FILE *fp, ds4_session *s, float *logits, int vocab) {
    require(ds4_session_copy_logits(s, logits, vocab) == vocab, "copy logits");
    for (int i = 0; i < vocab; i++)
        require(isfinite(logits[i]), "non-finite logit");
    require(fwrite(logits, sizeof(*logits), (size_t)vocab, fp) ==
            (size_t)vocab, "write logits");
}

int main(int argc, char **argv) {
    require(argc == 3, "usage: test_k3_stream MODEL.gguf TRACE.bin");
    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = 1024, .prefill_chunk = 64, .power_percent = 100,
        .ssd_streaming = true,
    };
    ds4_engine *engine = NULL;
    require(ds4_engine_open(&engine, &opt) == 0, "open engine");
    require(ds4_engine_is_k3(engine), "not a K3 model");
    ds4_session *s = NULL;
    require(ds4_session_create(&s, engine, 1024) == 0, "create session");
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    FILE *fp = fopen(argv[2], "wb");
    require(fp && logits, "allocate trace");
    char err[512] = {0};
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL,
        "Explain what a database transaction is in one short paragraph.",
        DS4_THINK_NONE, &prompt);
    double start = now();
    require(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0, err);
    fprintf(stderr, "initial prefill: %d tokens %.3f s\n", prompt.len, now() - start);
    record(fp, s, logits, vocab);
    start = now();
    for (int i = 0; i < 8; i++) {
        int token = ds4_session_argmax_ignoring_eos(s, DS4_THINK_NONE);
        require(token >= 0, "no greedy token");
        require(ds4_session_eval(s, token, err, sizeof(err)) == 0, err);
        ds4_tokens_push(&prompt, token);
        record(fp, s, logits, vocab);
    }
    fprintf(stderr, "decode: 8 tokens %.3f s\n", now() - start);

    ds4_chat_append_assistant_close(engine, &prompt);
    ds4_tokens_push(&prompt, ds4_token_assistant_turn_end(engine));
    ds4_chat_append_message(engine, &prompt, "user",
        "Now explain why rolling back a failed transaction matters.");
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    int before = ds4_session_pos(s);
    start = now();
    require(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0, err);
    require(ds4_session_pos(s) == prompt.len, "continued position");
    fprintf(stderr, "continued prefill: %d tokens %.3f s\n",
            prompt.len - before, now() - start);
    record(fp, s, logits, vocab);
    uint64_t bytes = ds4_session_payload_bytes(s);
    FILE *saved = tmpfile();
    require(saved && bytes, "create snapshot");
    require(ds4_session_save_payload(s, saved, err, sizeof(err)) == 0, err);
    require(fflush(saved) == 0 && fseek(saved, 0, SEEK_SET) == 0, "rewind snapshot");
    ds4_session_invalidate(s);
    require(ds4_session_load_payload(s, saved, bytes, err, sizeof(err)) == 0, err);
    require(ds4_session_common_prefix(s, &prompt) == prompt.len, "restore prefix");
    record(fp, s, logits, vocab);
    require(fclose(saved) == 0 && fclose(fp) == 0, "close trace");
    free(logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    ds4_engine_close(engine);
    puts("test_k3_stream: ok");
    return 0;
}
