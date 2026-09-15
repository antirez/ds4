#include "../ds4.c"
#include <assert.h>

/* A failed disk lookup must invalidate the recurrent frontier, including
 * speculative snapshots. Retrying must rebuild from the retained tokens. */
int main(int argc, char **argv) {
    const bool streaming = argc == 3 && strcmp(argv[2], "--ssd-streaming") == 0;
    if (argc != 2 && !streaming) {
        fprintf(stderr, "usage: %s QWEN_GGUF [--ssd-streaming]\n", argv[0]);
        return 2;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .glm_mtp = true, .prefill_chunk = 8};
    if (streaming) {
        /* Account for the actual context before opening the model and leave
         * room for the live/control sessions and their speculative snapshots. */
        opt.context_size = 256;
        opt.ssd_streaming = true;
        opt.ssd_streaming_cache_experts = 1024;
    }
    assert(ds4_engine_open(&engine, &opt) == 0);
    assert(ds4_engine_is_qwen4(engine) && engine->model.ngram_tensor);
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, "Count from one to ten.", DS4_THINK_NONE, &prompt);
    char error[256] = {0};
    for (int mode = 0; mode < 3; mode++) {
        ds4_session *live = NULL, *control = NULL;
        ds4_tokens frontier = {0};
        ds4_session_snapshot snapshot = {0};
        assert(ds4_session_create(&live, engine, 256) == 0);
        assert(ds4_session_create(&control, engine, 256) == 0);
        assert(ds4_session_sync(live, &prompt, error, sizeof(error)) == 0);
        int accepted[3];
        if (mode == 2) {
            /* Prefix preparation may already seed a proposal. Otherwise run
             * a cycle, checking its committed frontier rather than assuming
             * the first call always consumes exactly one token. */
            for (int attempt = 0; !live->glm_mtp_have && attempt < 3; attempt++) {
                const int before = ds4_session_pos(live);
                const int n = ds4_session_eval_speculative_argmax(live, ds4_session_argmax(live),
                    3, -1, accepted, 3, error, sizeof(error));
                assert(n >= 1 && n <= 3);
                assert(ds4_session_pos(live) == before + n);
            }
            assert(live->glm_mtp_have);
        }
        const ds4_tokens *current = ds4_session_tokens(live);
        for (int i = 0; i < current->len; i++) ds4_tokens_push(&frontier, current->v[i]);
        int token = ds4_session_argmax(live);
        const int fd = engine->model.ngram_fd;
        engine->model.ngram_fd = INT_MAX;
        if (mode == 0) {
            ds4_tokens_push(&frontier, token);
            assert(ds4_session_sync(live, &frontier, error, sizeof(error)) != 0);
        } else if (mode == 1) {
            assert(ds4_session_eval(live, token, error, sizeof(error)) != 0);
        } else {
            assert(ds4_session_eval_speculative_argmax(live, token,
                3, -1, accepted, 3, error, sizeof(error)) < 0);
        }
        engine->model.ngram_fd = fd;
        assert(!live->checkpoint_valid);
        assert(ds4_session_argmax(live) == -1);
        assert(ds4_session_save_snapshot(live, &snapshot, error, sizeof(error)) != 0);
        assert(ds4_session_sync(live, &frontier, error, sizeof(error)) == 0);
        assert(ds4_session_sync(control, &frontier, error, sizeof(error)) == 0);
        assert(!memcmp(live->logits, control->logits, DS4_N_VOCAB * sizeof(float)));
        token = ds4_session_argmax(control);
        assert(ds4_session_eval(live, token, error, sizeof(error)) == 0);
        assert(ds4_session_eval(control, token, error, sizeof(error)) == 0);
        assert(!memcmp(live->logits, control->logits, DS4_N_VOCAB * sizeof(float)));
        ds4_tokens_free(&frontier);
        ds4_session_free(control);
        ds4_session_free(live);
        printf("Qwen n-gram failure/recovery mode %d: exact logits OK\n", mode);
    }
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}
