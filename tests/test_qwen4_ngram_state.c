#include "../ds4.c"
#include <assert.h>

static void check_batch(ds4_engine *engine, const ds4_tokens *prompt, bool speculative) {
    enum { N = 4 };
    ds4_session *sessions[N] = {0};
    ds4_decode_item items[N];
    ds4_tokens frontier[N] = {0};
    ds4_session_snapshot snapshot = {0};
    float *saved_logits = xmalloc(DS4_N_VOCAB * sizeof(float));
    int accepted[N][2], counts[N];
    char error[256] = {0};
    for (int i = 0; i < N; i++) {
        assert(ds4_session_create(&sessions[i], engine, 256) == 0);
        assert(ds4_session_sync(sessions[i], prompt, error, sizeof(error)) == 0);
    }
    for (int step = 0; step < 12; step++) {
        if (step == 8) {
            assert(ds4_session_load_snapshot(sessions[0], &snapshot, error, sizeof(error)) == 0);
            assert(!memcmp(sessions[0]->logits, saved_logits, DS4_N_VOCAB * sizeof(float)));
        }
        for (int i = 0; i < N; i++) {
            ds4_session *s = sessions[(i + step) % N];
            items[i] = (ds4_decode_item){s, ds4_session_argmax(s)};
            ds4_tokens_copy(&frontier[i], &s->checkpoint);
        }
        /* Interleave ordinary cycles with the predictor, as exact-sampling
         * requests and output limits do in the server. */
        if (speculative && step % 4 != 3) {
            if (step == 2) {
                /* Counting has not finished: force rejection of an EOS draft. */
                assert(items[0].session->glm_mtp_have);
                items[0].session->glm_mtp_draft = ds4_token_eos(engine);
            }
            assert(ds4_sessions_eval_batch_speculative_argmax(items, N, accepted,
                        counts, error, sizeof(error)) == 0);
            if (step == 2) assert(counts[0] == 1);
            for (int i = 0; i < N; i++) {
                ds4_session *s = items[i].session;
                assert(counts[i] >= 1 && counts[i] <= 2);
                assert(accepted[i][0] == items[i].token);
                assert(s->checkpoint.len == frontier[i].len + counts[i]);
                for (int j = 0; j < counts[i]; j++)
                    assert(s->checkpoint.v[frontier[i].len + j] == accepted[i][j]);
                if (engine->backend == DS4_BACKEND_METAL) {
                    assert(!s->qwen4_graph.snap0_valid && !s->qwen4_graph.snap2_valid);
                    if (counts[i] == 1) assert(!s->qwen4_graph.snap_valid);
                }
            }
        } else {
            assert(ds4_sessions_eval_batch(items, N, error, sizeof(error)) == 0);
            for (int i = 0; i < N; i++) {
                assert(items[i].session->checkpoint.len == frontier[i].len + 1);
                assert(!items[i].session->glm_mtp_have);
            }
        }
        for (int i = 0; i < N; i++) {
            ds4_session *s = items[i].session;
            assert(s->checkpoint_valid && s->qwen4_graph.pos == (uint32_t)s->checkpoint.len);
            for (uint32_t j = 0; j < DS4_N_VOCAB; j++) assert(isfinite(s->logits[j]));
        }
        if (step == 1) {
            assert(ds4_session_save_snapshot(sessions[0], &snapshot, error, sizeof(error)) == 0);
            memcpy(saved_logits, sessions[0]->logits, DS4_N_VOCAB * sizeof(float));
        }
        if (step == 6 && speculative) {
            for (int i = 0; i < N; i++) {
                ds4_session *s = items[i].session;
                if (counts[i] != 2) continue;
                int pos = s->checkpoint.len - 1;
                ds4_session_rewind(s, pos);
                assert(s->checkpoint_valid && s->qwen4_graph.pos == (uint32_t)pos);
                assert(!memcmp(s->logits, s->qwen4_verify_logits, DS4_N_VOCAB * sizeof(float)));
            }
        }
    }
    ds4_session_snapshot_free(&snapshot);
    free(saved_logits);
    for (int i = 0; i < N; i++) {
        items[i] = (ds4_decode_item){sessions[i], ds4_session_argmax(sessions[i])};
        ds4_tokens_copy(&frontier[i], &sessions[i]->checkpoint);
    }
    ds4_decode_item duplicate[2] = {items[0], items[0]};
    assert(ds4_sessions_eval_batch_speculative_argmax(duplicate, 2, accepted,
                counts, error, sizeof(error)) != 0);
    for (int i = 0; i < N; i++) assert(sessions[i]->checkpoint.len == frontier[i].len);

    const int fd = engine->model.ngram_fd;
    engine->model.ngram_fd = INT_MAX;
    int rc = speculative ? ds4_sessions_eval_batch_speculative_argmax(items, N, accepted,
                                counts, error, sizeof(error)) :
                           ds4_sessions_eval_batch(items, N, error, sizeof(error));
    engine->model.ngram_fd = fd;
    assert(rc != 0);
    for (int i = 0; i < N; i++) {
        assert(!sessions[i]->checkpoint_valid);
        assert(ds4_session_argmax(sessions[i]) == -1);
        assert(ds4_session_sync(sessions[i], &frontier[i], error, sizeof(error)) == 0);
        ds4_session *control = NULL;
        assert(ds4_session_create(&control, engine, 256) == 0);
        assert(ds4_session_sync(control, &frontier[i], error, sizeof(error)) == 0);
        assert(!memcmp(sessions[i]->logits, control->logits, DS4_N_VOCAB * sizeof(float)));
        ds4_session_free(control);
        items[i].token = ds4_session_argmax(sessions[i]);
        /* Test the logical limit independently of rounded GPU allocation. */
        sessions[i]->ctx_size = sessions[i]->checkpoint.len + 1;
    }
    assert(ds4_sessions_eval_batch_speculative_argmax(items, N, accepted, counts,
                error, sizeof(error)) == 0);
    for (int i = 0; i < N; i++) {
        assert(counts[i] == 1);
        assert(sessions[i]->checkpoint.len == sessions[i]->ctx_size);
        items[i].token = ds4_session_argmax(sessions[i]);
    }
    assert(ds4_sessions_eval_batch_speculative_argmax(items, N, accepted, counts,
                error, sizeof(error)) != 0);
    for (int i = 0; i < N; i++) {
        assert(sessions[i]->checkpoint_valid);
        ds4_tokens_free(&frontier[i]);
        ds4_session_free(sessions[i]);
    }
    printf("Qwen batch %s: mixed cycles, rejection, disk failure/recovery, context limit OK\n",
           speculative ? "MTP" : "ordinary");
}

static void check_arena_resize(ds4_engine *engine, const ds4_tokens *prompt) {
    ds4_session *small = NULL, *large = NULL;
    char error[256] = {0};
    assert(ds4_session_create(&small, engine, 128) == 0);
    assert(ds4_session_sync(small, prompt, error, sizeof(error)) == 0);
    assert(ds4_session_create(&large, engine, 512) == 0);
    assert(ds4_session_sync(large, prompt, error, sizeof(error)) == 0);
    if (engine->backend == DS4_BACKEND_METAL) {
        assert(!small->qwen4_graph.owns_scratch);
        assert(large->qwen4_graph.owns_scratch);
    }
    ds4_session_free(large);
    ds4_session_free(small);
    assert(ds4_session_create(&large, engine, 1024) == 0);
    assert(ds4_session_sync(large, prompt, error, sizeof(error)) == 0);
    if (engine->backend == DS4_BACKEND_METAL) assert(!large->qwen4_graph.owns_scratch);
    ds4_session_free(large);
    assert(engine->qwen4_arena_users == 0);
    puts("Qwen shared arena: live growth fallback and idle replacement OK");
}

/* A failed disk lookup must invalidate the recurrent frontier, including
 * speculative snapshots. Retrying must rebuild from the retained tokens. */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s QWEN_GGUF\n", argv[0]);
        return 2;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {.model_path = argv[1],
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .glm_mtp = true, .prefill_chunk = 32,
        .share_session_prefill_workspace = true, .placement_session_count_hint = 4};
    assert(ds4_engine_open(&engine, &opt) == 0);
    assert(ds4_engine_is_qwen4(engine) && engine->model.ngram_tensor);
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, "Count from one to ten.", DS4_THINK_NONE, &prompt);
    char error[256] = {0};
    for (int mode = 0; mode < 4; mode++) {
        ds4_session *live = NULL, *control = NULL;
        ds4_tokens frontier = {0};
        ds4_session_snapshot snapshot = {0};
        assert(ds4_session_create(&live, engine, 256) == 0);
        assert(ds4_session_create(&control, engine, 256) == 0);
        assert(ds4_session_sync(live, &prompt, error, sizeof(error)) == 0);
        int accepted[3];
        if (mode == 2) {
            assert(ds4_session_eval_speculative_argmax(live, ds4_session_argmax(live),
                3, -1, accepted, 3, error, sizeof(error)) == 1);
            assert(live->glm_mtp_have);
        }
        const ds4_tokens *current = ds4_session_tokens(live);
        for (int i = 0; i < current->len; i++) ds4_tokens_push(&frontier, current->v[i]);
        int token = ds4_session_argmax(live);
        const int fd = engine->model.ngram_fd;
        engine->model.ngram_fd = INT_MAX;
        if (mode == 3) {
            for (int i = 0; i < 32; i++) ds4_tokens_push(&frontier,token);
            assert(ds4_session_sync(live,&frontier,error,sizeof(error)) != 0);
        } else if (mode == 0) {
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
    check_batch(engine, &prompt, false);
    check_batch(engine, &prompt, true);
    check_arena_resize(engine, &prompt);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}
