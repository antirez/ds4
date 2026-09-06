#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int calls;
    int current;
    int total;
} progress_counter;

static void count_progress(void *ud, const char *event, int current, int total) {
    (void)event;
    progress_counter *counter = ud;
    counter->calls++;
    counter->current = current;
    counter->total = total;
}

static float max_logit_delta(const float *a, const float *b, int count) {
    float max_delta = 0.0f;
    for (int i = 0; i < count; i++) {
        float delta = fabsf(a[i] - b[i]);
        if (delta > max_delta) max_delta = delta;
    }
    return max_delta;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MAIN.gguf VISION.gguf IMAGE\n", argv[0]);
        return 2;
    }
    ds4_engine_options options = {0};
    options.model_path = argv[1];
    options.vision_path = argv[2];
#ifdef __APPLE__
    options.backend = DS4_BACKEND_METAL;
#else
    options.backend = DS4_BACKEND_CUDA;
#endif
    options.context_size = 4096;
    options.quality = true;
#ifdef DS4_ROCM_BUILD
    options.ssd_streaming = true;
    options.ssd_streaming_cache_bytes = UINT64_C(32) << 30;
#endif

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    char error[256] = {0};
    ds4_vision_embedding embedding = {0};
    ds4_vision_span span = {0};
    ds4_tokens prompt = {0};
    ds4_session *session = NULL;
    float *image_logits = NULL;
    float *replayed_image_logits = NULL;
    float *zero_image_logits = NULL;
    float *restored_image_logits = NULL;
    float *image_embedding_data = NULL;
    int rc = 1;

    if (!ds4_engine_vision_encode_file(engine, argv[3], &embedding,
                                       error, sizeof(error))) goto done;
    ds4_chat_begin(engine, &prompt);
    ds4_tokens_push(&prompt, ds4_token_user(engine));
    if (!ds4_prompt_append_vision(engine, &prompt, &span, &embedding,
                                  error, sizeof(error))) goto done;
    ds4_tokenize_text(engine,
                      "\nDescribe the image briefly and state its dominant colors.",
                      &prompt);
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    if (ds4_session_create(&session, engine, 4096) != 0) {
        snprintf(error, sizeof(error), "session creation failed");
        goto done;
    }
    if (ds4_session_sync_multimodal(session, &prompt, &span, 1,
                                    error, sizeof(error)) != 0) goto done;
    if (!ds4_session_has_vision_state(session) ||
        !ds4_session_vision_state_matches(session, &span, 1) ||
        ds4_session_vision_state_matches(session, NULL, 0)) {
        snprintf(error, sizeof(error),
                 "live session did not retain exact image identity");
        goto done;
    }

    progress_counter progress = {0};
    ds4_session_set_progress(session, count_progress, &progress);
    if (ds4_session_sync_multimodal(session, &prompt, &span, 1,
                                    error, sizeof(error)) != 0) goto done;
    if (progress.calls != 0) {
        snprintf(error, sizeof(error),
                 "unchanged image unexpectedly repeated prefill");
        goto done;
    }
    const int n_vocab = ds4_engine_vocab_size(engine);
    const size_t image_embedding_elems =
        (size_t)span.embedding.token_count *
        (size_t)ds4_engine_embd_dim(engine);
    image_logits = malloc((size_t)n_vocab * sizeof(*image_logits));
    replayed_image_logits = malloc((size_t)n_vocab *
                                   sizeof(*replayed_image_logits));
    zero_image_logits = malloc((size_t)n_vocab * sizeof(*zero_image_logits));
    restored_image_logits = malloc((size_t)n_vocab *
                                   sizeof(*restored_image_logits));
    image_embedding_data = malloc(image_embedding_elems *
                                  sizeof(*image_embedding_data));
    if (!image_logits || !replayed_image_logits || !zero_image_logits ||
        !restored_image_logits || !image_embedding_data ||
        ds4_session_copy_logits(session, image_logits, n_vocab) != n_vocab) {
        snprintf(error, sizeof(error), "unable to save image-conditioned logits");
        goto done;
    }

    ds4_session_invalidate(session);
    if (ds4_session_sync_multimodal(session, &prompt, &span, 1,
                                    error, sizeof(error)) != 0 ||
        ds4_session_copy_logits(session, replayed_image_logits, n_vocab) !=
            n_vocab) {
        snprintf(error, sizeof(error),
                 "unable to replay image-conditioned prompt");
        goto done;
    }
    const float replayed_max_delta =
        max_logit_delta(image_logits, replayed_image_logits, n_vocab);
    if (replayed_max_delta > 1.0e-6f) {
        snprintf(error, sizeof(error),
                 "replayed image-conditioned logits changed (max %.9g)",
                 replayed_max_delta);
        goto done;
    }

    memcpy(image_embedding_data, span.embedding.data,
           image_embedding_elems * sizeof(*image_embedding_data));
    memset(span.embedding.data, 0,
           image_embedding_elems * sizeof(*span.embedding.data));
    span.embedding.fingerprint[0] ^= 1u;
    if (ds4_session_vision_state_matches(session, &span, 1)) {
        snprintf(error, sizeof(error),
                 "changed image fingerprint matched live session");
        goto done;
    }
    if (ds4_session_sync_multimodal(session, &prompt, &span, 1,
                                    error, sizeof(error)) != 0) goto done;
    if (!ds4_session_vision_state_matches(session, &span, 1)) {
        snprintf(error, sizeof(error),
                 "rebuilt session did not retain changed image identity");
        goto done;
    }
    /* GLM reports session prefill progress here. DeepSeek's one-shot GPU
     * prefill does not, so its rebuild is proven by the identity and logit
     * checks below instead. */
    if (progress.calls != 0 &&
        (progress.current != prompt.len || progress.total != prompt.len)) {
        snprintf(error, sizeof(error),
                 "changed image fingerprint did not rebuild prompt state");
        goto done;
    }
    if (ds4_session_copy_logits(session, zero_image_logits, n_vocab) != n_vocab) {
        snprintf(error, sizeof(error), "unable to save zero-image logits");
        goto done;
    }
    const float max_delta =
        max_logit_delta(image_logits, zero_image_logits, n_vocab);
    if (!(max_delta > 1.0e-4f)) {
        snprintf(error, sizeof(error),
                 "visual embedding did not affect output logits");
        goto done;
    }
    memcpy(span.embedding.data, image_embedding_data,
           image_embedding_elems * sizeof(*image_embedding_data));
    span.embedding.fingerprint[0] ^= 1u;
    if (ds4_session_sync_multimodal(session, &prompt, &span, 1,
                                    error, sizeof(error)) != 0) goto done;
    if (ds4_session_copy_logits(session, restored_image_logits, n_vocab) !=
        n_vocab) {
        snprintf(error, sizeof(error),
                 "unable to save restored image-conditioned logits");
        goto done;
    }
    const float restored_max_delta =
        max_logit_delta(image_logits, restored_image_logits, n_vocab);
    if (restored_max_delta > 1.0e-6f) {
        snprintf(error, sizeof(error),
                 "restored image-conditioned logits changed (max %.9g)",
                 restored_max_delta);
        goto done;
    }
    ds4_session_set_progress(session, NULL, NULL);
    for (int i = 0; i < 48; i++) {
        int token = ds4_session_argmax(session);
        if (token < 0) {
            snprintf(error, sizeof(error), "argmax failed");
            goto done;
        }
        if (ds4_token_is_stop(engine, token)) break;
        size_t len = 0;
        char *text = ds4_token_text(engine, token, &len);
        if (text && len) fwrite(text, 1, len, stdout);
        if (ds4_session_eval(session, token, error, sizeof(error)) != 0) goto done;
    }
    fputc('\n', stdout);
    rc = 0;

done:
    if (rc != 0) fprintf(stderr, "vision prompt failed: %s\n", error);
    free(image_embedding_data);
    free(restored_image_logits);
    free(zero_image_logits);
    free(replayed_image_logits);
    free(image_logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_vision_embedding_free(&span.embedding);
    ds4_vision_embedding_free(&embedding);
    ds4_engine_close(engine);
    return rc;
}
