#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_tp.h"

#include <float.h>
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
        if (!isfinite(a[i]) || !isfinite(b[i])) return INFINITY;
        float delta = fabsf(a[i] - b[i]);
        if (!isfinite(delta)) return INFINITY;
        if (delta > max_delta) max_delta = delta;
    }
    return max_delta;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--logit-oracle")) {
        const float a[] = {1.0f, 2.0f, 3.0f};
        float b[] = {1.0f, 2.25f, 3.0f};
        if (max_logit_delta(a, a, 3) != 0.0f ||
            max_logit_delta(a, b, 3) != 0.25f) return 1;
        const float invalid[] = {NAN, INFINITY, -INFINITY};
        for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
            b[1] = invalid[i];
            if (isfinite(max_logit_delta(a, b, 3)) ||
                isfinite(max_logit_delta(b, a, 3)) ||
                isfinite(max_logit_delta(b, b, 3))) return 1;
        }
        const float high[] = {FLT_MAX}, low[] = {-FLT_MAX};
        if (isfinite(max_logit_delta(high, low, 1))) return 1;
        puts("Vision logit comparison rejects non-finite values: PASS");
        return 0;
    }
    if (argc < 4) {
        fprintf(stderr, "usage: %s MAIN.gguf VISION.gguf IMAGE [--ssd-streaming] [--quality] [--tokens N] [TP options]\n", argv[0]);
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
#ifdef DS4_ROCM_BUILD
    options.ssd_streaming = true;
    options.ssd_streaming_cache_bytes = UINT64_C(32) << 30;
#endif

    ds4_dist_options dist = {0};
    int generate = 48;
    char parse_error[256] = {0};
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--quality")) {
            options.quality = true;
            continue;
        }
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            char *end = NULL;
            const long value = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end || value < 1 || value > 512) {
                fprintf(stderr, "--tokens must be between 1 and 512\n");
                return 2;
            }
            generate = (int)value;
            continue;
        }
        if (!strcmp(argv[i], "--ssd-streaming")) {
            options.ssd_streaming = true;
            continue;
        }
        const ds4_dist_cli_parse_result d = ds4_dist_parse_cli_arg(
            argv[i], &i, argc, argv, &dist, parse_error, sizeof(parse_error));
        if (d == DS4_DIST_CLI_MATCHED) continue;
        if (d != DS4_DIST_CLI_ERROR) {
            const ds4_tp_cli_parse_result t = ds4_tp_parse_cli_arg(
                argv[i], &i, argc, argv, &options.tp, parse_error, sizeof(parse_error));
            if (t == DS4_TP_CLI_MATCHED) continue;
        }
        fprintf(stderr, "invalid test option %s: %s\n", argv[i], parse_error);
        return 2;
    }
    if (!ds4_tp_adopt_distributed_options(&options.tp, &dist, parse_error, sizeof(parse_error)) ||
        !ds4_tp_validate_engine_options(&options, parse_error, sizeof(parse_error)) ||
        dist.role != DS4_DISTRIBUTED_NONE) {
        fprintf(stderr, "invalid vision test configuration: %s\n", parse_error);
        return 2;
    }

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    if (options.tp.role == DS4_TP_WORKER) {
        const int status = ds4_tp_worker_run(engine, &options.tp);
        ds4_engine_close(engine);
        return status;
    }
    ds4_tp *tp = NULL;
    if (options.tp.role == DS4_TP_LEADER) {
        ds4_tp_identity identity = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = options.context_size,
        };
        ds4_engine_tp_gate_schedule(engine, &identity.gate_slot_start,
                                    &identity.gate_slot_step, &identity.gates_per_token,
                                    identity.gate_slot_mask);
        if (!ds4_tp_create(&tp, &options.tp, &identity, parse_error, sizeof(parse_error)) ||
            !ds4_engine_tp_bind(engine, tp, parse_error, sizeof(parse_error))) {
            fprintf(stderr, "vision test TP setup failed: %s\n", parse_error);
            ds4_engine_close(engine);
            ds4_tp_free(tp);
            return 1;
        }
    }
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
    if (!isfinite(max_delta) || !(max_delta > 1.0e-4f)) {
        snprintf(error, sizeof(error),
                 "visual embedding produced invalid or unchanged output logits");
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
    for (int i = 0; i < generate; i++) {
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
    if (tp) (void)ds4_tp_send_stop(tp);
    ds4_engine_close(engine);
    ds4_tp_free(tp);
    return rc;
}
