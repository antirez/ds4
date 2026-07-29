/* Model-backed provider snapshot round-trip.
 *
 * This is intentionally not part of `make test`: it loads a full supported
 * model. Run with:
 *
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-session-snapshot
 */

#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 512

static void fail(const char *what, const char *detail) {
    fprintf(stderr, "FAIL: %s%s%s\n",
            what, detail && detail[0] ? ": " : "", detail ? detail : "");
    exit(1);
}

static void compare_logits(ds4_session *a, ds4_session *b,
                           float *a_logits, float *b_logits, int vocab,
                           const char *stage) {
    if (ds4_session_copy_logits(a, a_logits, vocab) != vocab ||
        ds4_session_copy_logits(b, b_logits, vocab) != vocab) {
        fail("copy logits", stage);
    }
    if (memcmp(a_logits, b_logits, (size_t)vocab * sizeof(float)) != 0) {
        fail("logits changed across snapshot round-trip", stage);
    }
    if (ds4_session_argmax(a) != ds4_session_argmax(b)) {
        fail("argmax changed across snapshot round-trip", stage);
    }
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) fail("DS4_TEST_MODEL is not set", NULL);

    ds4_engine_options opt = {
        .model_path = model,
#if defined(__APPLE__)
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .n_threads = 1,
        .context_size = TEST_CTX,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) fail("engine open", NULL);

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, "Reply with exactly: OK",
                           DS4_THINK_NONE, &prompt);

    ds4_session *source = NULL;
    ds4_session *restored = NULL;
    char err[256] = {0};
    if (ds4_session_create(&source, engine, TEST_CTX) != 0 ||
        ds4_session_create(&restored, engine, TEST_CTX) != 0) {
        fail("session create", NULL);
    }
    if (ds4_session_sync(source, &prompt, err, sizeof(err)) != 0) {
        fail("source prefill", err);
    }

    ds4_session_snapshot snapshot = {0};
    if (ds4_session_save_snapshot(source, &snapshot, err, sizeof(err)) != 0) {
        fail("snapshot save", err);
    }
    if (ds4_session_load_snapshot(restored, &snapshot, err, sizeof(err)) != 0) {
        fail("snapshot load", err);
    }
    if (ds4_session_pos(source) != ds4_session_pos(restored)) {
        fail("checkpoint position changed across snapshot round-trip", NULL);
    }

    const int vocab = ds4_engine_vocab_size(engine);
    float *source_logits = malloc((size_t)vocab * sizeof(float));
    float *restored_logits = malloc((size_t)vocab * sizeof(float));
    if (!source_logits || !restored_logits) fail("logit allocation", NULL);
    compare_logits(source, restored, source_logits, restored_logits,
                   vocab, "prefill");

    const int token = ds4_session_argmax(source);
    if (ds4_session_eval(source, token, err, sizeof(err)) != 0) {
        fail("source decode", err);
    }
    if (ds4_session_eval(restored, token, err, sizeof(err)) != 0) {
        fail("restored decode", err);
    }
    compare_logits(source, restored, source_logits, restored_logits,
                   vocab, "decode");

    fprintf(stderr,
            "test_session_snapshot: OK tokens=%d payload=%llu bytes\n",
            ds4_session_pos(source),
            (unsigned long long)snapshot.len);

    free(source_logits);
    free(restored_logits);
    ds4_session_snapshot_free(&snapshot);
    ds4_session_free(source);
    ds4_session_free(restored);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}
