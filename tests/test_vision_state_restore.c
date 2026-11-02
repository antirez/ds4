/* No model or GPU: restore operates only on an already-loaded token frontier. */
#include <stdlib.h>
static int fail_identity_alloc;
static void *test_calloc(size_t count, size_t bytes) {
    if (fail_identity_alloc) return NULL;
    return calloc(count, bytes);
}
#define calloc test_calloc
#include "../ds4.c"
#undef calloc
#include <assert.h>

static void expect_failure(ds4_session *s, const ds4_tokens *exact,
                           const ds4_vision_span *images, size_t count) {
    ds4_vision_identity *old = s->checkpoint_images;
    const size_t old_count = s->checkpoint_image_count;
    ds4_vision_identity saved[2] = {0};
    assert(old_count <= 2);
    if (old_count) memcpy(saved, old, old_count * sizeof(*old));
    const ds4_vision_span *borrowed = s->sync_images;
    const size_t borrowed_count = s->sync_image_count;
    const bool valid = s->checkpoint_valid;
    const int *tokens = s->checkpoint.v;
    const int token_count = s->checkpoint.len;
    char err[160] = {0};
    assert(ds4_session_restore_vision_state(s, exact, images, count,
                                           err, sizeof(err)) != 0);
    assert(err[0]);
    assert(s->checkpoint_images == old && s->checkpoint_image_count == old_count);
    assert(!old_count || !memcmp(saved, old, old_count * sizeof(*old)));
    assert(s->sync_images == borrowed && s->sync_image_count == borrowed_count);
    assert(s->checkpoint_valid == valid);
    assert(s->checkpoint.v == tokens && s->checkpoint.len == token_count);
}

int main(void) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    ds4_engine engine = {.backend = DS4_BACKEND_CPU, .vision_ready = true,
                         .vision_kind = DS4_VISION_QWEN4, .vision_image_token = 7};
    int tokens[] = {11, 12, 7, 7, 7, 7, 13, 11, 12, 7, 7, 7, 7, 13, 11};
    const int n = sizeof(tokens) / sizeof(*tokens);
    ds4_tokens exact = {.v = tokens, .len = n, .cap = n};
    ds4_session s = {.engine = &engine, .checkpoint_valid = true,
                    .checkpoint = {.v = tokens, .len = n, .cap = n}};
    float rows[4 * 2560] = {0};
    ds4_vision_span images[2] = {
        {.token_start = 2, .embedding = {.data = rows, .token_count = 4,
            .grid_width = 2, .grid_height = 2, .width = 64, .height = 64,
            .content_width = 64, .content_height = 64, .fingerprint = {1}}},
        {.token_start = 9, .embedding = {.data = rows, .token_count = 4,
            .grid_width = 2, .grid_height = 2, .width = 64, .height = 64,
            .content_width = 64, .content_height = 64, .fingerprint = {2}}},
    };
    char err[160] = {0};
    assert(!ds4_session_restore_vision_state(&s, &exact, images, 2, err, sizeof(err)));
    assert(ds4_session_vision_state_matches(&s, images, 2));
    assert(s.checkpoint_images && !s.sync_images && !s.sync_image_count);
    assert(s.checkpoint.v == tokens && s.checkpoint.len == n && s.checkpoint_valid);
    images[0].embedding.fingerprint[0] = 9;
    assert(s.checkpoint_images[0].fingerprint[0] == 1); /* copied, not borrowed */
    assert(!ds4_session_restore_vision_state(&s, &exact, images, 1, err, sizeof(err)));
    assert(s.checkpoint_image_count == 1 && s.checkpoint_images[0].fingerprint[0] == 9);

    int different[sizeof(tokens) / sizeof(*tokens)];
    memcpy(different, tokens, sizeof(tokens)); different[n - 1]++;
    ds4_tokens wrong = {.v = different, .len = n, .cap = n};
    expect_failure(&s, &wrong, images, 1);
    wrong = exact; wrong.len--;
    expect_failure(&s, &wrong, images, 1);
    wrong = exact; wrong.v = NULL;
    expect_failure(&s, &wrong, images, 1);
    wrong = exact; wrong.len = -1;
    expect_failure(&s, &wrong, images, 1);
    s.checkpoint_valid = false;
    expect_failure(&s, &exact, images, 1);
    s.checkpoint_valid = true;
    expect_failure(&s, &exact, NULL, 1);
    expect_failure(&s, &exact, images, SIZE_MAX);

    ds4_vision_span bad[2];
#define REJECT_IMAGE_CHANGE(change) do { \
    memcpy(bad, images, sizeof(bad)); change; \
    expect_failure(&s, &exact, bad, 2); \
} while (0)
    REJECT_IMAGE_CHANGE(bad[0].embedding.data = NULL);
    REJECT_IMAGE_CHANGE(bad[0].embedding.token_count = 0);
    REJECT_IMAGE_CHANGE(bad[1].token_start = 5);
    REJECT_IMAGE_CHANGE(bad[0].token_start = UINT32_MAX);
    REJECT_IMAGE_CHANGE(bad[0].embedding.grid_width = 0);
    REJECT_IMAGE_CHANGE(bad[0].embedding.grid_height = 3);
    REJECT_IMAGE_CHANGE(bad[0].embedding.layout = 1);
    REJECT_IMAGE_CHANGE(bad[0].embedding.content_width = 63);
    REJECT_IMAGE_CHANGE(bad[0].embedding.width = 0);
    REJECT_IMAGE_CHANGE(bad[0].embedding.height = DS4_IMAGE_MAX_DIMENSION + 1);
    REJECT_IMAGE_CHANGE(bad[0].embedding.width = 16384; bad[0].embedding.height = 16384);
#undef REJECT_IMAGE_CHANGE
    const int old_token = tokens[2]; tokens[2] = 42;
    expect_failure(&s, &exact, images, 2);
    tokens[2] = old_token;
    engine.vision_ready = false;
    expect_failure(&s, &exact, images, 2);
    engine.vision_ready = true;
    s.engine = NULL;
    expect_failure(&s, &exact, images, 2);
    s.engine = &engine;

    s.sync_images = images; /* Even a borrowed pointer with zero count is active. */
    expect_failure(&s, &exact, images, 2);
    s.sync_images = NULL; s.sync_image_count = 1;
    expect_failure(&s, &exact, images, 2);
    s.sync_image_count = 0;
#ifndef DS4_NO_GPU
    s.graph.prefill_vision_spans = images;
    expect_failure(&s, &exact, images, 2);
    s.graph.prefill_vision_spans = NULL;
#ifdef DS4_HAS_QWEN4_GPU
    s.qwen4_graph.vis_span_count = 1;
    expect_failure(&s, &exact, images, 2);
    s.qwen4_graph.vis_span_count = 0;
#endif
#endif
    fail_identity_alloc = 1;
    expect_failure(&s, &exact, images, 2);
    ds4_vision_identity *old = s.checkpoint_images;
    assert(!ds4_session_replace_vision_identities(&s, images, SIZE_MAX));
    assert(s.checkpoint_images == old);
    fail_identity_alloc = 0;
    assert(!ds4_session_restore_vision_state(&s, &exact, images, 2, err, sizeof(err)));
    assert(ds4_session_vision_state_matches(&s, images, 2));

    /* Ordinary sync and synthetic test setup share the checked copy helper. */
    s.sync_images = images; s.sync_image_count = 1;
    assert(ds4_session_store_vision_identities(&s));
    assert(s.sync_images == images && s.sync_image_count == 1);
    assert(s.checkpoint_image_count == 1);
    s.sync_images = NULL; s.sync_image_count = 0;
    ds4_session_set_test_images(&s, images, 2);
    assert(ds4_session_vision_state_matches(&s, images, 2));

    /* A special token alone does not prove image input; count zero clears ids. */
    engine.vision_ready = false;
    assert(!ds4_session_restore_vision_state(&s, &exact, NULL, 0, err, sizeof(err)));
    assert(!s.checkpoint_images && !s.checkpoint_image_count);
    assert(s.checkpoint_valid && s.checkpoint.v == tokens && s.checkpoint.len == n);
    g_ds4_shape = saved_shape;
    puts("vision restore state: PASS (no model, no GPU)");
    return 0;
}
