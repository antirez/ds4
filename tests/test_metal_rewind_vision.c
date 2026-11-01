/* Model-backed check that ds4_session_rewind is exact for a V4.1 session whose
 * retained prefix contains an image.
 *
 * The text case (test_metal_rewind) proves a rewound session matches one that
 * reached the same position directly. This is the same proof with an image in
 * the kept prefix: the image's raw rows come back from the rewind ring like
 * any other, and the rebuilt Engram history must record image rows as dead
 * instead of hashing their placeholder token ids as text.
 *
 * Layout: [image message][long text tail]. The rewind targets sit past the
 * 128-row raw window so the ring is exercised, and after the image so the
 * image is inside the state being kept: one well into the text tail, one
 * right after the image, where the Engram history tail still covers image
 * rows.
 *
 *   DS4_TEST_MODEL=... DS4_TEST_VISION=... DS4_TEST_IMAGE=... \
 *   DS4_TEST_SSD_STREAMING=1 ./tests/test_metal_rewind_vision
 */

#include "ds4.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static char *read_file(const char *path, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = malloc(cap + 1);
    size_t n = fread(buf, 1, cap, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

static void slice(ds4_tokens *out, const ds4_tokens *src, int to) {
    out->len = 0;
    for (int i = 0; i < to; i++) ds4_tokens_push(out, src->v[i]);
}

static float *logits_of(ds4_session *s, int vocab) {
    float *l = malloc((size_t)vocab * sizeof(float));
    if (!l || ds4_session_copy_logits(s, l, vocab) != vocab) fail("copy logits");
    return l;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void sync_mm(ds4_session *s, const ds4_tokens *t,
                    const ds4_vision_span *span, const char *what) {
    char err[256] = {0};
    if (ds4_session_sync_multimodal(s, t, span, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL: %s: %s\n", what, err);
        exit(1);
    }
}

/* The text test's method, verbatim, with an image in the prefix.
 *
 *   staged:  sync(prefix) then sync(full)
 *   rewound: sync(prefix) then sync(full), rewind to `at`, sync(full) again
 *
 * Both compute the prefix rows with the same sweep and the tail with the same
 * extension, so a correct rewind must match staged exactly. Two comparisons
 * that look reasonable are wrong and were tried first: reading logits straight
 * after an even rewind returns the stale logits of the old frontier; and
 * comparing a direct sweep to `at` against a longer sweep rewound computes the
 * prefix in a different batch shape, which the text test documents as up to
 * ~1 logit of noise on its own. Fails if the rewind dropped state, since a
 * rebuild would still match without exercising the rewind. */
static float compare_at(ds4_engine *e, int ctx, int vocab,
                        const ds4_tokens *full, const ds4_vision_span *span,
                        int at, const char *label) {
    ds4_tokens prefix = {0};
    slice(&prefix, full, at);
    ds4_session *staged = NULL, *rewound = NULL;
    if (ds4_session_create(&staged, e, ctx) ||
        ds4_session_create(&rewound, e, ctx)) fail("session create");

    sync_mm(staged, &prefix, span, "staged prefix");
    sync_mm(staged, full, span, "staged tail");
    float *l_staged = logits_of(staged, vocab);

    sync_mm(rewound, &prefix, span, "rewound prefix");
    sync_mm(rewound, full, span, "rewound full");
    ds4_session_rewind(rewound, at);
    if (ds4_session_pos(rewound) != at) fail("rewind position");
    const int kept = ds4_session_common_prefix(rewound, full);
    if (kept != at) {
        fprintf(stderr, "FAIL: %s: rewind dropped state (kept %d of %d)\n",
                label, kept, at);
        exit(1);
    }
    sync_mm(rewound, full, span, "rewound tail");
    float *l_rewound = logits_of(rewound, vocab);

    const float d = max_abs_diff(l_staged, l_rewound, vocab);
    fprintf(stderr, "%s: rewind to %d with image rows %u..%u in the kept prefix, "
                    "tail replayed to %d: rewound-vs-staged %g\n",
            label, at, span->token_start,
            span->token_start + span->embedding.token_count, full->len, d);
    free(l_staged); free(l_rewound);
    ds4_session_free(staged); ds4_session_free(rewound);
    free(prefix.v);
    return d;
}


static int argmax_of(ds4_session *s, int vocab) {
    float *l = logits_of(s, vocab);
    int best = 0;
    for (int i = 1; i < vocab; i++) if (l[i] > l[best]) best = i;
    free(l);
    return best;
}

static void greedy_n(ds4_session *s, int n, int vocab, int *out) {
    char err[160];
    for (int i = 0; i < n; i++) {
        out[i] = argmax_of(s, vocab);
        if (ds4_session_eval(s, out[i], err, sizeof(err)) != 0) fail("greedy eval");
    }
}

/* The exact check for an odd target. ds4_session_rewind reaches pos = at+1
 * (at even) by rewinding to `at` and replaying token `at` as one decode step.
 * The matching reference therefore computes row `at` the same way:
 *
 *   staged:  sync(prefix[0..at]), eval(full[at]), sync(full)
 *   rewound: sync(prefix[0..at]), sync(full), rewind to at+1, sync(full)
 *
 * Rows 0..at-1 come from the same sweep, row `at` from a decode step whose
 * Engram history, compressor pair state and raw window the rewind rebuilt, and
 * rows at+1.. from the same extension, so a correct odd path is bit-exact. The
 * "odd (info)" delta and the greedy check below compare against routes that
 * compute row `at` inside a batched sweep instead; their distance is the
 * decode-vs-sweep arithmetic (the text test measures ~1.3 logits between a
 * single sweep and prefix-then-tail on its own), not a rewind error. */
static float compare_odd_exact(ds4_engine *e, int ctx, int vocab,
                               const ds4_tokens *full, const ds4_vision_span *span,
                               int at, const char *label) {
    if (at & 1) fail("compare_odd_exact needs an even base");
    ds4_tokens prefix = {0};
    slice(&prefix, full, at);
    ds4_session *staged = NULL, *rewound = NULL;
    if (ds4_session_create(&staged, e, ctx) ||
        ds4_session_create(&rewound, e, ctx)) fail("odd exact sessions");
    char err[160];

    sync_mm(staged, &prefix, span, "odd staged prefix");
    if (ds4_session_eval(staged, full->v[at], err, sizeof(err)) != 0)
        fail("odd staged decode");
    sync_mm(staged, full, span, "odd staged tail");
    float *l_staged = logits_of(staged, vocab);

    sync_mm(rewound, &prefix, span, "odd rewound prefix");
    sync_mm(rewound, full, span, "odd rewound full");
    ds4_session_rewind(rewound, at + 1);
    if (ds4_session_pos(rewound) != at + 1) fail("odd exact rewind position");
    if (ds4_session_common_prefix(rewound, full) != at + 1) {
        fprintf(stderr, "FAIL: %s: odd rewind dropped state\n", label);
        exit(1);
    }
    sync_mm(rewound, full, span, "odd rewound tail");
    float *l_rewound = logits_of(rewound, vocab);

    const float d = max_abs_diff(l_staged, l_rewound, vocab);
    fprintf(stderr, "%s: rewind to %d (replay row %d as decode), tail to %d: "
                    "rewound-vs-staged-decode %g\n", label, at + 1, at, full->len, d);
    free(l_staged); free(l_rewound);
    ds4_session_free(staged); ds4_session_free(rewound);
    free(prefix.v);
    return d;
}

/* Informational: greedy tokens after an odd rewind against a single sweep
 * over the whole prompt. That reference computes row at and the tail in
 * different batch shapes, so a token can flip at a near-tie without any
 * error in the rewind (it did at 1201 on the upstream image run: 588 2413 201
 * vs 588 2413 9365). Not a pass criterion. */
static int odd_greedy_image(ds4_engine *e, int ctx, int vocab,
                            const ds4_tokens *full, const ds4_vision_span *span,
                            int odd) {
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx)) fail("odd session");
    int g_c[4], g_d[4];
    sync_mm(s, full, span, "odd setup");
    greedy_n(s, 4, vocab, g_c);
    ds4_session_rewind(s, odd);
    if (ds4_session_pos(s) != odd) fail("odd rewind position");
    if (ds4_session_common_prefix(s, full) != odd) fail("odd rewind kept no state");
    sync_mm(s, full, span, "odd tail");
    greedy_n(s, 4, vocab, g_d);
    fprintf(stderr, "odd image rewind to %d: greedy %d %d %d %d vs %d %d %d %d\n", odd,
            g_c[0], g_c[1], g_c[2], g_c[3], g_d[0], g_d[1], g_d[2], g_d[3]);
    ds4_session_free(s);
    return memcmp(g_c, g_d, sizeof(g_c)) == 0;
}

/* Control: the same odd logit comparison on a text-only prompt of the same
 * length. If text shows a delta of the same size, the image-prefix odd delta
 * is the one-token replay's batch shape, not the image. */
static float odd_delta_text(ds4_engine *e, int ctx, int vocab,
                            const ds4_tokens *full, int odd) {
    ds4_tokens prefix = {0};
    slice(&prefix, full, odd);
    ds4_session *staged = NULL, *rewound = NULL;
    if (ds4_session_create(&staged, e, ctx) ||
        ds4_session_create(&rewound, e, ctx)) fail("text control sessions");
    char err[256];
    if (ds4_session_sync(staged, &prefix, err, sizeof(err)) ||
        ds4_session_sync(staged, full, err, sizeof(err))) fail("text staged");
    float *a = logits_of(staged, vocab);
    if (ds4_session_sync(rewound, &prefix, err, sizeof(err)) ||
        ds4_session_sync(rewound, full, err, sizeof(err))) fail("text rewound");
    ds4_session_rewind(rewound, odd);
    if (ds4_session_sync(rewound, full, err, sizeof(err))) fail("text rewound tail");
    float *b = logits_of(rewound, vocab);
    const float d = max_abs_diff(a, b, vocab);
    free(a); free(b);
    ds4_session_free(staged); ds4_session_free(rewound);
    free(prefix.v);
    return d;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    const char *vision = getenv("DS4_TEST_VISION");
    const char *image = getenv("DS4_TEST_IMAGE");
    if (!model || !vision || !image) {
        fprintf(stderr, "SKIP: set DS4_TEST_MODEL, DS4_TEST_VISION, DS4_TEST_IMAGE\n");
        return 0;
    }
    char *text = read_file("README.md", 12000);
    if (!text) fail("README.md");

    const int ctx = 4096;
    ds4_engine_options opt = {
        .model_path = model,
        .vision_path = vision,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = ctx,
        .ssd_streaming = getenv("DS4_TEST_SSD_STREAMING") != NULL,
        .ssd_streaming_cache_bytes = getenv("DS4_TEST_SSD_CACHE_GB") ?
            (uint64_t)strtoull(getenv("DS4_TEST_SSD_CACHE_GB"), NULL, 10) << 30 : 0,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    const int vocab = ds4_engine_vocab_size(e);

    char err[256] = {0};
    ds4_vision_embedding emb = {0};
    if (!ds4_engine_vision_encode_file(e, image, &emb, err, sizeof(err))) {
        fprintf(stderr, "FAIL: vision encode: %s\n", err);
        return 1;
    }

    ds4_tokens full = {0};
    ds4_vision_span span = {0};
    const char *parts[2] = { "Look at this image.", " Now read the text that follows." };
    if (!ds4_chat_append_multimodal_message(e, &full, "user", parts, &emb, 1,
                                           &span, err, sizeof(err))) {
        fprintf(stderr, "FAIL: multimodal message: %s\n", err);
        return 1;
    }
    const int img_end = (int)(span.token_start + span.embedding.token_count);

    ds4_tokens tail = {0};
    ds4_tokenize_text(e, text, &tail);
    for (int i = 0; i < tail.len && full.len < 1500; i++) ds4_tokens_push(&full, tail.v[i]);

    int at = full.len - 300;          /* past the 128-row raw window */
    at &= ~1;                         /* even target first */
    if (at <= img_end + 8) fail("prompt too short to rewind behind the image");
    fprintf(stderr, "prompt %d tokens, image at %u..%d\n", full.len, span.token_start, img_end);

    const float d_even = compare_at(e, ctx, vocab, &full, &span, at, "even");
    /* Even target one or two rows past the image: of the three history
     * positions before it, at least one is an image row. */
    const int after = (img_end + 1) & ~1;
    if (after < 128) fail("image ends inside the raw window; need a longer image");
    const float d_after = compare_at(e, ctx, vocab, &full, &span, after,
                                     "even, right after the image");
    const float d_odd_exact = compare_odd_exact(e, ctx, vocab, &full, &span, at, "odd");
    /* Odd target right after the image: the replayed decode row's Engram
     * history tail is image rows (dead). */
    const float d_odd_after = compare_odd_exact(e, ctx, vocab, &full, &span, after,
                                                "odd, right after the image");
    const float d_odd = compare_at(e, ctx, vocab, &full, &span, at + 1, "odd (info)");
    const int odd_ok = odd_greedy_image(e, ctx, vocab, &full, &span, at + 1);

    ds4_tokens text_full = {0};
    for (int i = 0; i < tail.len && text_full.len < full.len; i++)
        ds4_tokens_push(&text_full, tail.v[i]);
    const float d_odd_text = odd_delta_text(e, ctx, vocab, &text_full, at + 1);
    fprintf(stderr, "odd logit delta: image prefix %g, text-only control %g\n",
            d_odd, d_odd_text);
    free(text_full.v);

    ds4_vision_embedding_free(&span.embedding);
    free(full.v); free(tail.v); free(text);
    ds4_engine_close(e);

    if (d_even > 1e-4f) fail("even rewind with an image in the prefix differs from staged");
    if (d_after > 1e-4f) fail("even rewind right after the image differs from staged");
    if (d_odd_exact > 1e-4f) fail("odd rewind with an image in the prefix differs from staged decode");
    if (d_odd_after > 1e-4f) fail("odd rewind right after the image differs from staged decode");
    if (!odd_ok)
        fprintf(stderr, "note: odd greedy differs from the single-sweep reference "
                        "(batch-shape noise; the exact check above passed)\n");
    fprintf(stderr, "V4.1 image-prefix rewind PASS\n");
    return 0;
}
