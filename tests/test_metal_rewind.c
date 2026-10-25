/* Model-backed check of ds4_session_rewind for V4.1: a session rewound to
 * an earlier position and re-synced must match a session that reached the
 * same tokens through a fresh prefix and the same replay, and stay close to
 * a single-sweep prefill.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-metal-rewind
 * Options: DS4_TEST_PROMPT_FILE (default README.md), DS4_TEST_SSD_STREAMING=1.
 */

#include "ds4.h"
#include <stdbool.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static char *read_file(const char *path, size_t cap) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    char *buf = malloc(cap + 1);
    size_t n = buf ? fread(buf, 1, cap, fp) : 0;
    fclose(fp);
    if (!buf) return NULL;
    buf[n] = '\0';
    return buf;
}

static void slice(ds4_tokens *out, const ds4_tokens *src, int from, int to) {
    out->len = 0;
    for (int i = from; i < to; i++) ds4_tokens_push(out, src->v[i]);
}

static float *logits_of(ds4_session *s, int vocab) {
    float *l = malloc((size_t)vocab * sizeof(float));
    if (!l || ds4_session_copy_logits(s, l, vocab) != vocab) fail("copy logits");
    return l;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (!(d <= m)) m = d;
    }
    return m;
}

static int argmax(const float *l, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[best]) best = i;
    return best;
}

static void sync_or_fail(ds4_session *s, const ds4_tokens *t, const char *what) {
    char err[256];
    if (ds4_session_sync(s, t, err, sizeof(err)) != 0) {
        fprintf(stderr, "sync %s: %s\n", what, err);
        fail(what);
    }
}

/* Greedy-decode `steps` tokens through the sync API; returns the tokens. */
static void greedy(ds4_session *s, const ds4_tokens *prompt, int steps, int vocab, int *out) {
    ds4_tokens t = {0};
    slice(&t, prompt, 0, prompt->len);
    for (int i = 0; i < steps; i++) {
        float *l = logits_of(s, vocab);
        out[i] = argmax(l, vocab);
        free(l);
        ds4_tokens_push(&t, out[i]);
        sync_or_fail(s, &t, "greedy step");
    }
    ds4_tokens_free(&t);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: set DS4_TEST_MODEL\n");
        return 0;
    }
    const char *prompt_path = getenv("DS4_TEST_PROMPT_FILE");
    char *text = read_file(prompt_path && prompt_path[0] ? prompt_path : "README.md", 12000);
    if (!text) fail("prompt file");
    const int ctx = 4096;
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = 16384,
        .ssd_streaming = getenv("DS4_TEST_SSD_STREAMING") != NULL,
        /* DS4_TEST_SSD_CACHE_GB caps the expert bank (the long case adds two
         * 16k-context sessions on top of it). */
        .ssd_streaming_cache_bytes = getenv("DS4_TEST_SSD_CACHE_GB") ?
            (uint64_t)strtoull(getenv("DS4_TEST_SSD_CACHE_GB"), NULL, 10) << 30 : 0,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) fail("engine open");
    const int vocab = ds4_engine_vocab_size(engine);
    ds4_tokens all = {0};
    ds4_tokenize_text(engine, text, &all);
    int n = all.len < 1800 ? all.len : 1800;
    if (n < 700) fail("prompt too short (need 700 tokens)");
    ds4_tokens full = {0}, prefix = {0}, other = {0};
    slice(&full, &all, 0, n);
    const int deep = n - 300;       /* rewind past the 128-row window */
    slice(&prefix, &all, 0, deep);
    /* A prompt that shares the first deep tokens, then differs. */
    slice(&other, &all, 0, deep);
    for (int i = 0; i < 40; i++) ds4_tokens_push(&other, all.v[(deep + 7 * i + 3) % n]);

    ds4_session *fresh = NULL, *staged = NULL, *rewound = NULL;
    if (ds4_session_create(&fresh, engine, ctx) || ds4_session_create(&staged, engine, ctx) ||
        ds4_session_create(&rewound, engine, ctx)) fail("session create");

    /* Reference: one sweep over the full prompt, then a few greedy tokens. */
    sync_or_fail(fresh, &full, "fresh full");
    float *l_fresh = logits_of(fresh, vocab);
    int g_fresh[6];
    greedy(fresh, &full, 6, vocab, g_fresh);

    /* Staged: the prefix, then the tail as its own sweep (the extension path
     * every cache hit takes; its distance from the reference is the tail
     * path's own numerical noise). */
    sync_or_fail(staged, &prefix, "staged prefix");
    sync_or_fail(staged, &full, "staged tail");
    float *l_staged = logits_of(staged, vocab);
    int g_staged[6];
    greedy(staged, &full, 6, vocab, g_staged);

    /* Rewound: the same prefix and tail as `staged`, plus a few generated
     * tokens, rewound to the prefix, then the tail replayed: the prefix rows
     * were computed by the same sweep, so this must match `staged` exactly.
     * (A single sweep over the whole prompt computes the prefix rows in a
     * different batch shape and lands up to ~1 logit away -- the staged
     * distance above -- so the rewind is judged against `staged`.) */
    sync_or_fail(rewound, &prefix, "rewound prefix");
    sync_or_fail(rewound, &full, "rewound full");
    int g_tmp[6];
    greedy(rewound, &full, 6, vocab, g_tmp);
    if (ds4_session_pos(rewound) != n + 6) fail("rewound position before rewind");
    ds4_session_rewind(rewound, deep);
    if (ds4_session_pos(rewound) != deep) fail("rewind position");
    if (ds4_session_common_prefix(rewound, &full) != deep) fail("rewind kept no state");
    sync_or_fail(rewound, &full, "rewound tail");
    float *l_rewound = logits_of(rewound, vocab);
    int g_rewound[6];
    greedy(rewound, &full, 6, vocab, g_rewound);

    const float staged_vs_fresh = max_abs_diff(l_staged, l_fresh, vocab);
    const float rewound_vs_staged = max_abs_diff(l_rewound, l_staged, vocab);
    const float rewound_vs_fresh = max_abs_diff(l_rewound, l_fresh, vocab);
    fprintf(stderr, "logits: staged-vs-fresh %g  rewound-vs-staged %g  rewound-vs-fresh %g\n",
            staged_vs_fresh, rewound_vs_staged, rewound_vs_fresh);
    fprintf(stderr, "greedy: fresh %d %d %d %d %d %d | staged %d %d %d %d %d %d | rewound %d %d %d %d %d %d\n",
            g_fresh[0], g_fresh[1], g_fresh[2], g_fresh[3], g_fresh[4], g_fresh[5],
            g_staged[0], g_staged[1], g_staged[2], g_staged[3], g_staged[4], g_staged[5],
            g_rewound[0], g_rewound[1], g_rewound[2], g_rewound[3], g_rewound[4], g_rewound[5]);
    if (memcmp(g_rewound, g_staged, sizeof(g_staged)) != 0) fail("rewound greedy differs from staged");
    if (rewound_vs_staged > 1e-4f) fail("rewound logits differ from staged");
    if (argmax(l_rewound, vocab) != argmax(l_fresh, vocab)) fail("rewound argmax differs from fresh");

    /* Divergent prompt after a short rewind (inside the window), against a
     * fresh session that took the same prefix-then-tail route. */
    ds4_session *diverged = NULL;
    if (ds4_session_create(&diverged, engine, ctx)) fail("diverged session");
    sync_or_fail(diverged, &prefix, "diverged prefix");
    sync_or_fail(diverged, &other, "diverged tail");
    float *l_div = logits_of(diverged, vocab);
    ds4_session_rewind(rewound, deep);
    if (ds4_session_common_prefix(rewound, &other) != deep) fail("second rewind kept no state");
    sync_or_fail(rewound, &other, "rewound other");
    float *l_rw2 = logits_of(rewound, vocab);
    const float div_diff = max_abs_diff(l_rw2, l_div, vocab);
    fprintf(stderr, "diverged: rewound-vs-staged %g\n", div_diff);
    if (div_diff > 1e-4f) fail("diverged rewind logits differ");
    /* A rewind by only a few tokens, inside the window. */
    int g_a[4], g_b[4];
    greedy(rewound, &other, 4, vocab, g_a);
    ds4_session_rewind(rewound, other.len - 4);
    sync_or_fail(rewound, &other, "short rewind");
    greedy(rewound, &other, 4, vocab, g_b);
    if (memcmp(g_a, g_b, sizeof(g_a)) != 0) fail("short rewind changed greedy tokens");

    /* An odd target: the graph only rewinds to even positions, so the
     * session must land exactly on the odd position with its state kept
     * (one token replayed), not invalidate and force a rebuild. */
    {
        const int odd = deep | 1;
        int g_c[4], g_d[4];
        sync_or_fail(rewound, &full, "odd setup");
        greedy(rewound, &full, 4, vocab, g_c);
        ds4_session_rewind(rewound, odd);
        if (ds4_session_pos(rewound) != odd) fail("odd rewind position");
        if (ds4_session_common_prefix(rewound, &full) != odd) fail("odd rewind kept no state");
        sync_or_fail(rewound, &full, "odd tail");
        greedy(rewound, &full, 4, vocab, g_d);
        fprintf(stderr, "odd rewind to %d: greedy %d %d %d %d vs %d %d %d %d\n", odd,
                g_c[0], g_c[1], g_c[2], g_c[3], g_d[0], g_d[1], g_d[2], g_d[3]);
        if (memcmp(g_c, g_d, sizeof(g_c)) != 0) fail("odd rewind changed greedy tokens");
    }

    fprintf(stderr, "V4.1 rewind PASS n=%d deep=%d\n", n, deep);
    free(l_fresh); free(l_staged); free(l_rewound); free(l_div); free(l_rw2);
    ds4_session_free(fresh); ds4_session_free(staged); ds4_session_free(rewound); ds4_session_free(diverged);
    /* A 10000-token prompt is prefilled as an 8192-token decoder-suffix sweep
     * plus a tail; in that sweep layers 20-39 compute only its end (layer 39
     * the last 128 rows).  A rewind to 6000 lands where those layers never
     * computed raw rows, so it must drop the state, and the rebuild must match
     * a fresh session exactly.  A rewind into the decoded tail past the prompt
     * still keeps the state. */
    {
        const char *lp = getenv("DS4_TEST_LONG_PROMPT_FILE");
        char *ltext = read_file(lp && lp[0] ? lp : "ds4_server.c", 400000);
        if (!ltext) fail("long prompt file");
        ds4_tokens lall = {0}, lfull = {0}, lnext = {0};
        ds4_tokenize_text(engine, ltext, &lall);
        if (lall.len < 10000) fail("long prompt too short (need 10000 tokens)");
        slice(&lfull, &lall, 0, 10000);
        const int lctx = 16384, hole = 6000;
        ds4_session *lfresh = NULL, *lrw = NULL;
        if (ds4_session_create(&lfresh, engine, lctx) ||
            ds4_session_create(&lrw, engine, lctx)) fail("long session create");
        int g_f[4], g_r[4], g_x[4], g_t[1];
        /* Reference: the whole prompt from an empty session, which is what a
         * dropped state rebuilds. */
        sync_or_fail(lfresh, &lfull, "long fresh");
        float *l_lf = logits_of(lfresh, vocab);
        greedy(lfresh, &lfull, 4, vocab, g_f);
        sync_or_fail(lrw, &lfull, "long setup");
        greedy(lrw, &lfull, 4, vocab, g_x);
        ds4_session_rewind(lrw, hole);
        const bool kept = ds4_session_common_prefix(lrw, &lfull) == hole;
        sync_or_fail(lrw, &lfull, "long resync");
        float *l_lr = logits_of(lrw, vocab);
        greedy(lrw, &lfull, 4, vocab, g_r);
        if (kept) fail("long rewind kept state its raw rows never had");
        const float vs_ref = max_abs_diff(l_lr, l_lf, vocab);
        fprintf(stderr, "long rewind into a %d-token prompt at %d: state dropped; logits vs fresh sweep %g; greedy %d %d %d %d vs fresh %d %d %d %d\n",
                lfull.len, hole, vs_ref,
                g_r[0], g_r[1], g_r[2], g_r[3], g_f[0], g_f[1], g_f[2], g_f[3]);
        if (vs_ref > 1e-4f) fail("long rebuild differs from a fresh sweep");
        if (memcmp(g_r, g_f, sizeof(g_f)) != 0) fail("long rewind changed greedy tokens");
        free(l_lf); free(l_lr);

        /* Past the sweep: rewind by one decoded token (odd target) and
         * regenerate the last greedy token from the kept state. */
        slice(&lnext, &lfull, 0, lfull.len);
        for (int i = 0; i < 3; i++) ds4_tokens_push(&lnext, g_r[i]);
        ds4_session_rewind(lrw, lnext.len);
        if (ds4_session_pos(lrw) != lnext.len ||
            ds4_session_common_prefix(lrw, &lnext) != lnext.len)
            fail("rewind past a long sweep kept no state");
        greedy(lrw, &lnext, 1, vocab, g_t);
        fprintf(stderr, "rewind past the sweep to %d: state kept, greedy %d vs %d\n",
                lnext.len, g_t[0], g_r[3]);
        if (g_t[0] != g_r[3]) fail("rewind past a long sweep changed greedy token");
        ds4_session_free(lfresh); ds4_session_free(lrw);
        ds4_tokens_free(&lall); ds4_tokens_free(&lfull); ds4_tokens_free(&lnext);
        free(ltext);
        fprintf(stderr, "V4.1 long-sweep rewind PASS\n");
    }

    ds4_engine_close(engine);
    free(text);
    return 0;
}
