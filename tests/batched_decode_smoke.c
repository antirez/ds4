/* batched_decode_smoke.c — compare greedy decode through
 * ds4_session_eval_multi against single-token ds4_session_eval on the same
 * prompts.  The batched path runs the dense stages through the batch
 * kernels, so tokens are expected to match in practice but are not
 * guaranteed bit-identical; this harness reports the first divergence and
 * the logits distance at that step so a real bug (garbage logits, crossed
 * sequences) is obvious against benign numeric drift.
 *
 * Set DS4_BATCHED_DECODE_FORCE=1 in the environment before running to also
 * exercise the n==1 batched path (done by this binary automatically).
 *
 * Usage: tests/batched_decode_smoke <model.gguf> [n_tokens]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ds4.h"

#define MAX_GEN 64

typedef struct {
    int toks[MAX_GEN];
    int len;
    float *logits; /* n * vocab, logits BEFORE sampling step i */
} gen_result;

static int g_vocab = 0;

static void die(const char *msg, const char *detail) {
    fprintf(stderr, "batched_decode_smoke: %s%s%s\n", msg,
            detail ? ": " : "", detail ? detail : "");
    exit(1);
}

static void capture_logits(ds4_session *s, gen_result *r, int step) {
    if (ds4_session_copy_logits(s, r->logits + (size_t)step * g_vocab, g_vocab) < 0) {
        die("copy_logits failed", NULL);
    }
}

static void run_single(ds4_engine *e, const ds4_tokens *prompt, int n,
                       int ctx_size, gen_result *out) {
    ds4_session *s = NULL;
    char err[256] = "";
    if (ds4_session_create(&s, e, ctx_size) != 0) die("session_create failed", NULL);
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) die("sync failed", err);
    const int eos = ds4_token_eos(e);
    out->len = 0;
    for (int i = 0; i < n; i++) {
        capture_logits(s, out, i);
        const int tok = ds4_session_argmax(s);
        if (tok < 0 || tok == eos) break;
        if (ds4_session_eval(s, tok, err, sizeof(err)) != 0) die("eval failed", err);
        out->toks[out->len++] = tok;
    }
    ds4_session_free(s);
}

/* Drive n_seqs sessions through ds4_session_eval_multi with greedy sampling.
 * Each stream keeps stepping until its own EOS; finished streams are dropped
 * from the batch (the batch shrinks, mirroring the server scheduler). */
static void run_batched(ds4_engine *e, const ds4_tokens **prompts,
                        uint32_t n_seqs, int n, int ctx_size,
                        gen_result *out /* n_seqs entries */) {
    ds4_session *all[4] = {0};
    char err[256] = "";
    const int eos = ds4_token_eos(e);
    for (uint32_t b = 0; b < n_seqs; b++) {
        if (ds4_session_create(&all[b], e, ctx_size) != 0) die("session_create failed", NULL);
        if (ds4_session_sync(all[b], prompts[b], err, sizeof(err)) != 0) die("sync failed", err);
        out[b].len = 0;
    }
    bool done[4] = {false};
    for (int i = 0; i < n; i++) {
        ds4_session *live[4];
        int tokens[4];
        uint32_t live_idx[4];
        int nl = 0;
        for (uint32_t b = 0; b < n_seqs; b++) {
            if (done[b]) continue;
            capture_logits(all[b], &out[b], out[b].len);
            const int tok = ds4_session_argmax(all[b]);
            if (tok < 0 || tok == eos) {
                done[b] = true;
                continue;
            }
            live[nl] = all[b];
            tokens[nl] = tok;
            live_idx[nl] = b;
            nl++;
        }
        if (nl == 0) break;
        if (ds4_session_eval_multi(live, nl, tokens, err, sizeof(err)) != 0) {
            die("eval_multi failed", err);
        }
        for (int k = 0; k < nl; k++) {
            gen_result *r = &out[live_idx[k]];
            r->toks[r->len++] = tokens[k];
        }
    }
    for (uint32_t b = 0; b < n_seqs; b++) ds4_session_free(all[b]);
}

static void print_stream(ds4_engine *e, const char *label, const gen_result *r) {
    printf("%-14s (%2d tokens): ", label, r->len);
    for (int i = 0; i < r->len; i++) {
        size_t len = 0;
        char *txt = ds4_token_text(e, r->toks[i], &len);
        if (txt) fwrite(txt, 1, len, stdout);
    }
    printf("\n");
}

/* Compare a batched stream against its single reference: report the first
 * token divergence and the logits distance at that step. */
static int compare(const char *label, const gen_result *ref, const gen_result *got) {
    const int n = ref->len < got->len ? ref->len : got->len;
    int div = -1;
    for (int i = 0; i < n; i++) {
        if (ref->toks[i] != got->toks[i]) { div = i; break; }
    }
    if (div < 0 && ref->len == got->len) {
        printf("%s: MATCH (%d tokens)\n", label, ref->len);
        return 0;
    }
    if (div < 0) div = n;
    float max_diff = 0.0f;
    double sum_sq = 0.0;
    const float *lr = ref->logits + (size_t)div * g_vocab;
    const float *lg = got->logits + (size_t)div * g_vocab;
    for (int v = 0; v < g_vocab; v++) {
        const float d = fabsf(lr[v] - lg[v]);
        if (d > max_diff) max_diff = d;
        sum_sq += (double)d * d;
    }
    printf("%s: DIVERGED at step %d (ref len %d, got len %d); "
           "logits max|diff| %.6f, rms %.6f\n",
           label, div, ref->len, got->len,
           (double)max_diff, sqrt(sum_sq / g_vocab));
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: batched_decode_smoke <model.gguf> [n_tokens]", NULL);
    int n = argc > 2 ? atoi(argv[2]) : 24;
    if (n <= 0 || n > MAX_GEN) n = 24;
    const int ctx_size = 4096;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.prefill_chunk = 256;

    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) die("engine_open failed", argv[1]);
    if (!ds4_engine_supports_batched_decode(e)) die("engine does not support batched decode", NULL);
    g_vocab = ds4_engine_vocab_size(e);

    const char *sys = "You are a concise technical assistant.";
    ds4_tokens pa, pb;
    memset(&pa, 0, sizeof(pa));
    memset(&pb, 0, sizeof(pb));
    ds4_encode_chat_prompt(e, sys, "Briefly explain what a compressor does in a refrigeration cycle.", DS4_THINK_NONE, &pa);
    ds4_encode_chat_prompt(e, sys, "List three common causes of high condensing pressure.", DS4_THINK_NONE, &pb);

    gen_result iso_a = {0}, iso_b = {0}, b1_a = {0}, b2[2];
    memset(b2, 0, sizeof(b2));
    iso_a.logits = malloc((size_t)n * g_vocab * sizeof(float));
    iso_b.logits = malloc((size_t)n * g_vocab * sizeof(float));
    b1_a.logits = malloc((size_t)n * g_vocab * sizeof(float));
    b2[0].logits = malloc((size_t)n * g_vocab * sizeof(float));
    b2[1].logits = malloc((size_t)n * g_vocab * sizeof(float));
    if (!iso_a.logits || !iso_b.logits || !b1_a.logits || !b2[0].logits || !b2[1].logits) {
        die("logits alloc failed", NULL);
    }

    printf("== single reference ==\n");
    run_single(e, &pa, n, ctx_size, &iso_a);
    print_stream(e, "A/single", &iso_a);
    run_single(e, &pb, n, ctx_size, &iso_b);
    print_stream(e, "B/single", &iso_b);

    printf("== batched n=1 (forced) ==\n");
    setenv("DS4_BATCHED_DECODE_FORCE", "1", 1);
    const ds4_tokens *only_a[1] = {&pa};
    run_batched(e, only_a, 1, n, ctx_size, &b1_a);
    print_stream(e, "A/batched1", &b1_a);

    printf("== batched n=2 ==\n");
    const ds4_tokens *both[2] = {&pa, &pb};
    run_batched(e, both, 2, n, ctx_size, b2);
    print_stream(e, "A/batched2", &b2[0]);
    print_stream(e, "B/batched2", &b2[1]);

    int rc = 0;
    rc |= compare("A single-vs-batched1", &iso_a, &b1_a);
    rc |= compare("A single-vs-batched2", &iso_a, &b2[0]);
    rc |= compare("B single-vs-batched2", &iso_b, &b2[1]);

    ds4_tokens_free(&pa);
    ds4_tokens_free(&pb);
    ds4_engine_close(e);
    printf(rc == 0 ? "OK\n" : "DIVERGENCE REPORTED\n");
    return rc;
}
