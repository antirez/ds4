/* two_sessions_smoke.c — prove that two ds4_session objects sharing one
 * ds4_engine produce the same greedy token streams when driven interleaved
 * (one token each, alternating on the same thread) as when each session is
 * driven alone. This is the ground assumption for multi-session serving:
 * backend statics must not carry per-graph state across ds4_session_eval()
 * calls of different sessions.
 *
 * Usage: tests/two_sessions_smoke <model.gguf> [n_tokens]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ds4.h"

#define MAX_GEN 128

typedef struct {
    int toks[MAX_GEN];
    int len;
} gen_result;

static void die(const char *msg, const char *detail) {
    fprintf(stderr, "two_sessions_smoke: %s%s%s\n", msg,
            detail ? ": " : "", detail ? detail : "");
    exit(1);
}

static void encode_prompt(ds4_engine *e, const char *system, const char *user,
                          ds4_tokens *out) {
    memset(out, 0, sizeof(*out));
    ds4_encode_chat_prompt(e, system, user, DS4_THINK_NONE, out);
}

/* One greedy step: argmax over current logits, then advance the session.
 * Returns the sampled token, or -1 once the stream hit EOS. */
static int greedy_step(ds4_session *s, int eos, char *err, size_t errlen) {
    int tok = ds4_session_argmax(s);
    if (tok < 0 || tok == eos) return -1;
    if (ds4_session_eval(s, tok, err, errlen) != 0) die("eval failed", err);
    return tok;
}

static void run_isolated(ds4_engine *e, const ds4_tokens *prompt, int n,
                         int ctx_size, gen_result *out) {
    ds4_session *s = NULL;
    char err[256] = "";
    if (ds4_session_create(&s, e, ctx_size) != 0) die("session_create failed", NULL);
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) die("sync failed", err);
    int eos = ds4_token_eos(e);
    out->len = 0;
    for (int i = 0; i < n; i++) {
        int tok = greedy_step(s, eos, err, sizeof(err));
        if (tok < 0) break;
        out->toks[out->len++] = tok;
    }
    ds4_session_free(s);
}

static void run_interleaved(ds4_engine *e, const ds4_tokens *pa,
                            const ds4_tokens *pb, int n, int ctx_size,
                            gen_result *ra, gen_result *rb) {
    ds4_session *sa = NULL, *sb = NULL;
    char err[256] = "";
    if (ds4_session_create(&sa, e, ctx_size) != 0) die("session_create A failed", NULL);
    if (ds4_session_create(&sb, e, ctx_size) != 0) die("session_create B failed", NULL);
    if (ds4_session_sync(sa, pa, err, sizeof(err)) != 0) die("sync A failed", err);
    if (ds4_session_sync(sb, pb, err, sizeof(err)) != 0) die("sync B failed", err);
    int eos = ds4_token_eos(e);
    ra->len = rb->len = 0;
    bool a_done = false, b_done = false;
    for (int i = 0; i < n && (!a_done || !b_done); i++) {
        if (!a_done) {
            int tok = greedy_step(sa, eos, err, sizeof(err));
            if (tok < 0) a_done = true; else ra->toks[ra->len++] = tok;
        }
        if (!b_done) {
            int tok = greedy_step(sb, eos, err, sizeof(err));
            if (tok < 0) b_done = true; else rb->toks[rb->len++] = tok;
        }
    }
    ds4_session_free(sa);
    ds4_session_free(sb);
}

static void print_stream(ds4_engine *e, const char *label, const gen_result *r) {
    printf("%s (%d tokens): ", label, r->len);
    for (int i = 0; i < r->len; i++) {
        size_t len = 0;
        char *txt = ds4_token_text(e, r->toks[i], &len);
        if (txt) fwrite(txt, 1, len, stdout);
    }
    printf("\n");
}

static bool same(const gen_result *x, const gen_result *y) {
    if (x->len != y->len) return false;
    return memcmp(x->toks, y->toks, sizeof(int) * (size_t)x->len) == 0;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: two_sessions_smoke <model.gguf> [n_tokens]", NULL);
    int n = argc > 2 ? atoi(argv[2]) : 24;
    if (n <= 0 || n > MAX_GEN) n = 24;
    int ctx_size = 4096;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.prefill_chunk = 256;

    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) die("engine_open failed", argv[1]);

    const char *sys = "You are a concise technical assistant.";
    ds4_tokens pa, pb;
    encode_prompt(e, sys, "Briefly explain what a compressor does in a refrigeration cycle.", &pa);
    encode_prompt(e, sys, "List three common causes of high condensing pressure.", &pb);

    gen_result iso_a, iso_b, mix_a, mix_b;
    printf("== isolated runs ==\n");
    run_isolated(e, &pa, n, ctx_size, &iso_a);
    print_stream(e, "A/isolated", &iso_a);
    run_isolated(e, &pb, n, ctx_size, &iso_b);
    print_stream(e, "B/isolated", &iso_b);

    printf("== interleaved run ==\n");
    run_interleaved(e, &pa, &pb, n, ctx_size, &mix_a, &mix_b);
    print_stream(e, "A/interleaved", &mix_a);
    print_stream(e, "B/interleaved", &mix_b);

    bool ok_a = same(&iso_a, &mix_a);
    bool ok_b = same(&iso_b, &mix_b);
    printf("A: %s, B: %s\n", ok_a ? "MATCH" : "MISMATCH", ok_b ? "MATCH" : "MISMATCH");

    ds4_tokens_free(&pa);
    ds4_tokens_free(&pb);
    ds4_engine_close(e);

    if (!ok_a || !ok_b) { printf("FAILED\n"); return 1; }
    printf("OK\n");
    return 0;
}
