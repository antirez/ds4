/* batched_decode_bench.c — isolate the cost of a decode step.
 *
 * The end-to-end server test mixes prefill contention with decode, which
 * hides what batched decode actually does per token.  This bench prefills
 * once, warms up, then times pure decode steps for:
 *   - single session (ds4_session_eval)                  -> ms/token, tok/s
 *   - B sessions batched (ds4_session_eval_multi)        -> ms/step, agg tok/s
 * and reports the aggregate throughput ratio vs running the sessions one at
 * a time.  Ratio > 1 means batching more than pays for itself at that B.
 *
 * Usage: tests/batched_decode_bench <model.gguf> [steps]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../ds4.h"

#define MAX_B 4
#define WARMUP 8

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *msg, const char *detail) {
    fprintf(stderr, "batched_decode_bench: %s%s%s\n", msg,
            detail ? ": " : "", detail ? detail : "");
    exit(1);
}

static const char *PROMPTS[MAX_B] = {
    "Explain how a thermostatic expansion valve regulates superheat.",
    "Describe how condenser fan speed control stabilizes head pressure.",
    "Summarize why subcooling matters at the condenser outlet.",
    "Explain the role of the receiver in a refrigeration circuit.",
};

/* Time `steps` single-token decode steps on one session (greedy). */
static double bench_single(ds4_engine *e, const ds4_tokens *prompt, int ctx, int steps) {
    ds4_session *s = NULL;
    char err[256] = "";
    if (ds4_session_create(&s, e, ctx) != 0) die("session_create failed", NULL);
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) die("sync failed", err);
    for (int i = 0; i < WARMUP; i++) {
        int tok = ds4_session_argmax(s);
        if (ds4_session_eval(s, tok, err, sizeof(err)) != 0) die("eval failed", err);
    }
    const double t0 = now_sec();
    for (int i = 0; i < steps; i++) {
        int tok = ds4_session_argmax(s);
        if (ds4_session_eval(s, tok, err, sizeof(err)) != 0) die("eval failed", err);
    }
    const double ms = (now_sec() - t0) * 1000.0 / steps;
    ds4_session_free(s);
    return ms;
}

/* Time `steps` batched decode steps over B sessions (greedy, one token each
 * per step).  Returns ms per step (each step commits B tokens). */
static double bench_batched(ds4_engine *e, const ds4_tokens **prompts, int B,
                            int ctx, int steps) {
    ds4_session *all[MAX_B] = {0};
    char err[256] = "";
    for (int b = 0; b < B; b++) {
        if (ds4_session_create(&all[b], e, ctx) != 0) die("session_create failed", NULL);
        if (ds4_session_sync(all[b], prompts[b], err, sizeof(err)) != 0) die("sync failed", err);
    }
    int tokens[MAX_B];
    for (int i = 0; i < WARMUP; i++) {
        for (int b = 0; b < B; b++) tokens[b] = ds4_session_argmax(all[b]);
        if (ds4_session_eval_multi(all, B, tokens, err, sizeof(err)) != 0) die("eval_multi failed", err);
    }
    const double t0 = now_sec();
    for (int i = 0; i < steps; i++) {
        for (int b = 0; b < B; b++) tokens[b] = ds4_session_argmax(all[b]);
        if (ds4_session_eval_multi(all, B, tokens, err, sizeof(err)) != 0) die("eval_multi failed", err);
    }
    const double ms = (now_sec() - t0) * 1000.0 / steps;
    for (int b = 0; b < B; b++) ds4_session_free(all[b]);
    return ms;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: batched_decode_bench <model.gguf> [steps]", NULL);
    int steps = argc > 2 ? atoi(argv[2]) : 60;
    if (steps <= 0) steps = 60;
    const int ctx = 4096;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.prefill_chunk = 256;

    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) die("engine_open failed", argv[1]);
    if (!ds4_engine_supports_batched_decode(e)) die("engine does not support batched decode", NULL);

    const char *sys = "You are a concise technical assistant.";
    ds4_tokens pr[MAX_B];
    const ds4_tokens *pp[MAX_B];
    for (int b = 0; b < MAX_B; b++) {
        memset(&pr[b], 0, sizeof(pr[b]));
        ds4_encode_chat_prompt(e, sys, PROMPTS[b], DS4_THINK_NONE, &pr[b]);
        pp[b] = &pr[b];
    }

    printf("model warmed; timing %d decode steps each\n\n", steps);

    const double ms_single = bench_single(e, &pr[0], ctx, steps);
    const double single_tps = 1000.0 / ms_single;
    printf("single decode:      %7.3f ms/token   %6.1f tok/s\n", ms_single, single_tps);

    for (int B = 2; B <= MAX_B; B++) {
        const double ms_step = bench_batched(e, pp, B, ctx, steps);
        const double agg_tps = (double)B * 1000.0 / ms_step;
        const double ms_per_tok = ms_step / B;
        /* Aggregate throughput of B sequential single-streams is B*single_tps
         * only if the GPU could serve them in parallel for free; the honest
         * baseline for "was batching worth it" is B independent single
         * streams sharing the GPU sequentially, i.e. B/single_tps time for
         * B tokens => single_tps aggregate.  Ratio vs that: */
        const double ratio = agg_tps / single_tps;
        printf("batched B=%d:        %7.3f ms/step  (%6.3f ms/token)  "
               "%6.1f tok/s agg   %.2fx vs interleaved\n",
               B, ms_step, ms_per_tok, agg_tps, ratio);
    }

    for (int b = 0; b < MAX_B; b++) ds4_tokens_free(&pr[b]);
    ds4_engine_close(e);
    printf("\nnote: >1.00x means batched decode beats serving the same B "
           "streams one-token-at-a-time on the shared GPU.\n");
    return 0;
}
