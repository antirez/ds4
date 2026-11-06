/* Real-model mixed prefill/decode replay, including the sparse boundary.
 * Usage: test_qwen4_prefill MODEL PROMPT [MAX_CONTEXT [DUMP_DIRECTORY]] */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    assert(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct { int last, total, calls; } progress;
static void report(void *ud, const char *event, int current, int total) {
    if (strcmp(event, "prefill_chunk")) return;
    progress *p = ud;
    assert(current > p->last && current <= total && total == p->total);
    p->last = current;
    p->calls++;
}

static void sync_prefix(ds4_session *s, ds4_tokens *tokens, int n) {
    ds4_tokens prefix = *tokens;
    prefix.len = n;
    char error[256] = {0};
    int rc = ds4_session_sync(s, &prefix, error, sizeof(error));
    if (rc) fprintf(stderr, "sync %d: %s\n", n, error);
    assert(rc == 0 && ds4_session_pos(s) == n);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Usage: %s MODEL PROMPT [MAX_CONTEXT [DUMP_DIRECTORY]]\n", argv[0]);
        return 1;
    }
    char *end = NULL;
    long max = argc > 3 ? strtol(argv[3], &end, 10) : 8192;
    if ((end && *end) || max < 4096 || max > 131072) return 1;
    FILE *fp = fopen(argv[2], "rb");
    assert(fp && fseek(fp, 0, SEEK_END) == 0);
    long bytes = ftell(fp);
    assert(bytes > 0 && bytes < 64*1024*1024 && fseek(fp, 0, SEEK_SET) == 0);
    char *text = calloc((size_t)bytes + 1, 1);
    assert(text && fread(text, 1, bytes, fp) == (size_t)bytes);
    fclose(fp);
    ds4_engine_options opt = {.model_path = argv[1], .context_size = (int)max+8,
        .prefill_chunk = 1024, .glm_mtp = true, .dspark_exact_sampling = true,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL
#else
        .backend = DS4_BACKEND_CUDA
#endif
    };
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen4(engine));
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    assert(tokens.len > 0);
    const int original = tokens.len;
    while (tokens.len < max) ds4_tokens_push(&tokens, tokens.v[tokens.len % original]);
    ds4_session *live = NULL, *control = NULL;
    assert(ds4_session_create(&live, engine, (int)max+8) == 0);
    assert(ds4_session_create(&control, engine, (int)max+8) == 0);
    int vocab = ds4_engine_vocab_size(engine);
    float *a = malloc((size_t)vocab*4), *b = malloc((size_t)vocab*4);
    assert(a && b);
    int lengths[] = {128,129,132,511,512,1024,2047,2048,2049,2052,3072,4096,(int)max};
    int previous = 0;
    for (size_t j = 0; j < sizeof(lengths)/sizeof(*lengths); j++) {
        int n = lengths[j];
        if (n <= previous) continue;
        progress p = {.last = previous, .total = n};
        ds4_session_set_progress(live, report, &p);
        double start = now();
        sync_prefix(live, &tokens, n);
        double elapsed = now()-start;
        ds4_session_set_progress(live, NULL, NULL);
        assert(p.calls > 0 && p.last == n);
        assert(ds4_session_copy_logits(live, a, vocab) == vocab);

        ds4_session_invalidate(control);
        for (size_t k = 0; k <= j; k++) {
            if (k && lengths[k] <= lengths[k-1]) continue;
            sync_prefix(control, &tokens, lengths[k]);
        }
        assert(ds4_session_copy_logits(control, b, vocab) == vocab);
        float replay_error = 0;
        for (int i = 0; i < vocab; i++) {
            assert(isfinite(a[i]) && isfinite(b[i]));
            replay_error = fmaxf(replay_error, fabsf(a[i]-b[i]));
        }
        assert(replay_error < .001f);
        assert(ds4_session_argmax(live) == ds4_session_argmax(control));

        /* A different matrix shape changes rounding and can exchange nearly
         * tied experts. Report the distribution change separately from exact
         * state replay; large errors in low-probability logits alone are not
         * evidence of a state bug. Individual kernels have CPU oracles. */
        ds4_session_invalidate(control);
        sync_prefix(control, &tokens, n-3);
        char error[256] = {0};
        for (int t = n-3; t < n; t++)
            assert(ds4_session_eval(control, tokens.v[t], error, sizeof(error)) == 0);
        assert(ds4_session_copy_logits(control, b, vocab) == vocab);
        float worst = 0, max_a = -INFINITY, max_b = -INFINITY;
        for (int i = 0; i < vocab; i++) {
            assert(isfinite(a[i]) && isfinite(b[i]));
            worst = fmaxf(worst, fabsf(a[i]-b[i]));
            max_a = fmaxf(max_a, a[i]);
            max_b = fmaxf(max_b, b[i]);
        }
        double za = 0, zb = 0, tv = 0, kl = 0;
        for (int i = 0; i < vocab; i++) {
            za += exp((double)a[i]-max_a);
            zb += exp((double)b[i]-max_b);
        }
        const double la = max_a+log(za), lb = max_b+log(zb);
        for (int i = 0; i < vocab; i++) {
            double pa = exp(a[i]-la), pb = exp(b[i]-lb);
            tv += fabs(pa-pb)*.5;
            kl += pa*((a[i]-la)-(b[i]-lb));
        }
        printf("ctx=%d added=%d prefill=%.2f t/s calls=%d replay=%g "
               "serial_max=%g serial_tv=%.6g serial_kl=%.6g top1=%d/%d\n",
            n,n-previous,(n-previous)/elapsed,p.calls,replay_error,worst,tv,kl,
            ds4_session_argmax(live),ds4_session_argmax(control));
        fflush(stdout);
        if (argc == 5) {
            char path[4096];
            int written = snprintf(path,sizeof(path),"%s/frontier-%d.bin",argv[4],n);
            assert(written > 0 && (size_t)written < sizeof(path));
            fp = fopen(path,"wb");
            assert(fp && fwrite(a,4,vocab,fp) == (size_t)vocab);
            assert(fclose(fp) == 0);
        }
        previous = n;
    }
    free(b); free(a);
    ds4_session_free(control);
    ds4_session_free(live);
    ds4_tokens_free(&tokens);
    ds4_engine_close(engine);
    puts("Qwen mixed prefill replay and progress: OK (serial differences reported above)");
    return 0;
}
