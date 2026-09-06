#define _DARWIN_C_SOURCE
#include "ds4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int limit;
    int calls;
    int accepted;
    int ids[16];
} token_capture;

static bool capture_token(void *ctx, int token) {
    token_capture *capture = ctx;
    capture->calls++;
    if (capture->accepted == capture->limit) return false;
    capture->ids[capture->accepted++] = token;
    return true;
}

static int compare_state(ds4_session *control, ds4_session *chain,
                         float *a, float *b, int vocab,
                         const char *name, int prefix, const char *phase) {
    const ds4_tokens *ta = ds4_session_tokens(control);
    const ds4_tokens *tb = ds4_session_tokens(chain);
    if (!ta || !tb || ta->len != tb->len ||
        memcmp(ta->v, tb->v, (size_t)ta->len * sizeof(ta->v[0])) != 0) {
        fprintf(stderr, "FAIL %s prefix=%d phase=%s token checkpoint mismatch control=%d chain=%d\n",
                name, prefix, phase, ta ? ta->len : -1, tb ? tb->len : -1);
        return 1;
    }
    memset(a, 0xa5, (size_t)vocab * sizeof(*a));
    memset(b, 0x5a, (size_t)vocab * sizeof(*b));
    if (ds4_session_copy_logits(control, a, vocab) != vocab ||
        ds4_session_copy_logits(chain, b, vocab) != vocab) {
        fprintf(stderr, "FAIL %s prefix=%d phase=%s logits unavailable\n", name, prefix, phase);
        return 1;
    }
    int different = 0, first = -1;
    for (int i = 0; i < vocab; ++i) {
        uint32_t bits_a, bits_b;
        memcpy(&bits_a, &a[i], sizeof(bits_a));
        memcpy(&bits_b, &b[i], sizeof(bits_b));
        if ((bits_a & 0x7f800000u) == 0x7f800000u ||
            (bits_b & 0x7f800000u) == 0x7f800000u) {
            fprintf(stderr, "FAIL %s prefix=%d phase=%s non-finite logits id=%d\n",
                    name, prefix, phase, i);
            return 1;
        }
        if (memcmp(&a[i], &b[i], sizeof(a[i])) != 0) {
            if (first < 0) first = i;
            different++;
        }
    }
    if (different) {
        fprintf(stderr,
                "FAIL %s prefix=%d phase=%s different=%d/%d first=%d control=%a chain=%a argmax=%d/%d\n",
                name, prefix, phase, different, vocab, first,
                a[first], b[first], ds4_session_argmax(control), ds4_session_argmax(chain));
        return 1;
    }
    fprintf(stderr, "PASS %s prefix=%d phase=%s pos=%d logits=%d exact\n",
            name, prefix, phase, ta->len, vocab);
    return 0;
}

static int run_case(ds4_engine *engine, const ds4_tokens *source, int prefix_len,
                    int ctx_size, int burst, int accept_limit, int rounds, const char *name,
                    float *a, float *b, int vocab) {
    ds4_session *control = NULL, *chain = NULL;
    ds4_tokens prefix = { .v = source->v, .len = prefix_len, .cap = prefix_len };
    char err[256] = {0};
    int failures = 0;
    if (ds4_session_create(&control, engine, ctx_size) != 0 ||
        ds4_session_create(&chain, engine, ctx_size) != 0 ||
        ds4_session_sync(control, &prefix, err, sizeof(err)) != 0 ||
        ds4_session_sync(chain, &prefix, err, sizeof(err)) != 0) {
        fprintf(stderr, "FAIL %s prefix=%d setup: %s\n", name, prefix_len, err);
        failures++;
        goto done;
    }
    if (!ds4_session_chain_greedy_supported(chain)) {
        fprintf(stderr, "FAIL session does not support Metal greedy chain\n");
        failures++;
        goto done;
    }
    failures += compare_state(control, chain, a, b, vocab, name, prefix_len, "prefill");
    if (failures) goto done;

    for (int round = 0; round < rounds; ++round) {
        const int headroom = ctx_size - ds4_session_pos(control);
        const int capped_burst = burst < headroom ? burst : headroom;
        const bool early_reject = accept_limit < capped_burst;
        const int accepted = early_reject ? accept_limit : capped_burst;
        int expected[16] = {0};
        for (int i = 0; i < accepted; ++i) {
            expected[i] = ds4_session_argmax(control);
            if (expected[i] < 0 || ds4_session_eval(control, expected[i], err, sizeof(err)) != 0) {
                fprintf(stderr, "FAIL classic eval: %s\n", err);
                failures++;
                goto done;
            }
        }
        token_capture capture = { .limit = accept_limit };
        bool completed = false;
        const int approved = ds4_session_eval_chain_greedy(
            chain, burst, capture_token, &capture, &completed, err, sizeof(err));
        if (approved != accepted || capture.accepted != accepted ||
            completed != !early_reject ||
            capture.calls != accepted + (early_reject ? 1 : 0) ||
            memcmp(expected, capture.ids, (size_t)accepted * sizeof(expected[0])) != 0) {
            fprintf(stderr,
                    "FAIL %s prefix=%d burst approved=%d accepted=%d calls=%d completed=%d error=%s\n",
                    name, prefix_len, approved, capture.accepted, capture.calls, completed, err);
            failures++;
            goto done;
        }
        char phase[32];
        snprintf(phase, sizeof(phase), "after-burst-%d", round + 1);
        failures += compare_state(control, chain, a, b, vocab, name, prefix_len, phase);
    }
    /* Feed both sessions an identical forced continuation without consulting
     * candidate logits; this tests the retained KV/compressor state even when
     * the frontier comparison already failed. Avoid the rejected
     * prediction so a repeated identical token cannot hide state contamination. */
    for (int step = 0; step < 4 && ds4_session_pos(control) < ctx_size; ++step) {
        const int next = (ds4_session_argmax(control) + 1) % vocab;
        if (ds4_session_eval(control, next, err, sizeof(err)) != 0 ||
            ds4_session_eval(chain, next, err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL %s prefix=%d follow=%d eval: %s\n", name, prefix_len, step + 1, err);
            failures++;
            break;
        }
        char phase[32];
        snprintf(phase, sizeof(phase), "classic-follow-%d", step + 1);
        failures += compare_state(control, chain, a, b, vocab, name, prefix_len, phase);
    }

done:
    ds4_session_free(control);
    ds4_session_free(chain);
    return failures;
}

static int run_raw_scope_guard(ds4_engine *engine, const ds4_tokens *source) {
    ds4_session *s = NULL;
    ds4_tokens prefix = { .v = source->v, .len = 126, .cap = 126 };
    char err[256] = {0};
    int failed = ds4_session_create(&s, engine, 512) != 0 ||
        ds4_session_sync(s, &prefix, err, sizeof(err)) != 0;
    token_capture capture = { .limit = 2 };
    bool completed = false;
    if (!failed) {
        failed = ds4_session_eval_chain_greedy(
                s, 2, capture_token, &capture, &completed, err, sizeof(err)) != -1 ||
            capture.calls != 0;
    }
    fprintf(stderr, "%s raw-prefix strict128-row scope guard\n", failed ? "FAIL" : "PASS");
    ds4_session_free(s);
    return failed;
}

int main(int argc, char **argv) {
    const char *model = getenv("DS4_TEST_MODEL");
    bool raw_prefix = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--raw-prefix")) raw_prefix = true;
        else if (argv[i][0] != '-') model = argv[i];
        else {
            fprintf(stderr, "usage: %s [model.gguf] [--raw-prefix]\n", argv[0]);
            return 2;
        }
    }
    if (!model || !model[0]) model = "ds4flash.gguf";
    const char *disable = "DS4_METAL_DISABLE_SESSION_CHAIN_RAW_PREFIX";
    const char *require = "DS4_METAL_REQUIRE_SESSION_CHAIN_RAW_PREFIX";
    if (raw_prefix && (setenv(require, "1", 1) ||
                       unsetenv(disable))) return 1;
    enum { SOURCE_REPEATS = 1024, MAX_PREFIX = 8191, LONG_HEADROOM = 64 };
    ds4_engine_options opt = {
        .model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = MAX_PREFIX + LONG_HEADROOM,
        .power_percent = 100, .warm_weights = true,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;
    const char *sentence = "A lighthouse keeper watches the sea and records each passing ship. ";
    const size_t len = strlen(sentence);
    char *text = malloc(len * SOURCE_REPEATS + 1u);
    ds4_tokens source = {0};
    const int vocab = ds4_engine_vocab_size(engine);
    float *a = malloc((size_t)vocab * sizeof(*a));
    float *b = malloc((size_t)vocab * sizeof(*b));
    int failures = 0, cases = 0;
    if (!text || !a || !b) { failures = 1; goto done; }
    for (unsigned i = 0; i < SOURCE_REPEATS; ++i) memcpy(text + i * len, sentence, len);
    text[len * SOURCE_REPEATS] = '\0';
    ds4_tokenize_text(engine, text, &source);
    if (source.len < MAX_PREFIX) { failures = 1; goto done; }

    const int complete_prefixes[] = { 64, 127, 128 };
    const int reject_prefixes[] = { 126, 127, 128, 130, 131, 132 };
#define RUN_CASE_CTX(prefix_, ctx_, burst_, limit_, rounds_, name_) do { \
        failures += run_case(engine, &source, (prefix_), (ctx_), (burst_), (limit_), \
                             (rounds_), (name_), a, b, vocab); \
        cases++; \
    } while (0)
#define RUN_CASE(prefix_, burst_, limit_, rounds_, name_) \
    RUN_CASE_CTX(prefix_, 512, burst_, limit_, rounds_, name_)
    for (size_t i = 0; i < sizeof(complete_prefixes) / sizeof(complete_prefixes[0]); ++i)
        RUN_CASE(complete_prefixes[i], 2, 2, 1, "complete-two-token-burst");
    for (size_t i = 0; i < sizeof(reject_prefixes) / sizeof(reject_prefixes[0]); ++i)
        RUN_CASE(reject_prefixes[i], 4, 1, 1, "reject-second-id");
    RUN_CASE(127, 4, 0, 1, "reject-seed");
    RUN_CASE(64, 16, 16, 1, "complete-sixteen-token-burst");
    RUN_CASE(510, 16, 16, 1, "context-headroom-two");
    RUN_CASE(124, 16, 16, 3, "repeated-bursts-across-compressor-boundary");

    /* A 256-row ring wraps while the session still has room to continue.
     * Repeated rejections cross both this physical wrap and ratio128 emit. */
    const char *raw_cap = getenv("DS4_METAL_GRAPH_RAW_CAP");
    char *saved_raw_cap = raw_cap ? strdup(raw_cap) : NULL;
    if ((raw_cap && !saved_raw_cap) || setenv("DS4_METAL_GRAPH_RAW_CAP", "256", 1)) {
        free(saved_raw_cap);
        failures++;
        goto done;
    }
    RUN_CASE(254, 4, 1, 3, "repeated-reject-around-raw-wrap");
    RUN_CASE(254, 16, 16, 2, "repeated-complete-around-raw-wrap");
    if (setenv("DS4_METAL_GRAPH_RAW_CAP", "129", 1)) failures++;
    else RUN_CASE(128, 4, 1, 2, "single-row-slack-repeated-reject");
    /* Reject the token whose harmless raw prefix wraps at 2048/8192. Keep
     * the ring large enough for the initial prompt to use batched prefill. */
    if (setenv("DS4_METAL_GRAPH_RAW_CAP", "2048", 1)) failures++;
    else RUN_CASE_CTX(2047, 2047 + LONG_HEADROOM, 4, 1, 1, "long-reject-at-raw-wrap");
    if (setenv("DS4_METAL_GRAPH_RAW_CAP", "8192", 1)) failures++;
    else RUN_CASE_CTX(8191, MAX_PREFIX + LONG_HEADROOM, 4, 1, 1, "long-reject-at-raw-wrap");
    if (raw_prefix) {
        if (setenv("DS4_METAL_GRAPH_RAW_CAP", "128", 1)) failures++;
        else {
            failures += run_raw_scope_guard(engine, &source);
            cases++;
            /* DISABLE dominates REQUIRE and retains the strict-ring fallback. */
            if (setenv(disable, "1", 1)) failures++;
            else RUN_CASE(126, 2, 2, 1, "strict-raw-ring-rollback");
            unsetenv(disable);
        }
    }
    if (saved_raw_cap) setenv("DS4_METAL_GRAPH_RAW_CAP", saved_raw_cap, 1);
    else unsetenv("DS4_METAL_GRAPH_RAW_CAP");
    free(saved_raw_cap);
#undef RUN_CASE
#undef RUN_CASE_CTX
    fprintf(stderr, "Metal chain state: %d cases, %d failed checks\n", cases, failures);

done:
    ds4_tokens_free(&source);
    free(text);
    free(a);
    free(b);
    ds4_engine_close(engine);
    return failures ? 1 : 0;
}
