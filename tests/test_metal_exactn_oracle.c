/* Model-backed correctness oracle for the Metal exact-N speculative path.
 *
 * The DSpark proposer is intentionally not involved: the test derives five
 * target-greedy tokens, injects controlled full/partial/EOS draft blocks, and
 * sends them through the same production verifier/commit function.  Each
 * resulting session is compared with ordinary one-token decode at three
 * levels: serialized KV/compressor state, continuation logits, and a short
 * greedy continuation.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/model.gguf make test-metal-exactn-oracle
 *
 * Running the binary directly without a model is a developer-friendly skip.
 * The Make target is the release gate: it sets DS4_TEST_REQUIRE_MODEL=1 and
 * therefore fails when the configured model is absent.
 */

#include "ds4.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST_CTX 512
#define TEST_PREFILL_CHUNK 128u
/* Partial-N deliberately falls back to the legacy five-row verifier, whose
 * routed union may need 5 * top6 slots before replaying the accepted prefix. */
#define TEST_EXPERT_CACHE 32u
#define MAX_DRAFT 5
#define CONTINUATION_TOKENS 4

/* This symbol exists only in the ds4 core compiled with DS4_TEST_HOOKS. */
int ds4_test_session_eval_exact_drafts(
        ds4_session *s,
        const int   *drafts,
        int          draft_n,
        int          eos_token,
        int         *accepted,
        int          accepted_cap,
        char        *err,
        size_t       errlen);

enum {
    EXACTN_UNION_ATTEMPTS = 0,
    EXACTN_UNION_FULL_ACCEPTS,
    EXACTN_UNION_FALLBACKS,
    EXACTN_UNION_PARTIAL_FALLBACKS,
    EXACTN_UNION_ERROR_FALLBACKS,
    EXACTN_UNION_PARTIAL_REPLAYS,
    EXACTN_UNION_VERIFY_SKIPS,
    EXACTN_UNION_BATCH_HEAD_ATTEMPTS,
    EXACTN_UNION_BATCH_HEAD_USES,
    EXACTN_UNION_BATCH_HEAD_FALLBACKS,
    EXACTN_UNION_COUNTER_COUNT
};

int ds4_test_session_exactn_union_stats(
        const ds4_session *s,
        uint64_t           out[EXACTN_UNION_COUNTER_COUNT]);

typedef struct {
    const char *name;
    int draft_n;
    int reject_at; /* -1 means every draft is target-greedy. */
    int eos_at;    /* -1 uses the model EOS; otherwise a synthetic EOS row. */
} exactn_case;

static void fail(const char *what, const char *case_name, const char *detail) {
    fprintf(stderr, "FAIL: %s case=%s%s%s\n",
            what,
            case_name ? case_name : "setup",
            detail && detail[0] ? ": " : "",
            detail && detail[0] ? detail : "");
    exit(1);
}

static void restore_snapshot(ds4_session *s,
                             const ds4_session_snapshot *snap,
                             const char *case_name) {
    char err[256] = "";
    if (ds4_session_load_snapshot(s, snap, err, sizeof(err)) != 0) {
        fail("snapshot restore", case_name, err);
    }
}

static void save_snapshot(ds4_session *s,
                          ds4_session_snapshot *snap,
                          const char *case_name) {
    char err[256] = "";
    if (ds4_session_save_snapshot(s, snap, err, sizeof(err)) != 0) {
        fail("snapshot save", case_name, err);
    }
}

static void eval_tokens(ds4_session *s,
                        const int *tokens,
                        int count,
                        const char *case_name) {
    char err[256] = "";
    for (int i = 0; i < count; i++) {
        if (ds4_session_eval(s, tokens[i], err, sizeof(err)) != 0) {
            char detail[320];
            snprintf(detail, sizeof(detail), "row=%d token=%d err=%s",
                     i, tokens[i], err);
            fail("sequential decode", case_name, detail);
        }
    }
}

static int copy_logits(ds4_session *s, float *out, int vocab,
                       const char *case_name) {
    const int copied = ds4_session_copy_logits(s, out, vocab);
    if (copied != vocab) {
        char detail[96];
        snprintf(detail, sizeof(detail), "copied=%d vocab=%d", copied, vocab);
        fail("logits read", case_name, detail);
    }
    return copied;
}

static int lowest_finite_token(const float *logits, int vocab,
                               int excluded_a, int excluded_b) {
    int token = -1;
    float value = FLT_MAX;
    for (int i = 0; i < vocab; i++) {
        if (i == excluded_a || i == excluded_b) continue;
        if (token < 0 || logits[i] < value) {
            token = i;
            value = logits[i];
        }
    }
    return token;
}

static int greedy_continuation(ds4_session *s, int eos,
                               int out[CONTINUATION_TOKENS],
                               const char *case_name) {
    char err[256] = "";
    int n = 0;
    while (n < CONTINUATION_TOKENS) {
        const int token = ds4_session_argmax(s);
        if (token < 0) fail("continuation argmax", case_name, "negative token");
        out[n++] = token;
        if (token == eos || n == CONTINUATION_TOKENS) break;
        if (ds4_session_eval(s, token, err, sizeof(err)) != 0) {
            fail("continuation decode", case_name, err);
        }
    }
    return n;
}

static void compare_logits(const float *expected, const float *actual,
                           int vocab, const char *case_name) {
    if (memcmp(expected, actual, (size_t)vocab * sizeof(*actual)) == 0) return;

    int differing = 0;
    int first = -1;
    float max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        if (memcmp(&expected[i], &actual[i], sizeof(actual[i])) != 0) {
            if (first < 0) first = i;
            differing++;
        }
        float delta = fabsf(expected[i] - actual[i]);
        if (!isfinite(delta)) delta = FLT_MAX;
        if (delta > max_abs) max_abs = delta;
    }
    char detail[256];
    snprintf(detail, sizeof(detail),
             "differing=%d first=%d expected=%g actual=%g max_abs=%g",
             differing, first,
             first >= 0 ? expected[first] : 0.0f,
             first >= 0 ? actual[first] : 0.0f,
             max_abs);
    fail("continuation logits mismatch", case_name, detail);
}

static void compare_snapshots(const ds4_session_snapshot *expected,
                              const ds4_session_snapshot *actual,
                              const char *case_name) {
    if (expected->len != actual->len) {
        char detail[160];
        snprintf(detail, sizeof(detail), "expected_bytes=%llu actual_bytes=%llu",
                 (unsigned long long)expected->len,
                 (unsigned long long)actual->len);
        fail("state snapshot length mismatch", case_name, detail);
    }
    if (memcmp(expected->ptr, actual->ptr, (size_t)expected->len) == 0) return;

    uint64_t first = 0;
    while (first < expected->len &&
           expected->ptr[first] == actual->ptr[first]) {
        first++;
    }
    char detail[192];
    snprintf(detail, sizeof(detail),
             "first_byte=%llu expected=0x%02x actual=0x%02x bytes=%llu",
             (unsigned long long)first,
             first < expected->len ? expected->ptr[first] : 0,
             first < actual->len ? actual->ptr[first] : 0,
             (unsigned long long)expected->len);
    fail("KV/compressor snapshot mismatch", case_name, detail);
}

static void compare_continuations(const int *expected, int expected_n,
                                  const int *actual, int actual_n,
                                  const char *case_name) {
    if (expected_n == actual_n &&
        memcmp(expected, actual, (size_t)expected_n * sizeof(*actual)) == 0) {
        return;
    }
    int first = 0;
    const int common = expected_n < actual_n ? expected_n : actual_n;
    while (first < common && expected[first] == actual[first]) first++;
    char detail[192];
    snprintf(detail, sizeof(detail),
             "expected_n=%d actual_n=%d first=%d expected=%d actual=%d",
             expected_n, actual_n, first,
             first < expected_n ? expected[first] : -1,
             first < actual_n ? actual[first] : -1);
    fail("greedy continuation mismatch", case_name, detail);
}

static void read_union_stats(
        const ds4_session *session,
        uint64_t           out[EXACTN_UNION_COUNTER_COUNT],
        const char        *case_name) {
    if (ds4_test_session_exactn_union_stats(session, out) != 0) {
        fail("exact-N union stats", case_name, "hook failed");
    }
}

static void check_union_stats_delta(
    const uint64_t before[EXACTN_UNION_COUNTER_COUNT],
    const uint64_t after[EXACTN_UNION_COUNTER_COUNT],
    const exactn_case *tc,
    bool expect_batch_head) {
    uint64_t expected[EXACTN_UNION_COUNTER_COUNT] = {0};
    /* EOS in the first draft row truncates the block to N=1 before exact-N
     * dispatch.  Every other full block (including middle EOS) is committed
     * by the union path; a deliberately wrong row restores once and exactly
     * replays the already verified prefix. */
    if (tc->eos_at != 0) {
        expected[EXACTN_UNION_ATTEMPTS] = 1;
        if (tc->reject_at >= 0) {
            expected[EXACTN_UNION_FALLBACKS] = 1;
            expected[EXACTN_UNION_PARTIAL_FALLBACKS] = 1;
            expected[EXACTN_UNION_PARTIAL_REPLAYS] = 1;
            expected[EXACTN_UNION_VERIFY_SKIPS] = 1;
        } else {
            expected[EXACTN_UNION_FULL_ACCEPTS] = 1;
        }
        if (expect_batch_head) {
            expected[EXACTN_UNION_BATCH_HEAD_ATTEMPTS] = 1;
            expected[EXACTN_UNION_BATCH_HEAD_USES] = 1;
        }
    }

    static const char *const names[EXACTN_UNION_COUNTER_COUNT] = {
        "attempt", "full", "fallback", "partial", "error",
        "partial-replay", "verify-skip", "batch-head-attempt",
        "batch-head-use", "batch-head-fallback"
    };
    for (int i = 0; i < EXACTN_UNION_COUNTER_COUNT; i++) {
        if (after[i] < before[i] || after[i] - before[i] != expected[i]) {
            char detail[192];
            snprintf(detail, sizeof(detail),
                     "%s before=%llu after=%llu expected_delta=%llu",
                     names[i],
                     (unsigned long long)before[i],
                     (unsigned long long)after[i],
                     (unsigned long long)expected[i]);
            fail("exact-N union stats delta", tc->name, detail);
        }
    }
}

static void run_case(ds4_session *session,
                     const ds4_session_snapshot *base,
                     int base_pos,
                     ds4_session_snapshot *expected_state,
                     ds4_session_snapshot *actual_state,
                     const int correct[MAX_DRAFT],
                     const int wrong[MAX_DRAFT],
                     int model_eos,
                      int vocab,
                      float *expected_logits,
                      float *actual_logits,
                      const exactn_case *tc,
                      bool expect_batch_head) {
    int drafts[MAX_DRAFT];
    memcpy(drafts, correct, (size_t)tc->draft_n * sizeof(drafts[0]));
    if (tc->reject_at >= 0) drafts[tc->reject_at] = wrong[tc->reject_at];

    int cycle_eos = model_eos;
    if (tc->eos_at >= 0) {
        cycle_eos = drafts[tc->eos_at];
    }

    /* Run the production path first.  A partial union result already proves
     * the complete correct prefix and skips the legacy batch verifier, so the
     * commit must stop exactly at the deliberately wrong row. */
    restore_snapshot(session, base, tc->name);
    int accepted[MAX_DRAFT] = {-1, -1, -1, -1, -1};
    char err[256] = "";
    uint64_t union_before[EXACTN_UNION_COUNTER_COUNT];
    uint64_t union_after[EXACTN_UNION_COUNTER_COUNT];
    read_union_stats(session, union_before, tc->name);
    const int accepted_n = ds4_test_session_eval_exact_drafts(
            session, drafts, tc->draft_n, cycle_eos,
            accepted, MAX_DRAFT, err, sizeof(err));
    if (accepted_n < 0) fail("exact-N cycle", tc->name, err);
    read_union_stats(session, union_after, tc->name);
    check_union_stats_delta(union_before, union_after, tc,
                            expect_batch_head);
    int full_expected = tc->draft_n;
    if (tc->eos_at >= 0) {
        /* Production truncates at the first occurrence of EOS.  The fixture
         * chooses a unique middle token below, but computing the first row
         * here keeps the oracle correct if a future prompt changes. */
        for (int i = 0; i < tc->draft_n; i++) {
            if (drafts[i] == cycle_eos) {
                full_expected = i + 1;
                break;
            }
        }
    }
    if (tc->reject_at < 0 && accepted_n != full_expected) {
        char detail[160];
        snprintf(detail, sizeof(detail), "expected=%d actual=%d err=%s",
                 full_expected, accepted_n, err);
        fail("accepted prefix length", tc->name, detail);
    }
    if (tc->reject_at >= 0 && accepted_n != tc->reject_at) {
        char detail[160];
        snprintf(detail, sizeof(detail),
                 "reject_at=%d accepted=%d err=%s",
                 tc->reject_at, accepted_n, err);
        fail("partial accepted wrong row", tc->name, detail);
    }
    for (int i = 0; i < accepted_n; i++) {
        if (accepted[i] != correct[i]) {
            char detail[128];
            snprintf(detail, sizeof(detail),
                     "row=%d expected=%d actual=%d",
                     i, correct[i], accepted[i]);
            fail("accepted token", tc->name, detail);
        }
    }
    if (ds4_session_pos(session) != base_pos + accepted_n) {
        fail("checkpoint length", tc->name, "unexpected committed position");
    }

    copy_logits(session, actual_logits, vocab, tc->name);
    save_snapshot(session, actual_state, tc->name);
    int actual_cont[CONTINUATION_TOKENS];
    const int actual_cont_n = tc->eos_at >= 0 ? 0 :
        greedy_continuation(session, model_eos, actual_cont, tc->name);

    restore_snapshot(session, base, tc->name);
    eval_tokens(session, correct, accepted_n, tc->name);
    copy_logits(session, expected_logits, vocab, tc->name);
    save_snapshot(session, expected_state, tc->name);
    int expected_cont[CONTINUATION_TOKENS];
    const int expected_cont_n = tc->eos_at >= 0 ? 0 :
        greedy_continuation(session, model_eos, expected_cont, tc->name);

    compare_logits(expected_logits, actual_logits, vocab, tc->name);
    compare_snapshots(expected_state, actual_state, tc->name);
    compare_continuations(expected_cont, expected_cont_n,
                          actual_cont, actual_cont_n, tc->name);
    fprintf(stderr,
            "PASS: exact-N oracle case=%s drafted=%d committed=%d "
            "snapshot_bytes=%llu continuation=%d\n",
            tc->name, tc->draft_n, accepted_n,
            (unsigned long long)actual_state->len, actual_cont_n);
}

int main(void) {
#ifndef __APPLE__
    fprintf(stderr, "test_metal_exactn_oracle: skipped (Metal requires macOS)\n");
    return 0;
#else
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0] || access(model, R_OK) != 0) {
        const char *required = getenv("DS4_TEST_REQUIRE_MODEL");
        fprintf(stderr,
                "test_metal_exactn_oracle: %s "
                "(set DS4_TEST_MODEL to a readable target GGUF)\n",
                required && required[0] && strcmp(required, "0") != 0
                    ? "FAIL: required model is missing"
                    : "skipped");
        return required && required[0] && strcmp(required, "0") != 0 ? 1 : 0;
    }

    const char *batch_head_env =
        getenv("DS4_TEST_METAL_EXACTN_BATCH_HEAD");
    const bool expect_batch_head =
        batch_head_env && batch_head_env[0] &&
        strcmp(batch_head_env, "0") != 0;

    setenv("DS4_TEST_METAL_EXACTN_ORACLE", "1", 1);
    setenv("DS4_METAL_DSPARK_EXACTN_UNION", "1", 1);
    setenv("DS4_METAL_DSPARK_EXACTN", "1", 1);
    if (expect_batch_head) {
        setenv("DS4_METAL_DSPARK_EXACTN_BATCH_HEAD", "1", 1);
        unsetenv("DS4_METAL_DISABLE_DSPARK_EXACTN_BATCH_HEAD");
    } else {
        unsetenv("DS4_METAL_DSPARK_EXACTN_BATCH_HEAD");
    }
    setenv("DS4_DSPARK_STATS", "1", 1);
    setenv("DS4_DSPARK_SSD_VERIFY_BLOCK_MAX", "5", 1);
    setenv("DS4_METAL_DSPARK_ACCEPTANCE_ONLY_VERIFY", "0", 1);
    setenv("DS4_METAL_DSPARK_EXACT2", "0", 1);

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_METAL;
    opt.context_size = TEST_CTX;
    opt.prefill_chunk = TEST_PREFILL_CHUNK;
    opt.ssd_streaming = true;
    opt.ssd_streaming_cold = true;
    opt.ssd_streaming_cache_experts = TEST_EXPERT_CACHE;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        fail("engine open", NULL, model);
    }

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(
            engine,
            NULL,
            "Continue this sequence concisely: 1, 2, 3, 4, 5,",
            DS4_THINK_NONE,
            &prompt);
    if (prompt.len <= 0 || prompt.len >= TEST_CTX - 16) {
        fail("prompt tokenization", NULL, "invalid prompt length");
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, TEST_CTX) != 0 || !session) {
        fail("session create", NULL, "failed");
    }
    char err[256] = "";
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fail("session prefill", NULL, err);
    }

    ds4_session_snapshot base = {0};
    ds4_session_snapshot expected_state = {0};
    ds4_session_snapshot actual_state = {0};
    save_snapshot(session, &base, "setup");

    const int vocab = ds4_engine_vocab_size(engine);
    const int model_eos = ds4_token_eos(engine);
    float *probe_logits = malloc((size_t)vocab * sizeof(*probe_logits));
    float *expected_logits = malloc((size_t)vocab * sizeof(*expected_logits));
    float *actual_logits = malloc((size_t)vocab * sizeof(*actual_logits));
    if (!probe_logits || !expected_logits || !actual_logits) {
        fail("host allocation", NULL, "logits");
    }

    int correct[MAX_DRAFT];
    int wrong[MAX_DRAFT];
    for (int i = 0; i < MAX_DRAFT; i++) {
        copy_logits(session, probe_logits, vocab, "draft derivation");
        correct[i] = ds4_session_argmax(session);
        if (correct[i] < 0 || correct[i] == model_eos) {
            fail("draft derivation", NULL, "unexpected early EOS");
        }
        wrong[i] = lowest_finite_token(probe_logits, vocab,
                                       correct[i], model_eos);
        if (wrong[i] < 0) fail("draft derivation", NULL, "no rejection token");
        eval_tokens(session, &correct[i], 1, "draft derivation");
    }
    restore_snapshot(session, &base, "setup");

    int eos_middle_at = -1;
    for (int candidate = 2; candidate < MAX_DRAFT && eos_middle_at < 0;
         candidate++) {
        bool unique = true;
        for (int previous = 0; previous < candidate; previous++) {
            if (correct[previous] == correct[candidate]) {
                unique = false;
                break;
            }
        }
        if (unique) eos_middle_at = candidate;
    }
    for (int candidate = 1; candidate < 2 && eos_middle_at < 0; candidate++) {
        if (correct[candidate] != correct[0]) eos_middle_at = candidate;
    }
    if (eos_middle_at < 1) {
        fail("EOS fixture derivation", NULL,
             "no target token unique after the first draft row");
    }

    const exactn_case cases[] = {
        {"full-2",       2, -1, -1},
        {"full-3",       3, -1, -1},
        {"full-4",       4, -1, -1},
        /* Five drafts plus the already-generated target token exercise the
         * requested six-token speculative cycle. */
        {"full-5",       5, -1, -1},
        {"partial-5-at1", 5,  1, -1},
        {"partial-5-at2", 5,  2, -1},
        {"partial-5-at3", 5,  3, -1},
        {"partial-5-at4", 5,  4, -1},
        {"eos-first",     5, -1,  0},
        {"eos-middle",    5, -1, eos_middle_at},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        run_case(session, &base, prompt.len, &expected_state, &actual_state,
                 correct, wrong, model_eos, vocab,
                 expected_logits, actual_logits, &cases[i],
                 expect_batch_head);
    }

    fprintf(stderr,
            "test_metal_exactn_oracle PASS cases=%zu N=2..5 "
            "(six-token cycle at N=5) partial_prefixes=1..4 "
            "eos=first,middle batch_head=%s\n",
            sizeof(cases) / sizeof(cases[0]),
            expect_batch_head ? "required" : "disabled");

    free(actual_logits);
    free(expected_logits);
    free(probe_logits);
    ds4_session_snapshot_free(&actual_state);
    ds4_session_snapshot_free(&expected_state);
    ds4_session_snapshot_free(&base);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
#endif
}
