/* Tier-2 MULTISEQ decode gate: co-scheduling neutrality + the first
 * aggregate-throughput measurement (increment 3's headline).
 *
 * MODEL-DEPENDENT: run manually on the GB10 via `make cuda-multiseq-gate`
 * (memory discipline in the Makefile target) — NOT part of `make test`.
 *
 * WHAT THIS GATE ASSERTS (and what it deliberately does not)
 * ---------------------------------------------------------
 * The batched multi-session sweep (gpu_graph_encode_layer_batch) is a
 * DIFFERENT KERNEL PATH from classic single-token decode
 * (gpu_graph_encode_decode_layer): different attention/indexer tiers,
 * different GEMM shapes, different accumulation order.  The two paths are
 * NOT bit-identical, and that predates this code: the batch sweep already
 * diverged from classic decode on the increment-2 baseline (2c16d73) at the
 * same step and the same token ids the 1-row multiseq path produces
 * (control harness temp/t2i3_pathctl.c, which uses only the pre-existing
 * gpu_graph_verify_suffix_tops: "batch-sweep DIVERGES from classic decode at
 * step 18 (batch 979 vs classic 339)" — byte-identical verdict on the
 * baseline and on this tree, and the same step/tokens this gate's N=1 run
 * reports).  Greedy argmax turns any near-tie into a different token, and
 * the streams walk apart from there.
 *
 * So "multiseq stream == classic-decode stream" is NOT a property this
 * engine has, and gating on it would gate on the wrong thing.  The property
 * that matters for multi-session serving — and the HARD gate here — is
 * CO-SCHEDULING NEUTRALITY:
 *
 *   a session's emitted token stream must not depend on WHICH other sessions
 *   share its batch, nor on how many: bank k's stream at N=2 must be
 *   token-identical to bank k's stream at N=3.
 *
 * (N=1 is excluded from that comparison by construction: a 1-row batch
 * dispatches the engine's dedicated single-row kernel tiers, so N=1 vs N>=2
 * compares two kernel paths again rather than co-scheduling.)
 *
 * Bit-level bank addressing/isolation (bank-slot swap invariance,
 * populate-order invariance, idle-bank bytes untouched) is the frontier
 * gate's job (tests/multiseq_frontier_gate.c); this gate is its end-to-end
 * token-stream complement.
 *
 * WHAT THIS GATE IS BLIND TO (do not add it here — it is S6's job):
 * per-row POSITIONS.  reqs[k].pos below is a function of the bank k and the
 * step j, never of the batch width n, so bank k occupies the same rows at
 * the same positions at N=2 and at N=3.  A kernel that ignored positions[]
 * entirely would apply the SAME wrong rotation at both widths, and the
 * neutrality comparison would still pass — co-scheduling neutrality is
 * width-invariant by construction in this instantiation.  The frontier
 * gate's S6 (batchmate-position independence) is the check with teeth for
 * the position path.
 *
 * DSPARK: `PULSAR_GATE_NO_DSPARK=1` opens the engine with speculation disabled
 * (the pulsar-bench/pulsar-eval/agent and `pulsar-server --no-dspark` config, and a
 * different allocation shape).  The driver must work there — it is plain
 * decode by contract at n >= 2 and must not depend on the speculation
 * machinery having been initialized.  Run via `make cuda-multiseq-gate-nodspark`.
 *
 * INFORMATIONAL (reported, never fatal): each bank's stream vs the same
 * prompt decoded solo through classic decode — first divergence step and the
 * logit gap between the two candidates at that step, measured on the live
 * row of the real run.  A small gap is the near-tie signature of the
 * two-path numerics above; the gap is printed so a systematic divergence
 * (large gap, or divergence at step 1) stays visible instead of hidden.
 *
 * THROUGHPUT (informational): aggregate tokens/sec over the timed multi loop
 * = N*STEPS/elapsed, vs the timed classic baseline.  Server wiring
 * (increment 4) adds scheduling overhead on top.
 *
 * usage: PULSAR_MSEQ_BANKS=3 ./tests/multiseq_decode_gate MODEL [MAXN] [STEPS]
 *        (from the repo root — reads tests/long_context_story_prompt.txt)
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GATE_MAX_N 8
#define GATE_MAX_STEPS 2048

static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;

/* Distinct prompts: different offsets AND different lengths so co-scheduled
 * banks sit at unrelated positions from step one. */
static const int g_prompt_off[GATE_MAX_N] = {0, 401, 907, 233, 601, 811, 101, 499};
static const int g_prompt_len[GATE_MAX_N] = {130, 258, 511, 187, 342, 419, 275, 158};

#define CHECK(cond, ...) do { \
        if (!(cond)) { \
            fprintf(stderr, "MULTISEQ GATE FAIL: " __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            g_fail = 1; \
        } \
    } while (0)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp);
    buf[n] = '\0';
    if (len_out) *len_out = (size_t)n;
    return buf;
}

static bool make_prompt(int k, pulsar_tokens *p) {
    memset(p, 0, sizeof(*p));
    const int off = g_prompt_off[k], len = g_prompt_len[k];
    if (off + len > g_toks.len) return false;
    p->v = (int *)malloc((size_t)len * sizeof(int));
    if (!p->v) return false;
    memcpy(p->v, g_toks.v + off, (size_t)len * sizeof(int));
    p->len = p->cap = len;
    return true;
}

/* Classic greedy solo reference for prompt k: prefill + STEPS plain decode
 * steps; stream[j] = argmax emitted at position len+j (stream[0] is the
 * prefill continuation).  *secs = the timed classic decode loop. */
static bool solo_stream(int k, int steps, int *stream, double *secs) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_tokens p;
    bool ok = make_prompt(k, &p);
    char err[256];
    if (ok && pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
        fprintf(stderr, "solo sync failed: %s\n", err);
        ok = false;
    }
    if (ok) {
        stream[0] = pulsar_session_argmax(s);
        const double t0 = now_s();
        for (int j = 1; ok && j <= steps; j++) {
            if (pulsar_session_eval(s, stream[j - 1], err, sizeof(err)) != 0) {
                fprintf(stderr, "solo eval failed at %d: %s\n", j, err);
                ok = false;
                break;
            }
            stream[j] = pulsar_session_argmax(s);
        }
        if (secs) *secs = now_s() - t0;
    }
    pulsar_tokens_free(&p);
    pulsar_session_free(s);
    return ok;
}

/* One multi run at batch width n: populate banks 0..n-1 through the classic
 * per-bank path, then run STEPS self-fed greedy steps through the engine
 * entry.  streams[k] gets steps+1 tokens (aligned with solo_stream).
 *
 * solo (optional): the first step at which bank k's token differs from
 * solo[k] is recorded in flip_step[k], and the LIVE logits row of that step
 * yields flip_gap[k] = logit(multi's pick) - logit(solo's pick) — measured in
 * the real run, at the real batch composition, with no re-decode. */
/* use_mixed: route the batched step through pulsar_session_decode_mixed (the plan-34
 * heap-descriptor entry the server worker now uses) instead of
 * pulsar_session_decode_multiseq. Increment-1 refactor must be byte-identical.
 * first_logits_out (optional, n*vocab floats): the STEP-1 logits, copied out for
 * a byte-level cross-entry comparison. */
static bool multi_run(int n, int steps, int **streams, int *const *solo,
                      int *flip_step, float *flip_gap, double *secs,
                      bool use_mixed, float *first_logits_out) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    if ((int)gpu_graph_bank_pool_count(g) < n) {
        fprintf(stderr, "pool too small: %u < %d (set PULSAR_MSEQ_BANKS)\n",
                gpu_graph_bank_pool_count(g), n);
        pulsar_session_free(s);
        return false;
    }
    const int vocab = (int)PULSAR_N_VOCAB;   /* the engine's logits row width */
    char err[256];
    bool ok = true;
    for (int k = 0; k < n; k++) {
        if (flip_step) flip_step[k] = -1;
        /* Repoint the graph's views at bank k, prefill THIS bank from zero
         * through the classic path (invalidate: no prefix reuse across
         * banks), and capture its frontier into the bank's ms counters. */
        if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) { ok = false; break; }
        pulsar_session_invalidate(s);
        pulsar_tokens p;
        ok = make_prompt(k, &p);
        if (ok && pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
            fprintf(stderr, "populate bank %d failed: %s\n", k, err);
            ok = false;
        }
        if (ok) {
            gpu_graph_bank_counters_capture(g, (uint32_t)k);
            streams[k][0] = pulsar_session_argmax(s);
        }
        pulsar_tokens_free(&p);
    }
    float *logits = ok ? (float *)malloc((size_t)n * vocab * sizeof(float)) : NULL;
    if (ok && !logits) ok = false;
    if (ok) {
        const double t0 = now_s();
        for (int j = 1; ok && j <= steps; j++) {
            pulsar_multiseq_req reqs[GATE_MAX_N];
            for (int k = 0; k < n; k++) {
                reqs[k].bank = (uint32_t)k;
                reqs[k].pos = g_prompt_len[k] + (j - 1);
                reqs[k].token = streams[k][j - 1];
            }
            const int rc = use_mixed
                ? pulsar_session_decode_mixed(s, reqs, (uint32_t)n,
                                           logits, n * vocab, NULL, 0u, err, sizeof(err))
                : pulsar_session_decode_multiseq(s, reqs, (uint32_t)n,
                                              logits, n * vocab, err, sizeof(err));
            if (rc != 0) {
                fprintf(stderr, "%s step %d failed (rc=%d): %s\n",
                        use_mixed ? "mixed" : "multiseq", j, rc, err);
                ok = false;
                break;
            }
            if (j == 1 && first_logits_out)
                memcpy(first_logits_out, logits, (size_t)n * vocab * sizeof(float));
            for (int k = 0; k < n; k++) {
                const float *row = logits + (size_t)k * vocab;
                streams[k][j] = (int)argmax_f32(row, (uint64_t)vocab);
                if (solo && flip_step && flip_step[k] < 0 &&
                    streams[k][j] != solo[k][j]) {
                    flip_step[k] = j;
                    if (flip_gap) flip_gap[k] = row[streams[k][j]] - row[solo[k][j]];
                }
            }
        }
        if (secs) *secs = now_s() - t0;
    }
    free(logits);
    pulsar_session_free(s);
    return ok;
}

/* HARD GATE: the classic single-bank entries must FAIL LOUD after a multiseq
 * step, not silently corrupt.
 *
 * After a multiseq step the graph's scalar frontier counters hold a
 * cross-bank SUPERSET, not any single bank's truth.  pulsar_session_eval decodes
 * against those scalars unconditionally (it never consulted checkpoint_valid,
 * so clearing that flag never covered this path): it would emit its
 * compressor row at the superset index and attend over the rows below it — a
 * previous tenant's bytes when the live bank's frontier is lower.  Wrong
 * logits, silently.  pulsar_session_decode_multiseq is public, so a server is
 * exactly the caller that hits this.
 *
 * Asserted here: (1) eval fails after a multiseq step, (2) the speculative
 * entries fail too (same counters), (3) a fresh pulsar_session_sync — the
 * documented escape hatch, and one available to public callers — clears the
 * condition and eval works again. */
static bool check_stale_classic_fails_loud(void) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    if ((int)gpu_graph_bank_pool_count(g) < 2) {
        printf("STALE-GUARD: skipped (pool %u < 2)\n", gpu_graph_bank_pool_count(g));
        pulsar_session_free(s);
        return true;
    }
    const int vocab = pulsar_engine_logits_width(g_e);
    char err[256];
    bool ok = true;
    int last[2] = {0, 0};
    for (int k = 0; ok && k < 2; k++) {
        if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) { ok = false; break; }
        pulsar_session_invalidate(s);
        pulsar_tokens p;
        ok = make_prompt(k, &p);
        if (ok && pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
            fprintf(stderr, "stale-guard populate bank %d failed: %s\n", k, err);
            ok = false;
        }
        if (ok) {
            gpu_graph_bank_counters_capture(g, (uint32_t)k);
            last[k] = pulsar_session_argmax(s);
        }
        pulsar_tokens_free(&p);
    }
    float *logits = ok ? (float *)malloc((size_t)2 * vocab * sizeof(float)) : NULL;
    if (ok && !logits) ok = false;
    if (ok) {
        /* Sanity: eval works BEFORE any multiseq step (so a later failure is
         * the guard firing, not an unrelated broken session). */
        if (pulsar_session_eval(s, last[1], err, sizeof(err)) != 0) {
            CHECK(0, "stale-guard: eval failed BEFORE any multiseq step (%s)", err);
            ok = false;
        }
    }
    if (ok) {
        /* Re-establish bank 1's frontier: the eval above advanced it. */
        pulsar_tokens p;
        ok = make_prompt(1, &p);
        if (ok && pulsar_session_sync(s, &p, err, sizeof(err)) != 0) ok = false;
        if (ok) gpu_graph_bank_counters_capture(g, 1u);
        pulsar_tokens_free(&p);
        if (!ok) CHECK(0, "stale-guard: bank 1 re-sync failed");
    }
    if (ok) {
        pulsar_multiseq_req reqs[2];
        for (int k = 0; k < 2; k++) {
            reqs[k].bank = (uint32_t)k;
            reqs[k].pos = g_prompt_len[k];
            reqs[k].token = last[k];
        }
        const int rc = pulsar_session_decode_multiseq(s, reqs, 2, logits,
                                                   2 * vocab, err, sizeof(err));
        if (rc != 0) {
            CHECK(0, "stale-guard: multiseq step failed (rc=%d): %s", rc, err);
            ok = false;
        }
    }
    if (ok) {
        err[0] = '\0';
        const int rc = pulsar_session_eval(s, last[0], err, sizeof(err));
        CHECK(rc != 0,
              "STALE CLASSIC STATE NOT CAUGHT: pulsar_session_eval SUCCEEDED after "
              "a multiseq step — it decoded against cross-bank superset frontier "
              "counters and silently corrupted KV");
        if (rc != 0) printf("STALE-GUARD: eval after multiseq fails loud OK (%s)\n", err);

        err[0] = '\0';
        int acc[4];
        const int rcs = pulsar_session_eval_speculative_block(s, last[0], 4, -1, acc,
                                                           4, err, sizeof(err));
        CHECK(rcs <= 0,
              "STALE CLASSIC STATE NOT CAUGHT: pulsar_session_eval_speculative_block "
              "SUCCEEDED after a multiseq step (returned %d)", rcs);
        if (rcs <= 0) printf("STALE-GUARD: spec block eval after multiseq fails loud OK\n");
    }
    if (ok) {
        /* The escape hatch: a fresh sync rebuilds per-bank state (reset zeroes
         * the counters) and classic decode must work again. */
        pulsar_tokens p;
        bool ok2 = make_prompt(0, &p);
        if (ok2 && pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
            CHECK(0, "stale-guard: re-sync after multiseq failed: %s", err);
            ok2 = false;
        }
        if (ok2) {
            CHECK(pulsar_session_eval(s, pulsar_session_argmax(s), err, sizeof(err)) == 0,
                  "stale-guard: eval still refused after a re-sync — the "
                  "documented recovery path does not clear the condition (%s)",
                  err);
            printf("STALE-GUARD: re-sync clears the condition, classic eval "
                   "works again OK\n");
        }
        pulsar_tokens_free(&p);
    }
    free(logits);
    pulsar_session_free(s);
    return ok;
}

/* Tier-1 SPEC lane timing (the mode-switch crossover measurement — plan
 * §2.2/§2.3: "MEASURE the N=2 boundary, do NOT hardcode 3").  This is the lane
 * the scheduler picks at n<=2: n INDEPENDENT sessions (each its own graph — the
 * Tier-1 time-slice model, NOT banks), fused DSpark speculation, round-robin
 * one generate_speculative quantum per session until each has emitted >= steps
 * tokens.  Timed apples-to-apples with multi_run's batched lane (aggregate =
 * emitted/elapsed) so the crossover is measured on THIS v0.2.3 build rather
 * than assumed.  Greedy (temp=0): exact-under-argmax spec, so each session's
 * stream matches its solo reference.  Returns false (caller skips + reports)
 * if the engine has no drafter — the PULSAR_GATE_NO_DSPARK / --no-dspark config,
 * where the scheduler has no spec lane and batches at every width. */
static bool spec_timeslice_run(int n, int steps, double *secs,
                               uint64_t *out_emitted) {
    if (!pulsar_engine_has_dspark(g_e)) return false;
    pulsar_session *ss[GATE_MAX_N];
    memset(ss, 0, sizeof(ss));
    char err[256];
    bool ok = true;
    for (int k = 0; k < n && ok; k++) {
        if (pulsar_session_create(&ss[k], g_e, 4096) != 0) { ok = false; break; }
        pulsar_tokens p;
        ok = make_prompt(k, &p);
        if (ok && pulsar_session_sync(ss[k], &p, err, sizeof(err)) != 0) {
            fprintf(stderr, "spec populate %d failed: %s\n", k, err);
            ok = false;
        }
        pulsar_tokens_free(&p);
    }
    uint64_t emitted_total = 0;
    if (ok) {
        int emitted[GATE_MAX_N];
        memset(emitted, 0, sizeof(emitted));
        int acc[32];
        uint64_t rng = 0x2545F4914F6CDD1Dull;
        const int eos = pulsar_token_eos(g_e);
        const double t0 = now_s();
        bool more = true;
        while (ok && more) {
            more = false;
            for (int k = 0; k < n && ok; k++) {
                if (emitted[k] >= steps) continue;
                const int ntok = pulsar_session_generate_speculative(
                        ss[k], 0.0f, 0, 1.0f, 0.0f, &rng,
                        steps - emitted[k], eos, acc,
                        (int)(sizeof(acc) / sizeof(acc[0])), err, sizeof(err));
                if (ntok < 0) {
                    fprintf(stderr, "spec %d failed: %s\n", k, err);
                    ok = false;
                    break;
                }
                if (ntok == 0) { emitted[k] = steps; continue; } /* eos/stall: retire */
                emitted[k] += ntok;
                emitted_total += (uint64_t)ntok;
                if (emitted[k] < steps) more = true;
            }
        }
        if (secs) *secs = now_s() - t0;
    }
    for (int k = 0; k < n; k++) if (ss[k]) pulsar_session_free(ss[k]);
    if (out_emitted) *out_emitted = emitted_total;
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL [MAXN] [STEPS]\n", argv[0]);
        return 2;
    }
    const int maxn = argc > 2 ? atoi(argv[2]) : 3;
    const int steps = argc > 3 ? atoi(argv[3]) : 512;
    if (maxn < 1 || maxn > GATE_MAX_N || steps < 1 || steps > GATE_MAX_STEPS) {
        fprintf(stderr, "bad MAXN/STEPS\n");
        return 2;
    }

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    /* PULSAR_GATE_NO_DSPARK: run the whole gate with speculation DISABLED, the
     * way pulsar-bench/pulsar-eval/pulsar-agent and `pulsar-server --no-dspark` open the
     * engine.  This is a real serving config (and the one increment 4's
     * server will plausibly run), and it is a DIFFERENT allocation shape: the
     * DSpark graph state is never allocated.  The driver used to reject every
     * step here because the shared multi-row logits slab was allocated only
     * as a side effect of DSpark init — a config no gate covered, because
     * both gates default to the drafter-merged FRONTIER_MODEL.  Read once at
     * startup (this is the test's own config, not a hot path). */
    if (getenv("PULSAR_GATE_NO_DSPARK") != NULL) {
        opt.dspark_disable = true;
        printf("CONFIG: DSpark DISABLED (dspark_disable=1) — the driver must "
               "work with no speculation machinery allocated\n");
    }
    if (pulsar_engine_open(&g_e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (getenv("PULSAR_GATE_NO_DSPARK") != NULL && pulsar_engine_has_dspark(g_e)) {
        fprintf(stderr, "MULTISEQ GATE FAIL: PULSAR_GATE_NO_DSPARK set but the "
                        "engine still reports a drafter — the no-dspark case "
                        "is not actually being exercised\n");
        return 1;
    }

    size_t text_len = 0;
    char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
    if (!text) { fprintf(stderr, "prompt file read failed\n"); return 1; }
    memset(&g_toks, 0, sizeof(g_toks));
    pulsar_tokenize_text(g_e, text, &g_toks);
    free(text);
    int need = 0;
    for (int k = 0; k < maxn; k++) {
        if (g_prompt_off[k] + g_prompt_len[k] > need) need = g_prompt_off[k] + g_prompt_len[k];
    }
    if (g_toks.len < need) { fprintf(stderr, "prompt too short (%d < %d)\n", g_toks.len, need); return 1; }

    /* Classic-decode solo references + the single-session baseline rate. */
    int *solo[GATE_MAX_N];
    double solo_secs[GATE_MAX_N];
    for (int k = 0; k < maxn; k++) {
        solo[k] = (int *)malloc((size_t)(steps + 1) * sizeof(int));
        if (!solo[k] || !solo_stream(k, steps, solo[k], &solo_secs[k])) {
            fprintf(stderr, "solo reference %d failed\n", k);
            return 1;
        }
        printf("solo[%d]: prompt off=%d len=%d, %d steps in %.1fs (%.2f tok/s classic plain)\n",
               k, g_prompt_off[k], g_prompt_len[k], steps, solo_secs[k],
               (double)steps / solo_secs[k]);
    }

    /* Multi runs at N = 1..maxn; every stream is kept for the cross-N gate. */
    int *multi[GATE_MAX_N][GATE_MAX_N];   /* [n-1][bank] */
    bool have[GATE_MAX_N];
    const int vocab_w = (int)PULSAR_N_VOCAB;
    float *ref_l1[GATE_MAX_N];            /* [n-1] step-1 logits, multiseq entry */
    memset(multi, 0, sizeof(multi));
    memset(have, 0, sizeof(have));
    memset(ref_l1, 0, sizeof(ref_l1));
    for (int n = 1; n <= maxn; n++) {
        int flip_step[GATE_MAX_N];
        float flip_gap[GATE_MAX_N];
        memset(flip_gap, 0, sizeof(flip_gap));
        for (int k = 0; k < n; k++) multi[n - 1][k] = (int *)malloc((size_t)(steps + 1) * sizeof(int));
        ref_l1[n - 1] = (float *)malloc((size_t)n * vocab_w * sizeof(float));
        double secs = 0.0;
        if (!multi_run(n, steps, multi[n - 1], solo, flip_step, flip_gap, &secs,
                       false, ref_l1[n - 1])) {
            CHECK(0, "N=%d: multi run failed", n);
            continue;
        }
        have[n - 1] = true;
        printf("N=%d: %d steps x %d sessions in %.1fs -> aggregate %.2f tok/s "
               "(%.2f tok/s/session)\n",
               n, steps, n, secs, (double)n * steps / secs, (double)steps / secs);
        /* INFORMATIONAL: divergence from classic decode (see the header). */
        for (int k = 0; k < n; k++) {
            if (flip_step[k] < 0) {
                printf("  N=%d bank %d: vs classic-solo IDENTICAL over %d tokens\n",
                       n, k, steps + 1);
            } else {
                printf("  N=%d bank %d: vs classic-solo diverges at step %d "
                       "(batch %d vs classic %d, logit gap %.6f) — two-path "
                       "numerics, informational\n",
                       n, k, flip_step[k],
                       multi[n - 1][k][flip_step[k]], solo[k][flip_step[k]],
                       (double)flip_gap[k]);
            }
        }
    }

    /* Tier-1 SPEC lane vs the batched lane above: the mode-switch crossover,
     * measured on this build (plan §2.2/§2.3 mandate — do NOT hardcode 3).
     * Informational, like the batched throughput lines; compare SPEC N=k
     * aggregate against the "N=k ... aggregate" batched line above.  Skipped
     * with no drafter (the scheduler then batches at every width). */
    if (pulsar_engine_has_dspark(g_e)) {
        for (int n = 1; n <= maxn; n++) {
            double secs = 0.0;
            uint64_t emitted = 0;
            if (spec_timeslice_run(n, steps, &secs, &emitted) && secs > 0.0) {
                printf("SPEC N=%d: %llu tokens (%d session(s) time-sliced) in "
                       "%.1fs -> aggregate %.2f tok/s (%.2f tok/s/session)\n",
                       n, (unsigned long long)emitted, n, secs,
                       (double)emitted / secs, (double)emitted / secs / n);
            } else {
                printf("SPEC N=%d: spec lane run failed/unavailable\n", n);
            }
        }
    } else {
        printf("SPEC lane: skipped (no drafter — scheduler batches at every "
               "width in this config)\n");
    }

    /* HARD GATE (plan-34 inc 1): the mixed-descriptor entry
     * (pulsar_session_decode_mixed, heap scratch) the server worker now routes
     * through must be BYTE-IDENTICAL to pulsar_session_decode_multiseq (fixed stack
     * scratch) for a decode-only batch. Re-run each width through the mixed entry
     * and assert (a) the full per-bank token STREAM is identical (all steps) and
     * (b) the STEP-1 logits are byte-identical (memcmp of the float rows). A
     * heap-scratch addressing/lifetime bug — the stated risk — surfaces here. */
    for (int n = 1; n <= maxn; n++) {
        if (!have[n - 1]) continue;
        int *mix[GATE_MAX_N];
        memset(mix, 0, sizeof(mix));
        for (int k = 0; k < n; k++) mix[k] = (int *)malloc((size_t)(steps + 1) * sizeof(int));
        float *mix_l1 = (float *)malloc((size_t)n * vocab_w * sizeof(float));
        double secs = 0.0;
        if (!multi_run(n, steps, mix, NULL, NULL, NULL, &secs, true, mix_l1)) {
            CHECK(0, "N=%d: mixed-entry run failed", n);
        } else {
            int sdiff = -1;
            for (int k = 0; k < n && sdiff < 0; k++)
                for (int j = 0; j <= steps; j++)
                    if (mix[k][j] != multi[n - 1][k][j]) { sdiff = j; break; }
            CHECK(sdiff < 0,
                  "MIXED ENTRY NOT byte-identical: N=%d token stream diverges from "
                  "the multiseq entry at step %d — heap descriptor addressing is wrong",
                  n, sdiff);
            const size_t lb = (size_t)n * vocab_w * sizeof(float);
            const int ldiff = memcmp(mix_l1, ref_l1[n - 1], lb);
            CHECK(ldiff == 0,
                  "MIXED ENTRY NOT byte-identical: N=%d STEP-1 logits differ from "
                  "the multiseq entry (memcmp != 0)", n);
            if (sdiff < 0 && ldiff == 0)
                printf("MIXED-ENTRY EQUIV: N=%d decode_mixed == decode_multiseq "
                       "(streams over %d tokens + step-1 logits byte-identical)\n",
                       n, steps + 1);
        }
        for (int k = 0; k < n; k++) free(mix[k]);
        free(mix_l1);
    }

    /* HARD GATE: co-scheduling neutrality — bank k's stream must not depend
     * on which/how many OTHER sessions share the batch (n >= 2; N=1 excluded,
     * it dispatches the single-row kernel tiers). */
    for (int n = 3; n <= maxn; n++) {
        if (!have[n - 1] || !have[1]) continue;
        for (int k = 0; k < 2; k++) {
            int diff = -1;
            for (int j = 0; j <= steps; j++) {
                if (multi[n - 1][k][j] != multi[1][k][j]) { diff = j; break; }
            }
            CHECK(diff < 0,
                  "co-scheduling NOT neutral: bank %d's stream at N=%d differs "
                  "from its stream at N=2 at step %d (%d vs %d) — a batchmate "
                  "changed another session's tokens",
                  k, n, diff,
                  diff >= 0 ? multi[n - 1][k][diff] : -1,
                  diff >= 0 ? multi[1][k][diff] : -1);
            if (diff < 0) {
                printf("CO-SCHED NEUTRALITY: bank %d identical at N=2 and N=%d "
                       "(%d tokens)\n", k, n, steps + 1);
            }
        }
    }

    if (!check_stale_classic_fails_loud()) CHECK(0, "stale-classic guard check failed to run");

    for (int k = 0; k < maxn; k++) free(solo[k]);
    for (int n = 0; n < GATE_MAX_N; n++) free(ref_l1[n]);
    for (int n = 0; n < GATE_MAX_N; n++)
        for (int k = 0; k < GATE_MAX_N; k++) free(multi[n][k]);
    pulsar_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "MULTISEQ DECODE GATE: FAIL\n"); return 1; }
    printf("MULTISEQ DECODE GATE: PASS\n");
    return 0;
}
