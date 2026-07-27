/* Tier-2 PATH A / Option F GATE: fused DSpark speculation on a BANK.
 *
 * MODEL-DEPENDENT: run manually on the GB10 via `make cuda-bank-spec-gate`
 * (a drafter-merged model — this gate is about speculation, so it needs one).
 * Hold temp/gpu.lock; sync + drop_caches + free-check first.
 *
 * Proves the Step-2 checkpoint the server restructure depends on, i.e. that
 * the unified bank model can run the spec lanes on banks:
 *
 *  TEST 1 (N=1 spec-on-a-bank == classic).  Prefill a NON-ZERO bank, restore
 *    it, and run fused DSpark speculation over it.  The greedy stream must be
 *    token-identical to the same prompt run through spec on a plain standalone
 *    single-session (the §2.4 gate: co-scheduling neutrality + run-to-run
 *    stability — NOT batched==classic).  Two independent runs must match.
 *
 *  TEST 2 (N=2 spec-time-slice, WARM rings, no cross-cool).  Two banks, two
 *    prompts, alternated one spec quantum at a time via
 *    pulsar_session_bank_state_save/restore.  Each bank's stream must match its
 *    solo reference (independence), AND each bank's per-bank accept rate must
 *    NOT be materially depressed vs solo — the whole reason Option F banks the
 *    drafter ring.  If the ring cross-cooled, accept would crater (86% -> ~5%).
 *
 *  TEST 3 (mseq_dirty cheap resume).  After a batched decode_multiseq step
 *    poisons the scalar frontier (pulsar_session_eval must fail), a bank-state
 *    restore clears it via counters_install (NOT a full pulsar_session_sync
 *    re-prefill) and classic/spec work resumes on that bank.
 *
 * usage: PULSAR_MSEQ_BANKS=2 ./tests/bank_spec_gate MODEL [STEPS]
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GATE_MAX_STEPS 1024

static pulsar_engine *g_e;
static pulsar_tokens g_toks;
static int g_fail;

static const int g_prompt_off[2] = {0, 401};
static const int g_prompt_len[2] = {130, 258};

#define CHECK(cond, ...) do { \
        if (!(cond)) { \
            fprintf(stderr, "BANK-SPEC GATE FAIL: " __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            g_fail = 1; \
        } \
    } while (0)

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

/* Greedy fused-spec stream on a session, capturing the emitted tokens (read
 * straight out of the advancing checkpoint) and the accept counters. */
static bool spec_stream(pulsar_session *s, int steps, int *stream,
                        uint64_t *accepted, uint64_t *drafted) {
    char err[256];
    int acc[32];
    uint64_t rng = 0x2545F4914F6CDD1Dull;
    const int eos = pulsar_token_eos(g_e);
    const uint64_t acc0 = s->spec.spec_accepted_tokens, drf0 = s->spec.spec_draft_tokens;
    int emitted = 0;
    while (emitted < steps) {
        const int base = s->checkpoint.len;
        const int ntok = pulsar_session_generate_speculative(
                s, 0.0f, 0, 1.0f, 0.0f, &rng, steps - emitted, eos, acc,
                (int)(sizeof(acc) / sizeof(acc[0])), err, sizeof(err));
        if (ntok < 0) { fprintf(stderr, "spec failed: %s\n", err); return false; }
        if (ntok == 0) break; /* eos/stall */
        for (int i = 0; i < ntok && emitted < steps; i++)
            stream[emitted++] = s->checkpoint.v[base + i];
    }
    if (accepted) *accepted = s->spec.spec_accepted_tokens - acc0;
    if (drafted)  *drafted  = s->spec.spec_draft_tokens - drf0;
    /* pad a short (eos) stream so comparisons stay in-bounds */
    for (int j = emitted; j < steps; j++) stream[j] = -1;
    return true;
}

/* Standalone single-session reference (NOT banked): the classic spec path. */
static bool solo_spec(int k, int steps, int *stream,
                      uint64_t *accepted, uint64_t *drafted) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_tokens p; char err[256];
    bool ok = make_prompt(k, &p);
    if (ok && pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
        fprintf(stderr, "solo sync failed: %s\n", err); ok = false;
    }
    if (ok) ok = spec_stream(s, steps, stream, accepted, drafted);
    pulsar_tokens_free(&p);
    pulsar_session_free(s);
    return ok;
}

/* Prefill bank k of the pooled session through the classic path and save its
 * per-bank host carry + frontier counters. */
static bool bank_prefill(pulsar_session *s, int k) {
    pulsar_gpu_graph *g = &s->graph;
    char err[256];
    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) return false;
    pulsar_session_invalidate(s);
    pulsar_tokens p;
    bool ok = make_prompt(k, &p);
    if (ok && pulsar_session_sync(s, &p, err, sizeof(err)) != 0) {
        fprintf(stderr, "bank %d prefill failed: %s\n", k, err); ok = false;
    }
    if (ok) pulsar_session_bank_state_save(s, (uint32_t)k);
    pulsar_tokens_free(&p);
    return ok;
}

static int streams_first_diff(const int *a, const int *b, int n) {
    for (int j = 0; j < n; j++) if (a[j] != b[j]) return j;
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [STEPS]\n", argv[0]); return 2; }
    const int steps = argc > 2 ? atoi(argv[2]) : 128;
    if (steps < 1 || steps > GATE_MAX_STEPS) { fprintf(stderr, "bad STEPS\n"); return 2; }

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&g_e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    if (!pulsar_engine_has_dspark(g_e)) {
        fprintf(stderr, "BANK-SPEC GATE: model has no drafter — this gate needs "
                        "speculation (use the drafter-merged FRONTIER_MODEL)\n");
        pulsar_engine_close(g_e);
        return 1;
    }

    size_t text_len = 0;
    char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
    if (!text) { fprintf(stderr, "prompt file read failed\n"); return 1; }
    memset(&g_toks, 0, sizeof(g_toks));
    pulsar_tokenize_text(g_e, text, &g_toks);
    free(text);

    /* Solo spec references (classic single-session path). */
    int solo[2][GATE_MAX_STEPS];
    uint64_t solo_acc[2], solo_drf[2];
    for (int k = 0; k < 2; k++) {
        if (!solo_spec(k, steps, solo[k], &solo_acc[k], &solo_drf[k])) {
            fprintf(stderr, "solo spec %d failed\n", k); return 1;
        }
        printf("solo[%d]: %d spec tokens, accept %.1f%% (%llu/%llu)\n", k, steps,
               solo_drf[k] ? 100.0 * (double)solo_acc[k] / (double)solo_drf[k] : 0.0,
               (unsigned long long)solo_acc[k], (unsigned long long)solo_drf[k]);
    }

    /* ---- TEST 1: N=1 spec on a NON-ZERO bank == classic, run-to-run stable ---- */
    for (int pass = 0; pass < 2; pass++) {
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, g_e, 4096) != 0) { CHECK(0, "T1 create"); break; }
        if ((int)gpu_graph_bank_pool_count(&s->graph) < 2) {
            printf("TEST1: SKIP (pool < 2; set DS4_MSEQ_BANKS>=2)\n");
            pulsar_session_free(s); break;
        }
        /* Populate both banks so bank 1's spec runs with a live bankmate present
         * in the pool (bank 0 occupied) — the co-scheduling-neutrality setup. */
        bool ok = bank_prefill(s, 0) && bank_prefill(s, 1);
        int bstream[GATE_MAX_STEPS];
        uint64_t bacc = 0, bdrf = 0;
        if (ok) ok = pulsar_session_bank_state_restore(s, 1);
        if (ok) ok = spec_stream(s, steps, bstream, &bacc, &bdrf);
        if (!ok) { CHECK(0, "T1 pass %d run failed", pass); pulsar_session_free(s); break; }
        const int d = streams_first_diff(bstream, solo[1], steps);
        CHECK(d < 0, "TEST1 pass %d: bank-1 spec stream diverges from classic solo "
                     "at step %d (bank %d vs solo %d)", pass, d,
                     d >= 0 ? bstream[d] : -1, d >= 0 ? solo[1][d] : -1);
        if (d < 0)
            printf("TEST1 pass %d: bank-1 spec == classic solo over %d tokens, "
                   "accept %.1f%% OK\n", pass, steps,
                   bdrf ? 100.0 * (double)bacc / (double)bdrf : 0.0);
        pulsar_session_free(s);
    }

    /* ---- TEST 2: N=2 spec-time-slice, warm rings, no cross-cool ---- */
    {
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, g_e, 4096) != 0) { CHECK(0, "T2 create"); goto done; }
        if ((int)gpu_graph_bank_pool_count(&s->graph) < 2) {
            printf("TEST2: SKIP (pool < 2)\n"); pulsar_session_free(s); goto done;
        }
        bool ok = bank_prefill(s, 0) && bank_prefill(s, 1);
        int ts[2][GATE_MAX_STEPS];
        int emitted[2] = {0, 0};
        uint64_t acc[2] = {0, 0}, drf[2] = {0, 0};
        char err[256];
        int accbuf[32];
        uint64_t rng[2] = {0x2545F4914F6CDD1Dull, 0x2545F4914F6CDD1Dull};
        const int eos = pulsar_token_eos(g_e);
        /* Alternate ONE generate_speculative quantum per bank, saving/restoring
         * the per-bank host carry across the switch (the time-slice lane). */
        while (ok && (emitted[0] < steps || emitted[1] < steps)) {
            for (int k = 0; k < 2 && ok; k++) {
                if (emitted[k] >= steps) continue;
                if (!pulsar_session_bank_state_restore(s, (uint32_t)k)) { ok = false; break; }
                const uint64_t a0 = s->spec.spec_accepted_tokens, d0 = s->spec.spec_draft_tokens;
                const int base = s->checkpoint.len;
                const int ntok = pulsar_session_generate_speculative(
                        s, 0.0f, 0, 1.0f, 0.0f, &rng[k], steps - emitted[k], eos,
                        accbuf, (int)(sizeof(accbuf)/sizeof(accbuf[0])), err, sizeof(err));
                if (ntok < 0) { fprintf(stderr, "T2 spec bank %d: %s\n", k, err); ok = false; break; }
                for (int i = 0; i < ntok && emitted[k] < steps; i++)
                    ts[k][emitted[k]++] = s->checkpoint.v[base + i];
                acc[k] += s->spec.spec_accepted_tokens - a0;
                drf[k] += s->spec.spec_draft_tokens - d0;
                if (ntok == 0) emitted[k] = steps; /* eos/stall: retire */
                pulsar_session_bank_state_save(s, (uint32_t)k);
            }
        }
        if (!ok) { CHECK(0, "TEST2 time-slice run failed"); pulsar_session_free(s); goto done; }
        for (int k = 0; k < 2; k++) {
            const int d = streams_first_diff(ts[k], solo[k], steps);
            CHECK(d < 0, "TEST2: bank %d time-sliced stream diverges from solo at "
                         "step %d (%d vs %d) — bankmate perturbed it", k, d,
                         d >= 0 ? ts[k][d] : -1, d >= 0 ? solo[k][d] : -1);
            const double a_ts = drf[k] ? 100.0*(double)acc[k]/(double)drf[k] : 0.0;
            const double a_solo = solo_drf[k] ? 100.0*(double)solo_acc[k]/(double)solo_drf[k] : 0.0;
            /* Warm-ring proof: time-sliced accept must track solo (a cross-cooled
             * ring would collapse it by >10x).  Allow a small slack. */
            CHECK(a_ts >= a_solo - 5.0,
                  "TEST2: bank %d accept rate DEPRESSED by time-slicing: %.1f%% vs "
                  "solo %.1f%% — the drafter ring cross-cooled (Option F broken)",
                  k, a_ts, a_solo);
            if (d < 0 && a_ts >= a_solo - 5.0)
                printf("TEST2: bank %d == solo (%d tokens), warm ring accept "
                       "%.1f%% vs solo %.1f%% OK\n", k, steps, a_ts, a_solo);
        }
        pulsar_session_free(s);
    }

    /* ---- TEST 3: mseq_dirty cheap resume (no full re-prefill) ---- */
    {
        pulsar_session *s = NULL;
        if (pulsar_session_create(&s, g_e, 4096) != 0) { CHECK(0, "T3 create"); goto done; }
        if ((int)gpu_graph_bank_pool_count(&s->graph) < 2) {
            printf("TEST3: SKIP (pool < 2)\n"); pulsar_session_free(s); goto done;
        }
        const int vocab = pulsar_engine_logits_width(g_e);
        char err[256];
        bool ok = bank_prefill(s, 0) && bank_prefill(s, 1);
        int first[2] = {0, 0};
        /* first decode token per bank = argmax after its prefill */
        for (int k = 0; k < 2 && ok; k++) {
            ok = pulsar_session_bank_state_restore(s, (uint32_t)k);
            if (ok) first[k] = pulsar_session_argmax(s);
        }
        float *logits = ok ? (float *)malloc((size_t)2 * vocab * sizeof(float)) : NULL;
        if (ok && logits) {
            pulsar_multiseq_req reqs[2];
            for (int k = 0; k < 2; k++) {
                reqs[k].bank = (uint32_t)k;
                reqs[k].pos = g_prompt_len[k];
                reqs[k].token = first[k];
            }
            const int rc = pulsar_session_decode_multiseq(s, reqs, 2, logits, 2*vocab,
                                                        err, sizeof(err));
            CHECK(rc == 0, "TEST3: batched step failed (rc=%d): %s", rc, err);
            if (rc == 0) {
                /* Poisoned: classic eval must fail loud now. */
                CHECK(s->mseq_dirty, "TEST3: mseq_dirty not set after batched step");
                CHECK(pulsar_session_eval(s, first[0], err, sizeof(err)) != 0,
                      "TEST3: eval SUCCEEDED while dirty (silent corruption)");
                /* Cheap resume: restore bank 0 (counters_install clears the
                 * poison — NO pulsar_session_sync re-prefill), reconcile the host
                 * checkpoint to the batched frontier (server appends the decoded
                 * token), and classic eval must work again. */
                ok = pulsar_session_bank_state_restore(s, 0);
                CHECK(ok && !s->mseq_dirty, "TEST3: restore did not clear mseq_dirty");
                token_vec_push(&s->checkpoint, first[0]); /* the token decoded at pos L0 */
                const int argmax_next = (int)argmax_f32(logits, (uint64_t)vocab);
                err[0] = '\0';
                CHECK(pulsar_session_eval(s, argmax_next, err, sizeof(err)) == 0,
                      "TEST3: eval STILL fails after cheap restore (no re-prefill): %s",
                      err);
                if (!s->mseq_dirty && ok)
                    printf("TEST3: batched-poison cleared by restore (no re-prefill), "
                           "classic eval resumes OK\n");
            }
        } else CHECK(0, "TEST3 setup failed");
        free(logits);
        pulsar_session_free(s);
    }

done:
    pulsar_tokens_free(&g_toks);
    pulsar_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "BANK-SPEC GATE: FAIL\n"); return 1; }
    printf("BANK-SPEC GATE: PASS\n");
    return 0;
}
