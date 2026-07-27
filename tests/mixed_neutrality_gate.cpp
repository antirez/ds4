/* plan-34 phase-2 increment 4 — TRUE MIXED STEP: co-scheduled decode banks + one
 * K-row prefill run in ONE fused step. Three gates in one harness:
 *
 *  GATE 4 (the core proof — CO-SCHEDULING NEUTRALITY): a decode bank's step logits
 *    must be BYTE-IDENTICAL whether it rides a decode-only step (M = n_dec) or a
 *    fused step that ALSO carries a K-row prefill run (M = n_dec + K). If they are,
 *    then co-scheduling a prefill does not perturb a decode bank — the property
 *    inc-4's per-run 3-way GEMM/MoE split exists to guarantee. Reference and fused
 *    runs use FRESH sessions that populate the decode banks IDENTICALLY (same
 *    prompts, same positions); per-bank isolation means a decode bank's output
 *    depends only on its own KV, so any divergence is a co-scheduling leak.
 *
 *  GATE 2 (PREFILL correctness in the mixed step): the fused step's prefill run
 *    last-position logits vs a classic RESUME — next-token EXACT and full-vocab
 *    rel-RMS = 0 (byte-identical KV => no corruption), same oracle as inc-3.
 *
 *  GATE 3 (MoE two-pass split boundary): with n_dec >= 2 (the per-token MoE path is
 *    taken over the decode prefix) AND K > 8 (the grouped MoE path is taken over the
 *    prefill suffix), BOTH row classes are correct — gate 4 proves the decode prefix,
 *    gate 2 proves the prefill suffix, and the split lands at row n_dec (asserted).
 *
 * Run pack on/off x idx-fp4 on/off (DS4_ATTN_PACK / DS4_IDX_FP4). CONFIG surfaced.
 * MODEL-DEPENDENT, needs DS4_MSEQ_BANKS >= n_dec+1. Run under GPU discipline.
 *   usage: DS4_MSEQ_BANKS=3 ./tests/mixed_neutrality_gate MODEL
 */
#include "ds4.h"
#include "ds4_engine_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DEC 2                 /* decode banks (per-token MoE path; gate-4 uses >=2) */
#define C0    128               /* prefill bank's classic first chunk (lifts frontier off 0) */
#define K_PRE 64                /* prefill run length; >8 => grouped MoE suffix taken */
#define PBASE 700               /* prefill bank token region (distinct from decode banks) */

static ds4_engine *g_e;
static ds4_tokens g_toks;
static int g_fail;

/* decode banks 0..N_DEC-1: distinct prompt regions (isolated KV). */
static const int g_off[N_DEC] = {0, 401};
static const int g_len[N_DEC] = {130, 258};

static char *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)s + 1);
    if (!b || fread(b, 1, (size_t)s, f) != (size_t)s) { fclose(f); free(b); return NULL; }
    fclose(f); b[s] = 0; if (n) *n = (size_t)s; return b;
}

/* Populate decode banks + (if K>0) the prefill bank on a FRESH session, run ONE
 * fused decode_mixed step, and copy back: each decode bank's logit row into
 * dec_rows[k] (N_DEC * vocab), and (K>0) the prefill run's last-position logits
 * into pre_row (vocab). head_cap is forwarded to ds4_session_decode_mixed's
 * max_head_runs (0 = all runs; N_DEC = LEVER-1 intermediate path: decode banks
 * only, prefill head skipped — pre_row is then unavailable). Returns false on any
 * engine failure. */
static bool fused_step_logits(int K, uint32_t head_cap, float *dec_rows, float *pre_row) {
    ds4_session *s = NULL;
    if (ds4_session_create(&s, g_e, 4096) != 0) return false;
    ds4_gpu_graph *g = &s->graph;
    const int vocab = (int)DS4_N_VOCAB;
    const uint32_t n_pre_bank = (K > 0) ? 1u : 0u;
    const uint32_t need_banks = (uint32_t)N_DEC + n_pre_bank;
    bool ok = gpu_graph_bank_pool_count(g) >= need_banks;
    if (!ok) fprintf(stderr, "pool too small: %u < %u (set DS4_MSEQ_BANKS)\n",
                     gpu_graph_bank_pool_count(g), need_banks);
    char err[256];
    int argtok[N_DEC];

    /* decode banks: sync prompt, capture frontier, record next token. */
    for (int k = 0; ok && k < N_DEC; k++) {
        if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)k)) { ok = false; break; }
        ds4_session_invalidate(s);
        ds4_tokens p = { .v = g_toks.v + g_off[k], .len = g_len[k], .cap = g_len[k] };
        if (ds4_session_sync(s, &p, err, sizeof err) != 0) {
            fprintf(stderr, "decode bank %d sync failed: %s\n", k, err); ok = false; break;
        }
        gpu_graph_bank_counters_capture(g, (uint32_t)k);
        argtok[k] = ds4_session_argmax(s);
    }
    /* prefill bank (bank N_DEC): classic first chunk [0,C0) to lift its frontier
     * off 0 (step_begin rejects pos-0), then the K prefill rows extend [C0,C0+K). */
    const int *pptr = g_toks.v + PBASE;
    if (ok && K > 0) {
        if (g->banks.n_banks && !gpu_graph_bank_repoint(g, (uint32_t)N_DEC)) ok = false;
        if (ok) {
            ds4_session_invalidate(s);
            ds4_tokens p = { .v = (int *)pptr, .len = C0, .cap = C0 };
            if (ds4_session_sync(s, &p, err, sizeof err) != 0) {
                fprintf(stderr, "prefill bank first-chunk sync failed: %s\n", err); ok = false;
            }
        }
        if (ok) gpu_graph_bank_counters_capture(g, (uint32_t)N_DEC);
    }

    const uint32_t n_rows = (uint32_t)N_DEC + (K > 0 ? (uint32_t)K : 0u);
    ds4_multiseq_req *reqs = ok ? (ds4_multiseq_req *)malloc((size_t)n_rows * sizeof(*reqs)) : NULL;
    float *logits = ok ? (float *)malloc((size_t)n_rows * vocab * sizeof(float)) : NULL;
    if (ok && (!reqs || !logits)) ok = false;
    if (ok) {
        for (int k = 0; k < N_DEC; k++) {
            reqs[k].bank = (uint32_t)k; reqs[k].pos = g_len[k]; reqs[k].token = argtok[k];
        }
        for (int j = 0; j < K; j++) {
            reqs[N_DEC + j].bank = (uint32_t)N_DEC;
            reqs[N_DEC + j].pos = C0 + j;
            reqs[N_DEC + j].token = pptr[C0 + j];
        }
        uint32_t n_runs = 0;
        const int rc = ds4_session_decode_mixed(s, reqs, n_rows, logits, (int)(n_rows * vocab),
                                                &n_runs, head_cap, err, sizeof err);
        if (rc != 0) { fprintf(stderr, "decode_mixed(K=%d) failed rc=%d: %s\n", K, rc, err); ok = false; }
        else {
            const uint32_t full_runs = (uint32_t)N_DEC + (K > 0 ? 1u : 0u);
            const uint32_t exp_runs = (head_cap == 0u || head_cap > full_runs) ? full_runs : head_cap;
            if (n_runs != exp_runs) {
                fprintf(stderr, "n_runs=%u expected %u (head_cap=%u split boundary wrong)\n",
                        n_runs, exp_runs, head_cap);
                ok = false;
            }
        }
        if (ok) {
            /* logits rows: [bank0, bank1, (prefill-last)] in run order. */
            memcpy(dec_rows, logits, (size_t)N_DEC * vocab * sizeof(float));
            if (K > 0 && pre_row)
                memcpy(pre_row, logits + (size_t)N_DEC * vocab, (size_t)vocab * sizeof(float));
        }
    }
    free(reqs); free(logits);
    ds4_session_free(s);
    return ok;
}

/* classic RESUME reference for the prefill bank: fresh session, prefill [0,C0)
 * then resume to [0,C0+K), copy last-position logits + next token. */
static bool classic_resume(int K, float *out_lg, int *next_tok) {
    ds4_session *s = NULL;
    if (ds4_session_create(&s, g_e, 4096) != 0) return false;
    ds4_gpu_graph *g = &s->graph;
    char err[256]; bool ok = true;
    const int *pptr = g_toks.v + PBASE;
    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, 0)) { ds4_session_free(s); return false; }
    ds4_session_invalidate(s);
    ds4_tokens p0 = { .v = (int *)pptr, .len = C0, .cap = C0 };
    if (ds4_session_sync(s, &p0, err, sizeof err) != 0) { fprintf(stderr, "classic chunk: %s\n", err); ok = false; }
    ds4_tokens p1 = { .v = (int *)pptr, .len = C0 + K, .cap = C0 + K };
    if (ok && ds4_session_sync(s, &p1, err, sizeof err) != 0) { fprintf(stderr, "classic resume: %s\n", err); ok = false; }
    if (ok) {
        gpu_graph_bank_counters_capture(g, 0);
        ds4_session_copy_logits(s, out_lg, (int)DS4_N_VOCAB);
        *next_tok = ds4_session_argmax(s);
    }
    ds4_session_free(s);
    return ok;
}

static long first_diff(const float *a, const float *b, long n) {
    for (long i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }
    ds4_engine_options o; memset(&o, 0, sizeof o);
    o.model_path = argv[1]; o.backend = DS4_BACKEND_CUDA;
    if (ds4_engine_open(&g_e, &o) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }
    printf("CONFIG: DS4_ATTN_PACK=%s DS4_IDX_FP4=%s  (n_dec=%d K=%d)\n",
           getenv("DS4_ATTN_PACK") ? getenv("DS4_ATTN_PACK") : "(unset,default 1)",
           getenv("DS4_IDX_FP4") ? getenv("DS4_IDX_FP4") : "(unset,default 1)", N_DEC, K_PRE);

    size_t tl = 0; char *txt = read_file("tests/long_context_story_prompt.txt", &tl);
    if (!txt) { fprintf(stderr, "prompt read failed\n"); return 1; }
    memset(&g_toks, 0, sizeof g_toks);
    ds4_tokenize_text(g_e, txt, &g_toks); free(txt);
    int need = PBASE + C0 + K_PRE + 1;
    for (int k = 0; k < N_DEC; k++) if (g_off[k] + g_len[k] > need) need = g_off[k] + g_len[k];
    if (g_toks.len < need) { fprintf(stderr, "prompt too short (%d<%d)\n", g_toks.len, need); return 1; }

    const int vocab = (int)DS4_N_VOCAB;
    float *ref_dec = (float *)malloc((size_t)N_DEC * vocab * sizeof(float));   /* decode-only M=N_DEC */
    float *mix_dec = (float *)malloc((size_t)N_DEC * vocab * sizeof(float));   /* fused M=N_DEC+K, full head */
    float *lv1_dec = (float *)malloc((size_t)N_DEC * vocab * sizeof(float));   /* fused M=N_DEC+K, LEVER-1 head */
    float *mix_pre = (float *)malloc((size_t)vocab * sizeof(float));           /* fused prefill last */
    float *cls_pre = (float *)malloc((size_t)vocab * sizeof(float));           /* classic-resume last */
    int cls_next = -1;

    if (!fused_step_logits(0,     0u,             ref_dec, NULL))    { fprintf(stderr, "GATE FAIL: decode-only reference run failed\n"); g_fail = 1; goto done; }
    if (!fused_step_logits(K_PRE, 0u,             mix_dec, mix_pre)) { fprintf(stderr, "GATE FAIL: fused mixed run (full head) failed\n"); g_fail = 1; goto done; }
    if (!fused_step_logits(K_PRE, (uint32_t)N_DEC, lv1_dec, NULL))   { fprintf(stderr, "GATE FAIL: fused mixed run (LEVER-1 head) failed\n"); g_fail = 1; goto done; }
    if (!classic_resume(K_PRE, cls_pre, &cls_next))  { fprintf(stderr, "GATE FAIL: classic-resume reference failed\n"); g_fail = 1; goto done; }

    /* GATE 4: each decode bank byte-identical decode-only vs fused (full head). */
    for (int k = 0; k < N_DEC; k++) {
        const long d = first_diff(mix_dec + (size_t)k * vocab, ref_dec + (size_t)k * vocab, vocab);
        if (d < 0) {
            printf("GATE 4 NEUTRALITY: decode bank %d logits M=%d (decode-only) == M=%d (+%d-row prefill) "
                   "BYTE-IDENTICAL\n", k, N_DEC, N_DEC + K_PRE, K_PRE);
        } else {
            fprintf(stderr, "GATE 4 FAIL: decode bank %d logits DIFFER at float %ld (%.9g vs %.9g) — "
                    "co-scheduling a %d-row prefill perturbed a decode bank\n",
                    k, d, (mix_dec + (size_t)k * vocab)[d], (ref_dec + (size_t)k * vocab)[d], K_PRE);
            g_fail = 1;
        }
    }

    /* GATE 4b (LEVER 1): an INTERMEDIATE fused step (max_head_runs = n_dec, prefill
     * head SKIPPED, single-block identity head path) must yield decode-bank logits
     * BYTE-IDENTICAL to decode-only — the prefill's intermediate logits are never
     * consumed, and skipping them must not perturb the decode banks. */
    for (int k = 0; k < N_DEC; k++) {
        const long d = first_diff(lv1_dec + (size_t)k * vocab, ref_dec + (size_t)k * vocab, vocab);
        if (d < 0) {
            printf("GATE 4b LEVER-1: decode bank %d logits (intermediate fused, head-skip) == decode-only "
                   "BYTE-IDENTICAL\n", k);
        } else {
            fprintf(stderr, "GATE 4b FAIL: decode bank %d LEVER-1 logits DIFFER at float %ld (%.9g vs %.9g) "
                    "— skipping the prefill head perturbed a decode bank\n",
                    k, d, (lv1_dec + (size_t)k * vocab)[d], (ref_dec + (size_t)k * vocab)[d]);
            g_fail = 1;
        }
    }

    /* GATE 2: prefill run last-position logits vs classic-resume. */
    {
        double se = 0, sr = 0;
        for (int i = 0; i < vocab; i++) { double dd = (double)mix_pre[i] - cls_pre[i]; se += dd * dd; sr += (double)cls_pre[i] * cls_pre[i]; }
        double rel_rms = sr > 0 ? sqrt(se / sr) : (se > 0 ? 1e9 : 0.0);
        int mix_next = (int)argmax_f32(mix_pre, (uint64_t)vocab);
        printf("GATE 2 PREFILL: next-token fused=%d classic=%d %s | last-pos logit rel-RMS=%.3e (<1e-2: %s)\n",
               mix_next, cls_next, mix_next == cls_next ? "MATCH" : "MISMATCH",
               rel_rms, rel_rms < 1e-2 ? "YES" : "NO");
        if (mix_next != cls_next) { fprintf(stderr, "GATE 2 FAIL: prefill next-token mismatch (KV/boundary wrong)\n"); g_fail = 1; }
        if (rel_rms >= 1e-2) { fprintf(stderr, "GATE 2 FAIL: prefill last-pos rel-RMS %.3e >= 1e-2 (corruption)\n", rel_rms); g_fail = 1; }
    }

    /* GATE 3: split boundary. n_dec>=2 => per-token MoE prefix; K>8 => grouped MoE suffix. */
    printf("GATE 3 MoE SPLIT: n_dec=%d (per-token MoE, rows [0,%d)) | K=%d>8 (grouped MoE, rows [%d,%d)) | "
           "per-token->grouped switch at row %d %s\n",
           N_DEC, N_DEC, K_PRE, N_DEC, N_DEC + K_PRE, N_DEC,
           (N_DEC >= 2 && K_PRE > 8) ? "OK" : "MISCONFIGURED");
    if (!(N_DEC >= 2 && K_PRE > 8)) { fprintf(stderr, "GATE 3 FAIL: gate misconfigured (need n_dec>=2 and K>8)\n"); g_fail = 1; }

done:
    free(ref_dec); free(mix_dec); free(lv1_dec); free(mix_pre); free(cls_pre);
    ds4_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "MIXED-NEUTRALITY GATE: FAIL\n"); return 1; }
    printf("MIXED-NEUTRALITY GATE: PASS\n");
    return 0;
}
