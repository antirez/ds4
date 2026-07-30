#include "pulsar_engine_internal.h"

/* Confidence-scheduled draft trim threshold.  Defaults to tau=0.25.  At the
 * v0.2.2 default draft depth 3 the 2026-07-17 tau sweep found tau barely moves
 * GREEDY throughput (only 3 positions to trim: full range within 1-3% and the
 * peak wanders inside noise), but tau=0.25 clearly wins under T=1.0 SAMPLING
 * (+25% structured, +10% prose vs verify-all), where the low-confidence tail is
 * real.  The old "optimal trim loosens with depth" was a k=5 artifact — the
 * real driver is acceptance rate, not depth, and it washes out at k=3.  Trim is
 * exact and output-invariant on STRUCTURED, but narrowing the verify batch
 * shifts float accumulation ~1 ULP and can flip near-tie greedy argmax on flat
 * (prose) distributions (benign numerical tie, same class as yield-quench) — so
 * tau is distribution-preserving but NOT byte-identical on greedy prose.
 * PULSAR_DSPARK_CONF_SCHED=<tau> overrides; "0"/"off" disables (verify all
 * n_draft).  Adaptive tau is not worth building at k=3 (payoff ~2-6%, mostly
 * captured by 0.25). */
static float dspark_conf_sched_tau(void) {
    static float cached = -1.0f;
    if (cached < 0.0f) {
        const char *cs = getenv("PULSAR_DSPARK_CONF_SCHED");
        if (!cs || !cs[0]) cached = 0.25f;
        else if (!strcmp(cs, "off") || !strcmp(cs, "false")) cached = 0.0f;
        else {
            float v = (float)atof(cs);
            cached = v > 0.0f ? v : 0.0f;
        }
    }
    return cached;
}

/* --- Terminal yield-quench controller (spec-decode Item 4) ---------------
 * Controller design after the Entrpi ds4 yield quench (v0.1.1, MIT): per
 * request, every fused spec step accrues debt = breakeven yield minus
 * realized yield; when the request has provably lost more than a small
 * budget of plain tokens to speculation AND its recent yield is still below
 * breakeven, speculation turns off for the REMAINDER of that request
 * (terminal; a new request re-arms).
 *
 * Breakeven derivation (calibrated 2026-07-17 against per-step
 * PULSAR_DSPARK_STATS traces of the production v5mx serving path — 2585 steps
 * across prose/structured x greedy/T1.0, least-squares, resid rms 7.2 ms;
 * offline method after Entrpi's dspark_trace_replay, tool:
 * temp/quench/quench_replay.py):
 *   fused spec step   ~= FLAT + ROW * n_batch   milliseconds
 *     FLAT: pooled fit 101.6 ms, SHIPPED 95.7 ms — see the greedy-fit
 *           paragraph below for why the lower bound is the one compiled in
 *           (batched projections + shared + drafter dense/markov + sampled-q
 *           readback/dist walk — flat in verify rows; the old 53.6 ms nsys
 *           figure was the draft=3 build before temperature-matched drafting)
 *     ROW:  pooled fit 19.15 ms, SHIPPED 18.37 ms (marginal verify row —
 *           bandwidth-bound; matches the 2026-07-09 nsys 19.7 ms/row audit)
 *   plain decode token = PLAIN(pos), piecewise-linear through the measured
 *     served-plain depth table (2026-07-15, medians of 3): 59.7 ms @0.3k,
 *     67.3 @2.3k, 68.7 @9.3k, 74.5 @38k. Depth-dependence matters: spec step
 *     cost is ~flat in depth while plain slows, so a scalar PLAIN would
 *     overprice speculation exactly in the deep cells where it wins most
 *     (e.g. sweep-prose greedy @2.3k, +11.7%).
 * The breakeven yield of a step that verified K drafts at frontier pos is
 *   guard = (FLAT + ROW*(1+K)) / PLAIN(pos)     [plain tokens]
 * ~2.9 at full depth shallow, ~2.0 for a draft-only step. A request whose
 * committed tokens/step run below guard would have been faster plain.
 * Charging the ACTUAL n_batch (post conf-sched trim) rather than a scalar
 * guard prices exactly the steps the trimmer already shortened; committed
 * likewise is the post-trim realized yield. Measured operating points:
 * t2x prose y~2.05 vs guard ~2.8 (loses -> quench), structured y~5.1-5.3 vs
 * guard ~3.5 (wins by >1.5 -> never quenches).
 *
 * FLAT/ROW are the greedy-prose fit (the LOWER bound across the four
 * calibrated cells; the sampled cells fit ~6-11 ms higher FLAT). Deliberate:
 * underpricing the spec step biases against quenching, which keeps the
 * borderline deep greedy cells (sweep prose @2.3k: y~3.2 vs guard 3.06,
 * wins +11.7% measured) strictly on the no-quench side of the model.
 *
 * WARMUP: the first steps of every request are drafter pipeline fill —
 * n_batch ramps 1->2->3 with near-zero commits while the pendings build and
 * the prompt window seeds (a one-time ~40 ms not in the step model). That is
 * a fixed startup cost every request pays, not evidence about proposal
 * quality — charging it was measured (2026-07-17, first Load-2 gate run) to
 * book ~4.9 debt by step 6 at 2.3k ctx and spuriously quench the WINNING
 * sweep-prose cell (16.6 -> 14.8 t/s). The controller therefore ignores the
 * first PULSAR_QUENCH_WARMUP steps entirely, and MINEV=8 (Entrpi's replay
 * default) delays the verdict until the EWMA reflects steady state.
 * Re-validated offline over all 17 Load-1 traces + deep synthetic steady
 * states: losers (shallow y~2.05, deep y~2.5) fire at tokens ~11-25;
 * winners (struct, deep y>=3.2) never fire.
 *
 * Debt is deliberately NOT clamped below (matches Entrpi's shipped default):
 * unclamped, debt is exactly the request's NET plain-token-equivalents lost,
 * so the quench fires iff the request is genuinely >= BUDGET behind plain —
 * banked credit is real measured savings being spent, not optimism. Entrpi
 * measured the zero-clamped variant false-quenching long bursty winners, and
 * our offline replay selftest reproduces the same false quench for any
 * finite credit cap on a net-positive bursty request. Budget = 4 plain-token
 * equivalents (~250 ms): large enough that per-step yield variance on a
 * winning request cannot cross it (compounded with the EWMA and MINEV
 * conditions), small enough that a 400-token losing request recovers nearly
 * all of the loss.
 *
 * The controller reads only (commit, n_batch) — counts, never wall-clock —
 * so for a fixed token stream the quench point is deterministic. Constants
 * are compile-time (no hot-path env reads; project rule). */
/* REFIT 2026-07-21 (FLAT 95.7 -> 57.0; ROW unchanged). The 2026-07-17 fit went
 * stale: the fused step got ~30-40 ms cheaper (41 commits + the mxfp8head ->
 * type-40/MXFP8_LT model swap) while plain decode did not, so the shipped line
 * OVER-priced the step at EVERY width -- +57% at n_batch=1, +29% at 3, +14-15%
 * at 6..8. Guard = step/plain, so over-pricing raises the break-even, which
 * biases TOWARD quenching: the exact false-quench the design comment says it
 * deliberately biased against. Measured live at 2.7k/n_batch=6, yield 2.964 sat
 * below the shipped guard 3.057 (quench) but above the true 2.732 (spec was
 * ~8% faster than plain).
 *
 * Basis: 7591 steady-state steps over six cells (greedy/T1.0 x prose/structured
 * x 2.7k/9.4k/38k), pooled resid rms 5.88 ms (original fit: 7.2). Per-cell fits
 * are excellent (rms 0.08-2.4 ms), so ONE line is still the right model.
 *
 * Why 57.0 and why ROW stays 18.37:
 *  - Residual structure is by ENGINE DRAFT DEPTH, not temperature or context.
 *    FLAT rises ~3.3 ms per configured draft position (drafting work, which is
 *    NOT in the verify batch) while ROW falls to compensate: draft-2 fits
 *    FLAT 53.3/ROW 20.95, draft-7 fits FLAT 81.0/ROW 16.97. Since
 *    dspark_draft_tokens is fixed per engine and only conf-sched trim moves
 *    n_batch, the WITHIN-engine line is the one the controller actually rides.
 *    57.0 targets the shipped --dspark-draft 3 engine (~56.6); deeper drafts are
 *    then UNDER-priced, which biases against quenching = the safe direction.
 *  - Pooled ROW (20.3) is inflated by smearing cross-engine drafting cost into
 *    the slope. Keeping the lower shipped 18.37 lands the line 3-6% BELOW
 *    measured at every width 2..8, restoring the intended underprice margin.
 *  - Greedy remains the MINIMUM cell (T1.0 costs +3.5..12.4 ms at equal width),
 *    so a greedy-derived FLAT stays the conservative choice, as originally
 *    intended.
 * Depth enters as a small additive FLAT term only (~0.17 ms/1k, +6 ms over 35k)
 * and ROW is depth-invariant, so nothing depth-aware belongs in the step term.
 * NOTE the old comment's premise "spec step cost is ~flat in depth while plain
 * slows" is NOT what was measured: the step's depth slope (0.17 ms/1k) is close
 * to plain's (0.20). The guard ratio still improves with depth, but because the
 * step is 2-3x larger, not because it is flat. */
#define PULSAR_QUENCH_FLAT_MS    57.0f
#define PULSAR_QUENCH_ROW_MS     18.37f
#define PULSAR_QUENCH_ALPHA      0.125f   /* EWMA weight (Entrpi default) */
#define PULSAR_QUENCH_WARMUP     3u      /* ramp steps charged to no one (below) */
#define PULSAR_QUENCH_MINEV      8u      /* min spec steps before quench */
#define PULSAR_QUENCH_BUDGET     4.0f    /* plain-token equivalents */

/* Served plain-decode ms/token vs request depth: piecewise-linear through the
 * measured depth table (re-measured 2026-07-21 — see the block below for the
 * data and for why the 2026-07-15 values were retired). The bottom clamp
 * over-estimates plain and biases AGAINST quenching (conservative for the
 * no-spurious-quench gates). Plain ms/token keeps rising ~linearly with KV
 * depth, so beyond the last anchor we project that segment's slope rather than
 * flat-lining — a flat clamp under-estimates plain, which inflates the guard
 * and biases TOWARD quenching exactly where spec advantage is already marginal.
 * The projection is bounded at ~256k (PULSAR_QUENCH_PLAIN_CAP_POS), well past the
 * measured range, rather than extrapolating without limit. Deterministic
 * (constants only) so the quench point stays reproducible for a fixed stream. */
/* RE-MEASURED 2026-07-21 (medians of 3, run-to-run spread 0.01-0.11%), matching
 * the ORIGINAL instrument: the server's own `decoding ... avg=` line over a
 * 256-token greedy generation, `--no-dspark`, prose. Confirmed the shipped py
 * was exactly 1000/{16.74,14.85,14.55,13.43} from the 2026-07-15 prose table.
 *
 *   depth    old table   measured   ratio
 *     426      60.18       56.18    0.933
 *    2348      67.31       63.05    0.937
 *    9141      68.67       65.78    0.958
 *   38147      74.53       69.98    0.939
 *  100362      87.10*      80.24    0.921      (*extrapolated, now MEASURED)
 *
 * The table was uniformly ~6% HIGH — plain decode got ~6% faster since
 * 2026-07-15, alongside the fused spec step. The 100k anchor is new and real;
 * `-c` confound controlled (38k reads 70.12 at -c 131072 vs 70.00 at -c 40960,
 * +0.17%).
 *
 * A 2026-07-21 spot check that reported 85.89 ms at 38k (and an alarming
 * 0.675 ms/1k slope) was a MEASUREMENT ARTIFACT, reconstructed to 0.03 ms:
 * it used 128-vs-384 marginal differencing WITHOUT `--no-kv-disk`, so the two
 * legs took different disk-KV hits and prefilled 30011 vs 31163 tokens —
 * 70.20 true marginal + 12.55 unequal prefill + 3.17 client overhead = 85.92.
 * The real 9.1k->38k slope is 0.145 ms/1k, so the beyond-38k extrapolation was
 * if anything slightly too STEEP, never "3x too shallow". LESSON: when
 * differencing two runs to cancel prefill, DISABLE the disk KV cache or the
 * cancellation silently fails.
 *
 * DIRECTION (correcting the earlier note): the FLAT staleness and this table's
 * staleness did NOT compound — they partially CANCELLED. FLAT was over-priced
 * (guard too high) while plain was over-estimated (guard too low). Fixing FLAT
 * alone left the guard ~6-9% too LOW; this correction restores it (+7.0% @2.3k,
 * +6.4% @38k, +8.6% @100k, +11.9% @256k at n_batch=6).
 *
 * The measured curve is mildly convex (0.145 -> 0.165 ms/1k), so the linear
 * projection past 100k slightly UNDER-estimates plain, which inflates the guard
 * — the conservative direction this design already prefers, and now anchored on
 * a real 100k point instead of a 38k one. Past 100k is UNMEASURED: the 256k cap
 * (~105.9 ms) is a bounded guess. Also unmeasured: whether structured/tool
 * output shifts plain ms/token at depth (all five anchors are prose). */
#define PULSAR_QUENCH_PLAIN_CAP_POS 256000.0f
static float spec_quench_plain_ms(int pos) {
    static const float px[5] = { 300.0f, 2300.0f, 9300.0f, 38000.0f, 100000.0f };
    static const float py[5] = { 55.7f, 62.9f, 65.8f, 70.0f, 80.2f };
    const float p = (float)pos;
    if (p <= px[0]) return py[0];
    for (int i = 1; i < 5; i++)
        if (p <= px[i])
            return py[i - 1] + (py[i] - py[i - 1]) * (p - px[i - 1]) /
                                   (px[i] - px[i - 1]);
    /* pos > 100000: extend the last segment's slope, capped at the 256k value. */
    const float slope = (py[4] - py[3]) / (px[4] - px[3]);
    const float q = p < PULSAR_QUENCH_PLAIN_CAP_POS ? p : PULSAR_QUENCH_PLAIN_CAP_POS;
    return py[4] + slope * (q - px[4]);
}

static float spec_quench_guard(uint32_t n_batch, int pos) {
    return (PULSAR_QUENCH_FLAT_MS + PULSAR_QUENCH_ROW_MS * (float)n_batch) /
           spec_quench_plain_ms(pos);
}

/* Re-arm at request boundaries (the same sites that drop the carry and
 * pendings). All-zero == armed, matching the xcalloc'd session. */
void spec_quench_reset(pulsar_session *s) {
    s->spec.spec_quench_debt = 0.0f;
    s->spec.spec_quench_ewma = 0.0f;
    s->spec.spec_quench_steps = 0;
    s->spec.spec_quenched = false;
}

/* Test-only (identity gates): PULSAR_QUENCH_FORCE_STEP=<N> latches the quench
 * unconditionally once N fused spec steps have run, and disables the policy
 * decision. The check runs inside the fused step after steps++, so the
 * earliest possible latch is after step 1 completes: N=0 behaves like N=1
 * (a request that must be fully plain wants dspark_disable, not this hook).
 * Read once; absent in production. */
static int spec_quench_force_step(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *fs = getenv("PULSAR_QUENCH_FORCE_STEP");
        cached = fs && fs[0] ? atoi(fs) : -1;
        if (cached < -1) cached = -1;
    }
    return cached;
}

static void spec_frontier_free(pulsar_spec_frontier *f) {
    if (!f) return;
    memset(f, 0, sizeof(*f));
}



/* Build the batched-copy descriptor tables for the frontier snapshot/restore
 * copy sets. All source/destination tensors are fixed allocations, so the
 * tables are built once and replayed with one kernel launch per direction
 * (previously ~126 cudaMemcpy launches per snapshot, again per restore). A
 * NULL handle (prepare rejected a size, or alloc failed) keeps the loop path. */
static void spec_frontier_copy_tables_init(pulsar_gpu_graph *g) {
    if (g->spec_frontier_copy_init) return;
    g->spec_frontier_copy_init = 1;
    pulsar_gpu_tensor *dst[PULSAR_MAX_LAYER * 4];
    pulsar_gpu_tensor *src[PULSAR_MAX_LAYER * 4];
    uint64_t bytes[PULSAR_MAX_LAYER * 4];
    uint32_t n = 0;
    uint64_t mx = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = pulsar_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        dst[n] = g->spec_attn_state_kv[il];    src[n] = g->layer_attn_state_kv[il];    bytes[n++] = ab;
        dst[n] = g->spec_attn_state_score[il]; src[n] = g->layer_attn_state_score[il]; bytes[n++] = ab;
        if (ab > mx) mx = ab;
        if (ratio == 4) {
            const uint64_t ib = pulsar_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            dst[n] = g->spec_index_state_kv[il];    src[n] = g->layer_index_state_kv[il];    bytes[n++] = ib;
            dst[n] = g->spec_index_state_score[il]; src[n] = g->layer_index_state_score[il]; bytes[n++] = ib;
            if (ib > mx) mx = ib;
        }
    }
    if (n == 0) return;
    g->spec_snap_copies = pulsar_gpu_batched_copy_prepare(dst, src, bytes, n);
    /* restore = the same set with src/dst swapped */
    g->spec_restore_copies = pulsar_gpu_batched_copy_prepare(src, dst, bytes, n);
    g->spec_frontier_copy_n = n;
    g->spec_frontier_copy_max_bytes = mx;
}

static bool spec_frontier_snapshot(pulsar_spec_frontier *f, pulsar_session *s) {
    memset(f, 0, sizeof(*f));
    pulsar_gpu_graph *g = &s->graph;
    spec_frontier_copy_tables_init(g);

    bool ok = pulsar_gpu_begin_commands() != 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        f->n_comp[il] = g->layer_n_comp[il];
        f->n_index_comp[il] = g->layer_n_index_comp[il];
    }
    if (ok && g->spec_snap_copies) {
        ok = pulsar_gpu_batched_copy_run(g->spec_snap_copies,
                                      g->spec_frontier_copy_n,
                                      g->spec_frontier_copy_max_bytes) != 0;
    } else {
        for (uint32_t il = 0; ok && il < PULSAR_N_LAYER; il++) {
            const uint32_t ratio = pulsar_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint64_t ab = pulsar_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
            ok = pulsar_gpu_tensor_copy(g->spec_attn_state_kv[il], 0,
                                       g->layer_attn_state_kv[il], 0, ab) != 0 &&
                 pulsar_gpu_tensor_copy(g->spec_attn_state_score[il], 0,
                                       g->layer_attn_state_score[il], 0, ab) != 0;
            if (ratio == 4) {
                const uint64_t ib = pulsar_gpu_tensor_bytes(g->layer_index_state_kv[il]);
                ok = ok &&
                     pulsar_gpu_tensor_copy(g->spec_index_state_kv[il], 0,
                                           g->layer_index_state_kv[il], 0, ib) != 0 &&
                     pulsar_gpu_tensor_copy(g->spec_index_state_score[il], 0,
                                           g->layer_index_state_score[il], 0, ib) != 0;
            }
        }
    }
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    if (ok) return true;

    spec_frontier_free(f);
    return false;
}



static bool spec_frontier_restore(pulsar_spec_frontier *f, pulsar_session *s) {
    pulsar_gpu_graph *g = &s->graph;
    bool ok = pulsar_gpu_begin_commands() != 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        g->layer_n_comp[il] = f->n_comp[il];
        g->layer_n_index_comp[il] = f->n_index_comp[il];
    }
    if (ok && g->spec_restore_copies) {
        ok = pulsar_gpu_batched_copy_run(g->spec_restore_copies,
                                      g->spec_frontier_copy_n,
                                      g->spec_frontier_copy_max_bytes) != 0;
    } else {
        for (uint32_t il = 0; ok && il < PULSAR_N_LAYER; il++) {
            const uint32_t ratio = pulsar_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint64_t ab = pulsar_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
            ok = pulsar_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                       g->spec_attn_state_kv[il], 0, ab) != 0 &&
                 pulsar_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                       g->spec_attn_state_score[il], 0, ab) != 0;
            if (ok && ratio == 4) {
                const uint64_t ib = pulsar_gpu_tensor_bytes(g->layer_index_state_kv[il]);
                ok = pulsar_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                           g->spec_index_state_kv[il], 0, ib) != 0 &&
                     pulsar_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                           g->spec_index_state_score[il], 0, ib) != 0;
            }
        }
    }
    if (ok) ok = pulsar_gpu_end_commands() != 0;
    else (void)pulsar_gpu_synchronize();
    return ok;
}

static float dspark_base_top1_prob(const float *logits, int n) {
    float m = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > m) m = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += exp((double)(logits[i] - m));
    return sum > 0.0 ? (float)(1.0 / sum) : 1.0f;
}

/* Diagnostic: dump the DSpark drafter's per-step inputs (target_h[3], main_x)
 * and pre-markov base logits (spec_logits row 0) so an off-box reference forward
 * can be diffed against ds4 to localize acceptance loss. Enabled by
 * PULSAR_DSPARK_DUMP=<path>; caps at PULSAR_DSPARK_DUMP_STEPS (default 8) records.
 * Record layout (little-endian): pos i32, first_token i32, then f32 arrays
 * target_h[0..2] (PULSAR_N_EMBD each), main_x (PULSAR_N_EMBD), base0 (PULSAR_N_VOCAB). */
static void dspark_dump_step(pulsar_gpu_graph *g, int pos, int first_token,
                             const int32_t *refined_ids, int n_draft) {
    const char *path = getenv("PULSAR_DSPARK_DUMP");
    if (!path || !path[0]) return;
    static int dumped = 0;
    const char *lim = getenv("PULSAR_DSPARK_DUMP_STEPS");
    const int max_steps = lim ? atoi(lim) : 8;
    if (dumped >= max_steps) return;

    const uint64_t hcw = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
    float *emb = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(float));
    float *voc = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
    float *hc = (float *)xmalloc((size_t)hcw * sizeof(float));
    FILE *f = fopen(path, dumped == 0 ? "wb" : "ab");
    if (!f) { free(emb); free(voc); free(hc); return; }
    /* Lean mode (PULSAR_DSPARK_DUMP_LEAN=1): confidence-head training records only —
     * hdr + refined_ids + the post-hc_head hidden rows (batch_ffn_cur) that the
     * engine's confidence kernel consumes (Step 5c), ~64 KB/step at draft=4
     * instead of ~1 MB. Bulk-collectable for the drafter-retune Phase 0. */
    static int lean = -1;
    if (lean < 0) lean = getenv("PULSAR_DSPARK_DUMP_LEAN") != NULL;
    if (lean) {
        int32_t hdr[3] = { (int32_t)pos, (int32_t)first_token, (int32_t)n_draft };
        fwrite(hdr, sizeof(int32_t), 3, f);
        fwrite(refined_ids, sizeof(int32_t), (size_t)n_draft + 1, f);
        for (int p = 0; p < n_draft; p++) {
            memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
            (void)pulsar_gpu_tensor_read(g->batch_ffn_cur, (uint64_t)p * PULSAR_N_EMBD * 4,
                                      emb, (uint64_t)PULSAR_N_EMBD * 4);
            fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
        }
        fclose(f);
        dumped++;
        if ((dumped & 1023) == 1)
            fprintf(stderr, "pulsar: dspark lean dump step %d pos=%d -> %s\n", dumped, pos, path);
        free(emb); free(voc); free(hc);
        return;
    }
    /* Record: pos, tok, n_draft, refined_ids[0..n_draft], target_h[3], main_x,
     * base0(vocab), then cur_hc for each of the n_draft block positions. n_draft
     * is fixed per run -> fixed record size. Used to validate offline whether the
     * DSpark confidence head predicts per-position acceptance on our requant. */
    int32_t hdr[3] = { (int32_t)pos, (int32_t)first_token, (int32_t)n_draft };
    fwrite(hdr, sizeof(int32_t), 3, f);
    fwrite(refined_ids, sizeof(int32_t), (size_t)n_draft + 1, f);
    for (int i = 0; i < 3; i++) {
        memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
        (void)pulsar_gpu_tensor_read(g->dspark_target_h[i], 0, emb, (uint64_t)PULSAR_N_EMBD * 4);
        fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
    }
    memset(emb, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
    (void)pulsar_gpu_tensor_read(g->dspark_main_x, 0, emb, (uint64_t)PULSAR_N_EMBD * 4);
    fwrite(emb, sizeof(float), PULSAR_N_EMBD, f);
    memset(voc, 0, (size_t)PULSAR_N_VOCAB * sizeof(float));
    (void)gpu_graph_read_spec_logits_row(g, 0, voc);
    fwrite(voc, sizeof(float), PULSAR_N_VOCAB, f);
    /* block-2 output hidden (pre-hc_head) for each draft position [HC, EMBD]. */
    for (int p = 0; p < n_draft; p++) {
        memset(hc, 0, (size_t)hcw * sizeof(float));
        (void)pulsar_read_hc_carrier_f32(g->batch_cur_hc, (uint64_t)p * hcw, hc, hcw);
        fwrite(hc, sizeof(float), (size_t)hcw, f);
    }
    fclose(f);
    dumped++;
    fprintf(stderr, "pulsar: dspark dump step %d pos=%d tok=%d n_draft=%d -> %s\n",
            dumped, pos, first_token, n_draft, path);
    free(emb); free(voc); free(hc);
}

/* Fused-loop helper: make batch row `row`'s captured anchor hidden the current
 * drafter conditioning (target_h -> main_x) and seed one drafter-KV row from it.
 * Mirrors the reference invariant "drafter KV row j = f(hidden at position j)". */
static bool dspark_seed_from_batch_row(pulsar_session *s, uint32_t row) {
    pulsar_gpu_graph *g = &s->graph;
    pulsar_engine *e = s->engine;
    for (int i = 0; i < 3; i++) {
        if (!g->dspark_target_h_batch[i] || !g->dspark_target_h[i]) return false;
        if (!pulsar_gpu_tensor_copy(g->dspark_target_h[i], 0,
                                 g->dspark_target_h_batch[i],
                                 (uint64_t)row * PULSAR_N_EMBD * sizeof(float),
                                 (uint64_t)PULSAR_N_EMBD * sizeof(float))) return false;
    }
    if (!gpu_graph_dspark_project_main_x(g, &e->dspark_model, &e->dspark_weights)) return false;
    gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, &e->dspark_weights, 1);
    return true;
}

/* Fused DSpark loop (P2, PULSAR_DSPARK_FUSED=1): ONE batched target forward per
 * step instead of Step-1 decode + separate verify. The forward runs over
 * [first_token, pending_drafts...] (drafts made LAST step -- EAGLE pipeline
 * inversion), so position 0 is the base decode, positions 1..K verify the
 * drafts, and the batched anchor-hidden capture gives the drafter its
 * conditioning at whatever position ends up last-accepted. Drafting for the
 * NEXT step then conditions on the hidden that PRODUCED the next base token
 * (matching the reference generate.py forward_spec dataflow; the legacy loop
 * conditions on the hidden AFTER re-evaluating the base token -- a one-position
 * train/inference mismatch this path removes).
 * Greedy-only, like the legacy block (generate.c gates on temperature<=0).
 * Partial/zero accepts restore the frontier and replay the committed prefix
 * (Stage A; the Stage-B transactional state removes the replay). */
static int pulsar_session_eval_speculative_fused(pulsar_session *s, int first_token,
                                              int max_tokens, int eos_token,
                                              float temperature, int top_k,
                                              float top_p, float min_p,
                                              uint64_t *rng,
                                              int *accepted, int accepted_cap,
                                              char *err, size_t errlen) {
    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = &s->graph;
    const pulsar_dspark_weights *w = &e->dspark_weights;
    const uint32_t embed_dim = 256;
    const uint32_t vocab_size = w->vocab_size;
    const uint64_t vocab_bytes = (uint64_t)vocab_size * sizeof(float);
    const void *dmap = e->dspark_model.map;
    const uint64_t dsize = e->dspark_model.size;
    static int dspark_stats_env = -1;
    const int dspark_stats = gpu_graph_env_flag("PULSAR_DSPARK_STATS", &dspark_stats_env);
    static int dtree_stats_env = -1;
    const int dtree_stats = gpu_graph_env_flag("PULSAR_DTREE_STATS", &dtree_stats_env);
    const double t0 = dspark_stats ? now_sec() : 0.0;
    int n_accept = 0;

    /* Pending drafts continue from the greedy base we predicted last step; if
     * the caller committed something else (tool injection, sampling change),
     * they are stale. */
    int32_t pend[16];
    uint32_t K = s->spec.dspark_n_pending;
    if (K > 16u) K = 16u;
    if (K && s->spec.dspark_pending_base != (int32_t)first_token) K = 0;
    /* Position guard — ACCEPTANCE, not exactness. The drafts were conditioned on
     * the old position; dspark_pending_base is a token VALUE, so a plain eval
     * that advances the session (tool injection, </think> recovery) followed by
     * a first_token that happens to collide with it would otherwise resurrect
     * them. Dropping them is a throughput choice: a draft conditioned on the
     * wrong position is near-worthless and would just burn a verify row. It is
     * NOT what keeps the output exact — both accept rules are proposal-agnostic
     * (see the walk below), so q_oldpos is still exactly the distribution the
     * draft was drawn from and the rule still yields p at the new position.
     * Mirrors carry_pos_match. (checkpoint.len is still the drafting-step value
     * here: this runs before the first_token push below.) */
    if (K && s->spec.dspark_pending_pos != (int32_t)s->checkpoint.len) K = 0;
    /* Params guard — also acceptance, not exactness. Drafts drawn under params X
     * and verified under params Y are still verified EXACTLY: the residual
     * rebuilds q under the stored params X (see the walk below), so the accept
     * denominator and the residual name one proposal q_X, and the p/q rule
     * returns exactly p_Y for any q. What a param change costs is acceptance —
     * q_X is a poor proposal for p_Y — so we drop them and draft afresh.
     * Mirrors the spec_carry_* params guard in pulsar_session_generate_speculative.
     *
     * The same holds for argmax drafts (dspark_pending_sampled == false), which
     * carry no q: a temp<=0 -> temp>0 change cannot misroute them, because the
     * walk picks its rule from pend_sampled, not from the live temperature, so
     * they still meet the deterministic rule — itself exact for an arbitrary
     * proposal. The guard just keeps a badly-matched proposal from wasting a
     * row. */
    const bool pending_params_match =
        s->spec.dspark_pending_temp == temperature && s->spec.dspark_pending_top_k == top_k &&
        s->spec.dspark_pending_top_p == top_p && s->spec.dspark_pending_min_p == min_p;
    if (K && !pending_params_match) K = 0;
    /* Proposal rule the pendings were drafted under; the verify walk must apply
     * the matching rule (see dspark_pending_sampled). */
    const bool pend_sampled = K ? s->spec.dspark_pending_sampled : false;
    if ((int)K > accepted_cap - 1) K = accepted_cap > 1 ? (uint32_t)(accepted_cap - 1) : 0;
    if ((int)K > max_tokens - 1) K = max_tokens > 1 ? (uint32_t)(max_tokens - 1) : 0;
    for (uint32_t i = 0; i < K; i++) pend[i] = s->spec.dspark_pending[i];
    /* DTree Phase 0: carry last step's drafter #2 + conf for these pendings. */
    int32_t pend_alt[16];
    float pend_conf[16];
    if (dtree_stats)
        for (uint32_t i = 0; i < K; i++) {
            pend_alt[i] = s->spec.dspark_pending_alt[i];
            pend_conf[i] = s->spec.dspark_pending_conf[i];
        }
    s->spec.dspark_n_pending = 0;
    s->spec.spec_carry_valid = false;
    const uint32_t n_batch = 1u + K;

    /* Prompt-window seeding (one-time per prompt): fresh drafter state + a
     * captured prompt window -> replay the last <=128 prompt positions into
     * the drafter's context-KV ring, exactly as the reference prefills it.
     * Without this the window starts empty (or, before the invalidate fix,
     * stale from the previous request), and the drafter is near-useless
     * without a valid window (masked-window eval: 4.7% vs 86% top-1). */
    if (g->dspark_n_raw[0] == 0 && g->dspark_prompt_n > 0 && g->dspark_prompt_h[0]) {
        const uint32_t win = PULSAR_DSPARK_DRAFT_WINDOW;
        uint32_t avail = g->dspark_prompt_n - g->dspark_prompt_lo;
        const uint32_t take = avail < win ? avail : win;
        const uint32_t first = g->dspark_prompt_n - take;
        bool seed_ok = true;
        for (uint32_t j = 0; seed_ok && j < take; j++) {
            const uint32_t slot = (first + j) % win;
            for (int i = 0; seed_ok && i < 3; i++) {
                seed_ok = pulsar_gpu_tensor_copy(g->dspark_target_h[i], 0,
                                              g->dspark_prompt_h[i],
                                              (uint64_t)slot * PULSAR_N_EMBD * sizeof(float),
                                              (uint64_t)PULSAR_N_EMBD * sizeof(float)) != 0;
            }
            if (seed_ok && gpu_graph_dspark_project_main_x(g, &e->dspark_model, w))
                gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, w, 1);
        }
        if (dspark_stats)
            fprintf(stderr, "pulsar: dspark prompt-window seeded %u rows (prompt_n=%u)\n",
                    take, g->dspark_prompt_n);
        g->dspark_prompt_n = 0;   /* consumed; commits take over from here */
    }

    pulsar_spec_frontier frontier;
    memset(&frontier, 0, sizeof(frontier));
    if (!spec_frontier_snapshot(&frontier, s)) {
        snprintf(err, errlen, "DSpark fused frontier snapshot failed");
        s->checkpoint_valid = false;
        return -1;
    }

    const int saved_len = s->checkpoint.len;
    token_vec_push(&s->checkpoint, first_token);
    for (uint32_t i = 0; i < K; i++) token_vec_push(&s->checkpoint, (int)pend[i]);

    /* ONE batched forward: base decode + draft verify + anchor capture. */
    int row_tops[16];
    g->dspark_capture_batch_n = n_batch;
    g->spec_comp_save_n = n_batch;   /* Stage-B: save per-position comp projections */
    bool ok = gpu_graph_verify_suffix_tops(g, &e->model, &e->weights,
                                           &s->checkpoint,
                                           (uint32_t)saved_len, n_batch,
                                           K ? row_tops : NULL, NULL);
    g->dspark_capture_batch_n = 0;
    g->spec_comp_save_n = 0;
    if (!ok) {
        s->checkpoint.len = saved_len;
        (void)spec_frontier_restore(&frontier, s);
        spec_frontier_free(&frontier);
        snprintf(err, errlen, "DSpark fused verify failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Accept the longest prefix the target agrees with. Greedy: row i's
     * argmax must equal pend[i]. Sampled: exact speculative sampling under the
     * request's FILTERED target distribution p_i, with the rule matched to how
     * the draft was PROPOSED:
     *   - sampled proposal (the production path at temperature > 0): pend[i]
     *     was drawn from a temperature-matched q_i, so accept w.p.
     *     min(1, p_i/q_i) and on rejection draw the residual (p_i - q_i)+.
     *     Acceptance is NOT capped at p_i(mode).
     *   - argmax proposal (drafter/target vocab mismatch fallback): the
     *     deterministic rule — accept w.p. p_i(pend[i]), residual p_i with
     *     pend[i] excluded. Capped at p_i(mode).
     * The rejected row's replacement becomes the carry token. All three paths
     * yield the exact per-token target distribution. */
    int commit = 0;
    int carry_tok = -1;
    if (temperature <= 0.0f || K == 0) {
        while (commit < (int)K && row_tops[commit] == (int)pend[commit]) commit++;
    } else {
        if (!s->spec_row_scratch)
            s->spec_row_scratch = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
        float *row_logits = s->spec_row_scratch;
        bool walk_ok = true;
        while (commit < (int)K) {
            if (!gpu_graph_read_spec_logits_row(g, (uint32_t)commit, row_logits)) {
                walk_ok = false;
                break;
            }
            pulsar_sample_dist dist;
            pulsar_sample_dist_build(row_logits, PULSAR_N_VOCAB, temperature, top_k,
                                  top_p, min_p, &s->sample_scratch, &dist);
            const bool accepted_row = pend_sampled
                ? pulsar_sample_dist_accept_pq(&dist, (int)pend[commit],
                                            s->spec.dspark_pending_q[commit], rng)
                : pulsar_sample_dist_accept(&dist, (int)pend[commit], rng);
            if (accepted_row) {
                pulsar_sample_dist_free(&dist);
                commit++;
                continue;
            }
            if (pend_sampled) {
                /* Rebuild THIS position's q from the refined-logits row AND the
                 * params stored at draft time. Both halves of the rule must name
                 * the SAME proposal: the accept denominator above is the stored
                 * scalar q(pend[i]) computed under the draft-time params, so the
                 * residual's q must be rebuilt under those params too — not the
                 * live ones. dist_build is pure, so feeding it the persisted row
                 * + the persisted params reproduces the draft-time q bit-exactly,
                 * by construction rather than by the params guard's coincidence
                 * (never recomputed from live drafter state, which has advanced).
                 * Only the one rejecting position pays this — the walk stops
                 * here. */
                pulsar_sample_dist qd;
                pulsar_sample_dist_build(s->dspark_pending_qrows +
                                          (size_t)commit * PULSAR_N_VOCAB,
                                      PULSAR_N_VOCAB, s->spec.dspark_pending_temp,
                                      s->spec.dspark_pending_top_k, s->spec.dspark_pending_top_p,
                                      s->spec.dspark_pending_min_p,
                                      &s->sample_scratch, &qd);
                /* Holding two dists over one scratch is safe by dist_build's
                 * aliasing contract: out->ids/probs never point into scratch. */
                carry_tok = pulsar_sample_dist_draw_residual(&dist, &qd,
                                                          &s->sample_scratch, rng);
                pulsar_sample_dist_free(&qd);
            } else {
                carry_tok = pulsar_sample_dist_draw_excluding(&dist, (int)pend[commit], rng);
            }
            pulsar_sample_dist_free(&dist);
            break;
        }
        /* row_logits is the session-owned reusable row — not freed here. */
        if (!walk_ok) {
            s->checkpoint.len = saved_len;
            (void)spec_frontier_restore(&frontier, s);
            spec_frontier_free(&frontier);
            snprintf(err, errlen, "DSpark sampled-accept logits readback failed");
            s->checkpoint_valid = false;
            return -1;
        }
    }

    /* Prometheus /metrics spec-decode counters (server /metrics endpoint). The
     * base token is always emitted; K drafts were verified this step and the
     * accepted prefix is [0,commit). num_drafts counts draft rounds only. */
    e->spec_gen_tokens += 1u + (uint64_t)commit;
    s->spec.spec_gen_tokens += 1u + (uint64_t)commit;
    if (K > 0) {
        e->spec_draft_tokens += K;
        e->spec_accepted_tokens += (uint64_t)commit;
        e->spec_num_drafts += 1u;
        for (int i = 0; i < commit && i < 16; i++) e->spec_accepted_per_pos[i]++;
        s->spec.spec_draft_tokens += K;
        s->spec.spec_accepted_tokens += (uint64_t)commit;
        s->spec.spec_num_drafts += 1u;
    }

    /* Yield-quench controller update (see the constants block up top). Uses
     * the ACTUAL verify width n_batch (post conf-sched trim last step) and the
     * ACTUAL committed yield 1+commit — counts only, so the decision is
     * deterministic for a fixed stream. Once latched, generate_speculative
     * routes this request's remaining tokens down the plain-decode path and
     * the drafting block below is skipped; both paths sample the exact target
     * distribution, so quenching changes speed, never marginals. */
    if (!s->spec.spec_quenched) {
        s->spec.spec_quench_steps++;
        const int force = spec_quench_force_step();
        bool fire = false;
        if (force >= 0) {
            fire = s->spec.spec_quench_steps >= (uint32_t)force;
        } else if (s->spec.spec_quench_steps > PULSAR_QUENCH_WARMUP) {
            const float margin = (1.0f + (float)commit) -
                                 spec_quench_guard(n_batch, saved_len);
            s->spec.spec_quench_ewma = (1.0f - PULSAR_QUENCH_ALPHA) * s->spec.spec_quench_ewma +
                                  PULSAR_QUENCH_ALPHA * margin;
            s->spec.spec_quench_debt -= margin;   /* unclamped: NET tokens lost */
            fire = s->spec.spec_quench_steps >= PULSAR_QUENCH_MINEV &&
                   s->spec.spec_quench_ewma < 0.0f &&
                   s->spec.spec_quench_debt > PULSAR_QUENCH_BUDGET;
        }
        if (fire) {
            s->spec.spec_quenched = true;
            fprintf(stderr,
                    "pulsar: dspark yield-quench pos=%d steps=%u debt=%.2f ewma=%.2f "
                    "-> plain decode for request remainder%s\n",
                    saved_len + 1 + commit, s->spec.spec_quench_steps,
                    (double)s->spec.spec_quench_debt, (double)s->spec.spec_quench_ewma,
                    force >= 0 ? " (forced)" : "");
        }
    }

    /* DTree Phase 0 (§6): emit one record per ON-POLICY verified position — the
     * accepted prefix 0..commit-1 plus the split (first rejected draft) at
     * commit. Each carries that position's drafter confidence, whether the
     * chain #1 was accepted (acc), and — at the split — whether the target's
     * corrected token equals the drafter's #2 (alt_hit). This yields both the
     * accept-rate-by-conf a(c) and the p2-by-conf table the go/no-go gate needs
     * (the sibling row's marginal yield at a split is (1-a(c))*p2(c)). Positions
     * past commit are off-policy (conditioned on a rejected token) and skipped. */
    if (dtree_stats)
        for (int i = 0; i <= commit && i < (int)K; i++) {
            const int acc = (i < commit) ? 1 : 0;
            const int alt_hit = acc ? -1 : (row_tops[i] == pend_alt[i] ? 1 : 0);
            fprintf(stderr,
                    "DTREE_V pos=%d k=%d conf=%.4f acc=%d alt_hit=%d r1=%d r2=%d tgt=%d nb=%u\n",
                    saved_len + 1 + i, i, (double)pend_conf[i], acc, alt_hit,
                    (int)pend[i], pend_alt[i], row_tops[i], n_batch);
        }

    /* Refresh s->logits to the last committed position's distribution. */
    if (!gpu_graph_read_spec_logits_row(g, (uint32_t)commit, s->logits)) {
        s->checkpoint.len = saved_len;
        (void)spec_frontier_restore(&frontier, s);
        spec_frontier_free(&frontier);
        snprintf(err, errlen, "DSpark fused logits readback failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Finalize the carry token — the next base, drawn from the refreshed
     * s->logits (= row[commit]) when the walk did not already draw a residual:
     * greedy -> argmax (the old next_base); sampled full-accept -> the bonus
     * draw from the last accepted row's distribution. */
    if (carry_tok < 0) {
        if (temperature <= 0.0f) {
            carry_tok = sample_argmax(s->logits, PULSAR_N_VOCAB);
        } else {
            pulsar_sample_dist bonus;
            pulsar_sample_dist_build(s->logits, PULSAR_N_VOCAB, temperature, top_k,
                                  top_p, min_p, &s->sample_scratch, &bonus);
            carry_tok = pulsar_sample_dist_draw(&bonus, rng);
            pulsar_sample_dist_free(&bonus);
        }
    }

    bool ok_state = true;
    if (commit == (int)K) {
        /* Full accept: the batch advanced the target state by exactly the
         * committed tokens. Seed drafter rows for every committed position from
         * its own captured hidden (row j = f(h_j)); the last copy leaves
         * target_h = the drafting hidden. */
        s->checkpoint.len = saved_len + 1 + commit;
        for (int m = 0; ok_state && m <= commit && m < (int)n_batch; m++)
            ok_state = dspark_seed_from_batch_row(s, (uint32_t)m);
    } else {
        /* Partial/zero accept: target state includes rejected positions.
         * Stage A: restore the pre-batch frontier and replay the committed
         * prefix (first_token + accepted drafts). The decode path re-captures
         * each position's hidden; seed per position as the legacy loop does.
         * (Stage B replaces this replay with transactional state rollback.) */
        s->checkpoint.len = saved_len;
        ok_state = spec_frontier_restore(&frontier, s);
        static int no_replay = -1;
        if (no_replay < 0) no_replay = getenv("PULSAR_DSPARK_REPLAY") == NULL;
        if (ok_state && no_replay) {
            /* Stage B: transformer-free rollback. Roll only the recurrent
             * compressor/indexer pool state forward through the committed
             * prefix from the projections saved during the verify batch
             * (bit-identical: same update kernels, same rows, same order).
             * Raw KV + comp-cache rows are position-addressed and already
             * correct; counters are set by formula; s->logits was already
             * read from the fused batch's last committed row; drafter rows
             * seed from the batch capture. */
            ok_state = gpu_graph_dspark_compressor_rollforward(g, &e->model, &e->weights,
                                                               (uint32_t)saved_len,
                                                               (uint32_t)(1 + commit));
            if (ok_state) {
                s->checkpoint.len = saved_len + 1 + commit;
                for (int m = 0; ok_state && m <= commit; m++)
                    ok_state = dspark_seed_from_batch_row(s, (uint32_t)m);
            }
        } else if (ok_state) {
            int32_t replay[17];
            replay[0] = (int32_t)first_token;
            for (int i = 0; i < commit; i++) replay[1 + i] = pend[i];
            for (int i = 0; ok_state && i < 1 + commit; i++) {
                if (s->eval((int)replay[i], err, errlen) != 0) {
                    ok_state = false;
                    break;
                }
                if (gpu_graph_dspark_project_main_x(g, &e->dspark_model, &e->dspark_weights))
                    gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, &e->dspark_weights, 1);
            }
            /* Replay refreshed s->logits from the last committed token. */
        }
    }
    spec_frontier_free(&frontier);
    if (!ok_state) {
        snprintf(err, errlen, "DSpark fused state update failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Emit first_token + accepted drafts. */
    accepted[n_accept++] = first_token;
    bool hit_eos = first_token == eos_token;
    for (int i = 0; i < commit && n_accept < accepted_cap && !hit_eos; i++) {
        accepted[n_accept++] = (int)pend[i];
        if (pend[i] == (int32_t)eos_token) hit_eos = true;
    }

    /* The carry IS the next base (already correctly distributed). Persist it
     * so the next generate_speculative call forwards it as batch position 0;
     * pre-draft the NEXT block conditioned on it. */
    const int next_base = carry_tok;
    s->spec.spec_carry_token = (int32_t)carry_tok;
    s->spec.spec_carry_valid = !hit_eos;
    s->spec.spec_carry_pos = (int32_t)s->checkpoint.len;
    s->spec.spec_carry_temp = temperature;
    s->spec.spec_carry_top_k = top_k;
    s->spec.spec_carry_top_p = top_p;
    s->spec.spec_carry_min_p = min_p;
    uint32_t n_draft = (uint32_t)e->dspark_draft_tokens;
    if (n_draft > 16u) n_draft = 16u;
    if (hit_eos || next_base == eos_token || n_draft == 0 || s->spec.spec_quenched) {
        /* Quenched: don't draft the next chain — the carry persisted above is
         * still the correctly-distributed next base, which the next
         * generate_speculative call consumes before routing plain. */
        if (dspark_stats)
            fprintf(stderr, "pulsar: dspark fused n_batch=%u committed=%d nodraft step_ms=%.1f\n",
                    n_batch, commit, (now_sec() - t0) * 1000.0);
        return n_accept;
    }

    /* Draft forward + markov refine (mirrors the legacy block Steps 3-5).
     * NOTE: no seed here -- the committed positions' rows were seeded above
     * (row j = f(h_j)); next_base's own row is seeded NEXT step when it is
     * processed as batch position 0. main_x is re-projected for clarity (the
     * seeding loop left it at the same last-committed hidden). */
    if (!gpu_graph_dspark_project_main_x(g, &e->dspark_model, &e->dspark_weights))
        return n_accept;   /* drafting is best-effort; the step already succeeded */
    int32_t draft_ids[16];
    draft_ids[0] = (int32_t)next_base;
    for (uint32_t i = 1; i < n_draft; i++) draft_ids[i] = PULSAR_DSPARK_NOISE_TOKEN_ID;
    if (!gpu_graph_dspark_draft_forward(g, &e->model, &e->weights,
                                        &e->dspark_model, &e->dspark_weights,
                                        g->spec_logits, draft_ids, n_draft))
        return n_accept;

    pulsar_gpu_tensor *dspark_logits = g->dspark_markov_logits;   /* persistent scratch */
    if (!dspark_logits || pulsar_gpu_tensor_bytes(dspark_logits) < vocab_bytes) return n_accept;
    int32_t refined[17];
    /* DTree Phase 0: markov runner-up per draft position. CAVEAT since drafts
     * are sampled: this is the runner-up to the markov ARGMAX, which is no
     * longer necessarily the token we drafted. DTREE_V's alt_hit therefore no
     * longer means "target's correction == drafter's #2 given #1 rejected" —
     * the p2 table in memory was measured under argmax drafting. Diagnostic
     * only (dtree_stats), but re-derive p2 before reusing that conclusion. */
    int32_t refined2[17];
    refined[0] = (int32_t)next_base;
    refined2[0] = -1;
    /* Temperature-matched draft sampling: draw each draft from the refined
     * logits filtered at the REQUEST's params (q) instead of taking the
     * drafter's argmax, so the verify walk can use min(1, p/q) — whose
     * acceptance is not capped at p(mode).
     *
     * temperature <= 0 keeps the argmax path untouched: no readback, no
     * dist_build, and — critically — no rng draw, so the greedy token stream
     * stays byte-identical (dist_build would collapse to a point mass, but
     * pulsar_sample_dist_draw would still consume an rng word and shift the
     * stream). It is also the fast path we do not want to slow down.
     *
     * The vocab check is a correctness guard, not an optimization: p is built
     * over PULSAR_N_VOCAB target logits and q over the drafter's vocab_size. If
     * they disagree the two are not distributions over the same space (and the
     * row copy would overrun), so fall back to argmax proposals + the
     * deterministic accept rule, which needs no q. */
    const bool sample_drafts = temperature > 0.0f && vocab_size == PULSAR_N_VOCAB;
    if (sample_drafts) {
        /* n_draft rows, not 16: the depth is fixed per engine
         * (e->dspark_draft_tokens, 3 by default), so this is 1.6 MB rather than
         * 8.3 MB per session. Grow-only, so a depth change is still safe. */
        const uint32_t need = n_draft * PULSAR_N_VOCAB;
        if (s->dspark_pending_qrows_cap < need) {
            free(s->dspark_pending_qrows);
            s->dspark_pending_qrows = (float *)xmalloc((size_t)need * sizeof(float));
            s->dspark_pending_qrows_cap = need;
        }
    }
    bool draft_ok = true;
    for (uint32_t pos = 0; pos < n_draft && draft_ok; pos++) {
        pulsar_gpu_tensor *base_row = pulsar_gpu_tensor_view(
            g->spec_logits, (uint64_t)pos * vocab_bytes, vocab_bytes);
        draft_ok = base_row &&
            pulsar_gpu_dspark_markov_step_model(dspark_logits, &refined[pos + 1],
                                             dtree_stats ? &refined2[pos + 1] : NULL,
                                             base_row, dmap, dsize,
                                             w->markov_w1->abs_offset,
                                             w->markov_w2->abs_offset,
                                             refined[pos], vocab_size, embed_dim);
        pulsar_gpu_tensor_free(base_row);
        if (!draft_ok || !sample_drafts) continue;
        /* Read this position's refined logits back BEFORE the next markov step
         * overwrites the single-row scratch, and keep the row: the residual
         * needs the full q of whichever position rejects, which is not known
         * until verify. (This full-vocab D2H per draft position is the
         * deliberate temporary cost Item 2's fused GPU accept kernel removes.) */
        float *qrow = s->dspark_pending_qrows + (size_t)pos * PULSAR_N_VOCAB;
        if (!pulsar_gpu_tensor_read(dspark_logits, 0, qrow, vocab_bytes)) {
            draft_ok = false;
            break;
        }
        pulsar_sample_dist q;
        pulsar_sample_dist_build(qrow, PULSAR_N_VOCAB, temperature, top_k, top_p, min_p,
                              &s->sample_scratch, &q);
        const int drawn = pulsar_sample_dist_draw(&q, rng);
        refined[pos + 1] = (int32_t)drawn;   /* the chain continues SAMPLED */
        s->spec.dspark_pending_q[pos] = pulsar_sample_dist_prob(&q, drawn);
        /* Diagnostic: how much proposal entropy is there actually? The whole
         * premise of temperature-matched drafting is that q is a DISTRIBUTION.
         * If q.n == 1 (or q(top) ~ 1) the draw is the argmax, min(1,p/q)
         * degenerates to the deterministic rule, and Item 1 is a no-op. */
        if (dspark_stats)
            fprintf(stderr, "DSPARK_Q pos=%u q_n=%u q_top=%.4f q_drawn=%.4f "
                            "drawn_is_argmax=%d\n",
                    pos, q.n, (double)q.probs[0],
                    (double)s->spec.dspark_pending_q[pos], drawn == q.ids[0]);
        pulsar_sample_dist_free(&q);
    }
    if (!draft_ok) return n_accept;

    /* Offline-validation / confidence-training dump (same hook as the legacy
     * loop, which the production fused path previously never reached). Emitted
     * after markov refine, while batch_ffn_cur still holds the post-hc_head
     * hidden rows the confidence kernel consumes. */
    dspark_dump_step(g, (int)s->checkpoint.len, (int)next_base, refined, (int)n_draft);

    /* On-policy trajectory dump (PULSAR_DSPARK_DUMP_ONPOLICY=<max steps>,
     * _PATH=<file>): self-contained per-step records for teacher-forcing the
     * offline torch drafter on THE ENGINE'S OWN trajectory — discriminates
     * "remaining acceptance gap is on-policy distribution" (torch matches the
     * engine's level on identical data) from "engine draft-forward middle is
     * numerically wrong" (torch scores much higher). Record: {pos, next_base,
     * n_draft, n_batch, commit, refined[n_draft+1], th[n_batch][3][4096] f32};
     * batch row m holds the anchor hiddens of position pos-1-commit+m; rows
     * 0..commit are the committed positions, row commit the drafting anchor. */
    {
        static int opsteps = -1;
        if (opsteps < 0) {
            const char *e2 = getenv("PULSAR_DSPARK_DUMP_ONPOLICY");
            opsteps = e2 && e2[0] ? atoi(e2) : 0;
        }
        static int opdone = 0;
        const char *oppath = getenv("PULSAR_DSPARK_DUMP_ONPOLICY_PATH");
        if (opdone < opsteps && oppath && oppath[0]) {
            FILE *f2 = fopen(oppath, opdone == 0 ? "wb" : "ab");
            if (f2) {
                int32_t hdr2[5] = { (int32_t)s->checkpoint.len, (int32_t)next_base,
                                    (int32_t)n_draft, (int32_t)n_batch, (int32_t)commit };
                fwrite(hdr2, sizeof(int32_t), 5, f2);
                fwrite(refined, sizeof(int32_t), (size_t)n_draft + 1, f2);
                float *row2 = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(float));
                for (uint32_t m2 = 0; m2 < n_batch; m2++) {
                    for (int sl = 0; sl < 3; sl++) {
                        memset(row2, 0, (size_t)PULSAR_N_EMBD * sizeof(float));
                        (void)pulsar_gpu_tensor_read(g->dspark_target_h_batch[sl],
                                                  (uint64_t)m2 * PULSAR_N_EMBD * sizeof(float),
                                                  row2,
                                                  (uint64_t)PULSAR_N_EMBD * sizeof(float));
                        fwrite(row2, sizeof(float), PULSAR_N_EMBD, f2);
                    }
                }
                free(row2);
                /* Bug-#4 localization payload (PULSAR_DSPARK_DUMP_RING=1): the
                 * drafter's context-KV ring + counters as the NEXT draft
                 * forward will read them (post-seeding for this step's
                 * commits), plus this step's block outputs (cur_hc rows).
                 * Lets the torch reference attend over the ENGINE'S OWN ring
                 * to split "seed rows wrong" from "block compute wrong". */
                static int dump_ring = -1;
                if (dump_ring < 0) dump_ring = getenv("PULSAR_DSPARK_DUMP_RING") != NULL;
                if (dump_ring) {
                    const uint64_t ring_bytes =
                        (uint64_t)PULSAR_DSPARK_DRAFT_WINDOW * PULSAR_N_HEAD_DIM * sizeof(float);
                    float *ring = (float *)xmalloc(ring_bytes);
                    for (int li2 = 0; li2 < 3; li2++) {
                        int32_t nr = (int32_t)g->dspark_n_raw[li2];
                        fwrite(&nr, sizeof(int32_t), 1, f2);
                        memset(ring, 0, ring_bytes);
                        (void)pulsar_gpu_tensor_read(g->dspark_raw_cache[li2], 0, ring, ring_bytes);
                        fwrite(ring, sizeof(float), ring_bytes / sizeof(float), f2);
                    }
                    free(ring);
                    const uint64_t hcw2 = (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
                    float *hcrow = (float *)xmalloc(hcw2 * sizeof(float));
                    for (uint32_t p2 = 0; p2 < n_draft; p2++) {
                        memset(hcrow, 0, hcw2 * sizeof(float));
                        (void)pulsar_read_hc_carrier_f32(g->batch_cur_hc, (uint64_t)p2 * hcw2,
                                                      hcrow, hcw2);
                        fwrite(hcrow, sizeof(float), hcw2, f2);
                    }
                    free(hcrow);
                }
                fclose(f2);
                opdone++;
            }
        }
    }

    /* Confidence-scheduled pending length (P1 head; keep the confident prefix).
     * DTree Phase 0: compute conf even when conf-sched is off, so the p2 table
     * can bucket by the actual confidence at every drafted position (collect
     * with PULSAR_DSPARK_CONF_SCHED=off for an unbiased, fully-verified sample). */
    uint32_t keep = n_draft;
    float conf[16];
    bool have_conf = false;
    {
        const float tau = dspark_conf_sched_tau();
        if (tau > 0.0f || dtree_stats) {
            pulsar_gpu_tensor *conf_dev = pulsar_gpu_tensor_alloc((uint64_t)n_draft * sizeof(float));
            pulsar_gpu_tensor *tok_dev = pulsar_gpu_tensor_alloc((uint64_t)n_draft * sizeof(int32_t));
            if (conf_dev && tok_dev &&
                pulsar_gpu_tensor_write(tok_dev, 0, refined, (uint64_t)n_draft * sizeof(int32_t)) &&
                pulsar_gpu_dspark_confidence_score_model(conf_dev, g->batch_ffn_cur, tok_dev,
                                                      dmap, dsize,
                                                      w->markov_w1->abs_offset,
                                                      w->confidence_proj->abs_offset,
                                                      n_draft, PULSAR_N_EMBD, embed_dim, vocab_size) &&
                pulsar_gpu_tensor_read(conf_dev, 0, conf, (uint64_t)n_draft * sizeof(float))) {
                have_conf = true;
                if (tau > 0.0f) {
                    uint32_t k = 0;
                    while (k < n_draft && conf[k] >= tau) k++;
                    keep = k;   /* 0 pending = next step is a plain n=1 forward */
                }
            }
            pulsar_gpu_tensor_free(conf_dev);
            pulsar_gpu_tensor_free(tok_dev);
        }
    }
    s->spec.dspark_pending_base = (int32_t)next_base;
    s->spec.dspark_n_pending = keep;
    for (uint32_t i = 0; i < keep; i++) s->spec.dspark_pending[i] = refined[i + 1];
    /* The proposal rule and the exact params these drafts were sampled under.
     * Stamped unconditionally, in the same straight-line block as
     * dspark_n_pending above — this is the only site that makes pendings
     * non-zero, so a populated dspark_pending_q[]/qrows pool (filled in the
     * drafting loop above, under sample_drafts) always has its params alongside
     * it. Three consumers: the verify walk picks its accept rule from the flag,
     * next step's params guard compares against the params, and — load-bearing —
     * the residual rebuilds q from the qrows under THESE params, so the stored
     * accept denominator and the residual describe one proposal. */
    s->spec.dspark_pending_sampled = sample_drafts;
    s->spec.dspark_pending_pos = (int32_t)s->checkpoint.len;
    s->spec.dspark_pending_temp = temperature;
    s->spec.dspark_pending_top_k = top_k;
    s->spec.dspark_pending_top_p = top_p;
    s->spec.dspark_pending_min_p = min_p;
    if (dtree_stats)
        for (uint32_t i = 0; i < keep; i++) {
            s->spec.dspark_pending_alt[i] = refined2[i + 1];
            s->spec.dspark_pending_conf[i] = have_conf ? conf[i] : -1.0f;
        }

    /* DTree Phase 0: mid-band frequency — how often the drafted chain carries a
     * position whose confidence lands in the ~[0.25,0.65] split band (where a
     * confidence-gated sibling row could pay), and where the first such split
     * point falls. next_base sits at abs position s->checkpoint.len; drafted
     * position pos is next_base+1+pos. */
    if (dtree_stats && have_conf) {
        int firstmid = -1;
        for (uint32_t i = 0; i < n_draft; i++)
            if (conf[i] >= 0.25f && conf[i] <= 0.65f) { firstmid = (int)i; break; }
        char cbuf[160];
        int off = 0;
        for (uint32_t i = 0; i < n_draft && off < (int)sizeof(cbuf) - 8; i++)
            off += snprintf(cbuf + off, sizeof(cbuf) - off, "%s%.3f",
                            i ? "," : "", (double)conf[i]);
        fprintf(stderr, "DTREE_MID base_pos=%d nd=%u firstmid=%d conf=[%s]\n",
                (int)s->checkpoint.len, n_draft, firstmid, cbuf);
    }

    if (dspark_stats)
        fprintf(stderr, "pulsar: dspark fused n_batch=%u committed=%d pend=%u step_ms=%.1f\n",
                n_batch, commit, keep, (now_sec() - t0) * 1000.0);
    return n_accept;
}

/* Speculative generation that OWNS sampling: draws the base token from the
 * request's filtered distribution (or forwards the carry left by the previous
 * call), runs the fused draft/verify step with exact sampled acceptance, and
 * leaves the next correctly-distributed base as the carry. temperature <= 0
 * degenerates to the greedy argmax-equality path (byte-identical to the old
 * eval_speculative_block behavior). Returns the number of tokens emitted. */
int pulsar_session::generate_speculative(float temperature, int top_k,
                                     float top_p, float min_p, uint64_t *rng,
                                     int max_tokens, int eos_token,
                                     int *accepted, int accepted_cap,
                                     char *err, size_t errlen) {
    auto *s = this;
    if (!s || max_tokens <= 0 || accepted_cap <= 0 || !accepted) return 0;
    /* Same stale-classic-state guard as pulsar_session_eval: the spec loop
     * decodes and emits against the graph's scalar frontier counters, which
     * hold a cross-bank superset after a multiseq step. */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "speculative generate after a multiseq decode step: classic "
                 "per-bank state is stale; re-sync the session first");
        return 0;
    }
    int first;
    const bool carry_params_match =
        s->spec.spec_carry_temp == temperature && s->spec.spec_carry_top_k == top_k &&
        s->spec.spec_carry_top_p == top_p && s->spec.spec_carry_min_p == min_p;
    /* the carry is only valid at the exact position it was drawn at; any
     * session advance outside this path (sync, plain eval, tool injection)
     * means s->logits no longer matches the carry's source distribution */
    const bool carry_pos_match = s->spec.spec_carry_pos == (int32_t)s->checkpoint.len;
    if (s->spec.spec_carry_valid && carry_params_match && carry_pos_match) {
        first = (int)s->spec.spec_carry_token;
        s->spec.spec_carry_valid = false;
    } else {
        /* no carry, or params changed mid-stream (e.g. tool-call payloads
         * force greedy): redraw from the current distribution */
        s->spec.spec_carry_valid = false;
        first = sample_top_p_min_p(s->logits, PULSAR_N_VOCAB, temperature, top_k,
                                   top_p, min_p, rng);
    }
    if (first == eos_token) {
        /* never forward EOS through the target (matches the old caller loops,
         * which broke before eval) */
        accepted[0] = first;
        return 1;
    }
    /* Yield-quenched requests run plain for their remainder — the same route
     * as a drafterless engine, chosen per request. The carry consumed above is
     * already correctly distributed, so this is a pure speed decision. */
    if (!s->engine->has_dspark() || s->spec.spec_quenched) {
        if (s->eval(first, err, errlen) != 0) return -1;
        accepted[0] = first;
        return 1;
    }
    return pulsar_session_eval_speculative_fused(s, first, max_tokens, eos_token,
                                              temperature, top_k, top_p, min_p, rng,
                                              accepted, accepted_cap, err, errlen);
}



int pulsar_session::eval_speculative_block(int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    auto *s = this;
    if (!s || max_tokens <= 0 || accepted_cap <= 0 || !accepted) return 0;
    /* Same stale-classic-state guard as pulsar_session_eval (which the no-dspark
     * fallback below would otherwise hit one frame deeper). */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "speculative block eval after a multiseq decode step: classic "
                 "per-bank state is stale; re-sync the session first");
        return -1;
    }
    if (!s->engine->has_dspark() || s->spec.spec_quenched) {
        if (s->eval(first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    /* The fused loop (one batched forward/step + transactional no-replay
     * rollback) is the default: measured 16.4 vs 15.2 t/s legacy-vs-baseline
     * and byte-identical deterministic output. PULSAR_DSPARK_LEGACY_LOOP restores
     * the old Step1+verify loop as an operational fallback. */
    static int fused_cache = -1;
    if (fused_cache < 0) fused_cache = getenv("PULSAR_DSPARK_LEGACY_LOOP") == NULL;
    /* an externally chosen first_token invalidates any pending carry */
    s->spec.spec_carry_valid = false;
    if (fused_cache)
        return pulsar_session_eval_speculative_fused(s, first_token, max_tokens, eos_token,
                                                  0.0f, 0, 1.0f, 0.0f, NULL,
                                                  accepted, accepted_cap, err, errlen);

    pulsar_engine *e = s->engine;
    pulsar_gpu_graph *g = &s->graph;
    const uint32_t n_draft = (uint32_t)e->dspark_draft_tokens;
    int n_accept = 0;
    static int dspark_stats_env = -1;
    const int dspark_stats = gpu_graph_env_flag("PULSAR_DSPARK_STATS", &dspark_stats_env);
    const double dspark_t0 = dspark_stats ? now_sec() : 0.0;
    double dspark_draft_ms = 0.0;
    int dspark_base0 = -1;   /* draft-forward's row-0 argmax (pre-markov) */
    double dspark_markov_ms = 0.0, dspark_snap_ms = 0.0, dspark_verify_ms = 0.0;
    const int dump_pos = s->checkpoint.len;  /* first_token's sequence position */

    /* Step 1: run target decode for the first token */
    if (s->eval(first_token, err, errlen) != 0) return -1;

    /* Step 2: project main_x from captured target hidden states */
    if (!gpu_graph_dspark_project_main_x(g, &e->dspark_model, &e->dspark_weights))
        return -1;

    /* Step 2b: seed the current frontier's main_kv into the drafter KV BEFORE the
     * draft forward, matching the reference DSparkAttention (store main_kv at
     * start_pos, then attend).  The old code seeded only at the END of the step
     * (Step 7) and only on accepts, so the draft forward attended to an empty /
     * stale context.  One row per step -> dspark_n_raw tracks the frontier. */
    gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, &e->dspark_weights, 1);

    /* Step 3: build draft input — position 0 = first_token, rest = noise */
    int32_t draft_ids[16];
    draft_ids[0] = (int32_t)first_token;
    for (uint32_t i = 1; i < n_draft; i++)
        draft_ids[i] = PULSAR_DSPARK_NOISE_TOKEN_ID;

    /* Step 3b: confidence gate. Draft acceptance tracks the base model's own
     * certainty (the drafter is trained to mimic it), so only spend the
     * draft+verify budget when the target is peaked; on uncertain tokens the
     * verify would almost always reject and we'd pay ~100-150ms for nothing.
     * Skipping here costs exactly a plain decode (first_token already committed
     * in Step 1, s->logits intact). Steps 2/2b already seeded this position into
     * the drafter KV, so its rolling context stays complete for later steps.
     * PULSAR_DSPARK_MIN_CONF=0 (default) preserves the old always-draft behavior. */
    {
        static float min_conf = -1.0f;
        if (min_conf < 0.0f) {
            const char *mc = getenv("PULSAR_DSPARK_MIN_CONF");
            const float v = mc ? (float)atof(mc) : 0.0f;
            min_conf = v > 0.0f ? v : 0.0f;
        }
        if (min_conf > 0.0f) {
            const float p = dspark_base_top1_prob(s->logits, PULSAR_N_VOCAB);
            if (p < min_conf) {
                if (dspark_stats)
                    fprintf(stderr, "pulsar: dspark step draft_n=%d committed=0 conf_skip p=%.3f min=%.3f "
                                    "step_ms=%.1f\n",
                            (int)n_draft, (double)p, (double)min_conf,
                            (now_sec() - dspark_t0) * 1000.0);
                accepted[n_accept++] = first_token;
                return n_accept;
            }
        }
    }

    /* Step 4: run draft forward → N-token base logits in g->spec_logits */
    const double dspark_draft_t0 = dspark_stats ? now_sec() : 0.0;
    if (!gpu_graph_dspark_draft_forward(g, &e->model, &e->weights,
                                         &e->dspark_model, &e->dspark_weights,
                                         g->spec_logits, draft_ids, n_draft))
        return -1;
    if (dspark_stats) {
        (void)pulsar_gpu_synchronize();
        dspark_draft_ms = (now_sec() - dspark_draft_t0) * 1000.0;
        float *r0 = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
        if (gpu_graph_read_spec_logits_row(g, 0, r0))
            dspark_base0 = sample_argmax(r0, PULSAR_N_VOCAB);
        free(r0);
        /* conditioning diagnostics: is target_h captured, main_x sane, KV seeded? */
        float *tmp = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(float));
        double mn = 0.0, th[3] = {0, 0, 0};
        if (pulsar_gpu_tensor_read(g->dspark_main_x, 0, tmp, (uint64_t)PULSAR_N_EMBD * 4))
            for (int j = 0; j < (int)PULSAR_N_EMBD; j++) mn += (double)tmp[j] * tmp[j];
        for (int i = 0; i < 3; i++)
            if (pulsar_gpu_tensor_read(g->dspark_target_h[i], 0, tmp, (uint64_t)PULSAR_N_EMBD * 4))
                for (int j = 0; j < (int)PULSAR_N_EMBD; j++) th[i] += (double)tmp[j] * tmp[j];
        free(tmp);
        fprintf(stderr, "pulsar: dspark cond main_x=%.3f target_h=[%.2f,%.2f,%.2f] n_raw=[%u,%u,%u]\n",
                sqrt(mn), sqrt(th[0]), sqrt(th[1]), sqrt(th[2]),
                g->dspark_n_raw[0], g->dspark_n_raw[1], g->dspark_n_raw[2]);
    }

    /* Step 5: Markov refine — sequential over N positions */
    const pulsar_dspark_weights *w = &e->dspark_weights;
    const uint32_t embed_dim = 256;
    const uint32_t vocab_size = w->vocab_size;
    const uint64_t vocab_bytes = (uint64_t)vocab_size * sizeof(float);
    const void *dmap = e->dspark_model.map;
    const uint64_t dsize = e->dspark_model.size;

    pulsar_gpu_tensor *dspark_logits = g->dspark_markov_logits;   /* persistent scratch */
    if (!dspark_logits || pulsar_gpu_tensor_bytes(dspark_logits) < vocab_bytes) return -1;
    const double dspark_mk_t0 = dspark_stats ? now_sec() : 0.0;

    /* refined_ids[0] holds first_token; positions 1..n_draft hold the refined
     * drafts, so the array needs n_draft + 1 slots (17 at the clamp of 16). */
    int32_t refined_ids[17];
    refined_ids[0] = (int32_t)first_token;
    for (uint32_t pos = 0; pos < n_draft; pos++) {
        /* Create view of spec_logits row [pos] as base logits */
        pulsar_gpu_tensor *base_row = pulsar_gpu_tensor_view(
            g->spec_logits, (uint64_t)pos * vocab_bytes, vocab_bytes);
        if (!base_row) return -1;

        int32_t prev = refined_ids[pos];
        bool step_ok = pulsar_gpu_dspark_markov_step_model(dspark_logits, &refined_ids[pos + 1],
                                               NULL,
                                               base_row,
                                               dmap, dsize,
                                               w->markov_w1->abs_offset,
                                               w->markov_w2->abs_offset,
                                               (int32_t)prev, vocab_size, embed_dim);
        pulsar_gpu_tensor_free(base_row);
        if (!step_ok) return -1;
    }
    if (dspark_stats) dspark_markov_ms = (now_sec() - dspark_mk_t0) * 1000.0;

    dspark_dump_step(g, dump_pos, first_token, refined_ids, (int)n_draft);

    /* Step 5c: confidence-scheduled verification (P1). The trained DSpark
     * confidence head predicts, per block position, whether the draft will be
     * accepted (from the post-hc_head hidden in batch_ffn_cur + the drafted
     * token's markov embed). Size the verify budget to the confident prefix so
     * we don't spend the batch verify on low-confidence tail drafts. Threshold
     * PULSAR_DSPARK_CONF_SCHED (defaults to the measured tau in
     * dspark_conf_sched_tau(); "0"/"off" = verify all n_draft, classic behavior).
     * OUTPUT-PRESERVING: reducing the budget only limits how many drafts we
     * verify/commit; the emitted tokens are still exact greedy. */
    uint32_t eff_draft = n_draft;
    {
        const float tau = dspark_conf_sched_tau();
        if (tau > 0.0f) {
            pulsar_gpu_tensor *conf_dev = pulsar_gpu_tensor_alloc((uint64_t)n_draft * sizeof(float));
            pulsar_gpu_tensor *tok_dev  = pulsar_gpu_tensor_alloc((uint64_t)n_draft * sizeof(int32_t));
            if (conf_dev && tok_dev &&
                pulsar_gpu_tensor_write(tok_dev, 0, refined_ids, (uint64_t)n_draft * sizeof(int32_t)) &&
                pulsar_gpu_dspark_confidence_score_model(conf_dev, g->batch_ffn_cur, tok_dev,
                                                      dmap, dsize,
                                                      w->markov_w1->abs_offset,
                                                      w->confidence_proj->abs_offset,
                                                      n_draft, PULSAR_N_EMBD, embed_dim, vocab_size)) {
                float conf[16];
                if (pulsar_gpu_tensor_read(conf_dev, 0, conf, (uint64_t)n_draft * sizeof(float))) {
                    uint32_t k = 0;
                    while (k < n_draft && conf[k] >= tau) k++;
                    eff_draft = k < 1 ? 1u : k;   /* floor 1: draft_1 is free-validated below */
                    if (dspark_stats)
                        fprintf(stderr, "pulsar: dspark conf_sched tau=%.2f conf=[%.2f,%.2f,%.2f,%.2f] eff=%u/%u\n",
                                (double)tau, (double)conf[0],
                                n_draft > 1 ? (double)conf[1] : 0.0,
                                n_draft > 2 ? (double)conf[2] : 0.0,
                                n_draft > 3 ? (double)conf[3] : 0.0, eff_draft, n_draft);
                }
            }
            pulsar_gpu_tensor_free(conf_dev);
            pulsar_gpu_tensor_free(tok_dev);
        }
    }

    /* Step 5b: fast reject. The first draft is validated for FREE against
     * s->logits (Step 1's P(next | first_token)); only drafts 2..N need the
     * batch verify. If the first draft already disagrees, commit_drafts is 0 no
     * matter what the verify says -- so skip the snapshot/verify/restore
     * entirely (~100-150ms saved) and emit just first_token. Nothing beyond
     * Step 1 is allocated here (dspark_logits already freed). */
    if (sample_argmax(s->logits, PULSAR_N_VOCAB) != refined_ids[1]) {
        if (dspark_stats)
            fprintf(stderr, "pulsar: dspark step draft_n=%d committed=0 verify_skip "
                            "draft_ms=%.1f markov_ms=%.1f step_ms=%.1f\n",
                    (int)n_draft, dspark_draft_ms, dspark_markov_ms,
                    (now_sec() - dspark_t0) * 1000.0);
        accepted[n_accept++] = first_token;
        return n_accept;
    }

    /* Step 6: verify drafts against target, rejection sampling.
     *
     * Step 1 already committed first_token and left s->logits = P(next |
     * first_token), i.e. the target's prediction of the first draft.  The
     * batch verify below runs the target over the draft_n draft positions and
     * fills row_tops[i-1] with its argmax after committing refined_ids[1..i],
     * so it validates drafts 2..draft_n; the first draft is validated for free
     * against s->logits. */
    const int saved_len = s->checkpoint.len;
    const int draft_n = (int)eff_draft;   /* confidence-scheduled verify budget (Step 5c) */

    pulsar_spec_frontier frontier;
    memset(&frontier, 0, sizeof(frontier));
    const double dspark_snap_t0 = dspark_stats ? now_sec() : 0.0;
    bool have_frontier = spec_frontier_snapshot(&frontier, s);
    if (dspark_stats) dspark_snap_ms = (now_sec() - dspark_snap_t0) * 1000.0;

    for (int i = 0; i < draft_n; i++)
        token_vec_push(&s->checkpoint, refined_ids[i + 1]);

    int *row_tops = (int *)xmalloc((size_t)draft_n * sizeof(int));
    const double dspark_verify_t0 = dspark_stats ? now_sec() : 0.0;
    bool verify_ok = gpu_graph_verify_suffix_tops(g, &e->model, &e->weights,
                                                    &s->checkpoint,
                                                    (uint32_t)saved_len,
                                                    (uint32_t)draft_n,
                                                    row_tops, NULL);
    if (dspark_stats) dspark_verify_ms = (now_sec() - dspark_verify_t0) * 1000.0;
    if (!verify_ok) {
        s->checkpoint.len = saved_len;
        if (have_frontier) (void)spec_frontier_restore(&frontier, s);
        spec_frontier_free(&frontier);
        free(row_tops);
        snprintf(err, errlen, "DSpark verifier failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Accept the longest prefix the target agrees with.  The first draft is
     * gated by the free s->logits prediction; each subsequent draft by the
     * verifier's row_tops.  commit_drafts may be 0 when even the first draft
     * disagrees. */
    int commit_drafts = 0;
    if (sample_argmax(s->logits, PULSAR_N_VOCAB) == refined_ids[1]) {
        commit_drafts = 1;
        for (int i = 1; i < draft_n; i++) {
            if (row_tops[i - 1] != refined_ids[i + 1]) break;
            commit_drafts++;
        }
    }

    if (dspark_stats)
        fprintf(stderr, "pulsar: dspark step draft_n=%d committed=%d tgt_next=%d base0_top=%d refined1=%d "
                        "base0_hit=%d refined1_hit=%d draft_ms=%.1f markov_ms=%.1f snap_ms=%.1f "
                        "verify_ms=%.1f step_ms=%.1f\n",
                draft_n, commit_drafts,
                sample_argmax(s->logits, PULSAR_N_VOCAB), dspark_base0, refined_ids[1],
                dspark_base0 == sample_argmax(s->logits, PULSAR_N_VOCAB),
                refined_ids[1] == sample_argmax(s->logits, PULSAR_N_VOCAB),
                dspark_draft_ms, dspark_markov_ms, dspark_snap_ms, dspark_verify_ms,
                (now_sec() - dspark_t0) * 1000.0);

    bool ok_state = true;
    if (commit_drafts == draft_n) {
        /*
         * Full accept: every draft stays committed and the verifier already
         * advanced the target KV over all of them, so no rollback is needed.
         * s->logits is still P(next | first_token) though — refresh it to the
         * target distribution after the last committed draft (spec_logits row
         * draft_n-1).
         */
        if (!s->spec_row_scratch)
            s->spec_row_scratch = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(s->spec_row_scratch[0]));
        float *row_logits = s->spec_row_scratch;
        ok_state = gpu_graph_read_spec_logits_row(g, (uint32_t)(draft_n - 1), row_logits);
        if (ok_state)
            memcpy(s->logits, row_logits, (size_t)PULSAR_N_VOCAB * sizeof(s->logits[0]));
    } else {
        /*
         * Partial accept or full reject: the verifier ran the target over all
         * draft_n positions, so its KV/compressor frontier is now ahead of the
         * commit_drafts we are keeping.  Roll back to the pre-draft frontier and
         * replay exactly the committed drafts, which also refreshes s->logits.
         * This is only possible with a valid snapshot — treat its absence as a
         * hard error rather than leaving rejected drafts committed.
         */
        if (!have_frontier) {
            spec_frontier_free(&frontier);
            free(row_tops);
            snprintf(err, errlen, "DSpark frontier snapshot failed");
            s->checkpoint_valid = false;
            return -1;
        }
        s->checkpoint.len = saved_len;
        ok_state = spec_frontier_restore(&frontier, s);
        for (int i = 0; ok_state && i < commit_drafts; i++) {
            if (s->eval(refined_ids[i + 1], err, errlen) != 0) {
                ok_state = false;
                break;
            }
            /* The single-token decode above just captured THIS accepted
             * position's anchor-layer target hiddens.  Project them and seed the
             * drafter KV so each accepted draft gets its OWN conditioning instead
             * of first_token's (reusing first_token's was measured to hurt).
             * This is the per-position content fix -- the real lever on
             * acceptance -- and it is free here (the decode already ran). */
            if (gpu_graph_dspark_project_main_x(g, &e->dspark_model, &e->dspark_weights))
                gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, &e->dspark_weights, 1);
        }
    }

    spec_frontier_free(&frontier);
    free(row_tops);
    if (!ok_state) {
        snprintf(err, errlen, "DSpark state update failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Step 7: nothing to seed here.  Step 2b seeds first_token's main_kv before
     * the forward (one row per step, correct content).  Reusing this step's
     * main_x for the ACCEPTED positions was measured to HURT (28.6% -> 23.9%):
     * the drafter is sensitive to per-position content, so accepted positions
     * need their OWN captured target hidden (see verify-capture, TODO). */

    /* Step 8: return first_token followed by the accepted drafts.  The caller
     * emits only the tokens returned here, so first_token — committed to the
     * target KV in Step 1 — must lead the list or it is dropped from the
     * output while remaining in context. */
    accepted[n_accept++] = first_token;
    for (int i = 0; i < commit_drafts && n_accept < accepted_cap; i++) {
        if (refined_ids[i + 1] == eos_token) break;
        accepted[n_accept++] = refined_ids[i + 1];
    }

    return n_accept;
}
