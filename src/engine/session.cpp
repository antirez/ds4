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
        const char *cs = getenv("DS4_DSPARK_CONF_SCHED");
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
static void spec_quench_reset(pulsar_session *s) {
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
        const char *fs = getenv("DS4_QUENCH_FORCE_STEP");
        cached = fs && fs[0] ? atoi(fs) : -1;
        if (cached < -1) cached = -1;
    }
    return cached;
}

static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}



static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}



static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}



static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = (const uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}



static PULSAR_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = (uint8_t *)ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}



static PULSAR_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}



static PULSAR_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}



static int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > PULSAR_SESSION_IO_CHUNK ? PULSAR_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n) {
            payload_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}



static PULSAR_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * PULSAR_N_HEAD_DIM * coff * ratio * sizeof(float);
}



static PULSAR_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * PULSAR_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}



/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical GPU tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const pulsar_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : PULSAR_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}



/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const pulsar_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    /* Session files always store comp rows as f32 (packed caches dequant to
     * the f32 shadow on save), so payload sizing is format-independent. */
    const uint64_t comp_row = (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * PULSAR_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * comp_row;
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}



/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const pulsar_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > pulsar_gpu_tensor_bytes(tensor) ||
        bytes > pulsar_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (pulsar_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}



static int payload_read_tensor_span(FILE *fp, pulsar_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > pulsar_gpu_tensor_bytes(tensor) ||
        bytes > pulsar_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (pulsar_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}



/* Session files always store the indexer comp cache as f32 rows.  Under
 * PULSAR_IDX_FP4 the persistent cache is MXKV-FP4-packed, so save dequantizes
 * into the f32 staging first and load repacks from it.  The repack is
 * value-exact for all realistic rows (QAT-roundtripped fp4 values on
 * power-of-two block scales survive re-encoding); the one exception is a
 * 32-block whose amax sits below the mxkv encode floor (1e-20), which would
 * flush to zero — unreachable for RMS-normed indexer rows. */
static int payload_write_index_comp(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                    uint32_t n_rows, uint8_t *buf, size_t cap,
                                    char *err, size_t errlen) {
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    pulsar_gpu_tensor *src = g->layer_index_comp_cache[il];
    if (gpu_graph_idx_fp4_enabled() && n_rows != 0) {
        if (!g->idx_comp_stage ||
            pulsar_gpu_mxkv_dequant_tensor(g->layer_index_comp_cache[il],
                                        g->idx_comp_stage,
                                        PULSAR_ENGINE_MXKV_FMT_FP4,
                                        n_rows,
                                        PULSAR_N_INDEXER_HEAD_DIM) == 0) {
            payload_set_err(err, errlen, "failed to dequantize fp4 indexer cache for session save");
            return 1;
        }
        src = g->idx_comp_stage;
    }
    return payload_write_tensor_span(fp, src, 0, bytes, buf, cap, err, errlen);
}

static int payload_read_index_comp(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                   uint32_t n_rows, uint8_t *buf, size_t cap,
                                   uint64_t *remaining, char *err, size_t errlen) {
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    if (!gpu_graph_idx_fp4_enabled() || n_rows == 0) {
        return payload_read_tensor_span(fp, g->layer_index_comp_cache[il], 0, bytes,
                                        buf, cap, remaining, err, errlen);
    }
    if (!g->idx_comp_stage) {
        payload_set_err(err, errlen, "fp4 indexer cache staging missing on session load");
        return 1;
    }
    int rc = payload_read_tensor_span(fp, g->idx_comp_stage, 0, bytes,
                                      buf, cap, remaining, err, errlen);
    if (rc != 0) return rc;
    if (pulsar_gpu_mxkv_pack_tensor(g->idx_comp_stage,
                                 g->layer_index_comp_cache[il],
                                 PULSAR_ENGINE_MXKV_FMT_FP4,
                                 n_rows,
                                 PULSAR_N_INDEXER_HEAD_DIM) == 0) {
        payload_set_err(err, errlen, "failed to repack fp4 indexer cache on session load");
        return 1;
    }
    return 0;
}

/* Attn comp cache spans under PULSAR_ATTN_PACK: session files always store f32
 * rows, so save dequantizes the packed cache into the f32 shadow first and
 * load repacks from it.  Save is bit-exact by construction (packed rows decode
 * to exactly the fp8-roundtripped values the f32 pipeline holds).  Load
 * re-quantizes already-roundtripped rows; that is value-preserving except for
 * blocks whose amax sits exactly on a scale boundary, where the recomputed
 * block scale can shift one step and re-round small values (the same
 * non-idempotency that forced quantize_fp8=false in the pack prefill paths).
 * Sub-1e-3-relative on isolated dims; acceptable for session restore, but do
 * NOT rely on save/load being bit-exact under pack. */
static int payload_write_attn_comp_pack(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                        uint32_t n_rows, uint8_t *buf, size_t cap,
                                        char *err, size_t errlen) {
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_N_HEAD_DIM * sizeof(float);
    pulsar_gpu_tensor *src = g->attn_comp_dequant;
    if (n_rows != 0) {
        if (!src ||
            pulsar_gpu_attn_pack_dequant_tensor(g->layer_attn_comp_cache[il],
                                             src, n_rows,
                                             PULSAR_N_HEAD_DIM, PULSAR_N_ROT) == 0) {
            payload_set_err(err, errlen, "failed to dequantize packed attn comp cache for session save");
            return 1;
        }
    }
    if (n_rows == 0) return 0;
    return payload_write_tensor_span(fp, src, 0, bytes, buf, cap, err, errlen);
}

static int payload_read_attn_comp_pack(FILE *fp, pulsar_gpu_graph *g, uint32_t il,
                                       uint32_t n_rows, uint8_t *buf, size_t cap,
                                       uint64_t *remaining, char *err, size_t errlen) {
    if (n_rows == 0) return 0;
    const uint64_t bytes = (uint64_t)n_rows * PULSAR_N_HEAD_DIM * sizeof(float);
    if (!g->attn_comp_dequant) {
        payload_set_err(err, errlen, "packed attn comp cache staging missing on session load");
        return 1;
    }
    int rc = payload_read_tensor_span(fp, g->attn_comp_dequant, 0, bytes,
                                      buf, cap, remaining, err, errlen);
    if (rc != 0) return rc;
    /* Exact-scale repack: file rows are already roundtripped, and the
     * fast-math quantize bucket is not bit-idempotent at scale boundaries. */
    if (pulsar_gpu_attn_pack_repack_tensor(g->attn_comp_dequant,
                                        g->layer_attn_comp_cache[il],
                                        0, n_rows,
                                        PULSAR_N_HEAD_DIM, PULSAR_N_ROT) == 0) {
        payload_set_err(err, errlen, "failed to repack attn comp cache on session load");
        return 1;
    }
    return 0;
}



static PULSAR_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const pulsar_gpu_tensor *tensor,
                                                                 uint64_t offset_f16, uint64_t count,
                                                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > pulsar_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > pulsar_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (pulsar_gpu_tensor_read(tensor, offset_f16 + done * sizeof(uint16_t),
                                h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to read GPU F16 session tensor");
            return 1;
        }
        for (size_t i = 0; i < n; i++) f[i] = f16_to_f32(h[i]);
        if (payload_write_bytes(fp, f, (uint64_t)n * sizeof(float), err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}



static PULSAR_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, pulsar_gpu_tensor *tensor,
                                                                uint64_t offset_f16, uint64_t count,
                                                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                                                char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > pulsar_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > pulsar_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (payload_read_bytes(fp, f, (uint64_t)n * sizeof(float), remaining, err, errlen) != 0) return 1;
        for (size_t i = 0; i < n; i++) h[i] = f32_to_f16(f[i]);
        if (pulsar_gpu_tensor_write(tensor, offset_f16 + done * sizeof(uint16_t),
                                 h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to restore GPU F16 session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

/* Raw-ring row spans: session files always store f32 rows.  Under PULSAR_RAW_F16
 * the ring holds __half containers, so save expands f16->f32 and load packs
 * f32->f16 — bit-exact both ways because the values are f16-rounded at store
 * time in both modes. */
static int payload_write_raw_row(FILE *fp, pulsar_gpu_graph *g, uint32_t il, uint32_t phys,
                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (gpu_graph_raw_f16_enabled()) {
        return payload_write_tensor_span_f16_as_f32(fp, g->layer_raw_cache[il],
                (uint64_t)phys * PULSAR_N_HEAD_DIM * sizeof(uint16_t),
                (uint64_t)PULSAR_N_HEAD_DIM, buf, cap, err, errlen);
    }
    return payload_write_tensor_span(fp, g->layer_raw_cache[il],
            (uint64_t)phys * PULSAR_N_HEAD_DIM * sizeof(float),
            (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float), buf, cap, err, errlen);
}

static int payload_read_raw_row(FILE *fp, pulsar_gpu_graph *g, uint32_t il, uint32_t phys,
                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                char *err, size_t errlen) {
    if (gpu_graph_raw_f16_enabled()) {
        return payload_read_tensor_span_f32_as_f16(fp, g->layer_raw_cache[il],
                (uint64_t)phys * PULSAR_N_HEAD_DIM * sizeof(uint16_t),
                (uint64_t)PULSAR_N_HEAD_DIM, buf, cap, remaining, err, errlen);
    }
    return payload_read_tensor_span(fp, g->layer_raw_cache[il],
            (uint64_t)phys * PULSAR_N_HEAD_DIM * sizeof(float),
            (uint64_t)PULSAR_N_HEAD_DIM * sizeof(float), buf, cap, remaining, err, errlen);
}





int pulsar_engine_routed_quant_bits(pulsar_engine *e) {
    if (!e) return 0;
    /* Report the routed-expert precision tier actually present, derived from
     * the loaded tensor types (was hardcoded 2, which under-reported the mixed
     * IQ2 + MXFP4/type-40 build as pure 2-bit). Any 4-bit routed format
     * (MXFP4 E2M1 / CUTLASS type-40) anywhere in gate/up/down makes this a
     * 4-bit-tier model; otherwise the 2-bit floor (IQ2_XXS / Q2_K); 0 if no
     * routed experts. The kvstore snapshot-compat guards accept {2,4} and
     * pulsar_engine_model_id() is a compile-time constant, so this is the only
     * model-variant discriminator in the disk-KV key — a value change
     * invalidates old snapshots (one-time re-prefill; fine in dev). */
    int bits = 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const pulsar_tensor *proj[3] = {
            e->weights.layer[il].ffn_gate_exps,
            e->weights.layer[il].ffn_up_exps,
            e->weights.layer[il].ffn_down_exps,
        };
        for (int k = 0; k < 3; k++) {
            const pulsar_tensor *t = proj[k];
            if (!t) continue;
            if (t->type == PULSAR_TENSOR_FP4_E2M1 ||
                t->type == PULSAR_TENSOR_CUTLASS_MXFP4)
                return 4;
            if (bits == 0) bits = 2;
        }
    }
    return bits;
}



bool pulsar_engine_has_output_head(pulsar_engine *e) {
    return e && weights_have_output_head(&e->weights);
}





bool pulsar_engine_has_dspark(pulsar_engine *e) {
    return e && e->dspark_ready;
}

int pulsar_engine_dspark_draft_tokens(pulsar_engine *e) {
    return pulsar_engine_has_dspark(e) ? e->dspark_draft_tokens : 0;
}



const pulsar_tokens *pulsar_session_tokens(pulsar_session *s) {
    return s ? &s->checkpoint : NULL;
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



uint64_t pulsar_session_payload_bytes(pulsar_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    const pulsar_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)PULSAR_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)PULSAR_N_VOCAB * sizeof(float);
    bytes += (uint64_t)PULSAR_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)PULSAR_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
}



int pulsar_session_write_staged_payload(const pulsar_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}



void pulsar_session_payload_file_free(pulsar_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}



int pulsar_session_stage_payload(pulsar_session *s, pulsar_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!out) {
        payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        if (errlen) snprintf(err, errlen, "failed to open staged session payload: %s",
                             strerror(saved));
        return 1;
    }

    int rc = pulsar_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = pulsar_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}



int pulsar_session_save_payload(pulsar_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }

    pulsar_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[PULSAR_SESSION_PAYLOAD_U32_FIELDS] = {
        PULSAR_SESSION_PAYLOAD_MAGIC,
        PULSAR_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        PULSAR_N_LAYER,
        PULSAR_N_HEAD_DIM,
        PULSAR_N_INDEXER_HEAD_DIM,
        PULSAR_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < PULSAR_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < PULSAR_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_raw_row(fp, g, il, phys,
                                       buf, PULSAR_SESSION_IO_CHUNK, err, errlen);
        }
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        if (gpu_graph_attn_pack_enabled()) {
            rc = payload_write_attn_comp_pack(fp, g, il,
                                              g->layer_n_comp[il],
                                              buf,
                                              PULSAR_SESSION_IO_CHUNK,
                                              err,
                                              errlen);
        } else {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * PULSAR_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           PULSAR_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    PULSAR_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    PULSAR_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_index_comp(fp, g, il,
                                          g->layer_n_index_comp[il],
                                          buf,
                                          PULSAR_SESSION_IO_CHUNK,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        PULSAR_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        PULSAR_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
}



int pulsar_session_load_payload(pulsar_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    /* drop speculative lookahead up front, not just on success: a restore
     * that fails midway may already have overwritten GPU state the carry
     * and pendings were conditioned on */
    s->spec.spec_carry_valid = false;
    s->spec.dspark_n_pending = 0;
    spec_quench_reset(s);
    uint64_t remaining = payload_bytes;
    uint32_t h[PULSAR_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < PULSAR_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != PULSAR_SESSION_PAYLOAD_MAGIC || h[1] != PULSAR_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    pulsar_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != PULSAR_N_LAYER || h[9] != PULSAR_N_HEAD_DIM ||
        h[10] != PULSAR_N_INDEXER_HEAD_DIM || h[11] != PULSAR_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different Pulsar layout");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout.
     * Old checkpoints remain valid as long as the raw KV window matches. */
    (void)saved_prefill_cap;
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)PULSAR_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[PULSAR_MAX_LAYER];
    uint32_t n_index_comp[PULSAR_MAX_LAYER];
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    if (pulsar_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV restore");
        return 1;
    }
    s->checkpoint_valid = false;

    uint8_t *buf = (uint8_t *)xmalloc(PULSAR_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < PULSAR_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_raw_row(fp, g, il, phys,
                                      buf, PULSAR_SESSION_IO_CHUNK, &remaining, err, errlen);
        }
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (gpu_graph_attn_pack_enabled()) {
            rc = payload_read_attn_comp_pack(fp, g, il,
                                             n_comp[il],
                                             buf,
                                             PULSAR_SESSION_IO_CHUNK,
                                             &remaining,
                                             err,
                                             errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[il] * PULSAR_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          PULSAR_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   PULSAR_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   PULSAR_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_index_comp(fp, g, il,
                                         n_index_comp[il],
                                         buf,
                                         PULSAR_SESSION_IO_CHUNK,
                                         &remaining,
                                         err,
                                         errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       PULSAR_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       PULSAR_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }
    if (pulsar_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    /* a restored state invalidates any in-flight speculative lookahead: the
     * carry token, pre-drafted pendings, AND the drafter's context-KV ring
     * were all conditioned on the replaced state. Leaving the ring makes the
     * next drafts (and therefore the verify batch shapes) depend on whatever
     * ran before the restore — the source of run-to-run tie flips. */
    s->spec.spec_carry_valid = false;
    s->spec.dspark_n_pending = 0;
    spec_quench_reset(s);
    for (int li = 0; li < 3; li++) g->dspark_n_raw[li] = 0;
    g->dspark_prompt_n = 0;
    return 0;
}



int pulsar_session_save_snapshot(pulsar_session *s, pulsar_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    const uint64_t bytes = pulsar_session_payload_bytes(s);
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = (uint8_t *)realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = pulsar_session_save_payload(s, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}



int pulsar_session_load_snapshot(pulsar_session *s, const pulsar_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = pulsar_session_load_payload(s, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}



void pulsar_session_snapshot_free(pulsar_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}



void pulsar_engine_dump_tokens(pulsar_engine *e, const pulsar_tokens *tokens) {
    dump_tokens(&e->vocab, tokens);
}



int pulsar_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    pulsar_model model;
    pulsar_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    vocab_load(&vocab, &model);
    tokenize_rendered_chat_vocab(&vocab, text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}



static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "pulsar: failed to stat imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fprintf(stderr, "pulsar: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "pulsar: failed to open imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    size_t n = (size_t)st.st_size;
    char *buf = (char *)xmalloc(n + 1);
    if (n != 0 && fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "pulsar: failed to read imatrix dataset %s\n", path);
        fclose(fp);
        free(buf);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "pulsar: failed to close imatrix dataset %s: %s\n", path, strerror(errno));
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = n;
    return true;
}



static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}



int pulsar_engine_collect_imatrix(pulsar_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) {
    if (!e || !dataset_path || !output_path) return 1;
    if (e->backend != PULSAR_BACKEND_CUDA || !e->gpu_ready) {
        fprintf(stderr, "pulsar: imatrix collection currently requires --cuda\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 32768;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    const pulsar_model *model = &e->model;
    const pulsar_weights *weights = &e->weights;
    const uint32_t prefill_cap =
        gpu_graph_prefill_cap_for_prompt(ctx_size, e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);

    pulsar_gpu_graph g;
    bool ok = gpu_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false);
    if (!ok) {
        fprintf(stderr, "pulsar: failed to allocate imatrix GPU graph runtime\n");
        free(dataset);
        return 1;
    }
    g.quality = e->quality;

    pulsar_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, prefill_cap, dataset_path)) {
        fprintf(stderr, "pulsar: failed to allocate imatrix collector\n");
        gpu_graph_free(&g);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "pulsar: collecting routed-MoE imatrix from %s (model=%s, layers=%u, experts=%u, ctx=%d, chunk=%u)\n",
            dataset_path, PULSAR_MODEL_SHAPE_NAME, PULSAR_N_LAYER, PULSAR_N_EXPERT, ctx_size, prefill_cap);

    int prompts_done = 0;
    int tokens_done = 0;
    char *cursor = dataset;
    const char *marker_lit = "===== DS4_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = strstr(cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }

        char *next = strstr(start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            token_vec prompt = {0};
            pulsar_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                if (!gpu_graph_reset_prefill_state(&g)) {
                    fprintf(stderr, "pulsar: failed to reset imatrix graph state\n");
                    ok = false;
                } else if ((uint32_t)prompt.len > prefill_cap) {
                    ok = gpu_graph_prefill_chunked_range(&g, model, weights,
                                                           &prompt, 0,
                                                           (uint32_t)prompt.len,
                                                           NULL, false,
                                                           NULL, NULL,
                                                           NULL, NULL,
                                                           &collector,
                                                           NULL, NULL, NULL);
                } else {
                    ok = gpu_graph_prefill_layer_major(&g, model, weights,
                                                         &prompt, 0,
                                                         (uint32_t)prompt.len,
                                                         NULL, false,
                                                         &collector,
                                                         NULL, NULL);
                }
                if (!ok) {
                    fprintf(stderr, "pulsar: imatrix prefill failed at prompt %d\n", prompts_done + 1);
                    token_vec_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "pulsar: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)collector.observed_routes);
                    fflush(stderr);
                }
            }
            token_vec_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = imatrix_collector_save(&collector, weights, output_path);
        if (ok) {
            fprintf(stderr,
                    "pulsar: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }

    imatrix_collector_free(&collector);
    gpu_graph_free(&g);
    free(dataset);
    return ok ? 0 : 1;
}



int pulsar_engine_generate_argmax(
        pulsar_engine        *e,
        const pulsar_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        pulsar_token_emit_fn  emit,
        pulsar_generation_done_fn done,
        void              *emit_ud,
        pulsar_session_progress_fn progress,
        void              *progress_ud) {
    const pulsar_model *model = &e->model;
    const pulsar_vocab *vocab = &e->vocab;
    const pulsar_weights *weights = &e->weights;

    if (pulsar_backend_uses_graph(e->backend)) {
        if (!e->gpu_ready) {
            fprintf(stderr, "pulsar: %s generation requested but the graph backend is unavailable\n",
                    pulsar_backend_name(e->backend));
            return 1;
        }
        return generate_gpu_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size, e->quality,
                                            e->prefill_chunk,
                                            e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale,
                                            emit, done, emit_ud,
                                            progress, progress_ud);
    }
    return 1;
}



int pulsar_engine_gpu_graph_test(pulsar_engine *e, const pulsar_tokens *prompt) {
    if (!e->gpu_ready) {
        fprintf(stderr, "pulsar: %s graph test requested but backend is unavailable\n",
                pulsar_backend_name(e->backend));
        return 1;
    }
    return gpu_graph_decode_test(&e->model, &e->weights, prompt, e->quality);
}



int pulsar_engine_head_test(pulsar_engine *e, const pulsar_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "pulsar: head test requires a non-empty prompt\n");
        return 1;
    }

    const pulsar_model *model = &e->model;
    const pulsar_vocab *vocab = &e->vocab;
    const pulsar_weights *weights = &e->weights;
    const pulsar_layer_weights *layer0 = &weights->layer[0];

    float *prompt_embd = (float *)xmalloc((size_t)prompt->len * PULSAR_N_EMBD * sizeof(prompt_embd[0]));
    embed_prompt(model, weights, prompt, PULSAR_N_EMBD, prompt_embd);

    const uint32_t n_hc = PULSAR_N_HC;
    float *hc0 = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(hc0[0]));
    float *residual_hc = (float *)xmalloc((size_t)n_hc * PULSAR_N_EMBD * sizeof(residual_hc[0]));
    float hc_post[4];
    float hc_comb[16];
    layer_attn_pre_one(model, layer0,
        prompt_embd + (uint64_t)(prompt->len - 1) * PULSAR_N_EMBD,
        hc0, residual_hc, hc_post, hc_comb);
    print_vec_stats("blk.0 attn_pre", hc0, PULSAR_N_EMBD);

    float *attn_norm0 = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(attn_norm0[0]));
    layer_attn_norm_one(attn_norm0, model, layer0, hc0);

    const uint64_t q_dim = (uint64_t)PULSAR_N_HEAD * PULSAR_N_HEAD_DIM;
    float *q0 = (float *)xmalloc((size_t)q_dim * sizeof(q0[0]));
    layer_q_projection_normed_one(model, layer0, attn_norm0, q0);
    print_vec_stats("blk.0 q", q0, q_dim);

    float *kv0 = (float *)xmalloc((size_t)PULSAR_N_HEAD_DIM * sizeof(kv0[0]));
    layer_kv_projection_normed_one(model, layer0, attn_norm0, kv0);
    print_vec_stats("blk.0 kv", kv0, PULSAR_N_HEAD_DIM);
    rope_tail_layer_inplace(q0, PULSAR_N_HEAD, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    rope_tail_layer_inplace(kv0, PULSAR_N_HEAD_KV, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv0, PULSAR_N_HEAD_DIM, PULSAR_N_ROT);
    f16_round_inplace_cpu(kv0, PULSAR_N_HEAD_DIM);

    float *attn_heads = (float *)xmalloc((size_t)q_dim * sizeof(attn_heads[0]));
    layer_attention_one(attn_heads, model, layer0, q0, kv0);
    print_vec_stats("blk.0 attn_heads", attn_heads, q_dim);
    rope_tail_layer_inplace(attn_heads, PULSAR_N_HEAD, PULSAR_N_HEAD_DIM, PULSAR_N_ROT, (uint32_t)(prompt->len - 1), 0, true);

    float *attn_out = (float *)xmalloc((size_t)PULSAR_N_EMBD * sizeof(attn_out[0]));
    layer_grouped_out_one(attn_out, model, layer0, attn_heads);
    print_vec_stats("blk.0 attn_out", attn_out, PULSAR_N_EMBD);

    float *after_attn_hc = (float *)xmalloc((size_t)n_hc * PULSAR_N_EMBD * sizeof(after_attn_hc[0]));
    hc_post_one(after_attn_hc, attn_out, residual_hc, hc_post, hc_comb, PULSAR_N_EMBD, n_hc);
    print_vec_stats("blk.0 after_attn_hc", after_attn_hc, (uint64_t)n_hc * PULSAR_N_EMBD);

    float *after_ffn_hc = (float *)xmalloc((size_t)n_hc * PULSAR_N_EMBD * sizeof(after_ffn_hc[0]));
    layer_ffn_one(after_ffn_hc, model, layer0, after_attn_hc, 0, prompt->v[prompt->len - 1],
                  NULL, 0.0f, true);
    print_vec_stats("blk.0 after_ffn_hc", after_ffn_hc, (uint64_t)n_hc * PULSAR_N_EMBD);

    float *logits = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(logits[0]));
    output_logits_one(logits, model, weights, after_ffn_hc);
    print_vec_stats("logits", logits, PULSAR_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after native blk.0 slice:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(after_ffn_hc);
    free(after_attn_hc);
    free(attn_out);
    free(attn_heads);
    free(kv0);
    free(q0);
    free(attn_norm0);
    free(residual_hc);
    free(hc0);
    free(prompt_embd);
    return 0;
}






int pulsar_engine_open(pulsar_engine **out, const pulsar_engine_options *opt) {
    pulsar_engine *e = (pulsar_engine *)xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->dspark_model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->prefill_chunk = opt->prefill_chunk;
    /* Default draft depth 3: the measured v5mx optimum (2026-07-17 k-sweep on
     * the shipped ds4flash build at the tau=0.25 conf-sched default, quench
     * disarmed, conf-sched trimming active). k=3 beats k=5 by +15% structured
     * to +32% prose served decode; distribution-preserving (exact verify) —
     * byte-identical on structured, near-tie-equivalent on greedy prose (the
     * verify-width change flips ~1-ULP argmax ties, same class as yield-quench).
     * The DSpark drafter forward is autoregressive, so its cost scales with the
     * chain length ON TOP of the verify rows — ms/accepted-token stays flat
     * ~41-46 ms across k, i.e. depth never amortizes, so shallower wins.
     * The prior default 5 was a compact-model figure (2026-07-09, conf3 head,
     * tau 0.35) that does not hold on shipped v5mx at the tau=0.25 default. */
    e->dspark_draft_tokens = opt->dspark_draft_tokens > 0 ? opt->dspark_draft_tokens : 3;
    if (e->dspark_draft_tokens > 16) e->dspark_draft_tokens = 16;
    e->dspark_confidence = opt->dspark_confidence > 0.0f ? opt->dspark_confidence : 0.5f;
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "pulsar: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = pulsar_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    pulsar_acquire_instance_lock();

    const bool graph_backend = pulsar_backend_uses_graph(opt->backend);
    if (graph_backend) pulsar_linux_graph_backend_set_oom_score(opt->backend);
    model_open(&e->model, opt->model_path, graph_backend, !opt->inspect_only);
    if (opt->warm_weights) model_warm_weights(&e->model);
    if (!opt->inspect_only) vocab_load(&e->vocab, &e->model);
    config_validate_model(&e->model);
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        const char *sep = strrchr(opt->expert_overlay, ':');
        if (!sep || sep == opt->expert_overlay || !sep[1]) {
            fprintf(stderr, "pulsar: --expert-overlay expects FILE:PREFIX (e.g. donor.gguf:blk.17.)\n");
            pulsar_engine_close(e);
            *out = NULL;
            return 1;
        }
        char overlay_path[4096];
        const size_t path_len = (size_t)(sep - opt->expert_overlay);
        if (path_len >= sizeof(overlay_path)) {
            fprintf(stderr, "pulsar: --expert-overlay path is too long\n");
            pulsar_engine_close(e);
            *out = NULL;
            return 1;
        }
        memcpy(overlay_path, opt->expert_overlay, path_len);
        overlay_path[path_len] = '\0';
        model_open(&e->overlay_model, overlay_path, graph_backend, false);
        e->overlay_ready = true;
        /* PREFIX is a comma-separated list so several layers can be swapped
         * in one run (e.g. compose "anchor + candidate" from a cheap base
         * without materializing the combined model as a file). */
        char prefixes[2048];
        const size_t plist_len = strlen(sep + 1);
        if (plist_len >= sizeof(prefixes)) {
            fprintf(stderr, "pulsar: --expert-overlay prefix list is too long\n");
            pulsar_engine_close(e);
            *out = NULL;
            return 1;
        }
        memcpy(prefixes, sep + 1, plist_len + 1);
        uint32_t swapped = 0;
        for (char *p = strtok(prefixes, ","); p; p = strtok(NULL, ",")) {
            const uint32_t n = model_apply_expert_overlay(&e->model, &e->overlay_model, p);
            if (n == 0) {
                fprintf(stderr, "pulsar: --expert-overlay prefix '%s' matched no routed-expert tensors\n",
                        p);
                pulsar_engine_close(e);
                *out = NULL;
                return 1;
            }
            swapped += n;
        }
        fprintf(stderr, "pulsar: expert overlay: %u tensors swapped in from %s (prefixes %s)\n",
                swapped, overlay_path, sep + 1);
    }
    weights_bind(&e->weights, &e->model);
    if (opt->inspect_only) {
        *out = e;
        return 0;
    }
    if (!opt->dspark_disable && model_find_tensor(&e->model, "dspark.main_proj.weight")) {
        /* Drafter merged into the main GGUF: bind from the main model and
         * alias dspark_model to it by value (same map/fd; every dspark call
         * site reads e->dspark_model, and close is guarded on dspark_external
         * so the shared mapping is only torn down once). */
        dspark_weights_bind(&e->dspark_weights, &e->model);
        e->dspark_model = e->model;
        e->dspark_external = false;
        e->dspark_ready = true;
        fprintf(stderr, "pulsar: DSpark drafter found in model (draft=%d, confidence=%.2f)\n",
                e->dspark_draft_tokens,
                (double)e->dspark_confidence);
    } else if (!opt->dspark_disable && opt->dspark_path && opt->dspark_path[0]) {
        model_open(&e->dspark_model, opt->dspark_path, graph_backend, true);
        dspark_weights_bind(&e->dspark_weights, &e->dspark_model);
        e->dspark_external = true;
        e->dspark_ready = true;
        fprintf(stderr, "pulsar: DSpark support model loaded: %s (draft=%d, confidence=%.2f)\n",
                opt->dspark_path,
                e->dspark_draft_tokens,
                (double)e->dspark_confidence);
    }

    if (graph_backend) {
        e->gpu_ready = pulsar_gpu_init() != 0;
        if (!e->gpu_ready) {
            fprintf(stderr, "pulsar: %s backend unavailable; aborting startup\n",
                    pulsar_backend_name(e->backend));
            pulsar_engine_close(e);
            *out = NULL;
            return 1;
        }
        pulsar_gpu_set_quality(e->quality);
        (void)pulsar_gpu_set_model_fd(e->model.fd);
        const int model_map_ok =
            pulsar_gpu_set_model_map_range(e->model.map,
                                        e->model.size,
                                        e->model.tensor_data_pos,
                                        e->model.size - e->model.tensor_data_pos,
                                        e->model.max_tensor_bytes);
        if (!model_map_ok) {
            fprintf(stderr,
                    "pulsar: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    pulsar_backend_name(e->backend));
            pulsar_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->dspark_ready && e->dspark_external &&
            !pulsar_gpu_set_model_map_range(e->dspark_model.map,
                                           e->dspark_model.size,
                                           e->dspark_model.tensor_data_pos,
                                           e->dspark_model.size - e->dspark_model.tensor_data_pos,
                                           e->dspark_model.max_tensor_bytes))
        {
            fprintf(stderr,
                    "pulsar: %s failed to map DSpark model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    pulsar_backend_name(e->backend));
            pulsar_engine_close(e);
            *out = NULL;
            return 1;
        }
        (void)pulsar_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        if (!accelerator_cache_model_tensors(e->backend, &e->model,
                                             NULL, NULL, 0,
                                             e->dspark_ready ? NULL : "dspark.")) {
            fprintf(stderr, "pulsar: %s failed to prepare optional model cache\n",
                    pulsar_backend_name(e->backend));
            pulsar_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->dspark_ready && e->dspark_external) {
            (void)pulsar_gpu_set_model_fd_for_map(e->dspark_model.fd, e->dspark_model.map);
            if (!accelerator_cache_model_tensors(e->backend, &e->dspark_model,
                                                 NULL, NULL, 0, NULL)) {
                fprintf(stderr, "pulsar: %s failed to prepare optional DSpark model cache\n",
                        pulsar_backend_name(e->backend));
                pulsar_engine_close(e);
                *out = NULL;
                return 1;
            }
            (void)pulsar_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        }
        if (e->overlay_ready &&
            !accelerator_prepare_expert_overlay(e->backend, &e->model,
                                                &e->overlay_model)) {
            fprintf(stderr, "pulsar: %s failed to prepare expert-overlay spans\n",
                    pulsar_backend_name(e->backend));
            pulsar_engine_close(e);
            *out = NULL;
            return 1;
        }
        fprintf(stderr, "pulsar: %s backend initialized for graph diagnostics\n",
                pulsar_backend_name(e->backend));

        /* One MoE-tier boot line so a silent slow tier is no longer silent:
         * resolved expert weight type (grouped-CUTLASS type-40 vs per-expert
         * tiled type-39) + the resolved gate/up MXFP4 tile width (NT16/NT8).
         * Reuses the bound-layer expert tensors and the CUDA tile-width
         * accessor; log-only, runs once at open on every GPU serve. */
        {
            /* Count the tier PER LAYER.  Reporting the first bound layer's type
             * as "the" tier is actively misleading on a heterogeneous model:
             * v5mx has only 5 uniform-MXFP4 layers, so the old line printed
             * "per-expert-tiled" while a dozen layers were in fact running the
             * grouped CUTLASS path (2026-07-21 profile).  This observability
             * exists to catch a SILENT fall to a slow tier, so it has to be
             * truthful about the mix or it defeats its own purpose. */
            uint32_t n_grouped = 0, n_tiled = 0, n_routed = 0;
            const pulsar_layer_weights *ml = NULL;
            for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
                const pulsar_layer_weights *l = &e->weights.layer[il];
                if (!l->ffn_gate_exps || !l->ffn_down_exps) continue;
                if (!ml) ml = l;                    /* first routed layer, for the type sample */
                n_routed++;
                /* The grouped/GEMV CUTLASS dispatch is entered only when BOTH
                 * gate and down experts are type-40 (see the moe.cu batch
                 * predicate); any other mix takes the per-expert tiled path. */
                if (l->ffn_gate_exps->type == PULSAR_TENSOR_CUTLASS_MXFP4 &&
                    l->ffn_down_exps->type == PULSAR_TENSOR_CUTLASS_MXFP4) n_grouped++;
                else n_tiled++;
            }
            if (ml) {
                const uint32_t gt = ml->ffn_gate_exps->type;
                const uint32_t dt = ml->ffn_down_exps->type;
                fprintf(stderr,
                        "pulsar: MoE expert tier: %u/%u layers grouped-CUTLASS, %u/%u per-expert-tiled "
                        "(first routed layer gate=%s(%u) down=%s(%u)) mxfp4 tile=NT%u\n",
                        n_grouped, n_routed, n_tiled, n_routed,
                        tensor_type_name(gt), gt, tensor_type_name(dt), dt,
                        pulsar_gpu_moe_mxfp4_tile_width());
            }
        }
    }

    *out = e;
    return 0;
}



void pulsar_engine_summary(pulsar_engine *e) {
    model_summary(&e->model);
}



int pulsar_engine_vocab_size(pulsar_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}



/* The engine's logits ROW WIDTH — the shape profile's n_vocab, which is what
 * every logits buffer the engine writes is strided by.  This is NOT
 * pulsar_engine_vocab_size (the tokenizer table length): the loader never checks
 * the two against each other, and sizing a logits buffer from the tokenizer
 * length is exactly the mismatch that produced an unbounded-logits write. */
int pulsar_engine_logits_width(const pulsar_engine *e) {
    return e ? (int)PULSAR_N_VOCAB : 0;
}



const char *pulsar_engine_model_name(pulsar_engine *e) {
    (void)e;
    return PULSAR_MODEL_SHAPE_NAME;
}

void pulsar_engine_spec_metrics(pulsar_engine *e, pulsar_spec_metrics *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!e) return;
    out->accepted_tokens = e->spec_accepted_tokens;
    out->draft_tokens = e->spec_draft_tokens;
    out->num_drafts = e->spec_num_drafts;
    out->gen_tokens = e->spec_gen_tokens;
    for (int i = 0; i < 16; i++) out->accepted_per_pos[i] = e->spec_accepted_per_pos[i];
    out->max_draft = e->dspark_draft_tokens;
    out->has_dspark = e->dspark_ready;
}



/* Per-SESSION cumulative DSpark counters (accepted/draft/drafts/gen). Same
 * semantics as pulsar_engine_spec_metrics but scoped to one session, so a caller
 * can diff a snapshot across a single request for a per-response accept-rate.
 * accepted_per_pos is left zero (the per-position waterfall stays a /metrics,
 * cross-request concern). */
void pulsar_session_spec_metrics(const pulsar_session *s, pulsar_spec_metrics *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!s) return;
    out->accepted_tokens = s->spec.spec_accepted_tokens;
    out->draft_tokens = s->spec.spec_draft_tokens;
    out->num_drafts = s->spec.spec_num_drafts;
    out->gen_tokens = s->spec.spec_gen_tokens;
    out->max_draft = s->engine ? s->engine->dspark_draft_tokens : 0;
    out->has_dspark = s->engine ? s->engine->dspark_ready : false;
}



int pulsar_engine_layer_count(pulsar_engine *e) {
    (void)e;
    return (int)PULSAR_N_LAYER;
}



uint64_t pulsar_engine_weights_resident_bytes(pulsar_engine *e) {
    if (!e) return 0;
    /* The GGUF(s) are mmap'd read-only and shared across every session, so this
     * is a single resident copy competing with per-session KV for the unified
     * memory budget.  A merged/embedded drafter lives inside e->model and is
     * already counted; an external drafter and an expert overlay map their own
     * files and are added when present. */
    uint64_t bytes = e->model.size;
    if (e->dspark_ready && e->dspark_external) bytes += e->dspark_model.size;
    if (e->overlay_ready) bytes += e->overlay_model.size;
    return bytes;
}



uint32_t pulsar_engine_layer_compress_ratio(pulsar_engine *e, uint32_t layer) {
    (void)e;
    if (layer >= PULSAR_N_LAYER) return 0;
    return pulsar_layer_compress_ratio(layer);
}



uint64_t pulsar_engine_hidden_f32_values(pulsar_engine *e) {
    (void)e;
    return (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD;
}



int pulsar_engine_model_id(pulsar_engine *e) {
    (void)e;
    return (int)PULSAR_MODEL_VARIANT;
}



void pulsar_engine_close(pulsar_engine *e) {
    if (!e) return;
    weights_free(&e->weights);
    vocab_free(&e->vocab);
    pulsar_threads_shutdown();
    /* Tear down GPU state (which cudaHostUnregisters the mmap'd weight ranges)
     * before munmap'ing the model — unmapping still-registered pages is UB. */
    pulsar_gpu_cleanup();
    if (e->dspark_ready && e->dspark_external) model_close(&e->dspark_model);
    if (e->overlay_ready) model_close(&e->overlay_model);
    model_close(&e->model);
    pulsar_release_instance_lock();
    free(e->directional_steering_dirs);
    free(e->directional_steering_file);
    free(e);
}



int pulsar_session_create(pulsar_session **out, pulsar_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 1;

    pulsar_session *s = (pulsar_session *)xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                        e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) {
        fprintf(stderr, "pulsar: no transformer layers are loaded\n");
        free(s);
        return 1;
    }
    /* Measure the true GPU cost of this session (allocator delta across the
     * create) so callers can reconcile admission estimates against reality. */
    const uint64_t alloc_before = pulsar_gpu_tensor_alloc_bytes_current();
    if (!gpu_graph_alloc_raw_cap(&s->graph, &e->weights, shape_layer,
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap,
                                   e->dspark_ready))
    {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    if (!gpu_graph_load_directional_steering(&s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        gpu_graph_free(&s->graph);
        free(s);
        return 1;
    }
    s->logits = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(s->logits[0]));
    if (e->dspark_ready) {
        if (!gpu_graph_init_dspark_target(&s->graph, e->dspark_weights.target_layer_ids)) {
            fprintf(stderr, "pulsar: failed to allocate DSpark graph buffers\n");
            gpu_graph_free(&s->graph);
            free(s->logits);
            free(s);
            return 1;
        }
    }
    s->resident_bytes = pulsar_gpu_tensor_alloc_bytes_current() - alloc_before;
    *out = s;
    return 0;
}



uint64_t pulsar_session_resident_bytes(const pulsar_session *s) {
    return s ? s->resident_bytes : 0;
}



/* TRUE total per-session GPU byte cost of pulsar_session_create at this context
 * size — the admission-control price of a session.  Derives prefill/raw caps
 * exactly like pulsar_session_create and includes the DSpark drafter graph state
 * when the engine has a drafter loaded, so callers cannot pass mismatched
 * parameters.  Built on the same sizing code as the allocator
 * (gpu_graph_session_bytes, gpu_diag.c); reconcile against
 * pulsar_session_resident_bytes after the create. */
uint64_t pulsar_engine_session_cost_bytes(pulsar_engine *e, int ctx_size) {
    if (!e || ctx_size <= 0) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    const uint32_t prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                                  e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) return 0;
    return gpu_graph_session_bytes(&e->weights, shape_layer, raw_cap,
                                   (uint32_t)ctx_size, prefill_cap,
                                   e->dspark_ready);
}

uint64_t pulsar_engine_session_cost_bytes_banked(pulsar_engine *e, int ctx_size,
                                              int n_banks) {
    if (!e || ctx_size <= 0 || n_banks < 1) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    const uint32_t prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                                  e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);
    const pulsar_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) return 0;
    return gpu_graph_session_bytes_banked(&e->weights, shape_layer, raw_cap,
                                          (uint32_t)ctx_size, prefill_cap,
                                          e->dspark_ready, (uint32_t)n_banks);
}

uint64_t pulsar_engine_demand_paged_bytes_per_bank(pulsar_engine *e, int ctx_size) {
    if (!e || ctx_size <= 0) return 0;
    if (!pulsar_backend_uses_graph(e->backend) || !e->gpu_ready) return 0;
    return gpu_graph_demand_paged_bytes_per_bank((uint32_t)ctx_size);
}

uint64_t pulsar_session_touched_kv_bytes(const pulsar_session *s) {
    if (!s) return 0;
    return gpu_graph_touched_kv_bytes(&s->graph);
}

/* Tier-2 task #55 increment 2b — per-bank physical evict/restore.  The caller
 * (server guard) must have snapshotted the bank's KV to DISK before evict (host
 * RAM reclaims nothing on unified memory) and repointed away from it; and after
 * restore-alloc it reloads the KV H2D from that snapshot. */
bool pulsar_session_bank_free_physical(pulsar_session *s, uint32_t bank) {
    if (!s) return false;
    return gpu_graph_bank_free_physical(&s->graph, bank);
}

bool pulsar_session_bank_alloc_physical(pulsar_session *s, uint32_t bank) {
    if (!s) return false;
    return gpu_graph_bank_alloc_physical(&s->graph, bank);
}

bool pulsar_session_bank_is_evicted(const pulsar_session *s, uint32_t bank) {
    if (!s) return false;
    return gpu_graph_bank_is_evicted(&s->graph, bank);
}

uint64_t pulsar_session_bank_touched_kv_bytes(pulsar_session *s, uint32_t bank) {
    if (!s) return 0;
    return gpu_graph_bank_touched_kv_bytes(&s->graph, bank);
}

uint64_t pulsar_session_quantum_growth_bytes_per_bank(pulsar_session *s, uint32_t q) {
    (void)s;
    return gpu_graph_quantum_growth_bytes_per_bank(q);
}

static bool bank_carry_ensure(pulsar_session *s);   /* defined below */

/* Set d's token contents to EXACTLY toks[0..n) (reusing d's owned buffer). The
 * fork paths use this to stamp a dst's committed frontier from the VALIDATED
 * request prefix, rather than relying on truncating a copied/live checkpoint —
 * that truncate silently no-ops when the target starts shorter than n (a fresh
 * dst==cur carry at len 0), which is exactly what made the server re-prefill
 * from 0 and skip zero work (the fork "fired" but delivered no TTFT gain). */
static void token_vec_set_prefix(token_vec *d, const int *toks, int n) {
    d->len = 0;
    for (int i = 0; i < n; i++) token_vec_push(d, toks[i]);
}

/* Tier-2 PATH-A (plan-33 inc A) — deep-copy one bank carry into another (src's
 * conversation continues on dst). Reuses d's owned heap buffers; no alias/leak. */
static void bank_carry_copy(pulsar_bank_carry *d, const pulsar_bank_carry *sc) {
    token_vec d_ck = d->checkpoint;              /* keep d's own token buffer */
    float   *d_logits = d->logits;
    float   *d_qrows = d->dspark_pending_qrows;
    uint32_t d_qcap = d->dspark_pending_qrows_cap;
    *d = *sc;                                    /* scalars + (aliased) heap ptrs — fixed below */
    d->checkpoint = d_ck;
    pulsar_tokens_copy(&d->checkpoint, &sc->checkpoint);
    d->logits = d_logits ? d_logits : (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
    if (sc->logits) memcpy(d->logits, sc->logits, (size_t)PULSAR_N_VOCAB * sizeof(float));
    d->dspark_pending_qrows = d_qrows;
    d->dspark_pending_qrows_cap = d_qcap;
    if (sc->dspark_pending_qrows && sc->dspark_pending_qrows_cap > 0) {
        if (d->dspark_pending_qrows_cap < sc->dspark_pending_qrows_cap) {
            d->dspark_pending_qrows = (float *)xrealloc(d->dspark_pending_qrows,
                (size_t)sc->dspark_pending_qrows_cap * sizeof(float));
            d->dspark_pending_qrows_cap = sc->dspark_pending_qrows_cap;
        }
        memcpy(d->dspark_pending_qrows, sc->dspark_pending_qrows,
               (size_t)sc->dspark_pending_qrows_cap * sizeof(float));
    }
}

/* Tier-2 PATH-A FULL-PREFIX FORK (plan-33 increment A). Clone bank `src`'s
 * committed KV + host carry into bank `dst` so dst continues src's conversation,
 * then the caller re-prefills only the divergent suffix. STRUCTURAL ANTI-
 * CONTAMINATION: memcmp the request's tokens[0..n_cached) against src's committed
 * history BEFORE any device write — a mismatch (or a shorter/absent history)
 * REFUSES (return non-zero) so the caller cold-prefills; the fork never runs on a
 * non-matching source. Increment A is full-prefix only: n_cached must equal src's
 * committed length (a shorter shared prefix is the partial case, increment C).
 * Pins src against the eviction guard for the duration of the clone. Returns 0 on
 * success, non-zero on refusal/failure (caller falls back to cold prefill). */
int pulsar_session_bank_fork(pulsar_session *s, uint32_t src, uint32_t dst,
                          const int *tokens, int n_cached) {
    if (!s || !tokens || n_cached < 0) return 1;
    pulsar_gpu_graph *g = &s->graph;
    if (g->banks.n_banks == 0 || src >= g->banks.n_banks ||
        dst >= g->banks.n_banks || src == dst) return 1;
    /* 1. VALIDATE against src's committed history BEFORE any device write. */
    const uint32_t cur = g->banks.n_banks ? g->banks.cur_bank : 0u;
    const token_vec *hist = NULL;
    if (src == cur) {
        hist = s->checkpoint_valid ? &s->checkpoint : NULL;
    } else if (s->bank_carry && src < s->bank_carry_n &&
               s->bank_carry[src].valid && s->bank_carry[src].checkpoint_valid) {
        hist = &s->bank_carry[src].checkpoint;
    }
    if (!hist || (int)hist->len != n_cached) return 1;      /* full-prefix only */
    for (int i = 0; i < n_cached; i++) {
        if (hist->v[i] != tokens[i]) return 1;              /* mismatch -> cold */
    }
    /* 2. Eviction pin src (the guard's victim picker must not free it mid-clone). */
    g->fork_pin[src] = 1u;
    /* 3. src must have physical to clone (inc A refuses an evicted source). */
    if (gpu_graph_bank_is_evicted(g, src)) { g->fork_pin[src] = 0u; return 1; }
    if (src == cur) gpu_graph_bank_counters_capture(g, src);  /* current frontier */
    /* 4-5. Device D2D clone (allocs dst physical if freed, mirrors counters). */
    if (!gpu_graph_bank_fork_copy(g, src, dst)) { g->fork_pin[src] = 0u; return 1; }
    /* 6. Copy src host carry -> dst so dst owns src's conversation state. */
    if (!bank_carry_ensure(s)) { g->fork_pin[src] = 0u; return 1; }
    if (src == cur) pulsar_session_bank_state_save(s, src);      /* refresh from live session */
    if (src < s->bank_carry_n && dst < s->bank_carry_n) {
        bank_carry_copy(&s->bank_carry[dst], &s->bank_carry[src]);
    }
    /* dst owns tokens[0..n_cached) as its committed frontier. SET it explicitly
     * from the validated request prefix (do not trust the copied carry alone). */
    if (dst < s->bank_carry_n) {
        pulsar_bank_carry *c = &s->bank_carry[dst];
        token_vec_set_prefix(&c->checkpoint, tokens, n_cached);
        c->checkpoint_valid = true;
        c->valid = true;
    }
    /* BUG FIX (server dst==cur): provision_bank installs the fresh dst as cur and
     * invalidates the live session (s->checkpoint.len -> 0) BEFORE this fork runs.
     * bank_frontier_tokens(dst) then reads the LIVE s->checkpoint, so unless we
     * SET it here bank_pos(dst) stays 0 and the server re-prefills the WHOLE
     * prompt (fork fires, zero work skipped). Install dst's graph frontier and
     * stamp s->checkpoint to tokens[0..n_cached). */
    if (dst == cur) {
        gpu_graph_bank_counters_install(g, dst);
        token_vec_set_prefix(&s->checkpoint, tokens, n_cached);
        s->checkpoint_valid = true;
        s->spec.spec_carry_valid = false;
        s->spec.dspark_n_pending = 0;
        s->mseq_dirty = false;
    }
    /* 7. Full fork has no ratio-4 boundary stash — keep the emit hook inactive.
     * The drafter ring is NOT cloned: zero dst's ring counters so the spec path
     * re-warms instead of reading another conversation's window. */
    g->ms_emit_keep[dst] = 0u;
    for (int i = 0; i < 3; i++) g->ms_dspark_n_raw[dst][i] = 0u;
    g->ms_dspark_prompt_n[dst] = 0u;
    g->ms_dspark_prompt_lo[dst] = 0u;
    /* 8. (dst suffix KV-growth admission charge is the server's job in a later
     *    increment; the engine clone copies existing KV, touching nothing new.) */
    g->fork_pin[src] = 0u;
    return 0;
}

bool pulsar_session_bank_fork_pinned(const pulsar_session *s, uint32_t bank) {
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return false;
    return s->graph.fork_pin[bank] != 0u;
}

/* Tier-2 PATH-A PARTIAL-CUT FORK (plan-33 increment C). Like pulsar_session_bank_fork
 * but for a request that shares only tokens[0..n_cached) with src's committed
 * history: the cut is ALIGNED DOWN to R = ((n_cached-4)/align)*align (align = 128,
 * the ratio LCM) so every ratio-128 layer cuts on a closed group and only ratio-4
 * layers straddle (boundary row R/4, byte-stashed + emit-restored). VALIDATES
 * tokens[0..R+4) against src's history BEFORE any device write (the boundary row
 * pools [R-4, R+4), so the prefix must match through R+4). dst's committed
 * history becomes tokens[0..R): the caller re-prefills [R, ...) — the replay's
 * first ratio-4 emit recomputes row R/4 and the emit hook byte-replaces it with
 * the stash. src==dst is the in-place truncate-reuse degenerate (no copies).
 * Returns 0 on success; non-zero refusal (mismatch, unaligned/short cut, evicted
 * src, wrapped-out ring) -> caller cold-prefills. */
int pulsar_session_bank_fork_partial(pulsar_session *s, uint32_t src, uint32_t dst,
                                  const int *tokens, int n_cached) {
    if (!s || !tokens || n_cached < 4) return 1;
    pulsar_gpu_graph *g = &s->graph;
    if (g->banks.n_banks == 0 || src >= g->banks.n_banks || dst >= g->banks.n_banks)
        return 1;
    const uint32_t align = pulsar_partial_fork_base_align();
    const uint32_t R = (uint32_t)((n_cached - 4) / (int)align) * align;
    if (R < align) return 1;                        /* cut too shallow */
    /* 1. VALIDATE tokens[0..R+4) vs src's committed history BEFORE any write. */
    const uint32_t cur = g->banks.cur_bank;
    const token_vec *hist = NULL;
    if (src == cur) {
        hist = s->checkpoint_valid ? &s->checkpoint : NULL;
    } else if (s->bank_carry && src < s->bank_carry_n &&
               s->bank_carry[src].valid && s->bank_carry[src].checkpoint_valid) {
        hist = &s->bank_carry[src].checkpoint;
    }
    if (!hist || hist->len < (int)(R + 4u)) return 1;
    for (uint32_t i = 0; i < R + 4u; i++) {
        if (hist->v[i] != tokens[i]) return 1;      /* mismatch -> cold */
    }
    /* 2. Pin src; refuse evicted. Snapshot src's host carry FIRST: state_save
     * re-captures the live frontier counters, which MUST happen before copy_cut
     * writes the CUT counters into ms[dst] (src==dst truncate would otherwise
     * be clobbered back to the live frontier). */
    g->fork_pin[src] = 1u;
    if (gpu_graph_bank_is_evicted(g, src)) { g->fork_pin[src] = 0u; return 1; }
    if (!bank_carry_ensure(s)) { g->fork_pin[src] = 0u; return 1; }
    if (src == cur) pulsar_session_bank_state_save(s, src);
    /* 3-4. Clone-with-cut (cut counters + boundary stash + keep threshold). */
    if (!gpu_graph_bank_fork_copy_cut(g, src, dst, R, (uint32_t)hist->len)) {
        g->fork_pin[src] = 0u;
        return 1;
    }
    /* 5. Host carry: dst owns tokens[0..R) as its committed history. Copy src's
     * carry for the auxiliary state (logits/dspark), then SET the checkpoint
     * EXPLICITLY to the validated tokens[0..R) — do not rely on truncating a
     * possibly-stale/short carry (a fresh dst carry starts at len 0, so the old
     * `checkpoint.len = R` set a length past the buffer's real contents). */
    if (src != dst && src < s->bank_carry_n && dst < s->bank_carry_n) {
        bank_carry_copy(&s->bank_carry[dst], &s->bank_carry[src]);
    }
    if (dst < s->bank_carry_n) {
        pulsar_bank_carry *c = &s->bank_carry[dst];
        token_vec_set_prefix(&c->checkpoint, tokens, (int)R);
        c->checkpoint_valid = true;
        c->valid = true;
        /* Position-stamped state beyond R is meaningless on dst. */
        c->spec.spec_carry_valid = false;
        c->spec.dspark_n_pending = 0;
        c->spec.dspark_pending_sampled = false;
        /* (no carried mseq_dirty: bank_state_restore clears the live flag
         * unconditionally once this bank's frontier counters are installed) */
    }
    /* Drafter ring not cloned (and cut anyway): re-warm from empty. */
    for (int i = 0; i < 3; i++) g->ms_dspark_n_raw[dst][i] = 0u;
    g->ms_dspark_prompt_n[dst] = 0u;
    g->ms_dspark_prompt_lo[dst] = 0u;
    /* 6. If dst is the installed bank (server forks onto a freshly provisioned
     * cur, or the truncate-reuse of cur), refresh the LIVE session state and SET
     * (not truncate-if-larger) s->checkpoint to tokens[0..R). BUG FIX: on the
     * server dst==cur is fresh (len 0), so `if (len > R) len = R` never fired ->
     * bank_pos(dst) stayed 0 -> the server re-prefilled the WHOLE prompt (fork
     * fired, zero work skipped, no TTFT gain). Stamp it to R explicitly. */
    if (dst == cur) {
        gpu_graph_bank_counters_install(g, dst);
        token_vec_set_prefix(&s->checkpoint, tokens, (int)R);
        s->checkpoint_valid = true;
        s->spec.spec_carry_valid = false;
        s->spec.dspark_n_pending = 0;
        s->mseq_dirty = false;
    }
    g->fork_pin[src] = 0u;
    return 0;
}

/* Tier-2 task #55 increment 2b — RAW per-bank comp/index KV disk snapshot. The
 * eviction guard's bit-identical mechanism (the transcript-keyed kv_cache is a
 * semantic prefix-reuse cache, NOT a bitwise continuation — evict-restore-gate
 * proved raw D2H->H2D is the only bit-identical path). D2H each layer's frontier
 * rows to a TRANSIENT staging buffer, write to `fp`, and FREE it immediately —
 * holding KV in host RAM reclaims nothing (unified pool). Precondition: `bank`
 * is the installed (cur) bank so its live frontier is captured. */
#define PULSAR_BANK_KV_MAGIC   0x4B564232u   /* "KVB2" */
#define PULSAR_BANK_KV_VERSION 1u

int pulsar_session_bank_kv_save(pulsar_session *s, uint32_t bank, FILE *fp,
                             char *err, size_t errlen) {
    if (!s || !fp) { payload_set_err(err, errlen, "bank kv save: bad args"); return 1; }
    pulsar_gpu_graph *g = &s->graph;
    if (g->banks.n_banks == 0 || bank >= g->banks.n_banks) {
        payload_set_err(err, errlen, "bank kv save: no pool / bad bank"); return 1;
    }
    gpu_graph_bank_counters_capture(g, bank);   /* ms_n_*[bank] <- live layer_n_* */
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? PULSAR_ENGINE_IDXFP4_ROWBYTES : (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    uint32_t hdr[4] = { PULSAR_BANK_KV_MAGIC, PULSAR_BANK_KV_VERSION, bank, (uint32_t)PULSAR_N_LAYER };
    if (fwrite(hdr, sizeof hdr, 1, fp) != 1) { payload_set_err(err, errlen, "bank kv save: header write"); return 1; }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        uint32_t cnt[2] = { g->ms_n_comp[bank][il], g->ms_n_index_comp[bank][il] };
        if (fwrite(cnt, sizeof cnt, 1, fp) != 1) { payload_set_err(err, errlen, "bank kv save: count write"); return 1; }
    }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t csz = (uint64_t)g->ms_n_comp[bank][il] * attn_row;
        if (csz) {
            uint8_t *buf = (uint8_t *)xmalloc((size_t)csz);
            pulsar_gpu_tensor *v = gpu_graph_bank_attn_comp_view(g, il, bank);
            int ok = v && pulsar_gpu_tensor_read(v, 0, buf, csz) != 0;
            pulsar_gpu_tensor_free(v);
            if (ok) ok = fwrite(buf, 1, (size_t)csz, fp) == (size_t)csz;
            free(buf);   /* TRANSIENT: freed before free_physical so RAM is reclaimed */
            if (!ok) { payload_set_err(err, errlen, "bank kv save: comp D2H/write"); return 1; }
        }
        if (ratio == 4) {
            const uint64_t isz = (uint64_t)g->ms_n_index_comp[bank][il] * idx_row;
            if (isz) {
                uint8_t *buf = (uint8_t *)xmalloc((size_t)isz);
                pulsar_gpu_tensor *v = gpu_graph_bank_index_comp_view(g, il, bank);
                int ok = v && pulsar_gpu_tensor_read(v, 0, buf, isz) != 0;
                pulsar_gpu_tensor_free(v);
                if (ok) ok = fwrite(buf, 1, (size_t)isz, fp) == (size_t)isz;
                free(buf);
                if (!ok) { payload_set_err(err, errlen, "bank kv save: index D2H/write"); return 1; }
            }
        }
    }
    return 0;
}

/* Restore a bank's comp/index KV from a raw snapshot: realloc physical (+base
 * table rebuild), reinstall the frontier counters, and H2D each layer's rows.
 * Leaves `bank` installed (cur). Host conversation state (checkpoint/pos) is the
 * caller's bank_carry, restored separately by pulsar_session_bank_state_restore. */
int pulsar_session_bank_kv_load(pulsar_session *s, uint32_t bank, FILE *fp,
                             char *err, size_t errlen) {
    if (!s || !fp) { payload_set_err(err, errlen, "bank kv load: bad args"); return 1; }
    pulsar_gpu_graph *g = &s->graph;
    if (g->banks.n_banks == 0 || bank >= g->banks.n_banks) {
        payload_set_err(err, errlen, "bank kv load: no pool / bad bank"); return 1;
    }
    uint32_t hdr[4];
    if (fread(hdr, sizeof hdr, 1, fp) != 1 || hdr[0] != PULSAR_BANK_KV_MAGIC ||
        hdr[1] != PULSAR_BANK_KV_VERSION || hdr[3] != (uint32_t)PULSAR_N_LAYER) {
        payload_set_err(err, errlen, "bank kv load: bad header"); return 1;
    }
    uint32_t comp_cnt[PULSAR_MAX_LAYER] = {0}, idx_cnt[PULSAR_MAX_LAYER] = {0};
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        uint32_t cnt[2];
        if (fread(cnt, sizeof cnt, 1, fp) != 1) { payload_set_err(err, errlen, "bank kv load: count read"); return 1; }
        comp_cnt[il] = cnt[0]; idx_cnt[il] = cnt[1];
    }
    /* Review finding 3 — TRANSACTIONAL restore. Validate every frontier count
     * against the per-layer cap BEFORE any device change: a short read or corrupt
     * header must never leave the bank advertising row counts over unwritten KV
     * (an OOB managed read on the next decode). */
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        if (pulsar_layer_compress_ratio(il) == 0) continue;
        if (comp_cnt[il] > g->layer_comp_cap[il] || idx_cnt[il] > g->layer_comp_cap[il]) {
            payload_set_err(err, errlen,
                            "bank kv load: frontier count exceeds cap (corrupt snapshot)");
            return 1;
        }
    }
    if (!gpu_graph_bank_alloc_physical(g, bank)) { payload_set_err(err, errlen, "bank kv load: alloc_physical OOM"); return 1; }
    /* H2D every row into the bank's OWN views (which address comp[il][bank]
     * directly — no repoint) BEFORE touching the live frontier. On any read/H2D
     * failure the bank's ms counters stay 0 (from free_physical) so it advertises
     * an EMPTY frontier — no OOB — and the caller fails the request. */
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? PULSAR_ENGINE_IDXFP4_ROWBYTES : (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t csz = (uint64_t)comp_cnt[il] * attn_row;
        if (csz) {
            uint8_t *buf = (uint8_t *)xmalloc((size_t)csz);
            int ok = fread(buf, 1, (size_t)csz, fp) == (size_t)csz;
            if (ok) { pulsar_gpu_tensor *v = gpu_graph_bank_attn_comp_view(g, il, bank);
                      ok = v && pulsar_gpu_tensor_write(v, 0, buf, csz) != 0;
                      pulsar_gpu_tensor_free(v); }
            free(buf);
            if (!ok) { payload_set_err(err, errlen, "bank kv load: comp read/H2D"); return 1; }
        }
        if (ratio == 4) {
            const uint64_t isz = (uint64_t)idx_cnt[il] * idx_row;
            if (isz) {
                uint8_t *buf = (uint8_t *)xmalloc((size_t)isz);
                int ok = fread(buf, 1, (size_t)isz, fp) == (size_t)isz;
                if (ok) { pulsar_gpu_tensor *v = gpu_graph_bank_index_comp_view(g, il, bank);
                          ok = v && pulsar_gpu_tensor_write(v, 0, buf, isz) != 0;
                          pulsar_gpu_tensor_free(v); }
                free(buf);
                if (!ok) { payload_set_err(err, errlen, "bank kv load: index read/H2D"); return 1; }
            }
        }
    }
    /* Every row is now on-device — COMMIT: repoint (make bank live), then install
     * the frontier. This is the single point after which the bank advertises its
     * rows, and each is backed by written KV. If repoint fails the ms counters stay
     * 0 (empty, safe). */
    if (!gpu_graph_bank_repoint(g, bank)) { payload_set_err(err, errlen, "bank kv load: repoint"); return 1; }
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        g->ms_n_comp[bank][il] = comp_cnt[il];
        g->ms_n_index_comp[bank][il] = idx_cnt[il];
    }
    gpu_graph_bank_counters_install(g, bank);   /* layer_n_* <- ms_n_*[bank] */
    /* plan-33 inc C: a disk-restored bank carries full committed state — any
     * stale fork boundary threshold must not fire on it. */
    g->ms_emit_keep[bank] = 0u;
    return 0;
}



void pulsar_session_free(pulsar_session *s) {
    if (!s) return;
    gpu_graph_free(&s->graph);
    token_vec_free(&s->checkpoint);
    pulsar_sample_scratch_free(&s->sample_scratch);
    pulsar_session_bank_carry_free(s);
    free(s->dspark_pending_qrows);
    free(s->spec_row_scratch);
    free(s->logits);
    free(s);
}



void pulsar_session_set_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}



void pulsar_session_set_display_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}



void pulsar_session_set_cancel(pulsar_session *s, pulsar_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}



static bool pulsar_session_cancelled(pulsar_session *s) {
    return s && s->cancel && s->cancel(s->cancel_ud);
}



static bool pulsar_session_cancelled_cb(void *ud) {
    return pulsar_session_cancelled((pulsar_session *)ud);
}



void pulsar_session_report_progress(pulsar_session *s, const char *event, int current, int total) {
    if (!s || !s->progress || !event) return;
    s->progress(s->progress_ud, event, current, total);
}


static void pulsar_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    pulsar_sync_progress *p = (pulsar_sync_progress *)ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}



/* Bring the live backend state to exactly the supplied token prefix.
 *
 * pulsar-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
int pulsar_session_sync(pulsar_session *s, const pulsar_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (pulsar_session_cancelled(s)) {
        snprintf(err, errlen, "interrupted");
        return PULSAR_SESSION_SYNC_INTERRUPTED;
    }
    pulsar_engine *e = s->engine;
    const char *backend_name = pulsar_backend_name(e->backend);

    /* a sync begins a new request: any carry left by a max-tokens/stop-string
     * truncated generation belongs to the previous request's distribution.
     * (position stamping alone misses a same-length full rebuild.) */
    s->spec.spec_carry_valid = false;
    /* Same argument, same blind spot: the pendings' position stamp cannot see a
     * rebuild that lands on the same length, and a sampled draft's q belongs to
     * the previous request's distribution. Dropping them here costs one draft
     * round at the start of a request and is the only guard that covers it. */
    s->spec.dspark_n_pending = 0;
    /* A sync begins a new request: re-arm the terminal yield quench. */
    spec_quench_reset(s);

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        pulsar_tokens_starts_with(prompt, &s->checkpoint))
    {
        const int suffix = prompt->len - s->checkpoint.len;
        const uint32_t resume_min = gpu_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_min) {
            bool cancelled = false;
            pulsar_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            bool ok = gpu_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        pulsar_session_note_prefill_progress,
                                                        &progress,
                                                        s->display_progress,
                                                        s->display_progress_ud,
                                                        NULL,
                                                        pulsar_session_cancelled_cb,
                                                        s,
                                                        &cancelled);
            if (cancelled) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = true;
                return PULSAR_SESSION_SYNC_INTERRUPTED;
            }
            if (!ok) {
                snprintf(err, errlen, "%s resumed prefill failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            pulsar_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            return 0;
        }

        for (int i = s->checkpoint.len; i < prompt->len; i++) {
            if (pulsar_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = true;
                return PULSAR_SESSION_SYNC_INTERRUPTED;
            }
            if (!gpu_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                                (uint32_t)prompt->v[i],
                                                (uint32_t)s->checkpoint.len,
                                                s->logits))
            {
                snprintf(err, errlen, "%s decode failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            token_vec_push(&s->checkpoint, prompt->v[i]);
        }
        return 0;
    }

    bool ok;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    if (!gpu_graph_reset_prefill_state(&s->graph)) {
        snprintf(err, errlen, "%s prefill state reset failed", backend_name);
        return 1;
    }
    /* The rebuild path is the one place classic per-bank truth is legitimately
     * re-established: reset_prefill_state zeroes layer_n_comp /
     * layer_n_index_comp, so the scalars stop holding any multiseq superset
     * and the prefill below refills them from zero against the installed
     * bank.  (The prefix-resume path above cannot be reached while dirty —
     * decode_multiseq clears checkpoint_valid, which that path gates on.) */
    s->mseq_dirty = false;
    if (s->prefill_cap < (uint32_t)prompt->len) {
        bool cancelled = false;
        pulsar_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ok = gpu_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         pulsar_session_note_prefill_progress, &progress,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         pulsar_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            s->checkpoint_valid = s->checkpoint.len > 0;
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
    } else {
        bool cancelled = false;
        ok = gpu_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         pulsar_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            return PULSAR_SESSION_SYNC_INTERRUPTED;
        }
    }
    if (!ok) {
        snprintf(err, errlen, "%s prefill failed", backend_name);
        s->checkpoint_valid = false;
        return 1;
    }
    pulsar_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    return 0;
}



/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool pulsar_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}



/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
pulsar_session_rewrite_result pulsar_session_rewrite_from_common(
        pulsar_session *s, const pulsar_tokens *prompt, int common,
        char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return PULSAR_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return PULSAR_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return pulsar_session_sync(s, prompt, err, errlen) == 0 ?
            PULSAR_SESSION_REWRITE_OK : PULSAR_SESSION_REWRITE_ERROR;
    }

    if (pulsar_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return PULSAR_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return PULSAR_SESSION_REWRITE_ERROR;
}



int pulsar_session_common_prefix(pulsar_session *s, const pulsar_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}



int pulsar_session_argmax(pulsar_session *s) {
    return sample_argmax(s->logits, PULSAR_N_VOCAB);
}



int pulsar_session_argmax_excluding(pulsar_session *s, int excluded_id) {
    if (!s || !s->logits) return -1;
    int best = -1;
    float best_logit = PULSAR_NEG_INF;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}



int pulsar_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return 0;
    return sample_top_p_min_p(logits, (uint32_t)n_vocab, temperature, top_k, top_p, min_p, rng);
}



int pulsar_session_sample(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    return sample_top_p_min_p(s->logits, PULSAR_N_VOCAB, temperature, top_k, top_p, min_p, rng);
}



int pulsar_session_top_logprobs(pulsar_session *s, pulsar_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > (int)PULSAR_N_VOCAB) k = (int)PULSAR_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = PULSAR_NEG_INF;
        out[i].logprob = PULSAR_NEG_INF;
    }

    float max_logit = PULSAR_NEG_INF;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : PULSAR_NEG_INF;
    }
    return k;
}



int pulsar_session_token_logprob(pulsar_session *s, int token, pulsar_token_score *out) {
    if (!s || !out || token < 0 || token >= (int)PULSAR_N_VOCAB) return 0;

    float max_logit = PULSAR_NEG_INF;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < PULSAR_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : PULSAR_NEG_INF;
    return 1;
}



int pulsar_session_copy_logits(pulsar_session *s, float *out, int cap) {
    if (!s || !out || cap < (int)PULSAR_N_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)PULSAR_N_VOCAB * sizeof(out[0]));
    return (int)PULSAR_N_VOCAB;
}



int pulsar_session_set_logits(pulsar_session *s, const float *logits, int n) {
    if (!s || !logits || n != (int)PULSAR_N_VOCAB) return 1;
    memcpy(s->logits, logits, (size_t)PULSAR_N_VOCAB * sizeof(s->logits[0]));
    return 0;
}



int pulsar_session_eval(pulsar_session *s, int token, char *err, size_t errlen) {
    if (!s) return 1;
    /* Fail loud rather than corrupt: after a multiseq step the graph's scalar
     * frontier counters hold a cross-bank superset, so this decode would emit
     * its compressor row at the superset index and attend over another bank's
     * rows — wrong logits, silently.  pulsar_session_sync re-establishes per-bank
     * state (rebuild path) and clears the flag. */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "session eval after a multiseq decode step: classic per-bank "
                 "state is stale (frontier counters hold a cross-bank "
                 "superset); re-sync the session first");
        return 1;
    }
    pulsar_engine *e = s->engine;
    if (!gpu_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                        (uint32_t)token,
                                        (uint32_t)s->checkpoint.len,
                                        s->logits))
    {
        snprintf(err, errlen, "%s decode failed", pulsar_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    /* a token evaluated outside the speculative path (tool injection, plain
     * fallback loops) advances the state past any in-flight carry */
    s->spec.spec_carry_valid = false;
    return 0;
}



/* Batched multi-session decode over the session's bank pool — see the pulsar.h
 * declaration for the caller contract and gpu_graph_decode_multiseq_batch
 * (imatrix.c) for the step mechanics.
 *
 * The session's own single-bank bookkeeping is not merely un-advanced, it is
 * INVALIDATED on success: a multiseq step leaves the scalar frontier
 * counters holding a cross-bank superset (never any single bank's truth) and
 * advances a bank's KV past s->checkpoint.
 *
 * TWO separate guards are needed, because they cover different callers:
 *
 *   checkpoint_valid = false  stops pulsar_session_sync from taking its
 *       prefix-resume path (it gates on checkpoint_valid alone), forcing the
 *       rebuild path — which resets the graph's prefill state and so
 *       re-establishes per-bank truth.
 *
 *   mseq_dirty = true  stops pulsar_session_eval.  checkpoint_valid does NOT
 *       cover it: pulsar_session_eval never reads checkpoint_valid — it calls
 *       gpu_graph_eval_token_raw_swa(..., s->checkpoint.len, ...)
 *       unconditionally, which reads g->layer_n_comp[il] (the cross-bank
 *       superset) as its emit row, writes the compressor row there, and
 *       attends over every row below it — a previous tenant's bytes when the
 *       current bank's true frontier is lower.  Wrong logits, silently.  So
 *       eval fails loud while dirty instead of corrupting.
 *
 * The caller owns per-bank histories and must re-establish per-bank state
 * explicitly to resume classic work on a bank: a fresh pulsar_session_sync
 * (rebuild path) clears both flags. */
#define PULSAR_MULTISEQ_ERR(...) do { \
        if (err && errlen) snprintf(err, errlen, __VA_ARGS__); \
    } while (0)
int pulsar_session_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs,
                                uint32_t n, float *logits, int logits_cap,
                                char *err, size_t errlen) {
    if (!s || !reqs || !logits || n == 0 || n > PULSAR_MSEQ_MAX) {
        PULSAR_MULTISEQ_ERR("multiseq decode: bad args (n=%u)", n);
        return 1;
    }
    /* The engine writes n rows of PULSAR_N_VOCAB floats; without a capacity the
     * readback would overflow a caller buffer sized from a different vocab
     * notion (pulsar_engine_vocab_size is the tokenizer table length, which the
     * loader never checks against the shape profile's n_vocab). */
    if (logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)n * PULSAR_N_VOCAB) {
        PULSAR_MULTISEQ_ERR("multiseq decode: logits capacity %d < %u rows x %u",
                         logits_cap, n, (unsigned)PULSAR_N_VOCAB);
        return 1;
    }
    pulsar_engine *e = s->engine;
    int32_t pos[PULSAR_MSEQ_MAX];
    int32_t bank[PULSAR_MSEQ_MAX];
    int tokens[PULSAR_MSEQ_MAX];
    for (uint32_t k = 0; k < n; k++) {
        pos[k] = reqs[k].pos;
        bank[k] = (int32_t)reqs[k].bank;
        tokens[k] = reqs[k].token;
    }
    /* decode_multiseq is decode-only (1 row per bank), so n_runs == n; no
     * out-param needed (the caller reads n logit rows). max_head_runs = 0 (all). */
    const int rc = gpu_graph_decode_multiseq_batch(&s->graph, &e->model,
                                                   &e->weights, tokens, pos,
                                                   bank, n, logits, NULL, 0u);
    if (rc == 0) {
        /* Recoverable: the driver rejected before arming the step, so nothing
         * was mutated — the upload writes ahead of it touch scratch only, and
         * every gpu_graph_multiseq_step_begin rejection point precedes its
         * first scalar write (the superset refresh) and its cur-bank capture.
         * The classic view is therefore still true: leave both flags alone so
         * the caller can fix the batch and retry, or fall back to classic. */
        PULSAR_MULTISEQ_ERR("multiseq decode step rejected (recoverable; "
                         "reason on stderr)");
        return 1;
    }
    /* The step was armed: the scalar counters hold a superset on success, and
     * on a fatal mid-sweep failure their state is unknown.  Both leave the
     * classic single-bank view stale (see above); only the diagnosis differs. */
    s->checkpoint_valid = false;
    s->mseq_dirty = true;
    s->spec.spec_carry_valid = false;
    if (rc == 1) return 0;
    PULSAR_MULTISEQ_ERR("multiseq decode step failed mid-sweep "
                     "(session state fatal)");
    return -1;
}
#undef PULSAR_MULTISEQ_ERR

/* plan-34 phase-2 increment 1 — mixed prefill+decode descriptor entry.
 *
 * BYTE-IDENTICAL to pulsar_session_decode_multiseq for a decode-only (1-row-per-bank)
 * batch: the ONLY difference is that the per-row descriptor scratch
 * (positions / seq_id / tokens) lives on the HEAP, sized to n_rows up to
 * prefill_cap, instead of the fixed [PULSAR_MSEQ_MAX] stack arrays. This establishes
 * the >PULSAR_MSEQ_MAX-row representation the fused mixed-batch step grows into over
 * increments 2-5. The kernel path (gpu_graph_decode_multiseq_batch + step_begin)
 * is UNCHANGED: it still bounds the current ROW count to PULSAR_MSEQ_MAX and keeps
 * PULSAR_MSEQ_MAX as the BANK-count bound. No mixing, no scheduler change, no algo
 * work this increment — the descriptor container is the only thing that moved. */
#define PULSAR_MIXED_ERR(...) do { \
        if (err && errlen) snprintf(err, errlen, __VA_ARGS__); \
    } while (0)
int pulsar_session_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs,
                             uint32_t n_rows, float *logits, int logits_cap,
                             uint32_t *out_n_rows, uint32_t max_head_runs,
                             char *err, size_t errlen) {
    if (out_n_rows) *out_n_rows = 0;
    if (!s || !reqs || !logits || n_rows == 0 ||
        n_rows > s->graph.prefill_cap) {
        PULSAR_MIXED_ERR("mixed decode: bad args (n_rows=%u prefill_cap=%u)",
                      n_rows, s ? s->graph.prefill_cap : 0u);
        return 1;
    }
    /* plan-34 inc 3: the engine emits ONE logit row per BANK RUN (last-of-run),
     * not one per token-row, so the caller need only size `logits` for n_runs =
     * the number of contiguous per-bank runs (<= PULSAR_MSEQ_MAX). Compute it here
     * for the capacity check; the engine returns it via out_n_rows. */
    uint32_t n_runs = 0;
    for (uint32_t k = 0; k < n_rows; k++)
        if (k + 1 == n_rows || reqs[k + 1].bank != reqs[k].bank) n_runs++;
    if (logits_cap < 0 || (uint64_t)logits_cap < (uint64_t)n_runs * PULSAR_N_VOCAB) {
        PULSAR_MIXED_ERR("mixed decode: logits capacity %d < %u runs x %u",
                      logits_cap, n_runs, (unsigned)PULSAR_N_VOCAB);
        return 1;
    }
    pulsar_engine *e = s->engine;
    /* HEAP descriptor scratch (positions / seq_id / tokens), sized to n_rows —
     * this is the whole point of the refactor: no [PULSAR_MSEQ_MAX] stack ceiling. */
    int32_t *pos = (int32_t *)xmalloc((size_t)n_rows * sizeof(*pos));
    int32_t *bank = (int32_t *)xmalloc((size_t)n_rows * sizeof(*bank));
    int *tokens = (int *)xmalloc((size_t)n_rows * sizeof(*tokens));
    for (uint32_t k = 0; k < n_rows; k++) {
        pos[k]    = reqs[k].pos;
        bank[k]   = (int32_t)reqs[k].bank;
        tokens[k] = reqs[k].token;
    }
    const int rc = gpu_graph_decode_multiseq_batch(&s->graph, &e->model,
                                                   &e->weights, tokens, pos,
                                                   bank, n_rows, logits, out_n_rows,
                                                   max_head_runs);
    /* The batch call consumed the descriptor synchronously (tokens copied to a
     * stack row, pos/seq uploaded to device in step_begin); the host scratch is
     * dead now regardless of rc. */
    free(pos); free(bank); free(tokens);
    if (rc == 0) {
        /* Recoverable rejection: nothing mutated (see pulsar_session_decode_multiseq
         * for the precise argument — every step_begin reject precedes its first
         * scalar write). Leave the classic view flags alone. */
        PULSAR_MIXED_ERR("mixed decode step rejected (recoverable; reason on stderr)");
        return 1;
    }
    /* Step was armed: scalar counters now hold a cross-bank superset; the classic
     * single-bank view is stale exactly as after a multiseq step. */
    s->checkpoint_valid = false;
    s->mseq_dirty = true;
    s->spec.spec_carry_valid = false;
    if (rc == 1) return 0;
    PULSAR_MIXED_ERR("mixed decode step failed mid-sweep (session state fatal)");
    return -1;
}
#undef PULSAR_MIXED_ERR



/* ===== Tier-2 PATH A: per-bank HOST carry (pulsar_bank_carry) ================
 *
 * gpu_graph_bank_repoint swaps only the DEVICE cache views; the session's HOST
 * per-conversation state (checkpoint history, host logits, the DSpark fused-
 * loop / spec-carry shadow) is single-instance and must be saved for the bank
 * we leave and restored for the bank we enter.  save/restore below are that
 * host half; the graph frontier counters ride gpu_graph_bank_counters_*, and
 * (Option F) the per-bank drafter ring rides gpu_graph_bank_repoint. */

static void bank_carry_free_one(pulsar_bank_carry *c) {
    if (!c) return;
    token_vec_free(&c->checkpoint);
    free(c->logits);
    free(c->dspark_pending_qrows);
    memset(c, 0, sizeof(*c));
}

void pulsar_session_bank_carry_free(pulsar_session *s) {
    if (!s || !s->bank_carry) return;
    for (uint32_t i = 0; i < s->bank_carry_n; i++) bank_carry_free_one(&s->bank_carry[i]);
    free(s->bank_carry);
    s->bank_carry = NULL;
    s->bank_carry_n = 0;
}

static bool bank_carry_ensure(pulsar_session *s) {
    const uint32_t n = gpu_graph_bank_pool_count(&s->graph);
    if (s->bank_carry && s->bank_carry_n == n) return true;
    if (s->bank_carry) pulsar_session_bank_carry_free(s);
    s->bank_carry = (pulsar_bank_carry *)xcalloc(n, sizeof(*s->bank_carry));
    s->bank_carry_n = n;
    return true;
}

int pulsar_session_bank_count(pulsar_session *s) {
    return s ? (int)gpu_graph_bank_pool_count(&s->graph) : 0;
}

int pulsar_session_bank_repoint(pulsar_session *s, uint32_t bank) {
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return 1;
    /* Pool disabled: bank 0 is the classic tensors, nothing to repoint. */
    if (s->graph.banks.n_banks == 0) return bank == 0 ? 0 : 1;
    return gpu_graph_bank_repoint(&s->graph, bank) ? 0 : 1;
}

void pulsar_session_bank_state_save(pulsar_session *s, uint32_t bank) {
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return;
    if (!bank_carry_ensure(s)) return;
    /* Graph frontier counters (attn/index comp; Option F also the drafter ring
     * counters) are captured on the graph side so a later install re-arms this
     * bank's per-bank truth. */
    gpu_graph_bank_counters_capture(&s->graph, bank);
    pulsar_bank_carry *c = &s->bank_carry[bank];
    /* heap-backed deep copies */
    pulsar_tokens_copy(&c->checkpoint, &s->checkpoint);
    if (!c->logits) c->logits = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(float));
    memcpy(c->logits, s->logits, (size_t)PULSAR_N_VOCAB * sizeof(float));
    if (s->dspark_pending_qrows && s->dspark_pending_qrows_cap > 0) {
        if (c->dspark_pending_qrows_cap < s->dspark_pending_qrows_cap) {
            c->dspark_pending_qrows = (float *)xrealloc(
                c->dspark_pending_qrows,
                (size_t)s->dspark_pending_qrows_cap * sizeof(float));
            c->dspark_pending_qrows_cap = s->dspark_pending_qrows_cap;
        }
        memcpy(c->dspark_pending_qrows, s->dspark_pending_qrows,
               (size_t)s->dspark_pending_qrows_cap * sizeof(float));
    }
    /* scalar mirrors */
    c->checkpoint_valid       = s->checkpoint_valid;
    /* Whole speculative/DSpark shadow in one assignment — a new field added to
     * pulsar_spec_carry_state is carried here for free (the old field-by-field
     * mirror was a silent-corruption footgun: miss one and the entering bank
     * inherits another conversation's speculative state).
     * s->mseq_dirty is NOT saved: it describes the graph's scalar frontier
     * counters, not this bank's conversation, and _restore re-establishes
     * per-bank frontier truth and clears it unconditionally. */
    c->spec = s->spec;
    c->valid = true;
}

bool pulsar_session_bank_state_restore(pulsar_session *s, uint32_t bank) {
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return false;
    /* Point device views (incl. Option F drafter ring) at this bank, and
     * re-arm its frontier counters — this is what makes clearing mseq_dirty
     * cheap and safe (per-bank truth is re-established without a re-prefill). */
    if (s->graph.banks.n_banks != 0 && !gpu_graph_bank_repoint(&s->graph, bank))
        return false;
    gpu_graph_bank_counters_install(&s->graph, bank);
    if (!s->bank_carry || bank >= s->bank_carry_n || !s->bank_carry[bank].valid) {
        /* No saved host state for a fresh bank: the counters_install above set
         * the (zeroed) frontier; leave the session's host shadow as the caller
         * primed it (a fresh sync just ran).  Clear the multiseq poison so
         * classic work resumes. */
        s->mseq_dirty = false;
        return true;
    }
    pulsar_bank_carry *c = &s->bank_carry[bank];
    pulsar_tokens_copy(&s->checkpoint, &c->checkpoint);
    memcpy(s->logits, c->logits, (size_t)PULSAR_N_VOCAB * sizeof(float));
    if (c->dspark_pending_qrows_cap > 0) {
        if (s->dspark_pending_qrows_cap < c->dspark_pending_qrows_cap) {
            s->dspark_pending_qrows = (float *)xrealloc(
                s->dspark_pending_qrows,
                (size_t)c->dspark_pending_qrows_cap * sizeof(float));
            s->dspark_pending_qrows_cap = c->dspark_pending_qrows_cap;
        }
        memcpy(s->dspark_pending_qrows, c->dspark_pending_qrows,
               (size_t)c->dspark_pending_qrows_cap * sizeof(float));
    }
    s->checkpoint_valid       = c->checkpoint_valid;
    /* Mirror of the save above: one assignment restores the whole shadow. */
    s->spec = c->spec;
    /* Cheap resume: per-bank frontier truth is now installed, so the multiseq
     * superset poison no longer applies to this bank. */
    s->mseq_dirty = false;
    return true;
}



/* ===== Tier-2 per-bank frontier READERS (server routing/metrics) ==========
 *
 * A bank-pooled server shares one pulsar_session across N conversation banks, so
 * pulsar_session_pos / _tokens / _common_prefix (which read the single live host
 * checkpoint) describe ONLY the currently-installed bank.  Reading them for a
 * non-live bank returns the wrong conversation's frontier.  These readers give
 * the correct per-bank answer without repointing the device or disturbing the
 * live bank: the live bank (banks.cur_bank) reads the authoritative live
 * checkpoint; every other bank reads its saved host carry (which the server
 * keeps current for idle banks by bank_state_save'ing at job end).  Pure host
 * reads — no CUDA, safe on the worker thread at routing/publish time. */
static const pulsar_tokens *bank_frontier_tokens(pulsar_session *s, uint32_t bank) {
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return NULL;
    const uint32_t cur = s->graph.banks.n_banks ? s->graph.banks.cur_bank : 0u;
    if (bank == cur) return s->checkpoint_valid ? &s->checkpoint : NULL;
    if (s->bank_carry && bank < s->bank_carry_n &&
        s->bank_carry[bank].valid && s->bank_carry[bank].checkpoint_valid)
        return &s->bank_carry[bank].checkpoint;
    return NULL;
}

int pulsar_session_bank_pos(pulsar_session *s, uint32_t bank) {
    const pulsar_tokens *t = bank_frontier_tokens(s, bank);
    return t ? t->len : 0;
}

const pulsar_tokens *pulsar_session_bank_tokens(pulsar_session *s, uint32_t bank) {
    return bank_frontier_tokens(s, bank);
}

int pulsar_session_bank_common_prefix(pulsar_session *s, uint32_t bank,
                                   const pulsar_tokens *prompt) {
    const pulsar_tokens *t = bank_frontier_tokens(s, bank);
    if (!t || !prompt) return 0;
    int n = t->len < prompt->len ? t->len : prompt->len;
    int i = 0;
    while (i < n && t->v[i] == prompt->v[i]) i++;
    return i;
}

void pulsar_session_note_committed_tokens(pulsar_session *s, const int *toks, int n) {
    if (!s || !toks || n <= 0) return;
    for (int i = 0; i < n; i++) token_vec_push(&s->checkpoint, toks[i]);
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
    const char *path = getenv("DS4_DSPARK_DUMP");
    if (!path || !path[0]) return;
    static int dumped = 0;
    const char *lim = getenv("DS4_DSPARK_DUMP_STEPS");
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
    if (lean < 0) lean = getenv("DS4_DSPARK_DUMP_LEAN") != NULL;
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
    const int dspark_stats = gpu_graph_env_flag("DS4_DSPARK_STATS", &dspark_stats_env);
    static int dtree_stats_env = -1;
    const int dtree_stats = gpu_graph_env_flag("DS4_DTREE_STATS", &dtree_stats_env);
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
        if (no_replay < 0) no_replay = getenv("DS4_DSPARK_REPLAY") == NULL;
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
                if (pulsar_session_eval(s, (int)replay[i], err, errlen) != 0) {
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
            const char *e2 = getenv("DS4_DSPARK_DUMP_ONPOLICY");
            opsteps = e2 && e2[0] ? atoi(e2) : 0;
        }
        static int opdone = 0;
        const char *oppath = getenv("DS4_DSPARK_DUMP_ONPOLICY_PATH");
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
                if (dump_ring < 0) dump_ring = getenv("DS4_DSPARK_DUMP_RING") != NULL;
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
int pulsar_session_generate_speculative(pulsar_session *s, float temperature, int top_k,
                                     float top_p, float min_p, uint64_t *rng,
                                     int max_tokens, int eos_token,
                                     int *accepted, int accepted_cap,
                                     char *err, size_t errlen) {
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
    if (!pulsar_engine_has_dspark(s->engine) || s->spec.spec_quenched) {
        if (pulsar_session_eval(s, first, err, errlen) != 0) return -1;
        accepted[0] = first;
        return 1;
    }
    return pulsar_session_eval_speculative_fused(s, first, max_tokens, eos_token,
                                              temperature, top_k, top_p, min_p, rng,
                                              accepted, accepted_cap, err, errlen);
}



int pulsar_session_eval_speculative_block(pulsar_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (!s || max_tokens <= 0 || accepted_cap <= 0 || !accepted) return 0;
    /* Same stale-classic-state guard as pulsar_session_eval (which the no-dspark
     * fallback below would otherwise hit one frame deeper). */
    if (s->mseq_dirty) {
        snprintf(err, errlen,
                 "speculative block eval after a multiseq decode step: classic "
                 "per-bank state is stale; re-sync the session first");
        return -1;
    }
    if (!pulsar_engine_has_dspark(s->engine) || s->spec.spec_quenched) {
        if (pulsar_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    /* The fused loop (one batched forward/step + transactional no-replay
     * rollback) is the default: measured 16.4 vs 15.2 t/s legacy-vs-baseline
     * and byte-identical deterministic output. PULSAR_DSPARK_LEGACY_LOOP restores
     * the old Step1+verify loop as an operational fallback. */
    static int fused_cache = -1;
    if (fused_cache < 0) fused_cache = getenv("DS4_DSPARK_LEGACY_LOOP") == NULL;
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
    const int dspark_stats = gpu_graph_env_flag("DS4_DSPARK_STATS", &dspark_stats_env);
    const double dspark_t0 = dspark_stats ? now_sec() : 0.0;
    double dspark_draft_ms = 0.0;
    int dspark_base0 = -1;   /* draft-forward's row-0 argmax (pre-markov) */
    double dspark_markov_ms = 0.0, dspark_snap_ms = 0.0, dspark_verify_ms = 0.0;
    const int dump_pos = s->checkpoint.len;  /* first_token's sequence position */

    /* Step 1: run target decode for the first token */
    if (pulsar_session_eval(s, first_token, err, errlen) != 0) return -1;

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
            const char *mc = getenv("DS4_DSPARK_MIN_CONF");
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
            if (pulsar_session_eval(s, refined_ids[i + 1], err, errlen) != 0) {
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



void pulsar_session_invalidate(pulsar_session *s) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->spec.dspark_n_pending = 0;
    s->spec.spec_carry_valid = false;
    spec_quench_reset(s);
    /* plan-33 inc C: an invalidated bank restarts from zero — a live keep
     * threshold would make the fresh prefill's early emits restore a STALE
     * boundary row over the new conversation's KV. Disarm it. */
    if (s->graph.banks.n_banks) {
        s->graph.ms_emit_keep[s->graph.banks.cur_bank] = 0u;
    }
    /* The drafter's context-KV ring must not survive into a new prompt: it was
     * never reset before, so in the server every request after the first
     * attended over the PREVIOUS request's window rows for its first ~128
     * generated tokens (and the drafter is near-useless without a valid
     * window: masked-window eval 4.7% vs 86% top-1). Positions are
     * drafter-relative, so restarting at 0 is exact. */
    for (int i = 0; i < 3; i++) s->graph.dspark_n_raw[i] = 0;
    s->graph.dspark_prompt_n = 0;
}



void pulsar_session_rewind(pulsar_session *s, int pos) {
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->spec.dspark_n_pending = 0;
    s->spec.spec_carry_valid = false;
    spec_quench_reset(s);
    /* Rewound positions' drafter rows are stale; empty the window (it refills
     * from the prompt capture on the next prefill, or from commits). */
    for (int i = 0; i < 3; i++) s->graph.dspark_n_raw[i] = 0;
    s->graph.dspark_prompt_n = 0;
}



int pulsar_session_pos(pulsar_session *s) {
    return s->checkpoint.len;
}



int pulsar_session_ctx(pulsar_session *s) {
    return s->ctx_size;
}



int pulsar_session_prefill_cap(pulsar_session *s) {
    return s ? (int)s->prefill_cap : 0;
}



/* Multi-session serving: is interrupting pulsar_session_sync() at a chunk
 * boundary (cancel callback) and re-issuing the sync bit-identical to letting
 * it run to completion?
 *
 * Two conditions must hold, and the return value encodes both:
 *
 *   - gpu_graph_prefill_chunked_range caps resumed (start != 0) chunks at
 *     raw_cap. If this session's cold chunks are larger (prefill_cap >
 *     raw_cap), a resumed prefill would re-chunk on different boundaries,
 *     changing batch shapes and therefore cuBLASLt algo selection; exact
 *     replay is lost. Return 0: the caller must not interrupt at all.
 *
 *   - Below gpu_graph_resume_prefill_min_tokens() remaining tokens,
 *     pulsar_session_sync extends the checkpoint by single-token decode evals
 *     instead of a final batched chunk. Interrupting with less than this left
 *     would change which path evaluates the tail. Return that minimum, so the
 *     caller only interrupts while (target - checkpoint) >= the returned
 *     value. (When resume is disabled via PULSAR_CUDA_RESUME_PREFILL_MIN<=0 the
 *     minimum is UINT32_MAX and the comparison never permits interruption.) */
uint32_t pulsar_session_prefill_quantum_min_suffix(const pulsar_session *s) {
    if (!s) return 0;
    if (s->graph.prefill_cap > s->graph.raw_cap) return 0;
    /* A cold (start==0) chunk loop trims each non-final chunk end DOWN to the
     * compress-ratio LCM, while a resumed (start!=0) loop snaps to absolute
     * prefill_cap boundaries. The two produce the same chunk ends only when
     * prefill_cap itself is LCM-aligned (true for the 4096/8192 defaults; a
     * hand-set --prefill-chunk may not be). */
    uint32_t align = 1;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t r = pulsar_layer_compress_ratio(il);
        if (r > 1 && align % r != 0) {
            uint32_t a = align, b = r;
            while (b) { const uint32_t t = a % b; a = b; b = t; }
            align *= r / a;
        }
    }
    if (align > 1 && s->graph.prefill_cap % align != 0) return 0;
    return gpu_graph_resume_prefill_min_tokens();
}

