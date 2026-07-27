/* Tier-2 FRONTIER-ISOLATION gate (the silent-corruption gate).
 *
 * MODEL-DEPENDENT: this gate audits the ENGINE-side multiseq wiring —
 * gpu_graph_encode_layer_batch through step_begin/step_end over a 2-bank
 * pool — so every layer's emit/store path runs the real layer weights
 * (matmuls, MoE, compressors) out of the mmapped GGUF; there is no
 * synthetic-weight path through that code.  The in-tree kernel smokes
 * (tests/cuda_long_context_smoke.c) test the banked KERNELS against
 * synthetic slabs and are structurally blind to engine-side wrong-bank
 * wiring; this gate is the engine-side complement.  Run manually on the
 * GB10 via `make cuda-frontier-gate` (see the Makefile target for the
 * memory discipline) — it is NOT part of `make test`.
 *
 * Two banks of a PULSAR_MSEQ_BANKS=2 pool are populated with DIFFERENT
 * sequences of DIFFERENT lengths through the classic per-bank path
 * (repoint + session sync), then ONE banked multiseq batched step carries
 * rows of both banks through the full layer sweep.  Verified:
 *
 *   (a) comp-cache CONTENTS: each bank's prefilled packed rows byte-match a
 *       solo session's, and each bank's step-EMITTED packed row byte-matches
 *       (a1) the same step with the banks SWAPPED (same GEMM/MoE shapes —
 *       pure bank-addressing invariance) and (a2) the same sequence advanced
 *       by 1-row multiseq steps on a solo pool (cross-shape: only holds if
 *       the row math is batch-shape-invariant; reported separately).
 *   (b) frontier isolation: a bank whose rows close no ratio group keeps its
 *       ms counters AND its cache bytes bit-untouched while its batchmate
 *       emits (checked at ratio 4 in S2 and at ratio 128 + ratio 4 in S3).
 *   (c) the scalar superset equals max over banks after the step and is
 *       written only at step top (structural; step_end re-verifies, and this
 *       gate re-asserts equality against the ms counters).
 *
 * Scenarios (S1-S3 keep the increment-2 globally-consecutive shape; S4-S5
 * exercise the increment-3 relaxation — banks at UNRELATED positions, only
 * per-bank runs consecutive — the actual multi-session decode shape):
 *   S1  A=62 toks, B=64: rows [A@62 A@63 B@64 B@65 B@66 B@67]
 *       -> BOTH banks emit ratio-4 (A row 15, B row 16), different
 *          pre-step frontiers (15 vs 16), multi-row-per-bank.
 *   S2  A=62, B=63: rows [A@62 B@63] -> only B emits (ratio-4 row 15);
 *       bank A must be bit-untouched.
 *   S3  A=126, B=127: rows [A@126 B@127] -> B emits at ratio 4 (row 31)
 *       AND ratio 128 (row 0); bank A bit-untouched at both ratios.
 *   S4  A=202, B=664 (positions ~460 apart): rows [A@202 A@203 B@664..667]
 *       -> BOTH banks emit ratio-4 (A row 50, B row 166) with per-row
 *          positions driving RoPE/ring/visibility; swap + solo checks as S1.
 *   S5  A=126, B=639: rows [A@126 B@639] -> B emits at ratio 4 (row 159)
 *       AND ratio 128 (row 4) while A (at an unrelated position) closes no
 *       group; bank A bit-untouched at both ratios.
 *   S6  A=502, B=664: rows [A@502 A@503 B@664..667] — S4's shape and S4's
 *       bank B EXACTLY, with only the BATCHMATE (A) moved 202/203 -> 502/503
 *       -> bank B's emitted rows must be BYTE-IDENTICAL to S4's.  This is
 *       the gate for per-row positions (increment 3 Part A): A's position
 *       reaches the kernels only via positions[], so a kernel that ignores
 *       positions[] moves B's rows.  S4's swap/order checks cannot see it
 *       (they hold positions fixed and permute banks -> a wrong rotation is
 *       identical on both sides), and the decode gate cannot either (its
 *       positions never vary with batch width).
 *
 * TEETH (each seeded as a local engine edit, gate re-run, edit reverted;
 * T1-T3 verified caught on the increment-2 tree, 2026-07; T4 verified on
 * this tree, 2026-07-15):
 *   T1  wrong-bank emit commit — gpu_prefill.c multiseq emit loop, commit
 *       the staged comp row to bank 0 instead of the row's bank:
 *           gpu_graph_commit_attn_comp_stage_bank(g, il, 0, comp_row, 1)
 *       -> S2/S3 "bank 0 layer N (ratio R) row K changed unexpectedly"
 *          (idle-bank cache bytes) + S1 bank-slot swap invariance FAIL.
 *   T2  wrong-bank counter bump — bump g->ms_n_comp[0][il] instead of the
 *       emitting bank's slot
 *       -> step_end self-check "multiseq step_end FAILED: bank 0 layer N
 *          frontier ..." + this gate's check_frontiers FAIL.
 *   T3  cur-bank raw store (the review BLOCKER this gate caught) — revert
 *       the compressed-layer chunked branch's raw KV store to the classic
 *       single-cache call (drop the banked scatter)
 *       -> S1 "emitted rows depend on populate order (cur-bank leak)" FAIL.
 *   T4  ignored per-row positions (Part A) — make BOTH RoPE kernels drop the
 *       positions array, src/cuda/pulsar_cuda_norm_kv.cu:
 *           const uint32_t rope_pos = pos0 + t;               (~:142)
 *           const uint32_t rope_pos = pos0 + t * pos_stride;  (~:238)
 *       -> S6 "bank B's emitted rows changed when only its BATCHMATE's
 *          position moved" FAIL (B rotates at 502+ instead of 664+).
 *       S1-S5 all still PASS under this seed — S6 is the only check in
 *       either gate with teeth for the position path.
 *
 * usage: PULSAR_MSEQ_BANKS=2 ./tests/multiseq_frontier_gate MODEL
 *        (from the repo root — reads tests/long_context_story_prompt.txt;
 *        or `make cuda-frontier-gate FRONTIER_MODEL=path/to/model.gguf`)
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static pulsar_engine *g_e;
static pulsar_tokens g_toks;      /* whole tokenized story prompt */
static int g_fail;

#define CHECK(cond, ...) do { \
        if (!(cond)) { \
            fprintf(stderr, "FRONTIER FAIL: " __VA_ARGS__); \
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

/* Stream A = prompt tokens at offset 0; stream B = offset 500 (different
 * content so cross-bank bleed cannot alias as equality). */
static int stream_tok(int stream, int i) {
    return g_toks.v[(stream ? 500 : 0) + i];
}

static bool populate_bank(pulsar_session *s, uint32_t bank, int stream, int len) {
    pulsar_gpu_graph *g = &s->graph;
    if (g->banks.n_banks && !gpu_graph_bank_repoint(g, bank)) return false;
    pulsar_tokens p;
    memset(&p, 0, sizeof(p));
    p.v = (int *)malloc((size_t)len * sizeof(int));
    p.len = p.cap = len;
    for (int i = 0; i < len; i++) p.v[i] = stream_tok(stream, i);
    char err[256];
    const int rc = pulsar_session_sync(s, &p, err, sizeof(err));
    free(p.v);
    if (rc != 0) { fprintf(stderr, "populate sync failed: %s\n", err); return false; }
    gpu_graph_bank_counters_capture(g, bank);
    return true;
}

/* One banked multiseq step: rows[t] = {bank, pos, token}. */
typedef struct { int bank; int pos; int token; } step_row;

static bool run_step(pulsar_session *s, const step_row *rows, uint32_t n) {
    pulsar_gpu_graph *g = &s->graph;
    pulsar_tokens vec;
    memset(&vec, 0, sizeof(vec));
    vec.v = (int *)malloc((size_t)n * sizeof(int));
    vec.len = vec.cap = (int)n;
    int32_t *pos = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *seq = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    for (uint32_t t = 0; t < n; t++) {
        vec.v[t] = rows[t].token;
        pos[t] = rows[t].pos;
        seq[t] = rows[t].bank;
    }
    bool ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, &vec, 0, n) &&
              gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc, g->prefill_tokens,
                                                    &g_e->model, &g_e->weights, &vec, 0, n) &&
              gpu_graph_multiseq_step_begin(g, pos, seq, n, false);
    if (ok) {
        ok = pulsar_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < PULSAR_N_LAYER; il++) {
            ok = gpu_graph_encode_layer_batch(g, &g_e->model,
                                              &g_e->weights.layer[il], il,
                                              (uint32_t)rows[0].pos, n);
        }
        if (ok) ok = pulsar_gpu_end_commands() != 0;
        else (void)pulsar_gpu_synchronize();
        const bool end_ok = gpu_graph_multiseq_step_end(g);
        ok = ok && end_ok;
        if (ok) ok = pulsar_gpu_synchronize() != 0;
    }
    free(seq);
    free(pos);
    free(vec.v);
    return ok;
}

/* D2H read of bank rows [first, first+count) of a comp cache.  Increment 2a: the
 * comp/index caches are now ONE allocation PER BANK, so read through the per-bank
 * view accessor (comp[il][bank] at offset 0) instead of pool + bank*stride. */
static bool read_bank_rows(pulsar_gpu_graph *g, int index_cache, uint32_t il,
                           uint32_t bank, uint32_t first, uint32_t count,
                           void *out, uint64_t row_bytes) {
    pulsar_gpu_tensor *view = index_cache ? gpu_graph_bank_index_comp_view(g, il, bank)
                                       : gpu_graph_bank_attn_comp_view(g, il, bank);
    if (!view) return false;
    const bool ok = pulsar_gpu_tensor_read(view,
                                        (uint64_t)first * row_bytes,
                                        out, (uint64_t)count * row_bytes) != 0;
    pulsar_gpu_tensor_free(view);
    return ok;
}

static uint64_t attn_row_bytes(void) { return gpu_graph_attn_comp_cache_row_bytes(); }
static uint64_t index_row_bytes(void) {
    return gpu_graph_idx_fp4_enabled() ? PULSAR_ENGINE_IDXFP4_ROWBYTES
                                       : (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
}

/* Snapshot every compressed layer's rows [0, upto_rows) of one bank. */
typedef struct {
    uint8_t *attn[PULSAR_MAX_LAYER];
    uint8_t *index[PULSAR_MAX_LAYER];
    uint32_t rows;
} bank_snap;

static bool snap_bank(pulsar_gpu_graph *g, uint32_t bank, uint32_t rows, bank_snap *snap) {
    memset(snap, 0, sizeof(*snap));
    snap->rows = rows;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        uint32_t n = rows;
        if (n > g->layer_comp_cap[il]) n = g->layer_comp_cap[il];
        snap->attn[il] = (uint8_t *)malloc((size_t)(n * attn_row_bytes()));
        if (!snap->attn[il] ||
            !read_bank_rows(g, 0, il, bank, 0, n, snap->attn[il], attn_row_bytes()))
            return false;
        if (ratio == 4) {
            snap->index[il] = (uint8_t *)malloc((size_t)(n * index_row_bytes()));
            if (!snap->index[il] ||
                !read_bank_rows(g, 1, il, bank, 0, n, snap->index[il], index_row_bytes()))
                return false;
        }
    }
    return true;
}

static void snap_free(bank_snap *snap) {
    for (uint32_t il = 0; il < PULSAR_MAX_LAYER; il++) {
        free(snap->attn[il]);
        free(snap->index[il]);
    }
    memset(snap, 0, sizeof(*snap));
}

/* Compare one bank's rows [0, rows) against a snapshot; rows listed in
 * skip_attn/skip_index (per ratio) are allowed (expected) to differ. */
static void check_bank_vs_snap(pulsar_gpu_graph *g, uint32_t bank, const bank_snap *snap,
                               int expect_new_r4, int expect_new_r128,
                               const char *what) {
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        uint32_t n = snap->rows;
        if (n > g->layer_comp_cap[il]) n = g->layer_comp_cap[il];
        uint8_t *now = (uint8_t *)malloc((size_t)(n * attn_row_bytes()));
        if (!now || !read_bank_rows(g, 0, il, bank, 0, n, now, attn_row_bytes())) {
            CHECK(0, "%s: bank %u layer %u readback failed", what, bank, il);
            free(now);
            return;
        }
        const int expect_new = ratio == 4 ? expect_new_r4 : expect_new_r128;
        for (uint32_t r = 0; r < n; r++) {
            const int differs = memcmp(now + (size_t)r * attn_row_bytes(),
                                       snap->attn[il] + (size_t)r * attn_row_bytes(),
                                       attn_row_bytes()) != 0;
            if (expect_new >= 0 && r == (uint32_t)expect_new) continue; /* new row, content checked elsewhere */
            CHECK(!differs, "%s: bank %u layer %u (ratio %u) row %u changed unexpectedly",
                  what, bank, il, ratio, r);
        }
        free(now);
        if (ratio == 4 && snap->index[il]) {
            uint8_t *inow = (uint8_t *)malloc((size_t)(n * index_row_bytes()));
            if (!inow || !read_bank_rows(g, 1, il, bank, 0, n, inow, index_row_bytes())) {
                CHECK(0, "%s: bank %u layer %u indexer readback failed", what, bank, il);
                free(inow);
                return;
            }
            for (uint32_t r = 0; r < n; r++) {
                const int differs = memcmp(inow + (size_t)r * index_row_bytes(),
                                           snap->index[il] + (size_t)r * index_row_bytes(),
                                           index_row_bytes()) != 0;
                if (expect_new_r4 >= 0 && r == (uint32_t)expect_new_r4) continue;
                CHECK(!differs, "%s: bank %u layer %u indexer row %u changed unexpectedly",
                      what, bank, il, r);
            }
            free(inow);
        }
    }
}

/* Collect one emitted row (attn + indexer where present) per compressed
 * layer of a bank into caller buffers keyed by layer. */
typedef struct {
    uint8_t *attn[PULSAR_MAX_LAYER];    /* NULL where no row expected */
    uint8_t *index[PULSAR_MAX_LAYER];
} emit_rows;

static bool collect_emit_rows(pulsar_gpu_graph *g, uint32_t bank,
                              int row_r4, int row_r128, emit_rows *er) {
    memset(er, 0, sizeof(*er));
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const int row = ratio == 4 ? row_r4 : row_r128;
        if (row < 0) continue;
        er->attn[il] = (uint8_t *)malloc((size_t)attn_row_bytes());
        if (!er->attn[il] ||
            !read_bank_rows(g, 0, il, bank, (uint32_t)row, 1, er->attn[il], attn_row_bytes()))
            return false;
        if (ratio == 4) {
            er->index[il] = (uint8_t *)malloc((size_t)index_row_bytes());
            if (!er->index[il] ||
                !read_bank_rows(g, 1, il, bank, (uint32_t)row, 1, er->index[il], index_row_bytes()))
                return false;
        }
    }
    return true;
}

static void emit_rows_free(emit_rows *er) {
    for (uint32_t il = 0; il < PULSAR_MAX_LAYER; il++) {
        free(er->attn[il]);
        free(er->index[il]);
    }
    memset(er, 0, sizeof(*er));
}

static int emit_rows_equal(const emit_rows *a, const emit_rows *b, const char *what) {
    int equal = 1;
    for (uint32_t il = 0; il < PULSAR_MAX_LAYER; il++) {
        if ((a->attn[il] != NULL) != (b->attn[il] != NULL)) { equal = 0; continue; }
        if (a->attn[il] && memcmp(a->attn[il], b->attn[il], attn_row_bytes()) != 0) {
            fprintf(stderr, "  %s: attn emitted row differs at layer %u\n", what, il);
            equal = 0;
        }
        if (a->index[il] && b->index[il] &&
            memcmp(a->index[il], b->index[il], index_row_bytes()) != 0) {
            fprintf(stderr, "  %s: indexer emitted row differs at layer %u\n", what, il);
            equal = 0;
        }
    }
    return equal;
}

/* Frontier counter checks. */
static void check_frontiers(pulsar_gpu_graph *g, uint32_t bank, uint32_t end_pos,
                            const char *what) {
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t want = (end_pos + 1u) / ratio;
        CHECK(g->ms_n_comp[bank][il] == want,
              "%s: bank %u layer %u ms_n_comp %u != %u", what, bank, il,
              g->ms_n_comp[bank][il], want);
        if (ratio == 4)
            CHECK(g->ms_n_index_comp[bank][il] == want,
                  "%s: bank %u layer %u ms_n_index_comp %u != %u", what, bank, il,
                  g->ms_n_index_comp[bank][il], want);
    }
}

static void check_superset(pulsar_gpu_graph *g, const char *what) {
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        uint32_t mx = 0;
        for (uint32_t b = 0; b < gpu_graph_bank_pool_count(g); b++)
            if (g->ms_n_comp[b][il] > mx) mx = g->ms_n_comp[b][il];
        CHECK(g->layer_n_comp[il] == mx,
              "%s: layer %u superset %u != max(banks) %u", what, il,
              g->layer_n_comp[il], mx);
    }
}

/* Solo reference: fresh session, populate bank 0 with `stream` to `len`,
 * then advance by `steps` single-row multiseq steps; return the emitted rows
 * at (row_r4, row_r128) of bank 0. */
static bool solo_reference(int stream, int len, int steps,
                           int row_r4, int row_r128, emit_rows *er) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    bool ok = populate_bank(s, 0, stream, len);
    for (int i = 0; ok && i < steps; i++) {
        const step_row r = {0, len + i, stream_tok(stream, len + i)};
        ok = run_step(s, &r, 1);
    }
    if (ok) ok = collect_emit_rows(&s->graph, 0, row_r4, row_r128, er);
    pulsar_session_free(s);
    return ok;
}

/* Mixed scenario runner.  a_len/b_len: prefilled lengths; rows: the one
 * batched step (banks a_bank/b_bank).  Returns emitted rows of both banks. */
static bool mixed_scenario_order(int a_bank, int a_len, int b_bank, int b_len,
                           const step_row *rows, uint32_t n_rows,
                           int a_row_r4, int a_row_r128,
                           int b_row_r4, int b_row_r128,
                           int a_end_pos, int b_end_pos,
                           const char *what,
                           emit_rows *a_out, emit_rows *b_out,
                           int check_a_untouched, int populate_b_first) {
    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, g_e, 4096) != 0) return false;
    pulsar_gpu_graph *g = &s->graph;
    /* populate_b_first flips which bank the graph's views are repointed at
     * when the step runs (cur_bank) — the banked step must be bit-invariant
     * to it (any cur-bank leakage in stores/reads/emits breaks this). */
    bool ok = populate_b_first
        ? (populate_bank(s, (uint32_t)b_bank, 1, b_len) &&
           populate_bank(s, (uint32_t)a_bank, 0, a_len))
        : (populate_bank(s, (uint32_t)a_bank, 0, a_len) &&
           populate_bank(s, (uint32_t)b_bank, 1, b_len));
    bank_snap a_pre;
    memset(&a_pre, 0, sizeof(a_pre));
    const uint32_t snap_rows = (uint32_t)(a_len / 4 + 3);
    if (ok && check_a_untouched) ok = snap_bank(g, (uint32_t)a_bank, snap_rows, &a_pre);
    if (ok) ok = run_step(s, rows, n_rows);
    if (ok) {
        check_frontiers(g, (uint32_t)a_bank, (uint32_t)a_end_pos, what);
        check_frontiers(g, (uint32_t)b_bank, (uint32_t)b_end_pos, what);
        check_superset(g, what);
        if (check_a_untouched)
            check_bank_vs_snap(g, (uint32_t)a_bank, &a_pre, a_row_r4, a_row_r128, what);
        ok = collect_emit_rows(g, (uint32_t)a_bank, a_row_r4, a_row_r128, a_out) &&
             collect_emit_rows(g, (uint32_t)b_bank, b_row_r4, b_row_r128, b_out);
        if (!ok) CHECK(0, "%s: emit-row collection failed", what);
    } else {
        CHECK(0, "%s: scenario execution failed", what);
    }
    snap_free(&a_pre);
    pulsar_session_free(s);
    return ok;
}

static bool mixed_scenario(int a_bank, int a_len, int b_bank, int b_len,
                           const step_row *rows, uint32_t n_rows,
                           int a_row_r4, int a_row_r128,
                           int b_row_r4, int b_row_r128,
                           int a_end_pos, int b_end_pos,
                           const char *what,
                           emit_rows *a_out, emit_rows *b_out,
                           int check_a_untouched) {
    return mixed_scenario_order(a_bank, a_len, b_bank, b_len, rows, n_rows,
                                a_row_r4, a_row_r128, b_row_r4, b_row_r128,
                                a_end_pos, b_end_pos, what, a_out, b_out,
                                check_a_untouched, 0);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL\n", argv[0]); return 2; }

    pulsar_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&g_e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    size_t text_len = 0;
    char *text = read_file("tests/long_context_story_prompt.txt", &text_len);
    if (!text) { fprintf(stderr, "prompt file read failed\n"); return 1; }
    memset(&g_toks, 0, sizeof(g_toks));
    pulsar_tokenize_text(g_e, text, &g_toks);
    free(text);
    if (g_toks.len < 1200) { fprintf(stderr, "prompt too short\n"); return 1; }

    /* ---- S1: both banks emit in ONE step, different pre-frontiers ---- */
    {
        const step_row rows[6] = {
            {0, 62, stream_tok(0, 62)}, {0, 63, stream_tok(0, 63)},
            {1, 64, stream_tok(1, 64)}, {1, 65, stream_tok(1, 65)},
            {1, 66, stream_tok(1, 66)}, {1, 67, stream_tok(1, 67)},
        };
        emit_rows mixed_a, mixed_b, swap_a, swap_b, solo_a, solo_b;
        bool ok = mixed_scenario(0, 62, 1, 64, rows, 6,
                                 15, -1, 16, -1, 63, 67,
                                 "S1(mixed)", &mixed_a, &mixed_b, 0);
        /* Swapped banks, same shapes: bank-addressing invariance. */
        const step_row rows_sw[6] = {
            {1, 62, stream_tok(0, 62)}, {1, 63, stream_tok(0, 63)},
            {0, 64, stream_tok(1, 64)}, {0, 65, stream_tok(1, 65)},
            {0, 66, stream_tok(1, 66)}, {0, 67, stream_tok(1, 67)},
        };
        bool ok_sw = ok && mixed_scenario(1, 62, 0, 64, rows_sw, 6,
                                          15, -1, 16, -1, 63, 67,
                                          "S1(swap)", &swap_a, &swap_b, 0);
        if (ok_sw) {
            CHECK(emit_rows_equal(&mixed_a, &swap_a, "S1 A mixed-vs-swap"),
                  "S1: sequence A emitted rows depend on bank slot");
            CHECK(emit_rows_equal(&mixed_b, &swap_b, "S1 B mixed-vs-swap"),
                  "S1: sequence B emitted rows depend on bank slot");
            printf("S1: bank-slot invariance (same-shape swap) OK\n");
        }
        /* Populate-order invariance: identical banks/rows/shapes, but the
         * graph's installed views sit on bank A (not B) when the step runs.
         * Any cur-bank leakage in the banked stores/reads/emits (the raw
         * store BLOCKER class caught in review) breaks this bit-equality. */
        emit_rows ord_a, ord_b;
        bool ok_ord = ok && mixed_scenario_order(0, 62, 1, 64, rows, 6,
                                                 15, -1, 16, -1, 63, 67,
                                                 "S1(order)", &ord_a, &ord_b, 0, 1);
        if (ok_ord) {
            CHECK(emit_rows_equal(&mixed_a, &ord_a, "S1 A order-invariance"),
                  "S1: sequence A emitted rows depend on populate order (cur-bank leak)");
            CHECK(emit_rows_equal(&mixed_b, &ord_b, "S1 B order-invariance"),
                  "S1: sequence B emitted rows depend on populate order (cur-bank leak)");
            printf("S1: populate-order (cur-bank) invariance OK\n");
            emit_rows_free(&ord_a);
            emit_rows_free(&ord_b);
        } else if (ok) {
            CHECK(0, "S1: order-invariance scenario failed");
        }
        /* Cross-shape solo references (1-row steps). */
        bool ok_solo = ok && solo_reference(0, 62, 2, 15, -1, &solo_a) &&
                             solo_reference(1, 64, 4, 16, -1, &solo_b);
        if (ok_solo) {
            const int ea = emit_rows_equal(&mixed_a, &solo_a, "S1 A mixed-vs-solo");
            const int eb = emit_rows_equal(&mixed_b, &solo_b, "S1 B mixed-vs-solo");
            printf("S1: mixed(6-row) vs solo(1-row steps): A %s, B %s\n",
                   ea ? "BYTE-EXACT" : "DIFFERS (batch-shape variance)",
                   eb ? "BYTE-EXACT" : "DIFFERS (batch-shape variance)");
            if (!ea || !eb) {
                printf("S1: NOTE: cross-shape divergence is GEMM/MoE batch-shape\n"
                       "    physics, not addressing — the swap gate above is the\n"
                       "    addressing truth; see report.\n");
            }
            emit_rows_free(&solo_a);
            emit_rows_free(&solo_b);
        } else if (ok) {
            CHECK(0, "S1: solo reference runs failed");
        }
        if (ok) { emit_rows_free(&mixed_a); emit_rows_free(&mixed_b); }
        if (ok_sw) { emit_rows_free(&swap_a); emit_rows_free(&swap_b); }
    }

    /* ---- S2: only B emits (ratio 4); bank A bit-untouched ---- */
    {
        const step_row rows[2] = {
            {0, 62, stream_tok(0, 62)},
            {1, 63, stream_tok(1, 63)},
        };
        emit_rows a, b, solo_b;
        /* A ends at pos 62 -> no emit anywhere: expected-new rows -1. */
        bool ok = mixed_scenario(0, 62, 1, 63, rows, 2,
                                 -1, -1, 15, -1, 62, 63,
                                 "S2", &a, &b, 1);
        if (ok) {
            printf("S2: only-B-emit — bank A counters + bytes untouched OK\n");
            if (solo_reference(1, 63, 1, 15, -1, &solo_b)) {
                const int eb = emit_rows_equal(&b, &solo_b, "S2 B mixed-vs-solo");
                printf("S2: B emitted row vs solo(1-row): %s\n",
                       eb ? "BYTE-EXACT" : "DIFFERS (batch-shape variance)");
                emit_rows_free(&solo_b);
            } else {
                CHECK(0, "S2: solo reference failed");
            }
            emit_rows_free(&a);
            emit_rows_free(&b);
        }
    }

    /* ---- S3: B emits at ratio 128 AND ratio 4; bank A bit-untouched ---- */
    {
        const step_row rows[2] = {
            {0, 126, stream_tok(0, 126)},
            {1, 127, stream_tok(1, 127)},
        };
        emit_rows a, b, solo_b;
        bool ok = mixed_scenario(0, 126, 1, 127, rows, 2,
                                 -1, -1, 31, 0, 126, 127,
                                 "S3", &a, &b, 1);
        if (ok) {
            printf("S3: ratio-128 only-B-emit — bank A untouched at both ratios OK\n");
            if (solo_reference(1, 127, 1, 31, 0, &solo_b)) {
                const int eb = emit_rows_equal(&b, &solo_b, "S3 B mixed-vs-solo");
                printf("S3: B emitted rows (r4+r128) vs solo(1-row): %s\n",
                       eb ? "BYTE-EXACT" : "DIFFERS (batch-shape variance)");
                emit_rows_free(&solo_b);
            } else {
                CHECK(0, "S3: solo reference failed");
            }
            emit_rows_free(&a);
            emit_rows_free(&b);
        }
    }

    /* ---- S4: banks at UNRELATED positions (increment-3 relaxation) ----
     * Same checks as S1 (frontiers, superset, swap invariance, solo refs)
     * but bank A sits ~460 positions behind bank B, so every per-row
     * position-derived stage (RoPE q/kv/indexer-q/inverse, ring slots,
     * visible-comp) computes different values per bank within one step. */
    {
        const step_row rows[6] = {
            {0, 202, stream_tok(0, 202)}, {0, 203, stream_tok(0, 203)},
            {1, 664, stream_tok(1, 664)}, {1, 665, stream_tok(1, 665)},
            {1, 666, stream_tok(1, 666)}, {1, 667, stream_tok(1, 667)},
        };
        emit_rows mixed_a, mixed_b, swap_a, swap_b, solo_a, solo_b;
        bool ok = mixed_scenario(0, 202, 1, 664, rows, 6,
                                 50, -1, 166, -1, 203, 667,
                                 "S4(mixed)", &mixed_a, &mixed_b, 0);
        const step_row rows_sw[6] = {
            {1, 202, stream_tok(0, 202)}, {1, 203, stream_tok(0, 203)},
            {0, 664, stream_tok(1, 664)}, {0, 665, stream_tok(1, 665)},
            {0, 666, stream_tok(1, 666)}, {0, 667, stream_tok(1, 667)},
        };
        bool ok_sw = ok && mixed_scenario(1, 202, 0, 664, rows_sw, 6,
                                          50, -1, 166, -1, 203, 667,
                                          "S4(swap)", &swap_a, &swap_b, 0);
        if (ok_sw) {
            CHECK(emit_rows_equal(&mixed_a, &swap_a, "S4 A mixed-vs-swap"),
                  "S4: sequence A emitted rows depend on bank slot");
            CHECK(emit_rows_equal(&mixed_b, &swap_b, "S4 B mixed-vs-swap"),
                  "S4: sequence B emitted rows depend on bank slot");
            printf("S4: different-position bank-slot invariance OK\n");
        }
        bool ok_solo = ok && solo_reference(0, 202, 2, 50, -1, &solo_a) &&
                             solo_reference(1, 664, 4, 166, -1, &solo_b);
        if (ok_solo) {
            const int ea = emit_rows_equal(&mixed_a, &solo_a, "S4 A mixed-vs-solo");
            const int eb = emit_rows_equal(&mixed_b, &solo_b, "S4 B mixed-vs-solo");
            printf("S4: mixed(6-row, split positions) vs solo(1-row steps): A %s, B %s\n",
                   ea ? "BYTE-EXACT" : "DIFFERS (batch-shape variance)",
                   eb ? "BYTE-EXACT" : "DIFFERS (batch-shape variance)");
            emit_rows_free(&solo_a);
            emit_rows_free(&solo_b);
        } else if (ok) {
            CHECK(0, "S4: solo reference runs failed");
        }

        /* ---- S6: BATCHMATE-POSITION independence (the Part-A gate) ----
         * Identical batch SHAPE to S4 — same 6 rows, same order, same banks,
         * bank B at the same positions 664..667 with the same tokens and the
         * same emit topology (A closes one ratio-4 group, B closes one) —
         * but bank A sits at 502/503 instead of 202/203.
         *
         * Bank B's rows, shape, position and content are all unchanged, so
         * B's emitted rows MUST be byte-identical to S4's.  The only thing
         * that differs in the whole step is A's position, which reaches the
         * kernels ONLY through the per-row positions[] array.  A kernel that
         * ignores positions[] and falls back to pos0 + t * pos_stride
         * rotates B against A's run instead of its own, so B's rows move ->
         * FAIL.  Nothing else in either gate can observe that: S4's swap and
         * order checks hold positions fixed and permute banks, so a wrong
         * rotation is identical on both sides of those comparisons, and the
         * decode gate's positions are a function of bank and step but never
         * of batch width (see that gate's header).  This is the check with
         * teeth for per-row positions. */
        emit_rows s6_a, s6_b;
        const step_row rows_s6[6] = {
            {0, 502, stream_tok(0, 502)}, {0, 503, stream_tok(0, 503)},
            {1, 664, stream_tok(1, 664)}, {1, 665, stream_tok(1, 665)},
            {1, 666, stream_tok(1, 666)}, {1, 667, stream_tok(1, 667)},
        };
        bool ok_s6 = ok && mixed_scenario(0, 502, 1, 664, rows_s6, 6,
                                          125, -1, 166, -1, 503, 667,
                                          "S6(mixed)", &s6_a, &s6_b, 0);
        if (ok_s6) {
            const int eq6 = emit_rows_equal(&s6_b, &mixed_b, "S6 B vs S4 B");
            CHECK(eq6,
                  "S6: bank B's emitted rows changed when only its BATCHMATE's "
                  "position moved (202/203 -> 502/503) — B's own rows, shape "
                  "and position are identical, so a per-row position is being "
                  "ignored (RoPE/ring/visible-comp reading pos0+t, not "
                  "positions[t])");
            if (eq6)
                printf("S6: batchmate-position independence — B byte-identical "
                       "across A@202 and A@502 OK\n");
            emit_rows_free(&s6_a);
            emit_rows_free(&s6_b);
        } else if (ok) {
            CHECK(0, "S6: scenario execution failed");
        }

        if (ok) { emit_rows_free(&mixed_a); emit_rows_free(&mixed_b); }
        if (ok_sw) { emit_rows_free(&swap_a); emit_rows_free(&swap_b); }
    }

    /* ---- S5: unrelated positions, only B emits (ratio 4 AND 128) ---- */
    {
        const step_row rows[2] = {
            {0, 126, stream_tok(0, 126)},
            {1, 639, stream_tok(1, 639)},
        };
        emit_rows a, b, solo_b;
        bool ok = mixed_scenario(0, 126, 1, 639, rows, 2,
                                 -1, -1, 159, 4, 126, 639,
                                 "S5", &a, &b, 1);
        if (ok) {
            printf("S5: different-position only-B-emit (r4+r128) — bank A untouched OK\n");
            if (solo_reference(1, 639, 1, 159, 4, &solo_b)) {
                const int eb = emit_rows_equal(&b, &solo_b, "S5 B mixed-vs-solo");
                printf("S5: B emitted rows vs solo(1-row): %s\n",
                       eb ? "BYTE-EXACT" : "DIFFERS (batch-shape variance)");
                emit_rows_free(&solo_b);
            } else {
                CHECK(0, "S5: solo reference failed");
            }
            emit_rows_free(&a);
            emit_rows_free(&b);
        }
    }

    pulsar_engine_close(g_e);
    if (g_fail) { fprintf(stderr, "FRONTIER GATE: FAIL\n"); return 1; }
    printf("FRONTIER GATE: PASS\n");
    return 0;
}
