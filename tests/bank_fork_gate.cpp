/* Tier-2 PATH-A FORK gate (plan-33 increment A) — the fork==cold oracle.
 *
 * Proves the full-prefix in-memory fork produces a bank BYTE-IDENTICAL to a cold
 * full prefill: fork src->dst, prefill an IDENTICAL suffix into (a) the forked dst
 * and (b) a fresh COLD bank, then assert their committed comp/index KV frontiers
 * are byte-identical (FNV over the D2H rows) AND their next-token argmax matches.
 * Also checks the STRUCTURAL anti-contamination: a fork whose request prefix does
 * NOT match the source's committed history is REFUSED (no device write).
 *
 * MODEL-DEPENDENT, GPU-resident, needs PULSAR_MSEQ_BANKS>=3 (src + fork-dst + cold-dst).
 * Run manually under the memory discipline (hold temp/gpu.lock, drop_caches, no
 * foreign ds4 process). NOT part of `make test`.
 *
 * usage: PULSAR_MSEQ_BANKS=3 ./tests/bank_fork_gate MODEL [N_CACHED L]
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
/* seed for greedy pulsar_session_sample calls (was &(uint64_t){7} compound literals;
 * greedy never consumes RNG state, so one shared object is behavior-identical). */
static uint64_t g_seed7 = 7;


static int g_fail;
#define CHECK(c, ...) do { if(!(c)){ fprintf(stderr,"FORK FAIL: " __VA_ARGS__); fprintf(stderr,"\n"); g_fail=1; } } while(0)

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb"); if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(b); return NULL; }
    fclose(fp); b[n] = '\0'; if (len_out) *len_out = (size_t)n; return b;
}

/* FNV-1a fold of bank `bank`'s captured comp+index frontier rows (raw D2H). */
static uint64_t checksum_bank_kv(pulsar_session *s, uint32_t bank) {
    pulsar_gpu_graph *g = &s->graph;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? PULSAR_ENGINE_IDXFP4_ROWBYTES : (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    uint64_t h = 1469598103934665603ull;
    uint8_t *buf = (uint8_t *)malloc(64u * 1024u * 1024u);
    if (!buf) return 0;
    for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
        const uint32_t ratio = pulsar_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t ncomp = g->ms_n_comp[bank][il];
        if (ncomp) {
            pulsar_gpu_tensor *v = gpu_graph_bank_attn_comp_view(g, il, bank);
            if (!v || pulsar_gpu_tensor_read(v, 0, buf, (uint64_t)ncomp * attn_row) == 0) { pulsar_gpu_tensor_free(v); free(buf); return 0; }
            pulsar_gpu_tensor_free(v);
            for (uint64_t i = 0; i < (uint64_t)ncomp * attn_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
        }
        if (ratio == 4) {
            const uint32_t nidx = g->ms_n_index_comp[bank][il];
            if (nidx) {
                pulsar_gpu_tensor *v = gpu_graph_bank_index_comp_view(g, il, bank);
                if (!v || pulsar_gpu_tensor_read(v, 0, buf, (uint64_t)nidx * idx_row) == 0) { pulsar_gpu_tensor_free(v); free(buf); return 0; }
                pulsar_gpu_tensor_free(v);
                for (uint64_t i = 0; i < (uint64_t)nidx * idx_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
            }
        }
    }
    free(buf);
    return h;
}

/* Prefill bank `bank` (fresh) to the first `len` tokens of toks. */
static bool prefill_bank_cold(pulsar_session *s, uint32_t bank, int *toks, int len) {
    if (!pulsar_session_bank_state_restore(s, bank)) return false;
    pulsar_session_invalidate(s);
    pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks; p.len = p.cap = len;
    char err[256];
    if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { fprintf(stderr, "sync bank %u: %s\n", bank, err); return false; }
    gpu_graph_bank_counters_capture(&s->graph, bank);
    return true;
}

/* P5 output-token oracle: greedy-decode `ngen` tokens from the bank CURRENTLY
 * installed at position `len`, appending each into toks[len..] and recording it
 * in out[0..ngen). Mirrors the server's decode loop (sample -> feed the token
 * back -> sync -> sample again). `toks` must have capacity >= len+ngen. On a sync
 * failure the stream is truncated with a -1 sentinel. */
static void decode_stream_greedy(pulsar_session *s, uint32_t bank, int *toks,
                                 int len, int ngen, int *out) {
    pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks;
    for (int i = 0; i < ngen; i++) {
        const int t = pulsar_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &g_seed7);
        out[i] = t;
        toks[len + i] = t;
        p.len = p.cap = len + i + 1;
        char err[256];
        if (pulsar_session_sync(s, &p, err, sizeof err) != 0) { out[i] = -1; break; }
        gpu_graph_bank_counters_capture(&s->graph, bank);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [N_CACHED L]\n", argv[0]); return 2; }
    /* ORACLE VALIDITY: n_cached must be a multiple of the PREFILL CHUNK (4096
     * default; also of 128 = LCM of compress ratios). A resumed prefill aligns
     * its chunks to absolute prefill_cap boundaries (imatrix.c chunked_range),
     * so with an aligned cut every resumed chunk has the SAME batch shape as the
     * cold run's — bit-identical kernels — and byte-identity is a fair oracle.
     * An UNALIGNED cut runs a short realign chunk through the per-token
     * compressor fallback (a different code path than cold's batched one):
     * last-ulp GEMM differences then propagate to every later row — that is the
     * engine's existing warm-continuation numerics, NOT a fork bug, and it
     * breaks the byte oracle spuriously. */
    const int n_cached = argc > 2 ? atoi(argv[2]) : 12288;
    const int L = argc > 3 ? atoi(argv[3]) : 20480;
    if ((n_cached % 4096) != 0) {
        fprintf(stderr, "n_cached %d must be a multiple of the prefill chunk (4096) "
                        "for the byte-identity oracle (see header comment)\n", n_cached);
        return 2;
    }
    const int ctx = L + 4096;
    if (n_cached >= L) { fprintf(stderr, "need N_CACHED < L\n"); return 2; }

    pulsar_engine *e = NULL; pulsar_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = argv[1]; opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    size_t tl = 0; char *text = read_file("tests/long_context_story_prompt.txt", &tl);
    if (!text) { fprintf(stderr, "prompt read failed\n"); return 1; }
    pulsar_tokens base; memset(&base, 0, sizeof base);
    pulsar_tokenize_text(e, text, &base); free(text);
    if (base.len < 256) { fprintf(stderr, "prompt too short\n"); return 1; }

    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session create failed\n"); return 1; }
    const uint32_t pool = gpu_graph_bank_pool_count(&s->graph);
    fprintf(stderr, "fork_gate: pool banks=%u ctx=%d n_cached=%d L=%d\n", pool, ctx, n_cached, L);
    if (pool < 3) { fprintf(stderr, "need PULSAR_MSEQ_BANKS>=3\n"); return 1; }

    int *toks = (int *)malloc((size_t)L * sizeof(int));
    for (int i = 0; i < L; i++) toks[i] = base.v[i % base.len];

    /* 1. SRC = bank 0 prefilled to the shared prefix [0, n_cached). */
    CHECK(prefill_bank_cold(s, 0, toks, n_cached), "src prefill");
    pulsar_gpu_synchronize();
    const uint64_t sum_src = checksum_bank_kv(s, 0);
    CHECK(sum_src != 0, "src checksum");
    fprintf(stderr, "fork_gate: src(bank0) prefilled to %d, checksum=%016" PRIx64 "\n", n_cached, sum_src);

    /* 2. NEGATIVE: a fork whose prefix does NOT match src is REFUSED (no write). */
    {
        int *wrong = (int *)malloc((size_t)n_cached * sizeof(int));
        for (int i = 0; i < n_cached; i++) wrong[i] = toks[i];
        wrong[n_cached / 2] ^= 0x1;   /* one token differs */
        const int rc = pulsar_session_bank_fork(s, 0, 2, wrong, n_cached);
        CHECK(rc != 0, "fork ACCEPTED a mismatched prefix (anti-contamination broken)");
        CHECK(!pulsar_session_bank_fork_pinned(s, 0), "src left fork-pinned after refusal");
        fprintf(stderr, "fork_gate: mismatch-prefix fork refused rc=%d : %s\n", rc, rc != 0 ? "OK" : "FAIL");
        free(wrong);
        /* src must be untouched by the refused fork. */
        gpu_graph_bank_counters_capture(&s->graph, 0);
        pulsar_gpu_synchronize();
        CHECK(checksum_bank_kv(s, 0) == sum_src, "src KV changed by a REFUSED fork");
    }

    /* 3. FORK src(0) -> dst(1). Must validate + clone + mirror counters. */
    {
        /* src is cur (bank 0 installed); its committed history is s->checkpoint. */
        const int rc = pulsar_session_bank_fork(s, 0, 1, toks, n_cached);
        CHECK(rc == 0, "full-prefix fork refused rc=%d", rc);
        CHECK(!pulsar_session_bank_fork_pinned(s, 0), "src still pinned after fork");
        /* immediate fork bit-identity: dst frontier == src frontier. */
        pulsar_gpu_synchronize();
        const uint64_t sum_fork = checksum_bank_kv(s, 1);
        CHECK(sum_fork == sum_src, "forked dst KV != src KV immediately after clone (%016" PRIx64 " vs %016" PRIx64 ")", sum_fork, sum_src);
        fprintf(stderr, "fork_gate: fork(0->1) rc=%d immediate dst checksum=%016" PRIx64 " (==src: %s)\n",
                rc, sum_fork, sum_fork == sum_src ? "YES" : "NO");
    }

    /* 4. Prefill the IDENTICAL suffix [n_cached, L) into the FORKED dst (bank 1)
     *    and a COLD full prefill into bank 2. */
    /* fork dst: install bank 1 (carries src's checkpoint at n_cached), sync to L. */
    CHECK(pulsar_session_bank_state_restore(s, 1), "install forked bank 1");
    {
        pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks; p.len = p.cap = L;
        char err[256];
        CHECK(pulsar_session_sync(s, &p, err, sizeof err) == 0, "fork-dst suffix sync: %s", err);
        /* LIVE frontier assert (catches a silently short-circuited sync HERE,
         * while bank 1 is still installed — pulsar_session_pos reads the live
         * checkpoint, not a possibly-stale carry). */
        CHECK(pulsar_session_pos(s) == L, "fork-dst live pos %d != %d after suffix sync",
              pulsar_session_pos(s), L);
        gpu_graph_bank_counters_capture(&s->graph, 1);
    }
    const int tok_fork = pulsar_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &g_seed7);
    pulsar_gpu_synchronize();
    const uint64_t sum_fork_full = checksum_bank_kv(s, 1);
    /* SAVE bank 1's live state before switching away — the final bank_pos(1)
     * read below reads the CARRY once bank 2 is installed; without this save it
     * reads the stale fork-time carry (the original gate-A false "pos" failure). */
    pulsar_session_bank_state_save(s, 1);

    /* cold: bank 2 full prefill of [0, L). */
    CHECK(prefill_bank_cold(s, 2, toks, L), "cold full prefill bank 2");
    const int tok_cold = pulsar_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &g_seed7);
    pulsar_gpu_synchronize();
    const uint64_t sum_cold = checksum_bank_kv(s, 2);

    /* 5. THE ORACLE: fork+suffix must be byte-identical to cold full prefill. */
    CHECK(sum_fork_full == sum_cold,
          "fork+suffix KV NOT byte-identical to cold (%016" PRIx64 " vs %016" PRIx64 ")",
          sum_fork_full, sum_cold);
    CHECK(tok_fork == tok_cold, "fork+suffix next-token %d != cold %d", tok_fork, tok_cold);
    CHECK(pulsar_session_bank_pos(s, 1) == L, "fork-dst pos %d != %d", pulsar_session_bank_pos(s, 1), L);
    fprintf(stderr, "fork_gate: fork+suffix checksum=%016" PRIx64 " cold=%016" PRIx64 " (byte-identical: %s); next-token fork=%d cold=%d\n",
            sum_fork_full, sum_cold, sum_fork_full == sum_cold ? "YES" : "NO", tok_fork, tok_cold);

    /* ================= increment C: PARTIAL-cut fork oracle ================
     * Mid-ratio-4-group cut on an UNWRAPPED source, chunk-aligned R (oracle
     * validity, see header): src bank0 @4300 (toks), request toks2 shares
     * [0,4102) then diverges; n_cached=4102 -> R=4096 (%4096==0, %128==0),
     * boundary row R/4=1024. fork_partial + replay vs COLD toks2 prefill:
     * byte-identical frontier + SPECIFIC row-1024 byte-diff per ratio-4 layer
     * (comp AND the MXFP4 indexer row separately — plan-33 risks 1+2). */
    {
        /* SLEN == L2: src must compute the boundary row 1024 in the SAME chunk
         * shape as the cold control ([4096,8192) full chunk) — a short src (e.g.
         * 4300) computes it in a 204-token batch whose last-ulp GEMM deltas make
         * the stash source differ from cold before any fork logic runs. */
        const int SLEN = 8192, NC = 4102, RCUT = 4096, L2 = 8192;
        int *toks2 = (int *)malloc((size_t)L2 * sizeof(int));
        for (int i = 0; i < L2; i++)
            toks2[i] = i < NC ? toks[i] : base.v[(i + 7777) % base.len];

        /* fresh short src on bank 0 */
        CHECK(prefill_bank_cold(s, 0, toks, SLEN), "P2 src prefill bank0@%d", SLEN);
        pulsar_session_bank_state_save(s, 0);

        /* shallow-cut refusal (R would be 0) */
        CHECK(pulsar_session_bank_fork_partial(s, 0, 1, toks2, 100) != 0,
              "P2 shallow cut not refused");

        /* partial fork 0->1 and replay toks2 to L2 */
        const int prc = pulsar_session_bank_fork_partial(s, 0, 1, toks2, NC);
        CHECK(prc == 0, "P2 partial fork refused rc=%d", prc);
        /* TRIAGE memcmp: stash slot vs SRC row 1024 (capture correctness). */
        {
            pulsar_gpu_graph *g = &s->graph;
            const uint64_t ar = gpu_graph_attn_comp_cache_row_bytes();
            uint8_t sa[4096], sb[4096];
            int bad = 0;
            for (uint32_t il = 0; il < PULSAR_N_LAYER && bad < 3; il++) {
                if (pulsar_layer_compress_ratio(il) != 4u) continue;
                pulsar_gpu_tensor *v = gpu_graph_bank_attn_comp_view(g, il, 0);
                pulsar_gpu_synchronize();
                if (v && pulsar_gpu_tensor_read(g->emit_stash_comp, ((uint64_t)1 * PULSAR_N_LAYER + il) * ar, sa, ar) &&
                    pulsar_gpu_tensor_read(v, (uint64_t)(RCUT/4) * ar, sb, ar) &&
                    memcmp(sa, sb, (size_t)ar) != 0) bad++;
                pulsar_gpu_tensor_free(v);
            }
            CHECK(bad == 0, "TRIAGE: stash != src row %d on %d layer(s) (CAPTURE bug)", RCUT/4, bad);
            fprintf(stderr, "fork_gate: TRIAGE stash==src row %d : %s\n", RCUT/4, bad ? "NO" : "YES");
        }
        CHECK(!pulsar_session_bank_fork_pinned(s, 0), "P2 src left pinned");
        CHECK(pulsar_session_bank_state_restore(s, 1), "P2 install fork dst");
        CHECK(pulsar_session_pos(s) == RCUT, "P2 dst pos %d != R %d", pulsar_session_pos(s), RCUT);
        {
            pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks2; p.len = p.cap = L2;
            char e2[256];
            CHECK(pulsar_session_sync(s, &p, e2, sizeof e2) == 0, "P2 replay sync: %s", e2);
            CHECK(pulsar_session_pos(s) == L2, "P2 live pos %d != %d", pulsar_session_pos(s), L2);
            gpu_graph_bank_counters_capture(&s->graph, 1);
        }
        const int tok_pf = pulsar_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &g_seed7);
        pulsar_gpu_synchronize();
        const uint64_t sum_pf = checksum_bank_kv(s, 1);
        pulsar_session_bank_state_save(s, 1);

        /* cold control on bank 2 */
        CHECK(prefill_bank_cold(s, 2, toks2, L2), "P2 cold control bank2");
        pulsar_session_bank_state_save(s, 2);
        const int tok_pc = pulsar_session_sample(s, 0.0f, 0, 1.0f, 0.0f, &g_seed7);
        pulsar_gpu_synchronize();
        const uint64_t sum_pc = checksum_bank_kv(s, 2);

        /* TRIAGE memcmp: SRC row 1024 vs COLD row 1024 (source-shape validity —
         * if these differ the oracle's stash source diverges from cold before
         * any fork logic; see SLEN note above). */
        {
            pulsar_gpu_graph *g = &s->graph;
            const uint64_t ar = gpu_graph_attn_comp_cache_row_bytes();
            uint8_t sa[4096], sb[4096];
            int bad = 0;
            for (uint32_t il = 0; il < PULSAR_N_LAYER && bad < 3; il++) {
                if (pulsar_layer_compress_ratio(il) != 4u) continue;
                pulsar_gpu_tensor *va = gpu_graph_bank_attn_comp_view(g, il, 0);
                pulsar_gpu_tensor *vb = gpu_graph_bank_attn_comp_view(g, il, 2);
                pulsar_gpu_synchronize();
                if (va && vb && pulsar_gpu_tensor_read(va, (uint64_t)(RCUT/4) * ar, sa, ar) &&
                    pulsar_gpu_tensor_read(vb, (uint64_t)(RCUT/4) * ar, sb, ar) &&
                    memcmp(sa, sb, (size_t)ar) != 0) bad++;
                pulsar_gpu_tensor_free(va); pulsar_gpu_tensor_free(vb);
            }
            fprintf(stderr, "fork_gate: TRIAGE src==cold row %d : %s\n", RCUT/4, bad ? "NO (source-shape divergence)" : "YES");
        }
        CHECK(sum_pf == sum_pc, "P2 partial-fork KV != cold (%016" PRIx64 " vs %016" PRIx64 ")",
              sum_pf, sum_pc);
        CHECK(tok_pf == tok_pc, "P2 next-token %d != cold %d", tok_pf, tok_pc);
        fprintf(stderr, "fork_gate: P2 partial(mid-group R=%d) checksum=%016" PRIx64
                " cold=%016" PRIx64 " (byte-identical: %s) tok %d/%d\n",
                RCUT, sum_pf, sum_pc, sum_pf == sum_pc ? "YES" : "NO", tok_pf, tok_pc);

        /* SPECIFIC boundary-row byte-diff: comp + MXFP4 index row R/4 per
         * ratio-4 layer, fork vs cold (risks 1+2 — a checksum can hide a
         * single half-restored row only if another row compensates; this
         * cannot). */
        {
            pulsar_gpu_graph *g = &s->graph;
            const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
            const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
                ? PULSAR_ENGINE_IDXFP4_ROWBYTES : (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
            uint8_t *ra = (uint8_t *)malloc((size_t)attn_row), *rb = (uint8_t *)malloc((size_t)attn_row);
            int diffc = 0, diffi = 0, checked = 0;
            for (uint32_t il = 0; il < PULSAR_N_LAYER; il++) {
                if (pulsar_layer_compress_ratio(il) != 4u) continue;
                pulsar_gpu_tensor *va = gpu_graph_bank_attn_comp_view(g, il, 1);
                pulsar_gpu_tensor *vb = gpu_graph_bank_attn_comp_view(g, il, 2);
                if (va && vb &&
                    pulsar_gpu_tensor_read(va, (uint64_t)(RCUT/4) * attn_row, ra, attn_row) &&
                    pulsar_gpu_tensor_read(vb, (uint64_t)(RCUT/4) * attn_row, rb, attn_row)) {
                    if (memcmp(ra, rb, (size_t)attn_row) != 0) diffc++;
                } else diffc += 1000;   /* read failure = loud */
                pulsar_gpu_tensor_free(va); pulsar_gpu_tensor_free(vb);
                va = gpu_graph_bank_index_comp_view(g, il, 1);
                vb = gpu_graph_bank_index_comp_view(g, il, 2);
                if (va && vb &&
                    pulsar_gpu_tensor_read(va, (uint64_t)(RCUT/4) * idx_row, ra, idx_row) &&
                    pulsar_gpu_tensor_read(vb, (uint64_t)(RCUT/4) * idx_row, rb, idx_row)) {
                    if (memcmp(ra, rb, (size_t)idx_row) != 0) diffi++;
                } else diffi += 1000;
                pulsar_gpu_tensor_free(va); pulsar_gpu_tensor_free(vb);
                checked++;
            }
            free(ra); free(rb);
            CHECK(diffc == 0, "P2 boundary COMP row %d differs on %d ratio-4 layer(s)", RCUT/4, diffc);
            CHECK(diffi == 0, "P2 boundary MXFP4 INDEX row %d differs on %d layer(s)", RCUT/4, diffi);
            fprintf(stderr, "fork_gate: P2 boundary row %d byte-diff: comp %d, index %d "
                    "(over %d ratio-4 layers) : %s\n", RCUT/4, diffc, diffi, checked,
                    (diffc == 0 && diffi == 0) ? "OK" : "FAIL");
        }

        /* P3: src==dst TRUNCATE-reuse — rewind bank 1 (@L2) to R and replay the
         * SAME tokens; must again be byte-identical to the cold control. */
        CHECK(pulsar_session_bank_state_restore(s, 1), "P3 install bank1");
        const int trc = pulsar_session_bank_fork_partial(s, 1, 1, toks2, NC);
        CHECK(trc == 0, "P3 truncate refused rc=%d", trc);
        CHECK(pulsar_session_pos(s) == RCUT, "P3 truncated pos %d != %d", pulsar_session_pos(s), RCUT);
        {
            pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks2; p.len = p.cap = L2;
            char e2[256];
            CHECK(pulsar_session_sync(s, &p, e2, sizeof e2) == 0, "P3 replay sync: %s", e2);
            gpu_graph_bank_counters_capture(&s->graph, 1);
        }
        pulsar_gpu_synchronize();
        const uint64_t sum_tr = checksum_bank_kv(s, 1);
        CHECK(sum_tr == sum_pc, "P3 truncate-replay KV != cold (%016" PRIx64 " vs %016" PRIx64 ")",
              sum_tr, sum_pc);
        fprintf(stderr, "fork_gate: P3 truncate(src==dst) checksum=%016" PRIx64
                " (==cold: %s)\n", sum_tr, sum_tr == sum_pc ? "YES" : "NO");

        /* P4: wrapped-ring window guard — bank 2 is @8192 (> raw_cap), a cut at
         * R=3840 has raw rows [3840-128, 3840) scrolled out (oldest 3840+) ->
         * the engine primitive must REFUSE. */
        CHECK(!gpu_graph_bank_fork_copy_cut(&s->graph, 2, 1, 3840, 8192),
              "P4 wrapped-ring cut NOT refused (window guard broken)");
        fprintf(stderr, "fork_gate: P4 wrapped-ring cut refused : OK\n");
        free(toks2);
    }

    /* ============ P5: UNALIGNED-SOURCE output-token COHERENCE ============
     * P2 proves BYTE-identity at a chunk-ALIGNED source/cut. The PRODUCTION case
     * is different: a trunk prefilled to a NON-chunk-multiple length, cut at R,
     * then resumed. The resumed [R,L) prefill does not chunk-align to the cold
     * run's 4096 boundaries, so it runs the per-token compressor-fallback realign
     * — the engine's accepted warm-continuation numerics, a last-ulp KV delta.
     * BYTE-identity therefore does NOT hold (and asserting it would be wrong).
     * The honest oracle is OUTPUT-TOKEN COHERENCE: the fork+suffix must greedy-
     * decode the SAME token stream as a cold full-prefill of the identical text.
     * This closes the gap between P2's aligned proof and how forks actually run. */
    {
        const int SLEN5 = 5000;     /* trunk length: NOT a multiple of 4096 */
        const int NC5   = 4700;     /* shared-prefix tokens (R aligns down from here) */
        const int L5    = 6000;     /* full request length (SLEN5..L5 diverges from trunk) */
        const int NGEN  = 24;       /* decoded token-stream length to compare */
        CHECK(SLEN5 % 4096 != 0, "P5 source must be chunk-UNALIGNED (got %d)", SLEN5);
        CHECK(NC5 < SLEN5 && L5 > SLEN5, "P5 length ordering");
        const int R5 = ((NC5 - 4) / 128) * 128;   /* engine's ratio-4 align-down */

        int *toks5 = (int *)malloc((size_t)(L5 + NGEN) * sizeof(int));
        for (int i = 0; i < L5; i++)
            toks5[i] = i < NC5 ? toks[i] : base.v[(i + 31337) % base.len];

        /* trunk on bank 0, prefilled to the UNALIGNED SLEN5. */
        CHECK(prefill_bank_cold(s, 0, toks, SLEN5), "P5 trunk prefill @%d", SLEN5);
        pulsar_session_bank_state_save(s, 0);

        /* partial fork 0->1 at NC5 (cut R5), replay the divergent suffix to L5. */
        const int prc5 = pulsar_session_bank_fork_partial(s, 0, 1, toks5, NC5);
        CHECK(prc5 == 0, "P5 partial fork refused rc=%d", prc5);
        CHECK(!pulsar_session_bank_fork_pinned(s, 0), "P5 src left pinned");
        CHECK(pulsar_session_bank_state_restore(s, 1), "P5 install fork dst");
        CHECK(pulsar_session_pos(s) == R5, "P5 dst pos %d != R %d", pulsar_session_pos(s), R5);
        {
            pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks5; p.len = p.cap = L5;
            char e5[256];
            CHECK(pulsar_session_sync(s, &p, e5, sizeof e5) == 0, "P5 replay sync: %s", e5);
            CHECK(pulsar_session_pos(s) == L5, "P5 live pos %d != %d", pulsar_session_pos(s), L5);
            gpu_graph_bank_counters_capture(&s->graph, 1);
        }
        const uint64_t sum_pf5 = checksum_bank_kv(s, 1);
        /* decode the fork's stream (bank 1 installed at L5). */
        int *fbuf = (int *)malloc((size_t)(L5 + NGEN) * sizeof(int));
        memcpy(fbuf, toks5, (size_t)L5 * sizeof(int));
        int fork_stream[64];
        decode_stream_greedy(s, 1, fbuf, L5, NGEN, fork_stream);
        pulsar_session_bank_state_save(s, 1);

        /* COLD control: bank 2 full prefill of the identical text, same decode. */
        CHECK(prefill_bank_cold(s, 2, toks5, L5), "P5 cold prefill @%d", L5);
        const uint64_t sum_pc5 = checksum_bank_kv(s, 2);
        int *cbuf = (int *)malloc((size_t)(L5 + NGEN) * sizeof(int));
        memcpy(cbuf, toks5, (size_t)L5 * sizeof(int));
        int cold_stream[64];
        decode_stream_greedy(s, 2, cbuf, L5, NGEN, cold_stream);

        /* THE ORACLE: identical output-token STREAM (NOT KV bytes). */
        int firstdiff = -1;
        for (int i = 0; i < NGEN; i++)
            if (fork_stream[i] != cold_stream[i]) { firstdiff = i; break; }
        CHECK(firstdiff < 0,
              "P5 output-token stream diverges at %d (fork %d != cold %d)",
              firstdiff, firstdiff < 0 ? 0 : fork_stream[firstdiff],
              firstdiff < 0 ? 0 : cold_stream[firstdiff]);
        /* Informational: the KV IS expected to differ (unaligned last-ulp delta);
         * a byte-identical result here would mean the source happened to align. */
        fprintf(stderr,
                "fork_gate: P5 unaligned-source(SLEN=%d R=%d) coherence: %d/%d tokens "
                "match cold (KV byte-identical: %s — delta expected on unaligned)\n",
                SLEN5, R5, firstdiff < 0 ? NGEN : firstdiff, NGEN,
                sum_pf5 == sum_pc5 ? "YES" : "no");
        free(toks5); free(fbuf); free(cbuf);
    }

    /* ============ P6: SERVER-CONFIG work-skipped TEETH (dst == cur) ==========
     * The value bug: the server provisions dst, INSTALLS it as cur, and
     * INVALIDATES the live session (s->checkpoint.len -> 0) BEFORE forking.
     * P2/P3/P5 fork into a NON-cur dst and re-install it, so they never
     * exercised this path — and it left the frontier at 0, so the server
     * re-prefilled 100% of the prompt (fork "fired" but skipped ZERO work; the
     * correctness gates stayed green because re-prefill-from-0 IS byte-identical).
     * These teeth assert bank_pos(dst) == cut in the dst==cur config, so a
     * 0-benefit fork can never pass green again — for BOTH full and partial. */
    {
        const int FLEN = 8192, SLEN6 = 8192, NC6 = 4102, R6 = 4096, L6 = 8192;

        /* ---- FULL fork, dst==cur: bank_pos(dst) must be n_cached, not 0 ---- */
        CHECK(prefill_bank_cold(s, 0, toks, FLEN), "P6 full src prefill");
        pulsar_session_bank_state_save(s, 0);
        CHECK(pulsar_session_bank_state_restore(s, 1), "P6 install dst(1) as cur");
        pulsar_session_invalidate(s);                    /* mimic provision: cur len -> 0 */
        CHECK(pulsar_session_pos(s) == 0, "P6 precondition: dst==cur must start at 0");
        CHECK(pulsar_session_bank_fork(s, 0, 1, toks, FLEN) == 0, "P6 full fork (dst==cur) refused");
        CHECK(pulsar_session_bank_pos(s, 1) == FLEN,
              "P6 FULL dst==cur bank_pos %d != %d (WORK-SKIPPED REGRESSION: "
              "server would re-prefill from 0)", pulsar_session_bank_pos(s, 1), FLEN);
        CHECK(pulsar_session_pos(s) == FLEN, "P6 full live pos %d != %d", pulsar_session_pos(s), FLEN);
        fprintf(stderr, "fork_gate: P6 FULL fork dst==cur bank_pos=%d (==%d: %s)\n",
                pulsar_session_bank_pos(s, 1), FLEN,
                pulsar_session_bank_pos(s, 1) == FLEN ? "YES" : "NO");

        /* ---- PARTIAL fork, dst==cur: bank_pos(dst) must be R, and the tail
         *      [R,L) re-prefill lands byte-identical to cold (aligned oracle) --- */
        int *toks6 = (int *)malloc((size_t)L6 * sizeof(int));
        for (int i = 0; i < L6; i++)
            toks6[i] = i < NC6 ? toks[i] : base.v[(i + 5150) % base.len];
        CHECK(prefill_bank_cold(s, 0, toks, SLEN6), "P6 partial src prefill");
        pulsar_session_bank_state_save(s, 0);
        CHECK(pulsar_session_bank_state_restore(s, 1), "P6 install dst(1) as cur (partial)");
        pulsar_session_invalidate(s);
        CHECK(pulsar_session_bank_fork_partial(s, 0, 1, toks6, NC6) == 0,
              "P6 partial fork (dst==cur) refused");
        CHECK(pulsar_session_bank_pos(s, 1) == R6,
              "P6 PARTIAL dst==cur bank_pos %d != R %d (WORK-SKIPPED REGRESSION)",
              pulsar_session_bank_pos(s, 1), R6);
        CHECK(pulsar_session_pos(s) == R6, "P6 partial live pos %d != %d", pulsar_session_pos(s), R6);
        {   /* replay ONLY the tail [R6,L6) — the (L6-R6) tokens actually skipped */
            pulsar_tokens p; memset(&p, 0, sizeof p); p.v = toks6; p.len = p.cap = L6;
            char e6[256];
            CHECK(pulsar_session_sync(s, &p, e6, sizeof e6) == 0, "P6 tail replay: %s", e6);
            CHECK(pulsar_session_pos(s) == L6, "P6 post-replay pos %d != %d", pulsar_session_pos(s), L6);
            gpu_graph_bank_counters_capture(&s->graph, 1);
        }
        pulsar_gpu_synchronize();
        const uint64_t sum_p6 = checksum_bank_kv(s, 1);
        pulsar_session_bank_state_save(s, 1);
        CHECK(prefill_bank_cold(s, 2, toks6, L6), "P6 cold control");
        pulsar_gpu_synchronize();
        const uint64_t sum_c6 = checksum_bank_kv(s, 2);
        CHECK(sum_p6 == sum_c6,
              "P6 partial(dst==cur)+tail KV != cold (%016" PRIx64 " vs %016" PRIx64 ")",
              sum_p6, sum_c6);
        fprintf(stderr, "fork_gate: P6 PARTIAL fork dst==cur bank_pos=%d==R "
                "(tail-only replay %d tok; KV==cold: %s)\n",
                R6, L6 - R6, sum_p6 == sum_c6 ? "YES" : "NO");
        free(toks6);
    }

    free(toks);
    fprintf(stderr, "BANK-FORK GATE: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
