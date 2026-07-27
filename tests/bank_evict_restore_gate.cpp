/* Tier-2 BANK EVICT/RESTORE gate (task #55 increment 2b, the memory-safety core).
 *
 * Proves the per-bank physical evict/restore cycle the proactive-eviction guard
 * relies on is (a) a real physical reclaim and (b) KV-BIT-IDENTICAL on return —
 * WITHOUT the box-lock risk of the full-server smoke (single session, controlled,
 * fills modest, never approaches OOM).
 *
 * Flow (mirrors the server guard's evict/restore, minus the disk I/O — the
 * snapshot lives in host RAM here; on the server it is the disk KV cache):
 *   1. prefill bank 0 to L tokens; capture its frontier; CHECKSUM its comp/index
 *      rows (raw D2H fold).
 *   2. save_snapshot(bank 0)   — the D2H KV snapshot (server: kv_cache "evict").
 *   3. repoint to bank 1       — bank 0 is now idle (free_physical refuses cur).
 *   4. free_physical(bank 0)   — DIRECT cudaFree of bank 0's split comp/index;
 *      assert cudaMemGetInfo free ROSE (physical returned), is_evicted==true.
 *   5. alloc_physical(bank 0)  — fresh cudaMallocManaged + base-table rebuild;
 *      assert the rebuilt comp_bases[0] entry == the new comp[il][0] ptr.
 *   6. repoint to bank 0; load_snapshot — H2D reload (server: kv_cache_try_load).
 *   7. CHECKSUM bank 0's comp/index again; assert == step 1 (bit-identical).
 *
 * MODEL-DEPENDENT, GPU-resident; run manually under the memory discipline (hold
 * temp/gpu.lock, drop_caches, no foreign ds4 process). NOT part of `make test`.
 *
 * usage: PULSAR_MSEQ_BANKS=2 ./tests/bank_evict_restore_gate MODEL [L]
 */
#include "pulsar.h"
#include "pulsar_engine_internal.h"
#include "pulsar_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static const double GIB = 1024.0 * 1024.0 * 1024.0;
static int g_fail;
#define CHECK(cond, ...) do { if(!(cond)){ fprintf(stderr,"EVICT-RESTORE FAIL: " __VA_ARGS__); fprintf(stderr,"\n"); g_fail=1; } } while(0)

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb"); if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp); buf[n] = '\0'; if (len_out) *len_out = (size_t)n; return buf;
}

/* FNV-1a fold of bank `bank`'s captured comp+index frontier rows (raw D2H). 0 on
 * a read failure (e.g. an evicted bank) — the caller only checksums resident banks. */
static uint64_t checksum_bank_kv(pulsar_session *s, uint32_t bank) {
    pulsar_gpu_graph *g = &s->graph;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? PULSAR_ENGINE_IDXFP4_ROWBYTES : (uint64_t)PULSAR_N_INDEXER_HEAD_DIM * sizeof(float);
    uint64_t h = 1469598103934665603ull;
    uint8_t *buf = (uint8_t *)malloc(64u * 1024u * 1024u);   /* per-layer row block scratch */
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

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [L]\n", argv[0]); return 2; }
    const int L = argc > 2 ? atoi(argv[2]) : 24576;
    const int ctx = L + 4096;

    pulsar_engine *e = NULL;
    pulsar_engine_options opt; memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1]; opt.backend = PULSAR_BACKEND_CUDA;
    if (pulsar_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    size_t tl = 0; char *text = read_file("tests/long_context_story_prompt.txt", &tl);
    if (!text) { fprintf(stderr, "prompt read failed\n"); return 1; }
    pulsar_tokens base; memset(&base, 0, sizeof(base));
    pulsar_tokenize_text(e, text, &base); free(text);
    if (base.len < 256) { fprintf(stderr, "prompt too short\n"); return 1; }

    pulsar_session *s = NULL;
    if (pulsar_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session create failed\n"); return 1; }
    const uint32_t pool = gpu_graph_bank_pool_count(&s->graph);
    fprintf(stderr, "evict_restore_gate: pool banks=%u ctx=%d L=%d\n", pool, ctx, L);
    if (pool < 2) { fprintf(stderr, "need DS4_MSEQ_BANKS>=2\n"); return 1; }

    /* 1. Prefill bank 0 (cur=0), capture its frontier, checksum its KV. */
    int *toks = (int *)malloc((size_t)L * sizeof(int));
    for (int i = 0; i < L; i++) toks[i] = base.v[i % base.len];
    pulsar_tokens p; memset(&p, 0, sizeof(p)); p.v = toks; p.len = p.cap = L;
    char err[256];
    if (pulsar_session_sync(s, &p, err, sizeof(err)) != 0) { fprintf(stderr, "sync failed: %s\n", err); return 1; }
    gpu_graph_bank_counters_capture(&s->graph, 0);
    (void)pulsar_gpu_synchronize();
    const uint64_t sum_before = checksum_bank_kv(s, 0);
    const uint64_t touched0 = pulsar_session_bank_touched_kv_bytes(s, 0);
    CHECK(sum_before != 0, "checksum_before failed");
    fprintf(stderr, "evict_restore_gate: bank0 prefilled pos=%d touched=%.3f GiB checksum=%016" PRIx64 "\n",
            pulsar_session_pos(s), (double)touched0 / GIB, sum_before);

    /* 2. Save bank 0's comp/index KV to a DISK file (the real server mechanism;
     * bank 0 is cur). Measure disk-snapshot latency (an eviction stalls the
     * evicted session's next turn by save+reload time). */
    pulsar_gpu_graph *g = &s->graph;
    const char *snap_path = "/home/tyler/Projects/AI/temp/tier2-inc1/bank0.kvsnap";
    { double t0 = now_ms();
      FILE *fp = fopen(snap_path, "wb");
      CHECK(fp != NULL, "open snap for write");
      if (fp) { CHECK(pulsar_session_bank_kv_save(s, 0, fp, err, sizeof err) == 0, "kv_save: %s", err); fclose(fp); }
      double t1 = now_ms();
      fprintf(stderr, "evict_restore_gate: kv_save %.1f ms\n", (t1 - t0)); }

    uint64_t f_prefill = 0, tt = 0; (void)pulsar_gpu_synchronize(); pulsar_gpu_mem_info(&f_prefill, &tt);

    /* 3. Repoint to bank 1 so bank 0 is idle (free_physical refuses cur). */
    CHECK(pulsar_session_bank_state_restore(s, 1), "repoint to bank 1 failed");
    uint64_t f_repoint = 0; (void)pulsar_gpu_synchronize(); pulsar_gpu_mem_info(&f_repoint, &tt);

    /* 4. free_physical(bank 0) — DIRECT cudaFree; physical must return. */
    CHECK(pulsar_session_bank_free_physical(s, 0), "free_physical failed");
    (void)pulsar_gpu_synchronize();
    uint64_t f_free = 0; pulsar_gpu_mem_info(&f_free, &tt);
    const int64_t reclaimed = (int64_t)f_free - (int64_t)f_repoint;
    CHECK(pulsar_session_bank_is_evicted(s, 0), "bank 0 not marked evicted after free");
    fprintf(stderr, "evict_restore_gate: free[GiB] prefill=%.3f repoint=%.3f afterfree=%.3f reclaimed=%.3f (touched=%.3f)\n",
            (double)f_prefill/GIB, (double)f_repoint/GIB, (double)f_free/GIB, (double)reclaimed/GIB, (double)touched0/GIB);
    /* Reclaim must be POSITIVE (physical returned; refutes #50). The exact amount
     * is unreliable — cudaMemGetInfo is coarse/laggy on driver-610 UVM (under-
     * reports small managed frees; 2a saw a clean 0.264 GiB at 65k). The guard
     * decides on the deterministic touched accounting, never this gauge. */
    CHECK(reclaimed > 0, "free_physical reclaimed <= 0 (arena-pool regression?)");

    /* 5-6. Restore bank 0 from disk: alloc_physical + base-table rebuild + counter
     * reinstall + H2D reload, all inside kv_load (leaves bank 0 installed). */
    { double t0 = now_ms();
      FILE *fp = fopen(snap_path, "rb");
      CHECK(fp != NULL, "open snap for read");
      if (fp) { CHECK(pulsar_session_bank_kv_load(s, 0, fp, err, sizeof err) == 0, "kv_load: %s", err); fclose(fp); }
      double t1 = now_ms();
      fprintf(stderr, "evict_restore_gate: kv_load %.1f ms\n", (t1 - t0)); }
    CHECK(!pulsar_session_bank_is_evicted(s, 0), "bank 0 still evicted after restore");
    /* base-table entry must point at the fresh comp[il][0]. */
    { int checked = 0;
      for (uint32_t il = 0; il < PULSAR_N_LAYER && checked < 4; il++) {
          if (pulsar_layer_compress_ratio(il) == 0) continue;
          void *want = pulsar_gpu_tensor_device_ptr(g->banks.comp[il][0]);
          void *got = NULL; (void)pulsar_gpu_synchronize();
          pulsar_gpu_tensor_read(g->banks.comp_bases[il], 0, &got, sizeof(void *));
          CHECK(got == want, "comp_bases[%u][0] rebuild mismatch", il);
          checked++; } }

    /* 7. checksum again — must be bit-identical to before evict. */
    (void)pulsar_gpu_synchronize();
    const uint64_t sum_after = checksum_bank_kv(s, 0);
    CHECK(sum_after == sum_before, "KV NOT bit-identical after evict/restore (%016" PRIx64 " vs %016" PRIx64 ")",
          sum_after, sum_before);
    fprintf(stderr, "evict_restore_gate: restored checksum=%016" PRIx64 " (bit-identical: %s)\n",
            sum_after, sum_after == sum_before ? "YES" : "NO");

    /* ===== Review finding 2 — a freed bank stops counting toward touched (so the
     * guard evicts exactly the minimum, not the whole idle set / cascade). Prefill
     * bank 1 too; with bank 0 cur, free bank 1 and assert touched drops by EXACTLY
     * bank 1's frontier (bank 0's contribution remains). ===== */
    CHECK(pulsar_session_bank_state_restore(s, 1), "finding2: repoint to bank 1");
    pulsar_session_invalidate(s);   /* clear the carried-over bank-0 checkpoint so the
                                  * sync actually prefills bank 1 from scratch */
    { pulsar_tokens p1; memset(&p1, 0, sizeof p1); p1.v = toks; p1.len = p1.cap = L;
      char e2[256];
      CHECK(pulsar_session_sync(s, &p1, e2, sizeof e2) == 0, "finding2: bank1 sync: %s", e2); }
    gpu_graph_bank_counters_capture(&s->graph, 1);
    CHECK(pulsar_session_bank_pos(s, 1) == L, "finding2: bank1 prefill pos=%d != %d (test setup)",
          pulsar_session_bank_pos(s, 1), L);
    CHECK(pulsar_session_bank_state_restore(s, 0), "finding2: repoint back to bank 0");
    (void)pulsar_gpu_synchronize();
    const uint64_t t_b0 = pulsar_session_bank_touched_kv_bytes(s, 0);
    const uint64_t t_b1 = pulsar_session_bank_touched_kv_bytes(s, 1);
    const uint64_t t_both = pulsar_session_touched_kv_bytes(s);
    CHECK(t_both == t_b0 + t_b1, "finding2: pool touched (%.3f) != bank0+bank1 (%.3f+%.3f) GiB",
          (double)t_both/GIB, (double)t_b0/GIB, (double)t_b1/GIB);
    CHECK(pulsar_session_bank_free_physical(s, 1), "finding2: free bank 1 (cur is 0)");
    const uint64_t t_after = pulsar_session_touched_kv_bytes(s);
    CHECK(t_after == t_b0,
          "finding2 (CASCADE BUG): touched after free bank1 = %.3f GiB, expected bank0-only %.3f GiB "
          "— a freed bank still counts, so the guard would spill every idle bank on one breach",
          (double)t_after/GIB, (double)t_b0/GIB);
    fprintf(stderr, "evict_restore_gate: finding-2 no-cascade: touched both=%.3f -> after free bank1=%.3f "
            "(bank0=%.3f) : %s\n", (double)t_both/GIB, (double)t_after/GIB, (double)t_b0/GIB,
            t_after == t_b0 ? "OK" : "FAIL");

    /* ===== Review finding 3 — a corrupt/short snapshot must fail kv_load SAFELY:
     * validate counts <= cap, don't install rows over unwritten KV. Bank 1 is now
     * freed (frontier 0). ===== */
    const char *cpath = "/home/tyler/Projects/AI/temp/tier2-inc1/corrupt.kvsnap";
    /* (a) count > cap. */
    { FILE *fp = fopen(cpath, "wb");
      uint32_t hdr[4] = { 0x4B564232u, 1u, 1u, (uint32_t)PULSAR_N_LAYER };
      fwrite(hdr, sizeof hdr, 1, fp);
      for (uint32_t il = 0; il < (uint32_t)PULSAR_N_LAYER; il++) { uint32_t c[2] = { 0xFFFFFFFFu, 0u }; fwrite(c, sizeof c, 1, fp); }
      fclose(fp);
      fp = fopen(cpath, "rb");
      const int rc = pulsar_session_bank_kv_load(s, 1, fp, err, sizeof err);
      fclose(fp);
      CHECK(rc != 0, "finding3(a): kv_load ACCEPTED a count > cap");
      CHECK(pulsar_session_bank_touched_kv_bytes(s, 1) == 0, "finding3(a): bank1 advertises rows after failed load");
      fprintf(stderr, "evict_restore_gate: finding-3(a) count>cap rejected: rc=%d touched(1)=%.3f : %s\n",
              rc, (double)pulsar_session_bank_touched_kv_bytes(s, 1)/GIB, (rc != 0) ? "OK" : "FAIL"); }
    /* (b) truncated: valid small counts but the row bytes are missing (short read). */
    { FILE *fp = fopen(cpath, "wb");
      uint32_t hdr[4] = { 0x4B564232u, 1u, 1u, (uint32_t)PULSAR_N_LAYER };
      fwrite(hdr, sizeof hdr, 1, fp);
      for (uint32_t il = 0; il < (uint32_t)PULSAR_N_LAYER; il++) {
          uint32_t c[2] = { pulsar_layer_compress_ratio(il) ? 4u : 0u, 0u }; fwrite(c, sizeof c, 1, fp); }
      /* no row data written -> fread of rows fails */
      fclose(fp);
      fp = fopen(cpath, "rb");
      const int rc = pulsar_session_bank_kv_load(s, 1, fp, err, sizeof err);
      fclose(fp);
      CHECK(rc != 0, "finding3(b): kv_load ACCEPTED a truncated file");
      CHECK(pulsar_session_bank_touched_kv_bytes(s, 1) == 0, "finding3(b): bank1 advertises rows after truncated load");
      fprintf(stderr, "evict_restore_gate: finding-3(b) truncated rejected: rc=%d touched(1)=%.3f : %s\n",
              rc, (double)pulsar_session_bank_touched_kv_bytes(s, 1)/GIB, (rc != 0) ? "OK" : "FAIL"); }
    remove(cpath);

    remove(snap_path);
    free(toks);
    fprintf(stderr, "EVICT-RESTORE GATE: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
