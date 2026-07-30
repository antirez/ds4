#include "pulsar_engine_internal.h"

/* Tier-2 task #55 increment 2b — per-bank physical evict/restore.  The caller
 * (server guard) must have snapshotted the bank's KV to DISK before evict (host
 * RAM reclaims nothing on unified memory) and repointed away from it; and after
 * restore-alloc it reloads the KV H2D from that snapshot. */
bool pulsar_session::bank_free_physical(uint32_t bank) {
    auto *s = this;
    if (!s) return false;
    return gpu_graph_bank_free_physical(&s->graph, bank);
}

bool pulsar_session::bank_alloc_physical(uint32_t bank) {
    auto *s = this;
    if (!s) return false;
    return gpu_graph_bank_alloc_physical(&s->graph, bank);
}

bool pulsar_session::bank_is_evicted(uint32_t bank) const {
    auto *s = this;
    if (!s) return false;
    return gpu_graph_bank_is_evicted(&s->graph, bank);
}

uint64_t pulsar_session::bank_touched_kv_bytes(uint32_t bank) {
    auto *s = this;
    if (!s) return 0;
    return gpu_graph_bank_touched_kv_bytes(&s->graph, bank);
}

uint64_t pulsar_session::quantum_growth_bytes_per_bank(uint32_t q) {
    auto *s = this;
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
void pulsar_bank_carry::copy(const pulsar_bank_carry *sc) {
    auto *d = this;
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
int pulsar_session::bank_fork(uint32_t src, uint32_t dst,
                          const int *tokens, int n_cached) {
    auto *s = this;
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
    if (src == cur) s->bank_state_save(src);      /* refresh from live session */
    if (src < s->bank_carry_n && dst < s->bank_carry_n) {
        s->bank_carry[dst].copy(&s->bank_carry[src]);
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

bool pulsar_session::bank_fork_pinned(uint32_t bank) const {
    auto *s = this;
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
int pulsar_session::bank_fork_partial(uint32_t src, uint32_t dst,
                                  const int *tokens, int n_cached) {
    auto *s = this;
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
    if (src == cur) s->bank_state_save(src);
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
        s->bank_carry[dst].copy(&s->bank_carry[src]);
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

int pulsar_session::bank_kv_save(uint32_t bank, FILE *fp,
                             char *err, size_t errlen) {
    auto *s = this;
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
int pulsar_session::bank_kv_load(uint32_t bank, FILE *fp,
                             char *err, size_t errlen) {
    auto *s = this;
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

/* ===== Tier-2 PATH A: per-bank HOST carry (pulsar_bank_carry) ================
 *
 * gpu_graph_bank_repoint swaps only the DEVICE cache views; the session's HOST
 * per-conversation state (checkpoint history, host logits, the DSpark fused-
 * loop / spec-carry shadow) is single-instance and must be saved for the bank
 * we leave and restored for the bank we enter.  save/restore below are that
 * host half; the graph frontier counters ride gpu_graph_bank_counters_*, and
 * (Option F) the per-bank drafter ring rides gpu_graph_bank_repoint. */

void pulsar_bank_carry::free_one() {
    auto *c = this;
    if (!c) return;
    token_vec_free(&c->checkpoint);
    free(c->logits);
    free(c->dspark_pending_qrows);
    memset(c, 0, sizeof(*c));
}

void pulsar_session::bank_carry_free() {
    auto *s = this;
    if (!s || !s->bank_carry) return;
    for (uint32_t i = 0; i < s->bank_carry_n; i++) s->bank_carry[i].free_one();
    free(s->bank_carry);
    s->bank_carry = NULL;
    s->bank_carry_n = 0;
}

static bool bank_carry_ensure(pulsar_session *s) {
    const uint32_t n = gpu_graph_bank_pool_count(&s->graph);
    if (s->bank_carry && s->bank_carry_n == n) return true;
    if (s->bank_carry) s->bank_carry_free();
    s->bank_carry = (pulsar_bank_carry *)xcalloc(n, sizeof(*s->bank_carry));
    s->bank_carry_n = n;
    return true;
}

int pulsar_session::bank_count() {
    auto *s = this;
    return s ? (int)gpu_graph_bank_pool_count(&s->graph) : 0;
}

int pulsar_session::bank_repoint(uint32_t bank) {
    auto *s = this;
    if (!s || bank >= gpu_graph_bank_pool_count(&s->graph)) return 1;
    /* Pool disabled: bank 0 is the classic tensors, nothing to repoint. */
    if (s->graph.banks.n_banks == 0) return bank == 0 ? 0 : 1;
    return gpu_graph_bank_repoint(&s->graph, bank) ? 0 : 1;
}

void pulsar_session::bank_state_save(uint32_t bank) {
    auto *s = this;
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

bool pulsar_session::bank_state_restore(uint32_t bank) {
    auto *s = this;
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

int pulsar_session::bank_pos(uint32_t bank) {
    auto *s = this;
    const pulsar_tokens *t = bank_frontier_tokens(s, bank);
    return t ? t->len : 0;
}

const pulsar_tokens *pulsar_session::bank_tokens(uint32_t bank) {
    auto *s = this;
    return bank_frontier_tokens(s, bank);
}

int pulsar_session::bank_common_prefix(uint32_t bank,
                                   const pulsar_tokens *prompt) {
    auto *s = this;
    const pulsar_tokens *t = bank_frontier_tokens(s, bank);
    if (!t || !prompt) return 0;
    int n = t->len < prompt->len ? t->len : prompt->len;
    int i = 0;
    while (i < n && t->v[i] == prompt->v[i]) i++;
    return i;
}
