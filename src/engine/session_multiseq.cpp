#include "pulsar_engine_internal.h"

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
int pulsar_session::decode_multiseq(const pulsar_multiseq_req *reqs,
                                uint32_t n, float *logits, int logits_cap,
                                char *err, size_t errlen) {
    auto *s = this;
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
int pulsar_session::decode_mixed(const pulsar_multiseq_req *reqs,
                             uint32_t n_rows, float *logits, int logits_cap,
                             uint32_t *out_n_rows, uint32_t max_head_runs,
                             char *err, size_t errlen) {
    auto *s = this;
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
