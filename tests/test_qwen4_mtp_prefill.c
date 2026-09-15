#include "../ds4.c"
#include <assert.h>

#ifdef DS4_HAS_QWEN4_METAL
/* Compare cache-only prefix preparation with the existing complete MTP
 * layer on exactly the same trunk rows. This uses real Metal/model weights;
 * it never runs the full CPU model or substitutes a mock dequantizer. */
enum { TEST_CTX = 128, TEST_CHUNK = 8, TEST_PREFIX = 13, TEST_APPEND = 16, MAX_FRAMES = 4 };
static char error[256];

static void need(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "Qwen MTP prefill: %s: %s\n", what, error);
        exit(1);
    }
}

static void *read_bytes(const ds4_gpu_tensor *t, uint64_t bytes) {
    void *p = malloc(bytes ? (size_t)bytes : 1u);
    need(p != NULL, "host comparison allocation");
    need(!bytes || ds4_gpu_tensor_read(t, 0, p, bytes), "read comparison tensor");
    return p;
}

static void exact(const char *what, const void *a, const void *b, size_t bytes) {
    if (!memcmp(a, b, bytes)) return;
    const unsigned char *x = a, *y = b;
    size_t first = 0;
    while (first < bytes && x[first] == y[first]) first++;
    fprintf(stderr, "Qwen MTP prefill: %s differs at byte %zu/%zu\n", what, first, bytes);
    exit(1);
}

typedef struct {
    uint32_t rows, mtp_rows, residual_rows;
    uint64_t kv_bytes, ik_bytes, block_bytes, residual_bytes;
    void *k, *v, *ik, *block, *tail, *residual;
    float *logits;
} frontier;

static frontier capture(ds4_session *s, uint32_t residual_rows) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint32_t il = DS4_N_LAYER - 1u;
    const uint64_t row_bytes = (uint64_t)DS4_N_EMBD * DS4_N_HC * sizeof(float);
    frontier f = {.rows = g->pos, .mtp_rows = g->mtp_pos, .residual_rows = residual_rows};
    f.kv_bytes = qwen4_payload_kv_bytes(f.mtp_rows);
    f.ik_bytes = qwen4_payload_ik_bytes(f.mtp_rows);
    f.block_bytes = qwen4_payload_block_key_bytes(f.mtp_rows);
    f.residual_bytes = residual_rows * row_bytes;
    f.k = read_bytes(g->layer_k_cache[il], f.kv_bytes);
    f.v = read_bytes(g->layer_v_cache[il], f.kv_bytes);
    f.ik = read_bytes(g->layer_ik_cache[il], f.ik_bytes);
    f.block = read_bytes(g->layer_block_key[il], f.block_bytes);
    f.tail = read_bytes(g->mtp_tail_R, row_bytes);
    f.residual = read_bytes(g->R, f.residual_bytes);
    f.logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    need(f.logits != NULL, "logits comparison allocation");
    memcpy(f.logits, s->logits, (size_t)DS4_N_VOCAB * sizeof(float));
    return f;
}

static void free_frontier(frontier *f) {
    free(f->k); free(f->v); free(f->ik); free(f->block);
    free(f->tail); free(f->residual); free(f->logits);
    memset(f, 0, sizeof(*f));
}

static void compare_frontier(ds4_session *s, const frontier *want, bool residual) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    need(s->checkpoint_valid && g->pos == want->rows &&
         ds4_session_pos(s) == (int)want->rows, "target frontier position");
    need(g->mtp_pos == want->mtp_rows, "contiguous MTP prefix position");
    need(g->mtp_tail_valid && g->mtp_tail_pos + 1u == g->pos, "last trunk row retained");
    frontier got = capture(s, residual ? want->residual_rows : 0);
    exact("nextn K", want->k, got.k, want->kv_bytes);
    exact("nextn V", want->v, got.v, want->kv_bytes);
    exact("nextn indexer keys", want->ik, got.ik, want->ik_bytes);
    exact("nextn pooled keys", want->block, got.block, want->block_bytes);
    exact("retained trunk row", want->tail, got.tail,
          (size_t)DS4_N_EMBD * DS4_N_HC * sizeof(float));
    if (residual) exact("unmodified trunk residual", want->residual, got.residual, want->residual_bytes);
    exact("unmodified target logits", want->logits, got.logits, (size_t)DS4_N_VOCAB * sizeof(float));
    free_frontier(&got);
}

static void poison_nextn(ds4_session *s, unsigned char value) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint32_t il = DS4_N_LAYER - 1u;
    ds4_gpu_tensor *t[] = {g->layer_k_cache[il], g->layer_v_cache[il],
                          g->layer_ik_cache[il], g->layer_block_key[il]};
    for (unsigned i = 0; i < 4; i++) {
        const uint64_t bytes = ds4_gpu_tensor_bytes(t[i]);
        void *p = malloc((size_t)bytes);
        need(p != NULL, "poison allocation"); memset(p, value, (size_t)bytes);
        need(ds4_gpu_tensor_write(t[i], 0, p, bytes), "poison old nextn cache");
        free(p);
    }
}

/* Frozen pre-optimization oracle: keep the full zero-padded EH projection
 * and all-row MoE here. Sharing the optimized input/steps helpers would let
 * both sides reproduce the same regression. Full attention remains the
 * independent oracle for the separate cache-only attention preparation. */
static bool reference_mtp_input(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                   uint32_t row, const int *next_tokens, uint32_t T) {
    const uint32_t E = DS4_N_EMBD, hc = DS4_N_HC;
    const ds4_layer_weights *l = &w->layer[DS4_N_LAYER - 1u];
    for (uint32_t t = 0; t < T; t++) {
        if (next_tokens[t] < 0 || next_tokens[t] >= (int)DS4_N_VOCAB) return false;
        qwen4_ref_row(m, w->token_embd, (uint64_t)next_tokens[t], g->host_row + (uint64_t)t * E);
    }
    if (!ds4_gpu_tensor_write(g->mtp_e, 0, g->host_row, (uint64_t)T * E * sizeof(float)) ||
        !glm_graph_begin_commands_if_needed()) return false;
    ds4_gpu_tensor *R_save = g->R;
    bool ok = true;
    const uint64_t emb_bytes = (uint64_t)E * sizeof(float);
    const uint64_t cat_bytes = (hc + 1u) * 2u * emb_bytes;
    const uint64_t proj_bytes = (hc + 1u) * emb_bytes;
    for (uint32_t t = 0; t < T && ok; t++) {
        ds4_gpu_tensor *e_row = ds4_gpu_tensor_view(g->mtp_e, t * emb_bytes, emb_bytes);
        ds4_gpu_tensor *R_row = ds4_gpu_tensor_view(R_save, (uint64_t)(row + t) * hc * emb_bytes, hc * emb_bytes);
        ds4_gpu_tensor *cat_row = ds4_gpu_tensor_view(g->mtp_cat, t * cat_bytes, cat_bytes);
        ok = e_row && R_row && cat_row &&
             ds4_gpu_qwen4_mtp_stage_tensor(cat_row, e_row, R_row, m->map, m->size,
                                             l->nextn_enorm->abs_offset, l->nextn_hnorm->abs_offset,
                                             E, hc, DS4_RMS_EPS);
        ds4_gpu_tensor_free(cat_row);
        ds4_gpu_tensor_free(R_row);
        ds4_gpu_tensor_free(e_row);
    }
    if (ok) ok = qwen4_gemv(g->mtp_proj, m, l->nextn_eh_proj, g->mtp_cat, T * (hc + 1u));
    for (uint32_t t = 0; t < T && ok; t++) {
        ds4_gpu_tensor *proj_row = ds4_gpu_tensor_view(g->mtp_proj, t * proj_bytes, proj_bytes);
        ds4_gpu_tensor *R_row = ds4_gpu_tensor_view(g->mtp_R, t * hc * emb_bytes, hc * emb_bytes);
        ok = proj_row && R_row && ds4_gpu_qwen4_mtp_combine_tensor(R_row, proj_row, E, hc);
        ds4_gpu_tensor_free(R_row);
        ds4_gpu_tensor_free(proj_row);
    }
    g->R = g->mtp_R;
    if (ok) ok = qwen4_graph_hc_mix(g, m, l->hc_attn_norm, l->hc_attn_down, l->hc_attn_up, l->hc_attn_inject, T);
    g->R = R_save;
    return ok;
}

static bool reference_mtp_steps(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                 uint32_t row, const int *next_tokens, uint32_t T, uint32_t idx, bool want_logits,
                                 float *logits_out, int *draft_out) {
    if (!g->mtp_R || T == 0u || T > 3u || idx > g->mtp_pos || idx > g->ctx_cap || T > g->ctx_cap - idx ||
        row > g->cap_tokens || T > g->cap_tokens - row) return false;
    const uint32_t E = DS4_N_EMBD, hc = DS4_N_HC;
    const uint32_t il = DS4_N_LAYER - 1u;
    const ds4_layer_weights *l = &w->layer[il];
    ds4_gpu_tensor *R_save = g->R;
    const uint64_t emb_bytes = (uint64_t)E * sizeof(float);
    bool ok = reference_mtp_input(g, m, w, row, next_tokens, T);
    g->R = g->mtp_R;
    if (ok) ok = qwen4_graph_attention(g, m, l, il, idx, T);
    if (ok) ok = ds4_gpu_qwen4_hc_combine_tensor(g->R, g->blk, g->inj, T, E, hc) != 0;
    if (ok) ok = qwen4_graph_hc_mix(g, m, l->hc_ffn_norm, l->hc_ffn_down, l->hc_ffn_up, l->hc_ffn_inject, T);
    if (ok) ok = qwen4_graph_moe(g, m, l, il, T);
    ds4_gpu_tensor *last = NULL;
    const char *argmax_env = getenv("DS4_QWEN4_MTP_GPU_ARGMAX");
    const bool gpu_argmax = want_logits && draft_out && !logits_out &&
        (!argmax_env || strcmp(argmax_env, "0") != 0);
    /* draft-only rows: host logits consumers always see the full head */
    const bool gathered = gpu_argmax && qwen4_mtp_draft_head_load(g, m, w->output);
    const uint32_t head_rows = gathered ? g->draft_rows : gpu_argmax ? qwen4_mtp_draft_rows() : DS4_N_VOCAB;
    if (ok && want_logits) {
        last = ds4_gpu_tensor_view(g->mtp_R, (T - 1u) * hc * emb_bytes, hc * emb_bytes);
        g->R = last;
        ok = last && qwen4_graph_hc_mix(g, m, l->nextn_hc_head_norm, l->nextn_hc_head_down, l->nextn_hc_head_up, NULL, 1) &&
             (gathered ? ds4_gpu_qwen4_matmul_q8_0_weights_tensor(g->logits, g->draft_head, (uint32_t)w->output->dim[0],
                                                                  head_rows, g->mixed) != 0
                       : qwen4_gemv_rows(g->logits, m, w->output, g->mixed, 1, head_rows));
    }
    g->R = R_save;
    if (ok && gpu_argmax) ok = ds4_gpu_qwen4_argmax_tensor(g->mtp_argmax, g->mtp_argmax_tmp,
                                                          g->logits, head_rows) != 0;
    if (!ds4_gpu_end_commands()) ok = false;
    ds4_gpu_tensor_free(last);
    if (ok && gpu_argmax) {
        int32_t token = -1;
        ok = ds4_gpu_tensor_read(g->mtp_argmax, 0, &token, sizeof(token)) != 0;
        if (ok && gathered) {
            if (token < 0 || (uint32_t)token >= g->draft_rows) ok = false;
            else token = g->draft_ids[token];
        }
        if (ok && (token < 0 || token >= (int32_t)DS4_N_VOCAB)) ok = false;
        if (ok) *draft_out = token;
    } else if (ok && want_logits) {
        float *dst = logits_out ? logits_out : g->host_logits;
        ok = ds4_gpu_tensor_read(g->logits, 0, dst, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        if (ok && draft_out) *draft_out = sample_argmax(dst, DS4_N_VOCAB);
    }
    if (ok) {
        g->mtp_pos = idx + T;
        g->mtp_last_rows = T;
        if (g->mtp_tail_valid && idx + T - 1u == g->mtp_tail_pos) {
            g->mtp_tail_cached = true;
            g->mtp_tail_next_token = next_tokens[T - 1u];
        }
    }
    return ok;
}

/* Full MTP is the oracle. With a known lookahead it also fills the last row
 * of an intermediate chunk, so graph_forward_tokens never needs to repair
 * that boundary through the new cache-only helper. */
static unsigned reference_sync(ds4_session *s, const ds4_tokens *tokens,
                               uint32_t start, frontier out[MAX_FRAMES]) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    ds4_engine *e = s->engine;
    unsigned n = 0;
    if (!start) {
        ds4_session_invalidate(s); qwen4_graph_reset(g); poison_nextn(s, 0x55);
    } else if (g->mtp_pos < start) {
        need(g->mtp_pos + 1u == start && g->mtp_tail_valid, "reference append boundary");
        ds4_gpu_tensor *saved = g->R;
        g->R = g->mtp_tail_R;
        const bool ok = reference_mtp_steps(g, &e->model, &e->weights,
            0, tokens->v + start, 1u, start - 1u, false, NULL, NULL);
        g->R = saved;
        need(ok, "full MTP append boundary oracle");
    }
    for (uint32_t pos = start; pos < (uint32_t)tokens->len;) {
        uint32_t T = (uint32_t)tokens->len - pos;
        if (T > TEST_CHUNK) T = TEST_CHUNK;
        need(qwen4_graph_forward_tokens(g, &e->model, &e->weights,
            tokens->v + pos, T, s->logits, false), "reference trunk forward");
        uint32_t rows = T;
        if (pos + T == (uint32_t)tokens->len) rows--;
        for (uint32_t row = 0; row < rows;) {
            const uint32_t count = rows - row < 3u ? rows - row : 3u;
            need(reference_mtp_steps(g, &e->model, &e->weights, row,
                tokens->v + pos + row + 1u, count, pos + row, false, NULL, NULL),
                "full MTP prefix oracle");
            row += count;
        }
        for (uint32_t row = 0; row < T; row++) token_vec_push(&s->checkpoint, tokens->v[pos + row]);
        s->checkpoint_valid = true; s->qwen4_rewound = false;
        need(n < MAX_FRAMES, "bounded expected frontiers");
        out[n++] = capture(s, T);
        pos += T;
    }
    return n;
}

typedef struct { ds4_session *s; const frontier *frames; unsigned n, seen; } progress_check;
static void check_progress(void *opaque, const char *phase, int completed, int total) {
    progress_check *p = opaque;
    if (strcmp(phase, "prefill_chunk")) return;
    need(p->seen < p->n, "unexpected prefill callback");
    need(completed == (int)p->frames[p->seen].rows && completed <= total, "chunk callback frontier");
    compare_frontier(p->s, p->frames + p->seen++, true);
}

static void checked_sync(ds4_session *s, const ds4_tokens *tokens, frontier *f, unsigned n) {
    progress_check p = {.s = s, .frames = f, .n = n};
    ds4_session_set_progress(s, check_progress, &p);
    need(ds4_session_sync(s, tokens, error, sizeof(error)) == 0, "candidate prefix sync");
    ds4_session_set_progress(s, NULL, NULL);
    need(p.seen == n, "every completed chunk checked");
    need(s->qwen4_graph.mtp_last_rows == 0, "cache-only prefill does not publish predictor output");
    compare_frontier(s, f + n - 1u, true);
}

static void restore_mtp_case(ds4_session *s, const ds4_session_snapshot *snapshot,
                             const void *rows, bool verify) {
    need(ds4_session_load_snapshot(s, snapshot, error, sizeof(error)) == 0,
         "restore optimized-path comparison frontier");
    /* Payloads retain the final trunk row, not the whole last batch. Reuse
     * the same three actual trunk rows explicitly on both sides. */
    need(ds4_gpu_tensor_write(s->qwen4_graph.R, 0, rows,
        3ull * DS4_N_HC * DS4_N_EMBD * sizeof(float)), "restore identical trunk rows");
    s->qwen4_graph.verify_rows_exact = verify;
    ds4_gpu_qwen4_set_verify_rows_exact(verify);
}

static void compare_tensor(const char *what, const ds4_gpu_tensor *got,
                            const ds4_gpu_tensor *want, uint64_t offset, uint64_t bytes) {
    void *a = malloc((size_t)bytes), *b = malloc((size_t)bytes);
    need(a && b, "tensor comparison allocation");
    need(ds4_gpu_tensor_read(got, offset, a, bytes) &&
         ds4_gpu_tensor_read(want, offset, b, bytes), "read compared tensor range");
    exact(what, a, b, (size_t)bytes);
    free(b); free(a);
}

static void check_mtp_paths(ds4_session *live, ds4_session *reference,
                            const ds4_session_snapshot *snapshot, const int *next_tokens,
                            const void *rows) {
    ds4_engine *e = live->engine;
    ds4_qwen4_gpu_graph *g = &live->qwen4_graph, *r = &reference->qwen4_graph;
    const uint64_t residual_bytes = (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    float *want = malloc((size_t)logits_bytes), *got = malloc((size_t)logits_bytes);
    need(want && got, "predictor logits allocation");
    /* idx=13 means T3 closes pooled block [12,16), while T1/T2 retain a
     * partial block. T3 exercises both prefill and exact-verify reductions. */
    for (unsigned mode = 0; mode < 4u; mode++) {
        const uint32_t T = mode < 3u ? mode + 1u : 3u;
        const bool verify = mode == 3u;
        int want_draft = -1, got_draft = -1;
        restore_mtp_case(reference, snapshot, rows, verify);
        need(reference_mtp_steps(r, &e->model, &e->weights, 0, next_tokens, T,
             TEST_PREFIX, true, want, &want_draft), "original complete MTP rows");
        frontier expected = capture(reference, 3u);

        restore_mtp_case(live, snapshot, rows, verify);
        need(qwen4_graph_mtp_cache_steps(g, &e->model, &e->weights, 0, next_tokens,
             T, TEST_PREFIX), "optimized cache-only rows");
        need(g->mtp_last_rows == 0, "cache-only rows are not recursive outputs");
        compare_frontier(live, &expected, true);
        compare_tensor("zero-skipping EH projection (cache)", g->mtp_proj, r->mtp_proj, 0,
                       (uint64_t)T * (DS4_N_HC + 1u) * DS4_N_EMBD * sizeof(float));

        restore_mtp_case(live, snapshot, rows, verify);
        need(qwen4_graph_mtp_steps(g, &e->model, &e->weights, 0, next_tokens, T,
             TEST_PREFIX, true, got, &got_draft), "optimized last-row MTP output");
        compare_frontier(live, &expected, true);
        compare_tensor("zero-skipping EH projection (full)", g->mtp_proj, r->mtp_proj, 0,
                       (uint64_t)T * (DS4_N_HC + 1u) * DS4_N_EMBD * sizeof(float));
        compare_tensor("last predictor residual", g->mtp_R, r->mtp_R,
                       (T - 1u) * residual_bytes, residual_bytes);
        exact("all predictor logits", got, want, (size_t)logits_bytes);
        need(got_draft == want_draft && got_draft == sample_argmax(got, DS4_N_VOCAB),
             "last-row draft agrees with original complete MTP");
        need(g->mtp_last_rows == T && g->mtp_pos == TEST_PREFIX + T,
             "last-row layout and contiguous cache frontier");
        free_frontier(&expected);

        /* The recursive consumer must read row T-1 after earlier FFN rows
         * are skipped. Its oracle uses the frozen input/full-layer helper,
         * with the original last residual explicitly selected as input. */
        const uint32_t idx = TEST_PREFIX + T;
        uint32_t pos3[4] = {0};
        int32_t delta = r->mrope_delta;
        qwen4_mrope_pos(NULL, 0, idx, &delta, pos3);
        need(ds4_gpu_tensor_write(r->pos3, (uint64_t)idx * sizeof(pos3), pos3, sizeof(pos3)),
             "reference future position");
        ds4_gpu_tensor *saved_R = r->R;
        ds4_gpu_tensor *last = ds4_gpu_tensor_view(r->mtp_R, (T - 1u) * residual_bytes, residual_bytes);
        need(last != NULL, "reference recursive residual view");
        r->R = last;
        const int parent = want_draft;
        const bool ok = reference_mtp_steps(r, &e->model, &e->weights, 0, &parent,
                                           1u, idx, true, want, &want_draft);
        r->R = saved_R;
        ds4_gpu_tensor_free(last);
        need(ok, "original recursive predictor output");
        need(qwen4_graph_mtp_chain_step(g, &e->model, &e->weights, got_draft, idx,
                                       &got_draft), "optimized recursive predictor output");
        need(ds4_gpu_tensor_read(g->logits, 0, got, logits_bytes), "recursive predictor logits");
        exact("all recursive predictor logits", got, want, (size_t)logits_bytes);
        need(got_draft == want_draft && g->mtp_last_rows == 1u,
             "recursive draft and output layout");
        compare_tensor("recursive predictor residual", g->mtp_R, r->mtp_R, 0, residual_bytes);
        expected = capture(reference, 3u);
        compare_frontier(live, &expected, true);
        free_frontier(&expected);
        printf("PASS optimized MTP T%u verify=%u: EH, KV, final residual, full logits and recursive draft\n",
               T, verify);
    }
    g->verify_rows_exact = false;
    r->verify_rows_exact = false;
    ds4_gpu_qwen4_set_verify_rows_exact(false);
    free(got); free(want);
}

static void close_logits(const float *got, const float *want) {
    /* The existing 2/3-row verifier and one-row target have distinct
     * reductions; use the established MTP batch tolerance, not text alone. */
    float worst = 0.0f;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float d = fabsf(got[i] - want[i]);
        if (!isfinite(got[i]) || !isfinite(want[i]) || d > 2e-3f + 1e-4f * fabsf(want[i])) {
            fprintf(stderr, "Qwen MTP target mismatch id=%u got=%g want=%g error=%g\n",
                    i, (double)got[i], (double)want[i], (double)d);
            exit(1);
        }
        if (d > worst) worst = d;
    }
    printf("  all %u target logits: max error %.6g\n", DS4_N_VOCAB, (double)worst);
}

static void check_cycles(ds4_session *live, ds4_session *target,
                         const ds4_session_snapshot *snapshot, unsigned mode) {
    need(ds4_session_load_snapshot(live, snapshot, error, sizeof(error)) == 0 &&
         ds4_session_load_snapshot(target, snapshot, error, sizeof(error)) == 0, "restore cycle frontier");
    const bool forced = mode == 1u, stochastic = mode == 2u;
    const bool speculative = qwen4_graph_fused(&live->qwen4_graph, 2u);
    need(setenv("DS4_QWEN4_MTP_DEPTH", mode ? "3" : "2", 1) == 0, "set depth case");
    need((forced ? setenv("DS4_QWEN4_SPEC_FORCE_ACCEPT", "1", 1) :
                   unsetenv("DS4_QWEN4_SPEC_FORCE_ACCEPT")) == 0, "set acceptance case");
    live->engine->dspark_exact_sampling = stochastic;
    uint64_t rng = 0x3456789abcdefull;
    bool saw_verify = false, saw_three = false;
    for (unsigned step = 0; step < 3u; step++) {
        const int first = ds4_session_sample(target, stochastic ? 0.8f : 0.0f,
                                             0, 0.95f, 0.0f, &rng);
        int accepted[3] = {-1, -1, -1};
        const int before = ds4_session_pos(live);
        const int n = ds4_session_eval_speculative(live, first, 3, -1,
            stochastic ? 0.8f : 0.0f, 0, 0.95f, 0.0f, &rng, accepted, 3, error, sizeof(error));
        need(n >= 1 && n <= (stochastic ? 2 : 3), "bounded accepted count");
        need(accepted[0] == first && ds4_session_pos(live) == before + n, "committed prefix count");
        saw_verify |= n > 1; saw_three |= n == 3;
        for (int i = 0; i < n; i++) {
            need(accepted[i] >= 0 && accepted[i] < (int)DS4_N_VOCAB, "accepted token range");
            if (!forced && !stochastic)
                need(accepted[i] == ds4_session_argmax(target), "greedy proposal agrees with target");
            need(ds4_session_eval(target, accepted[i], error, sizeof(error)) == 0,
                 "independent one-row target continuation");
        }
        close_logits(live->logits, target->logits);
    }
    if (forced) need(speculative ? saw_verify && saw_three : !saw_verify && !saw_three,
                     "depth-three verifier or explicit unfused fallback exercised");
    printf("PASS MTP continuation %s, independent target logits\n",
           !speculative ? "unfused ordinary fallback" : forced ? "forced depth3 state fixture" :
           stochastic ? "exact sampling depth2" : "greedy depth2");
}

static void check_payload_modes(ds4_session *live, ds4_session *reference,
                                const ds4_tokens *prefix, const ds4_session_snapshot *snapshot) {
    ds4_engine *e = live->engine;
    /* Share one model mapping, but allocate the third graph exactly as a
     * session created with MTP disabled. Ordinary eval uses that graph only. */
    ds4_session *off = NULL;
    const bool enabled = e->glm_mtp;
    e->glm_mtp = false;
    const int create_rc = ds4_session_create(&off, e, TEST_CTX);
    e->glm_mtp = enabled;
    need(create_rc == 0 && off && !off->qwen4_graph.mtp_R, "allocate MTP-disabled graph");
    need(ds4_session_load_snapshot(off, snapshot, error, sizeof(error)) == 0 &&
         ds4_session_load_snapshot(reference, snapshot, error, sizeof(error)) == 0,
         "MTP-on payload loads with MTP disabled");
    exact("MTP-disabled restored target logits", off->logits, reference->logits,
          (size_t)DS4_N_VOCAB * sizeof(float));
    const int next = (ds4_session_argmax(reference) + 1) % (int)DS4_N_VOCAB;
    need(ds4_session_eval(off, next, error, sizeof(error)) == 0 &&
         ds4_session_eval(reference, next, error, sizeof(error)) == 0,
         "MTP-disabled continuation from MTP-on payload");
    exact("MTP-disabled target continuation", off->logits, reference->logits,
          (size_t)DS4_N_VOCAB * sizeof(float));

    need(ds4_session_load_snapshot(off, snapshot, error, sizeof(error)) == 0,
         "restore MTP-disabled snapshot source");
    ds4_session_snapshot plain = {0};
    need(ds4_session_save_snapshot(off, &plain, error, sizeof(error)) == 0,
         "save actual MTP-disabled payload");
    ds4_session_free(off);
    poison_nextn(live, 0x77);
    need(ds4_session_load_snapshot(live, &plain, error, sizeof(error)) == 0,
         "MTP-off payload loads with MTP enabled");
    need(live->qwen4_graph.mtp_pos == 0 && !live->qwen4_graph.mtp_tail_valid,
         "MTP-off payload does not claim a predictor prefix");
    frontier frames[MAX_FRAMES] = {{0}};
    const unsigned n = reference_sync(reference, prefix, 0, frames);
    checked_sync(live, prefix, frames, n);
    for (unsigned i = 0; i < n; i++) free_frontier(frames + i);
    ds4_session_snapshot_free(&plain);

    /* The old tag cannot describe the deferred final row. Reject it before
     * any tensor writes instead of interpreting its high-water mark as KV. */
    ds4_session_snapshot old = {.len = snapshot->len, .cap = snapshot->len};
    old.ptr = malloc((size_t)old.len);
    need(old.ptr != NULL, "old payload allocation");
    memcpy(old.ptr, snapshot->ptr, (size_t)old.len);
    uint32_t tag = 0;
    memcpy(&tag, old.ptr + 12u * sizeof(uint32_t), sizeof(tag));
    need(tag == DS4_QWEN4_PAYLOAD_TAG, "payload family tag location");
    tag = 0x51573802u;
    memcpy(old.ptr + 12u * sizeof(uint32_t), &tag, sizeof(tag));
    need(ds4_session_load_snapshot(live, &old, error, sizeof(error)) != 0,
         "reject legacy nextn payload without a retained row");
    error[0] = '\0';
    ds4_session_snapshot_free(&old);
    puts("PASS MTP on/off payload compatibility, prefix replay and legacy-tag rejection");
}

int main(int argc, char **argv) {
    const bool streaming = argc == 3 && !strcmp(argv[2], "--ssd-streaming");
    if (argc != 2 && !streaming) {
        fprintf(stderr, "usage: %s QWEN_GGUF [--ssd-streaming]\n", argv[0]); return 2;
    }
    const char *names[] = {"DS4_QWEN4_SPEC_FORCE_ACCEPT", "DS4_QWEN4_MTP_DEPTH",
                          "DS4_QWEN4_MTP_DRAFT_ROWS", "DS4_QWEN4_MTP_DRAFT_VOCAB",
                          "DS4_QWEN4_MTP_GPU_ARGMAX"};
    char *saved[sizeof(names) / sizeof(names[0])] = {0};
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        const char *v = getenv(names[i]); saved[i] = v ? strdup(v) : NULL;
        need(!v || saved[i], "save test environment"); need(unsetenv(names[i]) == 0, "clear test override");
    }
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = TEST_CTX, .prefill_chunk = TEST_CHUNK, .glm_mtp = true,
        .ssd_streaming = streaming, .ssd_streaming_cache_experts = streaming ? 1024u : 0u};
    ds4_engine *engine = NULL;
    need(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen4(engine), "open Qwen model");
    ds4_session *live = NULL, *reference = NULL;
    need(ds4_session_create(&live, engine, TEST_CTX) == 0 &&
         ds4_session_create(&reference, engine, TEST_CTX) == 0, "allocate two bounded sessions");
    need(live->qwen4_graph.cap_tokens == TEST_CHUNK, "requested chunk size");
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, "Count from one to ten and explain the pattern.", DS4_THINK_NONE, &prompt);
    need(prompt.len > TEST_APPEND, "enough real prompt and lookahead token IDs");
    ds4_tokens prefix = prompt; prefix.len = 1;
    frontier frames[MAX_FRAMES] = {{0}};
    unsigned n = reference_sync(reference, &prefix, 0, frames);
    poison_nextn(live, 0xaa);
    checked_sync(live, &prefix, frames, n);
    need(live->qwen4_graph.mtp_pos == 0 && !live->qwen4_graph.mtp_tail_cached,
         "one-token prefix has only a deferred row");
    for (unsigned i = 0; i < n; i++) free_frontier(frames + i);
    ds4_session_invalidate(live);
    prefix.len = TEST_PREFIX;
    n = reference_sync(reference, &prefix, 0, frames);
    poison_nextn(live, 0xaa);
    checked_sync(live, &prefix, frames, n);
    need(live->qwen4_graph.mtp_pos + 1u == TEST_PREFIX, "last unknown token is not guessed");
    for (unsigned i = 0; i < n; i++) free_frontier(frames + i);
    puts("PASS cache-only prefix against full MTP: chunk lookahead and final retained row");

    /* Force the deferred row to use a real next input, not a guessed argmax. */
    prompt.v[TEST_PREFIX] = (ds4_session_argmax(reference) + 1) % (int)DS4_N_VOCAB;
    prefix.len = TEST_APPEND;
    n = reference_sync(reference, &prefix, TEST_PREFIX, frames);
    checked_sync(live, &prefix, frames, n);
    ds4_session_snapshot snapshot = {0};
    need(ds4_session_save_snapshot(live, &snapshot, error, sizeof(error)) == 0, "save prepared prefix");
    /* Load into a graph containing unrelated nextn rows and a different tail. */
    ds4_tokens other = {0};
    for (int i = 0; i < TEST_PREFIX; i++) ds4_tokens_push(&other, prompt.v[(i + 2) % TEST_APPEND]);
    need(ds4_session_sync(live, &other, error, sizeof(error)) == 0, "reuse with unrelated prefix");
    poison_nextn(live, 0x33);
    need(ds4_session_load_snapshot(live, &snapshot, error, sizeof(error)) == 0, "restore over stale cache");
    compare_frontier(live, frames + n - 1u, false);
    need(frames[n - 1u].residual_rows == 3u, "three real trunk rows for optimized-path oracle");
    check_mtp_paths(live, reference, &snapshot, prompt.v + TEST_PREFIX + 1u,
                    frames[n - 1u].residual);
    for (unsigned i = 0; i < n; i++) free_frontier(frames + i);
    puts("PASS append boundary and checkpoint restore over reused state");

    check_payload_modes(live, reference, &prefix, &snapshot);
    for (unsigned mode = 0; mode < 3; mode++) check_cycles(live, reference, &snapshot, mode);
    ds4_session_snapshot_free(&snapshot);
    ds4_tokens_free(&other); ds4_tokens_free(&prompt);
    ds4_session_free(reference); ds4_session_free(live); ds4_engine_close(engine);
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        need((saved[i] ? setenv(names[i], saved[i], 1) : unsetenv(names[i])) == 0, "restore test environment");
        free(saved[i]);
    }
    puts("Qwen MTP prefill: all checks passed");
    return 0;
}
#else
int main(void) { fputs("Qwen MTP prefill test requires Metal\n", stderr); return 2; }
#endif
