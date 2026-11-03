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
    if (ok) ok = qwen4_gemv(g, g->mtp_proj, m, l->nextn_eh_proj, g->mtp_cat, T * (hc + 1u));
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
                       : qwen4_gemv_rows(g, g->logits, m, w->output, g->mixed, 1, head_rows));
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

/* Cold preparation initializes prompt rows deterministically without running
 * the predictor projections. It changes the draft context, not the target
 * model or whether speculative verification remains available. */
static void check_cold_rows(ds4_session *s, uint32_t first, uint32_t rows) {
    const uint32_t il = DS4_N_LAYER - 1u;
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint64_t kv_row = qwen4_payload_kv_bytes(1u);
    const uint64_t ik_row = qwen4_payload_ik_bytes(1u);
    ds4_gpu_tensor *t[] = {g->layer_k_cache[il], g->layer_v_cache[il], g->layer_ik_cache[il]};
    const uint64_t row_bytes[] = {kv_row, kv_row, ik_row};
    for (unsigned i = 0; i < 3u; i++) {
        const uint64_t bytes = (uint64_t)rows * row_bytes[i];
        unsigned char *p = malloc(bytes ? (size_t)bytes : 1u);
        need(p != NULL && (!bytes || ds4_gpu_tensor_read(t[i],
             (uint64_t)first * row_bytes[i], p, bytes)), "read cold predictor rows");
        for (uint64_t j = 0; j < bytes; j++) need(p[j] == 0, "cold K/V/raw indexer rows are zero");
        free(p);
    }
    /* Only wholly cold, completed blocks have a zero pooled key. Half
     * signed zeros are valid after the ordinary norm/RoPE pool kernel. */
    const uint32_t first_block = (first + 3u) / 4u, end_block = (first + rows) / 4u;
    if (end_block > first_block) {
        const uint64_t n = (uint64_t)(end_block - first_block) * DS4_N_INDEXER_HEAD_DIM;
        uint16_t *p = malloc((size_t)n * sizeof(*p));
        need(p && ds4_gpu_tensor_read(g->layer_block_key[il],
             (uint64_t)first_block * DS4_N_INDEXER_HEAD_DIM * sizeof(*p), p, n * sizeof(*p)),
             "read cold pooled indexer keys");
        for (uint64_t i = 0; i < n; i++) need(f16_to_f32(p[i]) == 0.0f, "cold pooled keys are zero");
        free(p);
    }
}

typedef struct { ds4_session *s; unsigned callbacks; bool stop_after_chunk, cancelled; } cold_progress;
static void check_cold_progress(void *opaque, const char *phase, int completed, int total) {
    cold_progress *p = opaque;
    if (strcmp(phase, "prefill_chunk")) return;
    ds4_qwen4_gpu_graph *g = &p->s->qwen4_graph;
    const uint32_t known = (uint32_t)completed - 1u;
    need(completed <= total, "cold progress remains within the prompt");
    need(p->s->checkpoint_valid && g->pos == (uint32_t)completed && g->mtp_pos == known,
         "cold chunk publishes an initialized predictor frontier");
    need(g->mtp_R && p->s->engine->glm_mtp && g->mtp_tail_valid &&
         g->mtp_tail_pos + 1u == g->pos && g->mtp_last_rows == 0u,
         "cold preparation retains enabled MTP and the final trunk row");
    for (uint32_t start = 0u; start < known; start += TEST_CHUNK) {
        const uint32_t rows = known - start < TEST_CHUNK - 1u ? known - start : TEST_CHUNK - 1u;
        check_cold_rows(p->s, start, rows);
    }
    p->callbacks++;
    if (p->stop_after_chunk) p->cancelled = true;
}

static bool cancel_cold_prefill(void *opaque) {
    return ((cold_progress *)opaque)->cancelled;
}

static void check_cold_prefill(ds4_session *live, ds4_session *reference,
                               const ds4_tokens *prompt, const frontier *eager13,
                               const ds4_session_snapshot *eager16) {
    const bool exact_sampling = live->engine->dspark_exact_sampling;
    live->engine->dspark_exact_sampling = false;
    need(setenv("DS4_QWEN4_MTP_PREFILL", "off", 1) == 0 &&
         unsetenv("DS4_QWEN4_SPEC_FORCE_ACCEPT") == 0 &&
         unsetenv("DS4_QWEN4_MTP_DEPTH") == 0, "select cold predictor preparation");
    ds4_tokens prefix = *prompt; prefix.len = TEST_PREFIX;
    ds4_session_invalidate(live);
    poison_nextn(live, 0xa5);
    cold_progress p = {.s = live};
    ds4_session_set_progress(live, check_cold_progress, &p);
    need(ds4_session_sync(live, &prefix, error, sizeof(error)) == 0 && p.callbacks == 2u,
         "cold prefix crosses a prefill chunk boundary");
    ds4_session_set_progress(live, NULL, NULL);
    frontier cold13 = capture(live, 0u);
    exact("cold/eager target prefix logits", cold13.logits, eager13->logits,
          (size_t)DS4_N_VOCAB * sizeof(float));
    exact("cold/eager retained target row", cold13.tail, eager13->tail,
          (size_t)DS4_N_EMBD * DS4_N_HC * sizeof(float));

    p.callbacks = 0u;
    ds4_session_set_progress(live, check_cold_progress, &p);
    need(ds4_session_sync(live, &prefix, error, sizeof(error)) == 0 && p.callbacks == 0u,
         "identical cold prompt reuses its checkpoint without replay");
    ds4_session_set_progress(live, NULL, NULL);
    compare_frontier(live, &cold13, false);
    /* Reject a typo before it can consume pending drafts or append rows. */
    live->glm_mtp_have = 1;
    live->glm_mtp_have2 = true;
    ds4_tokens invalid_append = *prompt; invalid_append.len = TEST_APPEND;
    need(setenv("DS4_QWEN4_MTP_PREFILL", "invalid", 1) == 0 &&
         ds4_session_sync(live, &invalid_append, error, sizeof(error)) != 0 &&
         strstr(error, "DS4_QWEN4_MTP_PREFILL") != NULL,
         "invalid preparation policy is rejected");
    need(live->glm_mtp_have == 1 && live->glm_mtp_have2 &&
         live->checkpoint.len == TEST_PREFIX &&
         !memcmp(live->checkpoint.v, prompt->v, TEST_PREFIX * sizeof(*prompt->v)),
         "invalid preparation policy leaves pending drafts and checkpoint untouched");
    compare_frontier(live, &cold13, false);
    live->glm_mtp_have = 0;
    live->glm_mtp_have2 = false;
    error[0] = '\0';
    need(setenv("DS4_QWEN4_MTP_PREFILL", "off", 1) == 0, "restore cold preparation policy");
    ds4_session_snapshot cold_snapshot = {0}, partial = {0}, appended = {0};
    need(ds4_session_save_snapshot(live, &cold_snapshot, error, sizeof(error)) == 0,
         "save cold predictor prefix");
    poison_nextn(live, 0x3c);
    need(ds4_session_load_snapshot(live, &cold_snapshot, error, sizeof(error)) == 0,
         "restore cold prefix over stale predictor cache");
    compare_frontier(live, &cold13, false);

    /* Each cold chunk leaves its tail deferred. Cancellation and snapshot
     * restore therefore prepare exactly the same real boundary row as an
     * uninterrupted prefill, including the resulting predictor caches. */
    ds4_session_invalidate(live);
    poison_nextn(live, 0x5a);
    p = (cold_progress){.s = live, .stop_after_chunk = true};
    ds4_session_set_progress(live, check_cold_progress, &p);
    ds4_session_set_cancel(live, cancel_cold_prefill, &p);
    need(ds4_session_sync(live, &prefix, error, sizeof(error)) == DS4_SESSION_SYNC_INTERRUPTED &&
         p.callbacks == 1u && ds4_session_pos(live) == TEST_CHUNK,
         "cold prefill cancellation retains a completed chunk");
    ds4_session_set_cancel(live, NULL, NULL);
    ds4_session_set_progress(live, NULL, NULL);
    error[0] = '\0';
    need(ds4_session_save_snapshot(live, &partial, error, sizeof(error)) == 0,
         "save cancelled cold prefill frontier");
    need(ds4_session_sync(live, &prefix, error, sizeof(error)) == 0, "resume cold prefill in place");
    compare_frontier(live, &cold13, false);
    poison_nextn(reference, 0x69);
    need(ds4_session_load_snapshot(reference, &partial, error, sizeof(error)) == 0 &&
         ds4_session_sync(reference, &prefix, error, sizeof(error)) == 0,
         "resume serialized cancelled cold prefill");
    compare_frontier(reference, &cold13, false);

    prefix.len = TEST_APPEND;
    need(ds4_session_sync(live, &prefix, error, sizeof(error)) == 0 &&
         ds4_session_load_snapshot(reference, eager16, error, sizeof(error)) == 0,
         "append to cold prefix and restore independent eager target");
    exact("cold/eager appended target logits", live->logits, reference->logits,
          (size_t)DS4_N_VOCAB * sizeof(float));
    check_cold_rows(live, 0u, TEST_CHUNK - 1u);
    check_cold_rows(live, TEST_CHUNK, TEST_PREFIX - TEST_CHUNK - 1u);
    check_cold_rows(live, TEST_PREFIX, TEST_APPEND - TEST_PREFIX - 1u);
    /* The deferred row 12 is filled by ordinary boundary preparation before
     * append. The cold helper must preserve it while clearing rows 13/14. */
    const uint32_t il = DS4_N_LAYER - 1u;
    const uint64_t kv_row = qwen4_payload_kv_bytes(1u), ik_row = qwen4_payload_ik_bytes(1u);
    compare_tensor("preserved append boundary K", live->qwen4_graph.layer_k_cache[il],
                   reference->qwen4_graph.layer_k_cache[il], (TEST_PREFIX - 1u) * kv_row, kv_row);
    compare_tensor("preserved append boundary V", live->qwen4_graph.layer_v_cache[il],
                   reference->qwen4_graph.layer_v_cache[il], (TEST_PREFIX - 1u) * kv_row, kv_row);
    compare_tensor("preserved append boundary indexer", live->qwen4_graph.layer_ik_cache[il],
                   reference->qwen4_graph.layer_ik_cache[il], (TEST_PREFIX - 1u) * ik_row, ik_row);
    need(ds4_session_save_snapshot(live, &appended, error, sizeof(error)) == 0,
         "save appended cold predictor prefix");
    need(ds4_session_eval(live, prompt->v[TEST_APPEND], error, sizeof(error)) == 0 &&
         ds4_session_eval(reference, prompt->v[TEST_APPEND], error, sizeof(error)) == 0,
         "ordinary decode continues after cold prefill");
    exact("cold/eager ordinary target continuation", live->logits, reference->logits,
          (size_t)DS4_N_VOCAB * sizeof(float));

    /* A longer append closes block [12,16): its first key is real and its
     * other three keys are cold. Recompute an independent output buffer
     * with the ordinary pool kernel to catch erasing the whole mixed block. */
    ds4_tokens mixed = {0};
    for (int i = 0; i < 18; i++) ds4_tokens_push(&mixed, prompt->v[i % prompt->len]);
    need(ds4_session_load_snapshot(live, &cold_snapshot, error, sizeof(error)) == 0 &&
         ds4_session_load_snapshot(reference, &cold_snapshot, error, sizeof(error)) == 0 &&
         ds4_session_sync(live, &mixed, error, sizeof(error)) == 0 &&
         setenv("DS4_QWEN4_MTP_PREFILL", "on", 1) == 0 &&
         ds4_session_sync(reference, &mixed, error, sizeof(error)) == 0 &&
         setenv("DS4_QWEN4_MTP_PREFILL", "off", 1) == 0,
         "cold and eager append complete a mixed predictor block");
    exact("mixed-block target logits", live->logits, reference->logits,
          (size_t)DS4_N_VOCAB * sizeof(float));
    check_cold_rows(live, TEST_PREFIX, 4u);
    compare_tensor("mixed-block retained real indexer key", live->qwen4_graph.layer_ik_cache[il],
                   reference->qwen4_graph.layer_ik_cache[il], (TEST_PREFIX - 1u) * ik_row, ik_row);
    const uint64_t block_bytes = (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t);
    ds4_gpu_tensor *pooled = ds4_gpu_tensor_alloc(4u * block_bytes);
    ds4_engine *e = live->engine;
    need(pooled && glm_graph_begin_commands_if_needed() &&
         ds4_gpu_qwen4_idx_block_key_tensor(pooled, live->qwen4_graph.layer_ik_cache[il],
             live->qwen4_graph.pos3, e->model.map, e->model.size,
             e->weights.layer[il].indexer_k_norm->abs_offset, 3u, 1u, 4u,
             DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT, DS4_ROPE_FREQ_BASE, DS4_RMS_EPS) &&
         ds4_gpu_end_commands(), "ordinary pooled-key oracle for mixed cold/real block");
    compare_tensor("mixed cold/real pooled block", live->qwen4_graph.layer_block_key[il],
                   pooled, 3u * block_bytes, block_bytes);
    ds4_gpu_tensor_free(pooled);
    ds4_tokens_free(&mixed);
    for (unsigned mode = 0; mode < 3u; mode++) check_cycles(live, reference, &appended, mode);
    ds4_session_snapshot_free(&appended);
    ds4_session_snapshot_free(&partial);
    ds4_session_snapshot_free(&cold_snapshot);
    free_frontier(&cold13);
    live->engine->dspark_exact_sampling = exact_sampling;
    need(setenv("DS4_QWEN4_MTP_PREFILL", "on", 1) == 0, "restore eager oracle policy");
    puts("PASS cold predictor prefix: deterministic initialization, reuse, append, restore, cancellation and active MTP");
}

/* The long fixture runs the trunk only once. At the first sparse query and
 * at the deep prefix, replay one predictor layer over three actual trunk
 * rows. Saving only the predictor cache keeps this feasible with SSD model
 * weights on a 32 GiB Mac; no second context or full-model snapshot is used.
 * This isolates optimized predictor arithmetic from upstream trunk changes.
 * It does not measure speculative acceptance or establish model quality. */
enum { LONG_PREFIX = 17408, LONG_CHUNK = 128 };

typedef struct {
    ds4_session *s;
    const ds4_tokens *tokens;
    uint32_t prefix, previous, boundary, probes;
    const char *dump_prefix;
    FILE *manifest;
} long_check;

typedef struct {
    frontier state;
    uint32_t last_rows, tail_pos;
    int tail_next;
    bool tail_valid, tail_cached, verify;
    qwen4_projection_phase phase;
    void *positions;
} predictor_seed;

static void finite_f32(const char *what, const float *v, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(v[i])) {
            fprintf(stderr, "Qwen MTP long prefill: %s has nonfinite value at %llu\n",
                    what, (unsigned long long)i);
            exit(1);
        }
    }
}

static void finite_f16(const char *what, const void *p, uint64_t bytes) {
    const uint16_t *v = p;
    for (uint64_t i = 0; i < bytes / sizeof(*v); i++) {
        if ((v[i] & 0x7c00u) == 0x7c00u) {
            fprintf(stderr, "Qwen MTP long prefill: %s has nonfinite half at %llu\n",
                    what, (unsigned long long)i);
            exit(1);
        }
    }
}

static void finite_frontier(const frontier *f) {
    finite_f16("nextn K", f->k, f->kv_bytes);
    finite_f16("nextn V", f->v, f->kv_bytes);
    finite_f32("nextn indexer keys", f->ik, f->ik_bytes / sizeof(float));
    finite_f16("nextn pooled keys", f->block, f->block_bytes);
    finite_f32("retained trunk row", f->tail, (uint64_t)DS4_N_EMBD * DS4_N_HC);
    finite_f32("target logits", f->logits, DS4_N_VOCAB);
}

static void dump_long(const long_check *p, uint32_t idx, uint32_t T, bool verify,
                       const char *kind, const void *data, uint64_t bytes) {
    if (!p->dump_prefix) return;
    char path[4096];
    const int n = snprintf(path, sizeof(path), "%s.pos%u.T%u.verify%u.%s",
                           p->dump_prefix, idx, T, verify, kind);
    need(n > 0 && (size_t)n < sizeof(path), "bounded dump path");
    FILE *f = fopen(path, "wb");
    need(f != NULL, "open long-context dump");
    const bool ok = fwrite(data, 1, (size_t)bytes, f) == (size_t)bytes;
    const int close_rc = fclose(f);
    need(ok && close_rc == 0, "write long-context dump");
}

static predictor_seed save_predictor(ds4_session *s, uint32_t batch, uint32_t idx) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint32_t il = DS4_N_LAYER - 1u;
    predictor_seed seed = {.state = capture(s, batch), .last_rows = g->mtp_last_rows,
        .tail_pos = g->mtp_tail_pos, .tail_next = g->mtp_tail_next_token,
        .tail_valid = g->mtp_tail_valid, .tail_cached = g->mtp_tail_cached,
        .verify = g->verify_rows_exact, .phase = g->projection_phase};
    finite_frontier(&seed.state);
    finite_f32("actual trunk rows", seed.state.residual,
               seed.state.residual_bytes / sizeof(float));
    /* A recursive probe may write one row beyond the actual prefix. Retain
     * that storage too, without interpreting uninitialized future bytes. */
    uint32_t saved_rows = seed.state.mtp_rows;
    if (saved_rows < idx + 4u) saved_rows = idx + 4u;
    free(seed.state.k); free(seed.state.v); free(seed.state.ik); free(seed.state.block);
    seed.state.kv_bytes = qwen4_payload_kv_bytes(saved_rows);
    seed.state.ik_bytes = qwen4_payload_ik_bytes(saved_rows);
    seed.state.block_bytes = qwen4_payload_block_key_bytes(saved_rows);
    seed.state.k = read_bytes(g->layer_k_cache[il], seed.state.kv_bytes);
    seed.state.v = read_bytes(g->layer_v_cache[il], seed.state.kv_bytes);
    seed.state.ik = read_bytes(g->layer_ik_cache[il], seed.state.ik_bytes);
    seed.state.block = read_bytes(g->layer_block_key[il], seed.state.block_bytes);
    seed.positions = malloc(4u * 4u * sizeof(uint32_t));
    need(seed.positions && ds4_gpu_tensor_read(g->pos3,
        (uint64_t)idx * 4u * sizeof(uint32_t), seed.positions, 4u * 4u * sizeof(uint32_t)),
        "save probe RoPE positions");
    return seed;
}

static void restore_predictor(ds4_session *s, const predictor_seed *seed,
                              uint32_t idx, bool probe, bool verify) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint32_t il = DS4_N_LAYER - 1u;
    const frontier *f = &seed->state;
    need(ds4_gpu_tensor_write(g->layer_k_cache[il], 0, f->k, f->kv_bytes) &&
         ds4_gpu_tensor_write(g->layer_v_cache[il], 0, f->v, f->kv_bytes) &&
         ds4_gpu_tensor_write(g->layer_ik_cache[il], 0, f->ik, f->ik_bytes) &&
         ds4_gpu_tensor_write(g->layer_block_key[il], 0, f->block, f->block_bytes) &&
         ds4_gpu_tensor_write(g->pos3, (uint64_t)idx * 4u * sizeof(uint32_t),
                              seed->positions, 4u * 4u * sizeof(uint32_t)),
         "restore predictor-only cache and positions");
    if (probe) {
        /* Existing real prefix rows would hide a skipped cache publication.
         * Poison only rows that the probe must recreate; older causal rows
         * remain the identical real prefix for both executions. */
        const uint64_t kv_bytes = qwen4_payload_kv_bytes(4u);
        const uint64_t ik_bytes = qwen4_payload_ik_bytes(4u);
        const uint64_t block_bytes = qwen4_payload_block_key_bytes(4u);
        const uint64_t max_bytes = kv_bytes > ik_bytes ? kv_bytes : ik_bytes;
        void *poison = malloc((size_t)max_bytes);
        need(poison != NULL && max_bytes >= block_bytes, "long cache poison allocation");
        memset(poison, 0xff, (size_t)max_bytes);
        need(ds4_gpu_tensor_write(g->layer_k_cache[il], qwen4_payload_kv_bytes(idx), poison, kv_bytes) &&
             ds4_gpu_tensor_write(g->layer_v_cache[il], qwen4_payload_kv_bytes(idx), poison, kv_bytes) &&
             ds4_gpu_tensor_write(g->layer_ik_cache[il], qwen4_payload_ik_bytes(idx), poison, ik_bytes) &&
             ds4_gpu_tensor_write(g->layer_block_key[il], qwen4_payload_block_key_bytes(idx),
                                  poison, block_bytes), "poison cache rows under test");
        free(poison);
    }
    g->mtp_pos = probe ? idx : f->mtp_rows;
    g->mtp_last_rows = seed->last_rows;
    g->mtp_tail_pos = seed->tail_pos; g->mtp_tail_next_token = seed->tail_next;
    g->mtp_tail_valid = seed->tail_valid; g->mtp_tail_cached = seed->tail_cached;
    g->projection_phase = seed->phase;
    g->verify_rows_exact = probe ? verify : seed->verify;
    ds4_gpu_qwen4_set_verify_rows_exact(g->verify_rows_exact);
}

static void check_long_trunk(ds4_session *s, const predictor_seed *seed) {
    void *got = read_bytes(s->qwen4_graph.R, seed->state.residual_bytes);
    exact("long unmodified trunk residual", got, seed->state.residual,
          (size_t)seed->state.residual_bytes);
    free(got);
    exact("long unmodified target logits", s->logits, seed->state.logits,
          (size_t)DS4_N_VOCAB * sizeof(float));
}

static void future_position(ds4_qwen4_gpu_graph *g, uint32_t idx) {
    uint32_t pos3[4] = {0};
    int32_t delta = g->mrope_delta;
    qwen4_mrope_pos(NULL, 0, idx, &delta, pos3);
    need(ds4_gpu_tensor_write(g->pos3, (uint64_t)idx * sizeof(pos3), pos3, sizeof(pos3)),
         "long recursive RoPE position");
}

static void check_long_probe(long_check *p, uint32_t idx, uint32_t row, uint32_t batch) {
    ds4_session *s = p->s;
    ds4_engine *e = s->engine;
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const uint64_t residual_bytes = (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    need(row + 3u <= batch && idx + 4u < g->ctx_cap && idx + 3u < (uint32_t)p->tokens->len,
         "long probe has actual rows, lookahead and context capacity");
    predictor_seed seed = save_predictor(s, batch, idx);
    const int *next = p->tokens->v + idx + 1u;
    float *want = malloc((size_t)logits_bytes), *recursive = malloc((size_t)logits_bytes);
    float *got = malloc((size_t)logits_bytes);
    need(want && recursive && got, "long predictor comparison buffers");
    dump_long(p, g->pos, 1u, false, "target.f32", s->logits, logits_bytes);
    dump_long(p, idx, 3u, false, "trunk.f32",
              (char *)seed.state.residual + row * residual_bytes, 3u * residual_bytes);
    for (unsigned mode = 0; mode < 4u; mode++) {
        const uint32_t T = mode < 3u ? mode + 1u : 3u;
        const bool verify = mode == 3u;
        const uint64_t proj_bytes = (uint64_t)T * (DS4_N_HC + 1u) * DS4_N_EMBD * sizeof(float);
        int want_draft = -1, recursive_draft = -1, got_draft = -1;
        restore_predictor(s, &seed, idx, true, verify);
        need(reference_mtp_steps(g, &e->model, &e->weights, row, next, T, idx,
                                 true, want, &want_draft), "long frozen complete predictor");
        frontier expected = capture(s, 0);
        finite_frontier(&expected); finite_f32("frozen predictor logits", want, DS4_N_VOCAB);
        void *proj = read_bytes(g->mtp_proj, proj_bytes);
        void *last = malloc((size_t)residual_bytes);
        need(last && ds4_gpu_tensor_read(g->mtp_R, (T - 1u) * residual_bytes, last, residual_bytes),
             "save frozen last predictor residual");
        finite_f32("frozen EH projection", proj, proj_bytes / sizeof(float));
        finite_f32("frozen predictor residual", last, residual_bytes / sizeof(float));
        check_long_trunk(s, &seed);

        future_position(g, idx + T);
        ds4_gpu_tensor *saved_R = g->R;
        ds4_gpu_tensor *last_view = ds4_gpu_tensor_view(g->mtp_R, (T - 1u) * residual_bytes, residual_bytes);
        need(last_view != NULL, "long frozen recursive input view");
        g->R = last_view;
        const bool reference_ok = reference_mtp_steps(g, &e->model, &e->weights,
            0, &want_draft, 1u, idx + T, true, recursive, &recursive_draft);
        g->R = saved_R;
        ds4_gpu_tensor_free(last_view);
        need(reference_ok, "long frozen recursive predictor");
        frontier recursive_expected = capture(s, 0);
        finite_frontier(&recursive_expected);
        finite_f32("frozen recursive logits", recursive, DS4_N_VOCAB);
        void *recursive_R = read_bytes(g->mtp_R, residual_bytes);
        finite_f32("frozen recursive residual", recursive_R, residual_bytes / sizeof(float));

        restore_predictor(s, &seed, idx, true, verify);
        need(qwen4_graph_mtp_cache_steps(g, &e->model, &e->weights, row, next, T, idx),
             "long optimized cache-only predictor");
        need(g->mtp_last_rows == 0, "long cache-only rows publish no predictor output");
        compare_frontier(s, &expected, false);
        void *actual = read_bytes(g->mtp_proj, proj_bytes);
        exact("long cache-only EH projection", actual, proj, (size_t)proj_bytes); free(actual);
        check_long_trunk(s, &seed);

        restore_predictor(s, &seed, idx, true, verify);
        need(qwen4_graph_mtp_steps(g, &e->model, &e->weights, row, next, T, idx,
                                 true, got, &got_draft), "long optimized last-row predictor");
        dump_long(p, idx, T, verify, "reference.f32", want, logits_bytes);
        dump_long(p, idx, T, verify, "candidate.f32", got, logits_bytes);
        finite_f32("optimized predictor logits", got, DS4_N_VOCAB);
        exact("long full predictor logits", got, want, (size_t)logits_bytes);
        need(got_draft == want_draft && got_draft == sample_argmax(got, DS4_N_VOCAB) &&
             g->mtp_pos == idx + T && g->mtp_last_rows == T, "long predictor draft and row layout");
        compare_frontier(s, &expected, false);
        actual = read_bytes(g->mtp_proj, proj_bytes);
        exact("long full EH projection", actual, proj, (size_t)proj_bytes); free(actual);
        actual = malloc((size_t)residual_bytes);
        need(actual && ds4_gpu_tensor_read(g->mtp_R, (T - 1u) * residual_bytes, actual, residual_bytes),
             "read long last predictor residual");
        exact("long last predictor residual", actual, last, (size_t)residual_bytes); free(actual);

        future_position(g, idx + T);
        need(qwen4_graph_mtp_chain_step(g, &e->model, &e->weights, got_draft,
                                       idx + T, &got_draft), "long optimized recursive predictor");
        need(ds4_gpu_tensor_read(g->logits, 0, got, logits_bytes), "read long recursive logits");
        dump_long(p, idx, T, verify, "reference-recursive.f32", recursive, logits_bytes);
        dump_long(p, idx, T, verify, "candidate-recursive.f32", got, logits_bytes);
        finite_f32("optimized recursive logits", got, DS4_N_VOCAB);
        exact("long recursive predictor logits", got, recursive, (size_t)logits_bytes);
        need(got_draft == recursive_draft && g->mtp_pos == idx + T + 1u && g->mtp_last_rows == 1u,
             "long recursive draft and cache frontier");
        compare_frontier(s, &recursive_expected, false);
        actual = read_bytes(g->mtp_R, residual_bytes);
        exact("long recursive residual", actual, recursive_R, (size_t)residual_bytes); free(actual);
        check_long_trunk(s, &seed);
        if (p->manifest) {
            fprintf(p->manifest, "{\"idx\":%u,\"T\":%u,\"verify\":%u,\"trunk_frontier\":%u,"
                    "\"next_tokens\":[%d,%d,%d],\"draft\":%d,\"recursive_draft\":%d}\n",
                    idx, T, verify, g->pos, next[0], next[1], next[2], want_draft, recursive_draft);
            need(fflush(p->manifest) == 0, "flush long-context manifest");
        }
        printf("PASS long MTP idx=%u T=%u verify=%u: cache, EH, residual, full logits and recursive draft exact\n",
               idx, T, verify);
        fflush(stdout);
        free_frontier(&recursive_expected); free_frontier(&expected);
        free(recursive_R); free(last); free(proj);
    }
    restore_predictor(s, &seed, idx, false, false);
    /* The session's host logits never changed. Restore the device head too,
     * so this callback remains transparent to ordinary continuation. */
    need(ds4_gpu_tensor_write(g->logits, 0, seed.state.logits, logits_bytes), "restore target device logits");
    check_long_trunk(s, &seed);
    free(seed.positions); free_frontier(&seed.state);
    free(got); free(recursive); free(want);
    p->probes++;
}

static void check_long_progress(void *opaque, const char *phase, int completed, int total) {
    long_check *p = opaque;
    if (strcmp(phase, "prefill_chunk")) return;
    need(completed > (int)p->previous && completed <= total && (uint32_t)total == p->prefix,
         "long prefill progress frontier");
    const uint32_t end = (uint32_t)completed, batch = end - p->previous;
    const uint32_t probes[] = {p->boundary - 2u, p->prefix - 3u};
    for (unsigned i = 0; i < 2u; i++) {
        if (probes[i] >= p->previous && probes[i] + 3u <= end)
            check_long_probe(p, probes[i], probes[i] - p->previous, batch);
    }
    printf("long prefill %u/%u tokens\n", end, p->prefix); fflush(stdout);
    p->previous = end;
}

static void run_long_test(ds4_engine *engine, uint32_t prefix_rows, uint32_t chunk,
                           const char *dump_prefix) {
    ds4_session *s = NULL;
    const uint32_t ctx = prefix_rows + 8u;
    need(ds4_session_create(&s, engine, (int)ctx) == 0, "allocate one long-context session");
    need(s->qwen4_graph.cap_tokens == chunk, "requested long-context chunk size");
    const uint32_t boundary = (s->qwen4_graph.k_blocks + 1u) * 4u - 1u;
    need(prefix_rows >= boundary + 5u && prefix_rows % 4u == 0,
         "long prefix covers sparse boundary and separate deep probe");
    ds4_tokens phrase = {0}, tokens = {0};
    ds4_encode_chat_prompt(engine, NULL,
        "Count from one to ten and explain the pattern. Then compare Roman roads, aqueducts, "
        "trade and laws with their modern equivalents. Keep each answer clear and factual.",
        DS4_THINK_NONE, &phrase);
    need(phrase.len >= 8, "real long-context fixture token IDs");
    /* Fixed teacher-forced IDs make both chunk sizes and historical builds
     * comparable. This repeated synthetic fixture is not a quality corpus. */
    for (uint32_t i = 0; i < prefix_rows + 4u; i++)
        ds4_tokens_push(&tokens, phrase.v[i % (uint32_t)phrase.len]);
    long_check p = {.s = s, .tokens = &tokens, .prefix = prefix_rows,
                   .boundary = boundary, .dump_prefix = dump_prefix};
    if (dump_prefix) {
        char path[4096];
        const int n = snprintf(path, sizeof(path), "%s.jsonl", dump_prefix);
        need(n > 0 && (size_t)n < sizeof(path), "bounded long manifest path");
        p.manifest = fopen(path, "w"); need(p.manifest != NULL, "open long manifest");
        fprintf(p.manifest, "{\"prefix_tokens\":%u,\"context\":%u,\"chunk\":%u,"
                "\"sparse_position\":%u,\"vocab\":%u,\"fixture\":\"repeated-chat-ids\"}\n",
                prefix_rows, ctx, chunk, boundary, DS4_N_VOCAB);
        dump_long(&p, 0, (uint32_t)tokens.len, false, "tokens.i32", tokens.v,
                  (uint64_t)tokens.len * sizeof(*tokens.v));
    }
    printf("long MTP fixture: prefix=%u ctx=%u chunk=%u sparse=%u, one graph, repeated token IDs\n",
           prefix_rows, ctx, chunk, boundary); fflush(stdout);
    ds4_tokens prefix = tokens; prefix.len = (int)prefix_rows;
    ds4_session_set_progress(s, check_long_progress, &p);
    need(ds4_session_sync(s, &prefix, error, sizeof(error)) == 0, "long real trunk prefill");
    ds4_session_set_progress(s, NULL, NULL);
    need(p.probes == 2u && s->qwen4_graph.pos == prefix_rows &&
         s->qwen4_graph.mtp_pos + 1u == prefix_rows && s->qwen4_graph.mtp_last_rows == 0,
         "both long probes ran and restored the original deferred frontier");
    if (p.manifest) need(fclose(p.manifest) == 0, "close long manifest");
    ds4_tokens_free(&tokens); ds4_tokens_free(&phrase); ds4_session_free(s);
    puts("PASS long-context MTP frozen-layer regression (not an acceptance-rate benchmark)");
}

int main(int argc, char **argv) {
    bool streaming = false, long_context = false, long_options = false;
    uint32_t chunk = LONG_CHUNK, prefix_rows = LONG_PREFIX;
    const char *dump_prefix = NULL;
    bool valid = argc >= 2;
    for (int i = 2; valid && i < argc; i++) {
        if (!strcmp(argv[i], "--ssd-streaming")) streaming = true;
        else if (!strcmp(argv[i], "--long-context")) long_context = true;
        else if ((!strcmp(argv[i], "--prefill-chunk") || !strcmp(argv[i], "--prefix-tokens")) && i + 1 < argc) {
            long_options = true;
            const bool is_chunk = !strcmp(argv[i], "--prefill-chunk");
            char *end = NULL;
            const unsigned long value = strtoul(argv[++i], &end, 10);
            valid = end && *end == '\0' && end != argv[i] && value <= 32768u;
            if (is_chunk) { valid = valid && (value == 128u || value == 2048u); chunk = (uint32_t)value; }
            else { valid = valid && value >= 128u && value % 4u == 0; prefix_rows = (uint32_t)value; }
        } else if (!strcmp(argv[i], "--dump-prefix") && i + 1 < argc) {
            long_options = true; dump_prefix = argv[++i];
        } else valid = false;
    }
    if (!valid || (!long_context && long_options)) {
        fprintf(stderr, "usage: %s QWEN_GGUF [--ssd-streaming] [--long-context "
                "[--prefill-chunk 128|2048] [--prefix-tokens N] [--dump-prefix PATH]]\n", argv[0]); return 2;
    }
    const char *names[] = {"DS4_QWEN4_SPEC_FORCE_ACCEPT", "DS4_QWEN4_MTP_DEPTH",
                          "DS4_QWEN4_MTP_DRAFT_ROWS", "DS4_QWEN4_MTP_DRAFT_VOCAB",
                          "DS4_QWEN4_MTP_GPU_ARGMAX", "DS4_QWEN4_MTP_PREFILL"};
    char *saved[sizeof(names) / sizeof(names[0])] = {0};
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        const char *v = getenv(names[i]); saved[i] = v ? strdup(v) : NULL;
        need(!v || saved[i], "save test environment"); need(unsetenv(names[i]) == 0, "clear test override");
    }
    need(setenv("DS4_QWEN4_MTP_PREFILL", "on", 1) == 0, "select eager full-layer oracle policy");
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .context_size = long_context ? prefix_rows + 8u : TEST_CTX,
        .prefill_chunk = long_context ? chunk : TEST_CHUNK, .glm_mtp = true,
        .ssd_streaming = streaming, .ssd_streaming_cache_experts = streaming ? 1024u : 0u};
    ds4_engine *engine = NULL;
    need(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen4(engine), "open Qwen model");
    if (long_context) {
        run_long_test(engine, prefix_rows, chunk, dump_prefix);
        ds4_engine_close(engine);
        goto restore_environment;
    }
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
    frontier eager13 = capture(reference, 0u);
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
    check_cold_prefill(live, reference, &prompt, &eager13, &snapshot);
    free_frontier(&eager13);
    ds4_session_snapshot_free(&snapshot);
    ds4_tokens_free(&other); ds4_tokens_free(&prompt);
    ds4_session_free(reference); ds4_session_free(live); ds4_engine_close(engine);
restore_environment:
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
